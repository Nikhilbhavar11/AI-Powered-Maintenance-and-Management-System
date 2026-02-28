"""
FastAPI Application — REST APIs, WebSocket streaming, and lifespan.

Endpoints:
  GET  /api/devices                        — List all registered devices
  GET  /api/chart-data?device_id&seconds   — Recent sensor + prediction data
  GET  /api/status?device_id               — Latest status for a device
  POST /api/chat                           — Conversational AI endpoint
  WS   /ws/live?device_id                  — Real-time prediction stream

Lifespan:
  startup  → init Firebase, load ML model, discover devices, start scheduler
  shutdown → stop scheduler

Design decisions:
  • Static files served from /static for the dashboard.
  • CORS enabled for local development.
  • All Firebase I/O is offloaded to threads via asyncio.to_thread().
  • WebSocket pushes cached predictions every scheduler tick.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel

import config
import firebase_client
import ml_engine
import scheduler
import chat_engine
from device_registry import DeviceRegistry

# ─── Logging setup ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Shared state ───────────────────────────────────────────────
registry = DeviceRegistry()


# ─── Application Lifespan ───────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
      1. Initialize Firebase Admin SDK
      2. Load ML model from disk
      3. Discover devices from Firebase
      4. Start the background scheduler

    Shutdown:
      1. Stop the scheduler gracefully
    """
    logger.info("=" * 60)
    logger.info("PREDICTIVE MAINTENANCE PLATFORM — STARTING")
    logger.info("=" * 60)

    # 1. Firebase
    await asyncio.to_thread(firebase_client.init_firebase)

    # 2. ML model
    model_loaded = await asyncio.to_thread(ml_engine.load_model)
    if not model_loaded:
        logger.warning("Running in RULE-BASED fallback mode (no model file)")

    # 3. Device discovery
    device_ids = await asyncio.to_thread(registry.discover)
    logger.info("Discovered %d device(s): %s", len(device_ids), device_ids)

    # 4. Start scheduler
    scheduler.start(registry)

    logger.info("Platform ready — serving on %s:%s", config.HOST, config.PORT)
    logger.info("=" * 60)

    yield  # Application runs here

    # Shutdown
    logger.info("Shutting down scheduler...")
    await scheduler.stop()
    logger.info("Platform stopped")


# ─── FastAPI App ─────────────────────────────────────────────────

app = FastAPI(
    title="Predictive Maintenance Platform",
    description="Real-time IoT analytics with explainable AI",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")


# ═══════════════════════════════════════════════════════════════
# REST API ENDPOINTS
# ═══════════════════════════════════════════════════════════════


@app.get("/api/devices")
async def api_devices() -> JSONResponse:
    """
    List all registered devices with their latest status summary.
    """
    devices = registry.to_summary_list()
    return JSONResponse(content={
        "devices": devices,
        "count": len(devices),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


@app.get("/api/chart-data")
async def api_chart_data(
    device_id: str = Query(..., description="Device ID"),
    seconds: int = Query(60, ge=10, le=3600, description="Lookback window in seconds"),
) -> JSONResponse:
    """
    Return recent sensor readings and predictions for charting.

    Returns time-series arrays for current, temperature, vibration,
    health_score, and risk_level over the requested time window.
    """
    # Determine how many records to fetch (approx 1 reading per second)
    limit = min(seconds, 500)

    # Fetch sensor history
    history = await asyncio.to_thread(
        firebase_client.get_history, device_id, limit,
    )

    # Fetch prediction history
    pred_history = await asyncio.to_thread(
        firebase_client.get_predictions_history, device_id, limit,
    )

    # Build time-series arrays
    sensor_series = {
        "timestamps": [],
        "current": [],
        "temperature": [],
        "vibration": [],
    }
    for record in history:
        sensor_series["timestamps"].append(
            record.get("timestamp", record.get("_key", ""))
        )
        sensor_series["current"].append(
            _safe_float(record.get("current"))
        )
        sensor_series["temperature"].append(
            _safe_float(record.get("temperature"))
        )
        sensor_series["vibration"].append(
            _safe_float(record.get("vibration"))
        )

    prediction_series = {
        "timestamps": [],
        "health_score": [],
        "risk_level": [],
    }
    for pred in pred_history:
        prediction_series["timestamps"].append(
            pred.get("timestamp", pred.get("_key", ""))
        )
        prediction_series["health_score"].append(
            _safe_float(pred.get("health_score"))
        )
        prediction_series["risk_level"].append(
            pred.get("risk_level", "UNKNOWN")
        )

    return JSONResponse(content={
        "device_id": device_id,
        "seconds": seconds,
        "sensor_data": sensor_series,
        "prediction_data": prediction_series,
    })


@app.get("/api/status")
async def api_status(
    device_id: str = Query(..., description="Device ID"),
) -> JSONResponse:
    """
    Return the latest prediction and live sensor data for a device.
    First checks cached predictions, falls back to Firebase.
    """
    # Try cache first
    prediction = scheduler.get_cached_prediction(device_id)

    # Fall back to Firebase
    if prediction is None:
        prediction = await asyncio.to_thread(
            firebase_client.get_predictions_latest, device_id,
        )

    # Live data
    live_data = await asyncio.to_thread(
        firebase_client.get_live_data, device_id,
    )

    if prediction is None and live_data is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Device '{device_id}' not found or no data"},
        )

    return JSONResponse(content={
        "device_id": device_id,
        "live_data": live_data,
        "prediction": prediction,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ═══════════════════════════════════════════════════════════════
# WEBSOCKET STREAMING
# ═══════════════════════════════════════════════════════════════

@app.websocket("/ws/live")
async def ws_live(
    websocket: WebSocket,
    device_id: str = Query(..., description="Device ID"),
):
    """
    WebSocket endpoint for real-time prediction streaming.

    Pushes the latest cached prediction + live data every scheduler
    interval. Client receives JSON messages continuously.
    """
    await websocket.accept()
    logger.info("WebSocket connected for device %s", device_id)

    try:
        while True:
            # Build payload from cache
            prediction = scheduler.get_cached_prediction(device_id)
            device_info = registry.get(device_id)

            payload: Dict[str, Any] = {
                "device_id": device_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prediction": prediction,
                "live_data": (
                    device_info.last_live_data if device_info else None
                ),
            }

            await websocket.send_json(payload)

            # Wait for the next scheduler tick
            await asyncio.sleep(config.SCHEDULER_INTERVAL_SECONDS)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for device %s", device_id)
    except Exception:
        logger.exception("WebSocket error for device %s", device_id)
        try:
            await websocket.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════
# CONVERSATIONAL AI ENDPOINT
# ═══════════════════════════════════════════════════════════════


class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    device_id: str
    message: str
    question_id: Optional[str] = None  # WHY_MAINTENANCE, IS_MACHINE_SAFE, etc.


@app.post("/api/chat")
async def api_chat(req: ChatRequest) -> JSONResponse:
    """
    Conversational AI endpoint — data-grounded answers about devices.

    The LLM receives a structured context with real-time sensor data,
    predictions, and thresholds. It is strictly instructed to answer
    ONLY from this data.
    """
    # Validate device exists
    device_info = registry.get(req.device_id)
    if device_info is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Device '{req.device_id}' not found"},
        )

    # Call the chat engine
    result = await chat_engine.chat(
        device_id=req.device_id,
        user_message=req.message,
        registry=registry,
        question_id=req.question_id,
    )

    return JSONResponse(content=result)


# ─── Helpers ─────────────────────────────────────────────────────

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float."""
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


# ─── Entry Point ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=True,
        log_level="info",
    )
