"""
Background Scheduler — Real-time prediction loop.

Runs as an asyncio background task inside the FastAPI process.
Every SCHEDULER_INTERVAL_SECONDS (3–5 sec), for each registered
device:

  1. Pull recent history from Firebase
  2. Compute engineered features (analytics module)
  3. Run the ML model (ml_engine module)
  4. Write prediction back to Firebase
  5. Update the in-memory device registry cache

Design decisions:
  • Uses asyncio.to_thread() to offload the synchronous Firebase
    SDK calls to the default thread pool, keeping the event loop
    non-blocking.
  • Processes devices sequentially within each tick to avoid
    overwhelming Firebase with concurrent reads.
  • Maintains a separate in-memory prediction cache (dict) that
    the WebSocket handler can poll without Firebase round-trips.
  • Gracefully handles per-device errors without crashing the loop.
  • The scheduler task is started/stopped via start() and stop()
    lifecycle functions called from FastAPI's lifespan.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import config
import firebase_client
import analytics
import ml_engine
from device_registry import DeviceRegistry

logger = logging.getLogger(__name__)

# ─── In-memory prediction cache ─────────────────────────────────
# Key: device_id, Value: latest prediction dict
_prediction_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()

# ─── Scheduler state ────────────────────────────────────────────
_scheduler_task: Optional[asyncio.Task] = None
_running = False


# ─── Cache Access (thread-safe) ─────────────────────────────────

def get_cached_prediction(device_id: str) -> Optional[Dict[str, Any]]:
    """Return the latest cached prediction for a device."""
    with _cache_lock:
        return _prediction_cache.get(device_id)


def get_all_cached_predictions() -> Dict[str, Dict[str, Any]]:
    """Return a snapshot of the full prediction cache."""
    with _cache_lock:
        return dict(_prediction_cache)


def _update_cache(device_id: str, prediction: Dict[str, Any]) -> None:
    """Store a prediction in the cache."""
    with _cache_lock:
        _prediction_cache[device_id] = prediction


# ─── Single Device Tick ──────────────────────────────────────────

async def _process_device(
    device_id: str,
    registry: DeviceRegistry,
) -> Optional[Dict[str, Any]]:
    """
    Execute one prediction cycle for a single device.

    Returns the prediction dict, or None on error.
    """
    try:
        # 1. Pull recent history (sync → thread pool)
        history: List[Dict[str, Any]] = await asyncio.to_thread(
            firebase_client.get_history,
            device_id,
            config.HISTORY_FETCH_LIMIT,
        )
        if not history:
            logger.debug("No history for %s — skipping", device_id)
            return None

        # 1b. Also pull live data for registry caching
        live_data = await asyncio.to_thread(
            firebase_client.get_live_data,
            device_id,
        )
        if live_data:
            registry.update_live_data(device_id, live_data)

        # 2. Compute features
        features = analytics.build_feature_vector(history)

        # 3. Run ML model
        prediction = ml_engine.predict(features)

        # 4. Write to Firebase (sync → thread pool)
        await asyncio.to_thread(
            firebase_client.write_prediction,
            device_id,
            prediction,
        )

        # 5. Update caches
        _update_cache(device_id, prediction)
        registry.update_prediction(device_id, prediction)

        logger.info(
            "Tick  %s  |  health=%s  risk=%s  maint=%s",
            device_id,
            prediction["health_score"],
            prediction["risk_level"],
            prediction["maintenance_required"],
        )
        return prediction

    except Exception:
        logger.exception("Error processing device %s", device_id)
        return None


# ─── Main Loop ───────────────────────────────────────────────────

async def _scheduler_loop(registry: DeviceRegistry) -> None:
    """
    Infinite loop that runs every SCHEDULER_INTERVAL_SECONDS.
    Processes all registered devices each tick.
    """
    global _running
    interval = config.SCHEDULER_INTERVAL_SECONDS
    logger.info(
        "Scheduler started  |  interval=%ss  |  devices=%d",
        interval, registry.count(),
    )

    while _running:
        tick_start = asyncio.get_running_loop().time()

        # Refresh device list periodically (catches new devices)
        try:
            await asyncio.to_thread(registry.discover)
        except Exception:
            logger.exception("Error refreshing device registry")

        device_ids = registry.list_ids()
        for device_id in device_ids:
            if not _running:
                break
            await _process_device(device_id, registry)

        # Sleep for the remainder of the interval
        elapsed = asyncio.get_running_loop().time() - tick_start
        sleep_time = max(0.1, interval - elapsed)
        logger.debug(
            "Tick completed in %.2fs — sleeping %.2fs",
            elapsed, sleep_time,
        )
        await asyncio.sleep(sleep_time)

    logger.info("Scheduler stopped")


# ─── Lifecycle ───────────────────────────────────────────────────

def start(registry: DeviceRegistry) -> None:
    """
    Start the background scheduler as an asyncio task.
    Call this from FastAPI's lifespan startup.
    """
    global _scheduler_task, _running
    if _running:
        logger.warning("Scheduler already running")
        return

    _running = True
    loop = asyncio.get_running_loop()
    _scheduler_task = loop.create_task(_scheduler_loop(registry))
    logger.info("Scheduler task created")


async def stop() -> None:
    """
    Stop the background scheduler gracefully.
    Call this from FastAPI's lifespan shutdown.
    """
    global _running, _scheduler_task
    _running = False
    if _scheduler_task is not None:
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass
        _scheduler_task = None
    logger.info("Scheduler task cleaned up")
