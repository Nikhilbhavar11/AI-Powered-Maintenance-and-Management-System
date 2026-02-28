"""
Context Builder — Structured device context for LLM grounding.

Assembles a complete, data-grounded context object from the device
registry, cached predictions, and analytics results.  This JSON
context is injected into the LLM prompt so that the AI can ONLY
answer using real machine data.

Design decisions:
  • Pure data assembly — no LLM calls happen here.
  • Context includes: live sensor data, prediction results, feature
    summaries, trend information, and actionable thresholds.
  • The context is structured for easy reference in the system prompt.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import config
import scheduler
from device_registry import DeviceRegistry

logger = logging.getLogger(__name__)


def build_device_context(
    device_id: str,
    registry: DeviceRegistry,
) -> Dict[str, Any]:
    """
    Build a complete context object for a single device.

    This dict is injected verbatim into the LLM system prompt
    so the model can reference actual data.

    Structure:
      {
        "device_id": str,
        "timestamp": str,
        "live_sensors": {...},
        "prediction": {...},
        "thresholds": {...},
        "status_summary": str
      }
    """
    device_info = registry.get(device_id)
    prediction = scheduler.get_cached_prediction(device_id)

    # Live sensor data
    live = {}
    if device_info and device_info.last_live_data:
        raw = device_info.last_live_data
        live = {
            "current_amps": _sf(raw.get("current")),
            "temperature_celsius": _sf(raw.get("temperature")),
            "vibration_g": _sf(raw.get("vibration")),
            "sensor_timestamp": raw.get("timestamp", "N/A"),
        }

    # Prediction context
    pred = {}
    if prediction:
        pred = {
            "health_score": prediction.get("health_score"),
            "risk_level": prediction.get("risk_level"),
            "maintenance_required": prediction.get("maintenance_required"),
            "failure_reason": prediction.get("failure_reason"),
            "model_type": prediction.get("model_type"),
            "prediction_timestamp": prediction.get("timestamp"),
        }

        # Feature summary from prediction
        fs = prediction.get("features_summary", {})
        pred["features"] = {
            "current_trend": fs.get("current_trend", "N/A"),
            "temperature_trend": fs.get("temperature_trend", "N/A"),
            "vibration_trend": fs.get("vibration_trend", "N/A"),
            "stress_index": fs.get("stress_index", "N/A"),
        }

    # Operating thresholds (for LLM reference)
    thresholds = {
        "vibration_high_g": config.WEAK_LABEL_THRESHOLDS["vibration_high"],
        "temperature_high_celsius": config.WEAK_LABEL_THRESHOLDS["temperature_high"],
        "current_normal_range_amps": (
            f"{config.WEAK_LABEL_THRESHOLDS['current_abnormal_low']}–"
            f"{config.WEAK_LABEL_THRESHOLDS['current_abnormal_high']}"
        ),
    }

    # Human-readable status summary
    status_summary = _generate_status_summary(live, pred)

    return {
        "device_id": device_id,
        "context_generated_at": datetime.now(timezone.utc).isoformat(),
        "live_sensors": live,
        "prediction": pred,
        "operating_thresholds": thresholds,
        "status_summary": status_summary,
    }


def build_multi_device_context(
    registry: DeviceRegistry,
) -> Dict[str, Any]:
    """
    Build context for ALL devices (used for fleet-level questions).
    """
    device_ids = registry.list_ids()
    devices = {}
    for did in device_ids:
        devices[did] = build_device_context(did, registry)

    # Fleet summary
    total = len(device_ids)
    high_risk = sum(
        1 for d in devices.values()
        if d.get("prediction", {}).get("risk_level") == "HIGH"
    )
    medium_risk = sum(
        1 for d in devices.values()
        if d.get("prediction", {}).get("risk_level") == "MEDIUM"
    )
    maint_needed = sum(
        1 for d in devices.values()
        if d.get("prediction", {}).get("maintenance_required") is True
    )

    return {
        "fleet_summary": {
            "total_devices": total,
            "high_risk_count": high_risk,
            "medium_risk_count": medium_risk,
            "maintenance_required_count": maint_needed,
        },
        "devices": devices,
        "context_generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ─── Question ID Mapping ────────────────────────────────────────

QUESTION_TEMPLATES: Dict[str, str] = {
    "WHY_MAINTENANCE": (
        "Based on the provided machine data for {device_id}, explain "
        "exactly why maintenance is required. Reference specific sensor "
        "values, trends, and thresholds."
    ),
    "IS_MACHINE_SAFE": (
        "Based on the provided machine data for {device_id}, is this "
        "machine safe to operate right now? Reference health score, "
        "risk level, and any abnormal readings."
    ),
    "WHAT_IS_WRONG": (
        "Based on the provided machine data for {device_id}, what "
        "specific issues are detected? List each abnormal parameter "
        "with its current value vs. threshold."
    ),
    "WHAT_ACTION_REQUIRED": (
        "Based on the provided machine data for {device_id}, what "
        "specific actions should the maintenance team take? Prioritize "
        "by severity."
    ),
}


def resolve_question(
    question_id: Optional[str],
    user_message: str,
    device_id: str,
) -> str:
    """
    If the user message matches a known question_id, use the
    structured template. Otherwise, pass through the raw message.
    """
    if question_id and question_id.upper() in QUESTION_TEMPLATES:
        return QUESTION_TEMPLATES[question_id.upper()].format(
            device_id=device_id,
        )
    return user_message


# ─── Helpers ─────────────────────────────────────────────────────

def _sf(value: Any) -> Any:
    """Safe float conversion for context values."""
    try:
        if value is None:
            return None
        result: float = round(float(value), 3)
        return result
    except (TypeError, ValueError):
        return None


def _generate_status_summary(
    live: Dict[str, Any],
    pred: Dict[str, Any],
) -> str:
    """Generate a one-line human-readable status summary."""
    if not pred:
        if not live:
            return "No data available for this device."
        return "Live sensor data available but no prediction computed yet."

    risk = pred.get("risk_level", "UNKNOWN")
    health = pred.get("health_score", "N/A")
    maint = pred.get("maintenance_required", False)

    parts = [f"Risk: {risk}", f"Health: {health}/100"]
    if maint:
        parts.append("MAINTENANCE REQUIRED")
    else:
        parts.append("Operating normally")

    return " | ".join(parts)
