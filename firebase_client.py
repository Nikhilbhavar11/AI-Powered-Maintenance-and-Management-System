"""
Firebase Realtime Database client module.

Responsibilities:
  - Initialize Firebase Admin SDK (once, idempotent)
  - READ  /machines/{device_id}/live
  - READ  /machines/{device_id}/history
  - WRITE /machines/{device_id}/predictions/latest
  - WRITE /machines/{device_id}/predictions/history/{timestamp}
  - LIST  all device IDs under /machines

Design decisions:
  • Uses firebase-admin SDK with service-account credentials.
  • All functions are synchronous (Firebase Admin SDK is sync);
    the caller (scheduler, API) wraps them in asyncio.to_thread()
    when needed for async compatibility.
  • History reads are ordered by timestamp and limited to avoid
    pulling unbounded data.
  • Prediction writes are atomic per device — latest is overwritten,
    history is appended with a timestamp key.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import firebase_admin
from firebase_admin import credentials, db

import config

logger = logging.getLogger(__name__)

# ─── Module-level state ─────────────────────────────────────────
_firebase_app: Optional[firebase_admin.App] = None


# ─── Initialization ─────────────────────────────────────────────

def init_firebase() -> firebase_admin.App:
    """
    Initialize the Firebase Admin SDK.  Safe to call multiple times;
    subsequent calls return the existing app instance.
    """
    global _firebase_app
    if _firebase_app is not None:
        return _firebase_app

    cred = credentials.Certificate(config.FIREBASE_CREDENTIALS_PATH)
    _firebase_app = firebase_admin.initialize_app(cred, {
        "databaseURL": config.FIREBASE_DATABASE_URL,
    })
    logger.info(
        "Firebase initialized  |  DB URL: %s",
        config.FIREBASE_DATABASE_URL,
    )
    return _firebase_app


# ─── READ Operations ────────────────────────────────────────────

def list_device_ids() -> List[str]:
    """
    Return a list of all device IDs found under /machines.

    Each child key of /machines is treated as a device_id
    (e.g. "device_001", "device_002").
    """
    try:
        ref = db.reference(config.MACHINES_ROOT)
        snapshot: Optional[Dict] = ref.get(shallow=True)
        if not snapshot:
            logger.warning("No devices found under %s", config.MACHINES_ROOT)
            return []
        device_ids = sorted(snapshot.keys())
        logger.debug("Discovered %d device(s): %s", len(device_ids), device_ids)
        return device_ids
    except Exception as e:
        logger.warning("Could not list devices (%s) — database may be empty", e)
        return []


def get_live_data(device_id: str) -> Optional[Dict[str, Any]]:
    """
    Read the latest live sensor reading for a device.

    Returns dict with keys: current, temperature, vibration, timestamp
    or None if no live data exists.
    """
    path = config.LIVE_PATH_TEMPLATE.format(device_id=device_id)
    ref = db.reference(path)
    data = ref.get()
    if data is None:
        logger.warning("No live data for device %s", device_id)
    return data


def get_history(
    device_id: str,
    limit: int = config.HISTORY_FETCH_LIMIT,
) -> List[Dict[str, Any]]:
    """
    Read the most recent `limit` history entries for a device,
    ordered by key (timestamp).

    Returns a list of dicts sorted oldest → newest.
    Each dict contains: current, temperature, vibration, timestamp.
    """
    path = config.HISTORY_PATH_TEMPLATE.format(device_id=device_id)
    ref = db.reference(path)

    # Firebase orderByKey gives chronological order if keys are timestamps
    snapshot = ref.order_by_key().limit_to_last(limit).get()
    if not snapshot:
        logger.warning("No history for device %s", device_id)
        return []

    # snapshot is an OrderedDict {timestamp_key: {sensor_data}}
    records: List[Dict[str, Any]] = []
    for ts_key, reading in snapshot.items():
        if isinstance(reading, dict):
            reading["_key"] = ts_key
            records.append(reading)

    logger.debug(
        "Fetched %d history records for %s (limit=%d)",
        len(records), device_id, limit,
    )
    return records


def get_predictions_latest(device_id: str) -> Optional[Dict[str, Any]]:
    """
    Read the latest prediction written by the scheduler for a device.
    """
    path = config.PREDICTIONS_LATEST_TEMPLATE.format(device_id=device_id)
    return db.reference(path).get()


def get_predictions_history(
    device_id: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Read recent prediction history for a device.
    """
    path = config.PREDICTIONS_HISTORY_TEMPLATE.format(device_id=device_id)
    ref = db.reference(path)
    snapshot = ref.order_by_key().limit_to_last(limit).get()
    if not snapshot:
        return []
    return [
        {**v, "_key": k}
        for k, v in snapshot.items()
        if isinstance(v, dict)
    ]


# ─── WRITE Operations ───────────────────────────────────────────

def write_prediction(
    device_id: str,
    prediction: Dict[str, Any],
) -> None:
    """
    Write a prediction result for a device.

    1. Overwrites /machines/{device_id}/predictions/latest
    2. Appends to /machines/{device_id}/predictions/history/{timestamp}

    The prediction dict should contain:
      health_score, risk_level, maintenance_required,
      failure_reason, timestamp, features (summary).
    """
    ts = prediction.get(
        "timestamp",
        datetime.now(timezone.utc).isoformat(),
    )
    # Sanitize timestamp for use as Firebase key (no dots or special chars)
    ts_key = ts.replace(".", "_").replace(":", "-").replace("+", "p")

    # 1. Overwrite latest
    latest_path = config.PREDICTIONS_LATEST_TEMPLATE.format(device_id=device_id)
    db.reference(latest_path).set(prediction)

    # 2. Append to history
    history_path = config.PREDICTIONS_HISTORY_TEMPLATE.format(device_id=device_id)
    db.reference(f"{history_path}/{ts_key}").set(prediction)

    logger.info(
        "Prediction written for %s  |  health=%s  risk=%s",
        device_id,
        prediction.get("health_score"),
        prediction.get("risk_level"),
    )
