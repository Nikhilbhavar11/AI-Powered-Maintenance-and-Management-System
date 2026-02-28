"""
Device Registry module.

Maintains an in-memory registry of all known IoT devices.
Discovers devices from Firebase RTDB on startup and supports
periodic refresh to detect newly-added devices at runtime.

Design decisions:
  • Thread-safe via threading.Lock to support concurrent reads
    from the scheduler and API handlers.
  • Stores a lightweight DeviceInfo dataclass per device to hold
    metadata and last-seen timestamps.
  • Singleton-pattern registry—one instance is created at startup
    and shared across the application.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import firebase_client

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Metadata for a single registered device."""
    device_id: str
    first_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_live_data: Optional[dict] = None
    last_prediction: Optional[dict] = None


class DeviceRegistry:
    """
    In-memory registry of IoT devices.

    Usage:
        registry = DeviceRegistry()
        registry.discover()            # Pull device list from Firebase
        devices = registry.list_all()  # Get all DeviceInfo objects
        info = registry.get("device_001")
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._devices: Dict[str, DeviceInfo] = {}

    # ─── Discovery ───────────────────────────────────────────

    def discover(self) -> List[str]:
        """
        Query Firebase for all device IDs under /machines and
        register any that are not already known.

        Returns the full list of device IDs after discovery.
        """
        device_ids: List[str] = firebase_client.list_device_ids()
        with self._lock:
            for did in device_ids:
                device_id_str: str = str(did)
                if device_id_str not in self._devices:
                    self._devices[device_id_str] = DeviceInfo(device_id=device_id_str)
                    logger.info("Registered new device: %s", device_id_str)
        return self.list_ids()

    # ─── Accessors ───────────────────────────────────────────

    def list_ids(self) -> List[str]:
        """Return sorted list of all registered device IDs."""
        with self._lock:
            return sorted(self._devices.keys())

    def list_all(self) -> List[DeviceInfo]:
        """Return all DeviceInfo objects, sorted by device_id."""
        with self._lock:
            return [
                self._devices[k]
                for k in sorted(self._devices.keys())
            ]

    def get(self, device_id: str) -> Optional[DeviceInfo]:
        """Return DeviceInfo for a specific device, or None."""
        with self._lock:
            return self._devices.get(device_id)

    def count(self) -> int:
        """Number of registered devices."""
        with self._lock:
            return len(self._devices)

    # ─── Updates (called by scheduler) ───────────────────────

    def update_live_data(
        self, device_id: str, data: dict,
    ) -> None:
        """Cache the most recent live sensor reading."""
        with self._lock:
            info = self._devices.get(device_id)
            if info:
                info.last_live_data = data

    def update_prediction(
        self, device_id: str, prediction: dict,
    ) -> None:
        """Cache the most recent prediction result."""
        with self._lock:
            info = self._devices.get(device_id)
            if info:
                info.last_prediction = prediction

    # ─── Serialization (for API responses) ───────────────────

    def to_summary_list(self) -> List[dict]:
        """
        Return a JSON-serializable list of device summaries
        suitable for the /api/devices endpoint.
        """
        with self._lock:
            result: List[dict] = []
            for did in sorted(self._devices.keys()):
                info = self._devices[did]
                entry: Dict[str, Any] = {
                    "device_id": info.device_id,
                    "first_seen": info.first_seen,
                    "has_live_data": info.last_live_data is not None,
                    "has_prediction": info.last_prediction is not None,
                }
                pred = info.last_prediction
                if pred is not None:
                    entry["health_score"] = pred.get("health_score")
                    entry["risk_level"] = pred.get("risk_level")
                live = info.last_live_data
                if live is not None:
                    entry["last_reading"] = {
                        "current": live.get("current"),
                        "temperature": live.get("temperature"),
                        "vibration": live.get("vibration"),
                        "timestamp": live.get("timestamp"),
                    }
                result.append(entry)
            return result
