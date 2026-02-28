"""
Analytics & Feature Engineering module.

Transforms raw sensor history into ML-ready features:
  • Rolling averages (current, temperature, vibration)
  • Deltas / rate-of-change between consecutive readings
  • Trend detection (RISING / FALLING / STABLE)
  • Composite stress index (0–100)

Design decisions:
  • Pure functions — no side effects, no Firebase access.
  • Operates on a list of sensor dicts (as returned by
    firebase_client.get_history).
  • Uses numpy for efficient vectorized math.
  • Returns a flat feature dict ready for ML model input.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

import config

logger = logging.getLogger(__name__)

# Sensor columns that appear in every reading
SENSOR_KEYS = ("current", "temperature", "vibration")


# ─── Utility: safe extraction ───────────────────────────────────

def _extract_series(
    history: List[Dict[str, Any]],
    key: str,
) -> np.ndarray:
    """
    Extract a numeric series for `key` from the history list.
    Non-numeric values are replaced with NaN.
    """
    values = []
    for record in history:
        try:
            values.append(float(record.get(key, np.nan)))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.array(values, dtype=np.float64)


# ─── Rolling Averages ───────────────────────────────────────────

def compute_rolling_averages(
    history: List[Dict[str, Any]],
    window: int = config.ROLLING_WINDOW_SIZE,
) -> Dict[str, float]:
    """
    Compute rolling mean over the last `window` readings for each
    sensor channel.

    Returns:
        {"current_rolling_avg": ..., "temperature_rolling_avg": ...,
         "vibration_rolling_avg": ...}
    """
    result: Dict[str, float] = {}
    for key in SENSOR_KEYS:
        series = _extract_series(history, key)
        tail = series[-window:] if len(series) >= window else series
        # np.nanmean ignores NaN values gracefully
        result[f"{key}_rolling_avg"] = float(np.nanmean(tail)) if len(tail) > 0 else 0.0
    return result


# ─── Deltas (Rate of Change) ────────────────────────────────────

def compute_deltas(
    history: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute the rate of change between the last two readings
    for each sensor channel.

    Returns:
        {"current_delta": ..., "temperature_delta": ...,
         "vibration_delta": ...}
    """
    result: Dict[str, float] = {}
    for key in SENSOR_KEYS:
        series = _extract_series(history, key)
        if len(series) >= 2:
            delta = float(series[-1] - series[-2])
        else:
            delta = 0.0
        result[f"{key}_delta"] = delta
    return result


# ─── Trend Detection ────────────────────────────────────────────

def detect_trend(
    values: np.ndarray,
    window: int = config.TREND_WINDOW_SIZE,
) -> str:
    """
    Classify the trend over the last `window` values.

    Uses a simple linear regression slope:
      slope > +threshold → RISING
      slope < -threshold → FALLING
      otherwise          → STABLE

    Threshold is 1% of the mean value (or 0.01 absolute).
    """
    tail = values[-window:] if len(values) >= window else values
    tail = tail[~np.isnan(tail)]
    if len(tail) < 2:
        return "STABLE"

    x = np.arange(len(tail), dtype=np.float64)
    # np.polynomial.polynomial.polyfit returns coefficients in ascending order
    # so index [1] is the linear coefficient (slope)
    coeffs = np.polynomial.polynomial.polyfit(x, tail, deg=1)
    slope = float(coeffs[1])

    mean_val = float(np.mean(np.abs(tail)))
    threshold = max(mean_val * 0.01, 0.01)

    if slope > threshold:
        return "RISING"
    elif slope < -threshold:
        return "FALLING"
    return "STABLE"


def compute_trends(
    history: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Detect trend for each sensor channel.

    Returns:
        {"current_trend": "STABLE", "temperature_trend": "RISING",
         "vibration_trend": "FALLING"}
    """
    result: Dict[str, str] = {}
    for key in SENSOR_KEYS:
        series = _extract_series(history, key)
        result[f"{key}_trend"] = detect_trend(series)
    return result


# ─── Composite Stress Index ─────────────────────────────────────

def compute_stress_index(
    history: List[Dict[str, Any]],
    rolling_avgs: Optional[Dict[str, float]] = None,
    trends: Optional[Dict[str, str]] = None,
) -> float:
    """
    Compute a composite stress index (0–100) reflecting overall
    machine strain.

    Formula (weighted):
      40% — vibration contribution (normalized against threshold)
      35% — temperature contribution
      25% — current anomaly contribution

    Trend modifiers:
      RISING trend on any sensor adds +5 per rising channel.
    """
    if rolling_avgs is None:
        rolling_avgs = compute_rolling_averages(history)
    if trends is None:
        trends = compute_trends(history)

    thresholds = config.WEAK_LABEL_THRESHOLDS

    # Vibration stress: ratio of avg vibration to high threshold
    vib_avg = rolling_avgs.get("vibration_rolling_avg", 0.0)
    vib_stress = min(vib_avg / thresholds["vibration_high"], 1.5) * 40

    # Temperature stress: ratio of avg temp to high threshold
    temp_avg = rolling_avgs.get("temperature_rolling_avg", 0.0)
    temp_stress = min(temp_avg / thresholds["temperature_high"], 1.5) * 35

    # Current stress: deviation from normal range
    cur_avg = rolling_avgs.get("current_rolling_avg", 0.0)
    cur_low = thresholds["current_abnormal_low"]
    cur_high = thresholds["current_abnormal_high"]
    if cur_avg < cur_low:
        cur_stress = (1 - cur_avg / cur_low) * 25
    elif cur_avg > cur_high:
        cur_stress = min((cur_avg / cur_high), 1.5) * 25
    else:
        # Normal range — low stress
        mid = (cur_low + cur_high) / 2
        cur_stress = abs(cur_avg - mid) / (cur_high - cur_low) * 10

    raw_score = vib_stress + temp_stress + cur_stress

    # Trend modifiers
    trend_bonus = sum(
        5 for t in trends.values() if t == "RISING"
    )

    stress = min(raw_score + trend_bonus, 100.0)
    return round(max(stress, 0.0), 2)


# ─── Full Feature Vector ────────────────────────────────────────

def build_feature_vector(
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Full analytics pipeline: takes raw history and returns a flat
    dict containing all engineered features for ML consumption.

    Output keys:
      Sensor latest values:
        current, temperature, vibration

      Rolling averages:
        current_rolling_avg, temperature_rolling_avg, vibration_rolling_avg

      Deltas:
        current_delta, temperature_delta, vibration_delta

      Trends (encoded as int):
        current_trend_encoded, temperature_trend_encoded, vibration_trend_encoded
        (FALLING=-1, STABLE=0, RISING=1)

      Composite:
        stress_index
    """
    if not history:
        logger.warning("Empty history — returning zero feature vector")
        return _zero_feature_vector()

    # Latest reading
    latest = history[-1]
    features: Dict[str, Any] = {
        "current": float(latest.get("current", 0)),
        "temperature": float(latest.get("temperature", 0)),
        "vibration": float(latest.get("vibration", 0)),
    }

    # Rolling averages
    rolling = compute_rolling_averages(history)
    features.update(rolling)

    # Deltas
    deltas = compute_deltas(history)
    features.update(deltas)

    # Trends (with numeric encoding for ML)
    trends = compute_trends(history)
    trend_map = {"FALLING": -1, "STABLE": 0, "RISING": 1}
    for key in SENSOR_KEYS:
        trend_label = trends[f"{key}_trend"]
        features[f"{key}_trend"] = trend_label
        features[f"{key}_trend_encoded"] = trend_map.get(trend_label, 0)

    # Stress index
    features["stress_index"] = compute_stress_index(
        history, rolling_avgs=rolling, trends=trends,
    )

    return features


def get_feature_names() -> List[str]:
    """
    Return the ordered list of numeric feature names expected by the
    ML model. This must stay in sync with build_feature_vector().
    """
    names = []
    # Latest sensor values
    for key in SENSOR_KEYS:
        names.append(key)
    # Rolling averages
    for key in SENSOR_KEYS:
        names.append(f"{key}_rolling_avg")
    # Deltas
    for key in SENSOR_KEYS:
        names.append(f"{key}_delta")
    # Trend encoded
    for key in SENSOR_KEYS:
        names.append(f"{key}_trend_encoded")
    # Stress index
    names.append("stress_index")
    return names


def feature_vector_to_array(features: Dict[str, Any]) -> List[float]:
    """
    Convert a feature dict to an ordered numeric list matching
    get_feature_names().
    """
    return [float(features.get(name, 0.0)) for name in get_feature_names()]


# ─── Private helpers ─────────────────────────────────────────────

def _zero_feature_vector() -> Dict[str, Any]:
    """Return a feature vector with all zeros."""
    features = {}
    for key in SENSOR_KEYS:
        features[key] = 0.0
        features[f"{key}_rolling_avg"] = 0.0
        features[f"{key}_delta"] = 0.0
        features[f"{key}_trend"] = "STABLE"
        features[f"{key}_trend_encoded"] = 0
    features["stress_index"] = 0.0
    return features
