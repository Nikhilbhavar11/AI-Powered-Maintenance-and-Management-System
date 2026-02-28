"""
ML Inference Engine — Runtime prediction module.

Loaded once at FastAPI startup. Provides predictions for live
sensor data without retraining.

Responsibilities:
  • Load the persisted RandomForest model from disk
  • Accept a feature vector and return a structured prediction:
      health_score (0–100), risk_level, maintenance_required,
      failure_reason
  • Map model outputs to human-readable results

Design decisions:
  • Model is loaded ONCE at startup — no retraining during inference.
  • If no model file exists, the engine falls back to a rule-based
    heuristic so the system degrades gracefully.
  • The predict() function is pure — no Firebase/network calls.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

import config
import analytics

logger = logging.getLogger(__name__)

# ─── Module state ────────────────────────────────────────────────
_model: Optional[RandomForestClassifier] = None
_model_loaded: bool = False

# Risk level mapping (must match train_model.py)
RISK_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


# ─── Model Loading ──────────────────────────────────────────────

def load_model() -> bool:
    """
    Load the trained RandomForest model from disk.
    Returns True if successful, False if no model found (fallback mode).
    """
    global _model, _model_loaded

    model_path = config.MODEL_PATH
    if not Path(model_path).exists():
        logger.warning(
            "No trained model found at %s — using rule-based fallback",
            model_path,
        )
        _model = None
        _model_loaded = False
        return False

    _model = joblib.load(model_path)
    _model_loaded = True
    logger.info("ML model loaded from %s", model_path)
    return True


def is_model_loaded() -> bool:
    """Check if the ML model is available."""
    return _model_loaded


# ─── Prediction ──────────────────────────────────────────────────

def predict(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a prediction from engineered features.

    If an ML model is loaded, uses it for classification.
    Otherwise, falls back to rule-based heuristics.

    Args:
        features: Output of analytics.build_feature_vector()

    Returns:
        {
            "health_score": float (0–100),
            "risk_level": str ("LOW"/"MEDIUM"/"HIGH"),
            "maintenance_required": bool,
            "failure_reason": str,
            "model_type": str ("ml" or "rule_based"),
            "timestamp": str (ISO 8601),
            "features_summary": dict
        }
    """
    if _model is not None:
        return _predict_ml(features)
    return _predict_rule_based(features)


# ─── ML-Based Prediction ────────────────────────────────────────

def _predict_ml(features: Dict[str, Any]) -> Dict[str, Any]:
    """Use the trained RandomForest model."""
    assert _model is not None, "ML model not loaded"
    model = _model  # local ref for type narrowing

    feature_array = analytics.feature_vector_to_array(features)
    X = np.array([feature_array], dtype=np.float64)

    # Predicted class
    predicted_class = int(model.predict(X)[0])
    risk_level = RISK_LABELS.get(predicted_class, "UNKNOWN")

    # Class probabilities for health score
    probas = model.predict_proba(X)[0]
    # Health score: weighted inverse of risk probability
    # P(LOW)*100 + P(MEDIUM)*50 + P(HIGH)*0
    health_score = _compute_health_score(probas)

    # Maintenance decision
    maintenance_required = predicted_class >= 1
    failure_reason = _determine_failure_reason(features, risk_level)

    return _build_result(
        health_score=health_score,
        risk_level=risk_level,
        maintenance_required=maintenance_required,
        failure_reason=failure_reason,
        model_type="ml",
        features=features,
    )


def _compute_health_score(probas: np.ndarray) -> float:
    """
    Convert class probabilities to a 0–100 health score.

    Mapping:
      P(LOW)    contributes positively (health)
      P(MEDIUM) contributes moderately
      P(HIGH)   contributes negatively
    """
    weights = np.array([100.0, 50.0, 0.0])
    # Ensure probas has 3 elements (some edge cases with few classes)
    if len(probas) < 3:
        padded = np.zeros(3)
        padded[:len(probas)] = probas
        probas = padded

    score = float(np.dot(probas, weights))
    clamped: float = round(max(0.0, min(100.0, score)), 1)
    return clamped


# ─── Rule-Based Fallback ────────────────────────────────────────

def _predict_rule_based(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fallback prediction using threshold rules when no model is
    available. Uses the stress_index as primary signal.
    """
    stress = features.get("stress_index", 0.0)

    if stress > 70:
        risk_level = "HIGH"
        health_score = max(0, 100 - stress * 1.2)
    elif stress > 45:
        risk_level = "MEDIUM"
        health_score = max(20, 100 - stress)
    else:
        risk_level = "LOW"
        health_score = max(50, 100 - stress * 0.8)

    health_score = round(health_score, 1)
    maintenance_required = risk_level in ("MEDIUM", "HIGH")
    failure_reason = _determine_failure_reason(features, risk_level)

    return _build_result(
        health_score=health_score,
        risk_level=risk_level,
        maintenance_required=maintenance_required,
        failure_reason=failure_reason,
        model_type="rule_based",
        features=features,
    )


# ─── Failure Reason Determination ────────────────────────────────

def _determine_failure_reason(
    features: Dict[str, Any],
    risk_level: str,
) -> str:
    """
    Generate a human-readable failure reason based on which
    sensor channels are abnormal.
    """
    if risk_level == "LOW":
        return "All parameters within normal operating range."

    reasons: List[str] = []
    thresholds = config.WEAK_LABEL_THRESHOLDS

    # Check vibration
    vib = features.get("vibration", 0)
    vib_trend = features.get("vibration_trend", "STABLE")
    if vib > thresholds["vibration_high"]:
        reasons.append(
            f"High vibration detected ({vib:.2f}g, threshold: "
            f"{thresholds['vibration_high']}g, trend: {vib_trend})"
        )
    elif vib > thresholds["vibration_high"] * 0.7:
        reasons.append(
            f"Elevated vibration ({vib:.2f}g, approaching threshold)"
        )

    # Check temperature
    temp = features.get("temperature", 0)
    temp_trend = features.get("temperature_trend", "STABLE")
    if temp > thresholds["temperature_high"]:
        reasons.append(
            f"Overtemperature condition ({temp:.1f}°C, limit: "
            f"{thresholds['temperature_high']}°C, trend: {temp_trend})"
        )
    elif temp > thresholds["temperature_high"] * 0.85:
        reasons.append(
            f"Elevated temperature ({temp:.1f}°C, trend: {temp_trend})"
        )

    # Check current
    current = features.get("current", 0)
    if current < thresholds["current_abnormal_low"]:
        reasons.append(
            f"Abnormally low current draw ({current:.2f}A, min: "
            f"{thresholds['current_abnormal_low']}A) — possible open circuit"
        )
    elif current > thresholds["current_abnormal_high"]:
        reasons.append(
            f"Abnormally high current draw ({current:.2f}A, max: "
            f"{thresholds['current_abnormal_high']}A) — possible overload"
        )

    # Stress index
    stress = features.get("stress_index", 0)
    if stress > 60:
        reasons.append(f"Composite stress index elevated ({stress:.1f}/100)")

    if not reasons:
        reasons.append(
            "Multiple parameters trending toward abnormal thresholds."
        )

    return " | ".join(reasons)


# ─── Result Builder ──────────────────────────────────────────────

def _build_result(
    health_score: float,
    risk_level: str,
    maintenance_required: bool,
    failure_reason: str,
    model_type: str,
    features: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the standard prediction result dict."""
    return {
        "health_score": health_score,
        "risk_level": risk_level,
        "maintenance_required": maintenance_required,
        "failure_reason": failure_reason,
        "model_type": model_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "features_summary": {
            "current": features.get("current", 0),
            "temperature": features.get("temperature", 0),
            "vibration": features.get("vibration", 0),
            "stress_index": features.get("stress_index", 0),
            "current_trend": features.get("current_trend", "STABLE"),
            "temperature_trend": features.get("temperature_trend", "STABLE"),
            "vibration_trend": features.get("vibration_trend", "STABLE"),
        },
    }
