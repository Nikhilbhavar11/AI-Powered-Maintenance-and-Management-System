"""
ML Training Script — Offline RandomForest training.

Run this script ONCE (or periodically) to train a RandomForest
classifier from Firebase history data using weak-supervised labeling.

Workflow:
  1. Initialize Firebase
  2. Discover all devices
  3. Pull history for each device
  4. Apply weak-supervision threshold labeling to generate labels
  5. Build feature vectors via the analytics module
  6. Train a RandomForestClassifier
  7. Save the trained model to disk (models/rf_model.joblib)

Usage:
    python train_model.py

Design decisions:
  • Weak supervision: We label each history window as fault/no-fault
    using domain-expert thresholds (config.WEAK_LABEL_THRESHOLDS)
    rather than requiring manually labeled data.
  • Sliding window: We slide a window across each device's history
    to produce multiple training samples per device.
  • The model is a multi-class classifier predicting risk_level
    (0=LOW, 1=MEDIUM, 2=HIGH).
  • Model is saved via joblib for fast loading at FastAPI startup.
  • This script is NEVER called during inference.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import joblib

import config
import firebase_client
import analytics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Risk level encoding ────────────────────────────────────────
RISK_LABELS = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


def weak_label(features: Dict[str, Any]) -> int:
    """
    Assign a risk label (0/1/2) to a feature vector based on
    domain-expert thresholds.

    Rules (ordered by severity):
      HIGH (2):
        - vibration > threshold AND temperature trend is RISING
        - stress_index > 70
      MEDIUM (1):
        - vibration > threshold * 0.7
        - temperature > threshold * 0.85
        - current outside normal range
        - stress_index > 45
      LOW (0):
        - everything else
    """
    thresholds = config.WEAK_LABEL_THRESHOLDS

    vib = features.get("vibration", 0)
    temp = features.get("temperature", 0)
    current = features.get("current", 0)
    stress = features.get("stress_index", 0)
    temp_trend = features.get("temperature_trend", "STABLE")
    vib_trend = features.get("vibration_trend", "STABLE")

    # ── HIGH risk conditions ──
    if vib > thresholds["vibration_high"] and temp_trend == "RISING":
        return 2
    if stress > 70:
        return 2
    if (vib > thresholds["vibration_high"]
            and current > thresholds["current_abnormal_high"]):
        return 2

    # ── MEDIUM risk conditions ──
    if vib > thresholds["vibration_high"] * 0.7:
        return 1
    if temp > thresholds["temperature_high"] * 0.85:
        return 1
    if (current < thresholds["current_abnormal_low"]
            or current > thresholds["current_abnormal_high"] * 0.85):
        return 1
    if stress > 45:
        return 1

    # ── LOW risk ──
    return 0


def generate_training_data(
    all_histories: Dict[str, List[Dict[str, Any]]],
    window_size: int = config.HISTORY_FETCH_LIMIT,
    step: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (X, y) training data from multi-device histories
    using a sliding window approach.

    For each device, we slide a window across the history and at
    each position:
      1. Build a feature vector for the window
      2. Apply weak labeling

    Returns:
      X: np.ndarray of shape (n_samples, n_features)
      y: np.ndarray of shape (n_samples,) with values in {0, 1, 2}
    """
    X_list: List[List[float]] = []
    y_list: List[int] = []

    feature_names = analytics.get_feature_names()
    logger.info("Feature vector size: %d features", len(feature_names))

    for device_id, history in all_histories.items():
        if len(history) < 3:
            logger.warning(
                "Skipping %s — insufficient history (%d records)",
                device_id, len(history),
            )
            continue

        # Sliding window
        for end_idx in range(min(window_size, len(history)), len(history) + 1, step):
            start_idx = max(0, end_idx - window_size)
            window: List[Dict[str, Any]] = list(history[start_idx:end_idx])

            if len(window) < 3:
                continue

            features = analytics.build_feature_vector(window)
            label = weak_label(features)

            feature_array = analytics.feature_vector_to_array(features)
            X_list.append(feature_array)
            y_list.append(label)

    if not X_list:
        logger.error("No training samples generated!")
        return np.array([]), np.array([])

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)

    logger.info(
        "Training data: %d samples, %d features",
        X.shape[0], X.shape[1],
    )
    logger.info(
        "Label distribution: LOW=%d  MEDIUM=%d  HIGH=%d",
        np.sum(y == 0), np.sum(y == 1), np.sum(y == 2),
    )
    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier and report cross-validation accuracy.
    """
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",  # Handle class imbalance from weak labels
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation (if enough samples)
    if len(X) >= 10:
        cv_folds = min(5, len(X) // 3)
        if cv_folds >= 2:
            scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="accuracy")
            logger.info(
                "Cross-validation accuracy: %.3f ± %.3f  (k=%d)",
                scores.mean(), scores.std(), cv_folds,
            )

    # Final training on all data
    clf.fit(X, y)
    logger.info("Model trained on %d samples", len(X))

    # Feature importances
    feature_names = analytics.get_feature_names()
    importances = clf.feature_importances_
    ranked = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True,
    )
    logger.info("Top feature importances:")
    top_features = ranked[:5] if len(ranked) >= 5 else ranked  # type: ignore[index]
    for name, imp in top_features:
        logger.info("  %-30s  %.4f", name, imp)

    return clf


def save_model(clf: RandomForestClassifier) -> None:
    """Save trained model to disk."""
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, config.MODEL_PATH)
    logger.info("Model saved to %s", config.MODEL_PATH)


def main() -> None:
    """Full training pipeline."""
    logger.info("=" * 60)
    logger.info("PREDICTIVE MAINTENANCE — MODEL TRAINING")
    logger.info("=" * 60)

    # 1. Initialize Firebase
    firebase_client.init_firebase()

    # 2. Discover devices
    device_ids = firebase_client.list_device_ids()
    if not device_ids:
        logger.error("No devices found in Firebase. Aborting.")
        sys.exit(1)
    logger.info("Found %d device(s): %s", len(device_ids), device_ids)

    # 3. Pull history for all devices
    all_histories: Dict[str, List[Dict[str, Any]]] = {}
    for did in device_ids:
        history = firebase_client.get_history(did, limit=500)
        all_histories[did] = history
        logger.info("  %s: %d history records", did, len(history))

    # 4. Generate training data
    X, y = generate_training_data(all_histories)
    if X.size == 0:
        logger.error("No training data generated. Check Firebase history.")
        sys.exit(1)

    # 5. Train model
    clf = train_model(X, y)

    # 6. Save model
    save_model(clf)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
