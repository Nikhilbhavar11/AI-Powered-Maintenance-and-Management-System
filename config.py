"""
Configuration module for the Predictive Maintenance Platform.

Centralizes all settings: Firebase credentials, scheduler intervals,
model paths, and LLM configuration. Values can be overridden via
environment variables for production deployments.
"""

import os
from pathlib import Path

# ─── Project Root ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ─── Firebase Configuration ──────────────────────────────────────
FIREBASE_CREDENTIALS_PATH = os.getenv(
    "FIREBASE_CREDENTIALS_PATH",
    str(BASE_DIR / "firebase_credentials.json"),
)
FIREBASE_DATABASE_URL = os.getenv(
    "FIREBASE_DATABASE_URL",
    "https://ai-prediction-app-11e6a-default-rtdb.asia-southeast1.firebasedatabase.app/",  # Replace with your RTDB URL
)

# ─── Data Paths in Firebase RTDB ─────────────────────────────────
# Root node under which all devices reside
MACHINES_ROOT = "/machine"

# Per-device sub-paths (formatted with device_id at runtime)
LIVE_PATH_TEMPLATE = "/machine/{device_id}/live"
HISTORY_PATH_TEMPLATE = "/machine/{device_id}/history"
PREDICTIONS_LATEST_TEMPLATE = "/machine/{device_id}/predictions/latest"
PREDICTIONS_HISTORY_TEMPLATE = "/machine/{device_id}/predictions/history"

# ─── ML Model Configuration ─────────────────────────────────────
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "rf_model.joblib"

# Training parameters — calibrated to ESP32 sensor ranges
# Normal readings: current ~0.03-0.11A, temp ~28-33°C, vibration ~0.10-0.25g
WEAK_LABEL_THRESHOLDS = {
    "vibration_high": 0.50,        # g — high vibration for ESP32 accelerometer
    "temperature_high": 45.0,      # °C — elevated temperature threshold
    "current_abnormal_low": 0.01,  # A — abnormally low current draw
    "current_abnormal_high": 0.25, # A — abnormally high current draw
}

# ─── Analytics Configuration ─────────────────────────────────────
ROLLING_WINDOW_SIZE = 10          # Number of readings for rolling average
TREND_WINDOW_SIZE = 5             # Number of readings for trend detection
HISTORY_FETCH_LIMIT = 50          # Default number of history records to pull

# ─── Scheduler Configuration ─────────────────────────────────────
SCHEDULER_INTERVAL_SECONDS = float(os.getenv("SCHEDULER_INTERVAL", "4.0"))

# ─── LLM / Chat Configuration ───────────────────────────────────
# Supported providers: "groq", "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

# Groq (cloud) settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Ollama (local) settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

# ─── Server Configuration ───────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
