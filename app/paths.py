"""Central path constants for the PRANA modular layout."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Render persistent disk or custom data directory (writable on cloud hosts).
# Attach a Render disk at /var/data and set PRANA_DATA_DIR=/var/data
_DATA_ROOT = Path(os.environ.get("PRANA_DATA_DIR", PROJECT_ROOT / "database")).expanduser()
if not _DATA_ROOT.is_absolute():
    _DATA_ROOT = PROJECT_ROOT / _DATA_ROOT

DATABASE_DIR = _DATA_ROOT if os.environ.get("PRANA_DATA_DIR") else PROJECT_ROOT / "database"
DATASETS_DIR = PROJECT_ROOT / "datasets"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Writable paths — use DATABASE_DIR so SQLite survives on a Render persistent disk.
DB_PATH = str(DATABASE_DIR / "prana_records.db")
MODEL_PATH = str(DATABASE_DIR / "ecg_model.pkl")
REAL_ECG_MODEL_PATH = str(DATABASE_DIR / "real_ecg_model.pkl")

# Bundled read-only assets stay under the project tree.
LABELED_DATASET_CSV = str(DATASETS_DIR / "real_labeled_ecg_dataset.csv")
ECG_FEATURES_CSV = str(DATASETS_DIR / "real_ecg_features.csv")

# Per-modality PyTorch checkpoints (ship with deploy artifact or object storage).
MODULE_MODEL_PATHS = {
    "ECG": Path(MODEL_PATH),
    "EEG": PROJECT_ROOT / "modules" / "eeg" / "model" / "eeg_model.pth",
    "PPG": PROJECT_ROOT / "modules" / "ppg" / "model" / "ppg_model.pth",
    "SpO2": PROJECT_ROOT / "modules" / "spo2" / "model" / "spo2_model.pth",
    "Thermal": PROJECT_ROOT / "modules" / "thermal" / "model" / "thermal_model.pth",
}

# NATS messaging (optional — dashboard degrades gracefully when unavailable).
NATS_HOST = os.environ.get("NATS_HOST", "localhost")
NATS_PORT = int(os.environ.get("NATS_PORT", "4222"))


def ensure_runtime_dirs() -> None:
    """Create writable directories used at runtime (safe on ephemeral filesystems)."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for rel in ("eeg", "ppg", "spo2", "thermal"):
        (PROJECT_ROOT / "modules" / rel / "model").mkdir(parents=True, exist_ok=True)


def report_path(filename: str) -> str:
    """Return absolute path for a generated report artifact."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    return str(REPORTS_DIR / filename)
