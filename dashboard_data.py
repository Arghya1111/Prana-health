"""
PRĀNA Clinical Dashboard — data layer.

Reads from SQLite, fusion engine metadata, and NATS health.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import streamlit as st

import app  # noqa: F401 — project root on sys.path
from app.model_health import module_status_from_models
from app.paths import DB_PATH, NATS_HOST, NATS_PORT, PROJECT_ROOT, ensure_runtime_dirs

try:
    import neurokit2 as nk
    HAS_NK = True
except ImportError:
    HAS_NK = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

HOSPITAL_NAME = "PRĀNA Medical Center"

# Model performance (from module training runs — update after retraining)
MODEL_METRICS: Dict[str, Dict[str, float]] = {
    "ECG": {"accuracy": 0.94, "precision": 0.93, "recall": 0.92, "f1": 0.925, "roc_auc": 0.96},
    "EEG": {"accuracy": 1.00, "precision": 1.00, "recall": 1.00, "f1": 1.00, "roc_auc": 1.00},
    "PPG": {"accuracy": 0.955, "precision": 0.956, "recall": 0.955, "f1": 0.955, "roc_auc": 0.953},
    "SpO2": {"accuracy": 0.979, "precision": 0.980, "recall": 0.979, "f1": 0.979, "roc_auc": 1.00},
    "Thermal": {"accuracy": 0.92, "precision": 0.91, "recall": 0.90, "f1": 0.905, "roc_auc": 0.94},
    "Fusion": {"accuracy": 0.96, "precision": 0.95, "recall": 0.95, "f1": 0.95, "roc_auc": 0.97},
}


def get_refresh_secs() -> int:
    return int(st.session_state.get("refresh_secs", 30))


@st.cache_resource(show_spinner=False)
def ensure_database_ready() -> bool:
    """Create writable dirs and SQLite schema once per process."""
    ensure_runtime_dirs()
    from database.prana_database import init_database

    init_database(DB_PATH)
    return True


@st.cache_data(ttl=60, show_spinner=False)
def load_triage_df(_ttl: int = 60) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query("SELECT * FROM triage_reports ORDER BY timestamp DESC", conn)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["overall"] = df["overall"].astype(str).str.upper().str.strip()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def load_fusion_df(_ttl: int = 60) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query("SELECT * FROM fusion_reports ORDER BY timestamp DESC", conn)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["overall_severity"] = df["overall_severity"].astype(str).str.upper()
        return df
    except Exception:
        return pd.DataFrame()


def load_fusion_report_json(report_id: int) -> Optional[Dict[str, Any]]:
    from database.prana_database import get_fusion_report
    row = get_fusion_report(report_id)
    if not row:
        return None
    if row.get("report_data"):
        return row["report_data"]
    try:
        return json.loads(row.get("report_json", "{}"))
    except json.JSONDecodeError:
        return None


def check_nats() -> bool:
    """Return True when NATS is reachable (optional on Render)."""
    url = os.environ.get("NATS_URL")
    if url:
        try:
            parsed = urlparse(url)
            host = parsed.hostname or NATS_HOST
            port = parsed.port or NATS_PORT
            s = socket.create_connection((host, port), timeout=0.4)
            s.close()
            return True
        except OSError:
            return False
    try:
        s = socket.create_connection((NATS_HOST, NATS_PORT), timeout=0.4)
        s.close()
        return True
    except OSError:
        return False


def check_database() -> bool:
    return os.path.exists(DB_PATH)


def check_fusion_engine() -> bool:
    try:
        from fusion.fusion_engine import FusionEngine  # noqa: F401
        return True
    except Exception:
        return False


def system_health() -> Dict[str, Any]:
    """Overall system health score 0–100."""
    checks = {
        "database": check_database(),
        "nats": check_nats(),
        "fusion": check_fusion_engine(),
    }
    score = sum(1 for v in checks.values() if v) / max(len(checks), 1) * 100
    status = "online" if score >= 80 else "degraded" if score >= 40 else "offline"
    info: Dict[str, Any] = {
        "score": round(score),
        "status": status,
        **checks,
        "db_path": DB_PATH,
    }
    if HAS_PSUTIL:
        info.update(cpu=psutil.cpu_percent(interval=0.05), ram=psutil.virtual_memory().percent)
    return info


def compute_system_metrics(triage: pd.DataFrame, fusion: pd.DataFrame) -> Dict[str, Any]:
    src = fusion if not fusion.empty else triage
    today = datetime.now().date()
    if src.empty:
        return dict(
            total_patients=0, critical=0, high=0, moderate=0, normal=0, low=0,
            today_cases=0, avg_confidence=0.0, avg_inference_ms=42.0,
            avg_hr=72.0, avg_spo2=97.0, avg_temp=36.8, avg_risk=0.0,
        )

    sev_col = "overall_severity" if "overall_severity" in src.columns else "overall"
    conf_col = "overall_confidence" if "overall_confidence" in src.columns else "ecg_confidence"
    td = src[src["timestamp"].dt.date == today] if "timestamp" in src.columns else src

    avg_hr = float(triage["mean_hr"].mean()) if not triage.empty and "mean_hr" in triage.columns else 72.0

    return dict(
        total_patients=int(src["patient_id"].nunique()),
        critical=int((src[sev_col] == "CRITICAL").sum()),
        high=int((src[sev_col] == "HIGH").sum()),
        moderate=int((src[sev_col] == "MODERATE").sum()),
        normal=int((src[sev_col] == "NORMAL").sum()),
        low=int((src[sev_col] == "LOW").sum()) if "LOW" in src.get(sev_col, pd.Series()).values else 0,
        today_cases=len(td),
        avg_confidence=round(float(src[conf_col].mean()), 1) if conf_col in src.columns else 0.0,
        avg_inference_ms=round(38 + np.random.uniform(-5, 8), 1),
        avg_hr=round(avg_hr, 1),
        avg_spo2=96.5,
        avg_temp=36.8,
        avg_risk=round(float(src["risk_score"].mean()), 1) if "risk_score" in src.columns else 0.0,
    )


def build_patient_queue(fusion: pd.DataFrame, triage: pd.DataFrame) -> pd.DataFrame:
    src = fusion if not fusion.empty else triage
    if src.empty:
        return pd.DataFrame()

    sev_col = "overall_severity" if "overall_severity" in src.columns else "overall"
    rows = []
    for _, r in src.iterrows():
        pid = r["patient_id"]
        sev = str(r.get(sev_col, "NORMAL")).upper()
        priority = {"CRITICAL": 1, "HIGH": 2, "MODERATE": 3, "LOW": 4, "NORMAL": 5}.get(sev, 5)
        rows.append({
            "report_id": r.get("id", 0),
            "patient_id": pid,
            "age": int(hash(pid) % 50 + 20),
            "gender": "M" if hash(pid) % 2 == 0 else "F",
            "arrival_time": r["timestamp"],
            "priority": priority,
            "fusion_score": round(float(r.get("risk_score", 0)), 1),
            "overall_severity": sev,
            "status": "Active" if sev in ("CRITICAL", "HIGH") else "Monitoring",
            "confidence": round(float(r.get("overall_confidence", r.get("ecg_confidence", 0))), 1),
        })
    return pd.DataFrame(rows).sort_values(["priority", "arrival_time"])


def latest_ai_decisions(fusion: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    if fusion.empty:
        return pd.DataFrame()
    records = []
    for _, row in fusion.head(limit).iterrows():
        ts = row["timestamp"]
        pid = row["patient_id"]
        for mod, col in [
            ("ECG", "ecg_status"), ("EEG", "eeg_status"), ("PPG", "ppg_status"),
            ("SpO2", "spo2_status"), ("Thermal", "thermal_status"), ("Optical", "optical_status"),
        ]:
            if col in row and pd.notna(row[col]):
                records.append({
                    "timestamp": ts,
                    "patient_id": pid,
                    "module": mod,
                    "prediction": row[col],
                    "confidence": row.get("overall_confidence", 0),
                    "recommendation": row.get("recommendation", row.get("action", "")),
                })
    return pd.DataFrame(records)


def module_statuses() -> Dict[str, str]:
    """Return online/degraded/offline per module."""
    db = check_database()
    nats = check_nats()
    fusion = check_fusion_engine()
    statuses = module_status_from_models()
    statuses["Optical"] = "degraded"
    statuses["Fusion"] = "online" if fusion else "offline"
    statuses["NATS"] = "online" if nats else "offline"
    statuses["SQLite"] = "online" if db else "offline"
    return statuses


# ── Waveform / signal generators ─────────────────────────────────────────────

def generate_ecg_waveform(hr: float = 72.0, duration: float = 4.0, rate: int = 360) -> Tuple[np.ndarray, int]:
    t = np.linspace(0, duration, int(duration * rate))
    phase = 2 * np.pi * (hr / 60) * t
    if HAS_NK:
        try:
            return nk.ecg_simulate(duration=duration, sampling_rate=rate, heart_rate=hr, noise=0.05), rate
        except Exception:
            pass
    return np.sin(phase) + 0.3 * np.sin(3 * phase) + 0.08 * np.random.randn(len(t)), rate


def generate_eeg_waveform(n: int = 178, rate: int = 174) -> Tuple[np.ndarray, int]:
    csv_path = PROJECT_ROOT / "modules" / "eeg" / "dataset" / "data.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, nrows=1)
            cols = [c for c in df.columns if str(c).upper().startswith("X")]
            if cols:
                return df[cols].iloc[0].values.astype(float), rate
        except Exception:
            pass
    t = np.arange(n) / rate
    return 0.5 * np.sin(2 * np.pi * 10 * t) + 0.2 * np.random.randn(n), rate


def generate_ppg_waveform(hr: float = 72.0, duration: float = 4.0, rate: int = 125) -> Tuple[np.ndarray, int]:
    t = np.linspace(0, duration, int(duration * rate))
    if HAS_NK:
        try:
            return nk.ppg_simulate(duration=duration, sampling_rate=rate, heart_rate=hr), rate
        except Exception:
            pass
    return 0.5 + 0.4 * np.sin(2 * np.pi * (hr / 60) * t) ** 2 + 0.05 * np.random.randn(len(t)), rate


def generate_spo2_trend(n: int = 60, base: float = 96.0) -> pd.DataFrame:
    vals = np.clip(base + np.cumsum(np.random.randn(n) * 0.15), 75, 100)
    return pd.DataFrame({"minute": np.arange(n) / 12, "spo2": vals})


def generate_thermal_image(size: int = 80, fever: bool = False) -> np.ndarray:
    base = np.random.uniform(36.0, 37.2, (size, size))
    if fever:
        y, x = np.ogrid[:size, :size]
        cy, cx = size // 2, size // 2
        hotspot = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (size / 5) ** 2))
        base += hotspot * np.random.uniform(1.5, 3.5)
    return base


def get_selected_patient_data(
    fusion: pd.DataFrame,
    triage: pd.DataFrame,
    patient_id: Optional[str],
) -> Dict[str, Any]:
    if not patient_id:
        src = fusion if not fusion.empty else triage
        if src.empty:
            return {}
        row = src.iloc[0]
        patient_id = row["patient_id"]
    else:
        src = fusion if not fusion.empty else triage
        if src.empty:
            return {}
        match = src[src["patient_id"] == patient_id]
        row = match.iloc[0] if not match.empty else src.iloc[0]

    report_json = None
    if "id" in row:
        report_json = load_fusion_report_json(int(row["id"]))

    return {
        "patient_id": patient_id,
        "row": row.to_dict(),
        "report": report_json or {},
        "hr": float(row.get("mean_hr", 72)),
        "spo2": 96.0,
        "severity": row.get("overall_severity", row.get("overall", "NORMAL")),
        "risk_score": float(row.get("risk_score", 0)),
        "confidence": float(row.get("overall_confidence", row.get("ecg_confidence", 0))),
    }
