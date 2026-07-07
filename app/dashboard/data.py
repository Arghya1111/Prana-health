"""Data loaders and signal generators for the PRĀNA dashboard."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import app  # noqa: F401
from app.paths import DB_PATH, MODEL_PATH, PROJECT_ROOT

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


def get_refresh_secs() -> int:
    return int(st.session_state.get("refresh_secs", 5))


@st.cache_data(ttl=5, show_spinner=False)
def load_triage_df(_ttl: int = 5) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM triage_reports ORDER BY timestamp DESC", conn
            )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["overall"] = df["overall"].astype(str).str.upper().str.strip()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=5, show_spinner=False)
def load_fusion_df(_ttl: int = 5) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM fusion_reports ORDER BY timestamp DESC", conn
            )
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


def compute_kpis(triage: pd.DataFrame, fusion: pd.DataFrame) -> Dict[str, Any]:
    src = fusion if not fusion.empty else triage
    if src.empty:
        return dict(
            total=0, today=0, critical=0, high=0, moderate=0, normal=0, low=0,
            avg_risk=0.0, avg_conf=0.0, patients=0,
        )
    today = datetime.now().date()
    sev_col = "overall_severity" if "overall_severity" in src.columns else "overall"
    conf_col = "overall_confidence" if "overall_confidence" in src.columns else "ecg_confidence"
    td = src[src["timestamp"].dt.date == today] if "timestamp" in src.columns else src
    return dict(
        total=len(src),
        today=len(td),
        critical=int((src[sev_col] == "CRITICAL").sum()),
        high=int((src[sev_col] == "HIGH").sum()),
        moderate=int((src[sev_col] == "MODERATE").sum()),
        low=int((src[sev_col] == "LOW").sum()) if "LOW" in src[sev_col].values else 0,
        normal=int((src[sev_col] == "NORMAL").sum()),
        avg_risk=round(float(src["risk_score"].mean()), 1) if "risk_score" in src.columns else 0.0,
        avg_conf=round(float(src[conf_col].mean()), 1) if conf_col in src.columns else 0.0,
        patients=src["patient_id"].nunique(),
    )


def check_nats() -> bool:
    try:
        s = socket.create_connection(("localhost", 4222), timeout=0.5)
        s.close()
        return True
    except OSError:
        return False


def system_info() -> Dict[str, Any]:
    info = dict(
        db_exists=os.path.exists(DB_PATH),
        model_exists=os.path.exists(MODEL_PATH),
        nats_live=check_nats(),
        db_size_mb=round(os.path.getsize(DB_PATH) / 1024 / 1024, 2) if os.path.exists(DB_PATH) else 0,
    )
    if HAS_PSUTIL:
        info.update(
            cpu=psutil.cpu_percent(interval=0.1),
            ram=psutil.virtual_memory().percent,
            disk=psutil.disk_usage(".").percent,
        )
    return info


def latest_patient_module_status(fusion: pd.DataFrame, patient_id: str) -> Dict[str, Any]:
    if fusion.empty:
        return {}
    rows = fusion[fusion["patient_id"] == patient_id]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def filter_dataframe(
    df: pd.DataFrame,
    search: str = "",
    severity: str = "All",
    sev_col: str = "overall",
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if severity != "All" and sev_col in out.columns:
        out = out[out[sev_col].str.upper() == severity.upper()]
    if search.strip():
        q = search.strip().upper()
        mask = out["patient_id"].astype(str).str.upper().str.contains(q, na=False)
        for col in out.columns:
            if out[col].dtype == object:
                mask |= out[col].astype(str).str.upper().str.contains(q, na=False)
        out = out[mask]
    return out


# ── Waveform generators ───────────────────────────────────────────────────────

def generate_ecg_waveform(hr: float = 72.0, duration: float = 10.0, rate: int = 360) -> Tuple[np.ndarray, int]:
    if HAS_NK:
        return nk.ecg_simulate(duration=duration, sampling_rate=rate, heart_rate=hr, noise=0.06), rate
    t = np.linspace(0, duration, int(duration * rate))
    return np.sin(2 * np.pi * (hr / 60) * t) + 0.1 * np.random.randn(len(t)), rate


def generate_ppg_waveform(hr: float = 72.0, duration: float = 10.0, rate: int = 125) -> Tuple[np.ndarray, int]:
    if HAS_NK:
        return nk.ppg_simulate(duration=duration, sampling_rate=rate, heart_rate=hr), rate
    t = np.linspace(0, duration, int(duration * rate))
    return 0.5 + 0.4 * np.sin(2 * np.pi * (hr / 60) * t) ** 2 + 0.05 * np.random.randn(len(t)), rate


def load_eeg_sample() -> Tuple[np.ndarray, int]:
    csv_path = PROJECT_ROOT / "modules" / "eeg" / "dataset" / "data.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path, nrows=1)
            label_cols = [c for c in df.columns if c.lower() in ("label", "y", "class")]
            feat_cols = [c for c in df.columns if c not in label_cols]
            if feat_cols:
                return df[feat_cols].iloc[0].values.astype(float), 178
        except Exception:
            pass
    return np.random.randn(178) * 0.3, 178


def generate_spo2_series(n: int = 30, base: float = 96.0) -> pd.DataFrame:
    vals = np.clip(base + np.cumsum(np.random.randn(n) * 0.3), 75, 100)
    return pd.DataFrame({"reading": range(n), "spo2": vals})


def generate_thermal_grid(size: int = 64, fever: bool = False) -> np.ndarray:
    base = np.random.uniform(36.0, 37.0, (size, size))
    if fever:
        cy, cx = size // 2, size // 2
        y, x = np.ogrid[:size, :size]
        hotspot = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (size / 6) ** 2))
        base += hotspot * np.random.uniform(1.5, 3.0)
    return base
