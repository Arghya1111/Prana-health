"""
PRĀNA Clinical Dashboard — utility helpers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

import app  # noqa: F401 — project root on sys.path


def init_session_state() -> None:
    defaults = {
        "refresh_secs": 30,
        "live_mode": False,
        "hospital_name": "PRĀNA Medical Center",
        "selected_patient": None,
        "selected_report_id": None,
        "dark_theme": True,
        "search_query": "",
        "severity_filter": "All",
        "sort_by": "priority",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def format_timestamp(ts) -> str:
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return "—"
    try:
        return pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


def format_time_now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def format_date_now() -> str:
    return datetime.now().strftime("%A, %B %d, %Y")


def filter_queue(
    df: pd.DataFrame,
    search: str = "",
    severity: str = "All",
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if severity != "All" and "overall_severity" in out.columns:
        out = out[out["overall_severity"].str.upper() == severity.upper()]
    if search.strip():
        q = search.strip().upper()
        mask = out["patient_id"].astype(str).str.upper().str.contains(q, na=False)
        out = out[mask]
    return out


def sort_queue(df: pd.DataFrame, sort_by: str = "priority") -> pd.DataFrame:
    if df.empty:
        return df
    if sort_by == "priority" and "priority" in df.columns:
        return df.sort_values(["priority", "arrival_time"])
    if sort_by == "arrival" and "arrival_time" in df.columns:
        return df.sort_values("arrival_time", ascending=False)
    if sort_by == "risk" and "fusion_score" in df.columns:
        return df.sort_values("fusion_score", ascending=False)
    return df


def derive_fusion_module_scores(report: Dict[str, Any]) -> Dict[str, float]:
    """Extract per-module scores from fusion report JSON for radar chart."""
    default = {"ECG": 75, "EEG": 80, "PPG": 70, "SpO2": 85, "Thermal": 72, "Optical": 78}
    if not report:
        return default
    modules = report.get("report", {}).get("modules", report.get("modules", {}))
    scores: Dict[str, float] = {}
    for key in ["ECG", "EEG", "PPG", "SpO2", "THERMAL", "OPTICAL", "XRAY"]:
        mod = modules.get(key, modules.get(key.title(), {}))
        if isinstance(mod, dict):
            conf = mod.get("confidence", mod.get("overall_confidence", 0))
            scores[key.title() if key != "SpO2" else "SpO2"] = float(conf) if conf else default.get(key.title(), 70)
        else:
            scores[key.title() if key != "SpO2" else "SpO2"] = default.get(key.title(), 70)
    if "Thermal" not in scores and "THERMAL" in scores:
        scores["Thermal"] = scores.pop("THERMAL", 72)
    if "Optical" not in scores and "OPTICAL" in scores:
        scores["Optical"] = scores.pop("OPTICAL", 78)
    return scores or default


def compute_eeg_metrics(signal) -> Dict[str, float]:
    import numpy as np
    sig = np.asarray(signal)
    fft = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), d=1 / 174)
    dominant = float(freqs[np.argmax(fft[1:]) + 1]) if len(fft) > 1 else 10.0
    power = float(np.std(sig))
    return {
        "dominant_freq": round(dominant, 2),
        "brain_activity": round(power * 100, 1),
        "seizure_risk": "LOW" if power < 0.5 else "MODERATE",
        "stress_index": round(min(100, power * 120), 1),
    }


def build_analytics_daily(fusion: pd.DataFrame) -> pd.DataFrame:
    if fusion.empty or "timestamp" not in fusion.columns:
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        import numpy as np
        return pd.DataFrame({"day": days, "patients": np.random.randint(5, 25, 7)})
    d = fusion.copy()
    d["day"] = d["timestamp"].dt.strftime("%a")
    return d.groupby("day").size().reset_index(name="patients")


def severity_to_priority(sev: str) -> int:
    return {"CRITICAL": 1, "HIGH": 2, "MODERATE": 3, "LOW": 4, "NORMAL": 5}.get(str(sev).upper(), 5)
