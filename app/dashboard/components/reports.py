"""Downloadable report components."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st


def download_buttons(
    df: pd.DataFrame,
    report_json: Optional[Dict[str, Any]] = None,
    pdf_text: str = "",
    prefix: str = "PRANA",
) -> None:
    """Render CSV, JSON, and text report download buttons."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    c1, c2, c3 = st.columns(3)

    with c1:
        if not df.empty:
            st.download_button(
                "Download CSV",
                df.to_csv(index=False).encode(),
                file_name=f"{prefix}_export_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with c2:
        if report_json:
            st.download_button(
                "Download JSON",
                json.dumps(report_json, indent=2, default=str).encode(),
                file_name=f"{prefix}_report_{ts}.json",
                mime="application/json",
                use_container_width=True,
            )
    with c3:
        if pdf_text:
            st.download_button(
                "Download Report (TXT)",
                pdf_text.encode(),
                file_name=f"{prefix}_clinical_report_{ts}.txt",
                mime="text/plain",
                use_container_width=True,
            )


def fusion_report_text(row: dict, report_data: Optional[dict] = None) -> str:
    """Build plain-text clinical report from DB row."""
    lines = [
        "=" * 60,
        "PRANA CLINICAL FUSION REPORT",
        "=" * 60,
        f"Patient ID  : {row.get('patient_id', 'N/A')}",
        f"Timestamp   : {row.get('timestamp', 'N/A')}",
        f"Risk Score  : {row.get('risk_score', 0)} / 100",
        f"Severity    : {row.get('overall_severity', 'N/A')}",
        f"Confidence  : {row.get('overall_confidence', 0)}%",
        f"Action      : {row.get('action', 'N/A')}",
        "",
        "Module Status:",
    ]
    for mod, col in [
        ("ECG", "ecg_status"), ("EEG", "eeg_status"), ("PPG", "ppg_status"),
        ("SpO2", "spo2_status"), ("Thermal", "thermal_status"),
        ("Optical", "optical_status"), ("X-Ray", "xray_status"),
    ]:
        val = row.get(col)
        if val:
            lines.append(f"  {mod}: {val}")
    if row.get("recommendation"):
        lines.extend(["", "Recommendation:", row["recommendation"]])
    if report_data and report_data.get("report", {}).get("summary"):
        lines.extend(["", "Summary:", report_data["report"]["summary"]])
    lines.append("=" * 60)
    return "\n".join(lines)
