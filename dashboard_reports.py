"""
PRĀNA Clinical Dashboard — patient reports, export, and database actions.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

import app  # noqa: F401 — project root on sys.path

from app.paths import DB_PATH, REPORTS_DIR
from dashboard_data import load_fusion_report_json
from dashboard_theme import COLORS, severity_badge
from dashboard_utils import format_timestamp


def render_patient_report_dialog(patient_id: str, report_id: Optional[int] = None) -> None:
    """Display complete patient fusion report."""
    st.markdown(f"### Patient Report — {patient_id}")

    report: Dict[str, Any] = {}
    if report_id:
        report = load_fusion_report_json(report_id) or {}

    if report:
        st.markdown(
            f"**Risk Score:** {report.get('risk_score', '—')} &nbsp;|&nbsp; "
            f"**Severity:** {severity_badge(report.get('overall_severity', 'NORMAL'))} &nbsp;|&nbsp; "
            f"**Confidence:** {report.get('overall_confidence', '—')}%",
            unsafe_allow_html=True,
        )
        rec = report.get("recommendation", {})
        if isinstance(rec, dict):
            st.info(rec.get("combined_recommendation", rec.get("action", "No recommendation")))
        with st.expander("Full JSON Report"):
            st.json(report)
    else:
        st.warning("No detailed fusion report found for this patient.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Download CSV", key=f"csv_{patient_id}"):
            st.session_state["download_csv"] = patient_id
    with col2:
        if st.button("Download PDF", key=f"pdf_{patient_id}"):
            st.info("PDF export uses fusion report_generator — configure in production.")


def export_reports_csv(fusion_df: pd.DataFrame) -> bytes:
    return fusion_df.to_csv(index=False).encode("utf-8")


def backup_database() -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = REPORTS_DIR / f"prana_records_backup_{ts}.db"
    if Path(DB_PATH).exists():
        shutil.copy2(DB_PATH, dest)
    return str(dest)


def render_database_panel(fusion_df: pd.DataFrame, triage_df: pd.DataFrame) -> None:
    st.markdown("### Database — Latest Reports")

    search = st.text_input("Search Patient", placeholder="PAT001...")
    src = fusion_df if not fusion_df.empty else triage_df

    if src.empty:
        st.info("No reports in database. Run `python nats_pipeline.py` to populate.")
        return

    if search.strip():
        src = src[src["patient_id"].astype(str).str.contains(search.strip(), case=False, na=False)]

    display_cols = [c for c in [
        "id", "patient_id", "timestamp", "overall_severity", "risk_score",
        "overall_confidence", "recommendation", "action",
    ] if c in src.columns]
    st.dataframe(src[display_cols].head(25), use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Download CSV",
            data=export_reports_csv(src),
            file_name=f"prana_reports_{datetime.now():%Y%m%d}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        if st.button("Backup Database", use_container_width=True):
            path = backup_database()
            st.success(f"Backup: {path}")
    with col3:
        delete_id = st.number_input("Delete Report ID", min_value=0, step=1, value=0)
        if st.button("Delete Record", use_container_width=True) and delete_id > 0:
            st.warning("Delete disabled in demo — enable with admin auth in production.")

    st.caption(f"Database: `{DB_PATH}`")
