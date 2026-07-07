"""Clinical reports page."""

from __future__ import annotations

import streamlit as st

from app.dashboard.components import download_buttons, fusion_report_text, page_header
from app.dashboard.data import load_fusion_df, load_fusion_report_json
from app.dashboard.pages._helpers import patient_selector, get_patient_list


def render() -> None:
    page_header("Clinical Reports", "Generate and download patient fusion reports")
    fusion = load_fusion_df(st.session_state.get("refresh_secs", 5))

    if fusion.empty:
        st.info("No reports available. Run the pipeline to generate clinical reports.")
        return

    patients = get_patient_list()
    patient = patient_selector(patients, key="reports_patient")
    if not patient:
        return

    rows = fusion[fusion["patient_id"] == patient]
    if rows.empty:
        st.warning(f"No reports for {patient}")
        return

    report_idx = st.selectbox(
        "Select Report",
        range(len(rows)),
        format_func=lambda i: str(rows.iloc[i]["timestamp"]),
        key="report_idx",
    )
    row = rows.iloc[report_idx].to_dict()
    report_data = load_fusion_report_json(int(row["id"])) if row.get("id") else None
    text = fusion_report_text(row, report_data)

    st.markdown("### Report Preview")
    st.text_area("Clinical Report", text, height=400, key="report_preview")

    pdf_text = ""
    if report_data:
        pdf_text = report_data.get("report", {}).get("pdf_ready_text", text)
    else:
        pdf_text = text

    download_buttons(
        rows,
        report_json=report_data or row,
        pdf_text=pdf_text,
        prefix=f"PRANA_{patient}",
    )

    st.markdown("### All Reports")
    show = fusion[["id", "patient_id", "timestamp", "risk_score", "overall_severity", "action"]].head(30)
    st.dataframe(show, use_container_width=True, hide_index=True)
