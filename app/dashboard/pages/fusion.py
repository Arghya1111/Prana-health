"""Fusion AI page."""

from __future__ import annotations

import json

import streamlit as st

from app.dashboard.components import (
    confidence_gauge,
    kpi_row,
    module_confidence_bar,
    module_status_card,
    page_header,
    risk_gauge,
)
from app.dashboard.data import load_fusion_df, load_fusion_report_json
from app.dashboard.pages._helpers import get_patient_row, modality_header
from app.dashboard.theme import severity_badge


def render() -> None:
    page_header("Fusion AI Engine", "Multi-modality clinical decision support", live=True)
    fusion = load_fusion_df(st.session_state.get("refresh_secs", 5))

    if fusion.empty:
        st.warning("No fusion reports yet. Run `python nats_pipeline.py` to generate fused assessments.")
        return

    patient = modality_header("FUSION", "Fusion")
    row = get_patient_row(patient) if patient else fusion.iloc[0].to_dict()
    if hasattr(row, "to_dict"):
        row = row.to_dict()

    risk = float(row.get("risk_score", 0))
    sev = row.get("overall_severity", "NORMAL")
    conf = float(row.get("overall_confidence", 0))
    action = row.get("action", "")

    kpi_row([
        (f"{risk:.1f}", "Risk Score", "0–100"),
        (sev, "Overall Severity", ""),
        (f"{conf:.1f}%", "AI Confidence", ""),
        (str(int(row.get("module_count", 7))), "Modules", "active"),
    ])

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(risk_gauge(risk), use_container_width=True, key="fusion_risk")
    with c2:
        confidence_gauge(conf, "Fused AI Confidence")
        st.markdown(severity_badge(sev), unsafe_allow_html=True)
        st.info(action or "No recommendation available.")

    st.markdown("### Module Predictions")
    modules = [
        ("ECG", "ecg_status"), ("EEG", "eeg_status"), ("PPG", "ppg_status"),
        ("SpO2", "spo2_status"), ("Thermal", "thermal_status"),
        ("Optical", "optical_status"), ("X-Ray", "xray_status"),
    ]
    cols = st.columns(4)
    mod_data = {}
    for i, (name, col) in enumerate(modules):
        status = row.get(col, "—") or "—"
        mod_conf = conf
        mod_sev = "HIGH" if status in ("ANOMALOUS", "ABNORMAL", "LOW", "PNEUMONIA", "FEVER", "CRITICAL") else (
            "MODERATE" if status in ("ATTENTION", "MODERATE") else "NORMAL"
        )
        mod_data[name] = {"confidence": mod_conf, "status": status}
        cols[i % 4].markdown(
            module_status_card(name, status, mod_conf, mod_sev),
            unsafe_allow_html=True,
        )

    st.plotly_chart(module_confidence_bar(mod_data), use_container_width=True, key="fusion_mod_conf")

    report_id = row.get("id")
    if report_id:
        report_data = load_fusion_report_json(int(report_id))
        if report_data:
            with st.expander("Full Fusion JSON"):
                st.json(report_data)
