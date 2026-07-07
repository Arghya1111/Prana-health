"""SpO2 monitoring page."""

from __future__ import annotations

import streamlit as st

from app.dashboard.components import confidence_gauge, page_header, spo2_gauge, timeline_chart
from app.dashboard.data import generate_spo2_series
from app.dashboard.pages._helpers import get_patient_row, modality_header
from app.dashboard.theme import severity_badge


def render() -> None:
    page_header("SpO2 Intelligence", "Oxygen saturation and hypoxia monitoring", live=True)
    patient = modality_header("SpO2", "SpO2")
    row = get_patient_row(patient) if patient else None

    status = row.get("spo2_status", "NORMAL") if row is not None else "NORMAL"
    conf = float(row.get("overall_confidence", 97)) if row is not None else 97.0
    spo2_val = 98.0
    if status == "LOW" or status == "HIGH":
        spo2_val = 87.0
    elif status == "CRITICAL":
        spo2_val = 82.0
    elif status == "MODERATE":
        spo2_val = 92.0

    sev_map = {"NORMAL": "NORMAL", "MODERATE": "MODERATE", "LOW": "HIGH", "HIGH": "HIGH", "CRITICAL": "CRITICAL"}
    sev = sev_map.get(status, "NORMAL")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.plotly_chart(spo2_gauge(spo2_val), use_container_width=True, key="spo2_gauge")
    with c2:
        confidence_gauge(conf, "SpO2 Confidence")
        st.markdown(severity_badge(sev), unsafe_allow_html=True)
    with c3:
        st.metric("Status", status)
        st.metric("Respiratory Risk", sev)
        if spo2_val < 90:
            st.error("Supplemental oxygen recommended.")

    import pandas as pd
    trend = generate_spo2_series(40, base=spo2_val)
    trend["timestamp"] = pd.date_range(end=pd.Timestamp.now(), periods=len(trend), freq="min")
    st.plotly_chart(
        timeline_chart(trend, "spo2", "SpO2 Trend", "#00C896"),
        use_container_width=True, key="spo2_trend",
    )
