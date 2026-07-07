"""PPG monitoring page."""

from __future__ import annotations

import streamlit as st

from app.dashboard.components import confidence_gauge, page_header, ppg_chart
from app.dashboard.data import generate_ppg_waveform
from app.dashboard.pages._helpers import get_patient_row, modality_header
from app.dashboard.theme import severity_badge


def render() -> None:
    page_header("PPG Intelligence", "Photoplethysmography pulse waveform analysis", live=True)
    patient = modality_header("PPG", "PPG")
    row = get_patient_row(patient) if patient else None

    status = row.get("ppg_status", "NORMAL") if row is not None else "NORMAL"
    conf = float(row.get("overall_confidence", 93)) if row is not None else 93.0
    sev = "MODERATE" if status == "ABNORMAL" else "NORMAL"
    hr = float(row.get("mean_hr", 72)) if row is not None and "mean_hr" in row else 72.0

    signal, rate = generate_ppg_waveform(hr=hr)
    c1, c2 = st.columns([3, 1])
    with c1:
        st.plotly_chart(
            ppg_chart(signal, rate, f"PPG — {patient or 'Demo'}"),
            use_container_width=True, key="ppg_wave",
        )
    with c2:
        confidence_gauge(conf, "PPG Confidence")
        st.markdown(severity_badge(sev), unsafe_allow_html=True)
        st.metric("Est. Heart Rate", f"{hr:.0f} BPM")
        st.metric("Pulse Quality", "GOOD" if status == "NORMAL" else "FAIR")
