"""EEG monitoring page."""

from __future__ import annotations

import streamlit as st

from app.dashboard.components import confidence_gauge, eeg_multi_band, page_header
from app.dashboard.data import load_eeg_sample
from app.dashboard.pages._helpers import get_patient_row, modality_header
from app.dashboard.theme import severity_badge


def render() -> None:
    page_header("EEG Intelligence", "Electroencephalogram seizure and abnormality detection", live=True)
    patient = modality_header("EEG", "EEG")
    row = get_patient_row(patient) if patient else None

    status = row.get("eeg_status", "NORMAL") if row is not None else "NORMAL"
    conf = float(row.get("overall_confidence", 94)) if row is not None else 94.0
    sev = "HIGH" if status == "ABNORMAL" else "NORMAL"

    signal, rate = load_eeg_sample()
    c1, c2 = st.columns([3, 1])
    with c1:
        st.plotly_chart(
            eeg_multi_band(signal, rate, f"EEG — {patient or 'Demo'}"),
            use_container_width=True, key="eeg_wave",
        )
    with c2:
        confidence_gauge(conf, "EEG Confidence")
        st.markdown(severity_badge(sev), unsafe_allow_html=True)
        st.metric("Status", status)
        if status == "ABNORMAL":
            st.warning("Neurology consultation advised.")
