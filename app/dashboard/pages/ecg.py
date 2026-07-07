"""ECG monitoring page."""

from __future__ import annotations

import streamlit as st

from app.dashboard.components import confidence_gauge, ecg_chart, hr_trend, page_header
from app.dashboard.data import generate_ecg_waveform, load_triage_df
from app.dashboard.pages._helpers import get_patient_row, modality_header
from app.dashboard.theme import severity_badge


def render() -> None:
    page_header("ECG Intelligence", "Electrocardiogram analysis and HRV monitoring", live=True)
    patient = modality_header("ECG", "ECG")
    row = get_patient_row(patient) if patient else None

    hr = float(row["mean_hr"]) if row is not None and "mean_hr" in row else 72.0
    status = row.get("ecg_status", "NORMAL") if row is not None else "NORMAL"
    conf = float(row.get("ecg_confidence", row.get("overall_confidence", 92))) if row is not None else 92.0
    sev = "HIGH" if status == "ANOMALOUS" else "NORMAL"

    c1, c2, c3 = st.columns([2, 1, 1])
    signal, rate = generate_ecg_waveform(hr=hr)
    with c1:
        st.plotly_chart(
            ecg_chart(signal, rate, f"ECG — {patient or 'Demo'} ({status})"),
            use_container_width=True, key="ecg_wave",
        )
    with c2:
        confidence_gauge(conf, "ECG Confidence")
        st.markdown(severity_badge(sev), unsafe_allow_html=True)
    with c3:
        if row is not None:
            st.metric("Heart Rate", f"{hr:.0f} BPM")
            for label, col in [("SDNN", "sdnn"), ("RMSSD", "rmssd"), ("pNN50", "pnn50")]:
                if col in row:
                    st.metric(label, f"{row[col]:.1f}")

    triage = load_triage_df(st.session_state.get("refresh_secs", 5))
    if not triage.empty:
        st.plotly_chart(hr_trend(triage), use_container_width=True, key="ecg_hr_trend")
