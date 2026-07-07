"""Thermal diagnostics page."""

from __future__ import annotations

import streamlit as st

from app.dashboard.components import confidence_gauge, page_header, thermal_heatmap
from app.dashboard.data import generate_thermal_grid
from app.dashboard.pages._helpers import get_patient_row, modality_header
from app.dashboard.theme import severity_badge


def render() -> None:
    page_header("Thermal Diagnostics", "Infrared thermography fever and inflammation detection", live=True)
    patient = modality_header("THERMAL", "Thermal")
    row = get_patient_row(patient) if patient else None

    status = row.get("thermal_status", "NORMAL") if row is not None else "NORMAL"
    conf = float(row.get("overall_confidence", 90)) if row is not None else 90.0
    fever = status in ("FEVER", "HIGH", "MODERATE", "CRITICAL")
    sev = {"NORMAL": "NORMAL", "MODERATE": "MODERATE", "FEVER": "MODERATE", "HIGH": "HIGH"}.get(status, "NORMAL")

    grid = generate_thermal_grid(fever=fever)
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(thermal_heatmap(grid, f"Thermal — {patient or 'Demo'}"), use_container_width=True, key="thermal_map")
    with c2:
        confidence_gauge(conf, "Thermal Confidence")
        st.markdown(severity_badge(sev), unsafe_allow_html=True)
        st.metric("Status", status)
        st.metric("Fever Detected", "YES" if fever else "NO")
        st.metric("Mean Temp", f"{grid.mean():.1f} C")
