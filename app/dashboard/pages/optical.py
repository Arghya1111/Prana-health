"""Optical diagnostics page."""

from __future__ import annotations

import streamlit as st

from app.dashboard.components import confidence_gauge, module_status_card, page_header
from app.dashboard.pages._helpers import get_patient_row, modality_header
from app.dashboard.theme import severity_badge


def render() -> None:
    page_header("Optical Diagnostics", "Ophthalmic and surface imaging analysis", live=True)
    patient = modality_header("OPTICAL", "Optical")
    row = get_patient_row(patient) if patient else None

    disease = row.get("optical_status", "NORMAL") if row is not None else "NORMAL"
    conf = float(row.get("overall_confidence", 95)) if row is not None else 95.0
    sev = "NORMAL" if disease == "NORMAL" else "MODERATE"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="med-card" style="text-align:center;padding:50px 20px;">'
            '<div style="font-size:56px;">👁</div>'
            '<div style="font-weight:700;font-size:16px;margin-top:12px;">Retinal Scan</div>'
            f'<div style="color:#64748B;margin-top:6px;">{disease}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        confidence_gauge(conf, "Optical Confidence")
    with c3:
        st.markdown(severity_badge(sev), unsafe_allow_html=True)
        st.markdown(module_status_card("OPTICAL", disease, conf, sev), unsafe_allow_html=True)
        if disease != "NORMAL":
            st.info("Ophthalmology referral recommended.")
