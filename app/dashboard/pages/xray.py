"""X-Ray intelligence page."""

from __future__ import annotations

import streamlit as st

from app.dashboard.components import confidence_gauge, page_header, xray_findings_bar
from app.dashboard.pages._helpers import get_patient_row, modality_header
from app.dashboard.theme import severity_badge


def render() -> None:
    page_header("X-Ray Intelligence", "Chest radiograph pathology detection", live=True)
    patient = modality_header("XRAY", "X-Ray")
    row = get_patient_row(patient) if patient else None

    status = row.get("xray_status", "NORMAL") if row is not None else "NORMAL"
    conf = float(row.get("overall_confidence", 91)) if row is not None else 91.0
    sev = {"NORMAL": "NORMAL", "ATTENTION": "MODERATE", "CRITICAL": "CRITICAL"}.get(status, "HIGH")

    flagged = {}
    if row is not None and "xray_flagged" in row and row["xray_flagged"] not in (None, "None", ""):
        try:
            for part in str(row["xray_flagged"]).split(","):
                if "(" in part:
                    name, val = part.strip().rsplit("(", 1)
                    flagged[name.strip()] = float(val.rstrip(")"))
        except Exception:
            pass
    if not flagged and status not in ("NORMAL",):
        flagged = {"Pneumonia": 0.72, "Lung Opacity": 0.58}

    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(xray_findings_bar(flagged), use_container_width=True, key="xray_bar")
        st.markdown(
            '<div class="med-card" style="text-align:center;padding:40px;">'
            '<div style="font-size:48px;">🫁</div>'
            '<div style="color:#64748B;">Chest X-Ray Preview</div>'
            f'<div style="font-weight:700;margin-top:8px;">{status}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        confidence_gauge(conf, "X-Ray Confidence")
        st.markdown(severity_badge(sev), unsafe_allow_html=True)
        if flagged:
            st.markdown("**Flagged Conditions**")
            for k, v in sorted(flagged.items(), key=lambda x: -x[1]):
                st.progress(min(1.0, v), text=f"{k}: {v:.0%}")
