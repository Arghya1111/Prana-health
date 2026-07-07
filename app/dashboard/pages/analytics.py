"""Analytics page."""

from __future__ import annotations

import streamlit as st

from app.dashboard.components import (
    cases_per_hour,
    confidence_trend,
    hr_trend,
    page_header,
    severity_pie,
    severity_timeline,
)
from app.dashboard.data import compute_kpis, load_fusion_df, load_triage_df


def render() -> None:
    page_header("Analytics", "Clinic-wide trends and AI performance metrics")
    fusion = load_fusion_df(st.session_state.get("refresh_secs", 5))
    triage = load_triage_df(st.session_state.get("refresh_secs", 5))
    use_fusion = not fusion.empty
    df = fusion if use_fusion else triage
    sev_col = "overall_severity" if use_fusion else "overall"
    kpis = compute_kpis(triage, fusion)

    kpi_row_data = [
        (str(kpis["total"]), "Total Cases", ""),
        (str(kpis["critical"] + kpis["high"]), "Urgent Cases", ""),
        (f"{kpis['avg_conf']:.0f}%", "Avg Confidence", ""),
        (f"{kpis['avg_risk']:.0f}" if kpis["avg_risk"] else "—", "Avg Risk", ""),
    ]
    from app.dashboard.components import kpi_row
    kpi_row(kpi_row_data)

    if df.empty:
        st.info("No analytics data. Populate the database first.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(severity_pie(df, sev_col), use_container_width=True, key="ana_pie")
        st.plotly_chart(cases_per_hour(df), use_container_width=True, key="ana_cases")
    with c2:
        st.plotly_chart(severity_timeline(df, sev_col), use_container_width=True, key="ana_timeline")
        conf_col = "overall_confidence" if use_fusion else "ecg_confidence"
        st.plotly_chart(confidence_trend(df, conf_col), use_container_width=True, key="ana_conf")

    if not triage.empty and "mean_hr" in triage.columns:
        st.plotly_chart(hr_trend(triage), use_container_width=True, key="ana_hr")

    if use_fusion and "risk_score" in fusion.columns:
        st.markdown("### Risk Score Distribution")
        st.bar_chart(fusion["risk_score"].value_counts().sort_index())
