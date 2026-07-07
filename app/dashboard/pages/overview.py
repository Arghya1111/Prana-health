"""Main dashboard overview page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.dashboard.components import alert_banner, kpi_row, page_header, severity_pie, severity_timeline
from app.dashboard.data import compute_kpis, load_fusion_df, load_triage_df
from app.dashboard.theme import severity_badge


def render() -> None:
    page_header("Clinical Command Center", "Real-time multi-modality patient monitoring", live=True)
    fusion = load_fusion_df(st.session_state.get("refresh_secs", 5))
    triage = load_triage_df(st.session_state.get("refresh_secs", 5))
    kpis = compute_kpis(triage, fusion)

    kpi_row([
        (str(kpis["total"]), "Total Reports", f"{kpis['today']} today"),
        (str(kpis["critical"]), "Critical", "Immediate action"),
        (str(kpis["high"]), "High Priority", "Specialist review"),
        (f"{kpis['avg_risk']:.0f}" if kpis["avg_risk"] else "—", "Avg Risk Score", "0–100 scale"),
        (f"{kpis['avg_conf']:.0f}%" if kpis["avg_conf"] else "—", "AI Confidence", f"{kpis['patients']} patients"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # Critical alerts
    src = fusion if not fusion.empty else triage
    if not src.empty:
        sev_col = "overall_severity" if "overall_severity" in src.columns else "overall"
        critical = src[src[sev_col].isin(["CRITICAL", "HIGH"])].head(5)
        if not critical.empty:
            st.markdown("### Active Alerts")
            for _, row in critical.iterrows():
                pid = row["patient_id"]
                sev = row[sev_col]
                action = row.get("action", row.get("recommendation", "Review required"))
                alert_banner(sev, f"{pid} — {sev}", str(action)[:120])

    c1, c2 = st.columns(2)
    with c1:
        sev_col = "overall_severity" if not fusion.empty else "overall"
        chart_df = fusion if not fusion.empty else triage
        st.plotly_chart(severity_pie(chart_df, sev_col), use_container_width=True, key="dash_pie")
    with c2:
        st.plotly_chart(severity_timeline(chart_df, sev_col), use_container_width=True, key="dash_timeline")

    st.markdown("### Recent Patients")
    if src.empty:
        st.info("No data yet. Run `python nats_pipeline.py` to populate the database.")
    else:
        show_cols = ["patient_id", "timestamp", sev_col, "action" if "action" in src.columns else "recommendation"]
        if "risk_score" in src.columns:
            show_cols.insert(3, "risk_score")
        if "overall_confidence" in src.columns:
            show_cols.append("overall_confidence")
        display = src[[c for c in show_cols if c in src.columns]].head(15)
        st.dataframe(display, use_container_width=True, hide_index=True)
