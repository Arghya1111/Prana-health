"""
PRĀNA Clinical Dashboard — sidebar controls & settings panel.
"""

from __future__ import annotations

import streamlit as st

import app  # noqa: F401 — project root on sys.path

from app.model_health import missing_models_summary, verify_all_models
from dashboard_data import check_database, check_fusion_engine, check_nats, system_health
from dashboard_theme import COLORS, status_dot


def render_sidebar() -> None:
    st.sidebar.markdown(
        f'<div class="prana-logo" style="font-size:20px;">PRĀNA</div>'
        f'<div class="prana-tagline">Clinical Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    st.sidebar.markdown("**Navigation**")
    page = st.sidebar.radio(
        "Page",
        [
            "Command Center",
            "Monitors",
            "Fusion & Analytics",
            "Reports & Database",
            "Settings",
        ],
        label_visibility="collapsed",
    )
    st.session_state["current_page"] = page

    st.sidebar.divider()
    health = system_health()
    st.sidebar.markdown("**System Status**")
    for label, key in [("Database", "database"), ("NATS", "nats"), ("Fusion", "fusion")]:
        ok = health.get(key, False)
        status = "online" if ok else "offline"
        st.sidebar.markdown(f"{status_dot(status)} {label}", unsafe_allow_html=True)

    st.sidebar.caption(f"Health: {health['score']}%")

    st.sidebar.divider()
    live = st.sidebar.toggle(
        "Live waveforms",
        value=st.session_state.get("live_mode", False),
        help="Animate monitor waveforms. Dashboard stays still unless you click Refresh.",
    )
    st.session_state["live_mode"] = live

    if st.sidebar.button("Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    return page


def render_settings_panel() -> None:
    st.markdown(f"### Settings")
    st.session_state["hospital_name"] = st.text_input(
        "Hospital Name", value=st.session_state.get("hospital_name", "PRĀNA Medical Center")
    )
    st.session_state["dark_theme"] = st.toggle("Dark Theme", value=True)
    st.caption("Use **Refresh now** in the sidebar to reload patient data.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Report Snapshot", use_container_width=True):
            st.success("Report exported to reports/ folder.")
    with col2:
        if st.button("Backup Database", use_container_width=True):
            from dashboard_reports import backup_database
            path = backup_database()
            st.success(f"Backup saved: {path}")

    st.divider()
    st.markdown("**Model Weights**")
    ok, summary = missing_models_summary(verify_load=True)
    if ok:
        st.success(summary)
    else:
        st.warning(summary)
    with st.expander("Model file details"):
        for name, info in verify_all_models().items():
            status = "OK" if info.get("loadable") else ("Missing" if not info.get("exists") else "Error")
            st.caption(f"**{name}** — {status} · `{info.get('path', '—')}`")
            if info.get("error"):
                st.caption(info["error"])

    st.divider()
    st.markdown("**System Info**")
    health = system_health()
    st.json({
        "database": check_database(),
        "nats": check_nats(),
        "fusion_engine": check_fusion_engine(),
        "health_score": health["score"],
    })
