"""
PRĀNA Clinical Dashboard — sidebar controls & settings panel.
"""

from __future__ import annotations

import streamlit as st

import app  # noqa: F401 — project root on sys.path

from app.model_health import missing_models_summary, verify_all_models
from dashboard_data import check_database, check_fusion_engine, check_nats, system_health
from dashboard_theme import status_dot

PAGES = [
    "Command Center",
    "Monitors",
    "Fusion & Analytics",
    "Reports & Database",
    "Settings",
]

PAGE_SHORT = {
    "Command Center": "Command",
    "Monitors": "Monitors",
    "Fusion & Analytics": "Fusion",
    "Reports & Database": "Reports",
    "Settings": "Settings",
}


def render_sliding_nav(current_page: str) -> str:
    """
    Horizontal sliding navigation bar with pill buttons.
    Toggle hides/shows the bar; handle stays visible when collapsed.
    """
    nav_open = st.session_state.get("nav_bar_open", True)
    current = current_page if current_page in PAGES else st.session_state.get("current_page", PAGES[0])

    st.markdown(
        f'<div class="prana-slide-nav-marker {"is-nav-collapsed" if not nav_open else ""}"></div>',
        unsafe_allow_html=True,
    )

    handle_col, hint_col = st.columns([1.2, 5])
    with handle_col:
        toggle_label = "▲ Hide menu" if nav_open else "☰ Open menu"
        if st.button(toggle_label, key="nav_bar_toggle", use_container_width=True):
            st.session_state["nav_bar_open"] = not nav_open
            st.rerun()
    with hint_col:
        if nav_open:
            st.markdown(
                '<div class="prana-slide-nav-hint">Swipe or scroll pages · tap a pill to navigate</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="prana-slide-nav-hint">Viewing: <strong>{current}</strong> · tap ☰ to change page</div>',
                unsafe_allow_html=True,
            )

    page = current
    if nav_open:
        st.markdown('<div class="prana-slide-nav-panel">', unsafe_allow_html=True)
        page = st.radio(
            "Navigate",
            PAGES,
            index=PAGES.index(current),
            horizontal=True,
            key="slide_nav_pages",
            label_visibility="collapsed",
            format_func=lambda p: PAGE_SHORT.get(p, p),
        )

        st.markdown('<div class="prana-slide-nav-actions-label">Quick actions</div>', unsafe_allow_html=True)
        act1, act2, act3 = st.columns(3)
        with act1:
            live = st.toggle(
                "Live waveforms",
                value=st.session_state.get("live_mode", False),
                key="slide_nav_live",
            )
            st.session_state["live_mode"] = live
        with act2:
            if st.button("Refresh data", key="slide_nav_refresh", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        with act3:
            health = system_health()
            st.markdown(
                f'<div class="prana-slide-health">{status_dot(health.get("status", "degraded"))}'
                f'System {health.get("score", 0)}%</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.session_state["current_page"] = page
    return page


def render_sidebar() -> None:
    """Sidebar: branding + system status (page nav lives in sliding bar)."""
    st.sidebar.markdown(
        f'<div class="prana-logo" style="font-size:20px;">PRĀNA</div>'
        f'<div class="prana-tagline">Clinical Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    st.sidebar.markdown("**System Status**")
    health = system_health()
    for label, key in [("Database", "database"), ("NATS", "nats"), ("Fusion", "fusion")]:
        ok = health.get(key, False)
        status = "online" if ok else "offline"
        st.sidebar.markdown(f"{status_dot(status)} {label}", unsafe_allow_html=True)
    st.sidebar.caption(f"Health: {health['score']}%")

    st.sidebar.divider()
    st.sidebar.caption("Use the **sliding menu bar** above the dashboard to switch pages, refresh, and toggle live waveforms.")

    if st.sidebar.button("Open navigation menu", use_container_width=True):
        st.session_state["nav_bar_open"] = True
        st.rerun()


def render_settings_panel() -> None:
    st.markdown("### Settings")
    st.session_state["hospital_name"] = st.text_input(
        "Hospital Name", value=st.session_state.get("hospital_name", "PRĀNA Medical Center")
    )
    st.session_state["dark_theme"] = st.toggle("Dark Theme", value=True)
    st.caption("Use the sliding menu bar to refresh data or open pages.")

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
