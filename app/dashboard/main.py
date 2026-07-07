"""
PRĀNA Hospital Dashboard — main entry point.

Run:  streamlit run dashboard.py
"""

from __future__ import annotations

from datetime import timedelta

import streamlit as st

from app.dashboard.components import sidebar_brand, sidebar_status
from app.dashboard.data import check_nats, load_triage_df
from app.dashboard.theme import inject_css
from app.dashboard.pages import (
    analytics,
    ecg,
    eeg,
    fusion,
    optical,
    overview,
    patients,
    ppg,
    reports_page,
    settings,
    spo2,
    thermal,
    xray,
)

PAGES = {
    "Dashboard": overview,
    "Patient Management": patients,
    "ECG": ecg,
    "EEG": eeg,
    "PPG": ppg,
    "SpO2": spo2,
    "X-Ray": xray,
    "Thermal": thermal,
    "Optical": optical,
    "Fusion AI": fusion,
    "Reports": reports_page,
    "Analytics": analytics,
    "Settings": settings,
}


def _init_session() -> None:
    defaults = dict(refresh_secs=5, api_base="http://localhost:8000", live_mode=True)
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def main() -> None:
    st.set_page_config(
        page_title="PRANA Clinical Intelligence",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_session()
    st.markdown(inject_css(), unsafe_allow_html=True)

    sidebar_brand()
    nats = check_nats()
    db_ok = __import__("os").path.exists(__import__("app.paths", fromlist=["DB_PATH"]).DB_PATH)
    sidebar_status(nats, db_ok)

    page = st.sidebar.radio(
        "Navigation",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )

    refresh = st.session_state.get("refresh_secs", 5)
    live = st.session_state.get("live_mode", True)
    st.sidebar.caption(f"Auto-refresh: {refresh}s" if live else "Manual refresh")

    if live:
        st.sidebar.markdown(
            f'<span class="live-dot"></span> Live mode',
            unsafe_allow_html=True,
        )

    st.sidebar.divider()
    if st.sidebar.button("Refresh now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Render selected page
    PAGES[page].render()

    # Auto-refresh via fragment (Streamlit 1.33+) or sleep fallback
    if live:
        try:
            from app.dashboard.data import load_fusion_df

            @st.fragment(run_every=timedelta(seconds=refresh))
            def _auto_refresh():
                load_triage_df(refresh)
                load_fusion_df(refresh)

            _auto_refresh()
        except (TypeError, AttributeError):
            pass


if __name__ == "__main__":
    main()
