"""Settings page."""

from __future__ import annotations

import streamlit as st

from app.dashboard.components import page_header
from app.dashboard.data import HAS_PSUTIL, system_info
from app.paths import DB_PATH, MODEL_PATH


def render() -> None:
    page_header("Settings", "System configuration and health")
    info = system_info()

    st.markdown("### Display Settings")
    st.session_state["refresh_secs"] = st.slider(
        "Live refresh interval (seconds)",
        min_value=3, max_value=60, value=st.session_state.get("refresh_secs", 5),
    )
    st.session_state["api_base"] = st.text_input(
        "FastAPI base URL", value=st.session_state.get("api_base", "http://localhost:8000"),
    )
    st.session_state["live_mode"] = st.toggle("Live updates", value=st.session_state.get("live_mode", True))

    st.markdown("### System Status")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Database", "Connected" if info["db_exists"] else "Missing")
        st.metric("ECG Model", "Loaded" if info["model_exists"] else "Missing")
        st.metric("NATS", "Online" if info["nats_live"] else "Offline")
        st.metric("DB Size", f"{info['db_size_mb']} MB")
    with c2:
        if HAS_PSUTIL:
            st.metric("CPU", f"{info.get('cpu', 0):.0f}%")
            st.metric("RAM", f"{info.get('ram', 0):.0f}%")
            st.metric("Disk", f"{info.get('disk', 0):.0f}%")
        st.code(f"DB: {DB_PATH}\nModel: {MODEL_PATH}", language=None)

    st.markdown("### Pipeline")
    st.markdown("""
    ```
    Devices -> NATS -> AI Modules -> Fusion Engine -> SQLite -> FastAPI -> Dashboard
    ```

    **Start services:**
    1. `nats-server`
    2. `python nats_pipeline.py`
    3. `uvicorn app.api:app --reload`
    4. `streamlit run dashboard.py`
    """)

    if st.button("Clear cached data", type="secondary"):
        st.cache_data.clear()
        st.success("Cache cleared.")
        st.rerun()
