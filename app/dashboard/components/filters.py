"""Search and filter components."""

from __future__ import annotations

import streamlit as st


def patient_filters(key_prefix: str = "") -> tuple[str, str, str]:
    """Return (search, severity_filter, sort_order)."""
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        search = st.text_input("Search patients", placeholder="Patient ID, status...", key=f"{key_prefix}_search")
    with c2:
        severity = st.selectbox("Severity", ["All", "CRITICAL", "HIGH", "MODERATE", "LOW", "NORMAL"], key=f"{key_prefix}_sev")
    with c3:
        sort = st.selectbox("Sort", ["Newest", "Oldest", "Risk (High)", "Risk (Low)"], key=f"{key_prefix}_sort")
    return search, severity, sort


def patient_selector(patients: list[str], key: str = "patient_sel") -> str:
    if not patients:
        st.info("No patients in database.")
        return ""
    return st.selectbox("Select Patient", patients, key=key)


def date_range_filter(key: str = "date_range"):
    return st.date_input("Date range", value=[], key=key)
