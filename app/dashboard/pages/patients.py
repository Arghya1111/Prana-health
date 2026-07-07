"""Patient management page."""

from __future__ import annotations

import streamlit as st

from app.dashboard.components import download_buttons, page_header, patient_filters, severity_badge
from app.dashboard.data import filter_dataframe, load_fusion_df, load_triage_df
from app.dashboard.theme import severity_color


def render() -> None:
    page_header("Patient Management", "Search, filter, and review patient triage records")
    fusion = load_fusion_df(st.session_state.get("refresh_secs", 5))
    triage = load_triage_df(st.session_state.get("refresh_secs", 5))
    use_fusion = not fusion.empty
    df = fusion if use_fusion else triage
    sev_col = "overall_severity" if use_fusion else "overall"

    search, severity, sort = patient_filters("patients")

    if not df.empty:
        filtered = filter_dataframe(df, search, severity, sev_col)
        if sort == "Oldest":
            filtered = filtered.sort_values("timestamp")
        elif sort == "Risk (High)" and "risk_score" in filtered.columns:
            filtered = filtered.sort_values("risk_score", ascending=False)
        elif sort == "Risk (Low)" and "risk_score" in filtered.columns:
            filtered = filtered.sort_values("risk_score")
        else:
            filtered = filtered.sort_values("timestamp", ascending=False)
    else:
        filtered = df

    m1, m2, m3 = st.columns(3)
    m1.metric("Patients", df["patient_id"].nunique() if not df.empty else 0)
    m2.metric("Showing", len(filtered))
    m3.metric("Critical", int((filtered[sev_col] == "CRITICAL").sum()) if not filtered.empty else 0)

    if filtered.empty:
        st.info("No matching patients. Run the NATS pipeline to ingest data.")
        return

    for _, row in filtered.head(25).iterrows():
        pid = row["patient_id"]
        sev = row[sev_col]
        with st.expander(f"{pid}  ·  {sev}  ·  {row['timestamp']}", expanded=False):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(severity_badge(sev), unsafe_allow_html=True)
                if "risk_score" in row:
                    st.metric("Risk Score", f"{row['risk_score']:.1f}")
                if "overall_confidence" in row:
                    st.metric("AI Confidence", f"{row['overall_confidence']:.1f}%")
                elif "ecg_confidence" in row:
                    st.metric("ECG Confidence", f"{row['ecg_confidence']:.1f}%")
            with c2:
                st.write(row.get("action") or row.get("recommendation", ""))
                mod_cols = ["ecg_status", "eeg_status", "ppg_status", "spo2_status",
                            "thermal_status", "optical_status", "xray_status"]
                mods = {c.replace("_status", "").upper(): row[c] for c in mod_cols if c in row and row[c]}
                if mods:
                    st.json(mods)

    download_buttons(filtered, prefix="PRANA_Patients")
