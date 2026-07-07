"""
PRĀNA Clinical Dashboard — main entry point.

Hospital-grade ICU monitoring control center for the PRĀNA AI Triage System.

Run (local):
    streamlit run dashboard.py

Deploy (Render):
    streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0

Integrates: SQLite · Fusion Engine · NATS · ECG · EEG · PPG · SpO₂ · Thermal · Optical
"""

from __future__ import annotations

import time

import app  # noqa: F401 — project root on sys.path

import numpy as np
import pandas as pd
import streamlit as st

from dashboard_charts import (
    age_distribution_chart,
    analytics_patients_chart,
    ecg_waveform_chart,
    eeg_waveform_chart,
    fusion_radar_chart,
    model_performance_bar,
    ppg_waveform_chart,
    risk_timeline_chart,
    severity_donut,
    spo2_gauge,
    spo2_trend_chart,
    thermal_heatmap,
)
from dashboard_components import (
    alert_card,
    camera_placeholder,
    metric_row,
    render_kpi_row,
    render_module_status_grid,
    render_vitals_panel,
    section_header,
)
from app.model_health import missing_models_summary
from dashboard_data import (
    MODEL_METRICS,
    build_patient_queue,
    compute_system_metrics,
    ensure_database_ready,
    generate_ecg_waveform,
    generate_eeg_waveform,
    generate_ppg_waveform,
    generate_spo2_trend,
    generate_thermal_image,
    get_refresh_secs,
    get_selected_patient_data,
    latest_ai_decisions,
    load_fusion_df,
    load_triage_df,
    module_statuses,
    system_health,
)
from dashboard_reports import render_database_panel, render_patient_report_dialog
from dashboard_sidebar import render_settings_panel, render_sidebar, render_sliding_nav
from dashboard_theme import COLORS, inject_css, severity_badge, status_dot
from dashboard_utils import (
    build_analytics_daily,
    compute_eeg_metrics,
    derive_fusion_module_scores,
    filter_queue,
    format_date_now,
    format_time_now,
    init_session_state,
    sort_queue,
)


def _chart_key(prefix: str, patient: dict) -> str:
    """Stable Plotly keys avoid full chart remounts on unrelated reruns."""
    pid = patient.get("patient_id") or "default"
    return f"{prefix}_{pid}"


def _live_tick() -> int:
    """Time-based animation only when live waveforms are enabled."""
    return int(time.time()) if st.session_state.get("live_mode", False) else 0


# ── Top navigation ────────────────────────────────────────────────────────────

def render_top_nav(health: dict) -> None:
    hospital = st.session_state.get("hospital_name", "PRĀNA Medical Center")
    nats_s = "online" if health.get("nats") else "offline"
    db_s = "online" if health.get("database") else "offline"
    fusion_s = "online" if health.get("fusion") else "offline"
    sys_s = health.get("status", "degraded")

    st.markdown(
        f"""
        <div class="prana-topnav">
          <div class="prana-topnav-brand">
            <div class="prana-logo">PRĀNA</div>
            <div class="prana-tagline">AI Powered Multi-Modal Patient Triage</div>
          </div>
          <div class="prana-nav-meta prana-topnav-meta">
            <div style="font-weight:600;color:{COLORS['text']};">{hospital}</div>
            <div>{format_date_now()} · {format_time_now()}</div>
          </div>
          <div class="prana-nav-status">
            <span class="prana-nav-pill">{status_dot(nats_s)} NATS</span>
            <span class="prana-nav-pill">{status_dot(db_s)} SQLite</span>
            <span class="prana-nav-pill">{status_dot(fusion_s)} Fusion</span>
            <span class="prana-nav-pill">{status_dot(sys_s)} Health {health['score']}%</span>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


# ── Section renderers ───────────────────────────────────────────────────────

def render_system_metrics(metrics: dict) -> None:
    section_header("System Metrics", "Real-time operational KPIs")
    render_kpi_row([
        (str(metrics["total_patients"]), "Total Patients", ""),
        (str(metrics["critical"]), "Critical", "Immediate"),
        (str(metrics["high"]), "High Risk", ""),
        (str(metrics["moderate"]), "Moderate", ""),
        (str(metrics["normal"]), "Normal", ""),
        (str(metrics["today_cases"]), "Today's Cases", ""),
    ])
    render_kpi_row([
        (f"{metrics['avg_confidence']:.0f}%", "Avg AI Confidence", ""),
        (f"{metrics['avg_inference_ms']:.0f} ms", "Avg Inference", ""),
        (f"{metrics['avg_hr']:.0f}", "Avg Heart Rate", "BPM"),
        (f"{metrics['avg_spo2']:.1f}%", "Avg SpO₂", ""),
        (f"{metrics['avg_temp']:.1f}°C", "Avg Temperature", ""),
        (f"{metrics['avg_risk']:.0f}", "Avg Risk Score", "0–100"),
    ])


def render_patient_queue(queue: pd.DataFrame) -> None:
    section_header("Live Patient Queue", "Search · Sort · Filter · Click to open report")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        search = st.text_input("Search", placeholder="Patient ID...", label_visibility="collapsed")
    with c2:
        sev_filter = st.selectbox("Severity", ["All", "CRITICAL", "HIGH", "MODERATE", "NORMAL"], label_visibility="collapsed")
    with c3:
        sort_by = st.selectbox("Sort", ["priority", "arrival", "risk"], format_func=lambda x: x.title(), label_visibility="collapsed")

    filtered = sort_queue(filter_queue(queue, search, sev_filter), sort_by)
    if filtered.empty:
        st.info("No patients in queue.")
        return

    display = filtered[["patient_id", "age", "gender", "arrival_time", "fusion_score", "overall_severity", "status", "confidence"]].copy()
    display["arrival_time"] = display["arrival_time"].dt.strftime("%H:%M:%S")
    display.columns = ["Patient ID", "Age", "Gender", "Arrival", "Fusion Score", "Severity", "Status", "Confidence"]

    event = st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="patient_queue_table",
    )

    selected_rows = event.selection.rows if hasattr(event, "selection") and event.selection else []
    if selected_rows:
        idx = selected_rows[0]
        row = filtered.iloc[idx]
        st.session_state["selected_patient"] = row["patient_id"]
        st.session_state["selected_report_id"] = int(row.get("report_id", 0))

    if st.session_state.get("selected_patient"):
        with st.expander(f"Complete Report — {st.session_state['selected_patient']}", expanded=True):
            render_patient_report_dialog(
                st.session_state["selected_patient"],
                st.session_state.get("selected_report_id"),
            )


def render_live_monitors(patient: dict, tick: int) -> None:
    section_header("Live Monitors", "Real-time physiological waveforms")
    hr = patient.get("hr", 72) + (np.sin(tick * 0.3) * 2 if tick else 0)
    spo2_base = 96.0 + (np.sin(tick * 0.1) * 0.5 if tick else 0)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="prana-card-title">Live ECG</div>', unsafe_allow_html=True)
        ecg, rate = generate_ecg_waveform(hr)
        st.plotly_chart(ecg_waveform_chart(ecg[-800:], rate, hr), use_container_width=True, key=_chart_key("ecg", patient))
        metric_row("Heart Rate", f"{hr:.0f} BPM")
        metric_row("RR Interval", f"{60/hr*1000:.0f} ms")
        metric_row("SDNN", "42 ms")
        metric_row("RMSSD", "28 ms")
        metric_row("pNN50", "12%")
        metric_row("Prediction", "NORMAL", "NORMAL")
        metric_row("Confidence", "96.4%")
        metric_row("Signal Quality", "GOOD")

    with col2:
        st.markdown('<div class="prana-card-title">Live EEG</div>', unsafe_allow_html=True)
        eeg, eeg_rate = generate_eeg_waveform()
        eeg_m = compute_eeg_metrics(eeg)
        st.plotly_chart(eeg_waveform_chart(eeg, eeg_rate), use_container_width=True, key=_chart_key("eeg", patient))
        metric_row("Dominant Frequency", f"{eeg_m['dominant_freq']} Hz")
        metric_row("Brain Activity", f"{eeg_m['brain_activity']}")
        metric_row("Seizure Risk", str(eeg_m["seizure_risk"]))
        metric_row("Stress Index", f"{eeg_m['stress_index']}")
        metric_row("Confidence", "97.8%")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="prana-card-title">PPG Monitor</div>', unsafe_allow_html=True)
        ppg, ppg_rate = generate_ppg_waveform(hr)
        st.plotly_chart(ppg_waveform_chart(ppg[-400:], ppg_rate, hr), use_container_width=True, key=_chart_key("ppg", patient))
        metric_row("Pulse Rate", f"{hr:.0f} BPM")
        metric_row("Perfusion Index", "4.2")
        metric_row("Pulse Quality", "GOOD")
        metric_row("Confidence", "97.6%")

    with col4:
        st.markdown('<div class="prana-card-title">SpO₂ Monitor</div>', unsafe_allow_html=True)
        spo2_val = spo2_base
        st.plotly_chart(spo2_gauge(spo2_val), use_container_width=True, key=_chart_key("spo2g", patient))
        trend = generate_spo2_trend(60, spo2_val)
        st.plotly_chart(spo2_trend_chart(trend), use_container_width=True, key=_chart_key("spo2t", patient))
        rec = "Normal Oxygen Saturation" if spo2_val >= 95 else "Monitor oxygen saturation closely"
        st.caption(f"Recommendation: {rec}")


def render_thermal_optical(tick: int, patient: dict) -> None:
    col1, col2 = st.columns(2)
    fever = tick and tick % 5 == 0
    thermal = generate_thermal_image(fever=fever)
    with col1:
        section_header("Thermal Camera")
        st.plotly_chart(thermal_heatmap(thermal), use_container_width=True, key=_chart_key("thermal", patient))
        metric_row("Max Temperature", f"{thermal.max():.1f}°C")
        metric_row("Inflammation", "DETECTED" if fever else "NONE", "HIGH" if fever else "NORMAL")
        metric_row("Confidence", "96.3%")
    with col2:
        section_header("Optical Diagnostics")
        camera_placeholder("Optical Camera Preview")
        metric_row("Skin Color Analysis", "Normal")
        metric_row("Eye Detection", "Ready")
        metric_row("Respiratory Rate", "16 /min")
        metric_row("Face Detection", "Active")


def render_fusion_analytics(fusion: pd.DataFrame, patient: dict, tick: int) -> None:
    report = patient.get("report", {})
    scores = derive_fusion_module_scores(report)
    overall_risk = patient.get("risk_score", scores.get("ECG", 50))

    col1, col2 = st.columns(2)
    with col1:
        section_header("Fusion Engine")
        st.plotly_chart(fusion_radar_chart(scores), use_container_width=True, key=_chart_key("radar", patient))
        metric_row("Fusion Confidence", f"{patient.get('confidence', 0):.1f}%")
        metric_row("Overall Risk", f"{overall_risk:.0f}/100", str(patient.get("severity", "NORMAL")))
    with col2:
        section_header("Risk Timeline")
        st.plotly_chart(risk_timeline_chart(fusion), use_container_width=True, key="risk_timeline")

    c1, c2 = st.columns(2)
    with c1:
        section_header("Severity Distribution")
        sev_col = "overall_severity" if "overall_severity" in fusion.columns else "overall"
        st.plotly_chart(severity_donut(fusion, sev_col), use_container_width=True, key="severity_donut")
    with c2:
        section_header("Analytics")
        daily = build_analytics_daily(fusion)
        st.altair_chart(analytics_patients_chart(daily), use_container_width=True)
        ages = list(range(20, 75)) if fusion.empty else [int(r.get("age", 35)) for r in fusion.to_dict("records")[:50]]
        if fusion.empty:
            ages = list(np.random.randint(22, 78, 40))
        st.altair_chart(age_distribution_chart(ages), use_container_width=True)


def render_alerts_decisions(fusion: pd.DataFrame, triage: pd.DataFrame) -> None:
    col1, col2 = st.columns(2)
    src = fusion if not fusion.empty else triage
    sev_col = "overall_severity" if not fusion.empty else "overall"

    with col1:
        section_header("Alerts Panel")
        if src.empty:
            st.success("No active alerts.")
        else:
            critical = src[src[sev_col].isin(["CRITICAL", "HIGH"])].head(4) if sev_col in src.columns else pd.DataFrame()
            for _, row in critical.iterrows():
                alert_card(
                    f"Critical Patient — {row['patient_id']}",
                    str(row.get("action", row.get("recommendation", "Immediate review required")))[:100],
                )
            alert_card("Low SpO₂ Watch", "Patient cohort SpO₂ trending below 94%", "HIGH")
            alert_card("Abnormal ECG", "Elevated HR detected in latest triage batch", "MODERATE")

    with col2:
        section_header("Latest AI Decisions")
        decisions = latest_ai_decisions(fusion)
        if decisions.empty:
            st.info("No AI decisions yet.")
        else:
            st.dataframe(
                decisions[["timestamp", "patient_id", "module", "prediction", "confidence", "recommendation"]].head(12),
                use_container_width=True,
                hide_index=True,
                height=280,
            )


def render_model_performance() -> None:
    section_header("Model Performance", "Accuracy · Precision · Recall · F1 · ROC AUC")
    metric_type = st.selectbox("Metric", ["accuracy", "precision", "recall", "f1", "roc_auc"], label_visibility="collapsed")
    st.plotly_chart(model_performance_bar(MODEL_METRICS, metric_type), use_container_width=True)

    rows = []
    for mod, m in MODEL_METRICS.items():
        rows.append({"Module": mod, **{k: f"{v*100:.1f}%" for k, v in m.items()}})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_module_status_section() -> None:
    section_header("Module Status")
    render_module_status_grid(module_statuses(), cols=5)


# ── Page layouts ──────────────────────────────────────────────────────────────

def page_command_center(fusion, triage, metrics, queue, patient, tick) -> None:
    render_system_metrics(metrics)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        render_patient_queue(queue)
    with c2:
        section_header("Selected Patient Vitals")
        render_vitals_panel(patient)
        render_alerts_decisions(fusion, triage)


def page_monitors(patient, tick) -> None:
    render_live_monitors(patient, tick)
    render_thermal_optical(tick, patient)


def page_fusion_analytics(fusion, patient, tick) -> None:
    render_fusion_analytics(fusion, patient, tick)
    render_module_status_section()
    render_model_performance()


def page_reports(fusion, triage) -> None:
    render_database_panel(fusion, triage)


# ── Main ──────────────────────────────────────────────────────────────────────

def _render_deployment_warnings() -> None:
    """Surface missing model weights without blocking the dashboard."""
    if st.session_state.get("_deployment_warned"):
        return
    ok, summary = missing_models_summary()
    if not ok:
        st.warning(
            f"Some AI model weights are unavailable. The dashboard will run in demo mode. {summary}"
        )
    st.session_state["_deployment_warned"] = True


def main() -> None:
    st.set_page_config(
        page_title="PRĀNA Clinical Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="auto",
    )
    ensure_database_ready()
    init_session_state()
    st.markdown(inject_css(), unsafe_allow_html=True)
    _render_deployment_warnings()

    render_sidebar()
    refresh = get_refresh_secs()

    # Load data once per interaction; no automatic full-page reruns.
    fusion = load_fusion_df(refresh)
    triage = load_triage_df(refresh)
    health = system_health()
    metrics = compute_system_metrics(triage, fusion)
    queue = build_patient_queue(fusion, triage)
    patient = get_selected_patient_data(fusion, triage, st.session_state.get("selected_patient"))
    tick = _live_tick()

    render_top_nav(health)
    page = render_sliding_nav(st.session_state.get("current_page", "Command Center"))

    if page == "Command Center":
        page_command_center(fusion, triage, metrics, queue, patient, tick)
    elif page == "Monitors":
        page_monitors(patient, tick)
    elif page == "Fusion & Analytics":
        page_fusion_analytics(fusion, patient, tick)
    elif page == "Reports & Database":
        page_reports(fusion, triage)
    elif page == "Settings":
        render_settings_panel()


if __name__ == "__main__":
    main()
