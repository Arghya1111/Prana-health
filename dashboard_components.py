"""
PRĀNA Clinical Dashboard — reusable UI components.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

import app  # noqa: F401 — project root on sys.path

from dashboard_theme import COLORS, severity_badge, status_dot


def glass_card(title: str, body_html: str = "") -> None:
    st.markdown(
        f'<div class="prana-card"><div class="prana-card-title">{title}</div>{body_html}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(value: str, label: str, delta: str = "") -> str:
    delta_html = f'<div class="prana-kpi-label">{delta}</div>' if delta else ""
    return f"""
    <div class="prana-card" style="text-align:center;padding:14px 10px;">
        <div class="prana-kpi-value">{value}</div>
        <div class="prana-kpi-label">{label}</div>
        {delta_html}
    </div>"""


def render_kpi_row(items: List[Tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (val, label, delta) in zip(cols, items):
        with col:
            st.markdown(kpi_card(val, label, delta), unsafe_allow_html=True)


def alert_card(title: str, message: str, severity: str = "CRITICAL") -> None:
    st.markdown(
        f'<div class="prana-alert"><strong>{title}</strong><br><span style="color:{COLORS["muted"]};font-weight:400">{message}</span></div>',
        unsafe_allow_html=True,
    )


def module_status_card(name: str, status: str) -> str:
    label = {"online": "Online", "degraded": "Degraded", "offline": "Offline"}.get(status, status)
    return f"""
    <div class="prana-card" style="padding:12px 14px;text-align:center;">
        <div style="font-size:12px;font-weight:700;color:{COLORS['text']}">{name}</div>
        <div style="margin-top:6px;font-size:11px;">{status_dot(status)}{label}</div>
    </div>"""


def render_module_status_grid(statuses: Dict[str, str], cols: int = 5) -> None:
    names = list(statuses.keys())
    for i in range(0, len(names), cols):
        row = names[i : i + cols]
        columns = st.columns(cols)
        for j, name in enumerate(row):
            with columns[j]:
                st.markdown(module_status_card(name, statuses[name]), unsafe_allow_html=True)


def metric_row(label: str, value: str, badge: Optional[str] = None) -> None:
    badge_html = severity_badge(badge) if badge else ""
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;padding:4px 0;font-size:12px;">'
        f'<span style="color:{COLORS["muted"]}">{label}</span>'
        f'<span style="font-weight:600;">{value} {badge_html}</span></div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "") -> None:
    sub = f'<div style="font-size:11px;color:{COLORS["muted"]};margin-bottom:12px;">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div style="font-size:16px;font-weight:700;color:{COLORS["text"]};margin:16px 0 4px;">{title}</div>{sub}', unsafe_allow_html=True)


def camera_placeholder(label: str = "Optical Camera") -> None:
    st.markdown(
        f"""
        <div class="prana-card" style="height:180px;display:flex;align-items:center;justify-content:center;
             background:linear-gradient(135deg,{COLORS['bg_panel']},{COLORS['surface']});">
            <div style="text-align:center;color:{COLORS['muted']};">
                <div style="font-size:28px;margin-bottom:8px;">📷</div>
                <div style="font-size:12px;font-weight:600;">{label}</div>
                <div style="font-size:10px;margin-top:4px;">Future-ready preview</div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_vitals_panel(patient: Dict[str, Any]) -> None:
    if not patient:
        st.info("Select a patient from the queue.")
        return
    metric_row("Patient ID", str(patient.get("patient_id", "—")))
    metric_row("Severity", str(patient.get("severity", "—")), str(patient.get("severity", "")))
    metric_row("Risk Score", f"{patient.get('risk_score', 0):.0f}/100")
    metric_row("AI Confidence", f"{patient.get('confidence', 0):.1f}%")
