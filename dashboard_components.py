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
    return (
        f'<div class="prana-card" style="text-align:center;padding:14px 10px;">'
        f'<div class="prana-kpi-value">{value}</div>'
        f'<div class="prana-kpi-label">{label}</div>{delta_html}</div>'
    )


def _render_html(html: str) -> None:
    """Render raw HTML (st.html when available, else markdown)."""
    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(html, unsafe_allow_html=True)


def render_kpi_row(items: List[Tuple[str, str, str]]) -> None:
    cards = "".join(kpi_card(val, label, delta) for val, label, delta in items)
    _render_html(f'<div class="prana-kpi-grid">{cards}</div>')


def alert_card(title: str, message: str, severity: str = "CRITICAL") -> None:
    st.markdown(
        f'<div class="prana-alert"><strong>{title}</strong><br><span style="color:{COLORS["muted"]};font-weight:400">{message}</span></div>',
        unsafe_allow_html=True,
    )


def module_status_card(name: str, status: str) -> str:
    label = {"online": "Online", "degraded": "Degraded", "offline": "Offline"}.get(status, status)
    return (
        f'<div class="prana-card" style="padding:12px 14px;text-align:center;">'
        f'<div style="font-size:12px;font-weight:700;color:{COLORS["text"]}">{name}</div>'
        f'<div style="margin-top:6px;font-size:11px;">{status_dot(status)}{label}</div></div>'
    )


def render_module_status_grid(statuses: Dict[str, str], cols: int = 5) -> None:
    cards = "".join(module_status_card(name, statuses[name]) for name in statuses)
    _render_html(f'<div class="prana-module-grid">{cards}</div>')


def metric_row(label: str, value: str, badge: Optional[str] = None) -> None:
    badge_html = severity_badge(badge) if badge else ""
    st.markdown(
        f'<div class="prana-metric-row">'
        f'<span class="prana-metric-label">{label}</span>'
        f'<span class="prana-metric-value">{value} {badge_html}</span></div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "") -> None:
    sub = f'<div class="prana-section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="prana-section-title">{title}</div>{sub}', unsafe_allow_html=True)


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
