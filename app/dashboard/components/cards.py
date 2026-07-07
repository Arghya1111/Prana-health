"""KPI and status card components."""

from __future__ import annotations

from typing import Optional

import streamlit as st

from app.dashboard.theme import COLORS, severity_badge, severity_color


def kpi_row(items: list[tuple[str, str, Optional[str]]], cols: Optional[int] = None) -> None:
    """Render a row of KPI cards. Each item: (value, label, optional_sub)."""
    n = cols or len(items)
    columns = st.columns(n)
    for col, (value, label, sub) in zip(columns, items):
        sub_html = f'<div class="kpi-delta">{sub}</div>' if sub else ""
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div>{sub_html}</div>',
            unsafe_allow_html=True,
        )


def confidence_gauge(confidence: float, title: str = "AI Confidence") -> None:
    color = COLORS["success"] if confidence >= 90 else COLORS["warning"] if confidence >= 70 else COLORS["critical"]
    st.markdown(
        f'<div class="med-card" style="text-align:center;">'
        f'<div class="med-card-title">{title}</div>'
        f'<div style="font-size:36px;font-weight:700;color:{color};">{confidence:.1f}%</div>'
        f'<div style="background:{COLORS["border"]};border-radius:6px;height:8px;margin-top:10px;">'
        f'<div style="background:{color};width:{min(100,confidence):.0f}%;height:8px;border-radius:6px;"></div>'
        f"</div></div>",
        unsafe_allow_html=True,
    )


def module_status_card(
    module: str,
    status: str,
    confidence: float,
    severity: str = "NORMAL",
    extra: str = "",
) -> str:
    """Return HTML for a module status card."""
    color = severity_color(severity)
    extra_html = f'<div style="font-size:11px;color:{COLORS["muted"]};margin-top:4px;">{extra}</div>' if extra else ""
    return (
        f'<div class="med-card" style="border-left:4px solid {color};">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="font-weight:700;font-size:14px;">{module}</span>'
        f"{severity_badge(severity)}"
        f"</div>"
        f'<div style="margin-top:8px;font-size:13px;">Status: <b>{status}</b></div>'
        f'<div style="font-size:12px;color:{COLORS["muted"]};">Confidence: {confidence:.1f}%</div>'
        f"{extra_html}</div>"
    )


def alert_banner(severity: str, title: str, body: str) -> None:
    color = severity_color(severity)
    st.markdown(
        f'<div class="med-card" style="border-left:4px solid {color};background:{color}08;">'
        f'<div style="font-weight:700;color:{color};">{title}</div>'
        f'<div style="font-size:13px;margin-top:4px;">{body}</div></div>',
        unsafe_allow_html=True,
    )
