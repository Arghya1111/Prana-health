"""Reusable layout components."""

from __future__ import annotations

import streamlit as st

from app.dashboard.theme import COLORS


def page_header(title: str, subtitle: str = "", live: bool = False) -> None:
    live_html = '<span class="live-dot"></span> LIVE' if live else ""
    st.markdown(
        f'<div class="page-header">{title}</div>'
        f'<div class="page-sub">{subtitle} {live_html}</div>',
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="med-card-title">{title}</div>', unsafe_allow_html=True)


def med_card_start(title: str = "") -> None:
    if title:
        st.markdown(f'<div class="med-card"><div class="med-card-title">{title}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="med-card">', unsafe_allow_html=True)


def med_card_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.sidebar.markdown(
        f'<div class="prana-brand">PRANA</div>'
        f'<div class="prana-sub">AI Clinical Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.divider()


def sidebar_status(nats: bool, db: bool) -> None:
    nats_dot = "dot-g" if nats else "dot-r"
    db_dot = "dot-g" if db else "dot-r"
    st.sidebar.markdown(
        f'<div style="font-size:12px;color:{COLORS["muted"]};">'
        f'<span class="{nats_dot}"></span>NATS {"Online" if nats else "Offline"}<br>'
        f'<span class="{db_dot}"></span>Database {"Connected" if db else "Missing"}'
        f"</div>",
        unsafe_allow_html=True,
    )
