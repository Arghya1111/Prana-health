"""Waveform visualization components."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from app.dashboard.theme import COLORS, PLOTLY_LAYOUT


def _hex_rgba(hex_color: str, alpha: float = 0.15) -> str:
    h = hex_color.lstrip("#")[:6]
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _base(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def waveform_chart(
    signal: np.ndarray,
    sampling_rate: int,
    title: str,
    color: str = None,
    y_label: str = "Amplitude",
) -> go.Figure:
    color = color or COLORS["primary"]
    t = np.arange(len(signal)) / sampling_rate
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=signal, mode="lines",
        line=dict(color=color, width=1.2),
        fill="tozeroy", fillcolor=_hex_rgba(color),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title=y_label,
        height=280,
    )
    return _base(fig)


def eeg_multi_band(signal: np.ndarray, rate: int, title: str = "EEG Signal") -> go.Figure:
    return waveform_chart(signal, rate, title, color="#7C3AED", y_label="uV")


def ppg_chart(signal: np.ndarray, rate: int, title: str = "PPG Waveform") -> go.Figure:
    return waveform_chart(signal, rate, title, color="#DC2626", y_label="Intensity")


def ecg_chart(signal: np.ndarray, rate: int, title: str = "ECG Waveform") -> go.Figure:
    return waveform_chart(signal, rate, title, color=COLORS["primary"], y_label="mV")


def thermal_heatmap(grid: np.ndarray, title: str = "Thermal Map") -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=grid, colorscale="Hot", showscale=True,
        colorbar=dict(title="Temp"),
    ))
    fig.update_layout(title=title, height=320)
    return _base(fig)
