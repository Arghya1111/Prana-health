"""
PRĀNA Clinical Dashboard — Plotly & Altair chart builders.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import app  # noqa: F401 — project root on sys.path

from dashboard_theme import COLORS, PLOTLY_LAYOUT, SEVERITY_COLORS, severity_color


def _apply_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def _hex_rgba(hex_color: str, alpha: float = 0.3) -> str:
    h = hex_color.lstrip("#")[:6]
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Waveforms ─────────────────────────────────────────────────────────────────

def ecg_waveform_chart(signal: np.ndarray, rate: int, hr: float) -> go.Figure:
    t = np.arange(len(signal)) / rate
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=signal, mode="lines", name="ECG",
        line=dict(color=COLORS["secondary"], width=1.2),
    ))
    fig.update_layout(title=f"Live ECG — {hr:.0f} BPM", xaxis_title="Time (s)", yaxis_title="mV")
    return _apply_layout(fig)


def eeg_waveform_chart(signal: np.ndarray, rate: int) -> go.Figure:
    t = np.arange(len(signal)) / rate
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=signal, mode="lines", name="EEG",
        line=dict(color=COLORS["primary"], width=1),
    ))
    fig.update_layout(title="Live EEG", xaxis_title="Time (s)", yaxis_title="µV")
    return _apply_layout(fig)


def ppg_waveform_chart(signal: np.ndarray, rate: int, hr: float) -> go.Figure:
    t = np.arange(len(signal)) / rate
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=signal, mode="lines", name="PPG",
        line=dict(color="#EF4444", width=1.2),
        fill="tozeroy", fillcolor=_hex_rgba("#EF4444", 0.15),
    ))
    fig.update_layout(title=f"PPG — {hr:.0f} BPM", xaxis_title="Time (s)", yaxis_title="AU")
    return _apply_layout(fig)


# ── Gauges & distribution ─────────────────────────────────────────────────────

def spo2_gauge(value: float) -> go.Figure:
    color = COLORS["secondary"] if value >= 95 else COLORS["moderate"] if value >= 90 else COLORS["critical"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(suffix="%", font=dict(size=36, color=COLORS["text"])),
        gauge=dict(
            axis=dict(range=[70, 100], tickwidth=1, tickcolor=COLORS["muted"]),
            bar=dict(color=color, thickness=0.75),
            bgcolor=COLORS["bg_panel"],
            steps=[
                dict(range=[70, 85], color=_hex_rgba(COLORS["critical"], 0.35)),
                dict(range=[85, 90], color=_hex_rgba(COLORS["high"], 0.35)),
                dict(range=[90, 95], color=_hex_rgba(COLORS["moderate"], 0.35)),
                dict(range=[95, 100], color=_hex_rgba(COLORS["secondary"], 0.35)),
            ],
        ),
        title=dict(text="SpO₂", font=dict(size=14, color=COLORS["muted"])),
    ))
    return _apply_layout(fig)


def spo2_trend_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _apply_layout(go.Figure())
    fig = px.line(df, x="minute", y="spo2", title="SpO₂ Trend — Last 5 Min")
    fig.add_hline(y=95, line_dash="dash", line_color=COLORS["secondary"], opacity=0.6)
    fig.add_hline(y=90, line_dash="dash", line_color=COLORS["moderate"], opacity=0.6)
    fig.update_traces(line_color=COLORS["primary"], line_width=2)
    fig.update_layout(yaxis=dict(range=[75, 100]))
    return _apply_layout(fig)


def severity_donut(df: pd.DataFrame, col: str = "overall_severity") -> go.Figure:
    if df.empty or col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False, font=dict(color=COLORS["muted"]))
        return _apply_layout(fig)
    counts = df[col].value_counts().reset_index()
    counts.columns = ["severity", "count"]
    colors = [SEVERITY_COLORS.get(s, COLORS["muted"]) for s in counts["severity"]]
    fig = px.pie(counts, names="severity", values="count", hole=0.55, color="severity", color_discrete_sequence=colors)
    fig.update_layout(title="Severity Distribution", showlegend=True, legend=dict(orientation="h", y=-0.15))
    return _apply_layout(fig)


def risk_timeline_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty or "risk_score" not in df.columns:
        return _apply_layout(go.Figure())
    d = df.sort_values("timestamp").tail(40)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["timestamp"], y=d["risk_score"], mode="lines+markers",
        name="Risk Score", line=dict(color=COLORS["primary"], width=2),
        marker=dict(size=6, color=[severity_color(
            "CRITICAL" if v >= 81 else "HIGH" if v >= 61 else "MODERATE" if v >= 41 else "NORMAL"
        ) for v in d["risk_score"]]),
    ))
    fig.update_layout(title="Risk Timeline", yaxis_title="Risk Score", yaxis=dict(range=[0, 100]))
    return _apply_layout(fig)


def fusion_radar_chart(scores: Dict[str, float]) -> go.Figure:
    labels = list(scores.keys())
    values = list(scores.values())
    values.append(values[0])
    labels.append(labels[0])
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=labels, fill="toself",
        fillcolor=_hex_rgba(COLORS["primary"], 0.25),
        line=dict(color=COLORS["primary"], width=2),
        name="Module Scores",
    ))
    fig.update_layout(
        title="Fusion Engine — Module Scores",
        polar=dict(
            bgcolor=COLORS["bg_panel"],
            radialaxis=dict(visible=True, range=[0, 100], gridcolor=COLORS["border"]),
            angularaxis=dict(gridcolor=COLORS["border"]),
        ),
    )
    return _apply_layout(fig)


def thermal_heatmap(data: np.ndarray) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=data, colorscale="Inferno", colorbar=dict(title="°C"),
    ))
    fig.update_layout(title="Thermal Heatmap", xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    return _apply_layout(fig)


def model_performance_bar(metrics: Dict[str, Dict[str, float]], metric: str = "accuracy") -> go.Figure:
    modules = list(metrics.keys())
    vals = [metrics[m].get(metric, 0) * 100 for m in modules]
    colors = [COLORS["primary"] if v >= 90 else COLORS["moderate"] if v >= 80 else COLORS["critical"] for v in vals]
    fig = go.Figure(go.Bar(x=modules, y=vals, marker_color=colors, text=[f"{v:.1f}%" for v in vals], textposition="outside"))
    fig.update_layout(title=f"Model {metric.title()}", yaxis=dict(range=[0, 105], title="%"))
    return _apply_layout(fig)


def analytics_patients_chart(daily: pd.DataFrame) -> alt.Chart:
    """Altair daily patients bar chart."""
    return (
        alt.Chart(daily)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=COLORS["primary"])
        .encode(x=alt.X("day:N", title="Day"), y=alt.Y("patients:Q", title="Patients"))
        .properties(title="Daily Patients", height=180)
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor=COLORS["border"], labelColor=COLORS["muted"], titleColor=COLORS["text"])
        .configure_title(color=COLORS["text"], fontSize=13)
    )


def age_distribution_chart(ages: List[int]) -> alt.Chart:
    df = pd.DataFrame({"age": ages})
    return (
        alt.Chart(df)
        .mark_bar(color=COLORS["secondary"])
        .encode(
            x=alt.X("age:Q", bin=alt.Bin(maxbins=12), title="Age"),
            y=alt.Y("count()", title="Count"),
        )
        .properties(title="Age Distribution", height=160)
        .configure_view(strokeWidth=0)
        .configure_axis(gridColor=COLORS["border"], labelColor=COLORS["muted"])
    )
