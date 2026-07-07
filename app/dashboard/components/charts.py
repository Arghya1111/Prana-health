"""Plotly chart components."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.dashboard.theme import COLORS, PLOTLY_LAYOUT, SEVERITY_COLORS, severity_color


def _hex_rgba(hex_color: str, alpha: float = 0.25) -> str:
    """Convert #RRGGBB to rgba() — Plotly gauges do not accept 8-digit hex."""
    h = hex_color.lstrip("#")
    if len(h) == 8:
        h = h[:6]
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _base(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def severity_pie(df: pd.DataFrame, col: str = "overall") -> go.Figure:
    if df.empty or col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No data", showarrow=False)
        return _base(fig)
    counts = df[col].value_counts().reset_index()
    counts.columns = ["severity", "count"]
    colors = [SEVERITY_COLORS.get(s, COLORS["muted"]) for s in counts["severity"]]
    fig = px.pie(counts, names="severity", values="count", color="severity", color_discrete_sequence=colors, hole=0.45)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1))
    return _base(fig)


def risk_gauge(score: float) -> go.Figure:
    color = severity_color(
        "CRITICAL" if score >= 81 else "HIGH" if score >= 61 else "MODERATE" if score >= 41 else "LOW" if score >= 21 else "NORMAL"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix="/100", font=dict(size=28)),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1),
            bar=dict(color=color),
            steps=[
                dict(range=[0, 20], color=_hex_rgba(COLORS["success"])),
                dict(range=[20, 40], color=_hex_rgba(COLORS["low"])),
                dict(range=[40, 60], color=_hex_rgba(COLORS["warning"])),
                dict(range=[60, 80], color=_hex_rgba(COLORS["high"])),
                dict(range=[80, 100], color=_hex_rgba(COLORS["critical"])),
            ],
        ),
        title=dict(text="Patient Risk Score"),
    ))
    return _base(fig)


def spo2_gauge(value: float) -> go.Figure:
    color = COLORS["success"] if value >= 95 else COLORS["warning"] if value >= 90 else COLORS["critical"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number=dict(suffix="%", font=dict(size=32)),
        gauge=dict(
            axis=dict(range=[70, 100]),
            bar=dict(color=color),
            steps=[
                dict(range=[70, 85], color=_hex_rgba(COLORS["critical"], 0.35)),
                dict(range=[85, 90], color=_hex_rgba(COLORS["high"], 0.35)),
                dict(range=[90, 95], color=_hex_rgba(COLORS["warning"], 0.35)),
                dict(range=[95, 100], color=_hex_rgba(COLORS["success"], 0.35)),
            ],
        ),
        title=dict(text="SpO2"),
    ))
    return _base(fig)


def timeline_chart(df: pd.DataFrame, y_col: str, title: str, color: str = None) -> go.Figure:
    if df.empty:
        return _base(go.Figure())
    fig = px.line(df.sort_values("timestamp"), x="timestamp", y=y_col, markers=True, title=title)
    if color:
        fig.update_traces(line_color=color)
    return _base(fig)


def hr_trend(df: pd.DataFrame) -> go.Figure:
    if df.empty or "mean_hr" not in df.columns:
        return _base(go.Figure())
    d = df.sort_values("timestamp").tail(30)
    fig = px.area(d, x="timestamp", y="mean_hr", title="Heart Rate Trend", color_discrete_sequence=[COLORS["primary"]])
    fig.update_layout(yaxis_title="BPM")
    return _base(fig)


def confidence_trend(df: pd.DataFrame, col: str = "overall_confidence") -> go.Figure:
    if df.empty or col not in df.columns:
        if not df.empty and "ecg_confidence" in df.columns:
            col = "ecg_confidence"
        else:
            return _base(go.Figure())
    d = df.sort_values("timestamp").tail(30)
    fig = px.line(d, x="timestamp", y=col, title="AI Confidence Trend", markers=True,
                  color_discrete_sequence=[COLORS["secondary"]])
    fig.update_layout(yaxis_title="%", yaxis=dict(range=[0, 100]))
    return _base(fig)


def severity_timeline(df: pd.DataFrame, sev_col: str = "overall") -> go.Figure:
    if df.empty:
        return _base(go.Figure())
    order = {"NORMAL": 0, "LOW": 1, "MODERATE": 2, "HIGH": 3, "CRITICAL": 4}
    d = df.sort_values("timestamp").tail(40).copy()
    d["sev_num"] = d[sev_col].map(order).fillna(0)
    colors = [severity_color(s) for s in d[sev_col]]
    fig = go.Figure(go.Scatter(
        x=d["timestamp"], y=d["sev_num"], mode="markers+lines",
        marker=dict(size=10, color=colors), line=dict(color=COLORS["border"]),
    ))
    fig.update_layout(
        title="Severity Timeline",
        yaxis=dict(tickvals=[0, 1, 2, 3, 4], ticktext=["NORMAL", "LOW", "MODERATE", "HIGH", "CRITICAL"]),
    )
    return _base(fig)


def cases_per_hour(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        return _base(go.Figure())
    d = df.copy()
    d["hour"] = d["timestamp"].dt.floor("h")
    counts = d.groupby("hour").size().reset_index(name="cases")
    fig = px.bar(counts, x="hour", y="cases", title="Cases per Hour", color_discrete_sequence=[COLORS["primary"]])
    return _base(fig)


def xray_findings_bar(flagged: dict) -> go.Figure:
    if not flagged:
        return _base(go.Figure())
    items = sorted(flagged.items(), key=lambda x: x[1], reverse=True)[:8]
    fig = px.bar(
        x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
        title="X-Ray Pathology Scores", color_discrete_sequence=[COLORS["high"]],
    )
    fig.update_layout(xaxis_title="Probability", yaxis_title="")
    return _base(fig)


def module_confidence_bar(modules: dict) -> go.Figure:
    if not modules:
        return _base(go.Figure())
    names, confs = [], []
    for name, data in modules.items():
        if isinstance(data, dict):
            names.append(name)
            confs.append(data.get("confidence", 0))
    fig = px.bar(x=names, y=confs, title="Module Confidence", color_discrete_sequence=[COLORS["secondary"]])
    fig.update_layout(yaxis=dict(range=[0, 100]), yaxis_title="%")
    return _base(fig)
