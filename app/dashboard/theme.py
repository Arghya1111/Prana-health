"""PRĀNA dashboard theme — colors, CSS, severity styling."""

from __future__ import annotations

from typing import Dict

COLORS: Dict[str, str] = {
    "primary": "#38BDF8",
    "secondary": "#00C2FF",
    "success": "#34D399",
    "warning": "#FBBF24",
    "critical": "#F87171",
    "high": "#FB923C",
    "low": "#60A5FA",
    "bg": "#000000",
    "surface": "#111111",
    "surface_raised": "#1A1A1A",
    "border": "#2A2A2A",
    "muted": "#94A3B8",
    "text": "#F1F5F9",
    "card_shadow": "0 4px 16px rgba(0,0,0,0.45)",
}

SEVERITY_COLORS: Dict[str, str] = {
    "CRITICAL": COLORS["critical"],
    "HIGH": COLORS["high"],
    "MODERATE": COLORS["warning"],
    "LOW": COLORS["low"],
    "NORMAL": COLORS["success"],
}

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text"], size=12),
    margin=dict(l=40, r=20, t=40, b=40),
)


def severity_color(severity: str) -> str:
    return SEVERITY_COLORS.get(str(severity).upper(), COLORS["muted"])


def severity_badge(severity: str) -> str:
    sev = str(severity).upper()
    cls = {
        "CRITICAL": "b-critical",
        "HIGH": "b-high",
        "MODERATE": "b-moderate",
        "LOW": "b-low",
        "NORMAL": "b-normal",
    }.get(sev, "b-normal")
    return f'<span class="{cls}">{sev}</span>'


def inject_css() -> str:
    c = COLORS
    return f"""
<style>
  /* ── App shell — black background ─────────────────────── */
  .stApp, [data-testid="stAppViewContainer"], .main, .block-container {{
    background:{c["bg"]} !important;
    color:{c["text"]};
  }}
  [data-testid="stSidebar"] {{
    background:linear-gradient(180deg,{c["surface"]} 0%,{c["bg"]} 100%) !important;
    border-right:1px solid {c["border"]};
  }}
  [data-testid="stSidebar"] * {{
    color:{c["text"]} !important;
  }}
  #MainMenu, footer, header {{ visibility:hidden; }}
  [data-testid="stStatusWidget"] {{ display:none; }}

  /* ── Typography ─────────────────────────────────────── */
  h1, h2, h3, h4, p, label, span, .stMarkdown {{
    color:{c["text"]};
  }}
  .prana-brand {{
    font-size:22px; font-weight:800; letter-spacing:3px;
    color:{c["primary"]} !important; padding:8px 0;
  }}
  .prana-sub {{ font-size:11px; color:{c["muted"]} !important; letter-spacing:.6px; }}

  /* ── Cards ──────────────────────────────────────────── */
  .med-card {{
    background:{c["surface_raised"]}; border:1px solid {c["border"]};
    border-radius:14px; padding:18px 20px;
    box-shadow:{c["card_shadow"]}; margin-bottom:12px; color:{c["text"]};
  }}
  .med-card-title {{
    font-size:13px; font-weight:700; color:{c["primary"]};
    text-transform:uppercase; letter-spacing:.5px; margin-bottom:10px;
  }}

  .kpi-card {{
    background:{c["surface_raised"]}; border:1px solid {c["border"]};
    border-radius:14px; padding:18px; text-align:center;
    box-shadow:{c["card_shadow"]}; transition:transform .15s;
  }}
  .kpi-card:hover {{ transform:translateY(-2px); border-color:{c["primary"]}55; }}
  .kpi-value {{ font-size:28px; font-weight:700; color:{c["primary"]}; line-height:1.1; }}
  .kpi-label {{ font-size:11px; color:{c["muted"]}; font-weight:600;
                text-transform:uppercase; letter-spacing:.4px; margin-top:6px; }}
  .kpi-delta {{ font-size:11px; color:{c["muted"]}; margin-top:4px; }}

  /* ── Severity badges ────────────────────────────────── */
  .b-critical,.b-high,.b-moderate,.b-low,.b-normal {{
    border-radius:999px; padding:3px 12px; font-size:11px; font-weight:700;
    display:inline-block; letter-spacing:.3px;
  }}
  .b-critical {{ background:{c["critical"]}22; color:{c["critical"]}; border:1px solid {c["critical"]}55; }}
  .b-high     {{ background:{c["high"]}22; color:{c["high"]}; border:1px solid {c["high"]}55; }}
  .b-moderate {{ background:{c["warning"]}22; color:{c["warning"]}; border:1px solid {c["warning"]}55; }}
  .b-low      {{ background:{c["low"]}22; color:{c["low"]}; border:1px solid {c["low"]}55; }}
  .b-normal   {{ background:{c["success"]}22; color:{c["success"]}; border:1px solid {c["success"]}55; }}

  .page-header {{
    font-size:24px; font-weight:700; color:{c["text"]};
    margin:0 0 4px; letter-spacing:-.3px;
  }}
  .page-sub {{ font-size:13px; color:{c["muted"]}; margin-bottom:20px; }}

  .live-dot {{
    display:inline-block; width:8px; height:8px; background:{c["success"]};
    border-radius:50%; animation:blink 1.4s infinite; margin-right:6px;
  }}
  @keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:.2}} }}

  /* ── Streamlit widgets ──────────────────────────────── */
  [data-testid="metric-container"] {{
    background:{c["surface_raised"]} !important; border:1px solid {c["border"]};
    border-radius:12px; padding:12px;
  }}
  [data-testid="stMetricValue"] {{ color:{c["primary"]} !important; }}
  [data-testid="stMetricLabel"] {{ color:{c["muted"]} !important; }}

  .stDataFrame, [data-testid="stDataFrame"] {{
    background:{c["surface"]} !important;
  }}
  div[data-testid="stExpander"] {{
    background:{c["surface"]}; border:1px solid {c["border"]}; border-radius:10px;
  }}

  .stSelectbox > div > div, .stTextInput > div > div > input,
  .stTextArea > div > div > textarea {{
    background:{c["surface_raised"]} !important;
    color:{c["text"]} !important;
    border-color:{c["border"]} !important;
  }}

  .stButton > button {{
    background:{c["surface_raised"]}; color:{c["text"]};
    border:1px solid {c["border"]}; border-radius:8px;
  }}
  .stButton > button:hover {{
    border-color:{c["primary"]}; color:{c["primary"]};
  }}

  [data-testid="stSidebar"] .stRadio label {{
    font-size:14px !important; padding:6px 0 !important;
  }}

  hr {{ border-color:{c["border"]} !important; }}

  /* Plotly chart container */
  .js-plotly-plot .plotly .main-svg {{ background:transparent !important; }}
</style>
"""
