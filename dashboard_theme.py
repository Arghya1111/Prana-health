"""
PRĀNA Clinical Dashboard — theme, colors, and ICU-style CSS.

Hospital-grade dark theme inspired by Philips IntelliVue / GE CARESCAPE.
"""

from __future__ import annotations

from typing import Dict

# ── Design tokens ─────────────────────────────────────────────────────────────
COLORS: Dict[str, str] = {
    "primary": "#3B82F6",
    "secondary": "#10B981",
    "critical": "#EF4444",
    "high": "#F97316",
    "moderate": "#FACC15",
    "low": "#60A5FA",
    "normal": "#10B981",
    "bg": "#0B0F19",
    "bg_panel": "#111827",
    "surface": "#1F2937",
    "surface_glass": "rgba(31, 41, 55, 0.72)",
    "border": "#374151",
    "muted": "#9CA3AF",
    "text": "#F9FAFB",
    "text_dim": "#D1D5DB",
}

SEVERITY_COLORS: Dict[str, str] = {
    "CRITICAL": COLORS["critical"],
    "HIGH": COLORS["high"],
    "MODERATE": COLORS["moderate"],
    "LOW": COLORS["low"],
    "NORMAL": COLORS["normal"],
}

PLOTLY_LAYOUT: Dict = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(17,24,39,0.6)",
    "font": {"family": "Inter, Segoe UI, system-ui, sans-serif", "color": COLORS["text"], "size": 11},
    "margin": {"l": 36, "r": 16, "t": 36, "b": 36},
}


def severity_color(severity: str) -> str:
    return SEVERITY_COLORS.get(str(severity).upper(), COLORS["muted"])


def severity_badge(severity: str) -> str:
    sev = str(severity).upper()
    cls = {
        "CRITICAL": "prana-badge-critical",
        "HIGH": "prana-badge-high",
        "MODERATE": "prana-badge-moderate",
        "LOW": "prana-badge-low",
        "NORMAL": "prana-badge-normal",
    }.get(sev, "prana-badge-normal")
    return f'<span class="{cls}">{sev}</span>'


def status_dot(status: str) -> str:
    """Green / yellow / red status indicator."""
    color = {"online": COLORS["secondary"], "degraded": COLORS["moderate"], "offline": COLORS["critical"]}.get(
        status, COLORS["muted"]
    )
    return f'<span class="prana-status-dot" style="background:{color}"></span>'


def inject_css() -> str:
    c = COLORS
    return f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  .stApp, [data-testid="stAppViewContainer"], .main, .block-container {{
    background: radial-gradient(ellipse at top, #111827 0%, {c["bg"]} 55%) !important;
    color: {c["text"]};
    font-family: 'Inter', system-ui, sans-serif;
  }}
  .block-container {{ padding-top: 1rem; max-width: 100%; }}

  #MainMenu, footer, header {{ visibility: hidden; }}
  [data-testid="stStatusWidget"] {{ display: none; }}

  /* ── Top navigation bar ─────────────────────────────── */
  .prana-topnav {{
    display: flex; align-items: center; justify-content: space-between;
    background: {c["surface_glass"]}; backdrop-filter: blur(14px);
    border: 1px solid {c["border"]}; border-radius: 16px;
    padding: 14px 24px; margin-bottom: 18px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
  }}
  .prana-logo {{
    font-size: 26px; font-weight: 800; letter-spacing: 4px;
    color: {c["primary"]}; text-shadow: 0 0 24px {c["primary"]}44;
  }}
  .prana-tagline {{ font-size: 11px; color: {c["muted"]}; letter-spacing: 1px; margin-top: 2px; }}
  .prana-nav-meta {{ font-size: 12px; color: {c["text_dim"]}; text-align: right; }}
  .prana-nav-status {{ display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }}
  .prana-nav-pill {{
    background: {c["bg_panel"]}; border: 1px solid {c["border"]};
    border-radius: 999px; padding: 6px 14px; font-size: 11px; font-weight: 600;
  }}

  /* ── Glass cards ────────────────────────────────────── */
  .prana-card {{
    background: {c["surface_glass"]}; backdrop-filter: blur(12px);
    border: 1px solid {c["border"]}; border-radius: 16px;
    padding: 16px 18px; margin-bottom: 12px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
  }}
  .prana-card:hover {{
    transform: translateY(-2px);
    border-color: {c["primary"]}66;
    box-shadow: 0 8px 32px rgba(59,130,246,0.12);
  }}
  .prana-card-title {{
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.8px; color: {c["primary"]}; margin-bottom: 10px;
  }}
  .prana-kpi-value {{ font-size: 26px; font-weight: 700; color: {c["text"]}; line-height: 1.1; }}
  .prana-kpi-label {{ font-size: 10px; color: {c["muted"]}; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 4px; }}

  /* ── Severity badges ────────────────────────────────── */
  .prana-badge-critical, .prana-badge-high, .prana-badge-moderate,
  .prana-badge-low, .prana-badge-normal {{
    border-radius: 999px; padding: 3px 10px; font-size: 10px; font-weight: 700;
    display: inline-block; letter-spacing: 0.4px;
  }}
  .prana-badge-critical {{ background: {c["critical"]}22; color: {c["critical"]}; border: 1px solid {c["critical"]}55; }}
  .prana-badge-high     {{ background: {c["high"]}22; color: {c["high"]}; border: 1px solid {c["high"]}55; }}
  .prana-badge-moderate {{ background: {c["moderate"]}22; color: {c["moderate"]}; border: 1px solid {c["moderate"]}55; }}
  .prana-badge-low      {{ background: {c["low"]}22; color: {c["low"]}; border: 1px solid {c["low"]}55; }}
  .prana-badge-normal   {{ background: {c["normal"]}22; color: {c["normal"]}; border: 1px solid {c["normal"]}55; }}

  /* ── Alerts (blinking) ──────────────────────────────── */
  @keyframes prana-alert-pulse {{
    0%, 100% {{ opacity: 1; box-shadow: 0 0 0 0 {c["critical"]}66; }}
    50% {{ opacity: 0.85; box-shadow: 0 0 16px 4px {c["critical"]}44; }}
  }}
  .prana-alert {{
    background: {c["critical"]}18; border: 1px solid {c["critical"]}66;
    border-radius: 12px; padding: 10px 14px; margin-bottom: 8px;
    animation: prana-alert-pulse 1.8s infinite;
    font-size: 12px; font-weight: 600; color: {c["text"]};
  }}

  .prana-status-dot {{
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 6px; box-shadow: 0 0 8px currentColor;
  }}
  .prana-live-dot {{
    display: inline-block; width: 8px; height: 8px; background: {c["secondary"]};
    border-radius: 50%; animation: prana-blink 1.4s infinite; margin-right: 6px;
  }}
  @keyframes prana-blink {{ 0%,100%{{opacity:1}} 50%{{opacity:0.25}} }}

  /* ── Streamlit overrides ────────────────────────────── */
  [data-testid="stMetricValue"] {{ color: {c["primary"]} !important; }}
  [data-testid="stMetricLabel"] {{ color: {c["muted"]} !important; }}
  .stDataFrame {{ background: transparent !important; }}
  div[data-testid="stExpander"] {{
    background: {c["surface"]}; border: 1px solid {c["border"]}; border-radius: 12px;
  }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
  .stTabs [data-baseweb="tab"] {{
    background: {c["surface"]}; border-radius: 8px; color: {c["text_dim"]};
  }}
  hr {{ border-color: {c["border"]} !important; }}
</style>
"""
