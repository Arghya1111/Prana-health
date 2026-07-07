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
    "margin": {"l": 36, "r": 12, "t": 32, "b": 28},
    "autosize": True,
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
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;
    background: {c["surface_glass"]}; backdrop-filter: blur(14px);
    border: 1px solid {c["border"]}; border-radius: 16px;
    padding: 14px 24px; margin-bottom: 18px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
  }}
  .prana-topnav-brand {{ flex: 1 1 180px; min-width: 0; }}
  .prana-topnav-meta {{ flex: 1 1 200px; min-width: 0; }}
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

  /* Responsive KPI + module grids */
  .prana-kpi-grid {{
    display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin-bottom: 10px;
  }}
  .prana-module-grid {{
    display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px;
  }}
  .prana-section-title {{
    font-size: 16px; font-weight: 700; color: {c["text"]}; margin: 16px 0 4px;
  }}
  .prana-section-sub {{
    font-size: 11px; color: {c["muted"]}; margin-bottom: 12px; line-height: 1.4;
  }}
  .prana-metric-row {{
    display: flex; justify-content: space-between; align-items: flex-start; gap: 8px;
    padding: 6px 0; font-size: 12px; border-bottom: 1px solid {c["border"]}33;
  }}
  .prana-metric-label {{ color: {c["muted"]}; flex-shrink: 0; }}
  .prana-metric-value {{ font-weight: 600; text-align: right; word-break: break-word; }}

  /* Sliding nav — target Streamlit block that contains marker */
  div[data-testid="stVerticalBlock"]:has(.prana-slide-nav-marker) {{
    border: 1px solid {c["border"]};
    border-radius: 14px;
    background: {c["surface_glass"]};
    backdrop-filter: blur(12px);
    padding: 10px 12px 12px !important;
    margin-bottom: 14px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
  }}
  div[data-testid="stVerticalBlock"]:has(.prana-slide-nav-marker.is-collapsed-block) {{
    padding-bottom: 10px !important;
  }}
  .prana-slide-nav-marker {{ display: none; }}
  .prana-slide-nav-wrap {{ display: none; }}
  .prana-slide-nav-hint {{
    font-size: 11px; color: {c["muted"]}; padding-top: 8px; line-height: 1.4;
  }}
  .prana-slide-nav-hint strong {{ color: {c["primary"]}; }}
  .prana-slide-nav-panel {{
    overflow: hidden;
    max-height: 220px;
    opacity: 1;
    transition: max-height 0.4s cubic-bezier(0.4, 0, 0.2, 1), opacity 0.3s ease;
  }}
  div[data-testid="stVerticalBlock"]:has(.prana-slide-nav-marker) .stButton > button {{
    border-radius: 999px !important;
    border: 1px solid {c["primary"]}66 !important;
    background: {c["bg_panel"]} !important;
    color: {c["primary"]} !important;
    font-weight: 600 !important;
    min-height: 40px;
    transition: all 0.25s ease !important;
  }}
  div[data-testid="stVerticalBlock"]:has(.prana-slide-nav-marker) .stButton > button:hover {{
    background: {c["primary"]}22 !important;
    border-color: {c["primary"]} !important;
  }}
  div[data-testid="stVerticalBlock"]:has(.prana-slide-nav-marker) .stRadio > div {{
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    gap: 8px !important;
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch;
    padding: 4px 2px 8px !important;
    scrollbar-width: thin;
  }}
  div[data-testid="stVerticalBlock"]:has(.prana-slide-nav-marker) .stRadio label {{
    background: {c["bg_panel"]} !important;
    border: 1px solid {c["border"]} !important;
    border-radius: 999px !important;
    padding: 10px 18px !important;
    margin: 0 !important;
    white-space: nowrap !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: {c["text_dim"]} !important;
    transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
  }}
  div[data-testid="stVerticalBlock"]:has(.prana-slide-nav-marker) .stRadio label:hover {{
    border-color: {c["primary"]}88 !important;
    color: {c["text"]} !important;
    transform: translateY(-1px);
  }}
  div[data-testid="stVerticalBlock"]:has(.prana-slide-nav-marker) .stRadio label:has(input:checked) {{
    background: linear-gradient(135deg, {c["primary"]}, #2563EB) !important;
    border-color: {c["primary"]} !important;
    color: #fff !important;
    box-shadow: 0 4px 16px {c["primary"]}55 !important;
    transform: translateY(-1px);
  }}
  div[data-testid="stVerticalBlock"]:has(.prana-slide-nav-marker) .stRadio label > div:first-child {{
    display: none !important;
  }}
  div[data-testid="stVerticalBlock"]:has(.prana-slide-nav-marker.is-nav-collapsed) .prana-slide-nav-panel,
  div[data-testid="stVerticalBlock"]:has(.prana-slide-nav-marker.is-nav-collapsed) [data-testid="stHorizontalBlock"]:has(.stRadio),
  div[data-testid="stVerticalBlock"]:has(.prana-slide-nav-marker.is-nav-collapsed) .prana-slide-nav-actions-label,
  div[data-testid="stVerticalBlock"]:has(.prana-slide-nav-marker.is-nav-collapsed) [data-testid="stHorizontalBlock"]:has(.stToggle) {{
    display: none !important;
  }}
  .prana-slide-nav-actions-label {{
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.6px;
    color: {c["muted"]}; margin: 4px 0 6px;
  }}
  .prana-slide-health {{
    font-size: 12px; font-weight: 600; color: {c["text"]};
    padding: 10px 8px; text-align: center;
    background: {c["bg_panel"]}; border-radius: 10px;
    border: 1px solid {c["border"]};
    min-height: 40px; display: flex; align-items: center; justify-content: center;
  }}

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

  /* ── Tablet (≤1024px) ───────────────────────────────── */
  @media (max-width: 1024px) {{
    .prana-kpi-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .prana-module-grid {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
    .prana-kpi-value {{ font-size: 22px; }}
    .block-container {{ padding-left: 1rem !important; padding-right: 1rem !important; }}
  }}

  /* ── Mobile (≤768px) ────────────────────────────────── */
  @media (max-width: 768px) {{
    .prana-slide-nav-panel .stRadio label {{
      padding: 10px 14px !important; font-size: 12px !important;
    }}
    .prana-slide-nav-panel {{ max-height: 260px; }}

    .prana-topnav {{
      flex-direction: column; align-items: stretch; padding: 12px 14px; border-radius: 12px;
    }}
    .prana-topnav-meta {{ text-align: left !important; }}
    .prana-nav-status {{ justify-content: flex-start; gap: 8px; }}
    .prana-nav-pill {{ font-size: 10px; padding: 5px 10px; }}
    .prana-logo {{ font-size: 22px; letter-spacing: 3px; }}
    .prana-tagline {{ font-size: 10px; }}

    .prana-kpi-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
    .prana-module-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .prana-kpi-value {{ font-size: 20px; }}
    .prana-card {{ padding: 12px 14px; border-radius: 12px; }}
    .prana-card:hover {{ transform: none; }}
    .prana-section-title {{ font-size: 15px; margin-top: 12px; }}

    /* Stack Streamlit columns vertically */
    div[data-testid="stHorizontalBlock"] {{
      flex-direction: column !important; flex-wrap: wrap !important; gap: 0.75rem !important;
    }}
    div[data-testid="column"] {{
      width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important;
    }}

    /* Touch-friendly controls */
    .stButton > button, .stDownloadButton > button {{
      min-height: 44px; font-size: 14px;
    }}
    [data-testid="stSidebar"] .stButton > button {{
      min-height: 44px;
    }}

    /* Scrollable tables on small screens */
    [data-testid="stDataFrame"], .dvn-scroller {{
      overflow-x: auto !important; -webkit-overflow-scrolling: touch;
    }}

    /* Charts: fit narrow viewports */
    [data-testid="stPlotlyChart"], [data-testid="stArrowVegaLiteChart"] {{
      max-width: 100%; overflow-x: auto;
    }}

    .block-container {{ padding-top: 0.5rem !important; padding-left: 0.75rem !important; padding-right: 0.75rem !important; }}
  }}

  /* ── Small phones (≤480px) ──────────────────────────── */
  @media (max-width: 480px) {{
    .prana-kpi-grid {{ grid-template-columns: 1fr 1fr; }}
    .prana-nav-status {{ flex-direction: column; align-items: stretch; }}
    .prana-nav-pill {{ text-align: center; }}
    .prana-metric-row {{ flex-direction: column; gap: 2px; }}
    .prana-metric-value {{ text-align: left; }}
  }}
</style>
"""
