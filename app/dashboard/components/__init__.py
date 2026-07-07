"""Dashboard UI components package."""

from app.dashboard.theme import severity_badge, severity_color
from app.dashboard.components.cards import alert_banner, confidence_gauge, kpi_row, module_status_card
from app.dashboard.components.charts import (
    cases_per_hour,
    confidence_trend,
    hr_trend,
    module_confidence_bar,
    risk_gauge,
    severity_pie,
    severity_timeline,
    spo2_gauge,
    timeline_chart,
    xray_findings_bar,
)
from app.dashboard.components.filters import patient_filters, patient_selector
from app.dashboard.components.layout import page_header, section, sidebar_brand, sidebar_status
from app.dashboard.components.reports import download_buttons, fusion_report_text
from app.dashboard.components.waveforms import ecg_chart, eeg_multi_band, ppg_chart, thermal_heatmap, waveform_chart

__all__ = [
    "severity_badge", "severity_color",
    "alert_banner", "confidence_gauge", "kpi_row", "module_status_card",
    "cases_per_hour", "confidence_trend", "hr_trend", "module_confidence_bar",
    "risk_gauge", "severity_pie", "severity_timeline", "spo2_gauge",
    "timeline_chart", "xray_findings_bar",
    "patient_filters", "patient_selector",
    "page_header", "section", "sidebar_brand", "sidebar_status",
    "download_buttons", "fusion_report_text",
    "ecg_chart", "eeg_multi_band", "ppg_chart", "thermal_heatmap", "waveform_chart",
]
