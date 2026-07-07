"""
STEP 9 — Clinical rules for thermal diagnostics.

Maps thermal findings to risk levels and clinical flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# Temperature proxy thresholds on normalized 0-255 intensity scale
TEMP_NORMAL_MAX = 0.55       # relative heat index
TEMP_MILD_FEVER = 0.65
TEMP_HIGH_FEVER = 0.75


@dataclass(frozen=True)
class ThermalThresholds:
    normal_max: float = TEMP_NORMAL_MAX
    mild_fever: float = TEMP_MILD_FEVER
    high_fever: float = TEMP_HIGH_FEVER


DEFAULT_THRESHOLDS = ThermalThresholds()

CLASS_LABELS = ["normal", "inflammation", "fever", "severe"]
CLASS_TO_STATUS = {
    0: "NORMAL",
    1: "MODERATE",
    2: "HIGH",
    3: "CRITICAL",
}
CLASS_TO_RISK = {
    0: "LOW",
    1: "MODERATE",
    2: "HIGH",
    3: "CRITICAL",
}


def heat_index_to_risk(heat_index: float, thresholds: ThermalThresholds = DEFAULT_THRESHOLDS) -> str:
    """
    Map mean thermal intensity to risk level.

    Normal Temperature Distribution      -> LOW
    Localized Inflammation (moderate)    -> MODERATE
    High Body Temperature                -> HIGH
    Severe Thermal Abnormalities         -> CRITICAL
    """
    if heat_index < thresholds.normal_max:
        return "LOW"
    if heat_index < thresholds.mild_fever:
        return "MODERATE"
    if heat_index < thresholds.high_fever:
        return "HIGH"
    return "CRITICAL"


def heat_index_to_status(heat_index: float, thresholds: ThermalThresholds = DEFAULT_THRESHOLDS) -> str:
    risk = heat_index_to_risk(heat_index, thresholds)
    return {"LOW": "NORMAL", "MODERATE": "MODERATE", "HIGH": "HIGH", "CRITICAL": "CRITICAL"}[risk]


def class_index_to_status(class_idx: int, num_classes: int = 4) -> str:
    if num_classes == 2:
        return "NORMAL" if class_idx == 0 else "MODERATE"
    return CLASS_TO_STATUS.get(class_idx, "CRITICAL")


def class_index_to_risk(class_idx: int, num_classes: int = 4) -> str:
    if num_classes == 2:
        return "LOW" if class_idx == 0 else "MODERATE"
    return CLASS_TO_RISK.get(class_idx, "CRITICAL")


def detect_fever(heat_index: float, max_temp: float, thresholds: ThermalThresholds = DEFAULT_THRESHOLDS) -> str:
    """Return YES/NO fever flag."""
    if heat_index >= thresholds.mild_fever or max_temp >= thresholds.high_fever:
        return "YES"
    return "NO"


def detect_inflammation(hotspot_count: int, heat_index: float) -> str:
    """Return YES/NO localized inflammation flag."""
    if hotspot_count >= 2 and heat_index >= 0.45:
        return "YES"
    return "NO"


def clinical_summary(
    heat_index: float,
    max_temp: float,
    hotspot_count: int,
    class_idx: int | None = None,
    num_classes: int = 4,
) -> Dict[str, str]:
    """Full clinical interpretation dict."""
    status = class_index_to_status(class_idx, num_classes) if class_idx is not None else heat_index_to_status(heat_index)
    risk = class_index_to_risk(class_idx, num_classes) if class_idx is not None else heat_index_to_risk(heat_index)
    return {
        "status": status,
        "thermal_risk": risk,
        "fever": detect_fever(heat_index, max_temp),
        "inflammation": detect_inflammation(hotspot_count, heat_index),
        "severity": _severity_text(risk),
    }


def _severity_text(risk: str) -> str:
    mapping = {
        "LOW": "Normal Temperature Distribution",
        "MODERATE": "Localized Inflammation / Mild Fever",
        "HIGH": "High Body Temperature",
        "CRITICAL": "Severe Thermal Abnormality",
    }
    return mapping.get(risk, "Unknown")
