"""
STEP 9 — Clinical rules for SpO2 interpretation.

Reusable thresholds for oxygen saturation severity and recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

# Configurable clinical thresholds (%)
THRESHOLD_NORMAL = 95.0
THRESHOLD_MODERATE = 90.0
THRESHOLD_HIGH = 85.0


@dataclass(frozen=True)
class SpO2Thresholds:
    """Configurable SpO2 classification thresholds."""

    normal: float = THRESHOLD_NORMAL       # >= normal -> NORMAL
    moderate: float = THRESHOLD_MODERATE   # >= moderate -> MODERATE band
    high: float = THRESHOLD_HIGH           # >= high -> HIGH band; below -> CRITICAL


DEFAULT_THRESHOLDS = SpO2Thresholds()


def spo2_to_status(spo2: float, thresholds: SpO2Thresholds = DEFAULT_THRESHOLDS) -> str:
    """
    Map SpO2 value to clinical status label.

    >= 95%  -> NORMAL
    90–94%  -> MODERATE
    85–89%  -> HIGH
    < 85%   -> CRITICAL
    """
    if spo2 >= thresholds.normal:
        return "NORMAL"
    if spo2 >= thresholds.moderate:
        return "MODERATE"
    if spo2 >= thresholds.high:
        return "HIGH"
    return "CRITICAL"


def spo2_to_severity(spo2: float, thresholds: SpO2Thresholds = DEFAULT_THRESHOLDS) -> str:
    """
    Map SpO2 to severity level for fusion engine integration.

    >= 95%  -> LOW
    90–94%  -> MODERATE
    85–89%  -> HIGH
    < 85%   -> CRITICAL
    """
    if spo2 >= thresholds.normal:
        return "LOW"
    if spo2 >= thresholds.moderate:
        return "MODERATE"
    if spo2 >= thresholds.high:
        return "HIGH"
    return "CRITICAL"


def get_recommendation(spo2: float, thresholds: SpO2Thresholds = DEFAULT_THRESHOLDS) -> str:
    """Return clinical recommendation text for a SpO2 reading."""
    if spo2 >= thresholds.normal:
        return "Normal Oxygen Saturation"
    if spo2 >= thresholds.moderate:
        return "Monitor oxygen saturation closely; consider supplemental oxygen if symptomatic"
    if spo2 >= thresholds.high:
        return "Administer supplemental oxygen and reassess respiratory status"
    return "Immediate medical intervention required — severe hypoxia detected"


def spo2_to_severity_description(
    spo2: float,
    thresholds: SpO2Thresholds = DEFAULT_THRESHOLDS,
) -> str:
    """Return human-readable hypoxia severity description."""
    if spo2 >= thresholds.normal:
        return "Normal Oxygen Saturation"
    if spo2 >= thresholds.moderate:
        return "Mild Hypoxia"
    if spo2 >= thresholds.high:
        return "Moderate Hypoxia"
    return "Severe Hypoxia"


def spo2_to_respiratory_risk(
    spo2: float,
    thresholds: SpO2Thresholds = DEFAULT_THRESHOLDS,
) -> str:
    """Alias for spo2_to_severity — respiratory risk level."""
    return spo2_to_severity(spo2, thresholds)


def is_hypoxia(spo2: float, thresholds: SpO2Thresholds = DEFAULT_THRESHOLDS) -> bool:
    """True when SpO2 indicates hypoxia (below normal threshold)."""
    return spo2 < thresholds.normal


def status_to_class_index(status: str) -> int:
    """Map status string to multi-class index."""
    mapping = {"NORMAL": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
    return mapping.get(status.upper(), 3)


def class_index_to_status(class_idx: int, num_classes: int = 4) -> str:
    """Map model class index to status string."""
    if num_classes == 2:
        return "NORMAL" if class_idx == 0 else "ABNORMAL"
    labels = ["NORMAL", "MODERATE", "HIGH", "CRITICAL"]
    if 0 <= class_idx < len(labels):
        return labels[class_idx]
    return "CRITICAL"


def clinical_summary(spo2: float, thresholds: SpO2Thresholds = DEFAULT_THRESHOLDS) -> Dict[str, str]:
    """Return all clinical interpretations for one SpO2 reading."""
    return {
        "status": spo2_to_status(spo2, thresholds),
        "severity": spo2_to_severity(spo2, thresholds),
        "recommendation": get_recommendation(spo2, thresholds),
        "respiratory_risk": spo2_to_respiratory_risk(spo2, thresholds),
        "hypoxia": str(is_hypoxia(spo2, thresholds)),
    }


def spo2_range_for_class(class_idx: int, num_classes: int = 4) -> Tuple[float, float]:
    """Return approximate SpO2 range for a class label (for synthetic data)."""
    if num_classes == 2:
        return (95.0, 100.0) if class_idx == 0 else (80.0, 94.9)
    ranges = [(95.0, 100.0), (90.0, 94.9), (85.0, 89.9), (70.0, 84.9)]
    return ranges[min(class_idx, 3)]
