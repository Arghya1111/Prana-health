"""
Shared types, enums, and module-output standardization for the PRĀNA Fusion Engine.

Converts heterogeneous predictions from ECG, EEG, PPG, SpO₂, Thermal, Optical,
and X-Ray into a unified ``ModuleResult`` structure without modifying source modules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Union


class ModuleName(str, Enum):
    """Supported diagnostic modality identifiers."""

    ECG = "ECG"
    EEG = "EEG"
    PPG = "PPG"
    SPO2 = "SpO2"
    THERMAL = "THERMAL"
    OPTICAL = "OPTICAL"
    XRAY = "XRAY"


class SeverityLevel(str, Enum):
    """Unified clinical severity tiers used across the fusion engine."""

    NORMAL = "NORMAL"
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# Numeric contribution of each severity tier to the 0–100 risk score.
SEVERITY_RISK_SCORES: Dict[str, float] = {
    SeverityLevel.NORMAL.value: 0.0,
    SeverityLevel.LOW.value: 25.0,
    SeverityLevel.MODERATE.value: 50.0,
    SeverityLevel.HIGH.value: 80.0,
    SeverityLevel.CRITICAL.value: 100.0,
}

# Risk-score bands mapped to overall severity (Step 5).
RISK_SEVERITY_BANDS: List[tuple[int, int, str]] = [
    (0, 20, SeverityLevel.NORMAL.value),
    (21, 40, SeverityLevel.LOW.value),
    (41, 60, SeverityLevel.MODERATE.value),
    (61, 80, SeverityLevel.HIGH.value),
    (81, 100, SeverityLevel.CRITICAL.value),
]

# Default module weights for risk and confidence fusion (Step 4).
DEFAULT_MODULE_WEIGHTS: Dict[str, float] = {
    ModuleName.ECG.value: 20.0,
    ModuleName.EEG.value: 15.0,
    ModuleName.PPG.value: 10.0,
    ModuleName.SPO2.value: 20.0,
    ModuleName.THERMAL.value: 10.0,
    ModuleName.OPTICAL.value: 10.0,
    ModuleName.XRAY.value: 15.0,
}


@dataclass
class ModuleResult:
    """
    Standardized output for a single AI module.

    Every module must ultimately conform to this structure before fusion.
    """

    module: str
    status: str
    confidence: float
    severity: str
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return asdict(self)

    @property
    def is_abnormal(self) -> bool:
        """Return True when the module indicates a clinically significant finding."""
        normal_statuses = {
            "NORMAL",
            "NO_FINDING",
            "CLEAR",
        }
        return self.status.upper() not in normal_statuses and self.severity != SeverityLevel.NORMAL.value

    @property
    def severity_level(self) -> SeverityLevel:
        """Return the severity as an enum, defaulting to NORMAL for unknown values."""
        try:
            return SeverityLevel(self.severity.upper())
        except ValueError:
            return SeverityLevel.NORMAL


@dataclass(frozen=True)
class FusionConfig:
    """Configurable parameters for the fusion engine."""

    module_weights: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_MODULE_WEIGHTS))
    risk_bands: tuple[tuple[int, int, str], ...] = tuple(RISK_SEVERITY_BANDS)
    spo2_hypoxia_threshold: float = 85.0
    xray_alert_threshold: float = 0.3
    xray_critical_threshold: float = 0.5


class FusionError(Exception):
    """Base exception for fusion engine errors."""


class StandardizationError(FusionError):
    """Raised when a module output cannot be standardized."""


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _clamp_confidence(value: Optional[float]) -> float:
    """Normalize confidence to a 0–100 float."""
    if value is None:
        return 0.0
    try:
        return round(max(0.0, min(100.0, float(value))), 1)
    except (TypeError, ValueError) as exc:
        raise StandardizationError(f"Invalid confidence value: {value!r}") from exc


def _normalize_severity(raw: Optional[str], fallback: str = SeverityLevel.NORMAL.value) -> str:
    """Map assorted severity labels to the unified five-tier system."""
    if not raw:
        return fallback

    key = str(raw).strip().upper()

    direct_map = {
        "NORMAL": SeverityLevel.NORMAL.value,
        "LOW": SeverityLevel.LOW.value,
        "MODERATE": SeverityLevel.MODERATE.value,
        "HIGH": SeverityLevel.HIGH.value,
        "CRITICAL": SeverityLevel.CRITICAL.value,
        # Module-specific aliases
        "ANOMALOUS": SeverityLevel.HIGH.value,
        "ABNORMAL": SeverityLevel.HIGH.value,
        "ATTENTION": SeverityLevel.MODERATE.value,
        "FEVER": SeverityLevel.MODERATE.value,
        "INFLAMMATION": SeverityLevel.MODERATE.value,
        "HYPOXIA": SeverityLevel.HIGH.value,
        "PNEUMONIA": SeverityLevel.HIGH.value,
        "SEIZURE": SeverityLevel.CRITICAL.value,
    }
    if key in direct_map:
        return direct_map[key]

    # Human-readable thermal severity strings
    lowered = str(raw).lower()
    if "severe" in lowered or "critical" in lowered:
        return SeverityLevel.CRITICAL.value
    if "high" in lowered:
        return SeverityLevel.HIGH.value
    if "mild" in lowered or "moderate" in lowered or "inflammation" in lowered:
        return SeverityLevel.MODERATE.value
    if "normal" in lowered:
        return SeverityLevel.NORMAL.value

    return fallback


def _extract_features(raw: Dict[str, Any], exclude: Optional[set[str]] = None) -> Dict[str, Any]:
    """Copy module-specific fields into the standardized features dict."""
    skip = exclude or set()
    reserved = {
        "module",
        "status",
        "confidence",
        "severity",
        "disease",
        "spo2",
        "features",
    } | skip
    features: Dict[str, Any] = {}

    nested = raw.get("features")
    if isinstance(nested, dict):
        features.update(nested)

    for key, value in raw.items():
        if key not in reserved and not key.startswith("_"):
            features[key] = value

    return features


def standardize_ecg(raw: Dict[str, Any]) -> ModuleResult:
    """Standardize ECG module output."""
    status = str(raw.get("status", "NORMAL")).upper()
    severity = _normalize_severity(
        raw.get("severity"),
        SeverityLevel.HIGH.value if status == "ANOMALOUS" else SeverityLevel.NORMAL.value,
    )
    return ModuleResult(
        module=ModuleName.ECG.value,
        status=status,
        confidence=_clamp_confidence(raw.get("confidence")),
        severity=severity,
        features=_extract_features(raw),
    )


def standardize_eeg(raw: Dict[str, Any]) -> ModuleResult:
    """Standardize EEG module output."""
    status = str(raw.get("status", "NORMAL")).upper()
    severity = _normalize_severity(
        raw.get("severity"),
        SeverityLevel.HIGH.value if status == "ABNORMAL" else SeverityLevel.NORMAL.value,
    )
    return ModuleResult(
        module=ModuleName.EEG.value,
        status=status,
        confidence=_clamp_confidence(raw.get("confidence")),
        severity=severity,
        features=_extract_features(raw),
    )


def standardize_ppg(raw: Dict[str, Any]) -> ModuleResult:
    """Standardize PPG module output."""
    status = str(raw.get("status", "NORMAL")).upper()
    default_severity = SeverityLevel.MODERATE.value if status == "ABNORMAL" else SeverityLevel.NORMAL.value
    severity = _normalize_severity(raw.get("severity"), default_severity)

    # Poor pulse quality elevates severity to at least MODERATE.
    pulse_quality = str(raw.get("pulse_quality", "")).upper()
    if pulse_quality == "POOR" and severity == SeverityLevel.NORMAL.value:
        severity = SeverityLevel.MODERATE.value

    return ModuleResult(
        module=ModuleName.PPG.value,
        status=status,
        confidence=_clamp_confidence(raw.get("confidence")),
        severity=severity,
        features=_extract_features(raw),
    )


def standardize_spo2(raw: Dict[str, Any]) -> ModuleResult:
    """Standardize SpO₂ module output."""
    status = str(raw.get("status", "NORMAL")).upper()

    # Map SpO₂-specific status vocabulary to unified severity.
    status_to_severity = {
        "NORMAL": SeverityLevel.NORMAL.value,
        "MODERATE": SeverityLevel.MODERATE.value,
        "HIGH": SeverityLevel.HIGH.value,
        "CRITICAL": SeverityLevel.CRITICAL.value,
        "LOW": SeverityLevel.HIGH.value,
        "HYPOXIA": SeverityLevel.HIGH.value,
    }
    severity = _normalize_severity(
        raw.get("severity"),
        status_to_severity.get(status, SeverityLevel.NORMAL.value),
    )

    features = _extract_features(raw)
    if "spo2" in raw:
        features["spo2"] = raw["spo2"]

    return ModuleResult(
        module=ModuleName.SPO2.value,
        status=status,
        confidence=_clamp_confidence(raw.get("confidence")),
        severity=severity,
        features=features,
    )


def standardize_thermal(raw: Dict[str, Any]) -> ModuleResult:
    """Standardize Thermal module output."""
    status = str(raw.get("status", "NORMAL")).upper()

    # Prefer explicit severity, then thermal_risk, then status.
    severity_source = raw.get("severity") or raw.get("thermal_risk") or status
    severity = _normalize_severity(severity_source, SeverityLevel.NORMAL.value)

    if status == "FEVER" and severity == SeverityLevel.NORMAL.value:
        severity = SeverityLevel.MODERATE.value

    return ModuleResult(
        module=ModuleName.THERMAL.value,
        status=status,
        confidence=_clamp_confidence(raw.get("confidence")),
        severity=severity,
        features=_extract_features(raw),
    )


def _max_severity(a: str, b: str) -> str:
    """Return the more severe of two severity labels."""
    order = {level.value: idx for idx, level in enumerate(SeverityLevel)}
    return a if order.get(a, 0) >= order.get(b, 0) else b


def standardize_optical(raw: Dict[str, Any]) -> ModuleResult:
    """Standardize Optical module output."""
    disease = str(raw.get("disease") or raw.get("status", "NORMAL")).upper()
    status = disease
    default = (
        SeverityLevel.NORMAL.value
        if disease in ("NORMAL", "NO_FINDING")
        else SeverityLevel.MODERATE.value
    )
    severity = _normalize_severity(raw.get("severity"), default)
    if disease not in ("NORMAL", "NO_FINDING"):
        severity = _max_severity(severity, SeverityLevel.MODERATE.value)

    return ModuleResult(
        module=ModuleName.OPTICAL.value,
        status=status,
        confidence=_clamp_confidence(raw.get("confidence")),
        severity=severity,
        features=_extract_features(raw, exclude={"disease"}),
    )


def standardize_xray(raw: Dict[str, Any]) -> ModuleResult:
    """Standardize X-Ray module output."""
    status = str(raw.get("status", "NORMAL")).upper()

    # Derive status from top pathology when a condition name is provided.
    results = raw.get("results") or {}
    if results and status in ("NORMAL", "ATTENTION", "CRITICAL"):
        top_condition, top_score = max(results.items(), key=lambda item: item[1])
        if top_score >= raw.get("critical_threshold", 0.5):
            status = top_condition.upper().replace(" ", "_")
        elif top_score >= raw.get("alert_threshold", 0.3):
            status = top_condition.upper().replace(" ", "_")

    status_to_severity = {
        "NORMAL": SeverityLevel.NORMAL.value,
        "ATTENTION": SeverityLevel.MODERATE.value,
        "CRITICAL": SeverityLevel.CRITICAL.value,
        "PNEUMONIA": SeverityLevel.HIGH.value,
    }
    severity = _normalize_severity(
        raw.get("severity"),
        status_to_severity.get(status, SeverityLevel.HIGH.value if status != "NORMAL" else SeverityLevel.NORMAL.value),
    )

    features = _extract_features(raw)
    if results:
        features["results"] = results
    if "flagged" in raw:
        features["flagged"] = raw["flagged"]
    if "critical" in raw:
        features["critical"] = raw["critical"]

    confidence = raw.get("confidence")
    if confidence is None and results:
        confidence = max(results.values()) * 100.0

    return ModuleResult(
        module=ModuleName.XRAY.value,
        status=status,
        confidence=_clamp_confidence(confidence),
        severity=severity,
        features=features,
    )


_STANDARDIZERS = {
    ModuleName.ECG.value: standardize_ecg,
    ModuleName.EEG.value: standardize_eeg,
    ModuleName.PPG.value: standardize_ppg,
    ModuleName.SPO2.value: standardize_spo2,
    ModuleName.SPO2.value.upper(): standardize_spo2,
    ModuleName.THERMAL.value: standardize_thermal,
    ModuleName.OPTICAL.value: standardize_optical,
    ModuleName.XRAY.value: standardize_xray,
}


def standardize_module_output(
    raw: Dict[str, Any],
    module: Optional[str] = None,
) -> ModuleResult:
    """
    Convert a raw module prediction dict into a standardized ``ModuleResult``.

    Parameters
    ----------
    raw:
        Raw prediction dictionary from any PRĀNA module.
    module:
        Module name override. When omitted, ``raw['module']`` is used.

    Raises
    ------
    StandardizationError
        If the module name is unknown or the payload is invalid.
    """
    if not raw:
        raise StandardizationError("Module output cannot be empty.")

    module_key = (module or raw.get("module", "")).strip().upper()
    if not module_key:
        raise StandardizationError("Module name is required for standardization.")

    standardizer = _STANDARDIZERS.get(module_key)
    if standardizer is None:
        raise StandardizationError(f"Unsupported module: {module_key}")

    return standardizer(raw)


def standardize_batch(
    outputs: Mapping[str, Union[Dict[str, Any], ModuleResult, None]],
) -> Dict[str, ModuleResult]:
    """
    Standardize a mapping of module name → raw output.

    ``None`` values are skipped (module not available).
    Already-standardized ``ModuleResult`` instances are passed through.
    """
    standardized: Dict[str, ModuleResult] = {}
    for name, payload in outputs.items():
        if payload is None:
            continue
        if isinstance(payload, ModuleResult):
            standardized[name.upper()] = payload
        else:
            standardized[name.upper()] = standardize_module_output(payload, module=name)
    return standardized
