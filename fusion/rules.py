"""
Clinical rule-based overrides for the PRĀNA Fusion Engine.

Rule-based logic can elevate (or in limited cases reduce) the computed severity
when specific multi-modality clinical patterns are detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from fusion.patient_profile import PatientProfile
from fusion.utils import FusionConfig, ModuleName, SeverityLevel


@dataclass
class RuleMatch:
    """Record of a triggered clinical rule."""

    rule_id: str
    description: str
    resulting_severity: str
    modules_involved: List[str] = field(default_factory=list)


@dataclass
class RuleEvaluationResult:
    """Outcome of clinical rule evaluation."""

    overridden_severity: Optional[str]
    triggered_rules: List[RuleMatch] = field(default_factory=list)
    applied: bool = False

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "overridden_severity": self.overridden_severity,
            "applied": self.applied,
            "triggered_rules": [
                {
                    "rule_id": r.rule_id,
                    "description": r.description,
                    "resulting_severity": r.resulting_severity,
                    "modules_involved": r.modules_involved,
                }
                for r in self.triggered_rules
            ],
        }


# Type alias for rule predicate functions.
RulePredicate = Callable[[PatientProfile, FusionConfig], bool]


@dataclass
class ClinicalRule:
    """A single clinical override rule."""

    rule_id: str
    description: str
    predicate: RulePredicate
    resulting_severity: str
    modules_involved: List[str] = field(default_factory=list)
    priority: int = 0  # Higher priority rules are evaluated first.


def _is_ecg_anomalous(profile: PatientProfile, _: FusionConfig) -> bool:
    ecg = profile.ecg
    return ecg is not None and ecg.status.upper() in ("ANOMALOUS", "ABNORMAL")


def _spo2_below(profile: PatientProfile, config: FusionConfig, threshold: float) -> bool:
    spo2 = profile.spo2
    if spo2 is None:
        return False
    value = spo2.features.get("spo2")
    if value is None:
        return spo2.severity in (SeverityLevel.HIGH.value, SeverityLevel.CRITICAL.value)
    try:
        return float(value) < threshold
    except (TypeError, ValueError):
        return False


def _has_pneumonia(profile: PatientProfile, _: FusionConfig) -> bool:
    xray = profile.xray
    if xray is None:
        return False
    status = xray.status.upper()
    if "PNEUMONIA" in status:
        return True
    flagged = xray.features.get("flagged") or xray.features.get("critical") or {}
    return any("pneumonia" in str(k).lower() for k in flagged)


def _has_fever(profile: PatientProfile, _: FusionConfig) -> bool:
    thermal = profile.thermal
    if thermal is None:
        return False
    if thermal.status.upper() == "FEVER":
        return True
    if str(thermal.features.get("fever", "")).upper() == "YES":
        return True
    return thermal.severity in (SeverityLevel.MODERATE.value, SeverityLevel.HIGH.value, SeverityLevel.CRITICAL.value)


def _has_seizure(profile: PatientProfile, _: FusionConfig) -> bool:
    eeg = profile.eeg
    if eeg is None:
        return False
    status = eeg.status.upper()
    if status in ("ABNORMAL", "SEIZURE"):
        return True
    return "seizure" in str(eeg.features.get("class_label", "")).lower()


def _has_hypoxia(profile: PatientProfile, config: FusionConfig) -> bool:
    spo2 = profile.spo2
    if spo2 is None:
        return False
    if spo2.severity in (SeverityLevel.HIGH.value, SeverityLevel.CRITICAL.value):
        return True
    if spo2.status.upper() in ("LOW", "CRITICAL", "HIGH", "HYPOXIA"):
        return True
    return _spo2_below(profile, config, config.spo2_hypoxia_threshold)


def _count_high_modules(profile: PatientProfile, _: FusionConfig) -> int:
    return sum(
        1
        for result in profile.iter_modules()
        if result.severity == SeverityLevel.HIGH.value
    )


def _multiple_high_modules(profile: PatientProfile, config: FusionConfig) -> bool:
    return _count_high_modules(profile, config) >= 2


def _ecg_anomalous_and_hypoxia(profile: PatientProfile, config: FusionConfig) -> bool:
    return _is_ecg_anomalous(profile, config) and _spo2_below(
        profile, config, config.spo2_hypoxia_threshold
    )


def _pneumonia_and_fever(profile: PatientProfile, config: FusionConfig) -> bool:
    return _has_pneumonia(profile, config) and _has_fever(profile, config)


def _seizure_and_hypoxia(profile: PatientProfile, config: FusionConfig) -> bool:
    return _has_seizure(profile, config) and _has_hypoxia(profile, config)


# Default clinical rules ordered by priority (highest first).
DEFAULT_CLINICAL_RULES: List[ClinicalRule] = [
    ClinicalRule(
        rule_id="RULE_CRITICAL_SEIZURE_HYPOXIA",
        description="EEG seizure/abnormality combined with SpO₂ hypoxia → CRITICAL",
        predicate=_seizure_and_hypoxia,
        resulting_severity=SeverityLevel.CRITICAL.value,
        modules_involved=[ModuleName.EEG.value, ModuleName.SPO2.value],
        priority=100,
    ),
    ClinicalRule(
        rule_id="RULE_CRITICAL_MULTIPLE_HIGH",
        description="Two or more modules at HIGH severity → CRITICAL",
        predicate=_multiple_high_modules,
        resulting_severity=SeverityLevel.CRITICAL.value,
        modules_involved=[],
        priority=90,
    ),
    ClinicalRule(
        rule_id="RULE_HIGH_ECG_HYPOXIA",
        description="ECG anomalous combined with SpO₂ below threshold → HIGH",
        predicate=_ecg_anomalous_and_hypoxia,
        resulting_severity=SeverityLevel.HIGH.value,
        modules_involved=[ModuleName.ECG.value, ModuleName.SPO2.value],
        priority=80,
    ),
    ClinicalRule(
        rule_id="RULE_HIGH_PNEUMONIA_FEVER",
        description="X-Ray pneumonia combined with thermal fever → HIGH",
        predicate=_pneumonia_and_fever,
        resulting_severity=SeverityLevel.HIGH.value,
        modules_involved=[ModuleName.XRAY.value, ModuleName.THERMAL.value],
        priority=70,
    ),
]


class ClinicalRulesEngine:
    """
    Evaluate clinical override rules against a patient profile.

    The highest-severity triggered rule wins.  Severity ordering:
    CRITICAL > HIGH > MODERATE > LOW > NORMAL.
    """

    SEVERITY_ORDER = {
        SeverityLevel.NORMAL.value: 0,
        SeverityLevel.LOW.value: 1,
        SeverityLevel.MODERATE.value: 2,
        SeverityLevel.HIGH.value: 3,
        SeverityLevel.CRITICAL.value: 4,
    }

    def __init__(
        self,
        rules: Optional[List[ClinicalRule]] = None,
        config: Optional[FusionConfig] = None,
    ) -> None:
        self.config = config or FusionConfig()
        self.rules = sorted(
            rules if rules is not None else DEFAULT_CLINICAL_RULES,
            key=lambda r: r.priority,
            reverse=True,
        )

    def evaluate(
        self,
        profile: PatientProfile,
        base_severity: str,
    ) -> RuleEvaluationResult:
        """
        Evaluate all rules and return the highest applicable override.

        Parameters
        ----------
        profile:
            Patient profile with module results.
        base_severity:
            Severity computed from the risk score before rule overrides.

        Returns
        -------
        RuleEvaluationResult
            Override severity (if any) and list of triggered rules.
        """
        triggered: List[RuleMatch] = []
        best_severity: Optional[str] = None
        best_rank = self.SEVERITY_ORDER.get(base_severity.upper(), 0)

        for rule in self.rules:
            try:
                if rule.predicate(profile, self.config):
                    triggered.append(
                        RuleMatch(
                            rule_id=rule.rule_id,
                            description=rule.description,
                            resulting_severity=rule.resulting_severity,
                            modules_involved=rule.modules_involved,
                        )
                    )
                    rank = self.SEVERITY_ORDER.get(rule.resulting_severity, 0)
                    if rank > best_rank:
                        best_rank = rank
                        best_severity = rule.resulting_severity
            except Exception:
                # Rule predicates must not break the fusion pipeline.
                continue

        if best_severity is None:
            return RuleEvaluationResult(
                overridden_severity=None,
                triggered_rules=triggered,
                applied=False,
            )

        return RuleEvaluationResult(
            overridden_severity=best_severity,
            triggered_rules=triggered,
            applied=best_severity != base_severity,
        )

    def add_rule(self, rule: ClinicalRule) -> None:
        """Register an additional clinical rule."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority, reverse=True)
