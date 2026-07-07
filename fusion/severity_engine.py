"""
Overall severity assessment for the PRĀNA Fusion Engine.

Maps the 0–100 risk score to a five-tier severity level and applies
clinical rule-based overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fusion.patient_profile import PatientProfile
from fusion.rules import ClinicalRulesEngine, RuleEvaluationResult
from fusion.utils import FusionConfig, RISK_SEVERITY_BANDS, SeverityLevel


@dataclass
class SeverityResult:
    """Complete severity assessment output."""

    overall_severity: str
    base_severity: str
    risk_score: float
    rule_evaluation: RuleEvaluationResult
    rules_applied: bool

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "overall_severity": self.overall_severity,
            "base_severity": self.base_severity,
            "risk_score": self.risk_score,
            "rules_applied": self.rules_applied,
            "rule_evaluation": self.rule_evaluation.to_dict(),
        }


class SeverityEngine:
    """
    Determine overall patient severity from risk score and clinical rules.

    Base severity is derived from configurable risk bands:

        0–20   → NORMAL
        21–40  → LOW
        41–60  → MODERATE
        61–80  → HIGH
        81–100 → CRITICAL

    Clinical rules (see ``rules.py``) may override the base severity upward.
    """

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        rules_engine: Optional[ClinicalRulesEngine] = None,
    ) -> None:
        self.config = config or FusionConfig()
        self.rules_engine = rules_engine or ClinicalRulesEngine(config=self.config)

    def severity_from_risk(self, risk_score: float) -> str:
        """
        Map a numeric risk score to a severity tier.

        Parameters
        ----------
        risk_score:
            Patient risk score in the range 0–100.

        Returns
        -------
        str
            One of NORMAL, LOW, MODERATE, HIGH, CRITICAL.
        """
        score = max(0.0, min(100.0, risk_score))
        bands = self.config.risk_bands or RISK_SEVERITY_BANDS

        for low, high, severity in bands:
            if low <= score <= high:
                return severity

        return SeverityLevel.CRITICAL.value

    def assess(
        self,
        profile: PatientProfile,
        risk_score: float,
    ) -> SeverityResult:
        """
        Compute overall severity with optional rule-based overrides.

        Parameters
        ----------
        profile:
            Patient profile with module results.
        risk_score:
            Pre-computed patient risk score.

        Returns
        -------
        SeverityResult
            Final severity, base severity, and rule evaluation details.
        """
        base_severity = self.severity_from_risk(risk_score)
        rule_eval = self.rules_engine.evaluate(profile, base_severity)

        overall = (
            rule_eval.overridden_severity
            if rule_eval.overridden_severity is not None
            else base_severity
        )

        return SeverityResult(
            overall_severity=overall,
            base_severity=base_severity,
            risk_score=risk_score,
            rule_evaluation=rule_eval,
            rules_applied=rule_eval.applied,
        )


def determine_severity(
    profile: PatientProfile,
    risk_score: float,
    config: Optional[FusionConfig] = None,
) -> SeverityResult:
    """
    Convenience function to determine severity without instantiating the engine.

    Parameters
    ----------
    profile:
        Patient profile with module results.
    risk_score:
        Pre-computed risk score.
    config:
        Optional fusion configuration.
    """
    return SeverityEngine(config=config).assess(profile, risk_score)
