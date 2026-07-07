"""
Weighted patient risk scoring for the PRĀNA Fusion Engine.

Computes a 0–100 Overall Patient Risk Score from standardized module severities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from fusion.patient_profile import PatientProfile
from fusion.utils import (
    DEFAULT_MODULE_WEIGHTS,
    SEVERITY_RISK_SCORES,
    FusionConfig,
    SeverityLevel,
)


@dataclass
class RiskScoreResult:
    """Detailed risk scoring output."""

    risk_score: float
    module_contributions: Dict[str, float]
    weights_used: Dict[str, float]
    active_module_count: int

    def to_dict(self) -> Dict[str, object]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "risk_score": self.risk_score,
            "module_contributions": self.module_contributions,
            "weights_used": self.weights_used,
            "active_module_count": self.active_module_count,
        }


class RiskScoringEngine:
    """
    Compute a weighted 0–100 patient risk score.

    Each module contributes ``severity_score × weight`` to a weighted average.
    Severity scores are defined in ``SEVERITY_RISK_SCORES``:

        NORMAL   → 0
        LOW      → 25
        MODERATE → 50
        HIGH     → 80
        CRITICAL → 100

    Weights are configurable (default: ECG=20, EEG=15, PPG=10, SpO₂=20,
    Thermal=10, Optical=10, X-Ray=15).
    """

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        self.config = config or FusionConfig()

    def _severity_score(self, severity: str) -> float:
        """Map a severity label to its numeric risk contribution."""
        return SEVERITY_RISK_SCORES.get(
            severity.upper(),
            SEVERITY_RISK_SCORES[SeverityLevel.NORMAL.value],
        )

    def compute(self, profile: PatientProfile) -> RiskScoreResult:
        """
        Calculate the overall patient risk score.

        Parameters
        ----------
        profile:
            Patient profile with standardized module results.

        Returns
        -------
        RiskScoreResult
            Risk score and per-module contribution breakdown.
        """
        weights = self.config.module_weights
        entries: list[tuple[str, float, float]] = []  # (module_key, weight, severity_score)

        for result in profile.iter_modules():
            module_key = result.module.upper()
            weight = float(weights.get(module_key, weights.get(result.module, 0.0)))
            if weight <= 0:
                continue
            severity_score = self._severity_score(result.severity)
            entries.append((module_key, weight, severity_score))

        if not entries:
            return RiskScoreResult(
                risk_score=0.0,
                module_contributions={},
                weights_used={},
                active_module_count=0,
            )

        weight_total = sum(e[1] for e in entries)
        weighted_sum = sum(e[1] * e[2] for e in entries)
        risk_score = round(min(100.0, weighted_sum / weight_total), 1)

        contributions: Dict[str, float] = {}
        weights_used: Dict[str, float] = {}
        for module_key, weight, severity_score in entries:
            # Points contributed toward the 0–100 risk score.
            contributions[module_key] = round(severity_score * weight / weight_total, 2)
            weights_used[module_key] = weight

        return RiskScoreResult(
            risk_score=risk_score,
            module_contributions=contributions,
            weights_used=weights_used,
            active_module_count=len(entries),
        )


def calculate_risk_score(
    profile: PatientProfile,
    weights: Optional[Mapping[str, float]] = None,
) -> RiskScoreResult:
    """
    Convenience function to compute risk score without instantiating the engine.

    Parameters
    ----------
    profile:
        Patient profile with module results.
    weights:
        Optional custom module weights.
    """
    config = FusionConfig(
        module_weights=dict(weights) if weights else DEFAULT_MODULE_WEIGHTS
    )
    return RiskScoringEngine(config).compute(profile)
