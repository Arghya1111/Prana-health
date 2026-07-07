"""
Confidence fusion for the PRĀNA Fusion Engine.

Aggregates per-module confidence scores into overall AI confidence metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

from fusion.patient_profile import PatientProfile
from fusion.utils import DEFAULT_MODULE_WEIGHTS, FusionConfig


@dataclass
class ConfidenceFusionResult:
    """Aggregated confidence metrics across all available modules."""

    average_confidence: float
    weighted_confidence: float
    confidence_by_module: Dict[str, float] = field(default_factory=dict)
    overall_ai_confidence: float = 0.0
    module_count: int = 0

    def to_dict(self) -> Dict[str, float | int | Dict[str, float]]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "average_confidence": self.average_confidence,
            "weighted_confidence": self.weighted_confidence,
            "confidence_by_module": self.confidence_by_module,
            "overall_ai_confidence": self.overall_ai_confidence,
            "module_count": self.module_count,
        }


class ConfidenceFusionEngine:
    """
    Fuse per-module confidence scores into aggregate metrics.

    Computes:
      • Simple arithmetic mean (average confidence)
      • Weighted mean using configurable module weights
      • Per-module confidence breakdown
      • Overall AI confidence (weighted mean, primary fusion metric)
    """

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        self.config = config or FusionConfig()

    def fuse(self, profile: PatientProfile) -> ConfidenceFusionResult:
        """
        Compute confidence fusion metrics for a patient profile.

        Parameters
        ----------
        profile:
            Patient profile with standardized module results.

        Returns
        -------
        ConfidenceFusionResult
            Aggregated confidence metrics.
        """
        weights = self.config.module_weights
        by_module: Dict[str, float] = {}
        confidences: list[float] = []
        weighted_sum = 0.0
        weight_total = 0.0

        for result in profile.iter_modules():
            module_key = result.module.upper()
            confidence = round(result.confidence, 1)
            by_module[module_key] = confidence
            confidences.append(confidence)

            weight = float(weights.get(module_key, weights.get(result.module, 1.0)))
            weighted_sum += confidence * weight
            weight_total += weight

        module_count = len(confidences)

        if module_count == 0:
            return ConfidenceFusionResult(
                average_confidence=0.0,
                weighted_confidence=0.0,
                confidence_by_module={},
                overall_ai_confidence=0.0,
                module_count=0,
            )

        average = round(sum(confidences) / module_count, 1)
        weighted = round(weighted_sum / weight_total, 1) if weight_total > 0 else 0.0

        return ConfidenceFusionResult(
            average_confidence=average,
            weighted_confidence=weighted,
            confidence_by_module=by_module,
            overall_ai_confidence=weighted,
            module_count=module_count,
        )


def fuse_confidence(
    profile: PatientProfile,
    weights: Optional[Mapping[str, float]] = None,
) -> ConfidenceFusionResult:
    """
    Convenience function to fuse confidence without instantiating the engine.

    Parameters
    ----------
    profile:
        Patient profile with module results.
    weights:
        Optional custom module weights. Defaults to ``DEFAULT_MODULE_WEIGHTS``.
    """
    config = FusionConfig(
        module_weights=dict(weights) if weights else DEFAULT_MODULE_WEIGHTS
    )
    return ConfidenceFusionEngine(config).fuse(profile)
