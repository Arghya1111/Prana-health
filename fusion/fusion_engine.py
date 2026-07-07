"""
PRĀNA AI Fusion Engine — core orchestrator.

Integrates predictions from ECG, EEG, PPG, SpO₂, Thermal, Optical, and X-Ray
modules into a single intelligent clinical decision with risk score, severity,
confidence, and clinical recommendations.

This module is intentionally independent — it does NOT integrate with the
dashboard, NATS, SQLite, or API layers (Phase 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from fusion.confidence_fusion import ConfidenceFusionEngine, ConfidenceFusionResult
from fusion.patient_profile import PatientProfile
from fusion.recommendation_engine import RecommendationEngine, RecommendationResult
from fusion.report_generator import FusionReport, ReportGenerator
from fusion.risk_scoring import RiskScoringEngine, RiskScoreResult
from fusion.rules import ClinicalRulesEngine
from fusion.severity_engine import SeverityEngine, SeverityResult
from fusion.utils import (
    FusionConfig,
    FusionError,
    ModuleResult,
    StandardizationError,
    utc_now_iso,
)


@dataclass
class FusionResult:
    """
    Complete output of the AI Fusion Engine.

    Contains all intermediate results and the final structured report.
    """

    profile: PatientProfile
    risk: RiskScoreResult
    confidence: ConfidenceFusionResult
    severity: SeverityResult
    recommendation: RecommendationResult
    report: FusionReport

    @property
    def risk_score(self) -> float:
        return self.risk.risk_score

    @property
    def overall_severity(self) -> str:
        return self.severity.overall_severity

    @property
    def overall_confidence(self) -> float:
        return self.confidence.overall_ai_confidence

    @property
    def recommendation_text(self) -> str:
        return self.recommendation.combined_recommendation

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the complete fusion result to a JSON-compatible dict."""
        return {
            "patient_id": self.profile.patient_id,
            "timestamp": self.profile.timestamp,
            "risk_score": self.risk_score,
            "overall_severity": self.overall_severity,
            "overall_confidence": self.overall_confidence,
            "recommendation": self.recommendation.to_dict(),
            "report": self.report.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Return a formatted JSON string of the fusion result."""
        import json
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class FusionEngine:
    """
    Central AI Fusion Engine for PRĀNA.

    Pipeline:
        1. Assemble patient profile from module outputs
        2. Fuse confidence scores
        3. Compute weighted risk score
        4. Determine overall severity (with clinical rule overrides)
        5. Generate clinical recommendations
        6. Produce structured report (JSON + PDF-ready text)

    Parameters
    ----------
    config:
        Configurable weights, thresholds, and risk bands.
    """

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        self.config = config or FusionConfig()
        self._confidence_engine = ConfidenceFusionEngine(self.config)
        self._risk_engine = RiskScoringEngine(self.config)
        self._rules_engine = ClinicalRulesEngine(config=self.config)
        self._severity_engine = SeverityEngine(self.config, self._rules_engine)
        self._recommendation_engine = RecommendationEngine()
        self._report_generator = ReportGenerator()

    def fuse(
        self,
        patient_id: str,
        module_outputs: Dict[str, Union[Dict[str, Any], ModuleResult, None]],
        *,
        timestamp: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FusionResult:
        """
        Run the full fusion pipeline on module predictions.

        Parameters
        ----------
        patient_id:
            Unique patient identifier.
        module_outputs:
            Mapping of module name → raw prediction dict or ``ModuleResult``.
            Use ``None`` for unavailable modules.
        timestamp:
            Optional ISO timestamp override.
        metadata:
            Optional patient metadata (age, ward, etc.).

        Returns
        -------
        FusionResult
            Complete fusion output including structured report.

        Raises
        ------
        FusionError
            If the profile is invalid or fusion fails.
        """
        try:
            profile = PatientProfile.from_module_outputs(
                patient_id=patient_id,
                outputs=module_outputs,
                timestamp=timestamp,
                metadata=metadata,
            )
            profile.validate(require_at_least_one=True)

            confidence = self._confidence_engine.fuse(profile)
            risk = self._risk_engine.compute(profile)
            severity = self._severity_engine.assess(profile, risk.risk_score)
            recommendation = self._recommendation_engine.generate(
                profile, severity.overall_severity
            )
            report = self._report_generator.generate(
                profile, risk, severity, recommendation, confidence
            )

            return FusionResult(
                profile=profile,
                risk=risk,
                confidence=confidence,
                severity=severity,
                recommendation=recommendation,
                report=report,
            )
        except StandardizationError:
            raise
        except Exception as exc:
            raise FusionError(f"Fusion pipeline failed: {exc}") from exc

    def fuse_profile(self, profile: PatientProfile) -> FusionResult:
        """
        Run the fusion pipeline on an existing ``PatientProfile``.

        Parameters
        ----------
        profile:
            Pre-assembled patient profile.

        Returns
        -------
        FusionResult
            Complete fusion output.
        """
        try:
            profile.validate(require_at_least_one=True)

            confidence = self._confidence_engine.fuse(profile)
            risk = self._risk_engine.compute(profile)
            severity = self._severity_engine.assess(profile, risk.risk_score)
            recommendation = self._recommendation_engine.generate(
                profile, severity.overall_severity
            )
            report = self._report_generator.generate(
                profile, risk, severity, recommendation, confidence
            )

            return FusionResult(
                profile=profile,
                risk=risk,
                confidence=confidence,
                severity=severity,
                recommendation=recommendation,
                report=report,
            )
        except StandardizationError:
            raise
        except Exception as exc:
            raise FusionError(f"Fusion pipeline failed: {exc}") from exc


def run_fusion(
    patient_id: str,
    module_outputs: Dict[str, Union[Dict[str, Any], ModuleResult, None]],
    *,
    config: Optional[FusionConfig] = None,
    timestamp: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> FusionResult:
    """
    Convenience function to run the fusion engine in one call.

    Parameters
    ----------
    patient_id:
        Patient identifier.
    module_outputs:
        Module prediction dictionaries.
    config:
        Optional fusion configuration.
    timestamp:
        Optional timestamp override.
    metadata:
        Optional patient metadata.

    Returns
    -------
    FusionResult
        Complete fusion output.
    """
    return FusionEngine(config=config).fuse(
        patient_id=patient_id,
        module_outputs=module_outputs,
        timestamp=timestamp,
        metadata=metadata,
    )


# ── Demonstration with the Phase 7 example payloads ──────────────────────────

def _demo_module_outputs() -> Dict[str, Dict[str, Any]]:
    """Return the example module outputs from the Phase 7 specification."""
    return {
        "ECG": {
            "module": "ECG",
            "status": "ANOMALOUS",
            "confidence": 96.4,
            "severity": "HIGH",
            "features": {},
        },
        "EEG": {
            "module": "EEG",
            "status": "NORMAL",
            "confidence": 97.8,
            "severity": "NORMAL",
            "features": {},
        },
        "PPG": {
            "module": "PPG",
            "status": "ABNORMAL",
            "confidence": 93.1,
            "severity": "MODERATE",
            "features": {},
        },
        "SpO2": {
            "module": "SpO2",
            "spo2": 87,
            "status": "LOW",
            "confidence": 99.0,
            "severity": "HIGH",
        },
        "THERMAL": {
            "module": "THERMAL",
            "status": "FEVER",
            "confidence": 96.3,
            "severity": "MODERATE",
        },
        "OPTICAL": {
            "module": "OPTICAL",
            "disease": "NORMAL",
            "confidence": 95.8,
            "severity": "NORMAL",
        },
        "XRAY": {
            "module": "XRAY",
            "status": "PNEUMONIA",
            "confidence": 94.2,
            "severity": "HIGH",
        },
    }


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    engine = FusionEngine()
    result = engine.fuse(
        patient_id="PAT-PHASE7-001",
        module_outputs=_demo_module_outputs(),
        timestamp=utc_now_iso(),
    )

    print(result.report.pdf_ready_text)
    print()
    print("── JSON OUTPUT ──")
    print(result.to_json())
