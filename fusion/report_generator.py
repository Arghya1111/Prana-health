"""
Structured patient report generation for the PRĀNA Fusion Engine.

Produces JSON output and PDF-ready formatted text reports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from fusion.confidence_fusion import ConfidenceFusionResult
from fusion.patient_profile import PatientProfile
from fusion.recommendation_engine import RecommendationResult
from fusion.risk_scoring import RiskScoreResult
from fusion.severity_engine import SeverityResult
from fusion.utils import utc_now_iso


@dataclass
class FusionReport:
    """Complete fusion output suitable for JSON serialization and PDF rendering."""

    patient_id: str
    timestamp: str
    modules: Dict[str, Dict[str, Any]]
    confidence: Dict[str, Any]
    risk: Dict[str, Any]
    severity: Dict[str, Any]
    recommendation: Dict[str, Any]
    summary: str
    pdf_ready_text: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full report to a JSON-compatible dictionary."""
        return {
            "patient_id": self.patient_id,
            "timestamp": self.timestamp,
            "modules": self.modules,
            "confidence": self.confidence,
            "risk": self.risk,
            "severity": self.severity,
            "recommendation": self.recommendation,
            "summary": self.summary,
            "pdf_ready_text": self.pdf_ready_text,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


class ReportGenerator:
    """
    Generate structured clinical reports from fusion engine results.

    Outputs:
      • Full JSON report (machine-readable)
      • PDF-ready plain-text report (human-readable)
      • Executive summary string
    """

    def __init__(self, platform_name: str = "PRANA") -> None:
        self.platform_name = platform_name

    def _build_summary(
        self,
        profile: PatientProfile,
        risk: RiskScoreResult,
        severity: SeverityResult,
        recommendation: RecommendationResult,
        confidence: ConfidenceFusionResult,
    ) -> str:
        """Generate a concise executive summary."""
        abnormal = [
            f"{r.module} ({r.status})"
            for r in profile.iter_modules()
            if r.is_abnormal
        ]
        abnormal_text = ", ".join(abnormal) if abnormal else "none"

        return (
            f"Patient {profile.patient_id}: Overall severity {severity.overall_severity} "
            f"(risk score {risk.risk_score:.1f}/100, AI confidence {confidence.overall_ai_confidence:.1f}%). "
            f"Abnormal findings: {abnormal_text}. "
            f"Recommendation: {recommendation.action}."
        )

    def _build_pdf_text(
        self,
        profile: PatientProfile,
        risk: RiskScoreResult,
        severity: SeverityResult,
        recommendation: RecommendationResult,
        confidence: ConfidenceFusionResult,
        summary: str,
    ) -> str:
        """Build a PDF-ready plain-text report."""
        lines = [
            "=" * 62,
            f"  {self.platform_name} — AI FUSION CLINICAL REPORT",
            "=" * 62,
            "",
            f"  Patient ID     : {profile.patient_id}",
            f"  Timestamp      : {profile.timestamp}",
            f"  Modules Active : {profile.module_count()}",
            "",
            "-" * 62,
            "  MODULE PREDICTIONS",
            "-" * 62,
        ]

        for name in profile.available_modules():
            result = profile.require(name)
            lines.append(f"  [{result.module}]")
            lines.append(f"    Status     : {result.status}")
            lines.append(f"    Severity   : {result.severity}")
            lines.append(f"    Confidence : {result.confidence:.1f}%")
            if result.features:
                key_features = []
                for key in ("spo2", "heart_rate", "fever", "inflammation", "pulse_quality"):
                    if key in result.features:
                        key_features.append(f"{key}={result.features[key]}")
                if key_features:
                    lines.append(f"    Features   : {', '.join(key_features)}")
            lines.append("")

        lines.extend([
            "-" * 62,
            "  FUSION METRICS",
            "-" * 62,
            f"  Risk Score           : {risk.risk_score:.1f} / 100",
            f"  Base Severity        : {severity.base_severity}",
            f"  Overall Severity     : {severity.overall_severity}",
            f"  Rules Applied        : {'Yes' if severity.rules_applied else 'No'}",
            f"  Average Confidence   : {confidence.average_confidence:.1f}%",
            f"  Weighted Confidence  : {confidence.weighted_confidence:.1f}%",
            f"  Overall AI Confidence: {confidence.overall_ai_confidence:.1f}%",
            "",
            "-" * 62,
            "  CLINICAL RECOMMENDATION",
            "-" * 62,
            f"  Action : {recommendation.action}",
            f"  Detail : {recommendation.primary_recommendation}",
        ])

        if recommendation.module_advisories:
            lines.append("")
            lines.append("  Module Advisories:")
            for advisory in recommendation.module_advisories:
                lines.append(f"    • {advisory}")

        if severity.rule_evaluation.triggered_rules:
            lines.extend(["", "  Triggered Clinical Rules:"])
            for rule in severity.rule_evaluation.triggered_rules:
                lines.append(f"    • [{rule.rule_id}] {rule.description}")

        lines.extend([
            "",
            "-" * 62,
            "  EXECUTIVE SUMMARY",
            "-" * 62,
            f"  {summary}",
            "",
            "=" * 62,
            f"  Generated by {self.platform_name} AI Fusion Engine",
            "=" * 62,
        ])

        return "\n".join(lines)

    def generate(
        self,
        profile: PatientProfile,
        risk: RiskScoreResult,
        severity: SeverityResult,
        recommendation: RecommendationResult,
        confidence: ConfidenceFusionResult,
    ) -> FusionReport:
        """
        Build a complete fusion report.

        Parameters
        ----------
        profile:
            Patient profile with all module results.
        risk:
            Computed risk score result.
        severity:
            Severity assessment result.
        recommendation:
            Clinical recommendation result.
        confidence:
            Confidence fusion result.

        Returns
        -------
        FusionReport
            Structured report with JSON and PDF-ready output.
        """
        modules_dict = {
            name: result.to_dict()
            for name, result in sorted(profile.modules.items())
        }

        summary = self._build_summary(profile, risk, severity, recommendation, confidence)
        pdf_text = self._build_pdf_text(
            profile, risk, severity, recommendation, confidence, summary
        )

        return FusionReport(
            patient_id=profile.patient_id,
            timestamp=profile.timestamp or utc_now_iso(),
            modules=modules_dict,
            confidence=confidence.to_dict(),
            risk=risk.to_dict(),
            severity=severity.to_dict(),
            recommendation=recommendation.to_dict(),
            summary=summary,
            pdf_ready_text=pdf_text,
        )
