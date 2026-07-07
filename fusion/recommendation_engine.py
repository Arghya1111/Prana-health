"""
Clinical recommendation engine for the PRĀNA Fusion Engine.

Generates actionable clinical recommendations based on overall severity and
individual module findings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fusion.patient_profile import PatientProfile
from fusion.utils import SeverityLevel


# Primary recommendation per severity tier.
SEVERITY_RECOMMENDATIONS: Dict[str, str] = {
    SeverityLevel.NORMAL.value: "Routine monitoring — no immediate clinical action required.",
    SeverityLevel.LOW.value: "Follow-up recommended within standard care pathway.",
    SeverityLevel.MODERATE.value: "Consult physician — clinical review advised within 24–48 hours.",
    SeverityLevel.HIGH.value: "Immediate specialist review required.",
    SeverityLevel.CRITICAL.value: "Emergency intervention required — activate critical care protocol.",
}

# Short action labels for reports and dashboards.
SEVERITY_ACTIONS: Dict[str, str] = {
    SeverityLevel.NORMAL.value: "Routine monitoring",
    SeverityLevel.LOW.value: "Follow-up recommended",
    SeverityLevel.MODERATE.value: "Consult physician",
    SeverityLevel.HIGH.value: "Immediate specialist review",
    SeverityLevel.CRITICAL.value: "Emergency intervention required",
}

# Module-specific supplemental recommendations.
MODULE_ADVISORIES: Dict[str, Dict[str, str]] = {
    "ECG": {
        "ANOMALOUS": "ECG anomaly detected — consider cardiology referral and continuous cardiac monitoring.",
        "ABNORMAL": "ECG abnormality detected — consider cardiology referral and continuous cardiac monitoring.",
    },
    "EEG": {
        "ABNORMAL": "EEG abnormality detected — neurology consultation and seizure precautions advised.",
        "SEIZURE": "Seizure activity detected — immediate neurology review and airway protection required.",
    },
    "PPG": {
        "ABNORMAL": "Abnormal pulse waveform — verify sensor placement and repeat measurement.",
    },
    "SpO2": {
        "LOW": "Low oxygen saturation — administer supplemental oxygen and monitor respiratory status.",
        "HIGH": "Significant hypoxia — administer supplemental oxygen and escalate respiratory support.",
        "CRITICAL": "Critical hypoxia — immediate oxygen therapy and emergency respiratory assessment.",
        "MODERATE": "Mild hypoxia — monitor SpO₂ closely and assess need for supplemental oxygen.",
    },
    "THERMAL": {
        "FEVER": "Elevated thermal signature consistent with fever — assess for infection source.",
        "HIGH": "High body temperature detected — antipyretic therapy and infection workup advised.",
        "CRITICAL": "Critical thermal abnormality — immediate clinical assessment required.",
    },
    "OPTICAL": {
        "ABNORMAL": "Optical screening abnormality — specialist ophthalmology review recommended.",
    },
    "XRAY": {
        "PNEUMONIA": "Radiographic evidence of pneumonia — initiate antimicrobial therapy per protocol.",
        "CRITICAL": "Critical radiographic findings — urgent pulmonary specialist review.",
        "ATTENTION": "Radiographic findings require follow-up imaging or clinical correlation.",
    },
}


@dataclass
class RecommendationResult:
    """Clinical recommendation output."""

    severity: str
    primary_recommendation: str
    action: str
    module_advisories: List[str] = field(default_factory=list)
    combined_recommendation: str = ""

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "severity": self.severity,
            "primary_recommendation": self.primary_recommendation,
            "action": self.action,
            "module_advisories": self.module_advisories,
            "combined_recommendation": self.combined_recommendation,
        }


class RecommendationEngine:
    """
    Generate clinical recommendations from overall severity and module findings.

    Produces a primary recommendation based on severity tier, supplemented by
    module-specific advisories for abnormal findings.
    """

    def __init__(
        self,
        severity_recommendations: Optional[Dict[str, str]] = None,
        module_advisories: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> None:
        self.severity_recommendations = severity_recommendations or SEVERITY_RECOMMENDATIONS
        self.module_advisories = module_advisories or MODULE_ADVISORIES

    def _module_advisory(self, module: str, status: str) -> Optional[str]:
        """Look up a supplemental advisory for a module status."""
        advisories = self.module_advisories.get(module, {})
        return advisories.get(status.upper())

    def generate(
        self,
        profile: PatientProfile,
        overall_severity: str,
    ) -> RecommendationResult:
        """
        Generate clinical recommendations for a patient.

        Parameters
        ----------
        profile:
            Patient profile with module results.
        overall_severity:
            Final overall severity tier.

        Returns
        -------
        RecommendationResult
            Primary and supplemental clinical recommendations.
        """
        severity = overall_severity.upper()
        primary = self.severity_recommendations.get(
            severity,
            SEVERITY_RECOMMENDATIONS[SeverityLevel.MODERATE.value],
        )
        action = SEVERITY_ACTIONS.get(severity, "Clinical review advised")

        advisories: List[str] = []
        for result in profile.iter_modules():
            advisory = self._module_advisory(result.module, result.status)
            if advisory and result.is_abnormal:
                advisories.append(advisory)

        combined_parts = [primary]
        if advisories:
            combined_parts.append("Module-specific advisories: " + " ".join(advisories))

        return RecommendationResult(
            severity=severity,
            primary_recommendation=primary,
            action=action,
            module_advisories=advisories,
            combined_recommendation=" ".join(combined_parts),
        )


def generate_recommendation(
    profile: PatientProfile,
    overall_severity: str,
) -> RecommendationResult:
    """
    Convenience function to generate recommendations without instantiating the engine.

    Parameters
    ----------
    profile:
        Patient profile with module results.
    overall_severity:
        Final overall severity tier.
    """
    return RecommendationEngine().generate(profile, overall_severity)
