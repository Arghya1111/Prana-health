"""
PRĀNA AI Fusion Engine — Phase 7 multi-modality clinical decision support.

Combines ECG, EEG, PPG, SpO₂, Thermal, Optical, and X-Ray predictions into a
single risk score, severity assessment, confidence metric, and clinical
recommendation.

Usage::

    from fusion import FusionEngine, run_fusion

    result = run_fusion("PAT-001", {"ECG": {...}, "SpO2": {...}})
    print(result.overall_severity, result.risk_score)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fusion.confidence_fusion import ConfidenceFusionEngine, ConfidenceFusionResult
    from fusion.fusion_engine import FusionEngine, FusionResult
    from fusion.patient_profile import PatientProfile
    from fusion.recommendation_engine import RecommendationEngine, RecommendationResult
    from fusion.report_generator import FusionReport, ReportGenerator
    from fusion.risk_scoring import RiskScoreResult, RiskScoringEngine
    from fusion.rules import ClinicalRule, ClinicalRulesEngine, RuleEvaluationResult
    from fusion.severity_engine import SeverityEngine, SeverityResult
    from fusion.utils import (
        FusionConfig,
        FusionError,
        ModuleName,
        ModuleResult,
        SeverityLevel,
        StandardizationError,
    )

__all__ = [
    "FusionEngine",
    "FusionResult",
    "run_fusion",
    "PatientProfile",
    "ConfidenceFusionEngine",
    "ConfidenceFusionResult",
    "fuse_confidence",
    "RiskScoringEngine",
    "RiskScoreResult",
    "calculate_risk_score",
    "SeverityEngine",
    "SeverityResult",
    "determine_severity",
    "ClinicalRulesEngine",
    "ClinicalRule",
    "RuleEvaluationResult",
    "RecommendationEngine",
    "RecommendationResult",
    "ReportGenerator",
    "FusionReport",
    "ModuleResult",
    "ModuleName",
    "SeverityLevel",
    "FusionConfig",
    "FusionError",
    "StandardizationError",
    "DEFAULT_MODULE_WEIGHTS",
    "standardize_module_output",
    "standardize_batch",
]


def __getattr__(name: str):
    """Lazy imports to avoid circular loading when running submodules as __main__."""
    _exports = {
        "FusionEngine": ("fusion.fusion_engine", "FusionEngine"),
        "FusionResult": ("fusion.fusion_engine", "FusionResult"),
        "run_fusion": ("fusion.fusion_engine", "run_fusion"),
        "PatientProfile": ("fusion.patient_profile", "PatientProfile"),
        "ConfidenceFusionEngine": ("fusion.confidence_fusion", "ConfidenceFusionEngine"),
        "ConfidenceFusionResult": ("fusion.confidence_fusion", "ConfidenceFusionResult"),
        "fuse_confidence": ("fusion.confidence_fusion", "fuse_confidence"),
        "RiskScoringEngine": ("fusion.risk_scoring", "RiskScoringEngine"),
        "RiskScoreResult": ("fusion.risk_scoring", "RiskScoreResult"),
        "calculate_risk_score": ("fusion.risk_scoring", "calculate_risk_score"),
        "SeverityEngine": ("fusion.severity_engine", "SeverityEngine"),
        "SeverityResult": ("fusion.severity_engine", "SeverityResult"),
        "determine_severity": ("fusion.severity_engine", "determine_severity"),
        "ClinicalRulesEngine": ("fusion.rules", "ClinicalRulesEngine"),
        "ClinicalRule": ("fusion.rules", "ClinicalRule"),
        "RuleEvaluationResult": ("fusion.rules", "RuleEvaluationResult"),
        "RecommendationEngine": ("fusion.recommendation_engine", "RecommendationEngine"),
        "RecommendationResult": ("fusion.recommendation_engine", "RecommendationResult"),
        "ReportGenerator": ("fusion.report_generator", "ReportGenerator"),
        "FusionReport": ("fusion.report_generator", "FusionReport"),
        "ModuleResult": ("fusion.utils", "ModuleResult"),
        "ModuleName": ("fusion.utils", "ModuleName"),
        "SeverityLevel": ("fusion.utils", "SeverityLevel"),
        "FusionConfig": ("fusion.utils", "FusionConfig"),
        "FusionError": ("fusion.utils", "FusionError"),
        "StandardizationError": ("fusion.utils", "StandardizationError"),
        "DEFAULT_MODULE_WEIGHTS": ("fusion.utils", "DEFAULT_MODULE_WEIGHTS"),
        "standardize_module_output": ("fusion.utils", "standardize_module_output"),
        "standardize_batch": ("fusion.utils", "standardize_batch"),
    }
    if name in _exports:
        module_path, attr = _exports[name]
        import importlib
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
