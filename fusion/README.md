# PRĀNA AI Fusion Engine

Multi-modality clinical decision support that combines predictions from all
standalone AI modules into a single intelligent assessment.

## Architecture

```
Module Outputs (ECG, EEG, PPG, SpO₂, Thermal, Optical, X-Ray)
        │
        ▼
  standardize_module_output()          ← utils.py
        │
        ▼
  PatientProfile                       ← patient_profile.py
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
 ConfidenceFusionEngine  RiskScoringEngine   (parallel)
 confidence_fusion.py    risk_scoring.py
        │                  │
        └────────┬─────────┘
                 ▼
         SeverityEngine                  ← severity_engine.py
         (+ ClinicalRulesEngine)         ← rules.py
                 │
                 ▼
      RecommendationEngine               ← recommendation_engine.py
                 │
                 ▼
        ReportGenerator                  ← report_generator.py
                 │
                 ▼
          FusionResult (JSON + PDF-ready text)
```

The orchestrator (`fusion_engine.py`) wires these components together. The
engine is **fully independent** — it does not depend on the dashboard, NATS,
SQLite, or API layers.

## Quick Start

```python
from fusion import FusionEngine, run_fusion

module_outputs = {
    "ECG":     {"module": "ECG",     "status": "ANOMALOUS", "confidence": 96.4, "severity": "HIGH"},
    "EEG":     {"module": "EEG",     "status": "NORMAL",    "confidence": 97.8, "severity": "NORMAL"},
    "PPG":     {"module": "PPG",     "status": "ABNORMAL",  "confidence": 93.1, "severity": "MODERATE"},
    "SpO2":    {"module": "SpO2",    "spo2": 87, "status": "LOW", "confidence": 99.0, "severity": "HIGH"},
    "THERMAL": {"module": "THERMAL", "status": "FEVER",     "confidence": 96.3, "severity": "MODERATE"},
    "OPTICAL": {"module": "OPTICAL", "disease": "NORMAL",  "confidence": 95.8, "severity": "NORMAL"},
    "XRAY":    {"module": "XRAY",    "status": "PNEUMONIA", "confidence": 94.2, "severity": "HIGH"},
}

result = run_fusion(patient_id="PAT-001", module_outputs=module_outputs)

print(result.overall_severity)    # e.g. CRITICAL
print(result.risk_score)            # e.g. 62.9
print(result.overall_confidence)    # e.g. 96.1
print(result.recommendation_text)
print(result.to_json())
```

Run the built-in demonstration:

```bash
python -m fusion.fusion_engine
```

## Standardized Module Output

Every module is normalized to:

```json
{
  "module": "ECG",
  "status": "ANOMALOUS",
  "confidence": 96.4,
  "severity": "HIGH",
  "features": {}
}
```

Raw outputs from individual modules are adapted automatically via
`standardize_module_output()` in `utils.py`. Heterogeneous status vocabularies
(e.g. ECG `ANOMALOUS`, EEG `ABNORMAL`, X-Ray `ATTENTION`, Thermal `FEVER`) are
mapped to the unified five-tier severity system.

## Risk Calculation

The **Overall Patient Risk Score** (0–100) uses weighted severity scoring:

| Severity | Risk Points |
|----------|-------------|
| NORMAL   | 0           |
| LOW      | 25          |
| MODERATE | 50          |
| HIGH     | 80          |
| CRITICAL | 100         |

**Formula:**

```
risk_score = Σ(severity_points × module_weight) / Σ(module_weights)
```

**Default weights** (configurable via `FusionConfig`):

| Module  | Weight |
|---------|--------|
| ECG     | 20     |
| EEG     | 15     |
| PPG     | 10     |
| SpO₂    | 20     |
| Thermal | 10     |
| Optical | 10     |
| X-Ray   | 15     |

```python
from fusion import FusionConfig, FusionEngine

config = FusionConfig(module_weights={
    "ECG": 25, "SpO2": 25, "XRAY": 20,
    "EEG": 10, "PPG": 10, "THERMAL": 5, "OPTICAL": 5,
})
engine = FusionEngine(config=config)
```

## Severity Logic

Base severity is derived from the risk score:

| Risk Score | Severity  |
|------------|-----------|
| 0–20       | NORMAL    |
| 21–40      | LOW       |
| 41–60      | MODERATE  |
| 61–80      | HIGH      |
| 81–100     | CRITICAL  |

Clinical rules in `rules.py` may **override** the base severity upward when
multi-modality patterns are detected.

## Clinical Rules

| Rule ID | Condition | Override |
|---------|-----------|----------|
| `RULE_CRITICAL_SEIZURE_HYPOXIA` | EEG seizure/abnormality + SpO₂ hypoxia | CRITICAL |
| `RULE_CRITICAL_MULTIPLE_HIGH` | ≥ 2 modules at HIGH severity | CRITICAL |
| `RULE_HIGH_ECG_HYPOXIA` | ECG anomalous + SpO₂ < 85% | HIGH |
| `RULE_HIGH_PNEUMONIA_FEVER` | X-Ray pneumonia + thermal fever | HIGH |

Rules are extensible via `ClinicalRulesEngine.add_rule()`.

## Recommendation Engine

| Severity  | Action |
|-----------|--------|
| NORMAL    | Routine monitoring |
| LOW       | Follow-up recommended |
| MODERATE  | Consult physician |
| HIGH      | Immediate specialist review |
| CRITICAL  | Emergency intervention required |

Module-specific advisories are appended for abnormal findings (e.g. pneumonia,
hypoxia, ECG anomaly).

## Confidence Fusion

| Metric | Description |
|--------|-------------|
| `average_confidence` | Arithmetic mean across all modules |
| `weighted_confidence` | Weighted mean using module weights |
| `confidence_by_module` | Per-module confidence breakdown |
| `overall_ai_confidence` | Primary metric (weighted confidence) |

## Report Output

`ReportGenerator` produces:

- **JSON** — full structured report (`result.to_json()`)
- **PDF-ready text** — formatted plain-text report (`result.report.pdf_ready_text`)
- **Executive summary** — one-line clinical summary (`result.report.summary`)

## File Reference

| File | Responsibility |
|------|----------------|
| `utils.py` | Types, enums, standardization adapters |
| `patient_profile.py` | Patient object aggregation |
| `confidence_fusion.py` | Confidence aggregation |
| `risk_scoring.py` | Weighted 0–100 risk score |
| `severity_engine.py` | Risk → severity mapping |
| `rules.py` | Clinical rule-based overrides |
| `recommendation_engine.py` | Clinical recommendations |
| `fusion_engine.py` | Main orchestrator |
| `report_generator.py` | JSON + PDF-ready reports |

> **Note:** `prana_triage.py` (legacy ECG + X-Ray triage) remains unchanged.
> The new fusion engine supersedes it for multi-modality integration in Phase 8.

## Future Improvements

- **Dynamic weight calibration** — learn optimal module weights from clinician feedback
- **Temporal fusion** — trend analysis across serial measurements per patient
- **Uncertainty quantification** — Bayesian confidence intervals per module
- **FHIR integration** — export fusion reports as FHIR DiagnosticReport resources
- **Explainability layer** — SHAP/LIME attributions per module contribution
- **Optical module adapter** — extend when `modules/optical/` is added to the repo
- **Phase 8 integrations** — NATS streaming, SQLite persistence, REST API, dashboard
