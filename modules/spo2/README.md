# PRANA SpO2 Module (Phase 4)

Standalone SpO2 / photoplethysmography analysis module for the PRANA AI-Powered Multi-Modal Patient Triage System. **Not integrated** with NATS, dashboard, fusion, or database.

## Directory layout

```
modules/spo2/
├── dataset/
│   ├── spo2_data.csv
│   ├── dataset_meta.json
│   └── example_signal.png
├── model/
│   ├── spo2_model.pth
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── utils.py
├── inspect_dataset.py        # Step 1
├── preprocessing.py          # Step 2
├── feature_extraction.py     # Step 3
├── spo2_dataset.py           # Step 4
├── spo2_model.py             # Step 5
├── train_spo2.py             # Step 6
├── evaluate_model.py         # Step 7
├── clinical_rules.py         # Step 8
├── spo2_inference.py         # Step 9
├── run_pipeline.py
└── README.md
```

## Quick start

```bash
python modules/spo2/run_pipeline.py

# Or step-by-step
python modules/spo2/inspect_dataset.py
python modules/spo2/train_spo2.py
python modules/spo2/evaluate_model.py
python modules/spo2/spo2_inference.py
```

## Dataset

Default `spo2_data.csv` format:

| Column | Description |
|--------|-------------|
| subject_id | Patient/subject identifier |
| label | 0 = NORMAL, 1 = ABNORMAL (hypoxia) |
| spo2_mean | Reference SpO2 (%) |
| X1…X256 | PPG waveform samples |
| S1…S256 | SpO2 trace samples (%) |

| Property | Value |
|----------|-------|
| Samples | 480 |
| Signal length | 256 @ 64 Hz (~4 s) |
| Channels | 2 (PPG + SpO2) |
| Classes | 2 (binary hypoxia detection) |

## Clinical rules

| SpO2 (%) | Status | Severity |
|----------|--------|----------|
| ≥ 95 | NORMAL | LOW |
| 90–94 | MODERATE | MODERATE |
| 85–89 | HIGH | HIGH |
| < 85 | CRITICAL | CRITICAL |

## Inference

```python
from modules.spo2.spo2_inference import predict_spo2_sample

result = predict_spo2_sample(signal)  # (2, 256) PPG + SpO2 window
# {
#   "spo2": 97,
#   "status": "NORMAL",
#   "severity": "LOW",
#   "confidence": 98.6,
#   "recommendation": "Normal Oxygen Saturation"
# }
```

## Future integration

Planned connections (not implemented):

- `fusion/prana_triage.py` — multi-modal severity engine
- `app/nats_pipeline.py` — real-time SpO2 topic
- `database/prana_database.py` — persist SpO2 reports

## Dependencies

PyTorch, NumPy, SciPy, Pandas, scikit-learn, Matplotlib, neurokit2 (optional).
