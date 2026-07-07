# PRANA PPG Module (Phase 3)

Standalone photoplethysmography (PPG) analysis module for the PRANA AI-Powered Multi-Modal Patient Triage System. **Not integrated** with NATS, dashboard, fusion, or database.

## Directory layout

```
modules/ppg/
├── dataset/                  # PPG data (auto-generated if empty)
│   ├── ppg_data.csv          # or PPG_Dataset.csv
│   ├── dataset_meta.json
│   └── example_signal.png
├── model/
│   ├── ppg_model.pth         # Trained CNN checkpoint
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── utils.py                  # Paths, loading, synthetic data
├── inspect_dataset.py        # Step 1
├── preprocessing.py          # Step 2
├── feature_extraction.py     # Step 3
├── ppg_dataset.py            # Step 4
├── ppg_model.py              # Step 5
├── train_ppg.py              # Step 6
├── evaluate_model.py         # Step 7
├── ppg_inference.py          # Step 8
├── run_pipeline.py           # End-to-end runner
└── README.md
```

## Quick start

```bash
python modules/ppg/run_pipeline.py

# Or step-by-step
python modules/ppg/inspect_dataset.py
python modules/ppg/train_ppg.py
python modules/ppg/evaluate_model.py
python modules/ppg/ppg_inference.py
```

## Dataset

Supports `ppg_data.csv` or `PPG_Dataset.csv` in `dataset/`:

| Property | PPG_Dataset.csv (default) |
|----------|---------------------------|
| Samples | 2,576 |
| Signal length | 2,000 time points |
| Sampling rate | 125 Hz (~16 s window) |
| Classes | Normal, MI |
| Clinical mapping | Normal → NORMAL, MI → ABNORMAL |

If `dataset/` is empty, a synthetic reference dataset is generated automatically.

## Signal processing

`preprocessing.py` pipeline:

1. Baseline removal (moving-average subtraction)
2. Moving average smoothing
3. Butterworth band-pass (0.5–8 Hz)
4. Z-score normalization
5. Artifact detection (amplitude spike rejection)
6. Stratified train/val/test split (70/10/20)

## Feature extraction

`feature_extraction.py` extracts 12 features per window as a NumPy array:

heart_rate, pulse_rate_variability, peak_to_peak_interval, pulse_width, pulse_amplitude, pulse_energy, signal_entropy, peak_count, skewness, kurtosis, mean, standard_deviation

## CNN architecture

`PPG1DCNN` in `ppg_model.py` — identical style to `EEG1DCNN`:

```
Input (B, 1, L)
  → Conv1D(1→32) → BatchNorm → ReLU → MaxPool
  → Conv1D(32→64) → BatchNorm → ReLU → MaxPool
  → Conv1D(64→128) → BatchNorm → ReLU → MaxPool
  → Flatten → Linear → Dropout → Linear → Output
```

## Inference

```python
from modules.ppg.ppg_inference import predict_ppg_sample

result = predict_ppg_sample(ppg_signal)
# {"status": "NORMAL", "confidence": 97.5, "heart_rate": 72, "pulse_quality": "GOOD"}
```

## Future integration (Phase 4+)

Planned connections (not implemented):

- `fusion/prana_triage.py` — multi-modal severity engine
- `app/nats_pipeline.py` — real-time PPG topic
- `database/prana_database.py` — persist PPG reports

## Dependencies

Uses existing project packages: PyTorch, NumPy, SciPy, Pandas, scikit-learn, Matplotlib, neurokit2 (optional).
