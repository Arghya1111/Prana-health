# PRANA Thermal Module (Phase 5)

Standalone thermal diagnostics module for fever, inflammation, and thermal anomaly detection. **Not integrated** with NATS, dashboard, fusion, or database.

## Layout

```
modules/thermal/
├── dataset/                  # Class folders: normal/, inflammation/, fever/, severe/
├── model/                    # thermal_model.pth, ROC, visualizations
├── clinical_rules.py         # Step 9 — thermal risk thresholds
├── utils.py
├── inspect_dataset.py        # Step 1
├── preprocessing.py          # Step 2
├── image_dataset.py          # Step 3
├── thermal_model.py          # Step 4
├── train_thermal.py          # Step 5
├── evaluate_model.py         # Step 6
├── visualization.py          # Step 7
├── thermal_inference.py      # Step 8
└── README.md
```

## Quick start

```bash
python modules/thermal/inspect_dataset.py
python modules/thermal/train_thermal.py
python modules/thermal/evaluate_model.py
python modules/thermal/thermal_inference.py
python modules/thermal/visualization.py
```

## Dataset

Auto-generates synthetic thermal PNGs if `dataset/` is empty:

| Class folder | Meaning |
|--------------|---------|
| `normal/` | Uniform body heat |
| `inflammation/` | Localized hotspot |
| `fever/` | Elevated temperature + hotspots |
| `severe/` | Multiple intense hotspots |

Default: **2 classes** (`normal`, `inflammation`). Set `num_classes=4` in `generate_synthetic_dataset()`.

Custom data: place images in `dataset/<class_name>/`.

## Preprocessing

- Resize, denoise, CLAHE contrast enhancement
- Temperature normalization (z-score)
- Augmentation: horizontal/vertical flip, rotation, brightness
- Hotspot detection via threshold + connected components

## CNN architecture

2D CNN: 3x Conv2D blocks -> AdaptiveAvgPool -> Dropout -> Linear. Supports binary and multi-class.

## Clinical rules

| Finding | thermal_risk |
|---------|--------------|
| Normal temperature distribution | LOW |
| Localized inflammation / mild fever | MODERATE |
| High body temperature | HIGH |
| Severe thermal abnormality | CRITICAL |

## Inference

```python
from modules.thermal.thermal_inference import predict_thermal_image

result = predict_thermal_image(image_array)
# {
#   "status": "NORMAL",
#   "confidence": 98.1,
#   "fever": "NO",
#   "inflammation": "NO",
#   "thermal_risk": "LOW"
# }
```

## Visualization

`visualization.py` produces original image, inferno heatmap, temperature histogram, and hotspot overlay.

## Future integration (Phase 7)

Planned: `fusion/`, `app/nats_pipeline.py`, `database/`, `app/dashboard.py`.

## Dependencies

PyTorch, TorchVision, NumPy, OpenCV (`opencv-python-headless`), Matplotlib, scikit-learn, Pillow.
