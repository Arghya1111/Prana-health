"""
STEP 8 — Thermal inference for a single image.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

_MODULE = Path(__file__).resolve().parent
if str(_MODULE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_MODULE.parent.parent))

import numpy as np
import torch

from modules.thermal.clinical_rules import class_index_to_risk, class_index_to_status, clinical_summary
from modules.thermal.preprocessing import PreprocessConfig, detect_hotspots, preprocess_image
from modules.thermal.thermal_model import build_model
from modules.thermal.utils import META_PATH, ThermalDatasetInfo, compute_thermal_stats, get_model_path, load_image


def load_thermal_model(
    model_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    path = Path(model_path or get_model_path())
    if not path.exists():
        raise FileNotFoundError(f"No model at {path}. Run train_thermal.py first.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    model = build_model(
        ckpt["image_size"],
        ckpt.get("in_channels", 1),
        ckpt["num_classes"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, {
        "image_size": ckpt["image_size"],
        "in_channels": ckpt.get("in_channels", 1),
        "num_classes": ckpt["num_classes"],
        "dataset_info": ckpt.get("dataset_info", {}),
    }


@torch.no_grad()
def predict_thermal_image(
    image: Union[np.ndarray, str, Path],
    model: Optional[torch.nn.Module] = None,
    model_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Predict thermal status for one image.

    Returns spo2-style dict with status, confidence, fever, inflammation, thermal_risk.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model, meta = load_thermal_model(model_path, device)
    else:
        meta = {"image_size": model.image_size, "num_classes": model.num_classes}

    if isinstance(image, (str, Path)):
        raw = load_image(Path(image), size=meta["image_size"])
    else:
        raw = np.asarray(image, dtype=np.float32).squeeze()
        if raw.max() > 1.0:
            raw = raw / 255.0

    try:
        info = ThermalDatasetInfo.load(META_PATH)
        config = PreprocessConfig.from_dataset_info(info)
    except Exception:
        config = PreprocessConfig(image_size=meta["image_size"])

    processed = preprocess_image(raw, config, augment=False)
    tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx] * 100.0)
    nc = meta["num_classes"]

    stats = compute_thermal_stats(raw)
    hotspots = detect_hotspots(raw)
    hotspot_count = int(np.sum(hotspots))

    clinical = clinical_summary(
        stats["mean_temp"],
        stats["max_temp"],
        hotspot_count,
        class_idx=class_idx,
        num_classes=nc,
    )

    return {
        "status": clinical["status"],
        "confidence": round(confidence, 1),
        "fever": clinical["fever"],
        "inflammation": clinical["inflammation"],
        "thermal_risk": clinical["thermal_risk"],
        "class_index": class_idx,
        "model_status": class_index_to_status(class_idx, nc),
        "severity": clinical["severity"],
        "thermal_stats": stats,
        "hotspot_pixels": hotspot_count,
        "probabilities": probs.tolist(),
    }


def main() -> None:
    print("\n[Thermal] Inference demo...\n")
    try:
        from modules.thermal.utils import load_raw_dataset
        images, labels, names, paths = load_raw_dataset(auto_generate=True)
        idx = int(np.argmax(labels))  # pick abnormal if available
        result = predict_thermal_image(images[idx])
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print("  Result:")
    for k in ("status", "confidence", "fever", "inflammation", "thermal_risk"):
        print(f"    {k:<18s}: {result[k]}")
    print()


if __name__ == "__main__":
    main()
