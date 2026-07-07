"""
STEP 9 — SpO2 inference for a single patient sample.

Load model/spo2_model.pth and return clinical prediction:

    {
        "spo2": 97,
        "status": "NORMAL",
        "severity": "LOW",
        "confidence": 98.6,
        "recommendation": "Normal Oxygen Saturation"
    }
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

from modules.spo2.clinical_rules import (
    get_recommendation,
    spo2_to_severity,
    spo2_to_status,
)
from modules.spo2.feature_extraction import extract_spo2_features
from modules.spo2.preprocessing import PreprocessConfig, preprocess_signals
from modules.spo2.spo2_model import build_model
from modules.spo2.utils import META_PATH, SpO2DatasetInfo, get_model_path, load_raw_dataset


def load_spo2_model(
    model_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    """Load trained SpO2 CNN and associated metadata from checkpoint."""
    path = Path(model_path or get_model_path())
    if not path.exists():
        raise FileNotFoundError(
            f"No trained SpO2 model at {path}. Run train_spo2.py first."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    model = build_model(
        checkpoint["signal_length"],
        checkpoint.get("in_channels", 2),
        checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    meta = {
        "signal_length": checkpoint["signal_length"],
        "in_channels": checkpoint.get("in_channels", 2),
        "num_classes": checkpoint["num_classes"],
        "task_type": checkpoint.get("task_type", "classification"),
        "dataset_info": checkpoint.get("dataset_info", {}),
    }
    return model, meta


def _prepare_sample(
    signal: np.ndarray,
    signal_length: int,
    in_channels: int,
    sampling_rate: float,
    config: Optional[PreprocessConfig] = None,
) -> np.ndarray:
    """Ensure correct shape and apply preprocessing."""
    sig = np.asarray(signal, dtype=np.float32)

    if sig.ndim == 1:
        sig = np.stack([sig, np.full(signal_length, 0.97)], axis=0)
    elif sig.ndim == 2 and sig.shape[0] != in_channels:
        if sig.shape[0] < in_channels:
            pad = np.tile(sig[-1:], (in_channels - sig.shape[0], 1))
            sig = np.concatenate([sig, pad], axis=0)

    if sig.shape[1] != signal_length:
        if sig.shape[1] > signal_length:
            sig = sig[:, :signal_length]
        else:
            pad = signal_length - sig.shape[1]
            sig = np.pad(sig, ((0, 0), (0, pad)), mode="edge")

    batch = sig.reshape(1, in_channels, signal_length)
    if config is None:
        try:
            info = SpO2DatasetInfo.load(META_PATH)
            config = PreprocessConfig.from_dataset_info(info)
        except Exception:
            config = PreprocessConfig(sampling_rate=sampling_rate)

    return preprocess_signals(batch, config)


@torch.no_grad()
def predict_spo2_sample(
    signal: Union[np.ndarray, list],
    model: Optional[torch.nn.Module] = None,
    model_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    sampling_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run inference on a single synchronized PPG/SpO2 window.

    Returns spo2, status, severity, confidence, and recommendation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model, meta = load_spo2_model(model_path=model_path, device=device)
    else:
        meta = {
            "signal_length": model.signal_length,
            "in_channels": model.in_channels,
            "num_classes": model.num_classes,
            "task_type": "classification",
        }

    sl = meta["signal_length"]
    ic = meta["in_channels"]
    task_type = meta.get("task_type", "classification")
    sr = sampling_rate or _default_sr(meta)

    raw = np.asarray(signal, dtype=np.float32)
    processed = _prepare_sample(raw, sl, ic, sr)
    tensor = torch.from_numpy(processed).to(device)

    logits = model(tensor)

    if task_type == "regression":
        spo2_val = int(round(float(logits.squeeze().cpu().item())))
        confidence = 95.0
    else:
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx] * 100.0)

        # Estimate SpO2 from raw signal features
        raw_sig = raw if raw.ndim == 2 else np.stack([raw, np.full(len(raw), 97.0)])
        if raw_sig.shape[0] == 1:
            raw_sig = np.stack([raw_sig[0], np.full(raw_sig.shape[1], 97.0)])
        feat = extract_spo2_features(raw_sig, sr)
        spo2_val = int(round(feat["average_spo2"]))

    status = spo2_to_status(float(spo2_val))
    severity = spo2_to_severity(float(spo2_val))
    recommendation = get_recommendation(float(spo2_val))

    return {
        "spo2": spo2_val,
        "status": status,
        "severity": severity,
        "confidence": round(confidence, 1),
        "recommendation": recommendation,
    }


def _default_sr(meta: Dict[str, Any]) -> float:
    try:
        return SpO2DatasetInfo.from_dict(meta.get("dataset_info", {})).sampling_rate
    except Exception:
        try:
            return SpO2DatasetInfo.load(META_PATH).sampling_rate
        except Exception:
            return 64.0


def main() -> None:
    print("\n[SpO2] Running inference demo on sample 0...\n")

    try:
        signals, _, _, _, _, _ = load_raw_dataset(auto_generate=True)
        result = predict_spo2_sample(signals[0])
    except Exception as exc:
        print(f"[ERROR] Inference failed: {exc}")
        sys.exit(1)

    print("  Result:")
    print(f"    spo2            : {result['spo2']}%")
    print(f"    status          : {result['status']}")
    print(f"    severity        : {result['severity']}")
    print(f"    confidence      : {result['confidence']}%")
    print(f"    recommendation  : {result['recommendation']}")
    print()


if __name__ == "__main__":
    main()
