"""
STEP 8 — PPG inference for a single signal.

Returns clinical-style prediction with heart rate and pulse quality:

    {
        "status": "NORMAL",
        "confidence": 97.6,
        "heart_rate": 72,
        "pulse_quality": "GOOD"
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

from modules.ppg.feature_extraction import assess_pulse_quality, extract_ppg_features
from modules.ppg.ppg_model import build_model
from modules.ppg.preprocessing import PreprocessConfig, preprocess_signals
from modules.ppg.utils import META_PATH, PPGDatasetInfo, get_model_path, load_raw_dataset


def load_ppg_model(
    model_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    """Load trained PPG CNN from checkpoint."""
    path = Path(model_path or get_model_path())
    if not path.exists():
        raise FileNotFoundError(
            f"No trained PPG model at {path}. Run train_ppg.py first."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    signal_length = checkpoint["signal_length"]
    num_classes = checkpoint["num_classes"]

    model = build_model(signal_length=signal_length, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    meta = {
        "signal_length": signal_length,
        "num_classes": num_classes,
        "dataset_info": checkpoint.get("dataset_info", {}),
    }
    return model, meta


def _prepare_sample(
    signal: np.ndarray,
    signal_length: int,
    sampling_rate: float,
    config: Optional[PreprocessConfig] = None,
) -> np.ndarray:
    """Ensure correct length and apply preprocessing."""
    signal = np.asarray(signal, dtype=np.float32).squeeze()

    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}")

    if len(signal) != signal_length:
        if len(signal) > signal_length:
            signal = signal[:signal_length]
        else:
            signal = np.pad(signal, (0, signal_length - len(signal)), mode="edge")

    batch = signal.reshape(1, -1)
    if config is None:
        try:
            info = PPGDatasetInfo.load(META_PATH)
            config = PreprocessConfig.from_dataset_info(info)
        except Exception:
            config = PreprocessConfig(sampling_rate=sampling_rate)

    return preprocess_signals(batch, config)


@torch.no_grad()
def predict_ppg_sample(
    signal: Union[np.ndarray, list],
    model: Optional[torch.nn.Module] = None,
    model_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    sampling_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Run inference on a single PPG window.

    Returns status, confidence, heart_rate, and pulse_quality.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model, meta = load_ppg_model(model_path=model_path, device=device)
    else:
        meta = {"signal_length": model.signal_length, "num_classes": model.num_classes}

    signal_length = meta["signal_length"]
    sr = sampling_rate or _default_sampling_rate(meta)

    raw_signal = np.asarray(signal, dtype=np.float32).squeeze()
    processed = _prepare_sample(raw_signal, signal_length, sr)

    tensor = torch.from_numpy(processed).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx] * 100.0)

    status = "NORMAL" if class_idx == 0 else "ABNORMAL"

    # Physiological features from raw (pre-normalization) window
    feat = extract_ppg_features(raw_signal, sr)
    heart_rate = int(round(feat["heart_rate"])) if feat["heart_rate"] > 0 else 0
    pulse_quality = assess_pulse_quality(raw_signal, sr)

    return {
        "status": status,
        "confidence": round(confidence, 1),
        "heart_rate": heart_rate,
        "pulse_quality": pulse_quality,
    }


def _default_sampling_rate(meta: Dict[str, Any]) -> float:
    try:
        info = PPGDatasetInfo.from_dict(meta.get("dataset_info", {}))
        return info.sampling_rate
    except Exception:
        try:
            return PPGDatasetInfo.load(META_PATH).sampling_rate
        except Exception:
            return 128.0


def main() -> None:
    print("\n[PPG] Running inference demo on sample 0...\n")

    try:
        signals, _, _, _, _ = load_raw_dataset(auto_generate=True)
        result = predict_ppg_sample(signals[0])
    except Exception as exc:
        print(f"[ERROR] Inference failed: {exc}")
        sys.exit(1)

    print("  Result:")
    print(f"    status        : {result['status']}")
    print(f"    confidence    : {result['confidence']}%")
    print(f"    heart_rate    : {result['heart_rate']} BPM")
    print(f"    pulse_quality : {result['pulse_quality']}")
    print()


if __name__ == "__main__":
    main()
