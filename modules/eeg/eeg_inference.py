"""
STEP 7 — EEG inference for a single sample.

Load the trained CNN and return a clinical-style prediction dict:

    {"status": "NORMAL" | "ABNORMAL", "confidence": 97.8}
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

from modules.eeg.eeg_model import build_model
from modules.eeg.preprocessing import PreprocessConfig, preprocess_signals
from modules.eeg.utils import (
    EEGDatasetInfo,
    META_PATH,
    class_index_to_status,
    get_model_path,
    load_raw_dataset,
)


def load_eeg_model(
    model_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load trained EEG CNN and associated metadata from checkpoint.

    Returns (model, checkpoint_meta).
    """
    path = Path(model_path or get_model_path())
    if not path.exists():
        raise FileNotFoundError(
            f"No trained EEG model at {path}. Run train_eeg.py first."
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
    config: Optional[PreprocessConfig] = None,
) -> np.ndarray:
    """Ensure signal is 1D, correct length, and preprocessed."""
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
            info = EEGDatasetInfo.load(META_PATH)
            config = PreprocessConfig.from_dataset_info(info)
        except Exception:
            config = PreprocessConfig()

    processed = preprocess_signals(batch, config)
    return processed


@torch.no_grad()
def predict_eeg_sample(
    signal: Union[np.ndarray, list],
    model: Optional[torch.nn.Module] = None,
    model_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Run inference on a single EEG window.

    Parameters
    ----------
    signal : 1D array-like
        Raw or pre-segmented EEG samples.

    Returns
    -------
    dict with keys:
        status     — "NORMAL" or "ABNORMAL"
        confidence — float percentage (0–100)
        class_index — raw model class index
        probabilities — list of class probabilities
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model, meta = load_eeg_model(model_path=model_path, device=device)
    else:
        meta = {"signal_length": model.signal_length, "num_classes": model.num_classes}

    signal_length = meta["signal_length"]
    processed = _prepare_sample(signal, signal_length)

    tensor = torch.from_numpy(processed).unsqueeze(0)  # (1, 1, L)
    tensor = tensor.to(device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    class_idx = int(np.argmax(probs))
    confidence = float(probs[class_idx] * 100.0)

    num_classes = meta.get("num_classes", 2)
    if num_classes == 2:
        status = "NORMAL" if class_idx == 0 else "ABNORMAL"
    else:
        try:
            info = EEGDatasetInfo.from_dict(meta.get("dataset_info", {}))
        except Exception:
            info = EEGDatasetInfo.load(META_PATH)
        status = class_index_to_status(class_idx, info)

    return {
        "status": status,
        "confidence": round(confidence, 1),
        "class_index": class_idx,
        "probabilities": probs.tolist(),
    }


def main() -> None:
    """Demo: predict on the first sample from the loaded dataset."""
    print("\n[EEG] Running inference demo on sample 0...\n")

    try:
        signals, _, _, _ = load_raw_dataset(auto_download=True)
        result = predict_eeg_sample(signals[0])
    except Exception as exc:
        print(f"[ERROR] Inference failed: {exc}")
        sys.exit(1)

    print("  Result:")
    print(f"    status     : {result['status']}")
    print(f"    confidence : {result['confidence']}%")
    print(f"    class_idx  : {result['class_index']}")
    print()


if __name__ == "__main__":
    main()
