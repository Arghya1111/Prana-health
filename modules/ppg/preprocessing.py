"""
STEP 2 — PPG signal preprocessing pipeline.

Baseline removal, filtering, normalization, segmentation,
artifact detection, and train/test splitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split

from modules.ppg.utils import META_PATH, PPGDatasetInfo, raw_to_binary_labels


@dataclass
class PreprocessConfig:
    """Preprocessing hyper-parameters adapted from dataset inspection."""

    sampling_rate: float = 128.0
    apply_baseline_removal: bool = True
    moving_avg_window: int = 5
    apply_bandpass: bool = True
    bandpass_low: float = 0.5
    bandpass_high: float = 8.0      # PPG fundamental ~0.5-4 Hz; allow margin
    filter_order: int = 4
    normalize: str = "zscore"
    window_size: Optional[int] = None
    window_stride: Optional[int] = None
    artifact_threshold: float = 5.0   # z-score spike threshold
    binary_labels: bool = True
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42

    @classmethod
    def from_dataset_info(cls, info: PPGDatasetInfo, **overrides) -> "PreprocessConfig":
        cfg = cls(
            sampling_rate=info.sampling_rate,
            window_size=info.signal_length if info.is_pre_segmented else 512,
            window_stride=info.signal_length // 2 if not info.is_pre_segmented else None,
        )
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


def remove_baseline(signal: np.ndarray) -> np.ndarray:
    """Remove slow baseline drift via moving-average subtraction."""
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    out = np.empty_like(signal, dtype=np.float32)
    for i, row in enumerate(signal):
        kernel = np.ones(5) / 5.0
        baseline = np.convolve(row, kernel, mode="same")
        out[i] = row - baseline

    return out.squeeze() if squeeze else out


def moving_average_filter(signal: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply moving average smoothing."""
    if window <= 1:
        return signal.astype(np.float32)

    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    kernel = np.ones(window) / window
    out = np.array([np.convolve(row, kernel, mode="same") for row in signal], dtype=np.float32)
    return out.squeeze() if squeeze else out


def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: float,
    low: float = 0.5,
    high: float = 8.0,
    order: int = 4,
) -> np.ndarray:
    """Butterworth band-pass filter along the last axis."""
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    nyquist = 0.5 * sampling_rate
    high = min(high, nyquist * 0.95)
    if low <= 0 or low >= high:
        return signal.squeeze() if squeeze else signal.astype(np.float32)

    try:
        b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
        filtered = np.array([filtfilt(b, a, row) for row in signal], dtype=np.float32)
    except Exception:
        filtered = signal.astype(np.float32)

    return filtered.squeeze() if squeeze else filtered


def normalize_signal(
    signal: np.ndarray,
    method: str = "zscore",
    eps: float = 1e-8,
) -> np.ndarray:
    """Z-score or min-max normalization."""
    if method == "none":
        return signal.astype(np.float32)

    if signal.ndim == 1:
        signal = signal.reshape(1, -1)

    out = np.empty_like(signal, dtype=np.float32)
    for i, row in enumerate(signal):
        if method == "zscore":
            out[i] = (row - row.mean()) / (row.std() + eps)
        elif method == "minmax":
            lo, hi = row.min(), row.max()
            out[i] = (row - lo) / (hi - lo + eps)
        else:
            raise ValueError(f"Unknown normalization: {method}")

    return out


def detect_artifacts(signal: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """
    Flag samples containing amplitude spikes (motion artifacts).

    Returns boolean mask (N,) — True = clean, False = artifact.
    """
    if signal.ndim == 1:
        z = np.abs((signal - signal.mean()) / (signal.std() + 1e-8))
        return np.array([not np.any(z > threshold)])

    mask = np.ones(signal.shape[0], dtype=bool)
    for i, row in enumerate(signal):
        z = np.abs((row - row.mean()) / (row.std() + 1e-8))
        if np.any(z > threshold):
            mask[i] = False
    return mask


def remove_noise(signal: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Combined noise removal: baseline, moving average, band-pass."""
    out = signal.astype(np.float32)
    if config.apply_baseline_removal:
        out = remove_baseline(out)
    if config.moving_avg_window > 1:
        out = moving_average_filter(out, config.moving_avg_window)
    if config.apply_bandpass:
        out = bandpass_filter(
            out,
            config.sampling_rate,
            config.bandpass_low,
            config.bandpass_high,
            config.filter_order,
        )
    return out


def segment_windows(
    signal: np.ndarray,
    window_size: int,
    stride: int,
    label: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment long PPG recording into fixed windows."""
    if signal.ndim > 1:
        raise ValueError("segment_windows expects 1D input")

    if len(signal) < window_size:
        padded = np.pad(signal, (0, window_size - len(signal)), mode="edge")
        windows = padded.reshape(1, -1)
    else:
        starts = range(0, len(signal) - window_size + 1, stride)
        windows = np.array([signal[s: s + window_size] for s in starts], dtype=np.float32)

    labels = np.full(len(windows), label if label is not None else 0, dtype=np.int64)
    return windows, labels


def preprocess_signals(signals: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Full per-window preprocessing chain."""
    out = remove_noise(signals, config)
    out = normalize_signal(out, method=config.normalize)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def preprocess_pipeline(
    signals: np.ndarray,
    labels: np.ndarray,
    info: Optional[PPGDatasetInfo] = None,
    config: Optional[PreprocessConfig] = None,
    drop_artifacts: bool = True,
) -> Tuple[np.ndarray, np.ndarray, PreprocessConfig]:
    """End-to-end preprocessing with optional artifact rejection."""
    if info is None:
        info = PPGDatasetInfo.load(META_PATH)
    if config is None:
        config = PreprocessConfig.from_dataset_info(info)

    if not info.is_pre_segmented and signals.ndim == 2 and signals.shape[0] == 1:
        ws = config.window_size or 512
        stride = config.window_stride or ws // 2
        windows, win_labels = segment_windows(signals[0], ws, stride, label=int(labels[0]))
        signals = windows
        labels = win_labels

    clean_mask = detect_artifacts(signals, config.artifact_threshold)
    if drop_artifacts and not clean_mask.all():
        signals = signals[clean_mask]
        labels = labels[clean_mask]

    processed = preprocess_signals(signals, config)
    encoded = raw_to_binary_labels(labels, mapping=info.binary_mapping) if config.binary_labels else labels.astype(np.int64)

    return processed, encoded, config


def split_train_val_test(
    signals: np.ndarray,
    labels: np.ndarray,
    features: Optional[np.ndarray] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, ...]:
    """Stratified train/val/test split; optionally splits feature matrix too."""
    if features is not None:
        X_temp, X_test, F_temp, F_test, y_temp, y_test = train_test_split(
            signals, features, labels,
            test_size=test_size, random_state=random_state, stratify=labels,
        )
        val_fraction = val_size / (1.0 - test_size)
        X_train, X_val, F_train, F_val, y_train, y_val = train_test_split(
            X_temp, F_temp, y_temp,
            test_size=val_fraction, random_state=random_state, stratify=y_temp,
        )
        return X_train, X_val, X_test, F_train, F_val, F_test, y_train, y_val, y_test

    X_temp, X_test, y_temp, y_test = train_test_split(
        signals, labels, test_size=test_size, random_state=random_state, stratify=labels,
    )
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, random_state=random_state, stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
