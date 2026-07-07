"""
STEP 2 — EEG signal preprocessing pipeline.

Reusable functions for normalization, filtering, segmentation,
label encoding, and train/test splitting. Adapts to dataset
metadata produced by inspect_dataset.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from modules.eeg.utils import (
    EEGDatasetInfo,
    META_PATH,
    raw_to_binary_labels,
)


@dataclass
class PreprocessConfig:
    """Preprocessing hyper-parameters (adapted from dataset inspection)."""

    sampling_rate: float = 173.61
    apply_bandpass: bool = True
    bandpass_low: float = 0.5
    bandpass_high: float = 45.0
    filter_order: int = 4
    normalize: str = "zscore"          # "zscore" | "minmax" | "none"
    window_size: Optional[int] = None  # None = use full signal length
    window_stride: Optional[int] = None
    binary_labels: bool = True
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42

    @classmethod
    def from_dataset_info(cls, info: EEGDatasetInfo, **overrides) -> "PreprocessConfig":
        """Build config from inspection metadata."""
        cfg = cls(
            sampling_rate=info.sampling_rate or 173.61,
            window_size=info.signal_length if info.is_pre_segmented else 256,
            window_stride=info.signal_length // 2 if not info.is_pre_segmented else None,
        )
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


def normalize_signal(
    signal: np.ndarray,
    method: str = "zscore",
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize a 1D or 2D signal array.

    Parameters
    ----------
    signal : (L,) or (N, L)
    method : "zscore", "minmax", or "none"
    """
    if method == "none":
        return signal.astype(np.float32)

    if signal.ndim == 1:
        signal = signal.reshape(1, -1)

    out = np.empty_like(signal, dtype=np.float32)
    for i, row in enumerate(signal):
        if method == "zscore":
            mu, sigma = row.mean(), row.std()
            out[i] = (row - mu) / (sigma + eps)
        elif method == "minmax":
            lo, hi = row.min(), row.max()
            out[i] = (row - lo) / (hi - lo + eps)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    return out.squeeze() if out.shape[0] == 1 and signal.ndim == 1 else out


def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: float,
    low: float = 0.5,
    high: float = 45.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply Butterworth band-pass filter along the last axis.

    Skips filtering when Nyquist constraints cannot be met.
    """
    if signal.ndim == 1:
        signal = signal.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    nyquist = 0.5 * sampling_rate
    if high >= nyquist:
        high = nyquist * 0.95
    if low <= 0 or low >= high:
        return signal.squeeze() if squeeze else signal

    try:
        b, a = butter(order, [low / nyquist, high / nyquist], btype="band")
        filtered = np.array(
            [filtfilt(b, a, row) for row in signal],
            dtype=np.float32,
        )
    except Exception:
        filtered = signal.astype(np.float32)

    return filtered.squeeze() if squeeze else filtered


def segment_windows(
    signal: np.ndarray,
    window_size: int,
    stride: int,
    label: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment a long 1D signal into fixed-length windows.

    Returns (windows, labels) where labels repeat `label` for each window.
    """
    if signal.ndim > 1:
        raise ValueError("segment_windows expects a 1D signal")

    if len(signal) < window_size:
        padded = np.pad(signal, (0, window_size - len(signal)), mode="edge")
        windows = padded.reshape(1, -1)
    else:
        starts = range(0, len(signal) - window_size + 1, stride)
        windows = np.array([signal[s : s + window_size] for s in starts], dtype=np.float32)

    labels = np.full(len(windows), label if label is not None else 0, dtype=np.int64)
    return windows, labels


def encode_labels(
    labels: np.ndarray,
    binary: bool = True,
    mapping: Optional[dict] = None,
) -> np.ndarray:
    """
    Encode string/int labels to integer class indices.

    When binary=True, maps to 0=NORMAL, 1=ABNORMAL using clinical mapping.
    """
    if binary:
        return raw_to_binary_labels(labels, mapping)

    encoder = LabelEncoder()
    return encoder.fit_transform(labels.astype(str)).astype(np.int64)


def preprocess_signals(
    signals: np.ndarray,
    config: PreprocessConfig,
) -> np.ndarray:
    """Apply filtering and normalization to a batch of signals."""
    out = signals.astype(np.float32)

    if config.apply_bandpass and config.sampling_rate:
        out = bandpass_filter(
            out,
            sampling_rate=config.sampling_rate,
            low=config.bandpass_low,
            high=config.bandpass_high,
            order=config.filter_order,
        )

    out = normalize_signal(out, method=config.normalize)
    return out


def preprocess_pipeline(
    signals: np.ndarray,
    labels: np.ndarray,
    info: Optional[EEGDatasetInfo] = None,
    config: Optional[PreprocessConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, PreprocessConfig]:
    """
    Full preprocessing pipeline adapting to dataset format.

    Returns processed signals, encoded labels, and the config used.
    """
    if info is None:
        info = EEGDatasetInfo.load(META_PATH)

    if config is None:
        config = PreprocessConfig.from_dataset_info(info)

    # Handle long continuous recordings via sliding windows
    if not info.is_pre_segmented and signals.ndim == 2 and signals.shape[0] == 1:
        ws = config.window_size or 256
        stride = config.window_stride or ws // 2
        windows, win_labels = segment_windows(signals[0], ws, stride, label=int(labels[0]))
        signals = windows
        labels = win_labels

    processed = preprocess_signals(signals, config)
    encoded = encode_labels(labels, binary=config.binary_labels, mapping=info.binary_mapping)

    # Replace NaN from bad rows
    if np.isnan(processed).any():
        processed = np.nan_to_num(processed, nan=0.0)

    return processed, encoded, config


def split_train_val_test(
    signals: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, ...]:
    """
    Stratified split into train / validation / test sets.

    Returns X_train, X_val, X_test, y_train, y_val, y_test
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        signals, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # val_size relative to original dataset → adjust for remaining fraction
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_fraction,
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
