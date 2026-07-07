"""
STEP 2 — SpO2 / PPG signal preprocessing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split

from modules.spo2.utils import META_PATH, SpO2DatasetInfo, encode_labels


@dataclass
class PreprocessConfig:
    sampling_rate: float = 64.0
    normalize: str = "zscore"
    moving_avg_window: int = 5
    apply_bandpass_ppg: bool = True
    bandpass_low: float = 0.5
    bandpass_high: float = 8.0
    filter_order: int = 4
    artifact_threshold: float = 5.0
    window_size: Optional[int] = None
    window_stride: Optional[int] = None
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42

    @classmethod
    def from_dataset_info(cls, info: SpO2DatasetInfo, **overrides) -> "PreprocessConfig":
        cfg = cls(
            sampling_rate=info.sampling_rate,
            window_size=info.signal_length,
        )
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


def handle_missing_values(signal: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Replace NaN/Inf with fill value or linear interpolation."""
    out = signal.astype(np.float32).copy()
    if out.ndim == 1:
        mask = ~np.isfinite(out)
        if mask.any():
            idx = np.arange(len(out))
            valid = ~mask
            if valid.any():
                out[mask] = np.interp(idx[mask], idx[valid], out[valid])
            else:
                out[mask] = fill
        return out

    for i in range(out.shape[0]):
        for ch in range(out.shape[1]):
            row = out[i, ch]
            mask = ~np.isfinite(row)
            if mask.any():
                idx = np.arange(len(row))
                valid = ~mask
                if valid.any():
                    out[i, ch, mask] = np.interp(idx[mask], idx[valid], row[valid])
                else:
                    out[i, ch, mask] = fill
    return out


def moving_average_filter(signal: np.ndarray, window: int = 5) -> np.ndarray:
    if window <= 1:
        return signal.astype(np.float32)
    if signal.ndim == 1:
        k = np.ones(window) / window
        return np.convolve(signal, k, mode="same").astype(np.float32)

    out = signal.copy()
    for ch in range(signal.shape[1]):
        k = np.ones(window) / window
        for i in range(signal.shape[0]):
            out[i, ch] = np.convolve(signal[i, ch], k, mode="same")
    return out.astype(np.float32)


def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: float,
    low: float = 0.5,
    high: float = 8.0,
    order: int = 4,
) -> np.ndarray:
    """Apply band-pass to PPG channel only."""
    nyq = 0.5 * sampling_rate
    high = min(high, nyq * 0.95)
    if low <= 0 or low >= high:
        return signal

    try:
        b, a = butter(order, [low / nyq, high / nyq], btype="band")
        if signal.ndim == 1:
            return filtfilt(b, a, signal).astype(np.float32)
        out = signal.copy()
        for i in range(signal.shape[0]):
            out[i] = filtfilt(b, a, signal[i])
        return out.astype(np.float32)
    except Exception:
        return signal.astype(np.float32)


def normalize_signal(signal: np.ndarray, method: str = "zscore", eps: float = 1e-8) -> np.ndarray:
    if method == "none":
        return signal.astype(np.float32)

    out = signal.astype(np.float32).copy()
    if out.ndim == 1:
        rows = out.reshape(1, -1)
        squeeze = True
    elif out.ndim == 2:
        rows = out
        squeeze = False
    else:
        for i in range(out.shape[0]):
            for ch in range(out.shape[2] if out.ndim == 3 else 1):
                if out.ndim == 3:
                    row = out[i, ch]
                    if method == "zscore":
                        out[i, ch] = (row - row.mean()) / (row.std() + eps)
                    elif method == "minmax":
                        lo, hi = row.min(), row.max()
                        out[i, ch] = (row - lo) / (hi - lo + eps)
        return out

    for i, row in enumerate(rows):
        if method == "zscore":
            rows[i] = (row - row.mean()) / (row.std() + eps)
        elif method == "minmax":
            lo, hi = row.min(), row.max()
            rows[i] = (row - lo) / (hi - lo + eps)
    return rows.squeeze() if squeeze else rows


def detect_peaks(ppg: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Detect systolic peaks in a 1D PPG segment."""
    ppg = np.asarray(ppg, dtype=np.float64).squeeze()
    if len(ppg) < 10:
        return np.array([], dtype=int)
    min_dist = max(1, int(sampling_rate * 0.4))
    prominence = max(0.05, 0.2 * np.std(ppg))
    peaks, _ = find_peaks(ppg, distance=min_dist, prominence=prominence)
    return peaks


def remove_artifacts(signals: np.ndarray, threshold: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Return (clean_signals, boolean mask)."""
    mask = np.ones(signals.shape[0], dtype=bool)
    for i in range(signals.shape[0]):
        ppg = signals[i, 0]
        z = np.abs((ppg - ppg.mean()) / (ppg.std() + 1e-8))
        if np.any(z > threshold):
            mask[i] = False
    return signals[mask], mask


def segment_windows(
    signal: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Segment (2, L) recording into (N, 2, window_size)."""
    if signal.ndim != 2:
        raise ValueError("Expected (channels, length)")
    length = signal.shape[1]
    if length < window_size:
        pad = window_size - length
        signal = np.pad(signal, ((0, 0), (0, pad)), mode="edge")
        length = window_size
    starts = range(0, length - window_size + 1, stride)
    return np.array([signal[:, s: s + window_size] for s in starts], dtype=np.float32)


def preprocess_signals(signals: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """
    Preprocess (N, 2, L): channel 0 PPG filtered, channel 1 SpO2 smoothed.
    """
    out = handle_missing_values(signals)
    n = out.shape[0]

    for i in range(n):
        ppg = out[i, 0].copy()
        spo2 = out[i, 1].copy()

        # Preserve SpO2 percentage scale (do not z-score saturation channel)
        if np.max(spo2) > 10:
            spo2 = spo2 / 100.0

        ppg = moving_average_filter(ppg, config.moving_avg_window)
        if config.apply_bandpass_ppg:
            ppg = bandpass_filter(
                ppg, config.sampling_rate,
                config.bandpass_low, config.bandpass_high, config.filter_order,
            )
        spo2 = moving_average_filter(spo2, max(3, config.moving_avg_window))

        out[i, 0] = normalize_signal(ppg, config.normalize)
        out[i, 1] = np.clip(spo2, 0.0, 1.0).astype(np.float32)

    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def preprocess_pipeline(
    signals: np.ndarray,
    labels: np.ndarray,
    spo2_ref: np.ndarray,
    info: Optional[SpO2DatasetInfo] = None,
    config: Optional[PreprocessConfig] = None,
    drop_artifacts: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PreprocessConfig]:
    if info is None:
        info = SpO2DatasetInfo.load(META_PATH)
    if config is None:
        config = PreprocessConfig.from_dataset_info(info)

    if drop_artifacts:
        signals, mask = remove_artifacts(signals, config.artifact_threshold)
        labels = labels[mask]
        spo2_ref = spo2_ref[mask]

    processed = preprocess_signals(signals, config)
    encoded = encode_labels(labels, task_type=info.task_type)
    return processed, encoded, spo2_ref, config


def split_train_val_test(
    signals: np.ndarray,
    labels: np.ndarray,
    features: Optional[np.ndarray] = None,
    spo2_ref: Optional[np.ndarray] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, ...]:
    """Stratified train/val/test split with optional features and SpO2 reference."""
    if features is not None and spo2_ref is not None:
        X_temp, X_test, F_temp, F_test, S_temp, S_test, y_temp, y_test = train_test_split(
            signals, features, spo2_ref, labels,
            test_size=test_size, random_state=random_state, stratify=labels,
        )
        val_fraction = val_size / (1.0 - test_size)
        X_train, X_val, F_train, F_val, S_train, S_val, y_train, y_val = train_test_split(
            X_temp, F_temp, S_temp, y_temp,
            test_size=val_fraction, random_state=random_state, stratify=y_temp,
        )
        return X_train, X_val, X_test, F_train, F_val, F_test, S_train, S_val, S_test, y_train, y_val, y_test

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
