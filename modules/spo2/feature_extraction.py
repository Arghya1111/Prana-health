"""
STEP 3 — SpO2 physiological feature extraction.

Extracts oxygen saturation metrics and pulse morphology features.
Returns features as NumPy arrays.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.stats import entropy

from modules.spo2.preprocessing import detect_peaks

FEATURE_NAMES: List[str] = [
    "average_spo2",
    "minimum_spo2",
    "maximum_spo2",
    "spo2_variability",
    "signal_mean",
    "signal_standard_deviation",
    "signal_entropy",
    "peak_count",
    "pulse_interval",
    "pulse_amplitude",
]


def _denormalize_spo2(spo2: np.ndarray) -> np.ndarray:
    """Convert normalized SpO2 channel back to percentage scale."""
    if np.max(spo2) <= 1.5:
        return spo2 * 100.0
    if np.max(spo2) <= 20:
        return spo2 + 90.0
    return spo2.copy()


def extract_spo2_features(
    signal: np.ndarray,
    sampling_rate: float = 64.0,
) -> Dict[str, float]:
    """
    Extract physiological features from synchronized (2, L) PPG + SpO2 window.

    signal[0] = PPG, signal[1] = SpO2 trace (%)
    """
    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim == 1:
        ppg = sig
        spo2_pct = np.full_like(sig, 97.0)
    else:
        ppg = sig[0]
        spo2_pct = _denormalize_spo2(sig[1])

    peaks = detect_peaks(ppg, sampling_rate)
    if len(peaks) >= 2:
        intervals = np.diff(peaks) / sampling_rate
        pulse_interval = float(np.mean(intervals))
        pulse_amplitude = float(np.mean(ppg[peaks] - np.min(ppg)))
    else:
        pulse_interval = 0.0
        pulse_amplitude = float(np.max(ppg) - np.min(ppg))

    hist, _ = np.histogram(ppg, bins=32, density=True)
    hist = hist[hist > 0]

    return {
        "average_spo2": round(float(np.mean(spo2_pct)), 2),
        "minimum_spo2": round(float(np.min(spo2_pct)), 2),
        "maximum_spo2": round(float(np.max(spo2_pct)), 2),
        "spo2_variability": round(float(np.std(spo2_pct)), 4),
        "signal_mean": round(float(np.mean(ppg)), 4),
        "signal_standard_deviation": round(float(np.std(ppg)), 4),
        "signal_entropy": round(float(entropy(hist)) if hist.size else 0.0, 4),
        "peak_count": float(len(peaks)),
        "pulse_interval": round(pulse_interval, 4),
        "pulse_amplitude": round(pulse_amplitude, 4),
    }


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to ordered float32 vector."""
    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)


def extract_batch_features(signals: np.ndarray, sampling_rate: float = 64.0) -> np.ndarray:
    """
    Extract features for a batch of signals.

    Returns (N, len(FEATURE_NAMES)) array.
    """
    vectors = [
        features_to_vector(extract_spo2_features(s, sampling_rate))
        for s in signals
    ]
    return np.stack(vectors, axis=0)
