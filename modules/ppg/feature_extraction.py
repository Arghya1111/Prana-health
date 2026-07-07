"""
STEP 3 — Physiological feature extraction from PPG signals.

Extracts heart rate, pulse rate variability, morphology metrics,
and statistical descriptors. Returns a structured feature dictionary.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import entropy, kurtosis, skew

# Ordered feature names for tensor construction
FEATURE_NAMES: List[str] = [
    "heart_rate",
    "pulse_rate_variability",
    "peak_to_peak_interval",
    "pulse_width",
    "pulse_amplitude",
    "pulse_energy",
    "signal_entropy",
    "peak_count",
    "skewness",
    "kurtosis",
    "mean",
    "standard_deviation",
]


def _detect_peaks(
    signal: np.ndarray,
    sampling_rate: float,
    min_hr: float = 30.0,
    max_hr: float = 200.0,
) -> np.ndarray:
    """
    Simple peak detection for PPG pulsatile component.

    Uses adaptive threshold based on signal statistics.
    """
    signal = np.asarray(signal, dtype=np.float64).squeeze()
    if len(signal) < 10:
        return np.array([], dtype=int)

    # Smooth lightly for peak finding
    kernel = np.ones(5) / 5.0
    smoothed = np.convolve(signal, kernel, mode="same")

    min_distance = max(1, int(sampling_rate * 60.0 / max_hr))
    threshold = np.mean(smoothed) + 0.25 * np.std(smoothed)

    peaks: list[int] = []
    i = 1
    while i < len(smoothed) - 1:
        if (
            smoothed[i] > threshold
            and smoothed[i] >= smoothed[i - 1]
            and smoothed[i] >= smoothed[i + 1]
        ):
            if not peaks or (i - peaks[-1]) >= min_distance:
                peaks.append(i)
                i += min_distance
                continue
        i += 1

    return np.array(peaks, dtype=int)


def _pulse_width_at_half_max(
    signal: np.ndarray,
    peak_idx: int,
    sampling_rate: float,
) -> float:
    """Estimate pulse width (seconds) at half-maximum around a peak."""
    if peak_idx <= 0 or peak_idx >= len(signal) - 1:
        return 0.0

    peak_val = signal[peak_idx]
    half = peak_val - 0.5 * (peak_val - np.min(signal[max(0, peak_idx - 50): peak_idx + 1]))

    left = peak_idx
    while left > 0 and signal[left] > half:
        left -= 1

    right = peak_idx
    while right < len(signal) - 1 and signal[right] > half:
        right += 1

    return (right - left) / sampling_rate


def extract_ppg_features(
    signal: np.ndarray,
    sampling_rate: float = 128.0,
) -> Dict[str, float]:
    """
    Extract physiological and statistical features from one PPG window.

    Returns a dictionary with all FEATURE_NAMES keys.
    """
    signal = np.asarray(signal, dtype=np.float64).squeeze()
    if signal.size == 0:
        return {name: 0.0 for name in FEATURE_NAMES}

    peaks = _detect_peaks(signal, sampling_rate)

    # Inter-beat intervals (seconds)
    if len(peaks) >= 2:
        ibi = np.diff(peaks) / sampling_rate
        mean_ibi = float(np.mean(ibi))
        heart_rate = 60.0 / mean_ibi if mean_ibi > 0 else 0.0
        prv = float(np.std(ibi))
        peak_to_peak = mean_ibi
    else:
        heart_rate = 0.0
        prv = 0.0
        peak_to_peak = 0.0

    if len(peaks) >= 1:
        amplitudes = signal[peaks]
        pulse_amplitude = float(np.mean(amplitudes - np.min(signal)))
        widths = [_pulse_width_at_half_max(signal, p, sampling_rate) for p in peaks]
        pulse_width = float(np.mean(widths)) if widths else 0.0
    else:
        pulse_amplitude = float(np.max(signal) - np.min(signal))
        pulse_width = 0.0

    pulse_energy = float(np.sum(signal ** 2) / len(signal))

    # Histogram-based entropy (normalized)
    hist, _ = np.histogram(signal, bins=32, density=True)
    hist = hist[hist > 0]
    signal_entropy = float(entropy(hist)) if hist.size else 0.0

    return {
        "heart_rate": round(heart_rate, 2),
        "pulse_rate_variability": round(prv, 4),
        "peak_to_peak_interval": round(peak_to_peak, 4),
        "pulse_width": round(pulse_width, 4),
        "pulse_amplitude": round(pulse_amplitude, 4),
        "pulse_energy": round(pulse_energy, 4),
        "signal_entropy": round(signal_entropy, 4),
        "peak_count": float(len(peaks)),
        "skewness": round(float(skew(signal)), 4),
        "kurtosis": round(float(kurtosis(signal)), 4),
        "mean": round(float(np.mean(signal)), 4),
        "standard_deviation": round(float(np.std(signal)), 4),
    }


def features_to_vector(features: Dict[str, float]) -> np.ndarray:
    """Convert feature dict to ordered float32 vector."""
    return np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)


def extract_batch_features(
    signals: np.ndarray,
    sampling_rate: float = 128.0,
) -> np.ndarray:
    """
    Extract features for a batch of signals.

    Returns (N, len(FEATURE_NAMES)) array.
    """
    vectors = []
    for sig in signals:
        feat = extract_ppg_features(sig, sampling_rate)
        vectors.append(features_to_vector(feat))
    return np.stack(vectors, axis=0)


def assess_pulse_quality(
    signal: np.ndarray,
    sampling_rate: float = 128.0,
) -> str:
    """
    Rate pulse quality as GOOD, FAIR, or POOR.

    Based on SNR estimate and peak detection reliability.
    """
    signal = np.asarray(signal, dtype=np.float64).squeeze()
    peaks = _detect_peaks(signal, sampling_rate)
    duration_sec = len(signal) / sampling_rate
    expected_peaks = duration_sec * 1.2  # ~72 BPM baseline

    snr = float(np.std(signal) / (np.std(np.diff(signal)) + 1e-8))

    if len(peaks) >= max(2, int(expected_peaks * 0.5)) and snr > 0.8:
        return "GOOD"
    if len(peaks) >= 1 and snr > 0.4:
        return "FAIR"
    return "POOR"
