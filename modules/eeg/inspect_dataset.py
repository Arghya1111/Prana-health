"""
STEP 1 — Automatic EEG dataset inspection.

Run from project root:
    python modules/eeg/inspect_dataset.py

Prints dataset statistics and saves an example signal plot.
Metadata is written to dataset/dataset_meta.json for adaptive preprocessing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as script from project root or module directory
_MODULE = Path(__file__).resolve().parent
if str(_MODULE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_MODULE.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from modules.eeg.utils import (
    INSPECT_PLOT_PATH,
    META_PATH,
    ensure_dataset_dir,
    inspect_dataset,
    load_raw_dataset,
)


def plot_example_signal(signals: np.ndarray, sampling_rate: float | None) -> Path:
    """Plot the first sample and save to dataset/example_signal.png."""
    ensure_dataset_dir()
    sample = signals[0]
    n = len(sample)
    if sampling_rate:
        t = np.arange(n) / sampling_rate
        xlabel = "Time (s)"
    else:
        t = np.arange(n)
        xlabel = "Sample index"

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, sample, color="#0F4C81", linewidth=0.8)
    ax.set_title("Example EEG Signal (Sample 0)", fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude (a.u.)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(INSPECT_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return INSPECT_PLOT_PATH


def print_inspection_report(info, signals: np.ndarray) -> None:
    """Pretty-print all required inspection fields."""
    print("\n" + "=" * 60)
    print("  PRANA EEG - Dataset Inspection Report")
    print("=" * 60)

    print(f"\n  Dataset shape (signals) : {signals.shape}")
    print(f"  Number of samples       : {info.n_samples}")
    print(f"  Signal length           : {info.signal_length}")
    print(f"  Channels (used)         : {info.n_channels}")
    if info.sampling_rate:
        duration = info.signal_length / info.sampling_rate
        print(f"  Sampling frequency      : {info.sampling_rate:.2f} Hz")
        print(f"  Window duration         : {duration:.3f} s")
    else:
        print("  Sampling frequency      : not available (using sample index)")
    print(f"  Number of classes       : {info.n_classes}")
    print(f"  Format                  : {info.format_type}")
    print(f"  Source file(s)          : {', '.join(info.source_files)}")
    print(f"  Pre-segmented windows   : {info.is_pre_segmented}")
    if info.label_column:
        print(f"  Label column            : {info.label_column}")

    print("\n  Class distribution:")
    for cls, count in sorted(info.class_distribution.items(), key=lambda x: x[0]):
        pct = 100.0 * count / info.n_samples
        print(f"    Class {cls:>4s} : {count:>6d}  ({pct:5.1f}%)")

    print(f"\n  Missing values          : {info.missing_values}")

    print("\n  Binary clinical mapping (for inference):")
    for raw, status in sorted(info.binary_mapping.items(), key=lambda x: x[0]):
        print(f"    Raw {raw} -> {status}")

    print("\n" + "=" * 60)


def main() -> None:
    print("\n[EEG] Inspecting dataset in modules/eeg/dataset/ ...")

    try:
        signals, labels, fmt, sources = load_raw_dataset(auto_download=True)
        info = inspect_dataset(auto_download=False)
    except Exception as exc:
        print(f"\n[ERROR] Dataset inspection failed: {exc}")
        sys.exit(1)

    print_inspection_report(info, signals)

    try:
        plot_path = plot_example_signal(signals, info.sampling_rate)
        print(f"\n  Example signal plot saved -> {plot_path}")
    except Exception as exc:
        print(f"\n  [WARN] Could not save plot: {exc}")

    try:
        meta_path = info.save(META_PATH)
        print(f"  Metadata saved -> {meta_path}")
    except Exception as exc:
        print(f"\n  [WARN] Could not save metadata: {exc}")

    print("\n[Done] Inspection complete. Run train_eeg.py to train the CNN.\n")


if __name__ == "__main__":
    main()
