"""
STEP 1 — Automatic PPG dataset inspection.

Run from project root:
    python modules/ppg/inspect_dataset.py

Prints dataset statistics and saves an example signal plot.
Metadata is written to dataset/dataset_meta.json for adaptive preprocessing.
"""

from __future__ import annotations

import sys
from pathlib import Path

_MODULE = Path(__file__).resolve().parent
if str(_MODULE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_MODULE.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from modules.ppg.utils import (
    INSPECT_PLOT_PATH,
    META_PATH,
    ensure_dataset_dir,
    inspect_dataset,
    load_raw_dataset,
)


def plot_example_signal(signals: np.ndarray, sampling_rate: float) -> Path:
    """Save example PPG waveform plot."""
    ensure_dataset_dir()
    sample = signals[0]
    t = np.arange(len(sample)) / sampling_rate

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, sample, color="#C0392B", linewidth=0.9)
    ax.set_title("Example PPG Signal (Sample 0)", fontsize=12)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (a.u.)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(INSPECT_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return INSPECT_PLOT_PATH


def print_inspection_report(info, signals: np.ndarray) -> None:
    """Pretty-print all required inspection fields."""
    print("\n" + "=" * 60)
    print("  PRANA PPG - Dataset Inspection Report")
    print("=" * 60)

    print(f"\n  Dataset shape (signals) : {signals.shape}")
    print(f"  Number of samples       : {info.n_samples}")
    print(f"  Signal length           : {info.signal_length}")
    print(f"  Sampling frequency      : {info.sampling_rate:.2f} Hz")
    print(f"  Window duration         : {info.signal_length / info.sampling_rate:.3f} s")
    print(f"  Number of subjects      : {info.n_subjects}")
    print(f"  Number of classes       : {info.n_classes}")
    print(f"  Format                  : {info.format_type}")
    print(f"  Source file(s)          : {', '.join(info.source_files)}")
    print(f"  Pre-segmented windows   : {info.is_pre_segmented}")
    if info.label_column:
        print(f"  Label column            : {info.label_column}")
    if info.subject_column:
        print(f"  Subject column          : {info.subject_column}")

    if info.columns:
        print(f"\n  Columns ({len(info.columns)} total):")
        show = info.columns[:8]
        for col in show:
            print(f"    {col}")
        if len(info.columns) > 8:
            print(f"    ... and {len(info.columns) - 8} more")

    if info.dtypes:
        print("\n  Data types:")
        dtype_counts: dict[str, int] = {}
        for dtype in info.dtypes.values():
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        for dtype, count in sorted(dtype_counts.items()):
            print(f"    {dtype:>12s} : {count:>4d} columns")
        if info.label_column and info.label_column in info.dtypes:
            print(f"    Label column dtype    : {info.dtypes[info.label_column]}")

    print("\n  Class distribution:")
    for cls, count in sorted(info.class_distribution.items(), key=lambda x: x[0]):
        pct = 100.0 * count / info.n_samples
        status = info.binary_mapping.get(cls, "?")
        print(f"    Class {cls:>8s} ({status:>8s}) : {count:>6d}  ({pct:5.1f}%)")

    print(f"\n  Missing values          : {info.missing_values}")

    print("\n  Signal statistics:")
    print(f"    Mean                    : {info.signal_mean:.4f}")
    print(f"    Std                     : {info.signal_std:.4f}")
    print(f"    Min                     : {info.signal_min:.4f}")
    print(f"    Max                     : {info.signal_max:.4f}")

    print("\n  Binary clinical mapping (for inference):")
    for raw, status in sorted(info.binary_mapping.items(), key=lambda x: x[0]):
        print(f"    Raw {raw} -> {status}")

    print("\n" + "=" * 60)


def main() -> None:
    print("\n[PPG] Inspecting dataset in modules/ppg/dataset/ ...")

    try:
        signals, labels, subjects, _, _ = load_raw_dataset(auto_generate=True)
        info = inspect_dataset(auto_generate=False)
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

    print("\n[Done] Inspection complete. Run train_ppg.py to train the CNN.\n")


if __name__ == "__main__":
    main()
