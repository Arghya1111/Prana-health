"""
STEP 1 — Automatic SpO2 dataset inspection.

Run from project root:
    python modules/spo2/inspect_dataset.py

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

from modules.spo2.clinical_rules import spo2_to_status
from modules.spo2.utils import (
    INSPECT_PLOT_PATH,
    META_PATH,
    ensure_dataset_dir,
    inspect_dataset,
    load_raw_dataset,
)


def plot_example_signal(signals: np.ndarray, sampling_rate: float) -> Path:
    """Save synchronized PPG + SpO2 waveform plot."""
    ensure_dataset_dir()
    ppg = signals[0, 0]
    spo2 = signals[0, 1]
    if np.max(spo2) <= 1.5:
        spo2 = spo2 * 100
    t = np.arange(len(ppg)) / sampling_rate

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(t, ppg, color="#C0392B", linewidth=0.9)
    ax1.set_ylabel("PPG (norm.)")
    ax1.set_title("Example Synchronized PPG + SpO2 (Sample 0)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, spo2, color="#2980B9", linewidth=0.9)
    ax2.axhline(95, color="green", linestyle="--", alpha=0.6, label="95%")
    ax2.axhline(90, color="orange", linestyle="--", alpha=0.6, label="90%")
    ax2.axhline(85, color="red", linestyle="--", alpha=0.6, label="85%")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("SpO2 (%)")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(INSPECT_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return INSPECT_PLOT_PATH


def print_inspection_report(info, signals: np.ndarray, spo2_ref: np.ndarray) -> None:
    """Pretty-print all required inspection fields."""
    print("\n" + "=" * 60)
    print("  PRANA SpO2 - Dataset Inspection Report")
    print("=" * 60)

    print(f"\n  Dataset shape (signals) : {signals.shape}")
    print(f"  Number of samples       : {info.n_samples}")
    print(f"  Signal length           : {info.signal_length}")
    print(f"  Sampling frequency      : {info.sampling_rate:.2f} Hz")
    print(f"  Window duration         : {info.signal_length / info.sampling_rate:.3f} s")
    print(f"  Channels                : {info.n_channels} (PPG + SpO2)")
    print(f"  Number of subjects      : {info.n_subjects}")
    print(f"  Task type               : {info.task_type}")
    print(f"  Number of classes       : {info.n_classes}")
    print(f"  Format                  : {info.format_type}")
    print(f"  Source file(s)          : {', '.join(info.source_files)}")
    if info.label_column:
        print(f"  Label column            : {info.label_column}")
    if info.spo2_column:
        print(f"  SpO2 reference column   : {info.spo2_column}")

    if info.columns:
        print(f"\n  Columns ({len(info.columns)} total):")
        for col in info.columns[:8]:
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

    print(f"\n  SpO2 range              : {info.spo2_min:.1f}% - {info.spo2_max:.1f}%")
    print(f"  SpO2 mean               : {info.spo2_mean:.1f}%")
    print(f"  Missing values          : {info.missing_values}")

    print("\n  Class distribution:")
    for cls, count in sorted(info.class_distribution.items(), key=lambda x: x[0]):
        pct = 100.0 * count / info.n_samples
        print(f"    Class {cls:>4s} : {count:>6d}  ({pct:5.1f}%)")

    print("\n  Clinical status breakdown (from SpO2 reference):")
    status_counts: dict[str, int] = {}
    for s in spo2_ref:
        st = spo2_to_status(float(s))
        status_counts[st] = status_counts.get(st, 0) + 1
    for st, count in sorted(status_counts.items()):
        print(f"    {st:<10s} : {count}")

    print("\n  Signal statistics:")
    print(f"    Mean                    : {info.signal_mean:.4f}")
    print(f"    Std                     : {info.signal_std:.4f}")

    print("\n" + "=" * 60)


def main() -> None:
    print("\n[SpO2] Inspecting dataset in modules/spo2/dataset/ ...")

    try:
        signals, labels, subjects, spo2_ref, _, _ = load_raw_dataset(auto_generate=True)
        info = inspect_dataset(auto_generate=False)
    except Exception as exc:
        print(f"\n[ERROR] Dataset inspection failed: {exc}")
        sys.exit(1)

    print_inspection_report(info, signals, spo2_ref)

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

    print("\n[Done] Inspection complete. Run train_spo2.py to train the CNN.\n")


if __name__ == "__main__":
    main()
