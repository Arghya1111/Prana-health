"""
STEP 1 — Automatic thermal dataset inspection.
"""

from __future__ import annotations

import sys
from pathlib import Path

_MODULE = Path(__file__).resolve().parent
if str(_MODULE.parent.parent) not in sys.path:
    sys.path.insert(0, str(_MODULE.parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from modules.thermal.utils import (
    META_PATH,
    ensure_thermal_dataset,
    inspect_dataset,
    load_raw_dataset,
)


def show_random_samples(images: np.ndarray, labels: np.ndarray, class_names: list[str], n: int = 4) -> Path:
    """Save grid of random thermal samples."""
    out = _MODULE / "dataset" / "sample_grid.png"
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(images), size=min(n, len(images)), replace=False)

    cols = 2
    rows = (len(idxs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(8, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for ax, idx in zip(axes, idxs):
        ax.imshow(images[idx], cmap="inferno")
        name = class_names[labels[idx]] if labels[idx] < len(class_names) else str(labels[idx])
        ax.set_title(f"Class: {name}")
        ax.axis("off")

    for ax in axes[len(idxs):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def print_report(info) -> None:
    print("\n" + "=" * 60)
    print("  PRANA Thermal - Dataset Inspection Report")
    print("=" * 60)
    print(f"\n  Dataset size            : {info.n_samples} images")
    print(f"  Image resolution        : {info.image_width} x {info.image_height}")
    print(f"  Number of classes       : {info.n_classes}")
    print(f"  Image format            : {info.image_format}")
    print(f"  Missing images          : {info.missing_images}")
    print(f"  Corrupted images        : {info.corrupted_images}")
    print(f"  Source directories      :")
    for d in info.source_dirs:
        print(f"    - {d}")
    print("\n  Class distribution:")
    for cls, count in sorted(info.class_distribution.items()):
        pct = 100.0 * count / info.n_samples
        print(f"    {cls:<14s} : {count:>5d}  ({pct:5.1f}%)")
    print("\n" + "=" * 60)


def main() -> None:
    print("\n[Thermal] Inspecting dataset in modules/thermal/dataset/ ...")
    try:
        images, labels, class_names, paths = load_raw_dataset(auto_generate=True)
        info = inspect_dataset(auto_generate=False)
    except Exception as exc:
        print(f"\n[ERROR] Inspection failed: {exc}")
        sys.exit(1)

    print_report(info)

    try:
        grid = show_random_samples(images, labels, class_names)
        print(f"\n  Random sample grid saved -> {grid}")
    except Exception as exc:
        print(f"\n  [WARN] Sample grid failed: {exc}")

    try:
        print(f"  Metadata saved -> {info.save(META_PATH)}")
    except Exception as exc:
        print(f"\n  [WARN] Metadata save failed: {exc}")

    print("\n[Done] Run train_thermal.py next.\n")


if __name__ == "__main__":
    main()
