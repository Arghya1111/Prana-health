"""
STEP 7 — Thermal visualization: heatmaps, histograms, hotspots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from modules.thermal.clinical_rules import class_index_to_status
from modules.thermal.preprocessing import detect_hotspots, enhance_contrast, resize_image


def plot_temperature_histogram(image: np.ndarray, ax) -> None:
    """Plot pixel intensity histogram as temperature proxy."""
    flat = np.asarray(image).flatten()
    ax.hist(flat, bins=40, color="#E74C3C", alpha=0.75, edgecolor="white")
    ax.set_xlabel("Normalized Temperature")
    ax.set_ylabel("Pixel Count")
    ax.set_title("Temperature Histogram")
    ax.grid(True, alpha=0.3)


def visualize_thermal_analysis(
    image: np.ndarray,
    prediction: Optional[Dict[str, Any]] = None,
    save_path: Optional[Path] = None,
    image_size: int = 128,
) -> Path:
    """
    Display original, heatmap, histogram, and hotspot overlay.

    Returns path to saved figure.
    """
    img = resize_image(np.asarray(image, dtype=np.float32).squeeze(), image_size)
    img = enhance_contrast(img)
    hotspots = detect_hotspots(img)
    n_hotspots = int(hotspots.sum() > 0 and max(1, hotspots.sum() // 20))

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Original Thermal Image")
    axes[0, 0].axis("off")

    im = axes[0, 1].imshow(img, cmap="inferno")
    axes[0, 1].set_title("Thermal Heatmap")
    axes[0, 1].axis("off")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046)

    plot_temperature_histogram(img, axes[1, 0])

    overlay = np.stack([img, img, img], axis=-1)
    overlay[hotspots > 0] = [1.0, 0.2, 0.2]
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(f"Detected Hotspots (~{int(hotspots.sum())} px)")
    axes[1, 1].axis("off")

    if prediction:
        title = (
            f"Prediction: {prediction.get('status', '?')}  "
            f"({prediction.get('confidence', 0):.1f}%)  "
            f"Risk: {prediction.get('thermal_risk', '?')}"
        )
        fig.suptitle(title, fontsize=12, y=1.02)

    fig.tight_layout()
    out = Path(save_path or Path(__file__).parent / "model" / "thermal_analysis.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    """Demo visualization on a synthetic image."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from modules.thermal.utils import generate_synthetic_thermal_image

    img = generate_synthetic_thermal_image(class_idx=2)
    pred = {"status": "HIGH", "confidence": 92.5, "thermal_risk": "HIGH"}
    path = visualize_thermal_analysis(img, pred)
    print(f"Visualization saved -> {path}")


if __name__ == "__main__":
    main()
