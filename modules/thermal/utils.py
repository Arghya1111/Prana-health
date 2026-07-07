"""
Shared utilities for the PRANA Thermal module.

Dataset discovery, synthetic thermal image generation, and metadata.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from modules.thermal.clinical_rules import CLASS_LABELS

MODULE_ROOT = Path(__file__).resolve().parent
DATASET_DIR = MODULE_ROOT / "dataset"
MODEL_DIR = MODULE_ROOT / "model"
DEFAULT_MODEL_PATH = MODEL_DIR / "thermal_model.pth"
META_PATH = DATASET_DIR / "dataset_meta.json"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DEFAULT_IMAGE_SIZE = 128
DEFAULT_NUM_CLASSES = 2  # normal vs abnormal; set 4 for full severity tiers


@dataclass
class ThermalDatasetInfo:
    n_samples: int
    image_width: int
    image_height: int
    n_classes: int
    class_names: List[str]
    class_distribution: Dict[str, int]
    image_format: str
    missing_images: int
    corrupted_images: int
    source_dirs: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThermalDatasetInfo":
        return cls(**data)

    def save(self, path: Optional[Path] = None) -> Path:
        out = Path(path or META_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return out

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ThermalDatasetInfo":
        src = Path(path or META_PATH)
        if not src.exists():
            raise FileNotFoundError(f"Metadata not found at {src}. Run inspect_dataset.py first.")
        with open(src, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def ensure_dataset_dir() -> Path:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    return DATASET_DIR


def ensure_model_dir() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR


def get_model_path() -> Path:
    ensure_model_dir()
    return DEFAULT_MODEL_PATH


def discover_image_paths() -> List[Tuple[Path, int, str]]:
    """
    Scan dataset/ for class-folder layout.

    Returns list of (path, label_index, class_name).
    """
    ensure_dataset_dir()
    entries: List[Tuple[Path, int, str]] = []
    class_dirs = sorted(
        [d for d in DATASET_DIR.iterdir() if d.is_dir() and d.name != "__pycache__"],
        key=lambda d: CLASS_LABELS.index(d.name.lower()) if d.name.lower() in CLASS_LABELS else 999,
    )

    if not class_dirs:
        return entries

    for label_idx, class_dir in enumerate(class_dirs):
        class_name = class_dir.name.lower()
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                entries.append((path, label_idx, class_name))
    return entries


def dataset_is_empty() -> bool:
    return len(discover_image_paths()) == 0


def _save_image_array(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr_u8 = np.clip(arr * 255, 0, 255).astype(np.uint8)
    if HAS_CV2:
        cv2.imwrite(str(path), arr_u8)
    elif HAS_PIL:
        Image.fromarray(arr_u8).save(path)
    else:
        raise RuntimeError("Install opencv-python or Pillow to save thermal images.")


def load_image(path: Path, size: Optional[int] = None) -> np.ndarray:
    """Load grayscale thermal image as float32 array in [0, 1]."""
    if HAS_CV2:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Corrupted or unreadable image: {path}")
        if size:
            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 255.0

    if HAS_PIL:
        img = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        if size:
            img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((size, size))) / 255.0
        return img

    raise RuntimeError("Install opencv-python or Pillow to load thermal images.")


def generate_synthetic_thermal_image(
    class_idx: int,
    size: int = DEFAULT_IMAGE_SIZE,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a pseudo-thermal grayscale image.

    class 0 normal     — uniform warm body gradient
    class 1 inflammation — localized hotspot
    class 2 fever      — elevated global temperature + hotspots
    class 3 severe     — multiple intense hotspots
    """
    if rng is None:
        rng = np.random.default_rng()

    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size / 2 + rng.uniform(-5, 5), size / 2 + rng.uniform(-5, 5)
    base = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (size / 3) ** 2))

    if class_idx == 0:
        img = 0.35 + 0.15 * base + rng.normal(0, 0.02, (size, size))
    elif class_idx == 1:
        img = 0.35 + 0.12 * base + rng.normal(0, 0.02, (size, size))
        hx, hy = int(rng.integers(size // 4, 3 * size // 4)), int(rng.integers(size // 4, 3 * size // 4))
        img += 0.25 * np.exp(-((x - hx) ** 2 + (y - hy) ** 2) / (2 * (size / 10) ** 2))
    elif class_idx == 2:
        img = 0.50 + 0.20 * base + rng.normal(0, 0.03, (size, size))
        for _ in range(2):
            hx, hy = int(rng.integers(0, size)), int(rng.integers(0, size))
            img += 0.15 * np.exp(-((x - hx) ** 2 + (y - hy) ** 2) / (2 * (size / 12) ** 2))
    else:
        img = 0.55 + 0.25 * base + rng.normal(0, 0.04, (size, size))
        for _ in range(4):
            hx, hy = int(rng.integers(0, size)), int(rng.integers(0, size))
            img += 0.20 * np.exp(-((x - hx) ** 2 + (y - hy) ** 2) / (2 * (size / 14) ** 2))

    return np.clip(img, 0, 1).astype(np.float32)


def generate_synthetic_dataset(
    n_per_class: int = 80,
    num_classes: int = DEFAULT_NUM_CLASSES,
    size: int = DEFAULT_IMAGE_SIZE,
    seed: int = 42,
) -> Path:
    """Write synthetic PNGs into dataset/<class_name>/ folders."""
    ensure_dataset_dir()
    rng = np.random.default_rng(seed)
    class_names = CLASS_LABELS[:num_classes] if num_classes <= 4 else [f"class_{i}" for i in range(num_classes)]

    for cls_idx, cls_name in enumerate(class_names):
        out_dir = DATASET_DIR / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            img = generate_synthetic_thermal_image(cls_idx, size, rng)
            _save_image_array(img, out_dir / f"synthetic_{i:04d}.png")

    print(f"  Synthetic thermal dataset -> {DATASET_DIR}  ({n_per_class * len(class_names)} images)")
    return DATASET_DIR


def ensure_thermal_dataset(auto_generate: bool = True) -> Path:
    if not dataset_is_empty():
        return DATASET_DIR
    if auto_generate:
        print("  No thermal images found - generating synthetic reference dataset...")
        return generate_synthetic_dataset()
    raise FileNotFoundError(f"No images in {DATASET_DIR}.")


def load_raw_dataset(
    auto_generate: bool = True,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Path]]:
    """
    Load all images as (N, H, W) float32 and labels (N,).

    Returns images, labels, class_names, paths.
    """
    if dataset_is_empty():
        ensure_thermal_dataset(auto_generate)

    entries = discover_image_paths()
    if not entries:
        raise RuntimeError("Dataset folder exists but contains no images.")

    class_names = sorted({e[2] for e in entries}, key=lambda n: CLASS_LABELS.index(n) if n in CLASS_LABELS else 999)
    images, labels, paths = [], [], []

    corrupted = 0
    for path, label, _name in entries:
        try:
            img = load_image(path, size=image_size)
            images.append(img)
            labels.append(label)
            paths.append(path)
        except Exception:
            corrupted += 1

    if not images:
        raise RuntimeError(f"All {corrupted} images failed to load.")

    return np.stack(images), np.array(labels, dtype=np.int64), class_names, paths


def inspect_dataset(auto_generate: bool = True, image_size: int = DEFAULT_IMAGE_SIZE) -> ThermalDatasetInfo:
    images, labels, class_names, paths = load_raw_dataset(auto_generate, image_size)

    dist: Dict[str, int] = {}
    for lbl in labels:
        name = class_names[lbl] if lbl < len(class_names) else str(lbl)
        dist[name] = dist.get(name, 0) + 1

    fmt = paths[0].suffix.lower().lstrip(".") if paths else "unknown"
    expected = len(discover_image_paths())
    corrupted = max(0, expected - len(paths))

    return ThermalDatasetInfo(
        n_samples=len(images),
        image_width=images.shape[2],
        image_height=images.shape[1],
        n_classes=len(class_names),
        class_names=class_names,
        class_distribution=dist,
        image_format=fmt,
        missing_images=0,
        corrupted_images=corrupted,
        source_dirs=[str(DATASET_DIR / c) for c in class_names],
    )


def compute_thermal_stats(image: np.ndarray) -> Dict[str, float]:
    """Temperature proxy statistics from normalized image."""
    img = np.asarray(image, dtype=np.float32).squeeze()
    return {
        "mean_temp": float(np.mean(img)),
        "max_temp": float(np.max(img)),
        "min_temp": float(np.min(img)),
        "std_temp": float(np.std(img)),
    }
