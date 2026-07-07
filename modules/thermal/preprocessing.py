"""
STEP 2 — Thermal image preprocessing and augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from modules.thermal.utils import META_PATH, ThermalDatasetInfo


@dataclass
class PreprocessConfig:
    image_size: int = 128
    normalize_mean: float = 0.5
    normalize_std: float = 0.5
    augment_train: bool = True
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42

    @classmethod
    def from_dataset_info(cls, info: ThermalDatasetInfo, **overrides) -> "PreprocessConfig":
        cfg = cls(image_size=info.image_width)
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """Resize to (size, size)."""
    img = np.asarray(image, dtype=np.float32).squeeze()
    if img.shape[0] == size and img.shape[1] == size:
        return img
    if HAS_CV2:
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    from PIL import Image
    arr = (img * 255).astype(np.uint8)
    out = np.array(Image.fromarray(arr).resize((size, size), Image.Resampling.BILINEAR))
    return out.astype(np.float32) / 255.0


def normalize_temperature(image: np.ndarray, mean: float = 0.5, std: float = 0.5) -> np.ndarray:
    """Z-score style normalization for model input."""
    img = np.asarray(image, dtype=np.float32)
    return ((img - mean) / (std + 1e-8)).astype(np.float32)


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """CLAHE contrast enhancement when OpenCV available."""
    if not HAS_CV2:
        img = image.astype(np.float32)
        return np.clip((img - img.min()) / (img.max() - img.min() + 1e-8), 0, 1)

    u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    out = clahe.apply(u8)
    return out.astype(np.float32) / 255.0


def remove_noise(image: np.ndarray) -> np.ndarray:
    """Gaussian blur denoising."""
    if HAS_CV2:
        u8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        blurred = cv2.GaussianBlur(u8, (3, 3), 0)
        return blurred.astype(np.float32) / 255.0
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(image.astype(np.float32), sigma=0.8)


def detect_hotspots(image: np.ndarray, threshold: float = 0.65, min_area: int = 9) -> np.ndarray:
    """
    Detect hotspot regions above temperature threshold.

    Returns binary mask (H, W).
    """
    img = np.asarray(image, dtype=np.float32).squeeze()
    mask = (img >= threshold).astype(np.uint8)
    if HAS_CV2:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                filtered[labels == i] = 1
        return filtered
    return mask


def preprocess_image(
    image: np.ndarray,
    config: PreprocessConfig,
    augment: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Full single-image preprocessing chain."""
    img = resize_image(image, config.image_size)
    img = remove_noise(img)
    img = enhance_contrast(img)
    if augment and config.augment_train:
        img = apply_augmentation(img, rng)
    img = normalize_temperature(img, config.normalize_mean, config.normalize_std)
    return img


def apply_augmentation(image: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Random flip, rotation, brightness adjustment."""
    if rng is None:
        rng = np.random.default_rng()
    img = image.copy()

    if rng.random() > 0.5:
        img = np.fliplr(img)
    if rng.random() > 0.5:
        img = np.flipud(img)

    if HAS_CV2 and rng.random() > 0.5:
        angle = float(rng.uniform(-15, 15))
        h, w = img.shape
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REFLECT)

    brightness = float(rng.uniform(0.85, 1.15))
    img = np.clip(img * brightness, 0, 1)
    return img.astype(np.float32)


def preprocess_batch(
    images: np.ndarray,
    config: PreprocessConfig,
    augment: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(config.random_state)
    out = np.empty_like(images)
    for i, img in enumerate(images):
        r = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        out[i] = preprocess_image(img, config, augment=augment, rng=r)
    return out


def split_train_val_test(
    images: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, ...]:
    X_temp, X_test, y_temp, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels,
    )
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, random_state=random_state, stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
