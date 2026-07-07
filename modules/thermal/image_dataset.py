"""
STEP 3 — PyTorch Dataset for thermal images.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from modules.thermal.preprocessing import PreprocessConfig, preprocess_image


class ThermalImageDataset(Dataset):
    """Returns (image_tensor, label_tensor) — image shape (1, H, W)."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        config: PreprocessConfig,
        augment: bool = False,
        transform: Optional[Callable] = None,
    ) -> None:
        if len(images) != len(labels):
            raise ValueError("images/labels length mismatch")
        self.images = images
        self.labels = labels.astype(np.int64)
        self.config = config
        self.augment = augment
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(idx + int(self.config.random_state))
        img = preprocess_image(self.images[idx], self.config, augment=self.augment, rng=rng)
        if self.transform:
            img = self.transform(img)
        tensor = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label


def create_dataloader(
    images: np.ndarray,
    labels: np.ndarray,
    config: PreprocessConfig,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    ds = ThermalImageDataset(images, labels, config, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
