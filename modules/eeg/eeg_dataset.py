"""
STEP 3 — PyTorch Dataset for EEG windows.

Returns (signal_tensor, label_tensor) pairs compatible with DataLoader.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class EEGDataset(Dataset):
    """
    Custom PyTorch Dataset for 1D EEG classification.

    Each item is a tuple (signal, label):
        signal : FloatTensor of shape (1, signal_length)
        label  : LongTensor scalar class index
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        add_channel_dim: bool = True,
    ) -> None:
        if len(signals) != len(labels):
            raise ValueError(
                f"signals ({len(signals)}) and labels ({len(labels)}) length mismatch"
            )

        self.signals = signals.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.add_channel_dim = add_channel_dim

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx]
        if self.add_channel_dim and signal.ndim == 1:
            signal = signal.reshape(1, -1)

        x = torch.from_numpy(signal)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def create_dataloader(
    signals: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader from numpy arrays."""
    dataset = EEGDataset(signals, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def infer_signal_length(loader: DataLoader) -> int:
    """Read signal length from the first batch."""
    for batch_x, _ in loader:
        return batch_x.shape[-1]
    raise ValueError("DataLoader is empty")
