"""
STEP 4 — PyTorch Dataset for synchronized PPG/SpO2 windows.

Returns (signal_tensor, label_tensor) pairs compatible with DataLoader.
Identical interface to EEGDataset and PPGDataset.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SpO2Dataset(Dataset):
    """
    Custom PyTorch Dataset for SpO2 classification / regression.

    Each item is a tuple (signal, label):
        signal : FloatTensor of shape (C, signal_length)  — C=2 (PPG + SpO2)
        label  : LongTensor or FloatTensor scalar
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        task_type: str = "classification",
    ) -> None:
        if len(signals) != len(labels):
            raise ValueError(
                f"signals ({len(signals)}) and labels ({len(labels)}) length mismatch"
            )

        self.signals = signals.astype(np.float32)
        self.task_type = task_type

        if task_type == "regression":
            self.labels = labels.astype(np.float32)
        else:
            self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.signals[idx])
        if self.task_type == "regression":
            y = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def create_dataloader(
    signals: np.ndarray,
    labels: np.ndarray,
    task_type: str = "classification",
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader from numpy arrays."""
    dataset = SpO2Dataset(signals, labels, task_type=task_type)
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
