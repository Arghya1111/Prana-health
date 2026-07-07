"""
STEP 5 — 1D CNN architecture for PPG classification.

Modular Conv1D blocks with configurable depth and channel widths.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    """Conv1D -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        pool_size: int = 2,
        padding: Optional[int] = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PPG1DCNN(nn.Module):
    """
    1D CNN for PPG pulse pattern classification.

    Input (B, 1, L) -> ConvBlocks -> Flatten -> FC -> Output (B, num_classes)
    """

    def __init__(
        self,
        signal_length: int,
        num_classes: int = 2,
        conv_channels: Optional[List[int]] = None,
        kernel_size: int = 7,
        pool_size: int = 2,
        fc_hidden: int = 128,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 128]

        self.signal_length = signal_length
        self.num_classes = num_classes

        layers: List[nn.Module] = []
        in_ch = 1
        for out_ch in conv_channels:
            layers.append(
                ConvBlock1D(in_ch, out_ch, kernel_size=kernel_size, pool_size=pool_size)
            )
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, 1, signal_length)
            flat_size = self.conv_layers(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat_size, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.flatten(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


def build_model(signal_length: int, num_classes: int = 2, **kwargs) -> PPG1DCNN:
    """Factory helper for consistent model construction."""
    return PPG1DCNN(signal_length=signal_length, num_classes=num_classes, **kwargs)
