"""
STEP 5 — 1D CNN for SpO2 / respiratory risk classification.

Supports binary and multi-class output via num_classes parameter.
Accepts multi-channel input (PPG + SpO2 trace).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
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


class SpO2CNN(nn.Module):
    """
    1D CNN for SpO2 classification.

    Input (B, C, L) where C=2 (PPG + SpO2) by default.
    """

    def __init__(
        self,
        signal_length: int,
        in_channels: int = 2,
        num_classes: int = 4,
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
        self.in_channels = in_channels
        self.num_classes = num_classes

        layers: List[nn.Module] = []
        ch_in = in_channels
        for ch_out in conv_channels:
            layers.append(ConvBlock1D(ch_in, ch_out, kernel_size, pool_size))
            ch_in = ch_out
        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, signal_length)
            flat = self.conv_layers(dummy).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Linear(flat, fc_hidden),
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


def build_model(
    signal_length: int,
    in_channels: int = 2,
    num_classes: int = 4,
    **kwargs,
) -> SpO2CNN:
    return SpO2CNN(
        signal_length=signal_length,
        in_channels=in_channels,
        num_classes=num_classes,
        **kwargs,
    )
