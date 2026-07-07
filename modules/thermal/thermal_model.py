"""
STEP 4 — 2D CNN for thermal image classification.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class ConvBlock2D(nn.Module):
    """Conv2D -> BatchNorm -> ReLU -> MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: int = 2) -> None:
        super().__init__()
        padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ThermalCNN(nn.Module):
    """
    Modular 2D CNN for thermal anomaly classification.

    Input (B, 1, H, W) -> Conv blocks -> AdaptiveAvgPool -> Dropout -> Linear
    Supports binary and multi-class via num_classes.
    """

    def __init__(
        self,
        image_size: int = 128,
        in_channels: int = 1,
        num_classes: int = 2,
        conv_channels: Optional[List[int]] = None,
        dropout: float = 0.5,
        fc_hidden: int = 256,
    ) -> None:
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128]

        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        layers: List[nn.Module] = []
        ch_in = in_channels
        for ch_out in conv_channels:
            layers.append(ConvBlock2D(ch_in, ch_out))
            ch_in = ch_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(ch_in, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.forward(x), dim=-1)


def build_model(
    image_size: int = 128,
    in_channels: int = 1,
    num_classes: int = 2,
    **kwargs,
) -> ThermalCNN:
    return ThermalCNN(
        image_size=image_size,
        in_channels=in_channels,
        num_classes=num_classes,
        **kwargs,
    )
