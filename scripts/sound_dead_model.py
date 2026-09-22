"""Closest reproducible 3D CNN for the study sound and dead classifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

import torch
import torch.nn as nn

from sound_dead_config import PATCH_SHAPE


@dataclass(frozen=True)
class ModelConfig:
    """Numerical choices that were not published for the classifier."""

    input_shape: Tuple[int, int, int] = PATCH_SHAPE
    channels: Tuple[int, int, int] = (16, 32, 64)
    kernel_size: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


class ConvBlock(nn.Module):
    """Two convolutions and one pooling layer, following the thesis topology."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class SoundDeadCNN(nn.Module):
    """Three equal convolutional blocks followed by one binary output.

    The study specifies the 11 by 80 by 80 input and the classification task,
    but it does not publish classifier channel counts, kernels, activations,
    or pooling details. The accompanying thesis specifies three blocks, each
    with two convolutions and one pooling operation, followed by a fully
    connected output. This implementation uses that topology with conservative
    reconstructed settings: 16, 32, and 64 channels, 3 cubed kernels, ReLU,
    and 2 cubed max pooling.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        if tuple(self.config.input_shape) != PATCH_SHAPE:
            raise ValueError(f"Input shape must be {PATCH_SHAPE}.")
        if len(self.config.channels) != 3:
            raise ValueError("Exactly three convolutional blocks are required.")
        if self.config.kernel_size < 1 or self.config.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        channels = (1, *tuple(self.config.channels))
        self.features = nn.Sequential(
            *[
                ConvBlock(
                    channels[index],
                    channels[index + 1],
                    self.config.kernel_size,
                )
                for index in range(3)
            ]
        )
        self.classifier = nn.Linear(self._flattened_feature_count(), 1)
        self.apply(self._initialize_weights)

    def _flattened_feature_count(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, *PATCH_SHAPE)
            return int(self.features(dummy).numel())

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
                nonlinearity="relu",
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5:
            raise ValueError(
                f"Expected B, C, R, T, Z input, found {tuple(inputs.shape)}."
            )
        features = self.features(inputs)
        flattened = torch.flatten(features, start_dim=1)
        return self.classifier(flattened).squeeze(1)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return int(total), int(trainable)
