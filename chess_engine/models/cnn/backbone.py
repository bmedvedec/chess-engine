"""
CNN ARCHITECTURE - Backbone (Residual Tower)

Implements the convolutional backbone with residual blocks inspired by AlphaZero.
Transforms board encoding (22, 8, 8) into rich feature representation (256, 8, 8).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessResidualBlock(nn.Module):
    """Two-conv residual block with BatchNorm and optional dropout."""

    def __init__(self, num_filters: int = 256, dropout: float = 0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            padding=1,
            bias=False,  # BatchNorm handles bias
        )
        self.bn1 = nn.BatchNorm2d(num_filters)

        self.conv2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_filters)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)  # inplace for memory efficiency

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out, inplace=True)

        return out


class ChessCNN(nn.Module):
    """CNN backbone: initial conv + residual tower, outputs (batch, num_filters, 8, 8)."""

    def __init__(
        self,
        input_channels: int = 22,
        num_filters: int = 256,
        num_residual_blocks: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_residual_blocks = num_residual_blocks

        self.conv_initial = nn.Conv2d(
            in_channels=input_channels,
            out_channels=num_filters,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn_initial = nn.BatchNorm2d(num_filters)

        self.residual_blocks = nn.ModuleList(
            [
                ChessResidualBlock(num_filters, dropout)
                for _ in range(num_residual_blocks)
            ]
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_initial(x)
        x = self.bn_initial(x)
        x = F.relu(x, inplace=True)

        for block in self.residual_blocks:
            x = block(x)

        return x
