"""
CNN ARCHITECTURE - Backbone (Residual Tower)

Implements the convolutional backbone with residual blocks inspired by AlphaZero.
Transforms board encoding (22, 8, 8) into rich feature representation (256, 8, 8).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessResidualBlock(nn.Module):
    """
    Residual block for chess CNN.

    Architecture:
    Input (256 channels)
         ↓
    Conv 3×3 (256 filters) + BatchNorm + ReLU
         ↓
    Conv 3×3 (256 filters) + BatchNorm
         ↓
    Add skip connection (input + output)
         ↓
    ReLU
         ↓
    Output (256 channels)

    Why residual connections?
    - Allows training very deep networks (10-40+ layers)
    - Helps gradient flow (solves vanishing gradient problem)
    - Network can learn identity mapping if needed
    - Used successfully in AlphaZero and many other systems
    """

    def __init__(self, num_filters: int = 256, dropout: float = 0.0):
        """
        Initialize residual block.

        Args:
            num_filters: Number of convolutional filters (default: 256)
            dropout: Dropout probability (default: 0.0, no dropout)
        """
        super().__init__()

        # First convolution: 3×3 kernel, same padding
        self.conv1 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            padding=1,  # Padding=1 keeps spatial dimensions (8×8 → 8×8)
            bias=False,  # BatchNorm handles bias
        )
        self.bn1 = nn.BatchNorm2d(num_filters)

        # Second convolution: 3×3 kernel, same padding
        self.conv2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_filters)

        # Optional dropout for regularization
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.

        Args:
            x: Input tensor of shape (batch, 256, 8, 8)

        Returns:
            Output tensor of shape (batch, 256, 8, 8)
        """
        # Save input for skip connection
        residual = x

        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)  # Inplace for memory efficiency

        # Second conv block (no ReLU yet)
        out = self.conv2(out)
        out = self.bn2(out)

        # Add skip connection (residual learning)
        out += residual

        # Final ReLU
        out = F.relu(out, inplace=True)

        return out


class ChessCNN(nn.Module):
    """
    Main CNN backbone for processing chess board state.

    Transforms board encoding (22, 8, 8) into rich feature representation (256, 8, 8)
    that captures spatial patterns, piece relationships, and tactical motifs.

    Architecture:
    - Initial conv layer: Expands 22 input channels to 256 features
    - Residual tower: 10-20 residual blocks for deep learning
    - Output: 256 feature maps at 8×8 resolution

    This output feeds into policy and value heads.
    """

    def __init__(
        self,
        input_channels: int = 22,
        num_filters: int = 256,
        num_residual_blocks: int = 10,
        dropout: float = 0.0,
    ):
        """
        Initialize CNN backbone.

        Args:
            input_channels: Number of input channels (default: 22 for board encoding)
            num_filters: Number of convolutional filters throughout network (default: 256)
            num_residual_blocks: Number of residual blocks in tower (default: 10)
            dropout: Dropout probability for regularization (default: 0.0)
        """
        super().__init__()

        self.input_channels = input_channels
        self.num_filters = num_filters
        self.num_residual_blocks = num_residual_blocks

        # Initial convolution: Expand from input_channels to num_filters
        self.conv_initial = nn.Conv2d(
            in_channels=input_channels,
            out_channels=num_filters,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn_initial = nn.BatchNorm2d(num_filters)

        # Residual tower: Stack of residual blocks
        self.residual_blocks = nn.ModuleList(
            [
                ChessResidualBlock(num_filters, dropout)
                for _ in range(num_residual_blocks)
            ]
        )

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization for ReLU networks. Kaiming init for Conv2d, constant for BatchNorm"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN backbone.

        Args:
            x: Input tensor of shape (batch, 22, 8, 8)

        Returns:
            Feature tensor of shape (batch, 256, 8, 8)
        """
        # Initial convolution
        x = self.conv_initial(x)
        x = self.bn_initial(x)
        x = F.relu(x, inplace=True)

        # Process through residual tower
        for block in self.residual_blocks:
            x = block(x)

        return x
