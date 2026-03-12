"""
CNN ARCHITECTURE - Policy and Value Heads

Policy Head: Takes CNN features (256, 8, 8) and outputs move probabilities (4672,).
Value Head: Takes CNN features (256, 8, 8) and outputs position evaluation [-1, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """
    Policy head for move prediction.

    Takes CNN features (256, 8, 8) and outputs move probabilities (4672,).

    Architecture:
    Feature maps (256, 8, 8)
         ↓
    Conv 1×1 (2 filters) - Reduce channels
         ↓
    Flatten to (2 × 8 × 8 = 128)
         ↓
    Fully connected (128 → 4672)
         ↓
    Logits for each possible move

    Uses 4672 moves (AlphaZero-style encoding: 64 squares × 73 move types).
    """

    def __init__(self, input_channels: int = 256, num_actions: int = 4672):
        """
        Initialize policy head.

        Args:
            input_channels: Number of input channels from CNN (default: 256)
            num_actions: Number of possible moves (default: 4672)
        """
        super().__init__()

        self.num_actions = num_actions

        # Reduce channels with 1×1 convolution
        self.conv = nn.Conv2d(input_channels, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)

        # Fully connected layer to output move logits
        self.fc = nn.Linear(2 * 8 * 8, num_actions)

        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through policy head.

        Args:
            x: Input feature tensor of shape (batch, 256, 8, 8)

        Returns:
            Policy logits of shape (batch, num_actions)
        """
        # Reduce channels
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)

        # Flatten spatial dimensions
        x = x.flatten(1)  # (batch, 2*8*8)

        # Output move logits
        x = self.fc(x)

        return x


class ValueHead(nn.Module):
    """
    Value head for position evaluation.

    Takes CNN features (256, 8, 8) and outputs a single value in [-1, 1]
    representing position evaluation:
    - +1.0: White is completely winning
    -  0.0: Position is equal
    - -1.0: Black is completely winning

    Architecture:
    Feature maps (256, 8, 8)
         ↓
    Conv 1×1 (1 filter) - Reduce to single channel
         ↓
    Flatten to (1 × 8 × 8 = 64)
         ↓
    Fully connected (64 → 256)
         ↓
    ReLU
         ↓
    Fully connected (256 → 1)
         ↓
    Tanh (output in [-1, 1])
    """

    def __init__(self, input_channels: int = 256):
        """
        Initialize value head.

        Args:
            input_channels: Number of input channels from CNN (default: 256)
        """
        super().__init__()

        # Reduce channels with 1×1 convolution
        self.conv = nn.Conv2d(input_channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(1 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through value head.

        Args:
            x: Input feature tensor of shape (batch, 256, 8, 8)

        Returns:
            Position value of shape (batch, 1) in range [-1, 1]
        """
        # Reduce channels
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)

        # Flatten
        x = x.flatten(1)  # (batch, 64)

        # First FC layer
        x = self.fc1(x)
        x = F.relu(x, inplace=True)

        # Output layer with tanh (outputs in [-1, 1])
        x = self.fc2(x)
        x = torch.tanh(x)

        return x
