"""
CNN ARCHITECTURE IMPLEMENTATION
Complete Track Implementation

This module implements the Convolutional Neural Network (CNN) for spatial
understanding of chess positions. The architecture is inspired by AlphaZero
with residual blocks for deep feature learning.

Architecture Overview:
1. Initial Convolutional Layer (20 channels → 256 filters)
2. Residual Tower (10-20 residual blocks)
3. Policy Head (256 filters → 4672 move probabilities)
4. Value Head (256 filters → position evaluation [-1, 1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Optional, Union


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

    Transforms board encoding (20, 8, 8) into rich feature representation (256, 8, 8)
    that captures spatial patterns, piece relationships, and tactical motifs.

    Architecture:
    - Initial conv layer: Expands 20 input channels to 256 features
    - Residual tower: 10-20 residual blocks for deep learning
    - Output: 256 feature maps at 8×8 resolution

    This output feeds into policy and value heads.
    """

    def __init__(
        self,
        input_channels: int = 20,
        num_filters: int = 256,
        num_residual_blocks: int = 10,
        dropout: float = 0.0,
    ):
        """
        Initialize CNN backbone.

        Args:
            input_channels: Number of input channels (default: 20 for board encoding)
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
            x: Input tensor of shape (batch, 20, 8, 8)

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

    Note: Using 4672 moves (AlphaZero-style encoding) but we'll start with
    simpler 4096 (64×64) encoding. Can be upgraded later.
    """

    # TODO: upgrade to 4096 moves

    def __init__(self, input_channels: int = 256, num_actions: int = 4096):
        """
        Initialize policy head.

        Args:
            input_channels: Number of input channels from CNN (default: 256)
            num_actions: Number of possible moves (default: 4096)
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


class ChessNet(nn.Module):
    """
    Complete chess neural network combining CNN backbone with policy and value heads.

    This is the main model that will be trained. It takes a board position
    and outputs:
    1. Policy: Probability distribution over moves
    2. Value: Evaluation of the position

    Usage:
        model = ChessNet()
        policy_logits, value = model(board_tensor)
    """

    def __init__(
        self,
        input_channels: int = 20,
        num_filters: int = 256,
        num_residual_blocks: int = 10,
        num_actions: int = 4096,
        dropout: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Initialize complete chess network.

        Args:
            input_channels: Number of input channels (default: 20)
            num_filters: Number of CNN filters (default: 256)
            num_residual_blocks: Number of residual blocks (default: 10)
            num_actions: Number of possible moves (default: 4096)
            dropout: Dropout probability (default: 0.0)
            device: Device to use ('cuda' or 'cpu', auto-detect if None)
        """
        super().__init__()

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # CNN backbone
        self.backbone = ChessCNN(
            input_channels=input_channels,
            num_filters=num_filters,
            num_residual_blocks=num_residual_blocks,
            dropout=dropout,
        )

        # Policy head
        self.policy_head = PolicyHead(
            input_channels=num_filters, num_actions=num_actions
        )

        # Value head
        self.value_head = ValueHead(input_channels=num_filters)

        # Move to device
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through complete network.

        Args:
            x: Board tensor of shape (batch, 20, 8, 8)

        Returns:
            Tuple of (policy_logits, value):
            - policy_logits: shape (batch, num_actions)
            - value: shape (batch, 1) in range [-1, 1]
        """
        # Ensure input is on correct device
        if x.device != self.device:
            x = x.to(self.device)

        # Extract features with CNN backbone
        features = self.backbone(x)

        # Get policy and value predictions
        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        return policy_logits, value

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction with softmax applied to policy (for inference).

        Args:
            x: Board tensor of shape (batch, 20, 8, 8)

        Returns:
            Tuple of (policy_probs, value):
            - policy_probs: shape (batch, num_actions), sums to 1.0
            - value: shape (batch, 1) in range [-1, 1]
        """
        self.eval()

        policy_logits, value = self.forward(x)
        policy_probs = F.softmax(policy_logits, dim=1)

        return policy_probs, value

    def save(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "input_channels": self.backbone.input_channels,
                "num_filters": self.backbone.num_filters,
                "num_residual_blocks": self.backbone.num_residual_blocks,
                "num_actions": self.policy_head.num_actions,
            },
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "ChessNet":
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint
            device: Device to load to (auto-detect if None)

        Returns:
            Loaded ChessNet model
        """
        checkpoint = torch.load(path, map_location="cpu")
        config = checkpoint["config"]

        model = cls(
            input_channels=config["input_channels"],
            num_filters=config["num_filters"],
            num_residual_blocks=config["num_residual_blocks"],
            num_actions=config["num_actions"],
            device=device,
        )
        model.load_state_dict(checkpoint["state_dict"])
        print(f"Model loaded from {path}")
        return model

    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration and statistics."""
        return {
            "architecture": "AlphaZero-style CNN with Residual Blocks",
            "input_shape": f"(batch, {self.backbone.input_channels}, 8, 8)",
            "num_residual_blocks": self.backbone.num_residual_blocks,
            "num_filters": self.backbone.num_filters,
            "num_actions": self.policy_head.num_actions,
            "total_parameters": count_parameters(self),
            "backbone_parameters": count_parameters(self.backbone),
            "policy_head_parameters": count_parameters(self.policy_head),
            "value_head_parameters": count_parameters(self.value_head),
            "device": str(self.device),
        }


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
