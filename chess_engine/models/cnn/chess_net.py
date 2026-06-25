"""
CNN ARCHITECTURE - Complete Chess Network

Combines CNN backbone with policy and value heads into the main model.
Includes save/load, predict, and model info methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Optional, Union

from chess_engine.models.cnn.backbone import ChessCNN
from chess_engine.models.cnn.heads import PolicyHead, ValueHead
from chess_engine.models.cnn.utils import count_parameters


class ChessNet(nn.Module):
    """CNN backbone with policy and value heads for chess position evaluation."""

    def __init__(
        self,
        input_channels: int = 22,
        num_filters: int = 256,
        num_residual_blocks: int = 10,
        num_actions: int = 4672,
        dropout: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.backbone = ChessCNN(
            input_channels=input_channels,
            num_filters=num_filters,
            num_residual_blocks=num_residual_blocks,
            dropout=dropout,
        )

        self.policy_head = PolicyHead(
            input_channels=num_filters, num_actions=num_actions
        )

        self.value_head = ValueHead(input_channels=num_filters)

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.device != self.device:
            x = x.to(self.device)

        features = self.backbone(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        return policy_logits, value

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inference: forward pass with softmax applied to policy logits."""
        self.eval()

        policy_logits, value = self.forward(x)
        policy_probs = F.softmax(policy_logits, dim=1)

        return policy_probs, value

    def save(self, path: str):
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
