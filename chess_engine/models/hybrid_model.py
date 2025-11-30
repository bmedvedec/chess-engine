"""
HYBRID CNN-RNN MODEL INTEGRATION
Complete Track Implementation

This module integrates the CNN (spatial understanding) with RNN (temporal understanding)
to create a hybrid model that considers both the current board position and game history.

Architecture:
                    Current Board (20, 8, 8)
                            ↓
                    CNN Backbone
                            ↓
                    Features (256, 8, 8)
                            ↓
    ┌───────────────────────┴───────────────────────┐
    ↓                                               ↓
Move History                                  (CNN features)
    ↓                                               ↓
RNN (LSTM)                                    Policy/Value Heads
    ↓                                               ↓
Context (256)                               Initial Predictions
    ↓                                               ↓
    └───────────────────────┬───────────────────────┘
                            ↓
                    Fusion Layer
                            ↓
                    Combined Features
                            ↓
    ┌───────────────────────┴───────────────────────┐
    ↓                                               ↓
Policy Head (4096)                              Value Head (1)

The hybrid model provides:
- Spatial pattern recognition (from CNN)
- Sequential pattern recognition (from RNN)
- Better opening book understanding
- Game phase awareness
- Historical context for decisions
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple, Optional, Dict
import sys
import os
from chess_engine.models.chess_cnn import ChessCNN, PolicyHead, ValueHead
from chess_engine.models.chess_rnn import ChessRNN


class FeatureFusion(nn.Module):
    """
    Fusion layer that combines CNN spatial features with RNN temporal context.

    Multiple fusion strategies:
    1. Concatenation: Simply concat CNN + RNN features
    2. Gated fusion: Learn to weight CNN vs RNN importance
    3. Attention fusion: CNN features attend to RNN context
    """

    # Class constants
    SPATIAL_SIZE = 8

    def __init__(
        self,
        cnn_feature_size: int = 256,
        rnn_context_size: int = 256,
        fusion_type: str = "gated",
    ):
        """
        Initialize feature fusion layer.

        Args:
            cnn_feature_size: Size of CNN features (default: 256)
            rnn_context_size: Size of RNN context (default: 256)
            fusion_type: Type of fusion - "concat", "gated", or "attention" (default: "gated")
        """
        super().__init__()

        self.cnn_feature_size = cnn_feature_size
        self.rnn_context_size = rnn_context_size
        self.fusion_type = fusion_type

        if fusion_type == "concat":
            # Simple concatenation
            self.output_size = cnn_feature_size + rnn_context_size

        elif fusion_type == "gated":
            # Gated fusion - learn to weight CNN vs RNN
            self.gate = nn.Sequential(
                nn.Linear(cnn_feature_size + rnn_context_size, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
            self.output_size = cnn_feature_size

            # Project RNN context to match CNN size
            self.rnn_projection = nn.Linear(rnn_context_size, cnn_feature_size)

        elif fusion_type == "attention":
            # Attention-based fusion
            self.query_proj = nn.Linear(cnn_feature_size, 128)
            self.key_proj = nn.Linear(rnn_context_size, 128)
            self.value_proj = nn.Linear(rnn_context_size, cnn_feature_size)
            self.output_size = cnn_feature_size

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def _expand_to_spatial(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Expand 1D/2D tensor to spatial dimensions (B, C, H, W).

        Args:
            tensor (torch.Tensor): Input tensor (batch, features)
            batch_size (int): Batch size

        Returns:
            torch.Tensor: Expanded tensor (batch, features, H, W)
        """
        return tensor.view(batch_size, -1, 1, 1).expand(
            -1, -1, self.SPATIAL_SIZE, self.SPATIAL_SIZE
        )

    def forward(
        self, cnn_features: torch.Tensor, rnn_context: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse CNN and RNN features.

        Args:
            cnn_features (torch.Tensor): CNN features (batch, cnn_feature_size, 8, 8)
            rnn_context (torch.Tensor): RNN context (batch, rnn_context_size)

        Returns:
            torch.Tensor: Fused features (batch, output_size, 8, 8)
        """
        batch_size = cnn_features.size(0)

        if self.fusion_type == "concat":
            # Expand RNN context to match spatial dimensions
            rnn_expanded = self._expand_to_spatial(rnn_context, batch_size)

            # Concatenate along channel dimension
            fused = torch.cat([cnn_features, rnn_expanded], dim=1)

        elif self.fusion_type == "gated":
            # Global average pool CNN features for gate computation
            cnn_pooled = F.adaptive_avg_pool2d(cnn_features, 1).squeeze(-1).squeeze(-1)

            # Compute gate
            gate_input = torch.cat([cnn_pooled, rnn_context], dim=1)
            gate = self.gate(gate_input).view(batch_size, 1, 1, 1)  # (batch, 1, 1, 1)

            # Project and expand RNN context
            rnn_projected = self.rnn_projection(
                rnn_context
            )  # (batch, cnn_feature_size)
            # rnn_projected = rnn_projected.view(batch_size, -1, 1, 1).expand(
            #     -1, -1, 8, 8
            # )
            rnn_projected = self._expand_to_spatial(rnn_projected, batch_size)

            # Apply gated fusion
            fused = gate * cnn_features + (1 - gate) * rnn_projected

        elif self.fusion_type == "attention":
            # Global average pool CNN features
            cnn_pooled = F.adaptive_avg_pool2d(cnn_features, 1).squeeze(-1).squeeze(-1)

            # Compute attention score
            query = self.query_proj(cnn_pooled)  # (batch, 128)
            key = self.key_proj(rnn_context)  # (batch, 128)

            attention_score = torch.sigmoid(torch.sum(query * key, dim=1, keepdim=True))

            # Apply attention
            value = self.value_proj(rnn_context)  # (batch, cnn_feature_size)
            value_expanded = self._expand_to_spatial(value, batch_size)

            fused = (
                cnn_features
                + attention_score.view(batch_size, 1, 1, 1) * value_expanded
            )

        return fused


class HybridChessNet(nn.Module):
    """
    Hybrid CNN-RNN chess neural network.

    Combines:
    - CNN for spatial board understanding
    - RNN for sequential move understanding
    - Fusion layer to combine both
    - Policy and value heads for decision making

    This architecture captures both:
    - "What does the position look like?" (CNN)
    - "How did we get here?" (RNN)
    """

    # Class constants
    DEFAULT_BOARD_SIZE = 8
    DEFAULT_SPATIAL_DIM = (8, 8)
    DEFAULT_BOARD_CHANNELS = 20

    def __init__(
        self,
        # CNN parameters
        cnn_input_channels: int = 20,
        cnn_filters: int = 256,
        cnn_residual_blocks: int = 10,
        cnn_dropout: float = 0.0,
        # RNN parameters
        rnn_num_moves: int = 4096,
        rnn_embedding_dim: int = 64,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 2,
        rnn_dropout: float = 0.3,
        rnn_bidirectional: bool = False,
        rnn_use_attention: bool = False,
        rnn_use_layer_norm: bool = True,
        rnn_use_positional_encoding: bool = True,
        rnn_use_gradient_checkpointing: bool = False,
        rnn_gradient_checkpointing_threshold: int = 50,
        rnn_context_strategy: str = "last",
        # Fusion parameters
        fusion_type: str = "gated",
        # Output parameters
        num_actions: int = 4096,
        # Mode
        use_rnn: bool = True,
    ):
        """
        Initialize hybrid model.

        Args:
            cnn_input_channels: Number of input board channels (default: 20)
            cnn_filters: Number of CNN filters (default: 256)
            cnn_residual_blocks: Number of residual blocks (default: 10)
            cnn_dropout: CNN dropout (default: 0.0)

            rnn_num_moves: Number of possible moves (default: 4096)
            rnn_embedding_dim: Move embedding dimension (default: 64)
            rnn_hidden_size: RNN hidden size (default: 256)
            rnn_num_layers: Number of RNN layers (default: 2)
            rnn_dropout: RNN dropout (default: 0.3)
            rnn_bidirectional: Use bidirectional LSTM (default: False)
            rnn_use_attention: Use attention in RNN (default: False)
            rnn_use_layer_norm: Use layer normalization (default: True)
            rnn_use_positional_encoding: Add positional encoding (default: True)
            rnn_use_gradient_checkpointing: Enable gradient checkpointing (default: False)
            rnn_gradient_checkpointing_threshold: Min seq length for checkpointing (50)
            rnn_context_strategy: Context extraction - "last", "max", "mean", "multi" (default: "last")

            fusion_type: Type of feature fusion (default: "gated")

            num_actions: Number of possible actions (default: 4096)

            use_rnn: Whether to use RNN (if False, CNN-only mode) (default: True)
        """
        super().__init__()

        self.use_rnn = use_rnn
        self.fusion_type = fusion_type
        self.num_actions = num_actions

        # CNN backbone
        self.cnn = ChessCNN(
            input_channels=cnn_input_channels,
            num_filters=cnn_filters,
            num_residual_blocks=cnn_residual_blocks,
            dropout=cnn_dropout,
        )

        # RNN for move history (optional)
        if use_rnn:
            self.rnn = ChessRNN(
                num_moves=rnn_num_moves,
                embedding_dim=rnn_embedding_dim,
                hidden_size=rnn_hidden_size,
                num_layers=rnn_num_layers,
                dropout=rnn_dropout,
                bidirectional=rnn_bidirectional,
                use_attention=rnn_use_attention,
                use_layer_norm=rnn_use_layer_norm,
                use_positional_encoding=rnn_use_positional_encoding,
                use_gradient_checkpointing=rnn_use_gradient_checkpointing,
                gradient_checkpointing_threshold=rnn_gradient_checkpointing_threshold,
                context_strategy=rnn_context_strategy,
            )

            # Feature fusion
            self.fusion = FeatureFusion(
                cnn_feature_size=cnn_filters,
                rnn_context_size=rnn_hidden_size,
                fusion_type=fusion_type,
            )

            # Adjust policy/value head input size based on fusion
            head_input_size = self.fusion.output_size
        else:
            self.rnn = None
            self.fusion = None
            head_input_size = cnn_filters

        # Policy head
        self.policy_head = PolicyHead(
            input_channels=head_input_size, num_actions=num_actions
        )

        # Value head
        self.value_head = ValueHead(input_channels=head_input_size)

    @property
    def device(self) -> torch.device:
        """Get the device this model is on"""
        return next(self.parameters()).device

    def forward(
        self,
        board: torch.Tensor,
        move_history: Optional[torch.Tensor] = None,
        history_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through hybrid model.

        Args:
            board: Board tensor (batch, 20, 8, 8)
            move_history: Optional move history (batch, seq_len)
            history_lengths: Optional actual lengths (batch,)

        Returns:
            Tuple of:
            - policy_logits: Move logits (batch, num_actions)
            - value: Position evaluation (batch, 1)
            - attention_weights: Optional attention weights (batch, seq_len)

        Raises:
            AssertionError: If input shapes are invalid
        """
        # Input validation
        assert board.dim() == 4, f"Expected 4D board tensor, got {board.dim()}D"
        assert (
            board.size(1) == self.DEFAULT_BOARD_CHANNELS
        ), f"Expected {self.DEFAULT_BOARD_CHANNELS} channels, got {board.size(1)}"
        assert (
            board.size(2) == board.size(3) == self.DEFAULT_BOARD_SIZE
        ), f"Expected {self.DEFAULT_BOARD_SIZE}x{self.DEFAULT_BOARD_SIZE} board"

        if move_history is not None:
            assert (
                move_history.dim() == 2
            ), f"Expected 2D history, got {move_history.dim()}D"
            assert move_history.size(0) == board.size(
                0
            ), f"Batch size mismatch: board={board.size(0)}, history={move_history.size(0)}"

        # Ensure tensors on correct device
        device = self.device
        board = board.to(device)
        if move_history is not None:
            move_history = move_history.to(device)
        if history_lengths is not None:
            history_lengths = history_lengths.to(device)

        # Extract CNN features
        cnn_features = self.cnn(board)  # (batch, 256, 8, 8)

        # Process move history if provided and RNN is enabled
        attention_weights: Optional[torch.Tensor] = None
        features: torch.Tensor

        if (
            self.use_rnn
            and move_history is not None
            and self.rnn is not None
            and self.fusion is not None
        ):
            rnn_context, attention_weights = self.rnn(move_history, history_lengths)

            # Fuse CNN and RNN features
            features = self.fusion(cnn_features, rnn_context)
        else:
            # CNN-only mode
            features = cnn_features

        # Generate predictions
        policy_logits = self.policy_head(features)
        value = self.value_head(features)

        return policy_logits, value, attention_weights

    def predict(
        self,
        board: torch.Tensor,
        move_history: Optional[torch.Tensor] = None,
        history_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction mode with softmax applied.

        Args:
            board: Board tensor (batch, 20, 8, 8)
            move_history: Optional move history (batch, seq_len)
            history_lengths: Optional actual lengths (batch,)

        Returns:
            Tuple of:
            - policy_probs: Move probabilities (batch, num_actions)
            - value: Position evaluation (batch, 1)
        """
        self.eval()
        with torch.no_grad():
            policy_logits, value, _ = self.forward(board, move_history, history_lengths)
            policy_probs = F.softmax(policy_logits, dim=1)
        return policy_probs, value

    def benchmark_inference_speed(
        self, num_iterations: int = 100, batch_size: int = 1
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.

        Args:
            num_iterations (int): Number of iterations for benchmarking
            batch_size (int): Batch size to test

        Returns:
            Dict with timing statistics
        """
        self.eval()
        device = self.device

        # Create dummy inputs
        board = torch.randn(batch_size, 20, 8, 8, device=device)
        history = torch.randint(0, 4096, (batch_size, 50), device=device)
        lengths = torch.full((batch_size,), 50, dtype=torch.long, device=device)

        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = self.forward(board, history, lengths)

        # Synchronize if using CUDA
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()

        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.forward(board, history, lengths)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        avg_time_ms = (elapsed / num_iterations) * 1000

        results = {
            "avg_time_ms": avg_time_ms,
            "iterations": num_iterations,
            "batch_size": batch_size,
            "device": str(device),
            "passes_requirement": avg_time_ms < 50.0,
        }

        return results

    def summary(self) -> Dict[str, Any]:
        """
        Get model summary statistics.

        Returns:
            Dict containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": total_params * 4 / (1024**2),
            "use_rnn": self.use_rnn,
            "fusion_type": self.fusion_type if self.use_rnn else "none",
            "num_actions": self.num_actions,
            "device": str(self.device),
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
