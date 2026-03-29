import logging
from typing import Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FeatureFusion(nn.Module):
    """
    Combines CNN spatial features with RNN temporal context.

    Gate logging (gated fusion only):
        After each forward pass ``last_gate_mean`` holds the batch-mean gate
        value (float in [0, 1]).  Interpretation:
            ≈ 1.0  → output dominated by CNN (LSTM barely contributing)
            ≈ 0.0  → output dominated by RNN projection (CNN barely used)
            ≈ 0.5  → both branches contribute equally (healthy balance)
        The trainer can read this attribute after each step and write it to
        TensorBoard or a CSV for the ablation study.
    """

    # Declare plain-Python attributes at class level so that Pyright/Pylance
    # resolves their types directly instead of routing assignments through
    # nn.Module.__setattr__ (which is typed as accepting only Tensor | Module).
    fusion_type: str
    output_size: int
    log_gates: bool
    last_gate_mean: Optional[float]

    def __init__(
        self,
        cnn_feature_size: int,
        rnn_context_size: int,
        fusion_type: str = "gated",
        log_gates: bool = False,
        gate_bias: float = 2.2,
    ):
        """
        Args:
            cnn_feature_size: Channel count coming out of the CNN backbone.
            rnn_context_size: Size of the RNN context vector.
            fusion_type: One of "concat", "gated", or "attention".
            log_gates: If True, emit gate statistics to the module logger at
                DEBUG level on every forward pass (gated fusion only).
        """
        super().__init__()

        self.fusion_type = fusion_type
        self.log_gates = log_gates

        # Populated after each gated forward pass; None otherwise.
        self.last_gate_mean: Optional[float] = None

        if fusion_type == "concat":
            self.output_size = cnn_feature_size + rnn_context_size

        elif fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(cnn_feature_size + rnn_context_size, cnn_feature_size),
                nn.Sigmoid(),
            )
            nn.init.constant_(cast(nn.Linear, self.gate[0]).bias, gate_bias)
            self.rnn_projection = nn.Linear(rnn_context_size, cnn_feature_size)
            self.output_size = cnn_feature_size

        elif fusion_type == "attention":
            self.query_proj = nn.Linear(cnn_feature_size, 128)
            self.key_proj = nn.Linear(rnn_context_size, 128)
            self.value_proj = nn.Linear(rnn_context_size, cnn_feature_size)
            self.output_size = cnn_feature_size

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    @staticmethod
    def _expand_to_spatial(
        vector: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Expands (B, C) → (B, C, H, W) using target spatial size.
        """
        _, _, h, w = target.shape
        return vector[:, :, None, None].expand(-1, -1, h, w)

    def forward(
        self,
        cnn_features: torch.Tensor,
        rnn_context: torch.Tensor,
    ) -> torch.Tensor:

        if self.fusion_type == "concat":
            rnn_expanded = self._expand_to_spatial(rnn_context, cnn_features)
            return torch.cat([cnn_features, rnn_expanded], dim=1)

        if self.fusion_type == "gated":
            pooled = F.adaptive_avg_pool2d(cnn_features, 1).flatten(1)
            gate = self.gate(torch.cat([pooled, rnn_context], dim=1))

            # --- Gate logging (ablation study) ---
            # gate shape: (batch, cnn_feature_size)
            # Values near 1 → CNN dominates; near 0 → RNN dominates.
            self.last_gate_mean = gate.mean().item()
            if self.log_gates:
                logger.debug(
                    "fusion gate | mean=%.4f  min=%.4f  max=%.4f",
                    self.last_gate_mean,
                    gate.min().item(),
                    gate.max().item(),
                )

            gate = gate[:, :, None, None]

            rnn_proj = self.rnn_projection(rnn_context)
            rnn_proj = self._expand_to_spatial(rnn_proj, cnn_features)

            return gate * cnn_features + (1 - gate) * rnn_proj

        # attention
        pooled = F.adaptive_avg_pool2d(cnn_features, 1).flatten(1)
        q = self.query_proj(pooled)
        k = self.key_proj(rnn_context)
        score = torch.sigmoid((q * k).sum(dim=1, keepdim=True))

        v = self.value_proj(rnn_context)
        v = self._expand_to_spatial(v, cnn_features)

        return cnn_features + score[:, None, None] * v
