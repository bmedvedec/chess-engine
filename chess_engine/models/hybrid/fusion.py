import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureFusion(nn.Module):
    """
    Combines CNN spatial features with RNN temporal context.
    """

    def __init__(
        self,
        cnn_feature_size: int,
        rnn_context_size: int,
        fusion_type: str = "gated",
    ):
        super().__init__()

        self.fusion_type = fusion_type

        if fusion_type == "concat":
            self.output_size = cnn_feature_size + rnn_context_size

        elif fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(cnn_feature_size + rnn_context_size, cnn_feature_size),
                nn.Sigmoid(),
            )
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
