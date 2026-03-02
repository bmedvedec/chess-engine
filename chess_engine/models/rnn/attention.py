import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AttentionLayer(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.score = nn.Linear(input_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(x).squeeze(-1)

        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)

        weights = F.softmax(scores, dim=1)
        attended = torch.sum(x * weights.unsqueeze(-1), dim=1)

        return attended, weights
