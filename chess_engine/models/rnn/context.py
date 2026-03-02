import torch
from typing import Optional


class ContextExtractor:
    VALID = {"last", "max", "mean", "multi"}

    def __init__(self, strategy: str, bidirectional: bool):
        if strategy not in self.VALID:
            raise ValueError(f"Invalid context strategy: {strategy}")

        self.strategy = strategy
        self.bidirectional = bidirectional

    def __call__(
        self,
        output: torch.Tensor,
        hidden: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, seq_len, out_dim = output.shape
        _, hidden_batch, hidden_dim = hidden.shape

        assert batch == hidden_batch, "Batch mismatch between output and hidden"
        assert out_dim % hidden_dim == 0, "Invalid hidden/output dimensionality"

        # --- last ---
        if self.bidirectional:
            last = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            last = hidden[-1]

        if self.strategy == "last":
            return last

        # --- max ---
        max_ctx, _ = output.max(dim=1)
        if self.strategy == "max":
            return max_ctx

        # --- mean ---
        if lengths is not None:
            mask = (
                torch.arange(seq_len, device=output.device).unsqueeze(0)
                < lengths.unsqueeze(1)
            ).unsqueeze(-1)

            mean_ctx = (output * mask).sum(dim=1) / lengths.unsqueeze(1)
        else:
            mean_ctx = output.mean(dim=1)

        if self.strategy == "mean":
            return mean_ctx

        # --- multi ---
        return torch.cat([last, max_ctx, mean_ctx], dim=1)
