import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim: int, max_len: int = 512):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )

        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, dim)
        pe = self.pe[: x.size(1)].unsqueeze(0)
        return x + pe


class MoveEmbedding(nn.Module):
    def __init__(
        self,
        num_moves: int,
        embedding_dim: int,
        padding_idx: int = 0,
        use_positional_encoding: bool = True,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_moves,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].fill_(0)

        self.positional = (
            PositionalEncoding(embedding_dim, max_seq_len)
            if use_positional_encoding
            else None
        )

    def forward(self, move_indices: torch.Tensor) -> torch.Tensor:
        x = self.embedding(move_indices)

        if self.positional is not None:
            x = self.positional(x)

        return x
