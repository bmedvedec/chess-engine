import torch
import torch.nn as nn
from typing import Optional, Tuple

from chess_engine.models.rnn.attention import AttentionLayer
from chess_engine.models.rnn.cache import RNNCache
from chess_engine.models.rnn.context import ContextExtractor
from chess_engine.models.rnn.embedding import MoveEmbedding
from chess_engine.models.rnn.lstm_core import LSTMCore


class ChessRNN(nn.Module):
    def __init__(
        self,
        num_moves: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        context_strategy: str,
        use_attention: bool,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.embedding = MoveEmbedding(num_moves, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        self.core = LSTMCore(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.cache = RNNCache()

        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.context = ContextExtractor(context_strategy, bidirectional)

        self.projection = nn.Linear(
            lstm_out_dim * (3 if context_strategy == "multi" else 1),
            hidden_size,
        )

        self.norm = nn.LayerNorm(hidden_size) if use_layer_norm else nn.Identity()

        self.attention = AttentionLayer(lstm_out_dim) if use_attention else None

    def enable_cache(self):
        self.cache.enable()

    def disable_cache(self):
        self.cache.disable()

    def reset_cache(self):
        self.cache.reset()

    def forward(
        self,
        move_indices: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.dropout(self.embedding(move_indices))

        hidden = self.cache.get(x.size(0), lengths)
        output, (h, c) = self.core(x, hidden)
        self.cache.store(h, c)

        ctx = self.context(output, h, lengths)
        ctx = self.norm(self.projection(ctx))

        if self.attention is not None:
            mask = (
                (
                    torch.arange(output.size(1), device=output.device)
                    < lengths.unsqueeze(1)
                )
                if lengths is not None
                else None
            )

            ctx, weights = self.attention(output, mask)
            return ctx, weights

        return ctx, None
