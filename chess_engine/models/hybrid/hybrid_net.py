import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from chess_engine.models.cnn.backbone import ChessCNN
from chess_engine.models.cnn.heads import PolicyHead, ValueHead
from chess_engine.models.hybrid.config import HybridModelConfig
from chess_engine.models.hybrid.fusion import FeatureFusion
from chess_engine.models.rnn.chess_rnn import ChessRNN


class HybridChessNet(nn.Module):
    def __init__(self, config: HybridModelConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.cnn = ChessCNN(
            input_channels=config.cnn_input_channels,
            num_filters=config.cnn_filters,
            num_residual_blocks=config.cnn_residual_blocks,
        )

        if config.use_rnn:
            self.rnn = ChessRNN(
                num_moves=config.rnn_num_moves,
                embedding_dim=config.rnn_embedding_dim,
                hidden_size=config.rnn_hidden_size,
                num_layers=config.rnn_num_layers,
                dropout=config.rnn_dropout,
                bidirectional=config.rnn_bidirectional,
                use_attention=config.rnn_use_attention,
                use_layer_norm=config.rnn_use_layer_norm,
                context_strategy=config.rnn_context_strategy,
            )

            self.fusion = FeatureFusion(
                cnn_feature_size=config.cnn_filters,
                rnn_context_size=config.rnn_output_size,
                fusion_type=config.fusion_type,
            )

            head_channels = self.fusion.output_size
        else:
            self.rnn = None
            self.fusion = None
            head_channels = config.cnn_filters

        self.policy_head = PolicyHead(head_channels, config.num_actions)
        self.value_head = ValueHead(head_channels)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        board: torch.Tensor,
        move_history: Optional[torch.Tensor] = None,
        history_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        cnn_features: torch.Tensor = self.cnn(board)
        attention: Optional[torch.Tensor] = None

        if (
            self.rnn is not None
            and self.fusion is not None
            and move_history is not None
        ):
            rnn_context, attention = self.rnn(move_history, history_lengths)
            features = self.fusion(cnn_features, rnn_context)
        else:
            features = cnn_features

        policy = self.policy_head(features)
        value = self.value_head(features)

        return policy, value, attention

    def predict(
        self,
        board: torch.Tensor,
        move_history: Optional[torch.Tensor] = None,
        history_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self.eval()
        with torch.no_grad():
            policy, value, _ = self.forward(board, move_history, history_lengths)
            return F.softmax(policy, dim=1), value

    def summary(self) -> Dict[str, Any]:
        params = sum(p.numel() for p in self.parameters())
        return {
            "parameters": params,
            "model_size_mb": params * 4 / (1024**2),
            "use_rnn": self.config.use_rnn,
            "fusion": self.config.fusion_type,
            "actions": self.config.num_actions,
        }
