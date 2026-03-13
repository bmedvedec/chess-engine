from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal


@dataclass(frozen=True)
class HybridModelConfig:
    # ===== CNN =====
    cnn_input_channels: int = 22
    cnn_filters: int = 256
    cnn_residual_blocks: int = 10
    cnn_dropout: float = 0.0

    # ===== RNN =====
    rnn_num_moves: int = 4672
    rnn_embedding_dim: int = 64
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 2
    rnn_dropout: float = 0.3
    rnn_bidirectional: bool = False
    rnn_use_attention: bool = False
    rnn_use_layer_norm: bool = True
    rnn_context_strategy: Literal["last", "mean", "max", "multi"] = "last"

    # ===== Fusion =====
    fusion_type: Literal["concat", "gated", "attention"] = "gated"

    # ===== Output =====
    num_actions: int = 4672

    # ===== Mode =====
    use_rnn: bool = True

    @property
    def rnn_output_size(self) -> int:
        return self.rnn_hidden_size * (2 if self.rnn_bidirectional else 1)

    def validate(self) -> None:
        assert self.num_actions in (
            4096,
            4672,
        ), f"Unsupported action space: {self.num_actions}"

    # --------------------
    # Serialization
    # --------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to a JSON-safe dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridModelConfig":
        """Deserialize config from dict (e.g. checkpoint)."""
        return cls(**data)
