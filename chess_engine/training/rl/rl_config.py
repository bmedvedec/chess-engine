"""
RL TRAINER - Configuration

Contains all configuration parameters for the RL training loop.
"""

import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Literal, cast

import numpy as np


def _make_json_safe(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    elif hasattr(obj, "item"):  # numpy scalar types have .item() method
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


@dataclass
class RLTrainingConfig:
    """Configuration for RL training loop"""

    # =====================
    # Training iterations
    # =====================
    num_iterations: int = 100
    games_per_iteration: int = 100
    training_steps_per_iteration: int = 1000

    # =====================
    # Self-play configuration
    # =====================
    num_simulations: int = 200
    c_puct: float = 2.0
    temperature: float = 1.5
    temperature_threshold: int = 15
    max_moves_per_game: int = 200
    dirichlet_alpha: float = 0.3
    resign_threshold: float = -0.9

    # =====================
    # Parallel self-play
    # =====================
    use_parallel_selfplay: bool = True
    num_workers: Optional[int] = None

    # =====================
    # Model architecture
    # =====================
    cnn_blocks: int = 10
    cnn_filters: int = 256

    use_rnn: bool = False
    rnn_hidden_size: int = 256
    rnn_layers: int = 2
    rnn_use_attention: bool = False
    rnn_bidirectional: bool = False

    fusion_type: Literal["concat", "gated", "attention"] = "gated"
    num_actions: int = 4096  # 4096 or 4672

    # =====================
    # Training hyperparameters
    # =====================
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    # =====================
    # Optimizer settings
    # =====================
    optimizer: str = "adam"
    lr_schedule: str = "constant"
    lr_decay_steps: int = 10
    lr_decay_gamma: float = 0.5

    # =====================
    # Replay buffer
    # =====================
    buffer_size: int = 500000
    min_buffer_size: int = 10000
    sample_ratio: float = 1.0

    # =====================
    # Evaluation
    # =====================
    eval_frequency: int = 5
    eval_games: int = 20
    eval_simulations: int = 100
    win_threshold: float = 0.55

    # =====================
    # Checkpointing
    # =====================
    checkpoint_dir: str = "data/rl_checkpoints"
    save_frequency: int = 1
    keep_checkpoints: int = 20

    # =====================
    # Logging
    # =====================
    log_dir: str = "logs/rl_training"
    log_frequency: int = 10

    # =====================
    # Device
    # =====================
    device: str = "cuda"
    use_amp: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return cast(Dict[str, Any], _make_json_safe(asdict(self)))

    def save(self, filepath: str):
        """Save config to JSON"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "RLTrainingConfig":
        """Load config from JSON"""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)
