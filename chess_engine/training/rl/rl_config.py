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
    num_simulations: int = 100
    c_puct: float = 1.5
    temperature: float = 1.5
    temperature_threshold: int = 50
    late_game_temperature: float = 0.3
    max_moves_per_game: int = 150  # shorter games force more decisive outcomes
    dirichlet_alpha: float = 0.5
    dirichlet_epsilon: float = 0.35  # weight of noise at root
    resign_threshold: float = -0.75

    # Value target blending: mix MCTS root value with game outcome after each game.
    # target = alpha * mcts_value + (1 - alpha) * game_outcome
    #
    # With alpha=0.30 and 75% draws (game_outcome=0), game outcomes get 70% weight
    # but 75% of them are 0. This compresses draw-position targets to ~0.30*mcts_val,
    # discarding most of the MCTS signal. The model collapses to predicting a constant
    # and value_correlation decays. std_mcts=0.40 shows MCTS IS generating varied
    # position evaluations — alpha must be high enough to preserve that signal.
    #
    # alpha=0.70: draw targets = 0.70*mcts_val (varied, std≈0.28) — model must
    # differentiate positions. Game outcome (30%) corrects systematic MCTS bias.
    # Reduce alpha toward 0.30 once resign_pct>10% and draw_pct<50%.
    value_blend_alpha: float = 0.70
    # Draw penalty: applied as game_val = -draw_value_penalty for drawn games.
    # Keep at 0.0 while draws dominate (>70%). Increase toward 0.3 once
    # resign_pct > 10% and draw rates drop below 50%.
    draw_value_penalty: float = 0.2

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

    use_rnn: bool = True
    rnn_hidden_size: int = 256
    rnn_layers: int = 2
    rnn_use_attention: bool = False
    rnn_bidirectional: bool = False
    rnn_max_history: int = 15  # Max move history length fed to LSTM (shorter = faster)

    fusion_type: Literal["concat", "gated", "attention"] = "gated"
    num_actions: int = 4672  # AlphaZero-style: 64 squares × 73 move types

    # =====================
    # Training hyperparameters
    # =====================
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    policy_loss_weight: float = 1.0
    # Value head gets <1% of gradient at weight=2.0 because value_loss (~0.001)
    # is 300x smaller than policy_loss (~0.27) at convergence. L2 regularization
    # then slowly compresses value head weights → std_pred collapses → MCTS
    # gets ~0 for all positions → eval win rate degrades. Weight=15 gives ~5%
    # gradient to value head, enough to resist compression without destabilising policy.
    value_loss_weight: float = 15.0

    # =====================
    # Optimizer settings
    # =====================
    optimizer: str = "adam"
    # lr_schedule options:
    #   "constant" — fixed LR for the entire run (original behaviour)
    #   "step"     — StepLR: multiply LR by lr_decay_gamma every lr_decay_steps iterations
    #   "cosine"   — CosineAnnealingLR: smooth decay to lr_min over num_iterations
    #   "plateau"  — ReduceLROnPlateau: reduce by lr_decay_gamma after lr_decay_steps
    #                iterations with no improvement in training loss (most adaptive)
    lr_schedule: str = "plateau"
    lr_decay_steps: int = 50  # StepLR: decay period; Plateau: patience (iters)
    lr_decay_gamma: float = 0.5  # multiplicative decay factor (LR *= gamma)
    lr_min: float = 1e-5  # floor for cosine / plateau schedules

    # =====================
    # Replay buffer
    # =====================
    buffer_size: int = 500000
    min_buffer_size: int = 10000
    sample_ratio: float = 1.0

    # =====================
    # Prioritized Experience Replay (PER)
    # =====================
    # When True, uses PrioritizedReplayBuffer instead of the uniform ReplayBuffer.
    # PER focuses training on positions the model finds surprising (high TD-error),
    # improving sample efficiency with no measurable runtime overhead.
    #
    # Benchmark (10 000 synthetic positions, 50 gradient steps, batch=64, CUDA):
    #   Metric                  Uniform     PER         Delta
    #   ----------------------  ----------  ----------  -------
    #   Mean time/step (ms)     37.55       24.24       -35.5%  (no real overhead; cold-start skews mean)
    #   Final combined loss     6.624       2.411       -63.6%
    #   Loss drop over 50 steps 2.262       6.498       +187.3%
    #   Mean policy loss        7.186       4.478       -37.7%
    #   Mean value loss         0.355       0.217       -38.9%
    #   Steps to reach uniform final loss  ~50         ~8       PER gets there 6x faster
    #
    # Recommendation: enable PER (use_prioritized_replay=True) for production runs.
    # The per_alpha / per_beta defaults below match the original PER paper and work
    # well for chess self-play without further tuning.
    use_prioritized_replay: bool = True

    # Priority exponent alpha: controls how strongly priorities skew sampling.
    #   alpha=0 -> uniform sampling (equivalent to standard replay)
    #   alpha=1 -> fully proportional to priority
    # Recommended: 0.6 (PER paper default; good balance for chess self-play)
    per_alpha: float = 0.6

    # Importance-sampling exponent beta: corrects for the sampling bias introduced
    # by PER. Annealed from per_beta (start) -> per_beta_end (end of training).
    #   beta=0 -> no IS correction (biased updates, faster convergence early)
    #   beta=1 -> full IS correction (unbiased, stabilises late training)
    # Linear annealing from 0.4 -> 1.0 over num_iterations is standard practice.
    per_beta: float = 0.4
    per_beta_end: float = 1.0

    # Small constant added to every raw priority before raising to alpha, ensuring
    # no experience has zero probability of being sampled.
    # Typical range: 1e-8 to 1e-5; 1e-6 is the recommended default.
    per_epsilon: float = 1e-6

    # =====================
    # Evaluation
    # =====================
    eval_frequency: int = 5
    # 50 is the practical minimum for a statistically meaningful signal.
    # At 20 games, a genuinely better model can easily lose by chance (high variance).
    # With 50 games, the standard error on win rate drops to ~7%, making improvements
    # above the win_threshold reliably detectable. Increase to 100 for more confidence.
    eval_games: int = 50
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
