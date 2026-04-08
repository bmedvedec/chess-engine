"""
SELF-PLAY - Configuration
"""

from dataclasses import dataclass

# Constants — keep in sync with RLTrainingConfig in rl/rl_config.py
DEFAULT_MAX_MOVES = 150
DEFAULT_TEMPERATURE = 1.5
DEFAULT_TEMPERATURE_THRESHOLD = 50
LATE_GAME_TEMPERATURE = 0.3
DEFAULT_NUM_SIMULATIONS = 100
DEFAULT_C_PUCT = 1.5
DEFAULT_DIRICHLET_ALPHA = 0.5
DEFAULT_DIRICHLET_EPSILON = 0.35
DEFAULT_RESIGN_THRESHOLD = -0.45
DEFAULT_VALUE_BLEND_ALPHA = 0.30
DEFAULT_DRAW_VALUE_PENALTY = 0.50


@dataclass
class SelfPlayConfig:
    """Configuration for self-play."""

    num_simulations: int = DEFAULT_NUM_SIMULATIONS
    c_puct: float = DEFAULT_C_PUCT
    temperature: float = DEFAULT_TEMPERATURE
    temperature_threshold: int = DEFAULT_TEMPERATURE_THRESHOLD
    max_moves: int = DEFAULT_MAX_MOVES
    use_rnn: bool = False
    rnn_max_history: int = 15
    late_game_temperature: float = LATE_GAME_TEMPERATURE
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA
    dirichlet_epsilon: float = DEFAULT_DIRICHLET_EPSILON
    resign_threshold: float = DEFAULT_RESIGN_THRESHOLD
    value_blend_alpha: float = DEFAULT_VALUE_BLEND_ALPHA
    draw_value_penalty: float = DEFAULT_DRAW_VALUE_PENALTY

    @classmethod
    def from_args(cls, args) -> "SelfPlayConfig":
        """Create config from command line arguments."""
        return cls(
            num_simulations=getattr(args, "simulations", DEFAULT_NUM_SIMULATIONS),
            c_puct=getattr(args, "c_puct", DEFAULT_C_PUCT),
            temperature=getattr(args, "temperature", DEFAULT_TEMPERATURE),
            temperature_threshold=getattr(
                args, "temp_threshold", DEFAULT_TEMPERATURE_THRESHOLD
            ),
            max_moves=getattr(args, "max_moves", DEFAULT_MAX_MOVES),
            use_rnn=getattr(args, "use_rnn", False),
            rnn_max_history=getattr(args, "rnn_max_history", 15),
            dirichlet_alpha=getattr(args, "dirichlet_alpha", DEFAULT_DIRICHLET_ALPHA),
            dirichlet_epsilon=getattr(
                args, "dirichlet_epsilon", DEFAULT_DIRICHLET_EPSILON
            ),
            resign_threshold=getattr(
                args, "resign_threshold", DEFAULT_RESIGN_THRESHOLD
            ),
            value_blend_alpha=getattr(
                args, "value_blend_alpha", DEFAULT_VALUE_BLEND_ALPHA
            ),
            draw_value_penalty=getattr(
                args, "draw_value_penalty", DEFAULT_DRAW_VALUE_PENALTY
            ),
        )
