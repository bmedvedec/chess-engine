"""
SELF-PLAY - Configuration
"""

from dataclasses import dataclass

# Constants
DEFAULT_MAX_MOVES = 200
DEFAULT_TEMPERATURE = 1.5
DEFAULT_TEMPERATURE_THRESHOLD = 15
LATE_GAME_TEMPERATURE = 0.1
DEFAULT_NUM_SIMULATIONS = 100
DEFAULT_C_PUCT = 2.0
DEFAULT_DIRICHLET_ALPHA = 0.3
DEFAULT_RESIGN_THRESHOLD = -0.9


@dataclass
class SelfPlayConfig:
    """Configuration for self-play."""

    num_simulations: int = DEFAULT_NUM_SIMULATIONS
    c_puct: float = DEFAULT_C_PUCT
    temperature: float = DEFAULT_TEMPERATURE
    temperature_threshold: int = DEFAULT_TEMPERATURE_THRESHOLD
    max_moves: int = DEFAULT_MAX_MOVES
    use_rnn: bool = False
    late_game_temperature: float = LATE_GAME_TEMPERATURE
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA
    resign_threshold: float = DEFAULT_RESIGN_THRESHOLD

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
            dirichlet_alpha=getattr(args, "dirichlet_alpha", DEFAULT_DIRICHLET_ALPHA),
            resign_threshold=getattr(
                args, "resign_threshold", DEFAULT_RESIGN_THRESHOLD
            ),
        )
