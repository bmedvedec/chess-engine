"""
HYPERPARAMETER CONFIGURATION

Centralized configuration system: dataclasses for all hyperparameters,
parameter ranges for systematic search, validation, serialization, and presets.
"""

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
from enum import Enum
import copy
import numpy as np


class SearchSpace(Enum):
    """Types of search spaces for hyperparameters"""

    CATEGORICAL = "categorical"  # Discrete choices
    INT_LINEAR = "int_linear"  # Integer range, linear sampling
    INT_LOG = "int_log"  # Integer range, log-uniform sampling
    FLOAT_LINEAR = "float_linear"  # Float range, linear sampling
    FLOAT_LOG = "float_log"  # Float range, log-uniform sampling


@dataclass
class ParameterRange:
    """
    Definition of a hyperparameter's search range.

    Attributes:
        name: Parameter name (dot notation for nested, e.g., 'model.cnn_blocks')
        space_type: Type of search space
        values: For categorical, list of choices; for ranges, (min, max)
        default: Default value if not tuned
        description: Human-readable description
    """

    name: str
    space_type: SearchSpace
    values: Union[List[Any], Tuple[float, float]]
    default: Any
    description: str = ""

    def sample(self, rng=None) -> Any:
        """Sample a value from this parameter's range"""
        import numpy as np

        if rng is None:
            rng = np.random

        if self.space_type == SearchSpace.CATEGORICAL:
            idx = rng.randint(0, len(self.values))
            return self.values[idx]
        elif self.space_type == SearchSpace.INT_LINEAR:
            low, high = int(self.values[0]), int(self.values[1])
            return int(rng.randint(low, high + 1))
        elif self.space_type == SearchSpace.INT_LOG:
            log_min, log_max = math.log(self.values[0]), math.log(self.values[1])
            return int(round(math.exp(rng.uniform(log_min, log_max))))
        elif self.space_type == SearchSpace.FLOAT_LINEAR:
            return float(rng.uniform(self.values[0], self.values[1]))
        elif self.space_type == SearchSpace.FLOAT_LOG:
            log_min, log_max = math.log(self.values[0]), math.log(self.values[1])
            return float(math.exp(rng.uniform(log_min, log_max)))
        else:
            raise ValueError(f"Unknown space type: {self.space_type}")


@dataclass
class ModelConfig:
    """Model architecture hyperparameters"""

    # CNN parameters
    cnn_input_channels: int = 22
    cnn_filters: int = 256
    cnn_residual_blocks: int = 10
    cnn_dropout: float = 0.0

    # RNN parameters
    use_rnn: bool = False
    rnn_embedding_dim: int = 64
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 2
    rnn_dropout: float = 0.3
    rnn_bidirectional: bool = False
    rnn_use_attention: bool = False
    rnn_use_layer_norm: bool = True

    # Fusion parameters
    fusion_type: str = "gated"  # 'concat', 'gated', 'attention'


@dataclass
class TrainingConfig:
    """Training hyperparameters"""

    # Optimizer
    optimizer: str = "adam"  # 'adam', 'sgd', 'adamw'
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD
    betas: Tuple[float, float] = (0.9, 0.999)  # For Adam

    # Batch and steps
    batch_size: int = 256
    gradient_accumulation_steps: int = 1

    # Learning rate schedule
    lr_schedule: str = "constant"  # 'constant', 'step', 'cosine', 'plateau'
    lr_decay_steps: int = 10
    lr_decay_gamma: float = 0.5
    warmup_epochs: int = 2
    min_lr: float = 1e-7

    # Loss weights
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    # Regularization
    label_smoothing: float = 0.0
    gradient_clip: float = 1.0

    # Training duration (for RL)
    num_iterations: int = 100
    training_steps_per_iteration: int = 1000

    # Early stopping
    early_stopping_patience: int = 10

    # Mixed precision
    use_mixed_precision: bool = True


@dataclass
class MCTSConfig:
    """MCTS search hyperparameters"""

    num_simulations: int = 200
    c_puct: float = 1.5
    temperature: float = 1.0
    temperature_threshold: int = 15  # Move number after which temp drops

    # Dirichlet noise (for exploration)
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25

    # Caching
    enable_caching: bool = True
    max_cache_size: int = 10000

    # Batch evaluation
    eval_batch_size: int = 8

    # Progressive widening
    use_progressive_widening: bool = False

    # Early termination
    enable_early_termination: bool = True
    early_termination_threshold: float = 0.9


@dataclass
class SelfPlayConfig:
    """Self-play generation hyperparameters"""

    games_per_iteration: int = 100
    max_moves_per_game: int = 200

    # Parallelization
    use_parallel: bool = True
    num_workers: Optional[int] = None  # None = auto

    # Data augmentation
    use_augmentation: bool = True
    flip_probability: float = 0.5


@dataclass
class ReplayBufferConfig:
    """Replay buffer hyperparameters"""

    buffer_size: int = 500000
    min_buffer_size: int = 10000
    sample_ratio: float = 1.0
    prioritized: bool = False
    priority_alpha: float = 0.6
    priority_beta: float = 0.4


@dataclass
class EvaluationConfig:
    """Evaluation hyperparameters"""

    eval_frequency: int = 5
    eval_games: int = 20
    eval_simulations: int = 100
    win_threshold: float = 0.55

    # Stockfish comparison
    stockfish_levels: List[int] = field(default_factory=lambda: [1, 3, 5])
    stockfish_time_limit: float = 0.1


def _make_json_safe(obj: Any) -> Any:
    """
    Convert numpy types to native Python types for JSON serialization.

    This is a module-level utility function used by multiple classes.
    """
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
class HyperparameterConfig:
    """
    Complete hyperparameter configuration for chess engine.

    Aggregates all configuration sections into a single dataclass.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # Experiment metadata
    name: str = "default"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to nested dictionary (JSON-safe)"""
        raw_dict = {
            "name": self.name,
            "description": self.description,
            "model": asdict(self.model),
            "training": asdict(self.training),
            "mcts": asdict(self.mcts),
            "self_play": asdict(self.self_play),
            "buffer": asdict(self.buffer),
            "evaluation": asdict(self.evaluation),
        }
        return _make_json_safe(raw_dict)

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "HyperparameterConfig":
        """Load configuration from JSON file"""
        with open(filepath, "r") as f:
            data = json.load(f)

        return cls(
            name=data.get("name", "loaded"),
            description=data.get("description", ""),
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            mcts=MCTSConfig(**data.get("mcts", {})),
            self_play=SelfPlayConfig(**data.get("self_play", {})),
            buffer=ReplayBufferConfig(**data.get("buffer", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
        )

    def copy(self) -> "HyperparameterConfig":
        """Create a deep copy of this configuration"""
        return copy.deepcopy(self)

    def get_param(self, dotted_name: str) -> Any:
        """
        Get parameter value using dot notation.

        Example: config.get_param('model.cnn_blocks') returns model.cnn_blocks
        """
        parts = dotted_name.split(".")
        obj = self
        for part in parts:
            obj = getattr(obj, part)
        return obj

    def set_param(self, dotted_name: str, value: Any) -> None:
        """
        Set parameter value using dot notation.

        Example: config.set_param('model.cnn_blocks', 15)
        """
        parts = dotted_name.split(".")
        obj = self
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of warnings/errors.

        Returns:
            List of validation messages (empty if valid)
        """
        issues = []

        # Model validation
        if self.model.cnn_filters < 32:
            issues.append("Warning: cnn_filters < 32 may limit model capacity")
        if self.model.cnn_residual_blocks < 1:
            issues.append("Error: cnn_residual_blocks must be >= 1")
        if self.model.use_rnn and self.model.rnn_hidden_size < 64:
            issues.append("Warning: rnn_hidden_size < 64 may limit sequence modeling")

        # Training validation
        if self.training.learning_rate > 0.1:
            issues.append("Warning: learning_rate > 0.1 may cause training instability")
        if self.training.learning_rate < 1e-6:
            issues.append(
                "Warning: learning_rate < 1e-6 may cause very slow convergence"
            )
        if self.training.batch_size < 16:
            issues.append("Warning: batch_size < 16 may cause noisy gradients")

        # MCTS validation
        if self.mcts.num_simulations < 10:
            issues.append("Warning: num_simulations < 10 provides weak search")
        if self.mcts.c_puct < 0.1:
            issues.append("Warning: c_puct < 0.1 may under-explore")
        if self.mcts.c_puct > 10.0:
            issues.append("Warning: c_puct > 10.0 may over-explore")

        # Buffer validation
        if self.buffer.buffer_size < self.buffer.min_buffer_size:
            issues.append("Error: buffer_size must be >= min_buffer_size")

        return issues


# =============================================================================
# SEARCH SPACE DEFINITIONS
# =============================================================================


def get_default_search_space() -> Dict[str, ParameterRange]:
    """
    Get the default search space for hyperparameter tuning.

    Returns dictionary mapping parameter names to their ranges.
    """
    return {
        # Model architecture
        "model.cnn_filters": ParameterRange(
            name="model.cnn_filters",
            space_type=SearchSpace.CATEGORICAL,
            values=[64, 128, 256, 384],
            default=256,
            description="Number of CNN filters/channels",
        ),
        "model.cnn_residual_blocks": ParameterRange(
            name="model.cnn_residual_blocks",
            space_type=SearchSpace.INT_LINEAR,
            values=(3, 20),
            default=10,
            description="Number of residual blocks in CNN",
        ),
        "model.cnn_dropout": ParameterRange(
            name="model.cnn_dropout",
            space_type=SearchSpace.FLOAT_LINEAR,
            values=(0.0, 0.3),
            default=0.0,
            description="CNN dropout rate",
        ),
        "model.rnn_hidden_size": ParameterRange(
            name="model.rnn_hidden_size",
            space_type=SearchSpace.CATEGORICAL,
            values=[128, 256, 384, 512],
            default=256,
            description="RNN hidden layer size",
        ),
        "model.rnn_num_layers": ParameterRange(
            name="model.rnn_num_layers",
            space_type=SearchSpace.INT_LINEAR,
            values=(1, 4),
            default=2,
            description="Number of RNN layers",
        ),
        "model.rnn_dropout": ParameterRange(
            name="model.rnn_dropout",
            space_type=SearchSpace.FLOAT_LINEAR,
            values=(0.0, 0.5),
            default=0.3,
            description="RNN dropout rate",
        ),
        "model.fusion_type": ParameterRange(
            name="model.fusion_type",
            space_type=SearchSpace.CATEGORICAL,
            values=["concat", "gated", "attention"],
            default="gated",
            description="CNN-RNN fusion strategy",
        ),
        # Training parameters
        "training.learning_rate": ParameterRange(
            name="training.learning_rate",
            space_type=SearchSpace.FLOAT_LOG,
            values=(1e-5, 1e-2),
            default=0.001,
            description="Initial learning rate",
        ),
        "training.weight_decay": ParameterRange(
            name="training.weight_decay",
            space_type=SearchSpace.FLOAT_LOG,
            values=(1e-6, 1e-2),
            default=1e-4,
            description="Weight decay (L2 regularization)",
        ),
        "training.batch_size": ParameterRange(
            name="training.batch_size",
            space_type=SearchSpace.CATEGORICAL,
            values=[64, 128, 256, 512, 1024],
            default=256,
            description="Training batch size",
        ),
        "training.lr_schedule": ParameterRange(
            name="training.lr_schedule",
            space_type=SearchSpace.CATEGORICAL,
            values=["constant", "step", "cosine", "plateau"],
            default="constant",
            description="Learning rate schedule",
        ),
        "training.lr_decay_gamma": ParameterRange(
            name="training.lr_decay_gamma",
            space_type=SearchSpace.FLOAT_LINEAR,
            values=(0.1, 0.9),
            default=0.5,
            description="LR decay factor (for step schedule)",
        ),
        "training.warmup_epochs": ParameterRange(
            name="training.warmup_epochs",
            space_type=SearchSpace.INT_LINEAR,
            values=(0, 5),
            default=2,
            description="Number of warmup epochs",
        ),
        "training.policy_loss_weight": ParameterRange(
            name="training.policy_loss_weight",
            space_type=SearchSpace.FLOAT_LINEAR,
            values=(0.5, 2.0),
            default=1.0,
            description="Weight for policy loss",
        ),
        "training.value_loss_weight": ParameterRange(
            name="training.value_loss_weight",
            space_type=SearchSpace.FLOAT_LINEAR,
            values=(0.5, 2.0),
            default=1.0,
            description="Weight for value loss",
        ),
        "training.label_smoothing": ParameterRange(
            name="training.label_smoothing",
            space_type=SearchSpace.FLOAT_LINEAR,
            values=(0.0, 0.2),
            default=0.0,
            description="Label smoothing factor",
        ),
        "training.gradient_clip": ParameterRange(
            name="training.gradient_clip",
            space_type=SearchSpace.FLOAT_LINEAR,
            values=(0.5, 5.0),
            default=1.0,
            description="Gradient clipping norm",
        ),
        # MCTS parameters
        "mcts.num_simulations": ParameterRange(
            name="mcts.num_simulations",
            space_type=SearchSpace.INT_LOG,
            values=(50, 800),
            default=200,
            description="MCTS simulations per move",
        ),
        "mcts.c_puct": ParameterRange(
            name="mcts.c_puct",
            space_type=SearchSpace.FLOAT_LINEAR,
            values=(0.5, 4.0),
            default=1.5,
            description="MCTS exploration constant",
        ),
        "mcts.temperature": ParameterRange(
            name="mcts.temperature",
            space_type=SearchSpace.FLOAT_LINEAR,
            values=(0.5, 2.0),
            default=1.0,
            description="Move selection temperature",
        ),
        "mcts.temperature_threshold": ParameterRange(
            name="mcts.temperature_threshold",
            space_type=SearchSpace.INT_LINEAR,
            values=(10, 30),
            default=15,
            description="Move number for temperature drop",
        ),
        "mcts.dirichlet_alpha": ParameterRange(
            name="mcts.dirichlet_alpha",
            space_type=SearchSpace.FLOAT_LINEAR,
            values=(0.1, 1.0),
            default=0.3,
            description="Dirichlet noise alpha",
        ),
        "mcts.dirichlet_epsilon": ParameterRange(
            name="mcts.dirichlet_epsilon",
            space_type=SearchSpace.FLOAT_LINEAR,
            values=(0.0, 0.5),
            default=0.25,
            description="Dirichlet noise weight",
        ),
        # Self-play parameters
        "self_play.games_per_iteration": ParameterRange(
            name="self_play.games_per_iteration",
            space_type=SearchSpace.INT_LOG,
            values=(25, 500),
            default=100,
            description="Games per RL iteration",
        ),
        # Buffer parameters
        "buffer.buffer_size": ParameterRange(
            name="buffer.buffer_size",
            space_type=SearchSpace.INT_LOG,
            values=(100000, 2000000),
            default=500000,
            description="Replay buffer capacity",
        ),
    }


def get_quick_search_space() -> Dict[str, ParameterRange]:
    """
    Get a reduced search space for quick experiments.

    Focuses on the most impactful hyperparameters.
    """
    full_space = get_default_search_space()
    quick_params = [
        "training.learning_rate",
        "training.batch_size",
        "model.cnn_residual_blocks",
        "mcts.num_simulations",
        "mcts.c_puct",
    ]
    return {k: v for k, v in full_space.items() if k in quick_params}


def get_architecture_search_space() -> Dict[str, ParameterRange]:
    """Get search space focused on model architecture."""
    full_space = get_default_search_space()
    arch_params = [k for k in full_space.keys() if k.startswith("model.")]
    return {k: v for k, v in full_space.items() if k in arch_params}


def get_training_search_space() -> Dict[str, ParameterRange]:
    """Get search space focused on training hyperparameters."""
    full_space = get_default_search_space()
    train_params = [k for k in full_space.keys() if k.startswith("training.")]
    return {k: v for k, v in full_space.items() if k in train_params}


def get_mcts_search_space() -> Dict[str, ParameterRange]:
    """Get search space focused on MCTS parameters."""
    full_space = get_default_search_space()
    mcts_params = [k for k in full_space.keys() if k.startswith("mcts.")]
    return {k: v for k, v in full_space.items() if k in mcts_params}


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================


def get_preset_configs() -> Dict[str, HyperparameterConfig]:
    """
    Get preset configurations for different use cases.

    Returns:
        Dictionary mapping preset names to configurations
    """
    presets = {}

    # Fast development preset (minimal resources)
    fast_dev = HyperparameterConfig(
        name="fast_dev",
        description="Fast development/testing configuration",
    )
    fast_dev.model.cnn_residual_blocks = 3
    fast_dev.model.cnn_filters = 64
    fast_dev.training.batch_size = 64
    fast_dev.training.num_iterations = 10
    fast_dev.mcts.num_simulations = 50
    fast_dev.self_play.games_per_iteration = 20
    fast_dev.buffer.buffer_size = 50000
    presets["fast_dev"] = fast_dev

    # Balanced preset (moderate resources)
    balanced = HyperparameterConfig(
        name="balanced",
        description="Balanced configuration for typical use",
    )
    balanced.model.cnn_residual_blocks = 10
    balanced.model.cnn_filters = 256
    balanced.training.batch_size = 256
    balanced.training.num_iterations = 50
    balanced.mcts.num_simulations = 200
    balanced.self_play.games_per_iteration = 100
    presets["balanced"] = balanced

    # Strong preset (high resources)
    strong = HyperparameterConfig(
        name="strong",
        description="Strong configuration with more compute",
    )
    strong.model.cnn_residual_blocks = 15
    strong.model.cnn_filters = 256
    strong.training.batch_size = 512
    strong.training.num_iterations = 100
    strong.mcts.num_simulations = 400
    strong.self_play.games_per_iteration = 200
    strong.buffer.buffer_size = 1000000
    presets["strong"] = strong

    # AlphaZero-like preset
    alphazero = HyperparameterConfig(
        name="alphazero",
        description="AlphaZero-inspired configuration",
    )
    alphazero.model.cnn_residual_blocks = 19
    alphazero.model.cnn_filters = 256
    alphazero.model.use_rnn = False
    alphazero.training.learning_rate = 0.2  # AlphaZero uses high LR with momentum
    alphazero.training.optimizer = "sgd"
    alphazero.training.momentum = 0.9
    alphazero.training.weight_decay = 1e-4
    alphazero.training.batch_size = 4096
    alphazero.training.lr_schedule = "step"
    alphazero.mcts.num_simulations = 800
    alphazero.mcts.c_puct = 1.25
    alphazero.mcts.dirichlet_alpha = 0.3
    alphazero.mcts.dirichlet_epsilon = 0.25
    presets["alphazero"] = alphazero

    # Hybrid CNN-RNN preset
    hybrid = HyperparameterConfig(
        name="hybrid",
        description="Hybrid CNN-RNN configuration",
    )
    hybrid.model.cnn_residual_blocks = 10
    hybrid.model.cnn_filters = 256
    hybrid.model.use_rnn = True
    hybrid.model.rnn_hidden_size = 256
    hybrid.model.rnn_num_layers = 2
    hybrid.model.rnn_use_attention = True
    hybrid.model.fusion_type = "gated"
    presets["hybrid"] = hybrid

    return presets


def create_config_from_preset(preset_name: str) -> HyperparameterConfig:
    """
    Create a configuration from a preset name.

    Args:
        preset_name: Name of preset ('fast_dev', 'balanced', 'strong', etc.)

    Returns:
        HyperparameterConfig instance
    """
    presets = get_preset_configs()
    if preset_name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    return presets[preset_name].copy()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def generate_random_config(
    search_space: Optional[Dict[str, ParameterRange]] = None,
    base_config: Optional[HyperparameterConfig] = None,
    seed: Optional[int] = None,
) -> HyperparameterConfig:
    """
    Generate a random configuration by sampling from search space.

    Args:
        search_space: Parameter ranges to sample from (default: full search space)
        base_config: Base configuration to modify (default: new default config)
        seed: Random seed for reproducibility

    Returns:
        New HyperparameterConfig with sampled values
    """
    import numpy as np

    if search_space is None:
        search_space = get_default_search_space()

    if base_config is None:
        config = HyperparameterConfig()
    else:
        config = base_config.copy()

    rng = np.random.RandomState(seed)

    for param_name, param_range in search_space.items():
        value = param_range.sample(rng)
        config.set_param(param_name, value)

    return config


def config_to_flat_dict(config: HyperparameterConfig) -> Dict[str, Any]:
    """
    Flatten a config to a single-level dictionary with dotted keys.

    Useful for logging and comparison.
    """
    flat = {}

    def flatten(obj, prefix=""):
        if hasattr(obj, "__dataclass_fields__"):
            for field_name in obj.__dataclass_fields__:
                value = getattr(obj, field_name)
                key = f"{prefix}.{field_name}" if prefix else field_name
                flatten(value, key)
        else:
            flat[prefix] = obj

    flatten(config)
    return flat


def compare_configs(
    config1: HyperparameterConfig,
    config2: HyperparameterConfig,
) -> Dict[str, Tuple[Any, Any]]:
    """
    Compare two configurations and return differences.

    Returns:
        Dictionary mapping parameter names to (value1, value2) tuples
    """
    flat1 = config_to_flat_dict(config1)
    flat2 = config_to_flat_dict(config2)

    differences = {}
    all_keys = set(flat1.keys()) | set(flat2.keys())

    for key in all_keys:
        v1 = flat1.get(key)
        v2 = flat2.get(key)
        if v1 != v2:
            differences[key] = (v1, v2)

    return differences


if __name__ == "__main__":
    # Demo usage
    print("=== Hyperparameter Configuration System ===\n")

    # Create default config
    config = HyperparameterConfig(name="test", description="Test configuration")
    print(f"Default config created: {config.name}")

    # Validate
    issues = config.validate()
    print(f"Validation: {len(issues)} issues")

    # Show search space
    search_space = get_default_search_space()
    print(f"\nDefault search space: {len(search_space)} parameters")

    # Generate random config
    random_config = generate_random_config(seed=42)
    print(f"\nRandom config generated: {random_config.name}")

    # Compare
    diff = compare_configs(config, random_config)
    print(f"Differences from default: {len(diff)} parameters")

    # Show presets
    presets = get_preset_configs()
    print(f"\nAvailable presets: {list(presets.keys())}")

    # Save and load
    config.save("/tmp/test_config.json")
    loaded = HyperparameterConfig.load("/tmp/test_config.json")
    print(f"\nSaved and loaded config: {loaded.name}")
