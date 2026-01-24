"""
REINFORCEMENT LEARNING TRAINING LOOP

This module implements the complete RL training pipeline:
1. Self-play game generation using MCTS
2. Network training on collected experience
3. Model evaluation and selection
4. Iterative improvement loop

Key Features:
- Integrated self-play and training
- Replay buffer management
- Model evaluation via head-to-head matches
- Best model tracking and checkpointing
- Comprehensive metrics and logging
- Training resumption support
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN messages
import time
import json
import argparse
from typing import Optional, Dict, List, Tuple, Any, cast, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Mixed precision imports
from torch.amp.grad_scaler import GradScaler
from torch import autocast

AMP_DEVICE = "cuda"

import chess
from tqdm import tqdm

from chess_engine.models.hybrid_model import HybridChessNet
from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder
from chess_engine.data.replay_buffer import ReplayBuffer, GameExample
from chess_engine.search.mcts import MCTS
from chess_engine.training.self_play import (
    SelfPlayWorker,
    SelfPlayConfig,
    ParallelSelfPlayWorker,
)


def _make_json_safe(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    import numpy as np

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

    # Training iterations
    num_iterations: int = 100
    games_per_iteration: int = 100
    training_steps_per_iteration: int = 1000

    # Self-play configuration
    num_simulations: int = 200
    c_puct: float = 2.0
    temperature: float = 1.5
    temperature_threshold: int = 15
    max_moves_per_game: int = 200
    dirichlet_alpha: float = 0.3
    resign_threshold: float = -0.9

    # ASYMMETRIC SELF-PLAY (to break self-play symmetry)
    asymmetric_play: bool = True  # Enable asymmetric self-play
    temperature_variation: float = 0.3  # Random variation range (±N)
    white_temperature: Optional[float] = None  # Override temp for white
    black_temperature: Optional[float] = None  # Override temp for black

    # MIXED TRAINING OPPONENTS (to break self-play symmetry)
    use_mixed_opponents: bool = False  # Enable mixed opponent training
    self_play_ratio: float = 0.70  # 70% pure self-play
    old_opponent_ratio: float = 0.20  # 20% vs older checkpoint
    random_opponent_ratio: float = 0.10  # 10% vs random player
    old_opponent_iterations_back: int = 2  # Use checkpoint N iterations ago

    # VALUE TARGET SHARPENING (to amplify win/loss signals)
    use_value_sharpening: bool = True  # Enable value target sharpening
    value_sharpening_factor: float = 1.5  # Amplification factor for wins/losses
    draw_value_penalty: float = 0.0  # Optional penalty for draws (0.0 = no penalty)

    # Parallel self-play
    use_parallel_selfplay: bool = True
    num_workers: Optional[int] = None  # None = use all cores

    # Model architecture
    cnn_blocks: int = 10
    use_rnn: bool = False

    # Training hyperparameters
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    # Optimizer settings
    optimizer: str = "adam"  # 'adam' or 'sgd'
    lr_schedule: str = "constant"  # 'constant', 'step', 'cosine'
    lr_decay_steps: int = 10  # For step schedule
    lr_decay_gamma: float = 0.5

    # Replay buffer
    buffer_size: int = 500000
    min_buffer_size: int = 10000  # Minimum before training
    sample_ratio: float = 1.0  # Ratio of buffer to sample per training step

    # Evaluation
    eval_frequency: int = 5  # Evaluate every N iterations
    eval_games: int = 20  # Games for evaluation
    eval_simulations: int = 100  # MCTS simulations for evaluation
    win_threshold: float = 0.55  # Win rate to replace best model

    # Checkpointing
    checkpoint_dir: str = "data/rl_checkpoints"
    save_frequency: int = 1  # Save every N iterations
    keep_checkpoints: int = 20  # Number of checkpoints to keep

    # Logging
    log_dir: str = "logs/rl_training"
    log_frequency: int = 10  # Log every N training steps

    # Device
    device: str = "cuda"  # 'cuda' or 'cpu'
    use_amp: bool = True  # Enable mixed precision training (AMP)

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


class RLDataset(Dataset):
    """PyTorch Dataset for RL training from replay buffer"""

    def __init__(
        self,
        examples: List[GameExample],
        board_encoder: BoardEncoder,
        move_encoder: MoveEncoder,
    ):
        """
        Initialize dataset.

        Args:
            examples: List of training examples from replay buffer
            board_encoder: Board encoding utility
            move_encoder: Move encoding utility
        """
        self.examples = examples
        self.board_encoder = board_encoder
        self.move_encoder = move_encoder

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get single training example.

        Returns:
            Tuple of (board_tensor, policy_tensor, value)
        """
        example = self.examples[idx]

        # Handle both GameExample objects and dictionary format
        if isinstance(example, dict):
            fen = example["fen"]
            policy_dict = example["policy"]
            value_target = example["value"]
        else:
            # GameExample object
            fen = example.fen
            policy_dict = example.policy
            value_target = example.value

        # Parse board from FEN
        board = chess.Board(fen)

        # Encode board
        board_tensor = self.board_encoder.board_to_tensor(board)

        # Encode policy
        policy_tensor = torch.zeros(4096)
        for move_uci, prob in policy_dict.items():
            try:
                move = chess.Move.from_uci(move_uci)
                move_idx = self.move_encoder.encode_move(move)
                policy_tensor[move_idx] = prob
            except:
                # Skip invalid moves
                pass

        # Normalize policy (ensure it sums to 1)
        if policy_tensor.sum() > 0:
            policy_tensor = policy_tensor / policy_tensor.sum()

        # Get value
        value = torch.tensor([value_target], dtype=torch.float32)

        return board_tensor, policy_tensor, value


class RLTrainer:
    """
    Reinforcement Learning Trainer for Chess Engine.

    Implements complete AlphaZero-style training loop:
    - Self-play game generation
    - Network training
    - Model evaluation
    - Iterative improvement
    """

    def __init__(
        self,
        config: RLTrainingConfig,
        model: Optional[nn.Module] = None,
        resume_from: Optional[str] = None,
        pretrained_path: Optional[str] = None,
    ):
        """
        Initialize RL trainer.

        Args:
            config: Training configuration
            model: Optional pre-initialized model
            resume_from: Optional checkpoint path to resume from
        """
        self.config = config

        # Setup device
        self.device = torch.device(
            config.device
            if torch.cuda.is_available() and config.device == "cuda"
            else "cpu"
        )
        print(f"\n🖥️  Using device: {self.device}")

        # Create or load model
        if model is None:
            model = HybridChessNet(
                cnn_residual_blocks=config.cnn_blocks, use_rnn=config.use_rnn
            )
        self.model = model.to(self.device)

        # Load pretrained weights (Transfer Learning)
        if pretrained_path and not resume_from:
            print(f"\n📥 Loading pretrained weights from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device)

            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint

            self.model.load_state_dict(state_dict)
            print("   ✅ Weights loaded successfully")

        # Best model (for evaluation comparison)
        self.best_model = HybridChessNet(
            cnn_residual_blocks=config.cnn_blocks, use_rnn=config.use_rnn
        ).to(self.device)
        self.best_model.load_state_dict(self.model.state_dict())

        # Create encoders
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()

        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Loss functions
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()

        # Mixed precision training (AMP)
        self.use_amp = config.use_amp and self.device.type == "cuda"

        # Initialize GradScaler
        if self.use_amp:
            self.scaler: Optional[GradScaler] = GradScaler()
        else:
            self.scaler = None

        if self.use_amp:
            print(f"⚡ Mixed Precision (AMP): Enabled")
        else:
            print(f"⚠️  Mixed Precision (AMP): Disabled (CPU or manually disabled)")

        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            max_size=config.buffer_size, memory_efficient=True
        )

        # Training state
        self.current_iteration = 0
        self.total_games_played = 0
        self.total_training_steps = 0
        self.best_iteration = 0
        self.best_win_rate = 0.0

        # History tracking
        self.history = {
            "iterations": [],
            "games_played": [],
            "buffer_size": [],
            "train_loss": [],
            "train_policy_loss": [],
            "train_value_loss": [],
            "eval_win_rate": [],
            "eval_games": [],
            "learning_rate": [],
            "best_iteration": [],
        }

        # Setup logging
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=config.log_dir)

        # Resume from checkpoint if provided
        if resume_from:
            self._load_checkpoint(resume_from)

        # Save initial config
        config.save(os.path.join(config.checkpoint_dir, "config.json"))

        # Save initial best model (prevents deletion by rotation)
        self._save_best_model("Initial best model saved")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        if self.config.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler"""
        if self.config.lr_schedule == "constant":
            return None
        elif self.config.lr_schedule == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.lr_decay_steps,
                gamma=self.config.lr_decay_gamma,
            )
        elif self.config.lr_schedule == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_iterations,
            )
        else:
            return None

    def _save_best_model(self, message: str = ""):
        """
        Save best model to a separate file (won't be deleted by checkpoint rotation).

        Args:
            message: Optional message to print
        """
        best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")

        checkpoint = {
            "iteration": self.current_iteration,
            "model_state_dict": self.best_model.state_dict(),
            "win_rate": self.best_win_rate,
            "config": self.config.to_dict(),
            "model_config": {
                "cnn_residual_blocks": self.config.cnn_blocks,
                "use_rnn": self.config.use_rnn,
            },
        }

        torch.save(checkpoint, best_path)

        if message:
            print(f"\n💾 {message}")
        print(f"   Saved to: {best_path}")
        print(f"   Iteration: {self.current_iteration}")
        print(f"   Win rate: {self.best_win_rate:.1%}")

    def _sharpen_value_target(self, value: float) -> float:
        """
        Apply value target sharpening to amplify win/loss signals.

        This helps the model learn more decisive play by:
        1. Amplifying wins and losses (making signals stronger)
        2. Optionally penalizing draws (making them less attractive)

        Args:
            value: Raw game outcome value (-1.0, 0.0, or 1.0)

        Returns:
            Sharpened value target for training

        Example:
            Raw values:      -1.0,  0.0,  1.0
            Sharpened (1.5): -1.0,  0.0,  1.0  (clamped at ±1.0)
            With penalty:    -1.0, -0.2,  1.0  (draws penalized)
        """
        if not self.config.use_value_sharpening:
            return value

        # Special handling for draws
        if abs(value) < 0.01:  # Draw (value ≈ 0)
            return -self.config.draw_value_penalty

        # Amplify win/loss signals
        # sign(value) preserves direction, amplification increases magnitude
        sharpened = np.sign(value) * min(
            1.0, abs(value) * self.config.value_sharpening_factor
        )

        return float(sharpened)

    def _sharpen_game_values(self, examples: List[GameExample]) -> List[GameExample]:
        """
        Apply value sharpening to all examples from a game.

        Args:
            examples: List of game examples with raw values

        Returns:
            Same examples with sharpened values
        """
        if not self.config.use_value_sharpening:
            return examples

        for example in examples:
            example.value = self._sharpen_value_target(example.value)

        return examples

    def train(self):
        """
        Main training loop.

        Executes the complete AlphaZero training pipeline:
        1. Generate self-play games
        2. Train network on collected data
        3. Evaluate new model
        4. Update best model if improved
        5. Repeat
        """
        print("\n" + "=" * 80)
        print("STARTING RL TRAINING LOOP")
        print("=" * 80)
        print(f"\n📋 Configuration:")
        print(f"   Iterations: {self.config.num_iterations}")
        print(f"   Games per iteration: {self.config.games_per_iteration}")
        print(
            f"   Training steps per iteration: {self.config.training_steps_per_iteration}"
        )
        print(f"   MCTS simulations: {self.config.num_simulations}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Buffer size: {self.config.buffer_size}")

        start_time = time.time()

        try:
            start_iteration = (
                self.current_iteration + 1 if self.current_iteration > 0 else 0
            )
            for iteration in range(start_iteration, self.config.num_iterations):
                self.current_iteration = iteration
                iteration_start = time.time()

                print(f"\n{'='*80}")
                print(f"ITERATION {iteration}/{self.config.num_iterations}")
                print(f"{'='*80}")

                # Step 1: Self-play
                self._self_play_step()

                # Step 2: Training
                if len(self.replay_buffer) >= self.config.min_buffer_size:
                    train_metrics = self._training_step()
                else:
                    print(
                        f"\n⏳ Buffer size ({len(self.replay_buffer)}) below minimum "
                        f"({self.config.min_buffer_size}). Skipping training."
                    )
                    train_metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

                # Step 3: Evaluation
                eval_metrics = None
                if (iteration + 1) % self.config.eval_frequency == 0:
                    eval_metrics = self._evaluation_step()

                # Step 4: Update history and log
                self._update_history(train_metrics, eval_metrics)

                # Step 5: Save checkpoint
                if (iteration + 1) % self.config.save_frequency == 0:
                    self._save_checkpoint()

                # Step 6: Learning rate schedule
                if self.scheduler:
                    self.scheduler.step()

                # Print iteration summary
                iteration_time = time.time() - iteration_start
                self._print_iteration_summary(
                    iteration, train_metrics, eval_metrics, iteration_time
                )

        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user")
            print("Saving checkpoint...")
            self._save_checkpoint(name="interrupted")

        except Exception as e:
            print(f"\n\n❌ Training failed with error: {e}")

            traceback.print_exc()
            print("\nSaving checkpoint...")
            self._save_checkpoint(name="error")
            raise

        else:
            # Save final checkpoint after successful training
            print("\n💾 Saving final checkpoint...")
            self._save_checkpoint()

        finally:
            total_time = time.time() - start_time
            self._print_final_summary(total_time)
            self.writer.close()

    def _self_play_step(self):
        """Execute self-play game generation with mixed opponents"""
        print(f"\n🎮 Self-Play: Generating {self.config.games_per_iteration} games...")

        self.model.eval()

        # Calculate game distribution based on mixed opponent ratios
        if self.config.use_mixed_opponents:
            total_ratio = (
                self.config.self_play_ratio
                + self.config.old_opponent_ratio
                + self.config.random_opponent_ratio
            )

            # Normalize ratios
            self_play_games = int(
                self.config.games_per_iteration
                * self.config.self_play_ratio
                / total_ratio
            )
            old_opponent_games = int(
                self.config.games_per_iteration
                * self.config.old_opponent_ratio
                / total_ratio
            )
            random_opponent_games = (
                self.config.games_per_iteration - self_play_games - old_opponent_games
            )

            print(f"   📊 Game distribution:")
            print(
                f"      Pure self-play: {self_play_games} games ({self_play_games/self.config.games_per_iteration*100:.0f}%)"
            )
            print(
                f"      VS old checkpoint: {old_opponent_games} games ({old_opponent_games/self.config.games_per_iteration*100:.0f}%)"
            )
            print(
                f"      VS random player: {random_opponent_games} games ({random_opponent_games/self.config.games_per_iteration*100:.0f}%)"
            )
        else:
            # Pure self-play (original behavior)
            self_play_games = self.config.games_per_iteration
            old_opponent_games = 0
            random_opponent_games = 0

        # Create self-play configuration
        selfplay_config = SelfPlayConfig(
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
            temperature=self.config.temperature,
            temperature_threshold=self.config.temperature_threshold,
            max_moves=self.config.max_moves_per_game,
            use_rnn=self.config.use_rnn,
            dirichlet_alpha=self.config.dirichlet_alpha,
            resign_threshold=self.config.resign_threshold,
            # Asymmetric play parameters
            asymmetric_play=self.config.asymmetric_play,
            white_temperature=self.config.white_temperature,
            black_temperature=self.config.black_temperature,
            temperature_variation=self.config.temperature_variation,
        )

        all_examples = []
        start_time = time.time()

        # PART 1: Pure self-play games (current vs current)
        if self_play_games > 0:
            print(f"\n   🔄 Generating {self_play_games} pure self-play games...")
            examples = self._generate_selfplay_games(
                num_games=self_play_games, config=selfplay_config, opponent_type="self"
            )
            all_examples.extend(examples)
            print(f"      ✓ {len(examples)} examples")

        # PART 2: Games vs old checkpoint
        if (
            old_opponent_games > 0
            and self.current_iteration >= self.config.old_opponent_iterations_back
        ):
            old_iteration = (
                self.current_iteration - self.config.old_opponent_iterations_back
            )
            old_checkpoint_path = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_iteration_{old_iteration}.pt",
            )

            if os.path.exists(old_checkpoint_path):
                print(
                    f"\n   🔀 Generating {old_opponent_games} games vs old checkpoint (iteration {old_iteration})..."
                )
                examples = self._generate_vs_old_model_games(
                    num_games=old_opponent_games,
                    config=selfplay_config,
                    old_checkpoint_path=old_checkpoint_path,
                )
                all_examples.extend(examples)
                print(f"      ✓ {len(examples)} examples")
            else:
                print(f"\n   ⚠️  Old checkpoint not found: {old_checkpoint_path}")
                print(f"      Skipping old opponent games")

        # PART 3: Games vs random player
        if random_opponent_games > 0:
            print(
                f"\n   🎲 Generating {random_opponent_games} games vs random player..."
            )
            examples = self._generate_vs_random_games(
                num_games=random_opponent_games, config=selfplay_config
            )
            all_examples.extend(examples)
            print(f"      ✓ {len(examples)} examples")

        # Add all examples to replay buffer
        self.replay_buffer.add_game_examples(all_examples)

        elapsed = time.time() - start_time
        self.total_games_played += self.config.games_per_iteration

        print(f"\n✅ Generated {len(all_examples)} total training examples")
        print(
            f"   Time: {elapsed:.1f}s ({elapsed/self.config.games_per_iteration:.1f}s per game)"
        )
        print(f"   Buffer size: {len(self.replay_buffer)}")

        return all_examples

    def _generate_selfplay_games(
        self, num_games: int, config: SelfPlayConfig, opponent_type: str
    ) -> List[GameExample]:
        """Generate pure self-play games (current model vs itself)"""
        if self.config.use_parallel_selfplay:
            # Save current model for parallel workers
            temp_model_path = os.path.join(self.config.checkpoint_dir, "temp_model.pt")
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "config": self.config.to_dict(),
                    "model_config": {
                        "cnn_residual_blocks": self.config.cnn_blocks,
                        "use_rnn": self.config.use_rnn,
                    },
                },
                temp_model_path,
            )

            # Parallel self-play
            worker = ParallelSelfPlayWorker(
                model_path=temp_model_path,
                config=config,
                num_workers=self.config.num_workers,
            )
            examples = worker.play_games_parallel(num_games=num_games, buffer=None)
        else:
            # Sequential self-play
            worker = SelfPlayWorker(
                model=self.model,
                device=self.device,
                config=config,
            )
            examples = worker.play_games(num_games=num_games, buffer=None)

        # Apply value sharpening to all self-play examples
        if self.config.use_value_sharpening:
            for example in examples:
                example.value = self._sharpen_value_target(example.value)

        return examples

    def _generate_vs_old_model_games(
        self, num_games: int, config: SelfPlayConfig, old_checkpoint_path: str
    ) -> List[GameExample]:
        """Generate games where current model plays vs older checkpoint"""
        # Load old model
        old_model = HybridChessNet(
            cnn_residual_blocks=self.config.cnn_blocks, use_rnn=self.config.use_rnn
        ).to(self.device)

        checkpoint = torch.load(old_checkpoint_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            old_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            old_model.load_state_dict(checkpoint)

        old_model.eval()

        # Play games with current model as white, old model as black
        examples = []
        for game_idx in range(num_games):
            game_examples = self._play_vs_opponent_game(
                player_model=self.model,
                opponent_model=old_model,
                config=config,
                player_is_white=(game_idx % 2 == 0),  # Alternate colors
            )
            examples.extend(game_examples)

        return examples

    def _generate_vs_random_games(
        self, num_games: int, config: SelfPlayConfig
    ) -> List[GameExample]:
        """Generate games where current model plays vs random player"""
        examples = []
        for game_idx in range(num_games):
            game_examples = self._play_vs_random_game(
                model=self.model,
                config=config,
                model_is_white=(game_idx % 2 == 0),  # Alternate colors
            )
            examples.extend(game_examples)

        return examples

    def _play_vs_opponent_game(
        self,
        player_model: nn.Module,
        opponent_model: nn.Module,
        config: SelfPlayConfig,
        player_is_white: bool,
    ) -> List[GameExample]:
        """Play a single game with player vs opponent"""
        import random as rand_module

        board = chess.Board()
        examples = []
        move_count = 0

        # Create MCTS for both players
        player_mcts = MCTS(
            model=player_model,
            board_encoder=self.board_encoder,
            move_encoder=self.move_encoder,
            device=self.device,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            temperature=config.temperature,
            use_rnn=config.use_rnn,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=config.dirichlet_alpha,
        )

        opponent_mcts = MCTS(
            model=opponent_model,
            board_encoder=self.board_encoder,
            move_encoder=self.move_encoder,
            device=self.device,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            temperature=config.temperature,
            use_rnn=config.use_rnn,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=config.dirichlet_alpha,
        )

        try:
            while not board.is_game_over() and move_count < config.max_moves:
                move_count += 1

                # Select MCTS based on whose turn it is
                is_player_turn = (board.turn == chess.WHITE) == player_is_white
                current_mcts = player_mcts if is_player_turn else opponent_mcts

                # Adjust temperature
                if move_count < config.temperature_threshold:
                    current_mcts.temperature = config.temperature
                else:
                    current_mcts.temperature = 0.1  # Greedy in endgame

                # Get move and stats
                move, stats = current_mcts.search(board, return_stats=True)

                # Store training example (only for player's moves)
                if is_player_turn and stats:
                    policy = {}
                    if "visit_counts" in stats:
                        visit_counts = stats["visit_counts"]
                        total_visits = sum(visit_counts.values())
                        if total_visits > 0:
                            for move_obj in board.legal_moves:
                                move_uci = move_obj.uci()
                                visits = visit_counts.get(move_uci, 0)
                                policy[move_uci] = visits / total_visits

                    example = GameExample(
                        fen=board.fen(),
                        policy=policy,
                        value=0.0,  # Fill in later
                        move_number=move_count,
                    )
                    examples.append(example)

                # Make move
                board.push(move)

            # Determine result and assign values
            if board.is_game_over():
                result = board.result()
            else:
                result = "1/2-1/2"

            # Parse outcome from player's perspective
            if result == "1-0":
                outcome = 1.0 if player_is_white else -1.0
            elif result == "0-1":
                outcome = -1.0 if player_is_white else 1.0
            else:
                outcome = 0.0

            # Assign values to examples
            for example in examples:
                board_state = chess.Board(example.fen)
                if board_state.turn == chess.WHITE:
                    example.value = outcome if player_is_white else -outcome
                else:
                    example.value = -outcome if player_is_white else outcome

            # Apply value sharpening to amplify win/loss signals
            examples = self._sharpen_game_values(examples)

        except Exception as e:
            print(f"      ⚠️  Error in game: {e}")

        return examples

    def _play_vs_random_game(
        self, model: nn.Module, config: SelfPlayConfig, model_is_white: bool
    ) -> List[GameExample]:
        """Play a single game vs random player"""
        import random as rand_module

        board = chess.Board()
        examples = []
        move_count = 0

        # Create MCTS for model
        mcts = MCTS(
            model=model,
            board_encoder=self.board_encoder,
            move_encoder=self.move_encoder,
            device=self.device,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            temperature=config.temperature,
            use_rnn=config.use_rnn,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=config.dirichlet_alpha,
        )

        try:
            while not board.is_game_over() and move_count < config.max_moves:
                move_count += 1

                is_model_turn = (board.turn == chess.WHITE) == model_is_white

                if is_model_turn:
                    # Model's turn
                    if move_count < config.temperature_threshold:
                        mcts.temperature = config.temperature
                    else:
                        mcts.temperature = 0.1

                    move, stats = mcts.search(board, return_stats=True)

                    # Store training example
                    if stats:
                        policy = {}
                        if "visit_counts" in stats:
                            visit_counts = stats["visit_counts"]
                            total_visits = sum(visit_counts.values())
                            if total_visits > 0:
                                for move_obj in board.legal_moves:
                                    move_uci = move_obj.uci()
                                    visits = visit_counts.get(move_uci, 0)
                                    policy[move_uci] = visits / total_visits

                        example = GameExample(
                            fen=board.fen(),
                            policy=policy,
                            value=0.0,  # Fill in later
                            move_number=move_count,
                        )
                        examples.append(example)
                else:
                    # Random player's turn
                    legal_moves = list(board.legal_moves)
                    move = rand_module.choice(legal_moves)

                # Make move
                board.push(move)

            # Determine result and assign values
            if board.is_game_over():
                result = board.result()
            else:
                result = "1/2-1/2"

            # Parse outcome from model's perspective
            if result == "1-0":
                outcome = 1.0 if model_is_white else -1.0
            elif result == "0-1":
                outcome = -1.0 if model_is_white else 1.0
            else:
                outcome = 0.0

            # Assign values to examples
            for example in examples:
                board_state = chess.Board(example.fen)
                if board_state.turn == chess.WHITE:
                    example.value = outcome if model_is_white else -outcome
                else:
                    example.value = -outcome if model_is_white else outcome

            # Apply value sharpening to amplify win/loss signals
            examples = self._sharpen_game_values(examples)

        except Exception as e:
            print(f"      ⚠️  Error in game: {e}")

        return examples

    def _training_step(self) -> Dict[str, float]:
        """Execute network training on replay buffer"""
        print(f"\n📚 Training: {self.config.training_steps_per_iteration} steps...")

        self.model.train()

        # Sample from replay buffer - get all examples
        buffer_examples = list(self.replay_buffer.buffer)

        # Determine sample size
        sample_size = min(
            len(buffer_examples),
            int(len(self.replay_buffer) * self.config.sample_ratio),
        )

        if sample_size < len(buffer_examples):
            indices = np.random.choice(len(buffer_examples), sample_size, replace=False)
            buffer_examples = [buffer_examples[i] for i in indices]

        # Create dataset and dataloader
        dataset = RLDataset(buffer_examples, self.board_encoder, self.move_encoder)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # Training metrics
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        # Training loop
        steps_per_epoch = len(dataloader)
        num_epochs = max(1, self.config.training_steps_per_iteration // steps_per_epoch)

        pbar = tqdm(total=num_epochs * steps_per_epoch, desc="Training")

        for epoch in range(num_epochs):
            for batch_idx, (boards, policies, values) in enumerate(dataloader):
                # Move to device
                boards = boards.to(self.device)
                policies = policies.to(self.device)
                values = values.to(self.device)

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass with optional mixed precision
                if self.use_amp:
                    # MIXED PRECISION PATH
                    assert (
                        self.scaler is not None
                    ), "Scaler should be initialized when use_amp is True"

                    autocast_context = autocast(device_type=AMP_DEVICE)

                    with autocast_context:
                        # Forward pass in fp16
                        policy_logits, value_pred, _ = self.model(boards)

                        # Compute loss in fp16
                        policy_loss = self.policy_criterion(policy_logits, policies)
                        value_loss = self.value_criterion(value_pred, values)
                        loss = (
                            self.config.policy_loss_weight * policy_loss
                            + self.config.value_loss_weight * value_loss
                        )

                    # Backward pass with scaled gradients
                    self.scaler.scale(loss).backward()

                    # Unscale before gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    # Optimizer step with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:
                    # STANDARD FP32 PATH (fallback for CPU or when AMP disabled)
                    policy_logits, value_pred, _ = self.model(boards)

                    policy_loss = self.policy_criterion(policy_logits, policies)
                    value_loss = self.value_criterion(value_pred, values)
                    loss = (
                        self.config.policy_loss_weight * policy_loss
                        + self.config.value_loss_weight * value_loss
                    )

                    # Backward pass
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )

                    # Optimizer step
                    self.optimizer.step()

                # Track metrics
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1
                self.total_training_steps += 1

                # Log to tensorboard
                if self.total_training_steps % self.config.log_frequency == 0:
                    self.writer.add_scalar(
                        "train/loss", loss.item(), self.total_training_steps
                    )
                    self.writer.add_scalar(
                        "train/policy_loss",
                        policy_loss.item(),
                        self.total_training_steps,
                    )
                    self.writer.add_scalar(
                        "train/value_loss", value_loss.item(), self.total_training_steps
                    )
                    self.writer.add_scalar(
                        "train/learning_rate",
                        self.optimizer.param_groups[0]["lr"],
                        self.total_training_steps,
                    )
                    self.writer.add_scalar(
                        "train/epoch", epoch, self.total_training_steps
                    )

                pbar.update(1)

                # Break if we've done enough steps
                if num_batches >= self.config.training_steps_per_iteration:
                    break

            if num_batches >= self.config.training_steps_per_iteration:
                break

        pbar.close()

        # Average metrics
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0.0
        avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0.0

        metrics = {
            "loss": avg_loss,
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
        }

        print(f"\n✅ Training complete")
        print(
            f"   Loss: {avg_loss:.4f} (Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})"
        )
        print(f"   Total training steps: {self.total_training_steps}")

        return metrics

    def _evaluation_step(self) -> Dict[str, float]:
        """Evaluate current model against best model"""
        print(
            f"\n⚔️  Evaluation: Playing {self.config.eval_games} games vs best model..."
        )

        self.model.eval()
        self.best_model.eval()

        # Play evaluation games
        wins = 0
        losses = 0
        draws = 0

        eval_config = SelfPlayConfig(
            num_simulations=self.config.eval_simulations,
            c_puct=self.config.c_puct,
            temperature=0.1,  # Lower temperature for evaluation
            temperature_threshold=0,  # Deterministic play
            max_moves=self.config.max_moves_per_game,
            use_rnn=self.config.use_rnn,
            dirichlet_alpha=0.0,  # No exploration during evaluation
            resign_threshold=self.config.resign_threshold,
        )

        for game_num in tqdm(range(self.config.eval_games), desc="Evaluation games"):
            # Alternate colors
            if game_num % 2 == 0:
                current_model_plays_white = True
            else:
                current_model_plays_white = False

            result = self._play_evaluation_game(current_model_plays_white, eval_config)

            if result == "current_win":
                wins += 1
            elif result == "best_win":
                losses += 1
            else:
                draws += 1

        # Calculate win rate
        total_games = wins + losses + draws
        win_rate = (wins + 0.5 * draws) / total_games if total_games > 0 else 0.0

        metrics = {
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "draws": draws,
        }

        print(f"\n✅ Evaluation complete")
        print(f"   Win rate: {win_rate:.1%} ({wins}W / {losses}L / {draws}D)")

        # Update best model if improvement
        if win_rate >= self.config.win_threshold:
            print(
                f"\n🏆 New best model! (win rate: {win_rate:.1%} >= {self.config.win_threshold:.1%})"
            )
            self.best_model.load_state_dict(self.model.state_dict())
            self.best_iteration = self.current_iteration
            self.best_win_rate = win_rate

            # Save best model using helper
            self._save_best_model(
                f"New best model! (win rate: {win_rate:.1%} >= {self.config.win_threshold:.1%})"
            )
        else:
            print(
                f"\n   Current model not better than best (need {self.config.win_threshold:.1%})"
            )

            if win_rate > self.best_win_rate:
                print(
                    f"   However, this is better than previous best ({self.best_win_rate:.1%})"
                )
                self.best_model.load_state_dict(self.model.state_dict())
                self.best_iteration = self.current_iteration
                self.best_win_rate = win_rate
                self._save_best_model("Saving improved model (below threshold)")

        # Log to tensorboard
        self.writer.add_scalar("eval/win_rate", win_rate, self.current_iteration)
        self.writer.add_scalar("eval/wins", wins, self.current_iteration)
        self.writer.add_scalar("eval/losses", losses, self.current_iteration)
        self.writer.add_scalar("eval/draws", draws, self.current_iteration)

        return metrics

    def _play_evaluation_game(
        self, current_plays_white: bool, config: SelfPlayConfig
    ) -> str:
        """
        Play single evaluation game between current and best models.

        Args:
            current_plays_white: Whether current model plays white
            config: Self-play configuration

        Returns:
            "current_win", "best_win", or "draw"
        """

        board = chess.Board()

        # Create MCTS for both models
        current_mcts = MCTS(
            model=self.model,
            board_encoder=self.board_encoder,
            move_encoder=self.move_encoder,
            device=self.device,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            temperature=config.temperature,
            use_rnn=config.use_rnn,
        )

        best_mcts = MCTS(
            model=self.best_model,
            board_encoder=self.board_encoder,
            move_encoder=self.move_encoder,
            device=self.device,
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            temperature=config.temperature,
            use_rnn=config.use_rnn,
        )

        # Play game
        move_count = 0
        while not board.is_game_over() and move_count < config.max_moves:
            # Determine which model's turn
            if board.turn == chess.WHITE:
                mcts = current_mcts if current_plays_white else best_mcts
            else:
                mcts = best_mcts if current_plays_white else current_mcts

            # Get move
            move, _ = mcts.search(board)

            if move is None:
                break

            board.push(move)
            move_count += 1

        # Determine result
        if board.is_checkmate():
            # Winner is opposite of whose turn it is
            white_won = not board.turn
            if (white_won and current_plays_white) or (
                not white_won and not current_plays_white
            ):
                return "current_win"
            else:
                return "best_win"
        else:
            # Draw (stalemate, repetition, 50-move, insufficient material, or max moves)
            return "draw"

    def _update_history(
        self, train_metrics: Dict[str, float], eval_metrics: Optional[Dict[str, float]]
    ):
        """Update training history"""
        self.history["iterations"].append(self.current_iteration)
        self.history["games_played"].append(self.total_games_played)
        self.history["buffer_size"].append(len(self.replay_buffer))
        self.history["train_loss"].append(train_metrics["loss"])
        self.history["train_policy_loss"].append(train_metrics["policy_loss"])
        self.history["train_value_loss"].append(train_metrics["value_loss"])
        self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
        self.history["best_iteration"].append(self.best_iteration)

        if eval_metrics:
            self.history["eval_win_rate"].append(eval_metrics["win_rate"])
            self.history["eval_games"].append(self.config.eval_games)
        else:
            self.history["eval_win_rate"].append(None)
            self.history["eval_games"].append(0)

        # Log to tensorboard
        self.writer.add_scalar(
            "iteration/games_played", self.total_games_played, self.current_iteration
        )
        self.writer.add_scalar(
            "iteration/buffer_size", len(self.replay_buffer), self.current_iteration
        )
        self.writer.add_scalar(
            "iteration/train_loss", train_metrics["loss"], self.current_iteration
        )

    def _save_checkpoint(self, name: Optional[str] = None):
        """Save training checkpoint"""
        if name is None:
            name = f"iteration_{self.current_iteration}"

        checkpoint = {
            "iteration": self.current_iteration,
            "total_games_played": self.total_games_played,
            "total_training_steps": self.total_training_steps,
            "best_iteration": self.best_iteration,
            "best_win_rate": self.best_win_rate,
            "model_state_dict": self.model.state_dict(),
            "best_model_state_dict": self.best_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
            "config": self.config.to_dict(),
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save GradScaler state if using AMP
        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save checkpoint
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir, f"checkpoint_{name}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        # Save latest
        latest_path = os.path.join(self.config.checkpoint_dir, "latest.pt")
        torch.save(checkpoint, latest_path)

        # Save replay buffer
        buffer_path = os.path.join(self.config.checkpoint_dir, f"buffer_{name}.pkl")
        self.replay_buffer.save(buffer_path)

        print(f"\n💾 Saved checkpoint: {checkpoint_path}")

        # Clean old checkpoints
        self._clean_old_checkpoints()

    def _clean_old_checkpoints(self):
        """Remove old checkpoints, keeping only recent ones"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_iteration_*.pt"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        # Remove old checkpoints
        for checkpoint in checkpoints[self.config.keep_checkpoints :]:
            checkpoint.unlink()
            # Also remove corresponding buffer
            buffer_file = (
                checkpoint.parent
                / f"buffer_iteration_{checkpoint.stem.split('_')[-1]}.pkl"
            )
            if buffer_file.exists():
                buffer_file.unlink()

    def _load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        print(f"\n📂 Loading checkpoint: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        # Restore model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_model.load_state_dict(checkpoint["best_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Restore GradScaler state if using AMP
        if "scaler_state_dict" in checkpoint and self.scaler:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            print(f"   Loaded GradScaler state")

        # Restore training state
        self.current_iteration = checkpoint["iteration"]

        # Log resume information
        next_iteration = self.current_iteration + 1
        print(f"   Resuming from iteration {self.current_iteration}")
        print(f"   Next iteration will be: {next_iteration}")
        self.total_games_played = checkpoint.get("total_games_played", 0)
        self.total_training_steps = checkpoint.get("total_training_steps", 0)
        self.best_iteration = checkpoint.get("best_iteration", 0)
        self.best_win_rate = checkpoint.get("best_win_rate", 0.0)
        self.history = checkpoint.get("history", self.history)

        # Try to load replay buffer
        buffer_path = os.path.join(
            self.config.checkpoint_dir, f"buffer_iteration_{self.current_iteration}.pkl"
        )
        if os.path.exists(buffer_path):
            self.replay_buffer.load(buffer_path)
            print(f"   Loaded replay buffer: {len(self.replay_buffer)} examples")

        print(f"✅ Resumed from iteration {self.current_iteration}")

    def _print_iteration_summary(
        self,
        iteration: int,
        train_metrics: Dict[str, float],
        eval_metrics: Optional[Dict[str, float]],
        iteration_time: float,
    ):
        """Print summary of iteration"""
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration} SUMMARY")
        print(f"{'='*80}")
        print(f"\n📊 Metrics:")
        print(f"   Total games: {self.total_games_played}")
        print(f"   Buffer size: {len(self.replay_buffer)}")
        print(f"   Train loss: {train_metrics['loss']:.4f}")
        print(f"   Policy loss: {train_metrics['policy_loss']:.4f}")
        print(f"   Value loss: {train_metrics['value_loss']:.4f}")
        print(f"   Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")

        if eval_metrics:
            print(f"\n⚔️  Evaluation:")
            print(f"   Win rate: {eval_metrics['win_rate']:.1%}")
            print(
                f"   Results: {eval_metrics['wins']}W / {eval_metrics['losses']}L / {eval_metrics['draws']}D"
            )

        print(f"\n🏆 Best Model:")
        print(f"   Iteration: {self.best_iteration}")
        print(f"   Win rate: {self.best_win_rate:.1%}")

        print(f"\n⏱️  Time: {iteration_time:.1f}s")

    def _print_final_summary(self, total_time: float):
        """Print final training summary"""
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"\n📊 Final Statistics:")
        print(f"   Total iterations: {self.current_iteration + 1}")
        print(f"   Total games: {self.total_games_played}")
        print(f"   Total training steps: {self.total_training_steps}")
        print(f"   Final buffer size: {len(self.replay_buffer)}")
        print(f"\n🏆 Best Model:")
        print(f"   Iteration: {self.best_iteration}")
        print(f"   Win rate: {self.best_win_rate:.1%}")
        print(f"\n⏱️  Total Time: {total_time/3600:.1f} hours")
        print(
            f"   Avg time per iteration: {total_time/(self.current_iteration+1):.1f}s"
        )

        # Save final history
        history_path = os.path.join(self.config.checkpoint_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\n💾 Saved training history: {history_path}")


def main():
    """Main entry point for RL training"""

    parser = argparse.ArgumentParser(description="RL Training Loop for Chess Engine")

    # Training parameters
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of training iterations"
    )
    parser.add_argument(
        "--games-per-iter", type=int, default=100, help="Games per iteration"
    )
    parser.add_argument(
        "--training-steps", type=int, default=1000, help="Training steps per iteration"
    )

    # Self-play parameters
    parser.add_argument("--simulations", type=int, default=200, help="MCTS simulations")
    parser.add_argument(
        "--c-puct",
        type=float,
        default=2.0,
        help="MCTS exploration constant (default: 2.0)",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Use parallel self-play"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of parallel workers"
    )

    # Model parameters
    parser.add_argument("--cnn-blocks", type=int, default=10, help="CNN blocks")
    parser.add_argument("--use-rnn", action="store_true", help="Use RNN")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--buffer-size", type=int, default=500000, help="Replay buffer size"
    )

    # Evaluation
    parser.add_argument("--eval-freq", type=int, default=5, help="Evaluation frequency")
    parser.add_argument("--eval-games", type=int, default=20, help="Evaluation games")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.5,
        help="Temperature for move selection (default: 1.5)",
    )

    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.3,
        help="Dirichlet noise alpha for root exploration (default: 0.3)",
    )

    parser.add_argument(
        "--resign-threshold",
        type=float,
        default=-0.9,
        help="Resign if position value drops below this (default: -0.9)",
    )

    # ASYMMETRIC SELF-PLAY PARAMETERS
    parser.add_argument(
        "--asymmetric-play",
        action="store_true",
        default=True,
        help="Enable asymmetric self-play (different temps for white/black) (default: True)",
    )
    parser.add_argument(
        "--no-asymmetric-play",
        dest="asymmetric_play",
        action="store_false",
        help="Disable asymmetric self-play",
    )
    parser.add_argument(
        "--temperature-variation",
        type=float,
        default=0.3,
        help="Random temperature variation range (e.g., 0.3 = ±0.3) (default: 0.3)",
    )
    parser.add_argument(
        "--white-temperature",
        type=float,
        default=None,
        help="Fixed temperature for white (overrides --temperature)",
    )
    parser.add_argument(
        "--black-temperature",
        type=float,
        default=None,
        help="Fixed temperature for black (overrides --temperature)",
    )

    # MIXED OPPONENT PARAMETERS
    parser.add_argument(
        "--use-mixed-opponents",
        action="store_true",
        default=False,
        help="Enable mixed opponent training (default: False)",
    )
    parser.add_argument(
        "--self-play-ratio",
        type=float,
        default=0.70,
        help="Ratio of pure self-play games (default: 0.70)",
    )
    parser.add_argument(
        "--old-opponent-ratio",
        type=float,
        default=0.20,
        help="Ratio of games vs old checkpoint (default: 0.20)",
    )
    parser.add_argument(
        "--random-opponent-ratio",
        type=float,
        default=0.10,
        help="Ratio of games vs random player (default: 0.10)",
    )
    parser.add_argument(
        "--old-opponent-iterations-back",
        type=int,
        default=2,
        help="Use checkpoint N iterations ago as old opponent (default: 2)",
    )

    # VALUE TARGET SHARPENING PARAMETERS
    parser.add_argument(
        "--use-value-sharpening",
        action="store_true",
        default=True,
        help="Enable value target sharpening to amplify win/loss signals (default: True)",
    )
    parser.add_argument(
        "--value-sharpening-factor",
        type=float,
        default=1.5,
        help="Amplification factor for win/loss values (default: 1.5)",
    )
    parser.add_argument(
        "--draw-value-penalty",
        type=float,
        default=0.0,
        help="Penalty for draws, negative value makes draws less attractive (default: 0.0)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="data/rl_checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )

    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="Initialize from supervised model weights",
    )

    # Device
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable mixed precision training (AMP)"
    )

    args = parser.parse_args()

    # Create configuration
    config = RLTrainingConfig(
        num_iterations=args.iterations,
        games_per_iteration=args.games_per_iter,
        training_steps_per_iteration=args.training_steps,
        num_simulations=args.simulations,
        c_puct=getattr(args, "c_puct", 2.0),
        temperature=args.temperature,
        dirichlet_alpha=args.dirichlet_alpha,
        resign_threshold=args.resign_threshold,
        # Asymmetric play parameters
        asymmetric_play=args.asymmetric_play,
        temperature_variation=args.temperature_variation,
        white_temperature=args.white_temperature,
        black_temperature=args.black_temperature,
        # Mixed opponent parameters
        use_mixed_opponents=args.use_mixed_opponents,
        self_play_ratio=args.self_play_ratio,
        old_opponent_ratio=args.old_opponent_ratio,
        random_opponent_ratio=args.random_opponent_ratio,
        old_opponent_iterations_back=args.old_opponent_iterations_back,
        # Value sharpening parameters
        use_value_sharpening=args.use_value_sharpening,
        value_sharpening_factor=args.value_sharpening_factor,
        draw_value_penalty=args.draw_value_penalty,
        # Rest of config
        use_parallel_selfplay=args.parallel,
        num_workers=args.workers,
        cnn_blocks=args.cnn_blocks,
        use_rnn=args.use_rnn,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        eval_frequency=args.eval_freq,
        eval_games=args.eval_games,
        checkpoint_dir=args.checkpoint_dir,
        device="cpu" if args.cpu else "cuda",
        use_amp=not args.no_amp,
    )

    # Create trainer
    trainer = RLTrainer(
        config=config, resume_from=args.resume, pretrained_path=args.pretrained_model
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
