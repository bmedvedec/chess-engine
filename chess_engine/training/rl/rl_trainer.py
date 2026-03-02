"""
RL TRAINER - Main Trainer Class

Orchestrates the complete AlphaZero-style RL training pipeline:
- Self-play game generation
- Network training on collected experience
- Model evaluation and selection
- Iterative improvement loop

Delegates to:
  - self_play_runner.py: execute_self_play_step
  - train_step.py: execute_training_step
  - evaluation.py: execute_evaluation_step
"""

import os
import json
import time
import traceback
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from chess_engine.models.hybrid.hybrid_net import HybridChessNet
from chess_engine.models.hybrid.config import HybridModelConfig
from chess_engine.training.rl.evaluation import execute_evaluation_step
from chess_engine.training.rl.rl_config import RLTrainingConfig
from chess_engine.training.rl.self_play_runner import execute_self_play_step
from chess_engine.training.rl.train_step import execute_training_step
from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder
from chess_engine.data.replay.buffer import ReplayBuffer


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
        model: Optional[HybridChessNet] = None,
        resume_from: Optional[str] = None,
        pretrained_path: Optional[str] = None,
    ):
        """
        Initialize RL trainer.

        Args:
            config: Training configuration
            model: Optional pre-initialized model
            resume_from: Optional checkpoint path to resume from
            pretrained_path: Optional pretrained model for transfer learning
        """
        self.config = config

        # ---- device setup ----
        self.device = torch.device(
            config.device
            if torch.cuda.is_available() and config.device == "cuda"
            else "cpu"
        )
        print(f"\n🖥️  Using device: {self.device}")

        # ---- build or attach model ----
        if model is None:
            self.model_config = HybridModelConfig(
                cnn_input_channels=20,
                cnn_filters=config.cnn_filters,
                cnn_residual_blocks=config.cnn_blocks,
                use_rnn=config.use_rnn,
                rnn_hidden_size=config.rnn_hidden_size,
                rnn_num_layers=config.rnn_layers,
                rnn_use_attention=config.rnn_use_attention,
                rnn_bidirectional=config.rnn_bidirectional,
                fusion_type=config.fusion_type,
                num_actions=config.num_actions,
            )

            self.model = HybridChessNet(self.model_config).to(self.device)
        else:
            self.model = model.to(self.device)
            self.model_config = model.config

        # ---- load pretrained weights (transfer learning) ----
        if pretrained_path and not resume_from:
            print(f"\n📥 Loading pretrained weights from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=self.device)

            state_dict = (
                checkpoint["model_state_dict"]
                if "model_state_dict" in checkpoint
                else checkpoint
            )

            self.model.load_state_dict(state_dict, strict=False)
            print("   ✅ Weights loaded successfully")

        # ---- best model snapshot ----
        self.best_model = HybridChessNet(self.model_config).to(self.device)
        self.best_model.load_state_dict(self.model.state_dict())

        # ---- encoders ----
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()

        # ---- optimizer & scheduler ----
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # ---- loss functions ----
        self.value_criterion = nn.MSELoss()

        # ---- mixed precision ----
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.scaler = GradScaler() if self.use_amp else None

        print(
            "⚡ Mixed Precision (AMP): Enabled"
            if self.use_amp
            else "⚠️  Mixed Precision (AMP): Disabled"
        )

        # ---- replay buffer ----
        self.replay_buffer = ReplayBuffer(
            max_size=config.buffer_size,
            memory_efficient=True,
        )

        # ---- training state ----
        self.current_iteration = 0
        self.total_games_played = 0
        self.total_training_steps = 0
        self.best_iteration = 0
        self.best_win_rate = 0.0

        # ---- history ----
        self.history: Dict[str, list] = {
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

        # ---- logging ----
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=config.log_dir)

        # ---- resume training ----
        if resume_from:
            self._load_checkpoint(resume_from)

        # ---- save config snapshot ----
        config.save(os.path.join(config.checkpoint_dir, "config.json"))

        # ---- save initial best model ----
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
                _, self.total_games_played = execute_self_play_step(
                    model=self.model,
                    config=self.config,
                    replay_buffer=self.replay_buffer,
                    device=self.device,
                    total_games_played=self.total_games_played,
                )

                # Step 2: Training
                if len(self.replay_buffer) >= self.config.min_buffer_size:
                    train_metrics, self.total_training_steps = execute_training_step(
                        model=self.model,
                        replay_buffer=self.replay_buffer,
                        optimizer=self.optimizer,
                        config=self.config,
                        board_encoder=self.board_encoder,
                        move_encoder=self.move_encoder,
                        device=self.device,
                        use_amp=self.use_amp,
                        scaler=self.scaler,
                        value_criterion=self.value_criterion,
                        writer=self.writer,
                        current_iteration=self.current_iteration,
                        total_training_steps=self.total_training_steps,
                    )
                else:
                    print(
                        f"\n⏳ Buffer size ({len(self.replay_buffer)}) below minimum "
                        f"({self.config.min_buffer_size}). Skipping training."
                    )
                    train_metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

                # Step 3: Evaluation
                eval_metrics = None
                if (iteration + 1) % self.config.eval_frequency == 0:
                    (
                        eval_metrics,
                        should_update_best,
                        self.best_iteration,
                        self.best_win_rate,
                    ) = execute_evaluation_step(
                        current_model=self.model,
                        best_model=self.best_model,
                        board_encoder=self.board_encoder,
                        move_encoder=self.move_encoder,
                        device=self.device,
                        config=self.config,
                        writer=self.writer,
                        current_iteration=self.current_iteration,
                        best_iteration=self.best_iteration,
                        best_win_rate=self.best_win_rate,
                    )

                    if should_update_best:
                        self.best_model.load_state_dict(self.model.state_dict())
                        self._save_best_model(
                            f"Best model updated (win rate: {self.best_win_rate:.1%})"
                        )

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
        print(f"ITERATION {iteration + 1} SUMMARY")
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
        print(f"   Iteration: {self.best_iteration + 1}")
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
        print(f"   Iteration: {self.best_iteration + 1}")
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
