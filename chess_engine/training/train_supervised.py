"""
SUPERVISED LEARNING TRAINING

Train the chess engine on existing games to learn:
- What moves strong players make
- How to evaluate positions
- Opening principles
- Tactical patterns

Training Process:
1. Load games data
2. For each position:
   - Input: Board state
   - Target: Move played + Game outcome
3. Optimize network to predict these
4. Save checkpoints
5. Evaluate progress

Strong baseline before self-play RL.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN messages
import sys
import time
from typing import Dict, Tuple
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from tqdm import tqdm

from chess_engine.models.hybrid_model import HybridChessNet, count_parameters
from chess_engine.data.chess_dataset import (
    ChessDataset,
    load_dataset,
    create_dataloader,
    split_examples,
)
from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder


class ChessTrainer:
    """
    Trainer for supervised learning on chess games.

    Handles:
    - Training loop with mixed precision
    - Validation with metrics
    - Early stopping
    - Checkpointing
    - Logging
    - Learning rate scheduling with warmup
    - Gradient accumulation
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.001,
        checkpoint_dir: str = "data/models/checkpoints",
        log_dir: str = "logs/training",
        use_mixed_precision: bool = False,
        policy_weight: float = 1.0,
        value_weight: float = 1.0,
        early_stopping_patience: int = 10,
        warmup_epochs: int = 2,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Initialize trainer.

        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on (cuda/cpu)
            learning_rate: Initial learning rate (default: 0.001)
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            use_mixed_precision: Enable automatic mixed precision (default: True)
            policy_weight: Weight for policy loss (default: 1.0)
            value_weight: Weight for value loss (default: 1.0)
            early_stopping_patience: Epochs without improvement before stopping (default: 10)
            warmup_epochs: Number of warmup epochs (default: 2)
            gradient_accumulation_steps: Gradient accumulation steps (default: 1)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Loss weights
        self.policy_weight = policy_weight
        self.value_weight = value_weight

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.epochs_without_improvement = 0

        # Gradient accumulation
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Optimizer with explicit Adam parameters
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Main scheduler (ReduceLROnPlateau)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
        )

        # Warmup scheduler
        self.warmup_epochs = warmup_epochs

        def warmup_lambda(epoch):
            if epoch < self.warmup_epochs:
                return float(epoch + 1) / float(self.warmup_epochs)
            return 1.0

        self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=warmup_lambda)

        # Flag to track learning rate changes
        self.last_lr = learning_rate

        # Mixed precision training
        self.use_mixed_precision = use_mixed_precision and device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_mixed_precision else None

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Move encoder for policy targets
        self.move_encoder = MoveEncoder()

        print(f"✅ Trainer initialized")
        print(f"   Model parameters: {count_parameters(model):,}")
        print(f"   Device: {device}")
        print(f"   Mixed precision: {self.use_mixed_precision}")
        print(f"   Policy weight: {policy_weight}")
        print(f"   Value weight: {value_weight}")
        print(f"   Early stopping patience: {early_stopping_patience}")
        print(f"   Warmup epochs: {warmup_epochs}")
        print(f"   Gradient accumulation: {gradient_accumulation_steps}")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")

    def compute_loss(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        move_targets: torch.Tensor,
        outcome_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute training loss.

        Loss = (policy_weight × Policy Loss) + (value_weight × Value Loss)
        - Policy Loss: Cross-entropy between predicted policy and actual move
        - Value Loss: MSE between predicted value and game outcome

        Features:
        - Top-1, Top-3, Top-5 accuracy
        - Value MAE (Mean Absolute Error)

        Args:
            policy_logits: Model policy output (batch, 4096)
            value_pred: Model value output (batch, 1)
            move_targets: Target move indices (batch,)
            outcome_targets: Game outcomes (batch,)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Policy loss (cross-entropy)
        policy_loss = F.cross_entropy(policy_logits, move_targets)

        # Value loss (MSE)
        value_loss = F.mse_loss(value_pred.squeeze(), outcome_targets)

        # Total loss (weighted sum)
        total_loss = self.policy_weight * policy_loss + self.value_weight * value_loss

        # Calculate accuracy for logging
        with torch.no_grad():
            # Top-1 accuracy
            _, predicted_moves = torch.max(policy_logits, dim=1)
            top1_accuracy = (predicted_moves == move_targets).float().mean()

            # Top-3 accuracy
            _, top3_indices = torch.topk(policy_logits, k=3, dim=1)
            top3_accuracy = (
                (top3_indices == move_targets.unsqueeze(1)).any(dim=1).float().mean()
            )

            # Top-5 accuracy
            _, top5_indices = torch.topk(policy_logits, k=5, dim=1)
            top5_accuracy = (
                (top5_indices == move_targets.unsqueeze(1)).any(dim=1).float().mean()
            )

            # Value MAE (mean absolute error) - more interpretable than MSE
            value_mae = torch.abs(value_pred.squeeze() - outcome_targets).mean()

        loss_dict = {
            "total": total_loss.item(),
            "policy": policy_loss.item(),
            "value": value_loss.item(),
            "policy_top1_acc": top1_accuracy.item(),
            "policy_top3_acc": top3_accuracy.item(),
            "policy_top5_acc": top5_accuracy.item(),
            "value_mae": value_mae.item(),
        }

        return total_loss, loss_dict

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with mixed precision and gradient accumulation.

        Returns:
            Dictionary of average metrics for the epoch
        """
        self.model.train()

        total_losses = {
            "total": 0.0,
            "policy": 0.0,
            "value": 0.0,
            "policy_top1_acc": 0.0,
            "policy_top3_acc": 0.0,
            "policy_top5_acc": 0.0,
            "value_mae": 0.0,
        }

        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")

        for batch_idx, (boards, moves, outcomes) in enumerate(pbar):
            # Move to device
            boards = boards.to(self.device)
            moves = moves.to(self.device)
            outcomes = outcomes.to(self.device)

            # Mixed precision forward pass
            with autocast("cuda", enabled=self.use_mixed_precision):
                # Forward pass
                policy_logits, value_pred, _ = self.model(boards)

                # Compute loss
                loss, loss_dict = self.compute_loss(
                    policy_logits, value_pred, moves, outcomes
                )

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass with mixed precision
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    # Unscale before clipping
                    self.scaler.unscale_(self.optimizer)
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping (prevent exploding gradients)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()

                # Zero gradients
                self.optimizer.zero_grad()

            # Accumulate losses
            for key in total_losses:
                total_losses[key] += loss_dict[key]

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss_dict['total']:.4f}",
                    "top1": f"{loss_dict['policy_top1_acc']:.3f}",
                    "top5": f"{loss_dict['policy_top5_acc']:.3f}",
                }
            )

            # Log to TensorBoard
            if self.global_step % 10 == 0:
                for key, value in loss_dict.items():
                    self.writer.add_scalar(f"train/{key}", value, self.global_step)
                # Log learning rate
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar(
                    "train/learning_rate", current_lr, self.global_step
                )

            self.global_step += 1

        # Calculate averages
        num_batches = len(self.train_loader)
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}

        return avg_losses

    def validate(self) -> Dict[str, float]:
        """
        Run validation.

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_losses = {
            "total": 0.0,
            "policy": 0.0,
            "value": 0.0,
            "policy_top1_acc": 0.0,
            "policy_top3_acc": 0.0,
            "policy_top5_acc": 0.0,
            "value_mae": 0.0,
        }

        with torch.no_grad():
            for boards, moves, outcomes in tqdm(
                self.val_loader, desc="Validating", leave=False
            ):
                # Move to device
                boards = boards.to(self.device)
                moves = moves.to(self.device)
                outcomes = outcomes.to(self.device)

                # Forward pass
                policy_logits, value_pred, _ = self.model(boards)

                # Compute loss
                loss, loss_dict = self.compute_loss(
                    policy_logits, value_pred, moves, outcomes
                )

                # Accumulate losses
                for key in total_losses:
                    total_losses[key] += loss_dict[key]

        # Calculate averages
        num_batches = len(self.val_loader)
        avg_losses = {key: value / num_batches for key, value in total_losses.items()}

        # Log to TensorBoard
        for key, value in avg_losses.items():
            self.writer.add_scalar(f"val/{key}", value, self.epoch)

        return avg_losses

    def save_checkpoint(self, filename: str = "checkpoint.pt", is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "warmup_scheduler_state_dict": self.warmup_scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            # Save configuration
            "config": {
                "policy_weight": self.policy_weight,
                "value_weight": self.value_weight,
                "early_stopping_patience": self.early_stopping_patience,
                "warmup_epochs": self.warmup_epochs,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "use_mixed_precision": self.use_mixed_precision,
            },
        }

        # Save scaler state if using mixed precision
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"💾 Saved checkpoint: {filepath}")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"⭐ Saved best model: {best_path}")

    def load_checkpoint(self, filepath: str):
        """Load checkpoint and resume training"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load warmup scheduler if available
        if "warmup_scheduler_state_dict" in checkpoint:
            self.warmup_scheduler.load_state_dict(
                checkpoint["warmup_scheduler_state_dict"]
            )

        # Load scaler if using mixed precision
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        # Load early stopping state if available
        if "epochs_without_improvement" in checkpoint:
            self.epochs_without_improvement = checkpoint["epochs_without_improvement"]

        print(f"✅ Loaded checkpoint from epoch {self.epoch}")
        if "config" in checkpoint:
            print(f"   Configuration: {checkpoint['config']}")

    def train(self, num_epochs: int):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
        """
        print("\n" + "=" * 80)
        print("🎓 STARTING TRAINING")
        print("=" * 80)

        start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch

            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*80}")

            # Train
            train_losses = self.train_epoch()

            print(f"\n📊 Training metrics:")
            print(f"   Loss: {train_losses['total']:.4f}")
            print(f"   Policy loss: {train_losses['policy']:.4f}")
            print(f"   Value loss: {train_losses['value']:.4f}")
            print(f"   Top-1 accuracy: {train_losses['policy_top1_acc']:.3f}")
            print(f"   Top-3 accuracy: {train_losses['policy_top3_acc']:.3f}")
            print(f"   Top-5 accuracy: {train_losses['policy_top5_acc']:.3f}")
            print(f"   Value MAE: {train_losses['value_mae']:.4f}")

            # Validate
            val_losses = self.validate()

            print(f"\n📊 Validation metrics:")
            print(f"   Loss: {val_losses['total']:.4f}")
            print(f"   Policy loss: {val_losses['policy']:.4f}")
            print(f"   Value loss: {val_losses['value']:.4f}")
            print(f"   Top-1 accuracy: {val_losses['policy_top1_acc']:.3f}")
            print(f"   Top-3 accuracy: {val_losses['policy_top3_acc']:.3f}")
            print(f"   Top-5 accuracy: {val_losses['policy_top5_acc']:.3f}")
            print(f"   Value MAE: {val_losses['value_mae']:.4f}")

            # Learning rate scheduling
            if epoch < self.warmup_epochs:
                # Use warmup scheduler
                self.warmup_scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"\n🔥 Warmup phase - Learning rate: {current_lr:.6f}")
            else:
                # Use main scheduler
                self.scheduler.step(val_losses["total"])
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Check if learning rate changed
                if current_lr != self.last_lr:
                    print(
                        f"\n📉 Learning rate reduced: {self.last_lr:.6f} → {current_lr:.6f}"
                    )
                    self.last_lr = current_lr
                else:
                    print(f"\n📉 Learning rate: {current_lr:.6f}")

            # Save checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

            # Check for improvement (early stopping)
            if val_losses["total"] < self.best_val_loss:
                print(
                    f"✨ New best validation loss: {self.best_val_loss:.4f} → {val_losses['total']:.4f}"
                )
                self.best_val_loss = val_losses["total"]
                self.epochs_without_improvement = 0
                self.save_checkpoint(is_best=True)
            else:
                self.epochs_without_improvement += 1
                print(
                    f"⏸️  No improvement for {self.epochs_without_improvement} epoch(s)"
                )

                # Early stopping check
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\n⏹️  Early stopping triggered after {epoch + 1} epochs")
                    print(
                        f"   No improvement for {self.early_stopping_patience} epochs"
                    )
                    print(f"   Best validation loss: {self.best_val_loss:.4f}")
                    break

            # Time estimate
            elapsed = time.time() - start_time
            avg_epoch_time = elapsed / (epoch + 1)
            remaining_epochs = num_epochs - (epoch + 1)
            eta = avg_epoch_time * remaining_epochs

            print(f"\n⏱️  Time: {elapsed/60:.1f}m elapsed, {eta/60:.1f}m remaining")

        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Best validation loss: {self.best_val_loss:.4f}")
        print(f"   Final model: {os.path.join(self.checkpoint_dir, 'best_model.pt')}")

        self.writer.close()


def load_chunked_data(chunk_dir: str) -> list:
    """
    Load all chunks from a directory and combine them.

    Args:
        chunk_dir: Directory containing chunk_XXXX.pkl files

    Returns:
        Combined list of all examples from all chunks
    """
    import glob

    print(f"\n📦 Loading chunked data from: {chunk_dir}")

    # Find all chunk files
    chunk_pattern = os.path.join(chunk_dir, "chunk_*.pkl")
    chunk_files = sorted(glob.glob(chunk_pattern))

    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")

    print(f"   Found {len(chunk_files)} chunk files")

    # Load all chunks
    all_examples = []

    with tqdm(total=len(chunk_files), desc="Loading chunks", unit="chunk") as pbar:
        for chunk_file in chunk_files:
            try:
                chunk_examples = load_dataset(chunk_file)
                all_examples.extend(chunk_examples)
                pbar.update(1)
                pbar.set_postfix({"total_positions": len(all_examples)})
            except Exception as e:
                print(f"⚠️  Warning: Failed to load {chunk_file}: {e}")
                continue

    print(
        f"✅ Loaded {len(all_examples):,} total examples from {len(chunk_files)} chunks"
    )

    return all_examples


def main():
    """Main training script"""
    parser = argparse.ArgumentParser(description="Train chess engine")
    # Required arguments
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data (.pkl file or directory with chunks)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")

    # Model architecture arguments
    parser.add_argument(
        "--cnn-blocks", type=int, default=10, help="Number of CNN residual blocks"
    )
    parser.add_argument(
        "--cnn-filters", type=int, default=256, help="Number of CNN filters"
    )
    parser.add_argument(
        "--cnn-dropout", type=float, default=0.0, help="CNN dropout rate"
    )
    parser.add_argument("--use-rnn", action="store_true", help="Use RNN (hybrid model)")
    parser.add_argument(
        "--rnn-hidden-size", type=int, default=256, help="RNN hidden size"
    )
    parser.add_argument(
        "--rnn-layers", type=int, default=2, help="Number of RNN layers"
    )
    parser.add_argument(
        "--rnn-dropout", type=float, default=0.3, help="RNN dropout rate"
    )
    parser.add_argument(
        "--rnn-attention", action="store_true", help="Use attention in RNN"
    )
    parser.add_argument(
        "--fusion-type",
        type=str,
        default="gated",
        choices=["concat", "gated", "attention"],
        help="Feature fusion type",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Resume from checkpoint"
    )

    # Training options
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--policy-weight", type=float, default=1.0, help="Weight for policy loss"
    )
    parser.add_argument(
        "--value-weight", type=float, default=1.0, help="Weight for value loss"
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=10,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=2, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from: {args.data}")

    # Check if input is a directory (chunked) or file (single dataset)
    if os.path.isdir(args.data):
        # Chunked data - load all chunks
        examples = load_chunked_data(args.data)
    else:
        # Single file
        examples = load_dataset(args.data)

    # Split train/val using split_examples for proper shuffling and reproducibility
    train_examples, val_examples = split_examples(
        examples,
        train_ratio=1.0 - args.val_split,
        shuffle=True,
        seed=42,  # Reproducibility
    )

    print(f"Training examples: {len(train_examples):,}")
    print(f"Validation examples: {len(val_examples):,}")

    # Create datasets
    board_encoder = BoardEncoder()
    move_encoder = MoveEncoder()

    # Training dataset: Enable caching and augmentation
    train_dataset = ChessDataset(
        train_examples,
        board_encoder,
        move_encoder,
        cache_tensors=True,
        augment=True,
    )

    # Validation dataset: Enable caching, disable augmentation
    val_dataset = ChessDataset(
        val_examples,
        board_encoder,
        move_encoder,
        cache_tensors=True,
        augment=False,
    )

    # Create dataloaders
    # Note: num_workers=0 on Windows to avoid multiprocessing issues with cached tensors
    train_loader = create_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Create model
    print(f"\nCreating model...")
    model = HybridChessNet(
        # CNN parameters
        cnn_residual_blocks=args.cnn_blocks,
        cnn_filters=args.cnn_filters,
        cnn_dropout=args.cnn_dropout,
        # RNN parameters
        rnn_hidden_size=args.rnn_hidden_size,
        rnn_num_layers=args.rnn_layers,
        rnn_dropout=args.rnn_dropout,
        rnn_use_attention=args.rnn_attention,
        # Fusion parameters
        fusion_type=args.fusion_type,
        # Mode
        use_rnn=args.use_rnn,
    )

    print(f"Model: {'Hybrid CNN-RNN' if args.use_rnn else 'CNN-only'}")
    print(f"  CNN: {args.cnn_blocks} blocks, {args.cnn_filters} filters")
    if args.use_rnn:
        print(f"  RNN: {args.rnn_layers} layers, {args.rnn_hidden_size} hidden size")
        print(f"  RNN: dropout={args.rnn_dropout}, attention={args.rnn_attention}")
        print(f"  Fusion: {args.fusion_type}")
    print(f"Parameters: {count_parameters(model):,}")

    # Create trainer
    trainer = ChessTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        use_mixed_precision=not args.no_mixed_precision,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        early_stopping_patience=args.early_stopping,
        warmup_epochs=args.warmup_epochs,
        gradient_accumulation_steps=args.grad_accum,
    )

    # Resume from checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Train
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    # If run without arguments, use test mode
    if len(sys.argv) == 1:
        print("Running in TEST mode with sample data...")
        print(
            "For real training, use: python train_supervised.py --data <path> --epochs <n>"
        )
        print("\nTesting with sample data...")

        # Create sample data if needed
        from chess_engine.data.chess_dataset import (
            ChessGameParser,
            save_dataset,
        )

        pgn_path = "data/pgn/sample_games.pgn"
        data_path = "data/processed/test_train_data.pkl"

        if not os.path.exists(data_path):
            parser = ChessGameParser(min_elo=1500)
            examples = parser.parse_pgn_file(pgn_path)
            save_dataset(examples, data_path)

        # Test training with sample data
        sys.argv = [
            "train_supervised.py",
            "--data",
            data_path,
            "--epochs",
            "2",
            "--batch-size",
            "32",
            "--cnn-blocks",
            "3",
        ]

        main()
    else:
        main()
