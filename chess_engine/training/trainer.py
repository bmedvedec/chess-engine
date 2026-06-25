"""
SUPERVISED TRAINING - ChessTrainer Class

Main trainer orchestrator that ties together:
- Training and validation loops (supervised_loop.py)
- Learning rate scheduling (schedulers.py)
- Checkpointing (save/load)
- Early stopping
- TensorBoard logging
"""

import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp.grad_scaler import GradScaler

from chess_engine.training.schedulers import (
    create_warmup_scheduler,
    create_plateau_scheduler,
)
from chess_engine.training.supervised_loop import train_epoch, validate
from chess_engine.utils.move_encoder import MoveEncoder


def _count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ChessTrainer:
    """Supervised training with AMP, warmup scheduler, early stopping, and checkpointing."""

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
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.policy_weight = policy_weight
        self.value_weight = value_weight

        self.early_stopping_patience = early_stopping_patience
        self.epochs_without_improvement = 0

        self.gradient_accumulation_steps = gradient_accumulation_steps

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        self.scheduler = create_plateau_scheduler(self.optimizer)

        self.warmup_epochs = warmup_epochs
        self.warmup_scheduler = create_warmup_scheduler(self.optimizer, warmup_epochs)

        self.last_lr = learning_rate

        self.use_mixed_precision = use_mixed_precision and device.type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_mixed_precision else None

        self.writer = SummaryWriter(log_dir)

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        self.move_encoder = MoveEncoder()

        print(f"Trainer initialized")
        print(f"   Model parameters: {_count_parameters(model):,}")
        print(f"   Device: {device}")
        print(f"   Mixed precision: {self.use_mixed_precision}")
        print(f"   Policy weight: {policy_weight}")
        print(f"   Value weight: {value_weight}")
        print(f"   Early stopping patience: {early_stopping_patience}")
        print(f"   Warmup epochs: {warmup_epochs}")
        print(f"   Gradient accumulation: {gradient_accumulation_steps}")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")

    def save_checkpoint(self, filename: str = "checkpoint.pt", is_best: bool = False):
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "warmup_scheduler_state_dict": self.warmup_scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "config": {
                "policy_weight": self.policy_weight,
                "value_weight": self.value_weight,
                "early_stopping_patience": self.early_stopping_patience,
                "warmup_epochs": self.warmup_epochs,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "use_mixed_precision": self.use_mixed_precision,
            },
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint: {filepath}")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

    def load_checkpoint(self, filepath: str):
        """Load checkpoint and resume training."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "warmup_scheduler_state_dict" in checkpoint:
            self.warmup_scheduler.load_state_dict(
                checkpoint["warmup_scheduler_state_dict"]
            )

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        if "epochs_without_improvement" in checkpoint:
            self.epochs_without_improvement = checkpoint["epochs_without_improvement"]

        print(f"Loaded checkpoint from epoch {self.epoch}")
        if "config" in checkpoint:
            print(f"   Configuration: {checkpoint['config']}")

    def train(self, num_epochs: int):
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)

        start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch

            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*80}")

            train_losses, self.global_step = train_epoch(
                model=self.model,
                train_loader=self.train_loader,
                optimizer=self.optimizer,
                device=self.device,
                epoch=epoch,
                global_step=self.global_step,
                policy_weight=self.policy_weight,
                value_weight=self.value_weight,
                use_mixed_precision=self.use_mixed_precision,
                scaler=self.scaler,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                writer=self.writer,
            )

            print(f"\nTraining metrics:")
            print(f"   Loss: {train_losses['total']:.4f}")
            print(f"   Policy loss: {train_losses['policy']:.4f}")
            print(f"   Value loss: {train_losses['value']:.4f}")
            print(f"   Top-1 accuracy: {train_losses['policy_top1_acc']:.3f}")
            print(f"   Top-3 accuracy: {train_losses['policy_top3_acc']:.3f}")
            print(f"   Top-5 accuracy: {train_losses['policy_top5_acc']:.3f}")
            print(f"   Value MAE: {train_losses['value_mae']:.4f}")

            val_losses = validate(
                model=self.model,
                val_loader=self.val_loader,
                device=self.device,
                epoch=epoch,
                policy_weight=self.policy_weight,
                value_weight=self.value_weight,
                writer=self.writer,
            )

            print(f"\nValidation metrics:")
            print(f"   Loss: {val_losses['total']:.4f}")
            print(f"   Policy loss: {val_losses['policy']:.4f}")
            print(f"   Value loss: {val_losses['value']:.4f}")
            print(f"   Top-1 accuracy: {val_losses['policy_top1_acc']:.3f}")
            print(f"   Top-3 accuracy: {val_losses['policy_top3_acc']:.3f}")
            print(f"   Top-5 accuracy: {val_losses['policy_top5_acc']:.3f}")
            print(f"   Value MAE: {val_losses['value_mae']:.4f}")

            if epoch < self.warmup_epochs:
                self.warmup_scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"\nWarmup phase - Learning rate: {current_lr:.6f}")
            else:
                self.scheduler.step(val_losses["total"])
                current_lr = self.optimizer.param_groups[0]["lr"]

                if current_lr != self.last_lr:
                    print(
                        f"\nLearning rate reduced: {self.last_lr:.6f} -> {current_lr:.6f}"
                    )
                    self.last_lr = current_lr
                else:
                    print(f"\nLearning rate: {current_lr:.6f}")

            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

            if val_losses["total"] < self.best_val_loss:
                print(
                    f"New best validation loss: {self.best_val_loss:.4f} -> {val_losses['total']:.4f}"
                )
                self.best_val_loss = val_losses["total"]
                self.epochs_without_improvement = 0
                self.save_checkpoint(is_best=True)
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement} epoch(s)")

                if self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    print(
                        f"   No improvement for {self.early_stopping_patience} epochs"
                    )
                    print(f"   Best validation loss: {self.best_val_loss:.4f}")
                    break

            elapsed = time.time() - start_time
            avg_epoch_time = elapsed / (epoch + 1)
            remaining_epochs = num_epochs - (epoch + 1)
            eta = avg_epoch_time * remaining_epochs

            print(f"\nTime: {elapsed/60:.1f}m elapsed, {eta/60:.1f}m remaining")

        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"   Total time: {total_time/60:.1f} minutes")
        print(f"   Best validation loss: {self.best_val_loss:.4f}")
        print(f"   Final model: {os.path.join(self.checkpoint_dir, 'best_model.pt')}")

        self.writer.close()
