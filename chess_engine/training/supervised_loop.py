"""
SUPERVISED TRAINING - Training & Validation Loops

Contains the core training loop logic with mixed precision, gradient accumulation,
and comprehensive metrics (top-1/3/5 accuracy, value MAE).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def compute_loss(
    policy_logits: torch.Tensor,
    value_pred: torch.Tensor,
    move_targets: torch.Tensor,
    outcome_targets: torch.Tensor,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict]:
    """
    Compute training loss.

    Loss = (policy_weight * Policy Loss) + (value_weight * Value Loss)
    - Policy Loss: Cross-entropy between predicted policy and actual move
    - Value Loss: MSE between predicted value and game outcome

    Also computes Top-1, Top-3, Top-5 accuracy and Value MAE.

    Args:
        policy_logits: Model policy output (batch, 4096)
        value_pred: Model value output (batch, 1)
        move_targets: Target move indices (batch,)
        outcome_targets: Game outcomes (batch,)
        policy_weight: Weight for policy loss (default: 1.0)
        value_weight: Weight for value loss (default: 1.0)

    Returns:
        Tuple of (total_loss, loss_dict)
    """
    # Policy loss (cross-entropy)
    policy_loss = F.cross_entropy(policy_logits, move_targets)

    # Value loss (MSE)
    value_loss = F.mse_loss(value_pred.squeeze(), outcome_targets)

    # Total loss (weighted sum)
    total_loss = policy_weight * policy_loss + value_weight * value_loss

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


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    use_mixed_precision: bool = False,
    scaler: Optional[GradScaler] = None,
    gradient_accumulation_steps: int = 1,
    writer: Optional[SummaryWriter] = None,
) -> Tuple[Dict[str, float], int]:
    """
    Train for one epoch with mixed precision and gradient accumulation.

    Args:
        model: Neural network model
        train_loader: Training DataLoader
        optimizer: Optimizer
        device: Training device
        epoch: Current epoch number
        global_step: Current global step (for TensorBoard)
        policy_weight: Weight for policy loss
        value_weight: Weight for value loss
        use_mixed_precision: Enable AMP
        scaler: GradScaler for mixed precision
        gradient_accumulation_steps: Gradient accumulation steps
        writer: TensorBoard writer

    Returns:
        Tuple of (avg_losses dict, updated global_step)
    """
    model.train()

    total_losses = {
        "total": 0.0,
        "policy": 0.0,
        "value": 0.0,
        "policy_top1_acc": 0.0,
        "policy_top3_acc": 0.0,
        "policy_top5_acc": 0.0,
        "value_mae": 0.0,
    }

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for batch_idx, (boards, moves, outcomes) in enumerate(pbar):
        # Move to device
        boards = boards.to(device)
        moves = moves.to(device)
        outcomes = outcomes.to(device)

        # Mixed precision forward pass
        with autocast("cuda", enabled=use_mixed_precision):
            # Forward pass
            policy_logits, value_pred, _ = model(boards)

            # Compute loss
            loss, loss_dict = compute_loss(
                policy_logits,
                value_pred,
                moves,
                outcomes,
                policy_weight=policy_weight,
                value_weight=value_weight,
            )

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

        # Backward pass with mixed precision
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights every accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                # Unscale before clipping
                scaler.unscale_(optimizer)
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
            else:
                # Gradient clipping (prevent exploding gradients)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Zero gradients
            optimizer.zero_grad()

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
        if writer is not None and global_step % 10 == 0:
            for key, value in loss_dict.items():
                writer.add_scalar(f"train/{key}", value, global_step)
            # Log learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/learning_rate", current_lr, global_step)

        global_step += 1

    # Calculate averages
    num_batches = len(train_loader)
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}

    return avg_losses, global_step


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    epoch: int,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:
    """
    Run validation.

    Args:
        model: Neural network model
        val_loader: Validation DataLoader
        device: Device
        epoch: Current epoch number
        policy_weight: Weight for policy loss
        value_weight: Weight for value loss
        writer: TensorBoard writer

    Returns:
        Dictionary of validation metrics
    """
    model.eval()

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
        for boards, moves, outcomes in tqdm(val_loader, desc="Validating", leave=False):
            # Move to device
            boards = boards.to(device)
            moves = moves.to(device)
            outcomes = outcomes.to(device)

            # Forward pass
            policy_logits, value_pred, _ = model(boards)

            # Compute loss
            loss, loss_dict = compute_loss(
                policy_logits,
                value_pred,
                moves,
                outcomes,
                policy_weight=policy_weight,
                value_weight=value_weight,
            )

            # Accumulate losses
            for key in total_losses:
                total_losses[key] += loss_dict[key]

    # Calculate averages
    num_batches = len(val_loader)
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}

    # Log to TensorBoard
    if writer is not None:
        for key, value in avg_losses.items():
            writer.add_scalar(f"val/{key}", value, epoch)

    return avg_losses
