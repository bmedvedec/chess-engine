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
    policy_loss = F.cross_entropy(policy_logits, move_targets)
    value_loss = F.mse_loss(value_pred.squeeze(), outcome_targets)
    total_loss = policy_weight * policy_loss + value_weight * value_loss

    with torch.no_grad():
        _, predicted_moves = torch.max(policy_logits, dim=1)
        top1_accuracy = (predicted_moves == move_targets).float().mean()

        _, top3_indices = torch.topk(policy_logits, k=3, dim=1)
        top3_accuracy = (
            (top3_indices == move_targets.unsqueeze(1)).any(dim=1).float().mean()
        )

        _, top5_indices = torch.topk(policy_logits, k=5, dim=1)
        top5_accuracy = (
            (top5_indices == move_targets.unsqueeze(1)).any(dim=1).float().mean()
        )

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
        boards = boards.to(device)
        moves = moves.to(device)
        outcomes = outcomes.to(device)

        with autocast("cuda", enabled=use_mixed_precision):
            policy_logits, value_pred, _ = model(boards)

            loss, loss_dict = compute_loss(
                policy_logits,
                value_pred,
                moves,
                outcomes,
                policy_weight=policy_weight,
                value_weight=value_weight,
            )

            loss = loss / gradient_accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()

        for key in total_losses:
            total_losses[key] += loss_dict[key]

        pbar.set_postfix(
            {
                "loss": f"{loss_dict['total']:.4f}",
                "top1": f"{loss_dict['policy_top1_acc']:.3f}",
                "top5": f"{loss_dict['policy_top5_acc']:.3f}",
            }
        )

        if writer is not None and global_step % 10 == 0:
            for key, value in loss_dict.items():
                writer.add_scalar(f"train/{key}", value, global_step)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/learning_rate", current_lr, global_step)

        global_step += 1

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
            boards = boards.to(device)
            moves = moves.to(device)
            outcomes = outcomes.to(device)

            policy_logits, value_pred, _ = model(boards)

            loss, loss_dict = compute_loss(
                policy_logits,
                value_pred,
                moves,
                outcomes,
                policy_weight=policy_weight,
                value_weight=value_weight,
            )

            for key in total_losses:
                total_losses[key] += loss_dict[key]

    num_batches = len(val_loader)
    avg_losses = {key: value / num_batches for key, value in total_losses.items()}

    if writer is not None:
        for key, value in avg_losses.items():
            writer.add_scalar(f"val/{key}", value, epoch)

    return avg_losses
