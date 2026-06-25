"""
SUPERVISED TRAINING - Learning Rate Schedulers

Provides scheduler creation helpers used by ChessTrainer.
"""

import torch
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau


def create_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int = 2,
) -> LambdaLR:
    """Linearly increases LR from 0 to base LR over warmup_epochs."""

    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        return 1.0

    return LambdaLR(optimizer, lr_lambda=warmup_lambda)


def create_plateau_scheduler(
    optimizer: torch.optim.Optimizer,
    factor: float = 0.5,
    patience: int = 3,
    min_lr: float = 1e-7,
) -> ReduceLROnPlateau:
    """ReduceLROnPlateau: halve LR after `patience` epochs without improvement."""
    return ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr
    )
