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
    """
    Create a warmup learning rate scheduler.

    Linearly increases learning rate from 0 to base LR over warmup_epochs.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs (default: 2)

    Returns:
        LambdaLR scheduler for warmup
    """

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
    """
    Create a ReduceLROnPlateau scheduler.

    Reduces learning rate when validation loss plateaus.

    Args:
        optimizer: PyTorch optimizer
        factor: Factor to reduce LR by (default: 0.5)
        patience: Epochs to wait before reducing (default: 3)
        min_lr: Minimum learning rate (default: 1e-7)

    Returns:
        ReduceLROnPlateau scheduler
    """
    return ReduceLROnPlateau(
        optimizer, mode="min", factor=factor, patience=patience, min_lr=min_lr
    )
