"""
DATA LOADING & PREPROCESSING - DataLoader Factory

Creates PyTorch DataLoader with GPU-optimized settings.
"""

import torch
from torch.utils.data import DataLoader, Dataset

# Constants
DEFAULT_BATCH_SIZE = 256
DEFAULT_NUM_WORKERS = 4


def create_dataloader(
    dataset: Dataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> DataLoader:
    """
    Create DataLoader for training.

    Args:
        dataset: ChessDataset instance
        batch_size: Batch size (default: 256)
        shuffle: Whether to shuffle (default: True)
        num_workers: Number of worker processes (default: 4)

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Speed up GPU transfer
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
    )
