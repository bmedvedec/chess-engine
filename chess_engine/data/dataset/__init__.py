"""
Dataset Package

Re-exports all public classes and functions from the dataset module.
"""

from chess_engine.data.dataset.pgn_parser import ChessGameParser
from chess_engine.data.dataset.dataset import ChessDataset
from chess_engine.data.dataset.dataloader import create_dataloader
from chess_engine.data.dataset.splits import split_examples
from chess_engine.data.dataset.stats import (
    compute_dataset_statistics,
    print_dataset_statistics,
)
from chess_engine.data.dataset.storage import save_dataset, load_dataset


__all__ = [
    # Dataset
    "ChessGameParser",
    "ChessDataset",
    "create_dataloader",
    "split_examples",
    "compute_dataset_statistics",
    "print_dataset_statistics",
    "save_dataset",
    "load_dataset",
]
