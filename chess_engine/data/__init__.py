"""
Data Package

Re-exports all public classes and functions from the data module.
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

from chess_engine.data.replay.storage import (
    GameExample,
    convert_policy_dict_to_tensor,
    convert_policy_tensor_to_dict,
)
from chess_engine.data.replay.buffer import ReplayBuffer
from chess_engine.data.replay.prioritized import PrioritizedReplayBuffer
from chess_engine.data.replay.sampling import prioritized_sample
from chess_engine.data.replay.serialization import (
    save_replay_buffer,
    load_replay_buffer,
)


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
    # Replay
    "GameExample",
    "convert_policy_dict_to_tensor",
    "convert_policy_tensor_to_dict",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "prioritized_sample",
    "save_replay_buffer",
    "load_replay_buffer",
]
