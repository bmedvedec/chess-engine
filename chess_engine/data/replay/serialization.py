"""
REPLAY BUFFER - Serialization Utilities

Standalone save/load functions for replay buffers.
The main save/load logic lives on the buffer classes themselves.
Kept for backward-compatible imports.
"""

import os
import pickle

from chess_engine.data.replay.buffer import ReplayBuffer
from chess_engine.data.replay.prioritized import PrioritizedReplayBuffer


def save_replay_buffer(buffer: ReplayBuffer, filepath: str) -> None:
    """
    Save a replay buffer to disk.

    Args:
        buffer: ReplayBuffer or PrioritizedReplayBuffer to save
        filepath: Path to save to
    """
    buffer.save(filepath)


def load_replay_buffer(filepath: str, prioritized: bool = False) -> ReplayBuffer:
    """
    Load a replay buffer from disk.

    Args:
        filepath: Path to load from
        prioritized: If True, load as PrioritizedReplayBuffer

    Returns:
        Loaded buffer instance
    """
    if prioritized:
        buffer = PrioritizedReplayBuffer()
    else:
        buffer = ReplayBuffer()

    buffer.load(filepath)
    return buffer
