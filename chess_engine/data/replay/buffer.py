"""
REPLAY BUFFER - Base Buffer

Stores self-play training examples with FEN-based memory-efficient storage,
FIFO eviction, random sampling, weighted sampling, and save/load.
"""

import os
import random
import pickle
from collections import deque
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import chess
import torch

from chess_engine.data.replay.storage import GameExample


class ReplayBuffer:
    """
    Replay buffer for storing self-play training examples.

    Features:
    - Memory-efficient storage (FEN strings)
    - Fixed maximum size (FIFO when full)
    - Random sampling for training
    - Move history tracking for RNN
    - Weighted sampling for decisive games
    """

    def __init__(self, max_size: int = 100000, memory_efficient: bool = True):
        """
        Initialize replay buffer.

        Args:
            max_size: Maximum number of examples to store
            memory_efficient: If True, store FEN strings instead of Board objects
        """
        self.max_size = max_size
        self.memory_efficient = memory_efficient
        self.buffer: deque = deque(maxlen=max_size)

    def add(
        self,
        board: chess.Board,
        policy_target: Union[torch.Tensor, Dict[str, float]],
        value_target: float,
        move_history: Optional[Sequence[chess.Move]] = None,
    ) -> None:
        """
        Add a training example to the buffer.

        Args:
            board: Chess board position
            policy_target: Target policy distribution (visit counts from MCTS)
            value_target: Target value (game outcome: 1.0, 0.0, -1.0)
            move_history: Optional move history for RNN
        """
        if self.memory_efficient:
            example = {
                "fen": board.fen(),
                "policy": policy_target,
                "value": value_target,
                "move_history": (
                    [move.uci() for move in move_history] if move_history else []
                ),
            }
        else:
            example = {
                "board": board.copy(),
                "policy": policy_target,
                "value": value_target,
                "move_history": list(move_history) if move_history else [],
            }
        self.buffer.append(example)

    def add_game(
        self,
        positions: Sequence[chess.Board],
        policies: Sequence[Union[torch.Tensor, Dict[str, float]]],
        outcome: float,
        move_histories: Optional[Sequence[Sequence[chess.Move]]] = None,
    ) -> None:
        """
        Add all positions from a complete game.

        Args:
            positions: List of board positions from the game
            policies: List of policy targets (one per position)
            outcome: Game outcome from perspective of first player
            move_histories: Optional move histories for each position
        """
        if move_histories is None:
            move_histories = [[] for _ in positions]

        for i, (board, policy, history) in enumerate(
            zip(positions, policies, move_histories)
        ):
            value = outcome if board.turn == chess.WHITE else -outcome
            self.add(board, policy, value, history)

    def add_game_example(self, example: GameExample) -> None:
        """
        Add a single GameExample from self_play.

        Args:
            example: GameExample object from self_play module
        """
        internal_example = {
            "fen": example.fen,
            "policy": example.policy,
            "value": example.value,
            "move_history": [],
            "move_number": example.move_number,
        }
        self.buffer.append(internal_example)

    def add_game_examples(self, examples: List[GameExample]) -> None:
        """
        Bulk add GameExample objects from self_play.

        Args:
            examples: List of GameExample objects
        """
        for example in examples:
            self.add_game_example(example)

    def sample(
        self,
        batch_size: int,
        weighted: bool = False,
        weights: Optional[List[float]] = None,
    ) -> Dict[str, List]:
        """
        Sample a random batch of examples with optional weighting.

        Args:
            batch_size: Number of examples to sample
            weighted: If True, use weighted sampling based on provided weights
            weights: Optional sample weights (must match buffer size if provided)

        Returns:
            Dictionary with keys: 'boards', 'policies', 'values', 'move_histories'
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        if weighted:
            if weights is None:
                weights = [
                    2.0 if abs(example["value"]) > 0.1 else 1.0
                    for example in self.buffer
                ]

            if len(weights) != len(self.buffer):
                raise ValueError(
                    f"Weights length ({len(weights)}) must match buffer size ({len(self.buffer)})"
                )

            weights_array = np.array(weights, dtype=np.float64)

            if weights_array.sum() == 0:
                weights_array = np.ones_like(weights_array)

            probabilities = weights_array / weights_array.sum()

            try:
                indices = np.random.choice(
                    len(self.buffer), size=batch_size, replace=False, p=probabilities
                )
                batch = [self.buffer[i] for i in indices]
            except ValueError as e:
                print(
                    f"Warning: Weighted sampling failed ({e}), using uniform sampling"
                )
                batch = random.sample(self.buffer, batch_size)
        else:
            batch = random.sample(self.buffer, batch_size)

        if self.memory_efficient:
            return {
                "boards": [chess.Board(example["fen"]) for example in batch],
                "policies": [example["policy"] for example in batch],
                "values": [example["value"] for example in batch],
                "move_histories": [
                    [chess.Move.from_uci(m) for m in example["move_history"]]
                    for example in batch
                ],
            }
        else:
            return {
                "boards": [example["board"] for example in batch],
                "policies": [example["policy"] for example in batch],
                "values": [example["value"] for example in batch],
                "move_histories": [example["move_history"] for example in batch],
            }

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self) -> None:
        """Clear all examples from buffer."""
        self.buffer.clear()

    def is_full(self) -> bool:
        """Check if buffer is at maximum capacity."""
        return len(self.buffer) >= self.max_size

    def save(self, filepath: str) -> None:
        """Save buffer to disk."""
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "buffer": list(self.buffer),
                    "memory_efficient": self.memory_efficient,
                    "max_size": self.max_size,
                },
                f,
            )
        print(f"Saved {len(self.buffer)} examples to {filepath}")

    def load(self, filepath: str) -> None:
        """Load buffer from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict) and "buffer" in data:
            self.buffer = deque(data["buffer"], maxlen=self.max_size)
            self.memory_efficient = data.get("memory_efficient", self.memory_efficient)
        else:
            self.buffer = deque(data, maxlen=self.max_size)

        print(f"Loaded {len(self.buffer)} examples from {filepath}")

    def get_stats(self) -> Dict[str, float]:
        """Get statistics about buffer contents."""
        if len(self.buffer) == 0:
            return {
                "size": 0,
                "capacity": self.max_size,
                "fill_percentage": 0.0,
                "avg_value": 0.0,
                "value_std": 0.0,
                "positive_ratio": 0.0,
                "negative_ratio": 0.0,
                "draw_ratio": 0.0,
                "decisive_ratio": 0.0,
            }

        values = [example["value"] for example in self.buffer]

        positive_count = sum(1 for v in values if v > 0.1)
        negative_count = sum(1 for v in values if v < -0.1)
        draw_count = sum(1 for v in values if abs(v) <= 0.1)
        decisive_count = positive_count + negative_count

        return {
            "size": len(self.buffer),
            "capacity": self.max_size,
            "fill_percentage": float(len(self.buffer) / self.max_size * 100),
            "avg_value": float(np.mean(values)),
            "value_std": float(np.std(values)),
            "positive_ratio": float(positive_count / len(values)),
            "negative_ratio": float(negative_count / len(values)),
            "draw_ratio": float(draw_count / len(values)),
            "decisive_ratio": float(decisive_count / len(values)),
        }
