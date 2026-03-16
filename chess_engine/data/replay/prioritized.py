"""
REPLAY BUFFER - Prioritized Replay Buffer

Implements Prioritized Experience Replay (PER) where examples with higher
TD-error or priority are sampled more frequently.
"""

import os
import pickle
from collections import deque
from typing import Dict, List, Optional, Sequence, Union

import chess
import torch

from chess_engine.data.replay.buffer import ReplayBuffer
from chess_engine.data.replay.storage import GameExample


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Replay buffer with prioritized sampling.

    Samples examples based on their TD-error or other priority metric.
    Higher priority examples are sampled more frequently.

    Implements Prioritized Experience Replay (PER).
    """

    def __init__(
        self, max_size: int = 100000, alpha: float = 0.6, memory_efficient: bool = True
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            max_size: Maximum number of examples to store
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            memory_efficient: Use memory-efficient storage
        """
        super().__init__(max_size, memory_efficient)
        self.alpha = alpha
        self.priorities: deque = deque(maxlen=max_size)

    def add(
        self,
        board: chess.Board,
        policy_target: Union[torch.Tensor, Dict[str, float]],
        value_target: float,
        move_history: Optional[Sequence[chess.Move]] = None,
        priority: float = 1.0,
    ) -> None:
        """
        Add example with priority.

        Args:
            board: Chess board position
            policy_target: Target policy distribution
            value_target: Target value
            move_history: Optional move history
            priority: Sample priority (default: 1.0 = max priority)
        """
        super().add(board, policy_target, value_target, move_history)
        self.priorities.append(priority)

    def add_game_example(self, example: GameExample, priority: float = 1.0) -> None:
        """Add GameExample with priority."""
        super().add_game_example(example)
        self.priorities.append(priority)

    def clear(self) -> None:
        """Clear all examples and their priorities."""
        super().clear()
        self.priorities.clear()

    def sample(self, batch_size: int, beta: float = 0.4) -> Dict[str, List]:
        """
        Sample batch using priorities.

        Args:
            batch_size: Number of examples to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full)

        Returns:
            Dictionary with sampled examples and importance weights
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # Guard: priorities and buffer must be the same length
        assert len(self.priorities) == len(
            self.buffer
        ), f"Priority/buffer length mismatch: {len(self.priorities)} vs {len(self.buffer)}"

        # Compute sampling probabilities (add epsilon to prevent division by zero)
        priorities = torch.tensor(list(self.priorities), dtype=torch.float32)
        probs = (priorities + 1e-8) ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = torch.multinomial(probs, batch_size, replacement=False)

        # Get examples
        batch = [self.buffer[i] for i in indices]

        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize

        if self.memory_efficient:
            result = {
                "boards": [chess.Board(example["fen"]) for example in batch],
                "policies": [example["policy"] for example in batch],
                "values": [example["value"] for example in batch],
                "move_histories": [
                    [chess.Move.from_uci(m) for m in example["move_history"]]
                    for example in batch
                ],
                "weights": weights.tolist(),
                "indices": indices.tolist(),
            }
        else:
            result = {
                "boards": [example["board"] for example in batch],
                "policies": [example["policy"] for example in batch],
                "values": [example["value"] for example in batch],
                "move_histories": [example["move_history"] for example in batch],
                "weights": weights.tolist(),
                "indices": indices.tolist(),
            }

        return result

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        Update priorities for sampled examples.

        Args:
            indices: Indices of examples to update
            priorities: New priority values (usually |TD-error| + epsilon)
        """
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority

    def save(self, filepath: str) -> None:
        """Save prioritized buffer to disk."""
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "buffer": list(self.buffer),
                    "priorities": list(self.priorities),
                    "memory_efficient": self.memory_efficient,
                    "max_size": self.max_size,
                    "alpha": self.alpha,
                },
                f,
            )
        print(f"Saved {len(self.buffer)} prioritized examples to {filepath}")

    def load(self, filepath: str) -> None:
        """Load prioritized buffer from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.buffer = deque(data["buffer"], maxlen=self.max_size)
        self.priorities = deque(
            data.get("priorities", [1.0] * len(self.buffer)), maxlen=self.max_size
        )
        self.memory_efficient = data.get("memory_efficient", self.memory_efficient)
        self.alpha = data.get("alpha", self.alpha)

        print(f"Loaded {len(self.buffer)} prioritized examples from {filepath}")
