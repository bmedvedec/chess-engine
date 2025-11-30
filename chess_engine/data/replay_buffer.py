"""
CHAPTER 6+: REPLAY BUFFER FOR REINFORCEMENT LEARNING
Required for Self-Play Training

This module provides a replay buffer for storing and sampling
self-play training examples during reinforcement learning.

Usage:
    buffer = ReplayBuffer(max_size=100000)

    # During self-play
    buffer.add(board, policy_targets, value_target)

    # During training
    batch = buffer.sample(batch_size=256)
"""

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

import chess
import torch


class ReplayBuffer:
    """
    Replay buffer for storing self-play training examples.

    Stores positions from self-play games and provides sampling
    for training the neural network.

    Features:
    - Fixed maximum size (FIFO when full)
    - Random sampling for training
    - Optional prioritized sampling (future enhancement)
    """

    def __init__(self, max_size: int = 100000):
        """
        Initialize replay buffer.

        Args:
            max_size: Maximum number of examples to store
        """
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)

    def add(
        self,
        board: chess.Board,
        policy_target: torch.Tensor,
        value_target: float,
        move_history: Optional[List[chess.Move]] = None,
    ) -> None:
        """
        Add a training example to the buffer.

        Args:
            board: Chess board position
            policy_target: Target policy distribution (visit counts from MCTS)
            value_target: Target value (game outcome: 1.0, 0.0, -1.0)
            move_history: Optional move history for RNN
        """
        example = {
            "board": board.copy(),
            "policy": policy_target,
            "value": value_target,
            "move_history": move_history.copy() if move_history else [],
        }
        self.buffer.append(example)

    def add_game(
        self,
        positions: List[chess.Board],
        policies: List[torch.Tensor],
        outcome: float,
        move_histories: Optional[List[List[chess.Move]]] = None,
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
            # Flip value based on whose turn it is
            value = outcome if board.turn == chess.WHITE else -outcome

            self.add(board, policy, value, history)

    def sample(self, batch_size: int) -> Dict[str, List]:
        """
        Sample a random batch of examples.

        Args:
            batch_size: Number of examples to sample

        Returns:
            Dictionary with keys:
            - 'boards': List of chess.Board
            - 'policies': List of policy tensors
            - 'values': List of value targets
            - 'move_histories': List of move history lists
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        batch = random.sample(self.buffer, batch_size)

        return {
            "boards": [ex["board"] for ex in batch],
            "policies": [ex["policy"] for ex in batch],
            "values": [ex["value"] for ex in batch],
            "move_histories": [ex["move_history"] for ex in batch],
        }

    def __len__(self) -> int:
        """Return current number of examples in buffer"""
        return len(self.buffer)

    def clear(self) -> None:
        """Clear all examples from buffer"""
        self.buffer.clear()

    def is_full(self) -> bool:
        """Check if buffer is at maximum capacity"""
        return len(self.buffer) >= self.max_size

    def save(self, filepath: str) -> None:
        """Save buffer to disk"""
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(list(self.buffer), f)
        print(f"✅ Saved {len(self.buffer)} examples to {filepath}")

    def load(self, filepath: str) -> None:
        """Load buffer from disk"""
        import pickle

        with open(filepath, "rb") as f:
            examples = pickle.load(f)
        self.buffer = deque(examples, maxlen=self.max_size)
        print(f"✅ Loaded {len(self.buffer)} examples from {filepath}")


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Replay buffer with prioritized sampling.

    Samples examples based on their TD-error or other priority metric.
    Higher priority examples are sampled more frequently.

    Note: This is an advanced feature for later optimization.
    """

    def __init__(self, max_size: int = 100000, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer.

        Args:
            max_size: Maximum number of examples to store
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
        """
        super().__init__(max_size)
        self.alpha = alpha
        self.priorities: deque = deque(maxlen=max_size)

    def add(
        self,
        board: chess.Board,
        policy_target: torch.Tensor,
        value_target: float,
        move_history: Optional[List[chess.Move]] = None,
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

        # Compute sampling probabilities
        priorities = torch.tensor(list(self.priorities), dtype=torch.float32)
        probs = priorities**self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = torch.multinomial(probs, batch_size, replacement=False)

        # Get examples
        batch = [self.buffer[i] for i in indices]

        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize

        return {
            "boards": [ex["board"] for ex in batch],
            "policies": [ex["policy"] for ex in batch],
            "values": [ex["value"] for ex in batch],
            "move_histories": [ex["move_history"] for ex in batch],
            "weights": weights.tolist(),
            "indices": indices.tolist(),
        }

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """
        Update priorities for sampled examples.

        Args:
            indices: Indices of examples to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
