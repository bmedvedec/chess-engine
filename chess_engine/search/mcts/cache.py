"""
MCTS - Position Cache (Transposition Table)

Provides a simple dictionary-based cache for evaluated positions.
"""

import chess
from typing import Dict, Tuple, Optional


class PositionCache:
    """
    Simple dictionary-based position cache (transposition table) for MCTS.

    Caches neural network evaluations keyed by FEN string to avoid
    re-evaluating the same position.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize position cache.

        Args:
            max_size: Maximum number of cached positions (default: 10000)
        """
        self.max_size = max_size
        self._cache: Dict[str, Tuple[Dict[chess.Move, float], float]] = {}

    def get(self, fen: str) -> Optional[Tuple[Dict[chess.Move, float], float]]:
        """
        Look up a position in the cache.

        Args:
            fen: FEN string of the position

        Returns:
            Tuple of (policy_probs, value) if cached, None otherwise
        """
        return self._cache.get(fen, None)

    def put(
        self, fen: str, policy_probs: Dict[chess.Move, float], value: float
    ) -> None:
        """
        Store a position evaluation in the cache.

        Args:
            fen: FEN string of the position
            policy_probs: Policy probabilities for legal moves
            value: Position value
        """
        if len(self._cache) < self.max_size:
            self._cache[fen] = (policy_probs, value)

    def clear(self) -> None:
        """Clear the cache (call between moves)."""
        self._cache.clear()

    def __len__(self) -> int:
        """Return the number of cached positions."""
        return len(self._cache)
