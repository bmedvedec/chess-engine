"""
Provides a standalone decode_move function for use without MoveEncoder instance.
"""

import chess
from typing import Optional
from chess_engine.utils.move_encoder import MoveEncoder


# Module-level shared encoder instance
_shared_encoder = MoveEncoder()


def decode_move(index: int, board: Optional[chess.Board] = None) -> chess.Move:
    """
    Decode an integer index to a chess move.

    Convenience function that uses a shared MoveEncoder instance.

    Args:
        index: Integer index in range [0, 4671]
        board: Optional chess.Board to resolve queen promotions

    Returns:
        chess.Move object
    """
    return _shared_encoder.decode_move(index, board)
