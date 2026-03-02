import chess
import torch
import numpy as np
from typing import List, Tuple, Optional, Union

from chess_engine.utils.move_encoder import MoveEncoder, PAD_INDEX


class MoveHistory:
    """
    Manages move history for RNN input.
    Handles variable-length sequences with padding.
    """

    def __init__(self, max_length: int = 50):
        """
        Initialize move history manager.

        Args:
            max_length: Maximum number of moves to keep in history
        """
        self.max_length = max_length
        self.encoder = MoveEncoder()

    def encode_move_sequence(
        self, moves: List[chess.Move], pad: bool = True
    ) -> torch.Tensor:
        """
        Encode a sequence of moves to indices.

        Args:
            moves: List of chess.Move objects
            pad: Whether to pad to max_length

        Returns:
            torch.LongTensor of shape (seq_length,) or (max_length,) if padded
        """
        # Take only the most recent moves
        recent_moves = (
            moves[-self.max_length :] if len(moves) > self.max_length else moves
        )

        # Encode moves
        encoded = [self.encoder.encode_move(move) for move in recent_moves]

        if pad:
            # Pad at the beginning with zeros
            padded = [0] * (self.max_length - len(encoded)) + encoded
            return torch.LongTensor(padded)
        else:
            return torch.LongTensor(encoded)

    def encode_game_history(self, board: chess.Board, pad: bool = True) -> torch.Tensor:
        """
        Encode the move history from a chess.Board.

        Args:
            board: chess.Board object
            pad: Whether to pad to max_length

        Returns:
            torch.LongTensor of move indices
        """
        # Get move stack from board
        moves = list(board.move_stack)
        return self.encode_move_sequence(moves, pad=pad)

    def encode_board(
        self, board: chess.Board, pad: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """
        Encode the move history from a chess.Board.

        Backward-compatible method used by MCTS evaluator.

        Args:
            board: chess.Board object
            pad: Whether to pad to max_length

        Returns:
            Tuple of (encoded_sequence, actual_length)
        """
        moves = list(board.move_stack)
        actual_length = min(len(moves), self.max_length)
        encoded = self.encode_move_sequence(moves, pad=pad)
        return encoded, actual_length

    def batch_encode_histories(
        self, boards: List[chess.Board], pad: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode move histories for a batch of boards.

        Args:
            boards: List of chess.Board objects
            pad: Whether to pad sequences

        Returns:
            Tuple of (encoded_sequences, lengths)
            - encoded_sequences: torch.LongTensor of shape (batch, max_length)
            - lengths: torch.LongTensor of shape (batch,) containing actual lengths
        """
        sequences = []
        lengths = []

        for board in boards:
            moves = list(board.move_stack)
            seq_length = min(len(moves), self.max_length)
            lengths.append(seq_length)

            encoded = self.encode_move_sequence(moves, pad=pad)
            sequences.append(encoded)

        sequences = torch.stack(sequences)
        lengths = torch.LongTensor(lengths)

        return sequences, lengths

    def decode_move_sequence(
        self, encoded: Union[torch.Tensor, np.ndarray], length: Optional[int] = None
    ) -> List[chess.Move]:
        """
        Decode a sequence of move indices back to moves.

        Args:
            encoded: torch.LongTensor of move indices
            length: Actual length (ignores padding)

        Returns:
            List of chess.Move objects
        """
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.cpu().numpy()

        if length is not None:
            # Remove padding
            encoded = encoded[-length:]

        moves = []
        for idx in encoded:
            if idx > 0:  # Skip padding (index 0)
                move = self.encoder.decode_move(int(idx))
                moves.append(move)

        return moves
