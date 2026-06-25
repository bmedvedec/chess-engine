import chess
import torch
import numpy as np
from typing import List, Tuple, Optional, Union

from chess_engine.utils.move_encoder import MoveEncoder, PAD_INDEX


class MoveHistory:
    """Encodes move history sequences into padded index tensors for RNN input."""

    def __init__(self, max_length: int = 50):
        self.max_length = max_length
        self.encoder = MoveEncoder()

    def encode_move_sequence(
        self, moves: List[chess.Move], pad: bool = True
    ) -> torch.Tensor:
        recent_moves = (
            moves[-self.max_length :] if len(moves) > self.max_length else moves
        )

        encoded = [self.encoder.encode_move(move) for move in recent_moves]

        if pad:
            padded = [0] * (self.max_length - len(encoded)) + encoded
            return torch.LongTensor(padded)
        else:
            return torch.LongTensor(encoded)

    def encode_game_history(self, board: chess.Board, pad: bool = True) -> torch.Tensor:
        moves = list(board.move_stack)
        return self.encode_move_sequence(moves, pad=pad)

    def encode_board(
        self, board: chess.Board, pad: bool = True
    ) -> Tuple[torch.Tensor, int]:
        """Backward-compatible alias used by MCTS evaluator."""
        moves = list(board.move_stack)
        actual_length = min(len(moves), self.max_length)
        encoded = self.encode_move_sequence(moves, pad=pad)
        return encoded, actual_length

    def batch_encode_histories(
        self, boards: List[chess.Board], pad: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.cpu().numpy()

        if length is not None:
            encoded = encoded[-length:]

        moves = []
        for idx in encoded:
            if idx > 0:  # Skip padding (index 0)
                move = self.encoder.decode_move(int(idx))
                moves.append(move)

        return moves
