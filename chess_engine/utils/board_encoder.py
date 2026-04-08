"""
BOARD REPRESENTATION & DATA ENCODING

This module handles conversion between chess.Board objects and tensor representations.
The tensor format is designed for CNN input following AlphaZero's approach.

Tensor Structure: (22, 8, 8)
- Channels 0-11: Piece positions (6 for own pieces, 6 for opponent)
- Channel 12: Color to move
- Channel 13: Total move count
- Channels 14-17: Castling rights
- Channel 18: En passant square
- Channel 19: Halfmove clock (50-move rule)
- Channel 20: 2-fold repetition indicator (binary)
- Channel 21: 3-fold repetition indicator (binary)

Row/Rank mapping: row 0 = rank 1, row 7 = rank 8
Col/File mapping: col 0 = file 'a', col 7 = file 'h'
"""

import chess
import torch
from typing import List


class BoardEncoder:
    """
    Encodes chess boards into tensor representations suitable for neural network input.
    """

    # Piece to channel mapping
    PIECE_TO_CHANNEL = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    def __init__(self):
        """Initialize the board encoder"""
        self.num_channels = 22
        self.board_size = 8

    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """
        Convert a chess.Board to a tensor representation.

        Args:
            board: chess.Board object to encode

        Returns:
            torch.Tensor of shape (22, 8, 8)
        """
        # Allocate directly as a torch tensor (avoids numpy alloc + from_numpy copy)
        tensor = torch.zeros(
            (self.num_channels, self.board_size, self.board_size), dtype=torch.float32
        )

        # Encode piece positions (channels 0-11)
        # piece_map() returns only occupied squares (~16-32 entries vs 64 for chess.SQUARES)
        for square, piece in board.piece_map().items():
            piece_channel = self.PIECE_TO_CHANNEL[piece.piece_type]
            if piece.color != board.turn:
                piece_channel += 6
            tensor[piece_channel, square // 8, square % 8] = 1.0

        # Channel 12: Color to move (1 for white, 0 for black)
        tensor[12, :, :] = float(board.turn)

        # Channel 13: Total move count (normalized)
        tensor[13, :, :] = board.fullmove_number / 200.0

        # Channels 14-17: Castling rights
        tensor[14, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
        tensor[15, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
        tensor[16, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
        tensor[17, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))

        # Channel 18: En passant square
        if board.ep_square is not None:
            tensor[18, board.ep_square // 8, board.ep_square % 8] = 1.0

        # Channel 19: Halfmove clock (normalized by 100 for 50-move rule)
        tensor[19, :, :] = board.halfmove_clock / 100.0

        # Channel 20: 2-fold repetition indicator (binary)
        # True when the current position has occurred at least twice in the game.
        # Requires a populated move stack; returns False for positions loaded from FEN only.
        is_rep2 = board.is_repetition(2)
        tensor[20, :, :] = float(is_rep2)

        # Channel 21: 3-fold repetition indicator (binary)
        # True when the current position has occurred at least three times (draw claimable).
        tensor[21, :, :] = float(is_rep2 and board.is_repetition(3))

        return tensor

    def batch_boards_to_tensor(self, boards: List[chess.Board]) -> torch.Tensor:
        """
        Convert a batch of chess boards to tensor representation.

        Args:
            boards: List of chess.Board objects

        Returns:
            torch.Tensor of shape (batch_size, 22, 8, 8)
        """
        tensors = [self.board_to_tensor(board) for board in boards]
        return torch.stack(tensors)

    def validate_tensor(self, tensor: torch.Tensor) -> bool:
        """
        Validate tensor shape and value ranges.

        Args:
            tensor: Tensor to validate

        Returns:
            True if valid, False otherwise
        """
        if tensor.shape != (self.num_channels, self.board_size, self.board_size):
            return False

        # Channels 0-11 should be binary (piece positions)
        if not torch.all((tensor[:12] == 0) | (tensor[:12] == 1)):
            return False

        # Channel 12 should be binary (color)
        if not torch.all((tensor[12] == 0) | (tensor[12] == 1)):
            return False

        # Channels 14-17 should be binary (castling)
        if not torch.all((tensor[14:18] == 0) | (tensor[14:18] == 1)):
            return False

        return True
