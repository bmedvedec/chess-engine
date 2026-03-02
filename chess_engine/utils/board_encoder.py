"""
BOARD REPRESENTATION & DATA ENCODING

This module handles conversion between chess.Board objects and tensor representations.
The tensor format is designed for CNN input following AlphaZero's approach.

Tensor Structure: (20, 8, 8)
- Channels 0-11: Piece positions (6 for own pieces, 6 for opponent)
- Channel 12: Color to move
- Channel 13: Total move count
- Channels 14-17: Castling rights
- Channel 18: En passant square
- Channel 19: Halfmove clock (50-move rule)

Row/Rank mapping: row 0 = rank 1, row 7 = rank 8
Col/File mapping: col 0 = file 'a', col 7 = file 'h'
"""

import chess
import numpy as np
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
        self.num_channels = 20
        self.board_size = 8

    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        """
        Convert a chess.Board to a tensor representation.

        Args:
            board: chess.Board object to encode

        Returns:
            torch.Tensor of shape (20, 8, 8)
        """
        # Initialize empty tensor
        tensor = np.zeros(
            (self.num_channels, self.board_size, self.board_size), dtype=np.float32
        )
        # This creates a 3D array: 20 layers, each 8x8, with all values starting at 0.0

        # Encode piece positions (channels 0-11)
        for square in chess.SQUARES:  # Loop through all 64 squares (A1: 0..., H8: 63)
            piece = board.piece_at(square)
            if piece is not None:
                # Determine if piece belongs to current player or opponent
                is_own_piece = piece.color == board.turn

                # Get piece type channel
                piece_channel = self.PIECE_TO_CHANNEL[piece.piece_type]

                # Add 6 if opponent's piece
                if not is_own_piece:
                    piece_channel += 6

                # Convert square to row, col (rank, file)
                row = square // 8
                col = square % 8

                # Mark this square as occupied in the appropriate channel
                tensor[piece_channel, row, col] = 1.0

        # Channel 12: Color to move (1 for white, 0 for black)
        tensor[12, :, :] = float(board.turn)

        # Channel 13: Total move count (normalized)
        tensor[13, :, :] = board.fullmove_number / 100.0

        # Channels 14-17: Castling rights
        tensor[14, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
        tensor[15, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
        tensor[16, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
        tensor[17, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))

        # Channel 18: En passant square
        if board.ep_square is not None:
            row = board.ep_square // 8
            col = board.ep_square % 8
            tensor[18, row, col] = 1.0

        # Channel 19: Halfmove clock (normalized by 100 for 50-move rule)
        tensor[19, :, :] = board.halfmove_clock / 100.0

        return torch.from_numpy(tensor)

    def batch_boards_to_tensor(self, boards: List[chess.Board]) -> torch.Tensor:
        """
        Convert a batch of chess boards to tensor representation.

        Args:
            boards: List of chess.Board objects

        Returns:
            torch.Tensor of shape (batch_size, 20, 8, 8)
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
