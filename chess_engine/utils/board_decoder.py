import chess
import numpy as np
import torch
from typing import Union


def tensor_to_board(self, tensor: Union[torch.Tensor, np.ndarray]) -> chess.Board:
    """
    Reconstruct a chess.Board from a tensor representation.
    Useful for debugging and visualization.

    Args:
        tensor: torch.Tensor of shape (20, 8, 8)

    Returns:
        chess.Board object
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()

    board = chess.Board(fen=None)
    board.clear()

    # Reconstruct pieces (channels 0-11)
    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]

    for square in range(64):
        row = square // 8
        col = square % 8

        # Check own pieces (channels 0-5)
        for piece_idx, piece_type in enumerate(piece_types):
            if tensor[piece_idx, row, col] > 0.5:
                # Determine color based on channel 12
                color = chess.WHITE if tensor[12, row, col] > 0.5 else chess.BLACK
                board.set_piece_at(square, chess.Piece(piece_type, color))
                break

        # Check opponent pieces (channels 6-11)
        for piece_idx, piece_type in enumerate(piece_types):
            if tensor[piece_idx + 6, row, col] > 0.5:
                # Opponent color
                color = chess.BLACK if tensor[12, row, col] > 0.5 else chess.WHITE
                board.set_piece_at(square, chess.Piece(piece_type, color))
                break

    # Set turn
    board.turn = chess.WHITE if tensor[12, 0, 0] > 0.5 else chess.BLACK

    # Set castling rights
    board.castling_rights = 0
    if tensor[14, 0, 0] > 0.5:
        board.castling_rights |= chess.BB_H1
    if tensor[15, 0, 0] > 0.5:
        board.castling_rights |= chess.BB_A1
    if tensor[16, 0, 0] > 0.5:
        board.castling_rights |= chess.BB_H8
    if tensor[17, 0, 0] > 0.5:
        board.castling_rights |= chess.BB_A8

    # Set en passant square
    for square in range(64):
        row = square // 8
        col = square % 8
        if tensor[18, row, col] > 0.5:
            board.ep_square = square
            break

    # Set halfmove clock
    board.halfmove_clock = int(tensor[19, 0, 0] * 100)

    # Set fullmove number
    board.fullmove_number = int(tensor[13, 0, 0] * 100)

    return board
