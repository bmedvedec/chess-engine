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
from typing import Optional, List, Tuple, Union


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

    def tensor_to_board(self, tensor: Union[torch.Tensor, np.ndarray]) -> chess.Board:
        """
        Reconstruct a chess.Board from a tensor representation.
        Useful for debugging and visualization.

        Args:
            tensor: torch.Tensor of shape (20, 8, 8)

        Returns:
            chess.Board object
        """
        # Convert to numpy if necessary
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()

        board = chess.Board(fen=None)  # Empty board
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

    def visualize_tensor(
        self, tensor: Union[torch.Tensor, np.ndarray], channel: Optional[int] = None
    ):
        """
        Print a visual representation of the tensor.

        Args:
            tensor: torch.Tensor of shape (20, 8, 8)
            channel: If specified, show only this channel. Otherwise show summary.
        """
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.cpu().numpy()

        if channel is not None:
            print(f"\nChannel {channel}:")
            print("-" * 40)
            for row in range(7, -1, -1):  # Print from rank 8 to rank 1
                print(f"Rank {row + 1}: ", end="")
                for col in range(8):
                    val = tensor[channel, row, col]
                    print(f"{val:5.2f} ", end="")
                print()
            print("       ", end="")
            for col in range(8):
                print(f"  {chr(ord('a') + col)}   ", end="")
            print()
        else:
            # Show summary of all channels
            print("\nTensor Summary:")
            print("=" * 60)
            channel_names = [
                "Own Pawns",
                "Own Knights",
                "Own Bishops",
                "Own Rooks",
                "Own Queens",
                "Own King",
                "Opp Pawns",
                "Opp Knights",
                "Opp Bishops",
                "Opp Rooks",
                "Opp Queens",
                "Opp King",
                "Color to Move",
                "Move Count",
                "White K-Castle",
                "White Q-Castle",
                "Black K-Castle",
                "Black Q-Castle",
                "En Passant",
                "Halfmove Clock",
            ]

            for ch, name in enumerate(channel_names):
                non_zero = np.sum(tensor[ch] > 0)
                max_val = np.max(tensor[ch])
                print(
                    f"Ch {ch:2d} [{name:16s}]: {non_zero:2.0f} active squares, max={max_val:.2f}"
                )

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


class DataAugmentation:
    """
    Data augmentation techniques for chess positions.
    Includes horizontal flipping and color swapping.
    """

    @staticmethod
    def horizontal_flip(board: chess.Board) -> chess.Board:
        """
        Flip board horizontally (mirror across the vertical center line).

        Args:
            board: Original chess.Board

        Returns:
            Flipped chess.Board
        """
        flipped = chess.Board(fen=None)
        flipped.clear()

        # Map each piece to its horizontally flipped position
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Calculate flipped file
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                flipped_file = 7 - file
                flipped_square = chess.square(flipped_file, rank)

                flipped.set_piece_at(flipped_square, piece)

        # Copy game state
        flipped.turn = board.turn
        flipped.castling_rights = board.castling_rights

        # Flip en passant square if exists
        if board.ep_square is not None:
            ep_rank = chess.square_rank(board.ep_square)
            ep_file = chess.square_file(board.ep_square)
            flipped_ep_file = 7 - ep_file
            flipped.ep_square = chess.square(flipped_ep_file, ep_rank)

        flipped.halfmove_clock = board.halfmove_clock
        flipped.fullmove_number = board.fullmove_number

        return flipped

    @staticmethod
    def flip_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """
        Flip tensor representation horizontally.
        More efficient than converting to board and back.

        Args:
            tensor: torch.Tensor of shape (20, 8, 8) or (batch, 20, 8, 8)

        Returns:
            Horizontally flipped tensor
        """
        # Flip along the width dimension (last dimension)
        return torch.flip(tensor, dims=[-1])

    @staticmethod
    def flip_move(move: chess.Move) -> chess.Move:
        """
        Flip move coordinates horizontally to match flipped board.

        Args:
            move: Original move

        Returns:
            Flipped move with horizontally mirrored coordinates
        """
        from_rank = chess.square_rank(move.from_square)
        from_file = chess.square_file(move.from_square)
        to_rank = chess.square_rank(move.to_square)
        to_file = chess.square_file(move.to_square)

        # Flip files (a↔h, b↔g, c↔f, d↔e)
        flipped_from = chess.square(7 - from_file, from_rank)
        flipped_to = chess.square(7 - to_file, to_rank)

        return chess.Move(flipped_from, flipped_to, move.promotion)

    @staticmethod
    def flip_move_index(move_index: int) -> int:
        """
        Flip move index to match horizontally flipped board.

        Assumes move encoding: from_square * 64 + to_square (4096 possible moves)

        Args:
            move_index: Original move index (0-4095)

        Returns:
            Flipped move index with horizontally mirrored coordinates
        """
        from_square = move_index // 64
        to_square = move_index % 64

        # Flip files for both squares
        from_rank = from_square // 8
        from_file = from_square % 8
        to_rank = to_square // 8
        to_file = to_square % 8

        flipped_from = from_rank * 8 + (7 - from_file)
        flipped_to = to_rank * 8 + (7 - to_file)

        return flipped_from * 64 + flipped_to

    @staticmethod
    def color_swap(board: chess.Board) -> chess.Board:
        """
        Swap colors (white ↔ black) and flip board vertically.
        This maintains the perspective from the current player's view.

        Args:
            board: Original chess.Board

        Returns:
            Color-swapped chess.Board
        """
        swapped = chess.Board(fen=None)
        swapped.clear()

        # Swap pieces and flip vertically
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                # Flip square vertically (mirror rank)
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                flipped_rank = 7 - rank
                flipped_square = chess.square(file, flipped_rank)

                # Swap color
                new_color = not piece.color
                swapped.set_piece_at(
                    flipped_square, chess.Piece(piece.piece_type, new_color)
                )

        # Swap turn
        swapped.turn = not board.turn

        # Swap castling rights
        swapped.castling_rights = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            swapped.castling_rights |= chess.BB_H8
        if board.has_queenside_castling_rights(chess.WHITE):
            swapped.castling_rights |= chess.BB_A8
        if board.has_kingside_castling_rights(chess.BLACK):
            swapped.castling_rights |= chess.BB_H1
        if board.has_queenside_castling_rights(chess.BLACK):
            swapped.castling_rights |= chess.BB_A1

        # Flip en passant square vertically if exists
        if board.ep_square is not None:
            ep_rank = chess.square_rank(board.ep_square)
            ep_file = chess.square_file(board.ep_square)
            flipped_ep_rank = 7 - ep_rank
            swapped.ep_square = chess.square(ep_file, flipped_ep_rank)

        swapped.halfmove_clock = board.halfmove_clock
        swapped.fullmove_number = board.fullmove_number

        return swapped

    @staticmethod
    def augment_batch(
        boards: List[chess.Board],
        horizontal_flip: bool = True,
        color_swap: bool = False,
    ) -> List[chess.Board]:
        """
        Apply augmentation to a batch of boards.

        Args:
            boards: List of chess.Board objects
            horizontal_flip: Whether to include horizontal flips
            color_swap: Whether to include color swaps

        Returns:
            Augmented list of boards
        """
        augmented = list(boards)  # Include originals

        if horizontal_flip:
            augmented.extend([DataAugmentation.horizontal_flip(b) for b in boards])

        if color_swap:
            augmented.extend([DataAugmentation.color_swap(b) for b in boards])
            if horizontal_flip:
                augmented.extend(
                    [
                        DataAugmentation.horizontal_flip(DataAugmentation.color_swap(b))
                        for b in boards
                    ]
                )

        return augmented
