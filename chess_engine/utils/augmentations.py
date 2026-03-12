import chess
import torch
from typing import List

from chess_engine.utils.move_encoder import MoveEncoder


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

        Uses the AlphaZero-style 4672 move encoding (from_square * 73 + move_type).
        Decodes the index to a move, applies horizontal flip, then re-encodes.

        Args:
            move_index: Original move index (0-4671)

        Returns:
            Flipped move index with horizontally mirrored coordinates
        """
        encoder = MoveEncoder()
        move = encoder.decode_move(move_index)
        flipped = DataAugmentation.flip_move(move)
        return encoder.encode_move(flipped)

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

    # Backward-compatible aliases used by chess_dataset_old.py
    flip_tensor_horizontal = flip_tensor
    flip_move_horizontal = flip_move


# Backward-compatible alias: chess_dataset_old.py imports BoardAugmentations
BoardAugmentations = DataAugmentation
