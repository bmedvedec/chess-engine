"""
CHAPTER 2: MOVE REPRESENTATION & ENCODING
Complete Track Implementation

This module handles conversion between chess.Move objects and numerical indices.
We use a simplified encoding scheme: from_square * 64 + to_square = 4096 possible moves.

For a more sophisticated approach (AlphaZero-style with 73 move types per square),
this can be extended later.
"""

import chess
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union


class MoveEncoder:
    """
    Encodes chess moves into numerical indices for neural network processing.

    Simple encoding scheme:
    - move_index = from_square * 64 + to_square
    - Total possible moves: 64 * 64 = 4096

    Note: This includes illegal moves (e.g., a1 to a1), but simplifies implementation.
    The policy network will learn to assign near-zero probability to illegal moves.
    """

    def __init__(self):
        """Initialize the move encoder"""
        self.num_moves = 4096  # 64 from_squares * 64 to_squares

        # Pre-compute move to index mapping
        self.move_to_index = {}
        self.index_to_move_template = {}

        for from_square in range(64):
            for to_square in range(64):
                index = from_square * 64 + to_square
                self.move_to_index[(from_square, to_square)] = index
                self.index_to_move_template[index] = (from_square, to_square)

    def encode_move(self, move: chess.Move) -> int:
        """
        Encode a chess move to an integer index.

        Args:
            move: chess.Move object

        Returns:
            Integer index in range [0, 4095]
        """
        return self.move_to_index[(move.from_square, move.to_square)]

    def decode_move(
        self, index: int, board: Optional[chess.Board] = None
    ) -> chess.Move:
        """
        Decode an integer index to a chess move.

        Args:
            index: Integer index in range [0, 4095]
            board: Optional chess.Board to validate the move

        Returns:
            chess.Move object
        """
        from_square, to_square = self.index_to_move_template[index]

        # Create basic move
        move = chess.Move(from_square, to_square)

        # If board provided, check for promotions
        if board is not None:
            piece = board.piece_at(from_square)
            if piece and piece.piece_type == chess.PAWN:
                # Check if pawn reaches back rank
                to_rank = chess.square_rank(to_square)
                if (piece.color == chess.WHITE and to_rank == 7) or (
                    piece.color == chess.BLACK and to_rank == 0
                ):
                    # Default to queen promotion
                    move = chess.Move(from_square, to_square, promotion=chess.QUEEN)

                    # TODO: implement other promotions

        return move

    def encode_legal_moves(self, board: chess.Board) -> List[Tuple[chess.Move, int]]:
        """
        Encode all legal moves for a given position.

        Args:
            board: chess.Board object

        Returns:
            List of (move, index) tuples
        """
        encoded_moves = []
        for move in board.legal_moves:
            index = self.encode_move(move)
            encoded_moves.append((move, index))
        return encoded_moves

    def create_legal_moves_mask(self, board: chess.Board) -> torch.Tensor:
        """
        Create a binary mask for legal moves.

        Args:
            board: chess.Board object

        Returns:
            torch.Tensor of shape (4096,) with 1s for legal moves, 0s otherwise
        """
        mask = torch.zeros(self.num_moves)
        for move in board.legal_moves:
            index = self.encode_move(move)
            mask[index] = 1.0
        return mask

    def policy_to_move_probs(
        self, policy_logits: torch.Tensor, board: chess.Board, temperature: float = 1.0
    ) -> Dict[chess.Move, float]:
        """
        Convert policy network output to move probabilities.
        Only considers legal moves.

        Args:
            policy_logits: torch.Tensor of shape (4096,) - raw network output
            board: chess.Board object for legal move filtering
            temperature: Temperature for softmax (higher = more random)

        Returns:
            Dictionary mapping chess.Move to probability
        """
        # Apply temperature
        if temperature != 1.0:
            policy_logits = policy_logits / temperature

        # Mask illegal moves (set to very negative value)
        legal_mask = self.create_legal_moves_mask(board)
        masked_logits = policy_logits.clone()
        masked_logits[legal_mask == 0] = -1e10

        # Apply softmax
        probs = torch.softmax(masked_logits, dim=0)

        # Convert to dictionary
        move_probs = {}
        for move in board.legal_moves:
            index = self.encode_move(move)
            move_probs[move] = probs[index].item()

        return move_probs

    def sample_move(
        self, policy_logits: torch.Tensor, board: chess.Board, temperature: float = 1.0
    ) -> chess.Move:
        """
        Sample a move from the policy distribution.

        Args:
            policy_logits: torch.Tensor of shape (4096,)
            board: chess.Board object
            temperature: Temperature for sampling

        Returns:
            Sampled chess.Move
        """
        move_probs = self.policy_to_move_probs(policy_logits, board, temperature)

        moves = list(move_probs.keys())
        probs = list(move_probs.values())

        # Normalize (in case of floating point errors)
        probs = np.array(probs)
        probs = probs / probs.sum()

        # Sample move
        indices = np.arange(len(moves))
        chosen_idx = np.random.choice(indices, p=probs)
        chosen_move = moves[chosen_idx]

        return chosen_move

    def get_best_move(
        self, policy_logits: torch.Tensor, board: chess.Board
    ) -> Tuple[chess.Move, float]:
        """
        Get the move with highest probability.

        Args:
            policy_logits: torch.Tensor of shape (4096,)
            board: chess.Board object

        Returns:
            Tuple of (best_move, probability)
        """
        move_probs = self.policy_to_move_probs(policy_logits, board, temperature=1.0)

        best_move = max(move_probs.items(), key=lambda x: x[1])
        return best_move


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


def create_policy_target(
    board: chess.Board, target_move: chess.Move, smooth: float = 0.0
) -> torch.Tensor:
    """
    Create a target policy vector for supervised learning.

    Args:
        board: chess.Board object
        target_move: The correct move to play
        smooth: Label smoothing factor (0 = no smoothing)

    Returns:
        torch.Tensor of shape (4096,) with target probabilities
    """
    encoder = MoveEncoder()
    target = torch.zeros(encoder.num_moves)

    # Get legal moves for smoothing
    legal_moves = list(board.legal_moves)
    num_legal = len(legal_moves)

    if smooth > 0 and num_legal > 1:
        # Distribute smooth probability among all legal moves
        smooth_prob = smooth / num_legal
        for move in legal_moves:
            index = encoder.encode_move(move)
            target[index] = smooth_prob

        # Add remaining probability to target move
        target_index = encoder.encode_move(target_move)
        target[target_index] += 1.0 - smooth
    else:
        # One-hot encoding
        target_index = encoder.encode_move(target_move)
        target[target_index] = 1.0

    return target
