"""
MOVE REPRESENTATION & ENCODING
AlphaZero-style 4,672 move encoding

This module handles conversion between chess.Move objects and numerical indices.
We use the AlphaZero encoding scheme: from_square * 73 + move_type = 4,672 possible moves.

Move types (73 total per square):
  0-55:  Queen-style moves — 8 directions × 7 distances
  56-63: Knight moves — 8 L-shaped offsets
  64-72: Underpromotions — 3 directions × 3 pieces (knight, bishop, rook)
         Queen promotions are encoded via the matching queen-style move type.

Queen direction order (dir_idx → rank_delta, file_delta):
  0: N  (+1,  0)   4: S  (-1,  0)
  1: NE (+1, +1)   5: SW (-1, -1)
  2: E  ( 0, +1)   6: W  ( 0, -1)
  3: SE (-1, +1)   7: NW (+1, -1)

Knight move order (56 + k_idx → rank_delta, file_delta):
  0:(+2,+1)  1:(+2,-1)  2:(+1,+2)  3:(+1,-2)
  4:(-1,+2)  5:(-1,-2)  6:(-2,+1)  7:(-2,-1)

Underpromotion order (64 + dir_idx*3 + piece_idx):
  dir_idx: 0=left(file-1), 1=straight(file±0), 2=right(file+1)
  piece_idx: 0=knight, 1=bishop, 2=rook
"""

import chess
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
NUM_MOVES = 4672  # 64 from_squares × 73 move types
NUM_MOVE_TYPES = 73
PAD_INDEX = 0  # Padding index for move sequences

# Queen-style directions: (rank_delta, file_delta)
QUEEN_DIRECTIONS: List[Tuple[int, int]] = [
    (+1, 0),  # 0: N
    (+1, +1),  # 1: NE
    (0, +1),  # 2: E
    (-1, +1),  # 3: SE
    (-1, 0),  # 4: S
    (-1, -1),  # 5: SW
    (0, -1),  # 6: W
    (+1, -1),  # 7: NW
]

# Knight move offsets: (rank_delta, file_delta)
KNIGHT_DELTAS: List[Tuple[int, int]] = [
    (+2, +1),
    (+2, -1),
    (+1, +2),
    (+1, -2),
    (-1, +2),
    (-1, -2),
    (-2, +1),
    (-2, -1),
]

# Underpromotion pieces in order (piece_idx → chess piece type)
UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

# Underpromotion file deltas (dir_idx → file offset)
UNDERPROMOTION_FILE_DELTAS = [-1, 0, +1]  # left, straight, right


class MoveEncoder:
    """
    Encodes chess moves into numerical indices for neural network processing.

    AlphaZero encoding scheme:
    - move_index = from_square * 73 + move_type
    - Total possible moves: 64 * 73 = 4,672

    Move types (73 per square):
      0-55:  Queen-style (8 directions × 7 distances)
      56-63: Knight moves (8 L-shaped offsets)
      64-72: Underpromotions (3 directions × 3 pieces: knight/bishop/rook)

    Queen promotions share the same index as the corresponding 1-square queen-style
    move toward the back rank; decode_move resolves the promotion via the board.
    """

    def __init__(self):
        """Initialize the move encoder with precomputed lookup tables."""
        self.num_moves = NUM_MOVES

        # (from_sq, to_sq, promotion_or_None) → index
        self._encode_table: Dict[Tuple[int, int, Optional[int]], int] = {}
        # index → chess.Move  (queen promotions stored without promotion field;
        # decode_move adds it when a board is supplied)
        self._decode_table: Dict[int, chess.Move] = {}

        self._build_tables()

    # ------------------------------------------------------------------
    # Table construction
    # ------------------------------------------------------------------

    def _build_tables(self) -> None:
        """Precompute encode/decode lookup tables for all valid move indices."""
        for from_sq in range(64):
            from_rank = from_sq // 8
            from_file = from_sq % 8

            # --------------------------------------------------------------
            # Queen-style moves (move types 0–55)
            # --------------------------------------------------------------
            for dir_idx, (dr, df) in enumerate(QUEEN_DIRECTIONS):
                for dist in range(1, 8):
                    to_rank = from_rank + dr * dist
                    to_file = from_file + df * dist
                    if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
                        # Off-board; further distances in this direction also invalid
                        break
                    to_sq = to_rank * 8 + to_file
                    move_type = dir_idx * 7 + (dist - 1)
                    index = from_sq * NUM_MOVE_TYPES + move_type

                    # Decode: store basic move (no promotion); decode_move adds
                    # queen promotion contextually when a board is supplied.
                    self._decode_table[index] = chess.Move(from_sq, to_sq)

                    # Encode: non-promotion use (any piece including pawns not promoting)
                    self._encode_table[(from_sq, to_sq, None)] = index

                    # Encode: queen-promotion variant for pawn promotion squares
                    is_white_promo = from_rank == 6 and to_rank == 7
                    is_black_promo = from_rank == 1 and to_rank == 0
                    if is_white_promo or is_black_promo:
                        self._encode_table[(from_sq, to_sq, chess.QUEEN)] = index

            # --------------------------------------------------------------
            # Knight moves (move types 56–63)
            # --------------------------------------------------------------
            for k_idx, (dr, df) in enumerate(KNIGHT_DELTAS):
                to_rank = from_rank + dr
                to_file = from_file + df
                if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
                    continue
                to_sq = to_rank * 8 + to_file
                move_type = 56 + k_idx
                index = from_sq * NUM_MOVE_TYPES + move_type
                self._decode_table[index] = chess.Move(from_sq, to_sq)
                self._encode_table[(from_sq, to_sq, None)] = index

            # --------------------------------------------------------------
            # Underpromotions (move types 64–72)
            # Only valid from rank 6 (white) or rank 1 (black)
            # --------------------------------------------------------------
            for dir_idx, df in enumerate(UNDERPROMOTION_FILE_DELTAS):
                for piece_idx, piece in enumerate(UNDERPROMOTION_PIECES):
                    move_type = 64 + dir_idx * 3 + piece_idx
                    index = from_sq * NUM_MOVE_TYPES + move_type

                    if from_rank == 6:  # White pawn: rank 6 → 7
                        to_file = from_file + df
                        if 0 <= to_file <= 7:
                            to_sq = 7 * 8 + to_file
                            promo_move = chess.Move(from_sq, to_sq, promotion=piece)
                            self._decode_table[index] = promo_move
                            self._encode_table[(from_sq, to_sq, piece)] = index

                    elif from_rank == 1:  # Black pawn: rank 1 → 0
                        to_file = from_file + df
                        if 0 <= to_file <= 7:
                            to_sq = 0 * 8 + to_file
                            promo_move = chess.Move(from_sq, to_sq, promotion=piece)
                            self._decode_table[index] = promo_move
                            self._encode_table[(from_sq, to_sq, piece)] = index

    # ------------------------------------------------------------------
    # Core encode / decode
    # ------------------------------------------------------------------

    def encode_move(self, move: chess.Move) -> int:
        """
        Encode a chess move to an integer index.

        Args:
            move: chess.Move object

        Returns:
            Integer index in range [0, 4671]
        """
        return self._encode_table[(move.from_square, move.to_square, move.promotion)]

    def decode_move(
        self, index: int, board: Optional[chess.Board] = None
    ) -> chess.Move:
        """
        Decode an integer index to a chess move.

        Queen promotions are resolved when a board is provided; without a board
        the returned move has no promotion field (correct for all non-pawn moves).
        Underpromotions always carry the correct promotion piece regardless of board.

        Args:
            index: Integer index in range [0, 4671]
            board: Optional chess.Board to resolve queen promotions

        Returns:
            chess.Move object
        """
        move = self._decode_table[index]

        # Add queen promotion when the moving piece is a pawn reaching the back rank
        if board is not None and move.promotion is None:
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN:
                to_rank = chess.square_rank(move.to_square)
                if (piece.color == chess.WHITE and to_rank == 7) or (
                    piece.color == chess.BLACK and to_rank == 0
                ):
                    move = chess.Move(
                        move.from_square, move.to_square, promotion=chess.QUEEN
                    )

        return move

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def encode_legal_moves(self, board: chess.Board) -> List[Tuple[chess.Move, int]]:
        """
        Encode all legal moves for a given position.

        Args:
            board: chess.Board object

        Returns:
            List of (move, index) tuples
        """
        return [(move, self.encode_move(move)) for move in board.legal_moves]

    def create_legal_moves_mask(self, board: chess.Board) -> torch.Tensor:
        """
        Create a binary mask for legal moves.

        Args:
            board: chess.Board object

        Returns:
            torch.Tensor of shape (4672,) with 1s for legal moves, 0s otherwise
        """
        mask = torch.zeros(self.num_moves)
        for move in board.legal_moves:
            mask[self.encode_move(move)] = 1.0
        return mask

    def policy_to_move_probs(
        self,
        policy_logits: torch.Tensor,
        board: chess.Board,
        temperature: float = 1.0,
    ) -> Dict[chess.Move, float]:
        """
        Convert policy network output to move probabilities.
        Only legal moves receive non-negligible probability.

        Args:
            policy_logits: torch.Tensor of shape (4672,) — raw network output
            board: chess.Board object for legal move filtering
            temperature: Temperature for softmax (higher = more random)

        Returns:
            Dictionary mapping chess.Move to probability
        """
        if temperature != 1.0:
            policy_logits = policy_logits / temperature

        legal_mask = self.create_legal_moves_mask(board)
        masked_logits = policy_logits.clone()
        masked_logits[legal_mask == 0] = -1e10

        probs = torch.softmax(masked_logits, dim=0)

        return {
            move: probs[self.encode_move(move)].item() for move in board.legal_moves
        }

    def sample_move(
        self,
        policy_logits: torch.Tensor,
        board: chess.Board,
        temperature: float = 1.0,
    ) -> chess.Move:
        """
        Sample a move from the policy distribution.

        Args:
            policy_logits: torch.Tensor of shape (4672,)
            board: chess.Board object
            temperature: Temperature for sampling

        Returns:
            Sampled chess.Move
        """
        move_probs = self.policy_to_move_probs(policy_logits, board, temperature)
        moves = list(move_probs.keys())
        probs = np.array(list(move_probs.values()))
        probs = probs / probs.sum()
        return moves[np.random.choice(len(moves), p=probs)]

    def get_best_move(
        self, policy_logits: torch.Tensor, board: chess.Board
    ) -> Tuple[chess.Move, float]:
        """
        Get the move with highest probability.

        Args:
            policy_logits: torch.Tensor of shape (4672,)
            board: chess.Board object

        Returns:
            Tuple of (best_move, probability)
        """
        move_probs = self.policy_to_move_probs(policy_logits, board, temperature=1.0)
        return max(move_probs.items(), key=lambda x: x[1])
