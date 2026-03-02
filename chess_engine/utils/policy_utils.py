import chess
import torch

from chess_engine.utils.move_encoder import MoveEncoder


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
