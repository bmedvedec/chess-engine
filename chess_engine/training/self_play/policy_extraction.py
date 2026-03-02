"""
SELF-PLAY - Policy Extraction

Extracts complete policy distribution over ALL legal moves from MCTS statistics.
"""

import chess
from typing import Dict, Optional


def extract_policy(board: chess.Board, stats: Optional[Dict]) -> Dict[str, float]:
    """
    Extract complete policy distribution over ALL legal moves.

    Args:
        board: Current board state
        stats: MCTS statistics

    Returns:
        Dictionary mapping UCI strings to probabilities
    """
    policy = {}
    legal_moves = list(board.legal_moves)

    if stats and "visit_counts" in stats:
        visit_counts = stats["visit_counts"]
        total_visits = sum(visit_counts.values())

        if total_visits > 0:
            for move in legal_moves:
                move_uci = move.uci()
                visits = visit_counts.get(move_uci, 0)
                policy[move_uci] = visits / total_visits
        else:
            uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0.0
            policy = {move.uci(): uniform_prob for move in legal_moves}

    elif stats and "top_moves" in stats:
        top_moves_dict = {
            move_stat["move"]: move_stat["visit_pct"]
            for move_stat in stats["top_moves"]
        }

        for move in legal_moves:
            move_uci = move.uci()
            policy[move_uci] = top_moves_dict.get(move_uci, 0.0)

        total_prob = sum(policy.values())
        if total_prob > 0 and abs(total_prob - 1.0) > 1e-6:
            policy = {k: v / total_prob for k, v in policy.items()}

    else:
        uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0.0
        policy = {move.uci(): uniform_prob for move in legal_moves}

    return policy
