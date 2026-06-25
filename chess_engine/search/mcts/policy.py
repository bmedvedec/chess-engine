"""
MCTS - Policy Utilities

Provides Dirichlet noise injection for training exploration
and move selection based on visit counts with temperature scheduling.
"""

import numpy as np
import chess
from typing import Dict

from chess_engine.search.mcts.node import MCTSNode


def add_dirichlet_noise(
    policy_probs: Dict[chess.Move, float],
    epsilon: float = 0.25,
    alpha: float = 0.3,
) -> Dict[chess.Move, float]:
    """Mix policy with Dirichlet noise for self-play exploration."""
    moves = list(policy_probs.keys())
    probs = np.array([policy_probs[m] for m in moves])
    noise = np.random.dirichlet([alpha] * len(moves))
    noisy_probs = (1 - epsilon) * probs + epsilon * noise
    return {move: prob for move, prob in zip(moves, noisy_probs)}


def select_move(
    root: MCTSNode,
    move_number: int = 0,
    temperature: float = 1.0,
    temperature_threshold: int = 30,
    late_game_temperature: float = 0.1,
) -> chess.Move:
    """Select move from visit counts with temperature scheduling."""
    visit_counts = {move: child.visit_count for move, child in root.children.items()}

    if not visit_counts:
        legal_moves = list(root.board.legal_moves)
        return legal_moves[np.random.randint(len(legal_moves))]

    if move_number < temperature_threshold:
        temperature = temperature
    else:
        temperature = late_game_temperature

    if temperature < 0.01:
        best_move = max(visit_counts.keys(), key=lambda m: visit_counts[m])
    else:
        moves = list(visit_counts.keys())
        counts = np.array([visit_counts[m] for m in moves])
        counts = counts ** (1.0 / temperature)
        probs = counts / counts.sum()
        idx = np.random.choice(len(moves), p=probs)
        best_move = moves[idx]

    return best_move
