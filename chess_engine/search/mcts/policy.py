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
    """
    Add Dirichlet noise to policy for exploration (used in self-play training).

    Args:
        policy_probs: Original policy probabilities
        epsilon: Noise weight (0.0 = off, 0.25 typical)
        alpha: Dirichlet alpha parameter (default: 0.3)

    Returns:
        Policy with noise added
    """
    moves = list(policy_probs.keys())
    probs = np.array([policy_probs[m] for m in moves])

    # Generate Dirichlet noise
    noise = np.random.dirichlet([alpha] * len(moves))

    # Mix policy with noise
    noisy_probs = (1 - epsilon) * probs + epsilon * noise

    # Return as dictionary
    return {move: prob for move, prob in zip(moves, noisy_probs)}


def select_move(
    root: MCTSNode,
    move_number: int = 0,
) -> chess.Move:
    """
    Select move based on visit counts with temperature scheduling.

    Args:
        root: Root node after search
        move_number: Current move number (for temperature scheduling)

    Returns:
        Best move
    """
    # Get visit counts
    visit_counts = {move: child.visit_count for move, child in root.children.items()}

    if not visit_counts:
        # Fallback to random legal move
        legal_moves = list(root.board.legal_moves)
        return legal_moves[np.random.randint(len(legal_moves))]

    # Temperature scheduling: high early game, low late game
    if move_number < 30:
        temperature = 1.0  # More exploration
    else:
        temperature = 0.1  # More exploitation

    if temperature < 0.01:
        # Deterministic - select highest visit count
        best_move = max(visit_counts.keys(), key=lambda m: visit_counts[m])
    else:
        # Stochastic - sample proportional to visit_count^(1/temp)
        moves = list(visit_counts.keys())
        counts = np.array([visit_counts[m] for m in moves])

        # Apply temperature
        counts = counts ** (1.0 / temperature)
        probs = counts / counts.sum()

        # Sample move
        idx = np.random.choice(len(moves), p=probs)
        best_move = moves[idx]

    return best_move
