"""
MCTS - Node

Represents a single node in the MCTS tree. Stores board state, visit statistics,
prior probability, children, and virtual loss for parallel search support.
"""

import chess
import math
from typing import Dict, Optional


class MCTSNode:
    """MCTS tree node: board state, visit statistics, prior probability, and children."""

    def __init__(
        self,
        board: chess.Board,
        parent: Optional["MCTSNode"] = None,
        move: Optional[chess.Move] = None,
        prior: float = 0.0,
    ):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior

        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0  # for parallel search

        self.children: Dict[chess.Move, MCTSNode] = {}
        self.is_expanded = False

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0

        # Virtual loss reduces value to discourage multiple threads from exploring same path
        effective_value = self.value_sum - self.virtual_loss
        effective_visits = self.visit_count + self.virtual_loss
        return effective_value / effective_visits if effective_visits > 0 else 0.0

    def ucb_score(
        self, c_puct: float = 1.5, parent_visit_count: Optional[int] = None
    ) -> float:
        """PUCT: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))"""
        if parent_visit_count is None:
            parent_visit_count = self.parent.visit_count if self.parent else 1

        q_value = self.value()

        exploration = (
            c_puct
            * self.prior
            * math.sqrt(parent_visit_count)
            / (1 + self.visit_count + self.virtual_loss)
        )

        return q_value + exploration

    def select_child(self, c_puct: float = 1.5) -> "MCTSNode":
        """Return child with highest PUCT score."""
        best_score = -float("inf")
        best_child: Optional["MCTSNode"] = None

        # Pre-compute sqrt(parent_visits) once instead of once per child
        sqrt_parent = math.sqrt(self.visit_count) if self.visit_count > 0 else 0.0

        for child in self.children.values():
            vl = child.virtual_loss
            vc = child.visit_count
            eff_visits = vc + vl
            q = (child.value_sum - vl) / eff_visits if eff_visits > 0 else 0.0
            score = q + c_puct * child.prior * sqrt_parent / (1 + vc + vl)
            if score > best_score:
                best_score = score
                best_child = child

        # Should never be None if called correctly (when node has children)
        if best_child is None:
            raise ValueError("select_child called on node with no children")

        return best_child

    def expand(
        self, policy_probs: Dict[chess.Move, float], progressive: bool = False
    ) -> None:
        if self.is_expanded:
            return

        if progressive and len(policy_probs) > 20:
            # Progressive widening: expand only top moves initially
            sorted_moves = sorted(
                policy_probs.items(), key=lambda x: x[1], reverse=True
            )
            num_to_expand = min(
                len(sorted_moves), int(5 + 2 * math.sqrt(self.visit_count))
            )
            moves_to_expand = [m for m, _ in sorted_moves[:num_to_expand]]
        else:
            # Skip sort entirely — order doesn't matter when expanding all moves
            moves_to_expand = policy_probs.keys()

        for move in moves_to_expand:
            child_board = self.board.copy()
            child_board.push(move)

            prior = policy_probs.get(move, 1e-8)

            self.children[move] = MCTSNode(
                board=child_board, parent=self, move=move, prior=prior
            )

        self.is_expanded = True

    def update(self, value: float) -> None:
        self.visit_count += 1
        self.value_sum += value

    def add_virtual_loss(self) -> None:
        """Add virtual loss for parallel search."""
        self.virtual_loss += 1

    def remove_virtual_loss(self) -> None:
        """Remove virtual loss after simulation."""
        self.virtual_loss = max(0, self.virtual_loss - 1)

    def is_leaf(self) -> bool:
        """Check if node is a leaf (not expanded or terminal)"""
        return not self.is_expanded or self.board.is_game_over()
