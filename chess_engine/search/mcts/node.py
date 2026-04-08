"""
MCTS - Node

Represents a single node in the MCTS tree. Stores board state, visit statistics,
prior probability, children, and virtual loss for parallel search support.
"""

import chess
import math
from typing import Dict, Optional


class MCTSNode:
    """
    Node in the Monte Carlo Tree Search.

    Each node represents a board position and stores:
    - Visit count: How many times we've explored this position
    - Value sum: Accumulated value from simulations
    - Prior: Policy network's initial probability
    - Children: Child nodes (one per legal move)
    - Virtual loss: For parallel search support
    """

    def __init__(
        self,
        board: chess.Board,
        parent: Optional["MCTSNode"] = None,
        move: Optional[chess.Move] = None,
        prior: float = 0.0,
    ):
        """
        Initialize MCTS node.

        Args:
            board: Chess board state
            parent: Parent node
            move: Move that led to this node
            prior: Prior probability from policy network
        """
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior

        # Statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0  # For parallel search

        # Children nodes
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.is_expanded = False

    def value(self) -> float:
        """
        Average value of this node.

        Returns:
            Mean value from all simulations through this node
        """
        if self.visit_count == 0:
            return 0.0

        # Virtual loss reduces value to discourage multiple threads from exploring same path
        effective_value = self.value_sum - self.virtual_loss
        effective_visits = self.visit_count + self.virtual_loss
        return effective_value / effective_visits if effective_visits > 0 else 0.0

    def ucb_score(
        self, c_puct: float = 1.5, parent_visit_count: Optional[int] = None
    ) -> float:
        """
        Upper Confidence Bound score for node selection.

        Formula: Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

        Where:
        - Q(s,a): Average value (exploitation)
        - P(s,a): Prior probability from policy
        - N(s): Parent visit count
        - N(s,a): This node's visit count
        - c_puct: Exploration constant (higher = more exploration)

        Args:
            c_puct: Exploration constant (default: 1.5)
            parent_visit_count: Parent's visit count

        Returns:
            UCB score for this node
        """
        if parent_visit_count is None:
            parent_visit_count = self.parent.visit_count if self.parent else 1

        # Exploitation term: average value
        q_value = self.value()

        # Exploration term: balances exploration vs exploitation
        exploration = (
            c_puct
            * self.prior
            * math.sqrt(parent_visit_count)
            / (1 + self.visit_count + self.virtual_loss)
        )

        return q_value + exploration

    def select_child(self, c_puct: float = 1.5) -> "MCTSNode":
        """
        Select best child using UCB formula.

        Args:
            c_puct: Exploration constant

        Returns:
            Child node with highest UCB score
        """
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
        """
        Expand node by adding children for all legal moves.

        Args:
            policy_probs: Policy probabilities from neural network
            progressive: If True, use progressive widening (expand top moves first)
        """
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
            # Create child board
            child_board = self.board.copy()
            child_board.push(move)

            # Get prior probability
            prior = policy_probs.get(move, 1e-8)  # Small value if not in policy

            # Create child node
            self.children[move] = MCTSNode(
                board=child_board, parent=self, move=move, prior=prior
            )

        self.is_expanded = True

    def update(self, value: float) -> None:
        """
        Update node statistics after simulation.

        Args:
            value: Value from simulation (from perspective of player to move)
        """
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
