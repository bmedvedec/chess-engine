"""
MONTE CARLO TREE SEARCH (MCTS)
Implementation

MCTS allows the engine to "think ahead" by simulating games and building a search tree.

How MCTS Works:
1. Selection: Navigate tree using UCB (Upper Confidence Bound) formula
2. Expansion: Add new node to tree
3. Evaluation: Use neural network to evaluate position
4. Backpropagation: Update statistics up the tree

This makes the engine:
- See tactical threats
- Find combinations
- Avoid hanging pieces
- Detect checkmates
"""

import chess
import numpy as np
import torch
from typing import Optional, Dict, List, Tuple
import math
import time
from chess_engine.utils.move_encoder import MoveHistory


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

        for child in self.children.values():
            score = child.ucb_score(c_puct, self.visit_count)
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

        # Sort moves by prior probability
        sorted_moves = sorted(policy_probs.items(), key=lambda x: x[1], reverse=True)

        if progressive and len(sorted_moves) > 20:
            # Progressive widening: expand only top moves initially
            # Expand more as node is visited more
            num_to_expand = min(
                len(sorted_moves), int(5 + 2 * math.sqrt(self.visit_count))
            )
            moves_to_expand = [m for m, _ in sorted_moves[:num_to_expand]]
        else:
            moves_to_expand = [m for m, _ in sorted_moves]

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


class MCTS:
    """
    Monte Carlo Tree Search for chess.

    Uses neural network to guide search and evaluate positions.

    Features:
    - Position caching (transposition table)
    - Batch evaluations for GPU efficiency
    - Virtual loss for parallel search
    - Temperature scheduling
    - Progressive widening
    - Dirichlet noise
    - Early termination
    """

    def __init__(
        self,
        model,
        board_encoder,
        move_encoder,
        device: torch.device,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        use_rnn: bool = False,
        # Caching
        enable_caching: bool = True,
        max_cache_size: int = 10000,
        # Batch evaluation
        eval_batch_size: int = 8,
        # Progressive widening
        use_progressive_widening: bool = False,
        # Early termination
        enable_early_termination: bool = True,
        early_termination_threshold: float = 0.7,
        # Dirichlet noise (for training)
        dirichlet_epsilon: float = 0.0,
        dirichlet_alpha: float = 0.3,
    ):
        """
        Initialize MCTS.

        Args:
            model: Neural network model
            board_encoder: BoardEncoder instance
            move_encoder: MoveEncoder instance
            device: Device (cuda/cpu)
            num_simulations: Number of simulations per search (default: 100)
            c_puct: Exploration constant (default: 1.5)
            temperature: Temperature for move selection (default: 1.0)
            use_rnn: Whether model uses RNN

            enable_caching: Enable position caching (default: True)
            max_cache_size: Maximum cache size (default: 10000)

            eval_batch_size: Batch size for neural network evaluation (default: 8)

            use_progressive_widening: Use progressive widening (default: False)

            enable_early_termination: Enable early termination (default: True)
            early_termination_threshold: Threshold for early termination (default: 0.7)

            dirichlet_epsilon: Dirichlet noise weight (0.0 = off, 0.25 typical) (default: 0.0)
            dirichlet_alpha: Dirichlet alpha parameter (default: 0.3)
        """
        self.model = model
        self.board_encoder = board_encoder
        self.move_encoder = move_encoder
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.use_rnn = use_rnn

        # Position cache for transposition table
        self.enable_caching = enable_caching
        self.max_cache_size = max_cache_size
        self.position_cache: Dict[str, Tuple[Dict[chess.Move, float], float]] = {}

        # Batch evaluation
        self.eval_batch_size = eval_batch_size

        # Progressive widening
        self.use_progressive_widening = use_progressive_widening

        # Early termination
        self.enable_early_termination = enable_early_termination
        self.early_termination_threshold = early_termination_threshold

        # Dirichlet noise
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha

        self.model.eval()

    def clear_cache(self) -> None:
        """Clear position cache (call between moves)"""
        self.position_cache.clear()

    def search(
        self, board: chess.Board, return_stats: bool = False
    ) -> Tuple[chess.Move, Optional[Dict]]:
        """
        Run MCTS search from given position.

        Args:
            board: Current board state
            return_stats: Whether to return search statistics

        Returns:
            Tuple of (best_move, optional_stats)
        """
        # Create root node
        root = MCTSNode(board.copy())

        # Evaluate root to get initial policy
        policy_probs, root_value = self._evaluate_position(root.board)

        # Initialize root node with its value (improves initial estimates)
        root.visit_count = 1
        root.value_sum = root_value

        # Add Dirichlet noise if enabled (for training exploration)
        if self.dirichlet_epsilon > 0:
            policy_probs = self._add_dirichlet_noise(policy_probs)

        root.expand(policy_probs, progressive=self.use_progressive_widening)

        # Run simulations with early termination check
        check_interval = max(10, self.num_simulations // 10)

        for sim in range(self.num_simulations):
            node = root
            search_path = [node]

            # Selection: traverse tree to leaf
            while not node.is_leaf():
                node = node.select_child(self.c_puct)
                search_path.append(node)

            # Check if terminal
            if node.board.is_game_over():
                # Game over - get actual result
                value = self._get_game_result(node.board)
            else:
                # Expansion and Evaluation
                policy_probs, value = self._evaluate_position(node.board)
                if not node.board.is_game_over():
                    node.expand(policy_probs, progressive=self.use_progressive_widening)

            # Backpropagation
            self._backpropagate(search_path, value)

            # Check for early termination
            if self.enable_early_termination and sim > 30 and sim % check_interval == 0:
                if self._should_terminate_early(root, sim):
                    break

        # Select best move based on visit counts
        move_number = len(board.move_stack)
        best_move = self._select_move(root, move_number)

        # Gather statistics if requested
        stats = None
        if return_stats:
            stats = self._get_search_stats(root)
            stats["simulations_completed"] = min(sim + 1, self.num_simulations)

        return best_move, stats

    def search_batched(
        self, board: chess.Board, return_stats: bool = False
    ) -> Tuple[chess.Move, Optional[Dict]]:
        """
        MCTS search with batched neural network evaluations.
        More efficient on GPU.

        Args:
            board: Current board state
            return_stats: Whether to return search statistics

        Returns:
            Tuple of (best_move, optional_stats)
        """
        root = MCTSNode(board.copy())

        # Initial evaluation
        policy_probs, root_value = self._evaluate_position(root.board)

        # Initialize root node with its value (improves initial estimates)
        root.visit_count = 1
        root.value_sum = root_value

        if self.dirichlet_epsilon > 0:
            policy_probs = self._add_dirichlet_noise(policy_probs)

        root.expand(policy_probs, progressive=self.use_progressive_widening)

        # Run simulations in batches
        check_interval = max(10, self.num_simulations // 10)
        simulations_completed = 0

        for batch_start in range(0, self.num_simulations, self.eval_batch_size):
            batch_size = min(self.eval_batch_size, self.num_simulations - batch_start)

            # Collect leaf nodes for this batch
            batch_nodes = []
            batch_paths = []

            for _ in range(batch_size):
                node = root
                search_path = [node]

                # Selection
                while not node.is_leaf():
                    node = node.select_child(self.c_puct)
                    search_path.append(node)

                if not node.board.is_game_over():
                    batch_nodes.append(node)
                    batch_paths.append(search_path)
                else:
                    # Terminal node - backpropagate immediately
                    value = self._get_game_result(node.board)
                    self._backpropagate(search_path, value)

            # Batch evaluate all leaf nodes
            if batch_nodes:
                policies, values = self._evaluate_positions_batch(batch_nodes)

                # Expand and backpropagate
                for node, policy_probs, value, search_path in zip(
                    batch_nodes, policies, values, batch_paths
                ):
                    node.expand(policy_probs, progressive=self.use_progressive_widening)
                    self._backpropagate(search_path, value)

            simulations_completed = batch_start + batch_size

            # Check for early termination
            if (
                self.enable_early_termination
                and batch_start > 30
                and batch_start % check_interval == 0
            ):
                if self._should_terminate_early(root, batch_start):
                    break

        move_number = len(board.move_stack)
        best_move = self._select_move(root, move_number)

        stats = None
        if return_stats:
            stats = self._get_search_stats(root)
            stats["simulations_completed"] = min(
                simulations_completed, self.num_simulations
            )

        return best_move, stats

    def _evaluate_position(
        self, board: chess.Board
    ) -> Tuple[Dict[chess.Move, float], float]:
        """
        Evaluate position using neural network with caching.

        Args:
            board: Board to evaluate

        Returns:
            Tuple of (policy_probs, value)
        """
        # Check cache first
        if self.enable_caching:
            fen = board.fen()
            if fen in self.position_cache:
                return self.position_cache[fen]

        with torch.no_grad():
            # Encode board
            board_tensor = self.board_encoder.board_to_tensor(board).unsqueeze(0)
            board_tensor = board_tensor.to(self.device)

            # Get predictions
            if self.use_rnn:
                history_encoder = MoveHistory(max_length=50)
                move_history = history_encoder.encode_game_history(board, pad=True)
                move_history = move_history.unsqueeze(0).to(self.device)

                # Get actual sequence length (not padded length)
                actual_length = min(len(board.move_stack), 50)
                seq_length = torch.LongTensor([actual_length])

                policy_logits, value, _ = self.model(
                    board_tensor, move_history, seq_length
                )
            else:
                policy_logits, value, _ = self.model(board_tensor)

            # Convert to move probabilities
            policy_probs = self.move_encoder.policy_to_move_probs(
                policy_logits[0], board, temperature=self.temperature
            )

            value = value.item()

        # Cache result if enabled
        if self.enable_caching and len(self.position_cache) < self.max_cache_size:
            self.position_cache[fen] = (policy_probs, value)

        return policy_probs, value

    def _evaluate_positions_batch(
        self, nodes: List[MCTSNode]
    ) -> Tuple[List[Dict[chess.Move, float]], List[float]]:
        """
        Evaluate multiple positions in a batch (more efficient on GPU).

        Args:
            nodes: List of nodes to evaluate

        Returns:
            Tuple of (policies, values)
        """
        with torch.no_grad():
            # Encode all boards
            board_tensors = torch.stack(
                [self.board_encoder.board_to_tensor(node.board) for node in nodes]
            ).to(self.device)

            # Get predictions
            if self.use_rnn:
                history_encoder = MoveHistory(max_length=50)

                # Encode all move histories
                move_histories = []
                seq_lengths = []

                for node in nodes:
                    move_history = history_encoder.encode_game_history(
                        node.board, pad=True
                    )
                    move_histories.append(move_history)

                    actual_length = min(len(node.board.move_stack), 50)
                    seq_lengths.append(actual_length)

                move_histories = torch.stack(move_histories).to(self.device)
                seq_lengths = torch.LongTensor(seq_lengths)

                policy_logits, values, _ = self.model(
                    board_tensors, move_histories, seq_lengths
                )
            else:
                policy_logits, values, _ = self.model(board_tensors)

            # Convert to move probabilities
            policies = []
            for i, node in enumerate(nodes):
                policy_probs = self.move_encoder.policy_to_move_probs(
                    policy_logits[i], node.board, temperature=self.temperature
                )
                policies.append(policy_probs)

            values = values.squeeze().cpu().numpy()
            if len(nodes) == 1:
                values = [float(values)]
            else:
                values = values.tolist()

        return policies, values

    def _get_game_result(self, board: chess.Board) -> float:
        """
        Get game result for terminal position.

        Args:
            board: Terminal board position

        Returns:
            Result from perspective of side that just moved
        """
        result = board.result()

        # The side that just moved is opposite to current turn
        # (because turn switches after a move)
        side_that_moved = not board.turn

        if result == "1-0":  # White wins
            return 1.0 if side_that_moved == chess.WHITE else -1.0
        elif result == "0-1":  # Black wins
            return -1.0 if side_that_moved == chess.WHITE else 1.0
        else:  # Draw
            return 0.0

    def _backpropagate(self, search_path: List[MCTSNode], value: float) -> None:
        """
        Backpropagate value through search path.

        Args:
            search_path: Path from root to leaf
            value: Value from leaf node
        """
        # Alternate value sign for each level (player perspective)
        for node in reversed(search_path):
            node.update(value)
            value = -value  # Flip for opponent

    def _select_move(self, root: MCTSNode, move_number: int = 0) -> chess.Move:
        """
        Select move based on visit counts.

        Args:
            root: Root node after search
            move_number: Current move number (for temperature scheduling)

        Returns:
            Best move
        """
        # Get visit counts
        visit_counts = {
            move: child.visit_count for move, child in root.children.items()
        }

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

    def _add_dirichlet_noise(
        self, policy_probs: Dict[chess.Move, float]
    ) -> Dict[chess.Move, float]:
        """
        Add Dirichlet noise to policy for exploration (used in self-play training).

        Args:
            policy_probs: Original policy probabilities

        Returns:
            Policy with noise added
        """
        moves = list(policy_probs.keys())
        probs = np.array([policy_probs[m] for m in moves])

        # Generate Dirichlet noise
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))

        # Mix policy with noise
        noisy_probs = (
            1 - self.dirichlet_epsilon
        ) * probs + self.dirichlet_epsilon * noise

        # Return as dictionary
        return {move: prob for move, prob in zip(moves, noisy_probs)}

    def _should_terminate_early(self, root: MCTSNode, simulations_done: int) -> bool:
        """
        Check if we should terminate early when one move is clearly dominant.

        Args:
            root: Root node
            simulations_done: Number of simulations completed

        Returns:
            True if should terminate early
        """
        if simulations_done < 30:  # Need minimum samples
            return False

        visit_counts = [child.visit_count for child in root.children.values()]
        if len(visit_counts) < 2:
            return False

        visit_counts_sorted = sorted(visit_counts, reverse=True)

        # If best move has >threshold% of visits and 2x more than second best
        best_ratio = visit_counts_sorted[0] / sum(visit_counts)
        if best_ratio > self.early_termination_threshold:
            if visit_counts_sorted[0] > 2 * visit_counts_sorted[1]:
                return True

        return False

    def _get_search_stats(self, root: MCTSNode) -> Dict:
        """
        Get search statistics for analysis.

        Args:
            root: Root node after search

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_visits": root.visit_count,
            "root_value": root.value(),
            "top_moves": [],
            "cache_size": len(self.position_cache) if self.enable_caching else 0,
            "num_children": len(root.children),
        }

        # Sort children by visit count
        sorted_children = sorted(
            root.children.items(), key=lambda x: x[1].visit_count, reverse=True
        )

        # Get top 5 moves
        for move, child in sorted_children[:5]:
            stats["top_moves"].append(
                {
                    "move": move.uci(),
                    "visits": child.visit_count,
                    "value": child.value(),
                    "prior": child.prior,
                    "visit_pct": (
                        child.visit_count / root.visit_count
                        if root.visit_count > 0
                        else 0
                    ),
                }
            )

        return stats

    def search_with_time_limit(
        self,
        board: chess.Board,
        time_limit: float,
        return_stats: bool = False,
        min_simulations: int = 10,
    ) -> Tuple[chess.Move, Optional[Dict]]:
        """
        Run MCTS search with time limit (iterative deepening).

        Instead of fixed number of simulations, runs as many simulations
        as possible within the time limit.

        Args:
            board: Current board state
            time_limit: Maximum time in seconds
            return_stats: Whether to return search statistics
            min_simulations: Minimum simulations before time check (default: 10)

        Returns:
            Tuple of (best_move, optional_stats)
        """
        start_time = time.time()

        # Create root node
        root = MCTSNode(board.copy())

        # Evaluate root to get initial policy
        policy_probs, root_value = self._evaluate_position(root.board)

        if self.dirichlet_epsilon > 0:
            policy_probs = self._add_dirichlet_noise(policy_probs)

        root.expand(policy_probs, progressive=self.use_progressive_widening)

        # Run simulations until time expires
        simulations_done = 0

        while True:
            # Check time after minimum simulations
            if simulations_done >= min_simulations:
                elapsed = time.time() - start_time

                # Use 95% of time limit as safety margin
                if elapsed >= time_limit * 0.95:
                    break

                # Estimate if we have time for another batch
                if simulations_done > 0:
                    time_per_sim = elapsed / simulations_done
                    # Check if we have time for at least 5 more simulations
                    if elapsed + time_per_sim * 5 > time_limit * 0.95:
                        break

            # Run one simulation
            node = root
            search_path = [node]

            # Selection: traverse tree to leaf
            while not node.is_leaf():
                node = node.select_child(self.c_puct)
                search_path.append(node)

            # Check if terminal
            if node.board.is_game_over():
                value = self._get_game_result(node.board)
            else:
                # Expansion and Evaluation
                policy_probs, value = self._evaluate_position(node.board)
                if not node.board.is_game_over():
                    node.expand(policy_probs)

            # Backpropagation
            self._backpropagate(search_path, value)

            simulations_done += 1

        # Select best move based on visit counts
        best_move = self._select_move(root)

        # Gather statistics if requested
        stats = None
        if return_stats:
            stats = self._get_search_stats(root)
            stats["simulations_completed"] = simulations_done
            stats["time_used"] = time.time() - start_time

        return best_move, stats
