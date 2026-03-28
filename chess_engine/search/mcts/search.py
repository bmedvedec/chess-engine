"""
MCTS - Main Search Orchestrator

The main MCTS class that coordinates node expansion, neural network evaluation,
tree traversal, and move selection. Supports regular search, batched search,
and time-limited search.
"""

import chess
import time
import torch
from typing import Optional, Dict, Tuple

from chess_engine.search.mcts.node import MCTSNode
from chess_engine.search.mcts.cache import PositionCache
from chess_engine.search.mcts.evaluator import Evaluator
from chess_engine.search.mcts.tree import (
    backpropagate,
    should_terminate_early,
    get_search_stats,
)
from chess_engine.search.mcts.policy import add_dirichlet_noise, select_move


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
        rnn_max_history: int = 15,
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
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.use_progressive_widening = use_progressive_widening
        self.enable_early_termination = enable_early_termination
        self.early_termination_threshold = early_termination_threshold
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha
        self.eval_batch_size = eval_batch_size

        # Position cache
        self.enable_caching = enable_caching
        self.cache = PositionCache(max_cache_size) if enable_caching else None

        # Neural network evaluator
        self.evaluator = Evaluator(
            model=model,
            board_encoder=board_encoder,
            move_encoder=move_encoder,
            device=device,
            temperature=temperature,
            use_rnn=use_rnn,
            cache=self.cache,
            rnn_max_history=rnn_max_history,
        )

        model.eval()

    def clear_cache(self) -> None:
        """Clear position cache (call between moves)."""
        if self.cache is not None:
            self.cache.clear()

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
        policy_probs, root_value = self.evaluator.evaluate_position(root.board)

        # Initialize root node with its value (improves initial estimates)
        root.visit_count = 1
        root.value_sum = root_value

        # Add Dirichlet noise if enabled (for training exploration)
        if self.dirichlet_epsilon > 0:
            policy_probs = add_dirichlet_noise(
                policy_probs, self.dirichlet_epsilon, self.dirichlet_alpha
            )

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
                value = Evaluator.get_game_result(node.board)
            else:
                # Expansion and Evaluation
                policy_probs, value = self.evaluator.evaluate_position(node.board)
                if not node.board.is_game_over():
                    node.expand(policy_probs, progressive=self.use_progressive_widening)

            # Backpropagation
            backpropagate(search_path, value)

            # Check for early termination
            if self.enable_early_termination and sim > 30 and sim % check_interval == 0:
                if should_terminate_early(root, sim, self.early_termination_threshold):
                    break

        # Select best move based on visit counts
        move_number = len(board.move_stack)
        best_move = select_move(root, move_number)

        # Gather statistics if requested
        stats = None
        if return_stats:
            cache_size = len(self.cache) if self.cache is not None else 0
            stats = get_search_stats(root, cache_size)
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
        policy_probs, root_value = self.evaluator.evaluate_position(root.board)

        # Initialize root node with its value (improves initial estimates)
        root.visit_count = 1
        root.value_sum = root_value

        if self.dirichlet_epsilon > 0:
            policy_probs = add_dirichlet_noise(
                policy_probs, self.dirichlet_epsilon, self.dirichlet_alpha
            )

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
                    value = Evaluator.get_game_result(node.board)
                    backpropagate(search_path, value)

            # Batch evaluate all leaf nodes
            if batch_nodes:
                policies, values = self.evaluator.evaluate_positions_batch(batch_nodes)

                # Expand and backpropagate
                for node, policy_probs, value, search_path in zip(
                    batch_nodes, policies, values, batch_paths
                ):
                    node.expand(policy_probs, progressive=self.use_progressive_widening)
                    backpropagate(search_path, value)

            simulations_completed = batch_start + batch_size

            # Check for early termination
            if (
                self.enable_early_termination
                and batch_start > 30
                and batch_start % check_interval == 0
            ):
                if should_terminate_early(
                    root, batch_start, self.early_termination_threshold
                ):
                    break

        move_number = len(board.move_stack)
        best_move = select_move(root, move_number)

        stats = None
        if return_stats:
            cache_size = len(self.cache) if self.cache is not None else 0
            stats = get_search_stats(root, cache_size)
            stats["simulations_completed"] = min(
                simulations_completed, self.num_simulations
            )

        return best_move, stats

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
        policy_probs, root_value = self.evaluator.evaluate_position(root.board)

        if self.dirichlet_epsilon > 0:
            policy_probs = add_dirichlet_noise(
                policy_probs, self.dirichlet_epsilon, self.dirichlet_alpha
            )

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
                value = Evaluator.get_game_result(node.board)
            else:
                # Expansion and Evaluation
                policy_probs, value = self.evaluator.evaluate_position(node.board)
                if not node.board.is_game_over():
                    node.expand(policy_probs)

            # Backpropagation
            backpropagate(search_path, value)

            simulations_done += 1

        # Select best move based on visit counts
        best_move = select_move(root)

        # Gather statistics if requested
        stats = None
        if return_stats:
            cache_size = len(self.cache) if self.cache is not None else 0
            stats = get_search_stats(root, cache_size)
            stats["simulations_completed"] = simulations_done
            stats["time_used"] = time.time() - start_time

        return best_move, stats
