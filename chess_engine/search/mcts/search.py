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
    """Neural-network-guided Monte Carlo Tree Search with batched GPU evaluation."""

    def __init__(
        self,
        model,
        board_encoder,
        move_encoder,
        device: torch.device,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        temperature: float = 1.0,
        temperature_threshold: int = 30,
        late_game_temperature: float = 0.1,
        use_rnn: bool = False,
        # Caching
        enable_caching: bool = True,
        max_cache_size: int = 100000,
        # Batch evaluation
        eval_batch_size: int = 16,
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
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.late_game_temperature = late_game_temperature
        self.use_progressive_widening = use_progressive_widening
        self.enable_early_termination = enable_early_termination
        self.early_termination_threshold = early_termination_threshold
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha
        self.eval_batch_size = eval_batch_size

        self.enable_caching = enable_caching
        self.cache = PositionCache(max_cache_size) if enable_caching else None

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
        root = MCTSNode(board.copy())

        policy_probs, root_value = self.evaluator.evaluate_position(root.board)

        # Do NOT seed visit_count/value_sum here. Seeding with count=1 causes the
        # NN's raw estimate to be double-counted — once as the seed and again when
        # the first simulation backpropagates through the root. AlphaZero counts
        # only real simulations; root.value() will be purely simulation-derived.

        if self.dirichlet_epsilon > 0:
            policy_probs = add_dirichlet_noise(
                policy_probs, self.dirichlet_epsilon, self.dirichlet_alpha
            )

        root.expand(policy_probs, progressive=self.use_progressive_widening)

        check_interval = max(10, self.num_simulations // 10)

        for sim in range(self.num_simulations):
            node = root
            search_path = [node]

            while not node.is_leaf():
                node = node.select_child(self.c_puct)
                search_path.append(node)

            if node.board.is_game_over():
                value = Evaluator.get_game_result(node.board)
            else:
                policy_probs, value = self.evaluator.evaluate_position(node.board)
                if not node.board.is_game_over():
                    node.expand(policy_probs, progressive=self.use_progressive_widening)

            backpropagate(search_path, value)

            if self.enable_early_termination and sim > 30 and sim % check_interval == 0:
                if should_terminate_early(root, sim, self.early_termination_threshold):
                    break

        move_number = len(board.move_stack)
        best_move = select_move(
            root, move_number,
            temperature=self.temperature,
            temperature_threshold=self.temperature_threshold,
            late_game_temperature=self.late_game_temperature,
        )

        stats = None
        if return_stats:
            cache_size = len(self.cache) if self.cache is not None else 0
            stats = get_search_stats(root, cache_size)
            stats["simulations_completed"] = min(sim + 1, self.num_simulations)

        return best_move, stats

    def search_batched(
        self, board: chess.Board, return_stats: bool = False
    ) -> Tuple[chess.Move, Optional[Dict]]:
        """MCTS search with batched neural network evaluations for GPU efficiency."""
        root = MCTSNode(board.copy())

        policy_probs, root_value = self.evaluator.evaluate_position(root.board)

        # Do NOT seed visit_count/value_sum here — same reason as search().

        if self.dirichlet_epsilon > 0:
            policy_probs = add_dirichlet_noise(
                policy_probs, self.dirichlet_epsilon, self.dirichlet_alpha
            )

        root.expand(policy_probs, progressive=self.use_progressive_widening)

        check_interval = max(10, self.num_simulations // 10)
        simulations_completed = 0

        for batch_start in range(0, self.num_simulations, self.eval_batch_size):
            batch_size = min(self.eval_batch_size, self.num_simulations - batch_start)

            batch_nodes = []
            batch_paths = []
            visited_leaves: set = set()  # guard against evaluating the same leaf twice

            for _ in range(batch_size):
                node = root
                search_path = [node]
                node.add_virtual_loss()

                # Selection — virtual loss on each visited node steers subsequent
                # paths in this batch away from the same branch.
                while not node.is_leaf():
                    node = node.select_child(self.c_puct)
                    search_path.append(node)
                    node.add_virtual_loss()

                if node.board.is_game_over():
                    for n in search_path:
                        n.remove_virtual_loss()
                    value = Evaluator.get_game_result(node.board)
                    backpropagate(search_path, value)
                elif id(node) in visited_leaves:
                    # Duplicate leaf within this batch (can occur in positions with
                    # few legal moves even with virtual loss). Undo virtual loss and
                    # skip — the first occurrence will expand and backpropagate it.
                    for n in search_path:
                        n.remove_virtual_loss()
                else:
                    visited_leaves.add(id(node))
                    batch_nodes.append(node)
                    batch_paths.append(search_path)

            if batch_nodes:
                policies, values = self.evaluator.evaluate_positions_batch(batch_nodes)

                for node, policy_probs, value, search_path in zip(
                    batch_nodes, policies, values, batch_paths
                ):
                    for n in search_path:
                        n.remove_virtual_loss()
                    node.expand(policy_probs, progressive=self.use_progressive_widening)
                    backpropagate(search_path, value)

            simulations_completed = batch_start + batch_size

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
        best_move = select_move(
            root, move_number,
            temperature=self.temperature,
            temperature_threshold=self.temperature_threshold,
            late_game_temperature=self.late_game_temperature,
        )

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
        """Run MCTS until time_limit expires, then return best move."""
        start_time = time.time()

        root = MCTSNode(board.copy())
        policy_probs, root_value = self.evaluator.evaluate_position(root.board)

        if self.dirichlet_epsilon > 0:
            policy_probs = add_dirichlet_noise(
                policy_probs, self.dirichlet_epsilon, self.dirichlet_alpha
            )

        root.expand(policy_probs, progressive=self.use_progressive_widening)

        simulations_done = 0

        while True:
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

            node = root
            search_path = [node]

            while not node.is_leaf():
                node = node.select_child(self.c_puct)
                search_path.append(node)

            if node.board.is_game_over():
                value = Evaluator.get_game_result(node.board)
            else:
                policy_probs, value = self.evaluator.evaluate_position(node.board)
                if not node.board.is_game_over():
                    node.expand(policy_probs)

            backpropagate(search_path, value)

            simulations_done += 1

        move_number = len(board.move_stack)
        best_move = select_move(
            root, move_number,
            temperature=self.temperature,
            temperature_threshold=self.temperature_threshold,
            late_game_temperature=self.late_game_temperature,
        )

        stats = None
        if return_stats:
            cache_size = len(self.cache) if self.cache is not None else 0
            stats = get_search_stats(root, cache_size)
            stats["simulations_completed"] = simulations_done
            stats["time_used"] = time.time() - start_time

        return best_move, stats
