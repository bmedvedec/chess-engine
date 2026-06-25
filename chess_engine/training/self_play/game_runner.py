"""
SELF-PLAY - Game Runner (SelfPlayGameRunner)

Plays self-play games using MCTS and collects training examples.
"""

import os
import pickle
import random
from typing import List, Dict, Tuple, Optional

import chess
import torch
import torch.nn as nn
from tqdm import tqdm

from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder
from chess_engine.search.mcts.search import MCTS
from chess_engine.data.replay.buffer import ReplayBuffer
from chess_engine.data.replay.storage import GameExample
from chess_engine.training.self_play.config import SelfPlayConfig
from chess_engine.training.self_play.stats import SelfPlayStatistics
from chess_engine.training.self_play.policy_extraction import extract_policy
from chess_engine.training.self_play.value_targets import should_resign


class SelfPlayGameRunner:
    """
    Worker for generating self-play games.

    Plays games using MCTS and collects training examples.

    Backward-compatible alias: SelfPlayWorker = SelfPlayGameRunner
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Optional[SelfPlayConfig] = None,
    ):
        """
        Initialize self-play worker.

        Args:
            model: Neural network model
            device: Device (cuda/cpu)
            config: Self-play configuration
        """
        self.model = model
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.config = config or SelfPlayConfig()

        # Create encoders
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()

        # MCTS (pure search) — owns the evaluator and position cache internally
        self.mcts = MCTS(
            model=self.model,
            board_encoder=self.board_encoder,
            move_encoder=self.move_encoder,
            device=self.device,
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
            temperature=self.config.temperature,
            temperature_threshold=self.config.temperature_threshold,
            late_game_temperature=self.config.late_game_temperature,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
            dirichlet_alpha=self.config.dirichlet_alpha,
            use_rnn=self.config.use_rnn,
            rnn_max_history=self.config.rnn_max_history,
        )

    def play_game(
        self, max_moves: int = 100, verbose: bool = False
    ) -> Tuple[List[GameExample], str, bool]:
        """
        Play one self-play game.

        Args:
            max_moves: Maximum moves before declaring draw
            verbose: Print game progress

        Returns:
            Tuple of (examples, result, resigned)
        """
        if max_moves is None:
            max_moves = self.config.max_moves

        board = chess.Board()
        examples = []
        move_count = 0
        move_history: List[str] = []  # UCI moves played so far, for RNN input
        resigned = False
        resign_streak = {chess.WHITE: 0, chess.BLACK: 0}
        _RESIGN_STREAK_REQUIRED = 3

        if verbose:
            print("\nStarting self-play game...")

        # Random opening phase — play N random legal moves before MCTS takes over.
        # This diversifies openings and breaks color-specific patterns that cause
        # color bias when the model always sees the same deterministic opening lines.
        num_random = getattr(self.config, "random_opening_moves", 0)
        for _ in range(num_random):
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            non_captures = [m for m in legal if not board.is_capture(m)]
            opening_move = random.choice(non_captures if non_captures else legal)
            move_history.append(opening_move.uci())
            board.push(opening_move)

        try:
            while not board.is_game_over() and move_count < max_moves:
                move_count += 1

                # Adjust temperature (exploration vs exploitation)
                if move_count < self.config.temperature_threshold:
                    temp = self.config.temperature
                else:
                    temp = self.config.late_game_temperature

                self.mcts.evaluator.temperature = temp

                try:
                    # Run MCTS (batched: fewer GPU round-trips than search())
                    move, stats = self.mcts.search_batched(board, return_stats=True)

                    if stats is None or move not in board.legal_moves:
                        raise ValueError(f"MCTS returned illegal move: {move}")

                    if move_count > 10:
                        side = board.turn
                        if should_resign(stats, self.config.resign_threshold):
                            resign_streak[side] += 1
                        else:
                            resign_streak[side] = 0

                        if resign_streak[side] >= _RESIGN_STREAK_REQUIRED:
                            resigned = True
                            result = "0-1" if side == chess.WHITE else "1-0"
                            if verbose:
                                root_val = stats.get("root_value", float("nan"))
                                print(
                                    f"   Resignation at move {move_count} "
                                    f"(root_value={root_val:.3f}, "
                                    f"threshold={self.config.resign_threshold})"
                                )
                            break

                    # Extract complete policy
                    policy = extract_policy(board, stats)

                    # Value targets from MCTS root evaluation
                    mcts_root_value = stats.get("root_value", 0.0) if stats else 0.0
                    mcts_root_value = max(-1.0, min(1.0, mcts_root_value))

                    # Store example — snapshot history before this move is pushed
                    example = GameExample(
                        fen=board.fen(),
                        policy=policy,
                        value=mcts_root_value,
                        move_number=move_count,
                        move_history=list(move_history),
                    )
                    examples.append(example)

                    if verbose and move_count % 10 == 0:
                        print(f"   Move {move_count}: {move.uci()}")

                    move_history.append(move.uci())
                    board.push(move)

                except Exception as e:
                    print(f"Warning: Error at move {move_count}: {e}")
                    result = "1/2-1/2"
                    break

            # Determine game result (if not resigned)
            if not resigned:
                if board.is_game_over():
                    result = board.result()
                else:
                    result = "1/2-1/2"  # Max moves reached

            # Blend MCTS root values with game outcome to break draw-collapse loop.
            # Pure MCTS values collapse toward 0 when all games are draws
            # mixing in the actual outcome restores a meaningful training signal.
            if examples:
                alpha = self.config.value_blend_alpha
                for ex in examples:
                    fen_turn = ex.fen.split()[1]  # 'w' or 'b'
                    if result == "1-0":
                        # Win for white, loss for black — flip per side
                        game_val = 1.0 if fen_turn == "w" else -1.0
                    elif result == "0-1":
                        # Win for black, loss for white — flip per side
                        game_val = -1.0 if fen_turn == "w" else 1.0
                    else:
                        # Draw: penalise BOTH sides equally — no sign flip.
                        # Using -white_outcome here would reward black (+penalty),
                        # cancelling the penalty entirely across the dataset.
                        game_val = -self.config.draw_value_penalty
                    ex.value = alpha * ex.value + (1.0 - alpha) * game_val

            if verbose:
                print(f"   Game over: {result}")
                if resigned:
                    print(f"   (Resigned)")
                print(f"   Collected {len(examples)} training examples")

            return examples, result, resigned

        except Exception as e:
            print(f"Fatal error in play_game: {e}")
            if examples:
                result = "1/2-1/2"
                return examples, result, False
            else:
                return [], "1/2-1/2", False

    def play_games(
        self,
        num_games: int,
        buffer: Optional[ReplayBuffer] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[List[GameExample], SelfPlayStatistics]:
        """
        Play multiple self-play games sequentially.

        Args:
            num_games: Number of games to play
            buffer: Optional ReplayBuffer to add examples to
            save_path: Optional path to save examples (if no buffer)

        Returns:
            Tuple of (list of training examples, self-play statistics)
        """
        all_examples = []
        stats = SelfPlayStatistics()

        print(f"\nPlaying {num_games} self-play games...")

        for i in tqdm(range(num_games), desc="Self-play"):
            examples, result, resigned = self.play_game(max_moves=self.config.max_moves)
            all_examples.extend(examples)
            stats.update(result, len(examples), len(examples), resigned)

            if buffer is not None:
                buffer.add_game_examples(examples)

        stats.print_summary()

        if save_path and buffer is None:
            self._save_examples(all_examples, save_path)

        return all_examples, stats

    def _save_examples(self, examples: List[GameExample], filepath: str):
        """Save examples to disk."""
        serializable = [
            {
                "fen": example.fen,
                "policy": example.policy,
                "value": example.value,
                "move_number": example.move_number,
            }
            for example in examples
        ]

        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(serializable, f)

        print(f"\nSaved {len(examples)} examples to {filepath}")


# Backward-compatible alias
SelfPlayWorker = SelfPlayGameRunner
