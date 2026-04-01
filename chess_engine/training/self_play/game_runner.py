"""
SELF-PLAY - Game Runner (SelfPlayGameRunner)

Plays self-play games using MCTS and collects training examples.
"""

import os
import pickle
from typing import List, Dict, Tuple, Optional

import chess
import torch
import torch.nn as nn
from tqdm import tqdm

from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder
from chess_engine.search.mcts.evaluator import Evaluator
from chess_engine.search.mcts.cache import PositionCache
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

        # Evaluator (NN + exploration logic)
        self.evaluator = Evaluator(
            model=self.model,
            board_encoder=self.board_encoder,
            move_encoder=self.move_encoder,
            device=self.device,
            use_rnn=self.config.use_rnn,
            temperature=self.config.temperature,
            cache=PositionCache(max_size=100_000),
        )

        # MCTS (pure search)
        self.mcts = MCTS(
            model=self.model,
            board_encoder=self.board_encoder,
            move_encoder=self.move_encoder,
            device=self.device,
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
            dirichlet_epsilon=0.25,
            dirichlet_alpha=self.config.dirichlet_alpha,
            use_rnn=self.config.use_rnn,
            rnn_max_history=self.config.rnn_max_history,
        )

    def play_game(
        self, max_moves: int = 200, verbose: bool = False
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
        resigned = False

        if verbose:
            print("\nStarting self-play game...")

        try:
            while not board.is_game_over() and move_count < max_moves:
                move_count += 1

                # Adjust temperature (exploration vs exploitation)
                if move_count < self.config.temperature_threshold:
                    temp = self.config.temperature
                else:
                    temp = self.config.late_game_temperature

                self.evaluator.temperature = temp

                try:
                    # Run MCTS
                    move, stats = self.mcts.search(board, return_stats=True)

                    if stats is None or move not in board.legal_moves:
                        raise ValueError(f"MCTS returned illegal move: {move}")

                    # Check for resignation (only after move 10)
                    if move_count > 10 and should_resign(
                        stats, self.config.resign_threshold
                    ):
                        resigned = True
                        result = "0-1" if board.turn == chess.WHITE else "1-0"
                        if verbose:
                            print(f"   Resignation at move {move_count}")
                        break

                    # Extract complete policy
                    policy = extract_policy(board, stats)

                    # Value targets from MCTS root evaluation
                    mcts_root_value = stats.get("root_value", 0.0) if stats else 0.0
                    mcts_root_value = max(-1.0, min(1.0, mcts_root_value))

                    # Store example
                    example = GameExample(
                        fen=board.fen(),
                        policy=policy,
                        value=mcts_root_value,
                        move_number=move_count,
                    )
                    examples.append(example)

                    if verbose and move_count % 10 == 0:
                        print(f"   Move {move_count}: {move.uci()}")

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
            examples, result, resigned = self.play_game()
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
