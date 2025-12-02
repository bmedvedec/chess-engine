"""
CHAPTER 9: SELF-PLAY INFRASTRUCTURE

Generate training data by having the engine play against itself.

Self-Play Loop:
1. Load current model
2. Play games using MCTS
3. Store positions, moves, and outcomes using ReplayBuffer
4. Use this data to train better model
5. Repeat (model improves over time!)

This is the core of AlphaZero's training approach.
"""

import os
import sys
import time
from typing import List, Dict, Tuple, Optional
import pickle
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
import argparse

import numpy as np
import torch
import torch.nn as nn
import chess
from tqdm import tqdm

from chess_engine.models.hybrid_model import HybridChessNet
from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder
from chess_engine.search.mcts import MCTS
from chess_engine.data.replay_buffer import ReplayBuffer, GameExample

# Constants
DEFAULT_MAX_MOVES = 200
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TEMPERATURE_THRESHOLD = 15
LATE_GAME_TEMPERATURE = 0.1
DEFAULT_NUM_SIMULATIONS = 100
DEFAULT_C_PUCT = 1.5


@dataclass
class SelfPlayConfig:
    """Configuration for self-play"""

    num_simulations: int = DEFAULT_NUM_SIMULATIONS
    c_puct: float = DEFAULT_C_PUCT
    temperature: float = DEFAULT_TEMPERATURE
    temperature_threshold: int = DEFAULT_TEMPERATURE_THRESHOLD
    max_moves: int = DEFAULT_MAX_MOVES
    use_rnn: bool = False
    late_game_temperature: float = LATE_GAME_TEMPERATURE

    @classmethod
    def from_args(cls, args) -> "SelfPlayConfig":
        """Create config from command line arguments"""
        return cls(
            num_simulations=getattr(args, "simulations", DEFAULT_NUM_SIMULATIONS),
            c_puct=getattr(args, "c_puct", DEFAULT_C_PUCT),
            temperature=getattr(args, "temperature", DEFAULT_TEMPERATURE),
            temperature_threshold=getattr(
                args, "temp_threshold", DEFAULT_TEMPERATURE_THRESHOLD
            ),
            max_moves=getattr(args, "max_moves", DEFAULT_MAX_MOVES),
            use_rnn=getattr(args, "use_rnn", False),
        )


@dataclass
class SelfPlayStatistics:
    """Statistics for self-play games"""

    total_games: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    total_examples: int = 0
    total_moves: int = 0
    errors: int = 0

    @property
    def avg_moves_per_game(self) -> float:
        """Average moves per game"""
        return self.total_moves / self.total_games if self.total_games > 0 else 0.0

    @property
    def white_win_rate(self) -> float:
        """White win percentage"""
        return self.white_wins / self.total_games * 100 if self.total_games > 0 else 0.0

    @property
    def black_win_rate(self) -> float:
        """Black win percentage"""
        return self.black_wins / self.total_games * 100 if self.total_games > 0 else 0.0

    @property
    def draw_rate(self) -> float:
        """Draw percentage"""
        return self.draws / self.total_games * 100 if self.total_games > 0 else 0.0

    def update(self, result: str, num_examples: int, num_moves: int):
        """Update statistics with game result"""
        self.total_games += 1
        self.total_examples += num_examples
        self.total_moves += num_moves

        if result == "1-0":
            self.white_wins += 1
        elif result == "0-1":
            self.black_wins += 1
        else:
            self.draws += 1

    def print_summary(self):
        """Print statistics summary"""
        print(f"\n📊 Self-Play Statistics:")
        print(f"   Total games: {self.total_games}")
        print(f"   Total examples: {self.total_examples}")
        print(f"   Total moves: {self.total_moves}")
        print(f"   Avg moves/game: {self.avg_moves_per_game:.1f}")
        print(f"\n   Results:")
        print(f"      White wins: {self.white_wins} ({self.white_win_rate:.1f}%)")
        print(f"      Black wins: {self.black_wins} ({self.black_win_rate:.1f}%)")
        print(f"      Draws: {self.draws} ({self.draw_rate:.1f}%)")
        if self.errors > 0:
            print(f"   Errors: {self.errors}")


class SelfPlayWorker:
    """
    Worker for generating self-play games.

    Plays games using MCTS and collects training examples.
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

        # Create MCTS
        self.mcts = MCTS(
            model=model,
            board_encoder=self.board_encoder,
            move_encoder=self.move_encoder,
            device=self.device,
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
            temperature=self.config.temperature,
            use_rnn=self.config.use_rnn,
        )

    def _extract_policy(
        self, board: chess.Board, stats: Optional[Dict]
    ) -> Dict[str, float]:
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
            # Get visit counts for all moves
            visit_counts = stats["visit_counts"]
            total_visits = sum(visit_counts.values())

            if total_visits > 0:
                # Assign probabilities based on visit counts
                for move in legal_moves:
                    move_uci = move.uci()
                    visits = visit_counts.get(move_uci, 0)
                    policy[move_uci] = visits / total_visits
            else:
                # No visits recorded, uniform distribution
                uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0.0
                policy = {move.uci(): uniform_prob for move in legal_moves}

        elif stats and "top_moves" in stats:
            # Fallback: Use top_moves but include all legal moves
            top_moves_dict = {
                move_stat["move"]: move_stat["visit_pct"]
                for move_stat in stats["top_moves"]
            }

            # Create policy for all legal moves
            for move in legal_moves:
                move_uci = move.uci()
                policy[move_uci] = top_moves_dict.get(move_uci, 0.0)

            # Normalize if needed
            total_prob = sum(policy.values())
            if total_prob > 0 and abs(total_prob - 1.0) > 1e-6:
                policy = {k: v / total_prob for k, v in policy.items()}

        else:
            # No stats available, uniform distribution
            uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0.0
            policy = {move.uci(): uniform_prob for move in legal_moves}

        return policy

    def play_game(
        self, max_moves: int = 200, verbose: bool = False
    ) -> Tuple[List[GameExample], str]:
        """
        Play one self-play game.

        Args:
            max_moves: Maximum moves before declaring draw
            verbose: Print game progress

        Returns:
            Tuple of (examples, result)
        """
        if max_moves is None:
            max_moves = self.config.max_moves

        board = chess.Board()
        examples = []
        move_count = 0

        if verbose:
            print("\n🎮 Starting self-play game...")

        try:
            while not board.is_game_over() and move_count < max_moves:
                move_count += 1

                # Adjust temperature (exploration vs exploitation)
                if move_count < self.config.temperature_threshold:
                    temp = self.config.temperature
                else:
                    temp = self.config.late_game_temperature

                self.mcts.temperature = temp

                try:
                    # Run MCTS
                    move, stats = self.mcts.search(board, return_stats=True)

                    if stats is None or move not in board.legal_moves:
                        raise ValueError(f"MCTS returned illegal move: {move}")

                    # Extract complete policy
                    policy = self._extract_policy(board, stats)

                    # Store example (value will be filled in later)
                    example = GameExample(
                        fen=board.fen(),
                        policy=policy,
                        value=0.0,
                        move_number=move_count,
                    )
                    examples.append(example)

                    if verbose and move_count % 10 == 0:
                        print(f"   Move {move_count}: {move.uci()}")

                    # Make move
                    board.push(move)

                except Exception as e:
                    print(f"⚠️  Error at move {move_count}: {e}")
                    result = "1/2-1/2"
                    break

            # Determine game result
            if board.is_game_over():
                result = board.result()
            else:
                result = "1/2-1/2"  # Max moves reached

            # Fill in values based on game outcome
            outcome = self._parse_outcome(result)
            self._assign_values(examples, outcome)

            if verbose:
                print(f"   Game over: {result}")
                print(f"   Collected {len(examples)} training examples")

            return examples, result

        except Exception as e:
            print(f"❌ Fatal error in play_game: {e}")
            if examples:
                result = "1/2-1/2"
                outcome = 0.0
                self._assign_values(examples, outcome)
                return examples, result
            else:
                return [], "1/2-1/2"

    def _parse_outcome(self, result: str) -> float:
        """Parse game result to value"""
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        else:
            return 0.0

    def _assign_values(self, examples: List[GameExample], outcome: float):
        """Assign values based on actual turn in each position"""
        for example in examples:
            board = chess.Board(example.fen)
            # Value from perspective of player to move
            if board.turn == chess.WHITE:
                example.value = outcome
            else:
                example.value = -outcome

    def play_games(
        self,
        num_games: int,
        buffer: Optional[ReplayBuffer] = None,
        save_path: Optional[str] = None,
    ) -> List[GameExample]:
        """
        Play multiple self-play games sequentially.

        Args:
            num_games: Number of games to play
            buffer: Optional ReplayBuffer to add examples to
            save_path: Optional path to save examples (if no buffer provided)

        Returns:
            List of all training examples
        """
        all_examples = []
        stats = SelfPlayStatistics()

        print(f"\n🎮 Playing {num_games} self-play games (sequential)...")
        print(f"   MCTS simulations: {self.config.num_simulations}")
        print(f"   Temperature threshold: {self.config.temperature_threshold}")

        start_time = time.time()

        for game_num in tqdm(range(num_games), desc="Self-play"):
            try:
                examples, result = self.play_game(verbose=False)
                all_examples.extend(examples)
                stats.update(result, len(examples), len(examples))

                # Add to buffer if provided
                if buffer is not None:
                    buffer.add_game_examples(examples)

            except Exception as e:
                print(f"\n⚠️  Error in game {game_num + 1}: {e}")
                stats.errors += 1

        elapsed = time.time() - start_time

        print(f"\n✅ Self-play complete!")
        print(f"   Time: {elapsed/60:.1f} minutes")
        print(f"   Avg time/game: {elapsed/num_games:.1f}s")

        stats.print_summary()

        # Save if path provided and no buffer
        if save_path and buffer is None:
            self._save_examples(all_examples, save_path)

        return all_examples

    def _save_examples(self, examples: List[GameExample], filepath: str):
        """Save training examples to disk"""
        # Convert to serializable format
        serializable_examples = [
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
            pickle.dump(serializable_examples, f)

        print(f"\n💾 Saved {len(examples)} examples to {filepath}")

    @staticmethod
    def load_examples(filepath: str) -> List[Dict]:
        """Load training examples from disk"""
        with open(filepath, "rb") as f:
            examples = pickle.load(f)
        print(f"📂 Loaded {len(examples)} examples from {filepath}")
        return examples


# ============================================================================
# PARALLEL SELF-PLAY IMPLEMENTATION
# ============================================================================


def _play_single_game_worker(
    model_path: str, config_dict: Dict, game_seed: int
) -> Tuple[List[Dict], str]:
    """
    Worker function for parallel game execution.

    Args:
        model_path: Path to saved model checkpoint
        config_dict: Configuration dictionary
        game_seed: Random seed for this game

    Returns:
        Tuple of (serialized_examples, result)
    """
    # Set random seed
    np.random.seed(game_seed)
    torch.manual_seed(game_seed)

    # Load model
    device = torch.device("cpu")
    model = HybridChessNet(use_rnn=config_dict["use_rnn"])

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Create config and worker
    config = SelfPlayConfig(**config_dict)
    worker = SelfPlayWorker(model=model, device=device, config=config)

    # Play game
    examples, result = worker.play_game(verbose=False)

    # Serialize examples
    serialized = [
        {
            "fen": ex.fen,
            "policy": ex.policy,
            "value": ex.value,
            "move_number": ex.move_number,
        }
        for ex in examples
    ]

    return serialized, result


class ParallelSelfPlayWorker:
    """
    Worker for generating self-play games in parallel.

    Uses multiprocessing to play multiple games simultaneously.
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[SelfPlayConfig] = None,
        num_workers: Optional[int] = None,
    ):
        """
        Initialize parallel self-play worker.

        Args:
            model_path: Path to saved model checkpoint
            config: Self-play configuration
            num_workers: Number of parallel workers
        """
        self.model_path = model_path
        self.config = config or SelfPlayConfig()

        if num_workers is None:
            self.num_workers = max(1, cpu_count() - 1)
        else:
            self.num_workers = num_workers

        print(f"🔧 Initialized parallel worker with {self.num_workers} processes")

    def play_games_parallel(
        self,
        num_games: int,
        buffer: Optional[ReplayBuffer] = None,
        save_path: Optional[str] = None,
    ) -> List[GameExample]:
        """
        Play multiple games in parallel.

        Args:
            num_games: Number of games to play
            buffer: Optional ReplayBuffer to add examples to
            save_path: Optional path to save examples (if no buffer provided)

        Returns:
            List of all training examples
        """
        all_examples = []
        stats = SelfPlayStatistics()

        print(f"\n🎮 Playing {num_games} self-play games (parallel)...")
        print(f"   Workers: {self.num_workers}")
        print(f"   MCTS simulations: {self.config.num_simulations}")

        start_time = time.time()

        # Create config dictionary
        config_dict = {
            "num_simulations": self.config.num_simulations,
            "c_puct": self.config.c_puct,
            "temperature": self.config.temperature,
            "temperature_threshold": self.config.temperature_threshold,
            "max_moves": self.config.max_moves,
            "use_rnn": self.config.use_rnn,
            "late_game_temperature": self.config.late_game_temperature,
        }

        # Create arguments
        game_args = [
            (self.model_path, config_dict, game_num) for game_num in range(num_games)
        ]

        # Play games in parallel
        with Pool(processes=self.num_workers) as pool:
            results = list(
                tqdm(
                    pool.starmap(_play_single_game_worker, game_args),
                    total=num_games,
                    desc="Self-play (parallel)",
                )
            )

        # Process results
        for serialized_examples, result in results:
            examples = [
                GameExample(
                    fen=example["fen"],
                    policy=example["policy"],
                    value=example["value"],
                    move_number=example["move_number"],
                )
                for example in serialized_examples
            ]

            all_examples.extend(examples)
            stats.update(result, len(examples), len(examples))

            # Add to buffer if provided
            if buffer is not None:
                buffer.add_game_examples(examples)

        elapsed = time.time() - start_time

        print(f"\n✅ Parallel self-play complete!")
        print(f"   Time: {elapsed/60:.1f} minutes")
        print(f"   Avg time/game: {elapsed/num_games:.1f}s")
        print(f"   Speedup: ~{self.num_workers * 0.7:.1f}x (estimated)")

        stats.print_summary()

        # Save if path provided and no buffer
        if save_path and buffer is None:
            self._save_examples(all_examples, save_path)

        return all_examples

    def _save_examples(self, examples: List[GameExample], filepath: str):
        """Save examples to disk"""
        import pickle

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

        print(f"\n💾 Saved {len(examples)} examples to {filepath}")


# ============================================================================
# TEST AND MAIN
# ============================================================================


def test_self_play():
    """Test self-play with ReplayBuffer integration"""
    print("=" * 80)
    print("TESTING SELF-PLAY WITH REPLAY BUFFER INTEGRATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Create model
    print("\nCreating model...")
    model = HybridChessNet(cnn_residual_blocks=3, use_rnn=False)
    model.to(device)
    model.eval()

    # Test Self-play with ReplayBuffer
    print("\n1. Testing self-play with ReplayBuffer...")
    config = SelfPlayConfig(num_simulations=50, max_moves=50)
    worker = SelfPlayWorker(model=model, device=device, config=config)

    # Create replay buffer
    buffer = ReplayBuffer(max_size=1000, memory_efficient=True)

    # Play games and add to buffer
    examples = worker.play_games(num_games=3, buffer=buffer)

    print(f"\n   Buffer size: {len(buffer)}")
    print(f"   Total examples collected: {len(examples)}")

    # Test sampling from buffer
    print("\n2. Testing buffer sampling...")
    batch = buffer.sample(batch_size=5)
    print(f"   Sampled {len(batch['boards'])} examples")
    print(f"   First board: {batch['boards'][0].fen()[:50]}...")

    # Test buffer statistics
    print("\n3. Testing buffer statistics...")
    stats = buffer.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value:.3f}")

    # Test save/load
    print("\n4. Testing buffer save/load...")
    buffer.save("data/selfplay/test_buffer.pkl")

    buffer2 = ReplayBuffer()
    buffer2.load("data/selfplay/test_buffer.pkl")
    print(f"   Loaded buffer size: {len(buffer2)}")

    print("\n" + "=" * 80)
    print("✅ SELF-PLAY + REPLAY BUFFER TESTS COMPLETE")
    print("=" * 80)


def main():
    """Main self-play script"""
    parser = argparse.ArgumentParser(
        description="Generate self-play games with ReplayBuffer integration"
    )
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--games", type=int, default=100, help="Number of games")
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations")
    parser.add_argument("--output", type=str, required=True, help="Output path")
    parser.add_argument("--use-rnn", action="store_true", help="Model uses RNN")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument(
        "--parallel", action="store_true", help="Use parallel execution"
    )
    parser.add_argument("--workers", type=int, default=None, help="Number of workers")
    parser.add_argument("--c-puct", type=float, default=DEFAULT_C_PUCT)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--temp-threshold", type=int, default=DEFAULT_TEMPERATURE_THRESHOLD
    )
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES)
    parser.add_argument("--use-buffer", action="store_true", help="Use ReplayBuffer")

    args = parser.parse_args()

    config = SelfPlayConfig.from_args(args)

    if args.use_buffer:
        # Use ReplayBuffer
        buffer = ReplayBuffer(max_size=100000, memory_efficient=True)
        print("📦 Using ReplayBuffer for storage")
    else:
        buffer = None

    if args.parallel:
        # Parallel execution
        worker = ParallelSelfPlayWorker(
            model_path=args.model, config=config, num_workers=args.workers
        )
        examples = worker.play_games_parallel(
            num_games=args.games,
            buffer=buffer,
            save_path=args.output if not buffer else None,
        )
    else:
        # Sequential execution
        device = torch.device(
            "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        model = HybridChessNet(use_rnn=args.use_rnn)
        checkpoint = torch.load(args.model, map_location=device)

        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        worker = SelfPlayWorker(model=model, device=device, config=config)
        examples = worker.play_games(
            num_games=args.games,
            buffer=buffer,
            save_path=args.output if not buffer else None,
        )

    # Save buffer if used
    if buffer:
        buffer.save(args.output)

    print(f"\n🎉 Generated {len(examples)} training examples!")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        test_self_play()
    else:
        main()
