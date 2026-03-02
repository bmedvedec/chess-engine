"""
SELF-PLAY - Parallel Execution (ParallelSelfPlay)

Plays self-play games in parallel using multiprocessing.
"""

import os
import time
import pickle
import signal
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool, cpu_count

import torch
from tqdm import tqdm

from chess_engine.models.hybrid.config import HybridModelConfig
from chess_engine.models.hybrid.hybrid_net import HybridChessNet
from chess_engine.data.replay.buffer import ReplayBuffer
from chess_engine.data.replay.storage import GameExample
from chess_engine.training.self_play.config import SelfPlayConfig
from chess_engine.training.self_play.stats import SelfPlayStatistics
from chess_engine.training.self_play.game_runner import SelfPlayGameRunner


def _play_single_game_worker(
    model_path: str, config_dict: dict, game_num: int
) -> Tuple[List[Dict], str, bool]:
    """Worker function for parallel game execution."""
    # IMPORTANT: Ignore keyboard interrupts in worker processes
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Re-initialize model in worker process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_config = HybridModelConfig(
        cnn_input_channels=20,
        cnn_filters=config_dict["cnn_filters"],
        cnn_residual_blocks=config_dict["cnn_blocks"],
        cnn_dropout=config_dict.get("cnn_dropout", 0.0),
        use_rnn=config_dict["use_rnn"],
        rnn_hidden_size=config_dict.get("rnn_hidden_size", 256),
        rnn_num_layers=config_dict.get("rnn_layers", 2),
        rnn_dropout=config_dict.get("rnn_dropout", 0.0),
        rnn_use_attention=config_dict.get("rnn_use_attention", False),
        rnn_bidirectional=config_dict.get("rnn_bidirectional", False),
        fusion_type=config_dict.get("fusion_type", "gated"),
        num_actions=config_dict["num_actions"],
    )

    model = HybridChessNet(model_config).to(device)

    state = torch.load(model_path, map_location=device)
    state_dict = state["model_state_dict"] if "model_state_dict" in state else state
    model.load_state_dict(state_dict)
    model.eval()

    # Create config
    config = SelfPlayConfig(
        num_simulations=config_dict["num_simulations"],
        c_puct=config_dict["c_puct"],
        temperature=config_dict["temperature"],
        temperature_threshold=config_dict["temperature_threshold"],
        max_moves=config_dict["max_moves"],
        use_rnn=config_dict["use_rnn"],
        late_game_temperature=config_dict["late_game_temperature"],
        dirichlet_alpha=config_dict["dirichlet_alpha"],
        resign_threshold=config_dict["resign_threshold"],
    )

    # Create worker and play game
    worker = SelfPlayGameRunner(model=model, device=device, config=config)
    examples, result, resigned = worker.play_game()

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

    return serialized, result, resigned


class ParallelSelfPlay:
    """
    Worker for parallel self-play game generation.

    Backward-compatible alias: ParallelSelfPlayWorker = ParallelSelfPlay
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[SelfPlayConfig] = None,
        num_workers: Optional[int] = None,
    ):
        """
        Initialize parallel worker.

        Args:
            model_path: Path to model checkpoint
            config: Self-play configuration
            num_workers: Number of parallel workers (default: CPU count - 1)
        """
        self.model_path = model_path
        self.config = config or SelfPlayConfig()
        self.num_workers = num_workers or max(1, cpu_count() - 1)

        print(f"Initialized parallel worker with {self.num_workers} processes")

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
            save_path: Optional path to save examples (if no buffer)

        Returns:
            List of all training examples
        """
        all_examples = []
        stats = SelfPlayStatistics()

        print(f"\nPlaying {num_games} self-play games (parallel)...")
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
            "dirichlet_alpha": self.config.dirichlet_alpha,
            "resign_threshold": self.config.resign_threshold,
        }

        # Create arguments
        game_args = [
            (self.model_path, config_dict, game_num) for game_num in range(num_games)
        ]

        # Play games in parallel
        with Pool(processes=self.num_workers) as pool:
            try:
                results = list(
                    tqdm(
                        pool.starmap(_play_single_game_worker, game_args),
                        total=num_games,
                        desc="Self-play (parallel)",
                    )
                )
            except KeyboardInterrupt:
                print("\nStopping workers...")
                pool.terminate()
                pool.join()
                raise

        # Process results
        for serialized_examples, result, resigned in results:
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
            stats.update(result, len(examples), len(examples), resigned)

            if buffer is not None:
                buffer.add_game_examples(examples)

        elapsed = time.time() - start_time

        print(f"\nParallel self-play complete!")
        print(f"   Time: {elapsed/60:.1f} minutes")
        print(f"   Avg time/game: {elapsed/num_games:.1f}s")
        print(f"   Speedup: ~{self.num_workers * 0.7:.1f}x (estimated)")

        stats.print_summary()

        if save_path and buffer is None:
            self._save_examples(all_examples, save_path)

        return all_examples

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
ParallelSelfPlayWorker = ParallelSelfPlay
