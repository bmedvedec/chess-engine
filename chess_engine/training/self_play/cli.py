"""
SELF-PLAY - CLI Entrypoint

Usage:
    python -m chess_engine.training.self_play.cli --model <path> --games <n> --output <path>
"""

import argparse
import torch

from chess_engine.data.replay.buffer import ReplayBuffer
from chess_engine.models.hybrid.config import HybridModelConfig
from chess_engine.models.hybrid.hybrid_net import HybridChessNet
from chess_engine.training.self_play.config import (
    SelfPlayConfig,
    DEFAULT_C_PUCT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TEMPERATURE_THRESHOLD,
    DEFAULT_MAX_MOVES,
    DEFAULT_DIRICHLET_ALPHA,
    DEFAULT_RESIGN_THRESHOLD,
)
from chess_engine.training.self_play.game_runner import SelfPlayGameRunner
from chess_engine.training.self_play.parallel import ParallelSelfPlay


def main():
    """Main self-play script."""
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
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=DEFAULT_DIRICHLET_ALPHA,
        help="Dirichlet noise alpha",
    )
    parser.add_argument(
        "--resign-threshold",
        type=float,
        default=DEFAULT_RESIGN_THRESHOLD,
        help="Resign threshold",
    )

    args = parser.parse_args()

    config = SelfPlayConfig.from_args(args)

    if args.use_buffer:
        buffer = ReplayBuffer(max_size=100000, memory_efficient=True)
        print("Using ReplayBuffer for storage")
    else:
        buffer = None

    if args.parallel:
        worker = ParallelSelfPlay(
            model_path=args.model, config=config, num_workers=args.workers
        )
        examples = worker.play_games_parallel(
            num_games=args.games,
            buffer=buffer,
            save_path=args.output if not buffer else None,
        )
    else:
        device = torch.device(
            "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        checkpoint = torch.load(args.model, map_location=device)

        model_config = HybridModelConfig.from_dict(checkpoint["model_config"])

        model = HybridChessNet(model_config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        worker = SelfPlayGameRunner(model=model, device=device, config=config)
        examples = worker.play_games(
            num_games=args.games,
            buffer=buffer,
            save_path=args.output if not buffer else None,
        )

    if buffer:
        buffer.save(args.output)

    print(f"\nGenerated {len(examples)} training examples!")


if __name__ == "__main__":
    main()
