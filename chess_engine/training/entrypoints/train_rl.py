"""
RL TRAINER - CLI Entrypoint

Usage:
    python -m chess_engine.training.entrypoints.train_rl --iterations 100 --games-per-iter 100
"""

import os

from chess_engine.training.rl.rl_config import RLTrainingConfig
from chess_engine.training.rl.rl_trainer import RLTrainer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN messages

import argparse


def main():
    """Main entry point for RL training"""

    parser = argparse.ArgumentParser(description="RL Training Loop for Chess Engine")

    # Training parameters
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of training iterations"
    )
    parser.add_argument(
        "--games-per-iter", type=int, default=100, help="Games per iteration"
    )
    parser.add_argument(
        "--training-steps", type=int, default=1000, help="Training steps per iteration"
    )

    # Self-play parameters
    parser.add_argument("--simulations", type=int, default=None, help="MCTS simulations (default: from RLTrainingConfig)")
    parser.add_argument(
        "--c-puct",
        type=float,
        default=None,
        help="MCTS exploration constant (default: from RLTrainingConfig)",
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use parallel self-play (default: True). Pass --no-parallel to force sequential.",
    )
    parser.add_argument(
        "--workers", type=int, default=12, help="Number of parallel workers"
    )

    # Model parameters
    parser.add_argument("--cnn-blocks", type=int, default=10, help="CNN blocks")
    parser.add_argument("--use-rnn", action="store_true", help="Use RNN")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: from RLTrainingConfig)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (default: from RLTrainingConfig)")
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default=None,
        choices=["constant", "step", "cosine", "plateau"],
        help="LR schedule: constant|step|cosine|plateau (default: from RLTrainingConfig)",
    )
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=None,
        help="StepLR: decay every N iters; Plateau: patience (default: from RLTrainingConfig)",
    )
    parser.add_argument(
        "--lr-decay-gamma",
        type=float,
        default=None,
        help="Multiplicative LR decay factor (default: from RLTrainingConfig)",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=None,
        help="Minimum LR floor for cosine/plateau (default: from RLTrainingConfig)",
    )
    parser.add_argument(
        "--buffer-size", type=int, default=None, help="Replay buffer size (default: from RLTrainingConfig)"
    )

    # Evaluation
    parser.add_argument("--eval-freq", type=int, default=5, help="Evaluation frequency")
    parser.add_argument("--eval-games", type=int, default=50, help="Evaluation games (default: 50 — minimum for statistical significance)")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.5,
        help="Temperature for move selection (default: 1.5)",
    )

    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=None,
        help="Dirichlet noise alpha for root exploration (default: from RLTrainingConfig)",
    )

    parser.add_argument(
        "--resign-threshold",
        type=float,
        default=None,
        help="Resign if MCTS root value drops below this for 4 consecutive own-side moves (default: from RLTrainingConfig)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="data/rl_checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )

    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="Initialize from supervised model weights",
    )

    # Device
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument(
        "--no-amp", action="store_true", help="Disable mixed precision training (AMP)"
    )

    args = parser.parse_args()

    # Start from RLTrainingConfig defaults, then apply only the CLI args that
    # were explicitly provided (non-None).  This means rl_config.py is the
    # authoritative source for any flag not passed on the command line.
    config = RLTrainingConfig(
        num_iterations=args.iterations,
        games_per_iteration=args.games_per_iter,
        training_steps_per_iteration=args.training_steps,
        temperature=args.temperature,
        use_parallel_selfplay=args.parallel,
        # None-guarded: only override rl_config.py when explicitly passed on CLI
        **({"num_simulations": args.simulations} if args.simulations is not None else {}),
        **({"c_puct": args.c_puct} if args.c_puct is not None else {}),
        **({"dirichlet_alpha": args.dirichlet_alpha} if args.dirichlet_alpha is not None else {}),
        **({"resign_threshold": args.resign_threshold} if args.resign_threshold is not None else {}),
        num_workers=args.workers,
        cnn_blocks=args.cnn_blocks,
        use_rnn=args.use_rnn,
        eval_frequency=args.eval_freq,
        eval_games=args.eval_games,
        checkpoint_dir=args.checkpoint_dir,
        device="cpu" if args.cpu else "cuda",
        use_amp=not args.no_amp,
        # Optional overrides — only applied when explicitly passed on CLI:
        **({"batch_size": args.batch_size} if args.batch_size is not None else {}),
        **({"learning_rate": args.lr} if args.lr is not None else {}),
        **({"lr_schedule": args.lr_schedule} if args.lr_schedule is not None else {}),
        **({"lr_decay_steps": args.lr_decay_steps} if args.lr_decay_steps is not None else {}),
        **({"lr_decay_gamma": args.lr_decay_gamma} if args.lr_decay_gamma is not None else {}),
        **({"lr_min": args.lr_min} if args.lr_min is not None else {}),
        **({"buffer_size": args.buffer_size} if args.buffer_size is not None else {}),
    )

    trainer = RLTrainer(
        config=config, resume_from=args.resume, pretrained_path=args.pretrained_model
    )
    trainer.train()


if __name__ == "__main__":
    main()
