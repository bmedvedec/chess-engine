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
    parser.add_argument("--simulations", type=int, default=200, help="MCTS simulations")
    parser.add_argument(
        "--c-puct",
        type=float,
        default=2.0,
        help="MCTS exploration constant (default: 2.0)",
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Use parallel self-play"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Number of parallel workers"
    )

    # Model parameters
    parser.add_argument("--cnn-blocks", type=int, default=10, help="CNN blocks")
    parser.add_argument("--use-rnn", action="store_true", help="Use RNN")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--buffer-size", type=int, default=500000, help="Replay buffer size"
    )

    # Evaluation
    parser.add_argument("--eval-freq", type=int, default=5, help="Evaluation frequency")
    parser.add_argument("--eval-games", type=int, default=20, help="Evaluation games")

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.5,
        help="Temperature for move selection (default: 1.5)",
    )

    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.3,
        help="Dirichlet noise alpha for root exploration (default: 0.3)",
    )

    parser.add_argument(
        "--resign-threshold",
        type=float,
        default=-0.9,
        help="Resign if position value drops below this (default: -0.9)",
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

    # Create configuration
    config = RLTrainingConfig(
        num_iterations=args.iterations,
        games_per_iteration=args.games_per_iter,
        training_steps_per_iteration=args.training_steps,
        num_simulations=args.simulations,
        c_puct=getattr(args, "c_puct", 2.0),
        temperature=args.temperature,
        dirichlet_alpha=args.dirichlet_alpha,
        resign_threshold=args.resign_threshold,
        use_parallel_selfplay=args.parallel,
        num_workers=args.workers,
        cnn_blocks=args.cnn_blocks,
        use_rnn=args.use_rnn,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        eval_frequency=args.eval_freq,
        eval_games=args.eval_games,
        checkpoint_dir=args.checkpoint_dir,
        device="cpu" if args.cpu else "cuda",
        use_amp=not args.no_amp,
    )

    # Create trainer
    trainer = RLTrainer(
        config=config, resume_from=args.resume, pretrained_path=args.pretrained_model
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
