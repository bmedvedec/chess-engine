"""
SUPERVISED TRAINING - CLI Entrypoint

Usage:
    python -m chess_engine.training.entrypoints.train_supervised --data <path> --epochs <n>
"""

import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN messages

import sys
import argparse
import glob

import torch
from tqdm import tqdm

from chess_engine.training.trainer import ChessTrainer
from chess_engine.models.cnn.utils import count_parameters
from chess_engine.models.hybrid.hybrid_net import HybridChessNet
from chess_engine.models.hybrid.config import HybridModelConfig
from chess_engine.data.dataset import (
    ChessDataset,
    load_dataset,
    create_dataloader,
    split_examples,
)
from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder


def load_chunked_data(chunk_dir: str) -> list:
    """
    Load all chunks from a directory and combine them.

    Args:
        chunk_dir: Directory containing chunk_XXXX.pkl files

    Returns:
        Combined list of all examples from all chunks
    """
    print(f"\nLoading chunked data from: {chunk_dir}")

    # Find all chunk files
    chunk_pattern = os.path.join(chunk_dir, "chunk_*.pkl")
    chunk_files = sorted(glob.glob(chunk_pattern))

    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {chunk_dir}")

    print(f"   Found {len(chunk_files)} chunk files")

    # Load all chunks
    all_examples = []

    with tqdm(total=len(chunk_files), desc="Loading chunks", unit="chunk") as pbar:
        for chunk_file in chunk_files:
            try:
                chunk_examples = load_dataset(chunk_file)
                all_examples.extend(chunk_examples)
                pbar.update(1)
                pbar.set_postfix({"total_positions": len(all_examples)})
            except Exception as e:
                print(f"Warning: Failed to load {chunk_file}: {e}")
                continue

    print(f"Loaded {len(all_examples):,} total examples from {len(chunk_files)} chunks")

    return all_examples


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train chess engine")
    # Required arguments
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data (.pkl file or directory with chunks)",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split")

    # Model architecture arguments
    parser.add_argument(
        "--cnn-blocks", type=int, default=10, help="Number of CNN residual blocks"
    )
    parser.add_argument(
        "--cnn-filters", type=int, default=256, help="Number of CNN filters"
    )
    parser.add_argument(
        "--cnn-dropout", type=float, default=0.0, help="CNN dropout rate"
    )
    parser.add_argument("--use-rnn", action="store_true", help="Use RNN (hybrid model)")
    parser.add_argument(
        "--rnn-hidden-size", type=int, default=256, help="RNN hidden size"
    )
    parser.add_argument(
        "--rnn-layers", type=int, default=2, help="Number of RNN layers"
    )
    parser.add_argument(
        "--rnn-dropout", type=float, default=0.3, help="RNN dropout rate"
    )
    parser.add_argument(
        "--rnn-attention", action="store_true", help="Use attention in RNN"
    )
    parser.add_argument(
        "--fusion-type",
        type=str,
        default="gated",
        choices=["concat", "gated", "attention"],
        help="Feature fusion type",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Resume from checkpoint"
    )

    # Training options
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training",
    )
    parser.add_argument(
        "--policy-weight", type=float, default=1.0, help="Weight for policy loss"
    )
    parser.add_argument(
        "--value-weight", type=float, default=1.0, help="Weight for value loss"
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=10,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=2, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from: {args.data}")

    # Check if input is a directory (chunked) or file (single dataset)
    if os.path.isdir(args.data):
        # Chunked data - load all chunks
        examples = load_chunked_data(args.data)
    else:
        # Single file
        examples = load_dataset(args.data)

    # Split train/val using split_examples for proper shuffling and reproducibility
    train_examples, val_examples = split_examples(
        examples,
        train_ratio=1.0 - args.val_split,
        shuffle=True,
        seed=42,  # Reproducibility
    )

    print(f"Training examples: {len(train_examples):,}")
    print(f"Validation examples: {len(val_examples):,}")

    # Create datasets
    board_encoder = BoardEncoder()
    move_encoder = MoveEncoder()

    # Training dataset: Enable caching and augmentation
    train_dataset = ChessDataset(
        train_examples,
        board_encoder,
        move_encoder,
        cache_tensors=True,
        augment=True,
    )

    # Validation dataset: Enable caching, disable augmentation
    val_dataset = ChessDataset(
        val_examples,
        board_encoder,
        move_encoder,
        cache_tensors=True,
        augment=False,
    )

    # Create dataloaders
    # Note: num_workers=0 on Windows to avoid multiprocessing issues with cached tensors
    train_loader = create_dataloader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # Create model
    print(f"\nCreating model...")
    model = HybridChessNet(
        config=HybridModelConfig(
            # CNN parameters
            cnn_residual_blocks=args.cnn_blocks,
            cnn_filters=args.cnn_filters,
            cnn_dropout=args.cnn_dropout,
            # RNN parameters
            rnn_hidden_size=args.rnn_hidden_size,
            rnn_num_layers=args.rnn_layers,
            rnn_dropout=args.rnn_dropout,
            rnn_use_attention=args.rnn_attention,
            # Fusion parameters
            fusion_type=args.fusion_type,
            # Mode
            use_rnn=args.use_rnn,
        )
    )

    print(f"Model: {'Hybrid CNN-RNN' if args.use_rnn else 'CNN-only'}")
    print(f"  CNN: {args.cnn_blocks} blocks, {args.cnn_filters} filters")
    if args.use_rnn:
        print(f"  RNN: {args.rnn_layers} layers, {args.rnn_hidden_size} hidden size")
        print(f"  RNN: dropout={args.rnn_dropout}, attention={args.rnn_attention}")
        print(f"  Fusion: {args.fusion_type}")
    print(f"Parameters: {count_parameters(model):,}")

    # Create trainer
    trainer = ChessTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        use_mixed_precision=not args.no_mixed_precision,
        policy_weight=args.policy_weight,
        value_weight=args.value_weight,
        early_stopping_patience=args.early_stopping,
        warmup_epochs=args.warmup_epochs,
        gradient_accumulation_steps=args.grad_accum,
    )

    # Resume from checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Train
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    # If run without arguments, use test mode
    if len(sys.argv) == 1:
        print("Running in TEST mode with sample data...")
        print(
            "For real training, use: python train_supervised.py --data <path> --epochs <n>"
        )
        print("\nTesting with sample data...")

        # Create sample data if needed
        from chess_engine.data.dataset import (
            ChessGameParser,
            save_dataset,
        )

        pgn_path = "data/pgn/sample_games.pgn"
        data_path = "data/processed/test_train_data.pkl"

        if not os.path.exists(data_path):
            parser = ChessGameParser(min_elo=1500)
            examples = parser.parse_pgn_file(pgn_path)
            save_dataset(examples, data_path)

        # Test training with sample data
        sys.argv = [
            "train_supervised.py",
            "--data",
            data_path,
            "--epochs",
            "2",
            "--batch-size",
            "32",
            "--cnn-blocks",
            "3",
        ]

        main()
    else:
        main()
