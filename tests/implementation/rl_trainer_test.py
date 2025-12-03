#!/usr/bin/env python3
"""
Quick Test Script for RL Training Loop

This script runs a minimal test of the RL training system with reduced
parameters for fast verification.

Usage:
    python test_rl_training.py

Or with custom settings:
    python test_rl_training.py --iterations 5 --games 20
"""

import sys
import torch
from chess_engine.training.rl_trainer import RLTrainer, RLTrainingConfig
from chess_engine.models.hybrid_model import HybridChessNet


def test_basic():
    """Basic functionality test"""
    print("=" * 80)
    print("RL TRAINING LOOP - BASIC TEST")
    print("=" * 80)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    # Create minimal configuration for testing
    config = RLTrainingConfig(
        # Minimal iterations for quick test
        num_iterations=3,
        games_per_iteration=10,
        training_steps_per_iteration=50,
        # Reduced MCTS for speed
        num_simulations=50,
        max_moves_per_game=50,
        # Small model for speed
        cnn_blocks=3,
        use_rnn=False,
        # Small batch and buffer
        batch_size=32,
        buffer_size=5000,
        min_buffer_size=100,
        # Evaluate every iteration (for testing)
        eval_frequency=1,
        eval_games=4,
        eval_simulations=30,
        # Use parallel if available
        use_parallel_selfplay=False,  # Sequential for testing
        # Paths
        checkpoint_dir="data/test_rl_checkpoints",
        log_dir="logs/test_rl_training",
        # Device
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\n📋 Test Configuration:")
    print(f"   Iterations: {config.num_iterations}")
    print(f"   Games per iteration: {config.games_per_iteration}")
    print(f"   MCTS simulations: {config.num_simulations}")
    print(f"   Model: {config.cnn_blocks} CNN blocks")

    # Create model
    print(f"\n🏗️  Creating model...")
    model = HybridChessNet(
        cnn_residual_blocks=config.cnn_blocks, use_rnn=config.use_rnn
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")

    # Create trainer
    print(f"\n🚀 Creating trainer...")
    trainer = RLTrainer(config=config, model=model)

    # Run training
    print(f"\n▶️  Starting training...\n")

    try:
        trainer.train()
        print("\n" + "=" * 80)
        print("✅ TEST PASSED - RL Training Loop Works!")
        print("=" * 80)
        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_checkpoint_resume():
    """Test checkpoint saving and resuming"""
    print("\n" + "=" * 80)
    print("RL TRAINING LOOP - CHECKPOINT RESUME TEST")
    print("=" * 80)

    # First run - save checkpoint
    print("\n1️⃣  First run - training 2 iterations...")

    config = RLTrainingConfig(
        num_iterations=2,
        games_per_iteration=5,
        training_steps_per_iteration=20,
        num_simulations=30,
        cnn_blocks=2,
        batch_size=16,
        buffer_size=1000,
        min_buffer_size=50,
        eval_frequency=999,  # Skip evaluation
        checkpoint_dir="data/test_resume_checkpoints",
        log_dir="logs/test_resume_training",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    trainer = RLTrainer(config=config)

    # Train for 2 iterations
    try:
        # Temporarily modify to train only 2 iterations
        trainer.config.num_iterations = 2
        trainer.train()

        print("\n✅ First run complete")

        # Second run - resume from checkpoint
        print("\n2️⃣  Second run - resuming from checkpoint...")

        checkpoint_path = "data/test_resume_checkpoints/latest.pt"
        trainer2 = RLTrainer(config=config, resume_from=checkpoint_path)

        # Train for 1 more iteration
        trainer2.config.num_iterations = 3  # Total of 3
        trainer2.train()

        print("\n✅ Resume successful")

        # Verify
        assert (
            trainer2.current_iteration >= 2
        ), "Should have completed at least 2 iterations"

        print("\n" + "=" * 80)
        print("✅ CHECKPOINT RESUME TEST PASSED")
        print("=" * 80)
        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ CHECKPOINT RESUME TEST FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_parallel_selfplay():
    """Test parallel self-play"""
    print("\n" + "=" * 80)
    print("RL TRAINING LOOP - PARALLEL SELF-PLAY TEST")
    print("=" * 80)

    import multiprocessing

    num_cores = multiprocessing.cpu_count()
    print(f"\n🖥️  Available CPU cores: {num_cores}")

    if num_cores < 2:
        print("⚠️  Need at least 2 cores for parallel test, skipping...")
        return True

    config = RLTrainingConfig(
        num_iterations=2,
        games_per_iteration=8,  # Multiple of num_workers
        training_steps_per_iteration=20,
        num_simulations=30,
        cnn_blocks=2,
        batch_size=16,
        buffer_size=1000,
        min_buffer_size=50,
        eval_frequency=999,  # Skip evaluation
        use_parallel_selfplay=True,
        num_workers=min(2, num_cores),  # Use 2 workers
        checkpoint_dir="data/test_parallel_checkpoints",
        log_dir="logs/test_parallel_training",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    print(f"\n📋 Using {config.num_workers} parallel workers")

    try:
        trainer = RLTrainer(config=config)
        trainer.train()

        print("\n" + "=" * 80)
        print("✅ PARALLEL SELF-PLAY TEST PASSED")
        print("=" * 80)
        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ PARALLEL SELF-PLAY TEST FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("RUNNING ALL RL TRAINING TESTS")
    print("=" * 80)

    results = {}

    # Test 1: Basic functionality
    print("\n" + "=" * 80)
    print("TEST 1/3: Basic Functionality")
    print("=" * 80)
    results["basic"] = test_basic()

    # Test 2: Checkpoint resume
    print("\n" + "=" * 80)
    print("TEST 2/3: Checkpoint Resume")
    print("=" * 80)
    results["checkpoint"] = test_checkpoint_resume()

    # Test 3: Parallel self-play
    print("\n" + "=" * 80)
    print("TEST 3/3: Parallel Self-Play")
    print("=" * 80)
    results["parallel"] = test_parallel_selfplay()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.capitalize()}: {status}")

    all_passed = all(results.values())

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("=" * 80)

    return all_passed


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Test RL Training Loop")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--basic", action="store_true", help="Run basic test only")
    parser.add_argument(
        "--checkpoint", action="store_true", help="Run checkpoint test only"
    )
    parser.add_argument(
        "--parallel", action="store_true", help="Run parallel test only"
    )
    parser.add_argument("--iterations", type=int, help="Override iterations")
    parser.add_argument("--games", type=int, help="Override games per iteration")

    args = parser.parse_args()

    # If no specific test selected, run basic test
    if not (args.all or args.basic or args.checkpoint or args.parallel):
        args.basic = True

    # Run selected tests
    if args.all:
        success = run_all_tests()
    else:
        if args.basic:
            success = test_basic()
        elif args.checkpoint:
            success = test_checkpoint_resume()
        elif args.parallel:
            success = test_parallel_selfplay()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
