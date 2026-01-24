"""
Quick Test Script for RL Training Loop with AMP Testing

This script runs tests of the RL training system with special focus on
verifying that mixed precision (AMP) is working correctly.

Usage:
    python rl_trainer_test.py                    # Basic test
    python rl_trainer_test.py --all              # All tests
    python rl_trainer_test.py --amp              # AMP-specific test
"""

import sys
import torch
from chess_engine.training.rl_trainer import RLTrainer, RLTrainingConfig
from chess_engine.models.hybrid_model import HybridChessNet


def check_amp_available():
    """Check if AMP is available on this system"""
    print("\n" + "=" * 80)
    print("CHECKING AMP AVAILABILITY")
    print("=" * 80)

    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA available: {cuda_available}")

    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        compute_cap = torch.cuda.get_device_capability(0)
        print(f"Compute capability: {compute_cap[0]}.{compute_cap[1]}")

        # Check if AMP is supported (Compute capability >= 6.0)
        amp_supported = compute_cap[0] >= 6
        print(f"AMP supported: {amp_supported} {'✅' if amp_supported else '❌'}")

        if not amp_supported:
            print("⚠️  AMP requires compute capability >= 6.0")
            print("   AMP will be automatically disabled")

        return amp_supported
    else:
        print("⚠️  No CUDA device found - AMP not available")
        return False


def test_amp_imports():
    """Test that AMP imports work correctly"""
    print("\n" + "=" * 80)
    print("TEST: AMP IMPORTS")
    print("=" * 80)

    try:
        # Try new API (PyTorch 2.0+)
        from torch.amp import autocast_mode
        from torch.amp.grad_scaler import GradScaler

        print("\n✅ Using PyTorch 2.0+ API (torch.amp)")
        api_version = "2.0+"
    except (ImportError, AttributeError):
        # Fall back to old API (PyTorch 1.6-1.13)
        from torch.cuda.amp import autocast, GradScaler

        print("\n✅ Using PyTorch 1.x API (torch.cuda.amp)")
        api_version = "1.x"

    print(f"PyTorch version: {torch.__version__}")
    print(f"API version: {api_version}")

    return True


def test_amp_basic():
    """Test basic AMP functionality"""
    print("\n" + "=" * 80)
    print("TEST: BASIC AMP FUNCTIONALITY")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available, skipping AMP test")
        return True

    device = torch.device("cuda")

    # Test with minimal config
    config = RLTrainingConfig(
        num_iterations=1,
        games_per_iteration=5,
        training_steps_per_iteration=10,
        num_simulations=20,
        max_moves_per_game=20,
        cnn_blocks=2,
        use_rnn=False,
        batch_size=8,
        buffer_size=500,
        min_buffer_size=10,
        eval_frequency=999,  # Skip evaluation
        checkpoint_dir="data/test_amp_checkpoints",
        log_dir="logs/test_amp_training",
        device="cuda",
        use_amp=True,  # Enable AMP
    )

    print(f"\n📋 Configuration:")
    print(f"   Device: {config.device}")
    print(f"   AMP enabled: {config.use_amp}")
    print(f"   Iterations: {config.num_iterations}")
    print(f"   Games: {config.games_per_iteration}")

    try:
        # Create trainer
        print(f"\n🚀 Creating trainer with AMP enabled...")
        trainer = RLTrainer(config=config)

        # Check that scaler was initialized
        if trainer.scaler is None:
            print("❌ ERROR: GradScaler was not initialized!")
            return False

        print(f"✅ GradScaler initialized: {type(trainer.scaler).__name__}")
        print(f"✅ AMP mode active: {trainer.use_amp}")

        # Run training
        print(f"\n▶️  Running training with AMP...")
        trainer.train()

        print("\n✅ AMP training completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ AMP test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_amp_vs_fp32():
    """Compare AMP vs FP32 performance and memory usage"""
    print("\n" + "=" * 80)
    print("TEST: AMP vs FP32 COMPARISON")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available, skipping comparison")
        return True

    results = {}

    # Test with FP32 (no AMP)
    print("\n" + "-" * 80)
    print("Running with FP32 (AMP disabled)...")
    print("-" * 80)

    config_fp32 = RLTrainingConfig(
        num_iterations=1,
        games_per_iteration=10,
        training_steps_per_iteration=20,
        num_simulations=30,
        cnn_blocks=3,
        batch_size=16,
        buffer_size=1000,
        min_buffer_size=20,
        eval_frequency=999,
        checkpoint_dir="data/test_fp32_checkpoints",
        log_dir="logs/test_fp32_training",
        device="cuda",
        use_amp=False,  # Disable AMP
    )

    try:
        import time

        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        trainer_fp32 = RLTrainer(config=config_fp32)
        trainer_fp32.train()

        fp32_time = time.time() - start_time
        fp32_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB

        results["fp32"] = {"time": fp32_time, "memory": fp32_memory, "success": True}

        print(f"\n✅ FP32 completed:")
        print(f"   Time: {fp32_time:.1f}s")
        print(f"   Peak memory: {fp32_memory:.2f} GB")

    except Exception as e:
        print(f"❌ FP32 test failed: {e}")
        results["fp32"] = {"success": False}

    # Test with AMP
    print("\n" + "-" * 80)
    print("Running with AMP (mixed precision)...")
    print("-" * 80)

    config_amp = RLTrainingConfig(
        num_iterations=1,
        games_per_iteration=10,
        training_steps_per_iteration=20,
        num_simulations=30,
        cnn_blocks=3,
        batch_size=16,
        buffer_size=1000,
        min_buffer_size=20,
        eval_frequency=999,
        checkpoint_dir="data/test_amp_comparison_checkpoints",
        log_dir="logs/test_amp_comparison_training",
        device="cuda",
        use_amp=True,  # Enable AMP
    )

    try:
        import time

        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        trainer_amp = RLTrainer(config=config_amp)
        trainer_amp.train()

        amp_time = time.time() - start_time
        amp_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB

        results["amp"] = {"time": amp_time, "memory": amp_memory, "success": True}

        print(f"\n✅ AMP completed:")
        print(f"   Time: {amp_time:.1f}s")
        print(f"   Peak memory: {amp_memory:.2f} GB")

    except Exception as e:
        print(f"❌ AMP test failed: {e}")
        results["amp"] = {"success": False}

    # Compare results
    if results.get("fp32", {}).get("success") and results.get("amp", {}).get("success"):
        print("\n" + "=" * 80)
        print("COMPARISON RESULTS")
        print("=" * 80)

        time_speedup = results["fp32"]["time"] / results["amp"]["time"]
        memory_savings = (
            1 - results["amp"]["memory"] / results["fp32"]["memory"]
        ) * 100

        print(f"\n📊 Performance:")
        print(f"   FP32 time:  {results['fp32']['time']:.1f}s")
        print(f"   AMP time:   {results['amp']['time']:.1f}s")
        print(f"   Speedup:    {time_speedup:.2f}x {'✅' if time_speedup > 1 else '⚠️'}")

        print(f"\n💾 Memory:")
        print(f"   FP32 memory:  {results['fp32']['memory']:.2f} GB")
        print(f"   AMP memory:   {results['amp']['memory']:.2f} GB")
        print(
            f"   Savings:      {memory_savings:.1f}% {'✅' if memory_savings > 0 else '⚠️'}"
        )

        if time_speedup > 1.0 or memory_savings > 0:
            print(f"\n🎉 AMP provides benefits on your GPU!")
        else:
            print(f"\n⚠️  AMP may not provide significant benefits on this GPU")

        return True
    else:
        print("\n⚠️  Could not complete comparison")
        return False


def test_basic():
    """Basic functionality test"""
    print("\n" + "=" * 80)
    print("TEST: BASIC FUNCTIONALITY")
    print("=" * 80)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    # Create minimal configuration for testing
    config = RLTrainingConfig(
        num_iterations=2,
        games_per_iteration=8,
        training_steps_per_iteration=30,
        num_simulations=40,
        max_moves_per_game=40,
        cnn_blocks=3,
        use_rnn=False,
        batch_size=16,
        buffer_size=2000,
        min_buffer_size=50,
        eval_frequency=2,
        eval_games=4,
        eval_simulations=25,
        use_parallel_selfplay=False,
        checkpoint_dir="data/test_rl_checkpoints",
        log_dir="logs/test_rl_training",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=True if torch.cuda.is_available() else False,  # Enable AMP on CUDA
    )

    print(f"\n📋 Test Configuration:")
    print(f"   Iterations: {config.num_iterations}")
    print(f"   Games per iteration: {config.games_per_iteration}")
    print(f"   MCTS simulations: {config.num_simulations}")
    print(f"   Model: {config.cnn_blocks} CNN blocks")
    print(f"   AMP enabled: {config.use_amp}")

    try:
        print(f"\n🚀 Creating trainer...")
        trainer = RLTrainer(config=config)

        print(f"\n▶️  Starting training...\n")
        trainer.train()

        print("\n" + "=" * 80)
        print("✅ BASIC TEST PASSED")
        print("=" * 80)
        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ BASIC TEST FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_checkpoint_resume():
    """Test checkpoint saving and resuming"""
    print("\n" + "=" * 80)
    print("TEST: CHECKPOINT RESUME")
    print("=" * 80)

    config = RLTrainingConfig(
        num_iterations=2,  # Train 2 iterations first
        games_per_iteration=6,
        training_steps_per_iteration=20,
        num_simulations=30,
        cnn_blocks=2,
        batch_size=8,
        buffer_size=1000,
        min_buffer_size=30,
        eval_frequency=999,
        save_frequency=1,  # Save every iteration
        checkpoint_dir="data/test_resume_checkpoints",
        log_dir="logs/test_resume_training",
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=True if torch.cuda.is_available() else False,
    )

    try:
        # First run - train 2 iterations
        print("\n1️⃣  First run - training 2 iterations...")
        trainer = RLTrainer(config=config)
        trainer.train()  # This will train for 2 iterations

        first_iteration = trainer.current_iteration
        print(f"✅ First run complete (iteration: {first_iteration + 1})")

        # Second run - resume and train 1 more
        print("\n2️⃣  Second run - resuming from checkpoint...")
        checkpoint_path = "data/test_resume_checkpoints/latest.pt"

        # Create new config for 1 more iteration
        config2 = RLTrainingConfig(
            num_iterations=3,  # Total of 3 iterations
            games_per_iteration=6,
            training_steps_per_iteration=20,
            num_simulations=30,
            cnn_blocks=2,
            batch_size=8,
            buffer_size=1000,
            min_buffer_size=30,
            eval_frequency=999,
            save_frequency=1,
            checkpoint_dir="data/test_resume_checkpoints",
            log_dir="logs/test_resume_training",
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_amp=True if torch.cuda.is_available() else False,
        )

        trainer2 = RLTrainer(config=config2, resume_from=checkpoint_path)

        # Verify it resumed correctly
        if trainer2.current_iteration != first_iteration:
            print(
                f"❌ Resume failed: expected iteration {first_iteration}, got {trainer2.current_iteration}"
            )
            return False

        print(f"✅ Resumed from iteration {trainer2.current_iteration + 1}")

        # Train 1 more iteration (will go from current_iteration to num_iterations)
        trainer2.train()

        final_iteration = trainer2.current_iteration
        print(f"✅ Training complete (iteration: {final_iteration + 1})")

        # Verify it trained more
        if final_iteration <= first_iteration:
            print(
                f"❌ Training didn't progress: {final_iteration + 1} <= {first_iteration + 1}"
            )
            return False

        # Verify scaler was restored
        if trainer2.use_amp and trainer2.scaler is None:
            print("❌ GradScaler was not restored from checkpoint!")
            return False

        if trainer2.use_amp:
            print("✅ GradScaler successfully restored from checkpoint")

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


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("RUNNING ALL RL TRAINING TESTS")
    print("=" * 80)

    results = {}

    # Check AMP availability
    check_amp_available()

    # Test AMP imports
    print("\n" + "=" * 80)
    print("TEST 1/5: AMP Imports")
    print("=" * 80)
    results["amp_imports"] = test_amp_imports()

    # Test basic AMP
    if torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("TEST 2/5: Basic AMP")
        print("=" * 80)
        results["amp_basic"] = test_amp_basic()

        print("\n" + "=" * 80)
        print("TEST 3/5: AMP vs FP32 Comparison")
        print("=" * 80)
        results["amp_comparison"] = test_amp_vs_fp32()
    else:
        print("\n⚠️  Skipping GPU tests (no CUDA)")
        results["amp_basic"] = True
        results["amp_comparison"] = True

    # Test basic functionality
    print("\n" + "=" * 80)
    print("TEST 4/5: Basic Functionality")
    print("=" * 80)
    results["basic"] = test_basic()

    # Test checkpoint resume
    print("\n" + "=" * 80)
    print("TEST 5/5: Checkpoint Resume")
    print("=" * 80)
    results["checkpoint"] = test_checkpoint_resume()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")

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

    parser = argparse.ArgumentParser(description="Test RL Training Loop with AMP")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--basic", action="store_true", help="Run basic test only")
    parser.add_argument("--amp", action="store_true", help="Run AMP tests only")
    parser.add_argument(
        "--checkpoint", action="store_true", help="Run checkpoint test only"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Run AMP comparison test"
    )

    args = parser.parse_args()

    # If no specific test selected, run basic test
    if not (args.all or args.basic or args.amp or args.checkpoint or args.compare):
        args.basic = True

    success = True

    # Run selected tests
    if args.all:
        success = run_all_tests()
    else:
        if args.amp:
            check_amp_available()
            test_amp_imports()
            success = test_amp_basic()
        elif args.compare:
            check_amp_available()
            success = test_amp_vs_fp32()
        elif args.basic:
            success = test_basic()
        elif args.checkpoint:
            success = test_checkpoint_resume()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
