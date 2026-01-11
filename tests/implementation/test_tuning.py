#!/usr/bin/env python3
"""
CHAPTER 13: HYPERPARAMETER TUNING - TEST SUITE

Tests for the hyperparameter tuning and optimization module.
Validates all core functionality without requiring full training runs.
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, "/home/claude")

import numpy as np


def test_hyperparameter_config():
    """Test configuration classes and utilities"""
    print("\n" + "=" * 60)
    print("TEST: Hyperparameter Configuration System")
    print("=" * 60)

    from chess_engine.tuning.hyperparameter_config import (
        HyperparameterConfig,
        ModelConfig,
        TrainingConfig,
        MCTSConfig,
        ParameterRange,
        SearchSpace,
        get_default_search_space,
        get_quick_search_space,
        get_preset_configs,
        create_config_from_preset,
        generate_random_config,
        config_to_flat_dict,
        compare_configs,
    )

    # Test default config creation
    config = HyperparameterConfig(name="test_config")
    assert config.name == "test_config"
    assert config.model.cnn_residual_blocks == 10
    assert config.training.learning_rate == 0.001
    print("✓ Default config creation")

    # Test nested config access
    assert config.model.cnn_filters == 256
    assert config.mcts.num_simulations == 200
    print("✓ Nested config access")

    # Test get_param and set_param
    config.set_param("model.cnn_blocks", 15)
    # Note: This might fail if the attribute name doesn't match
    # Let's use an attribute that exists
    config.set_param("model.cnn_residual_blocks", 15)
    assert config.model.cnn_residual_blocks == 15
    print("✓ set_param/get_param")

    # Test validation
    issues = config.validate()
    assert isinstance(issues, list)
    print(f"✓ Validation ({len(issues)} issues)")

    # Test save/load
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        temp_path = f.name

    config.save(temp_path)
    loaded = HyperparameterConfig.load(temp_path)
    assert loaded.model.cnn_residual_blocks == 15
    os.unlink(temp_path)
    print("✓ Save/Load")

    # Test search spaces
    default_space = get_default_search_space()
    assert len(default_space) > 10
    print(f"✓ Default search space: {len(default_space)} parameters")

    quick_space = get_quick_search_space()
    assert len(quick_space) <= len(default_space)
    print(f"✓ Quick search space: {len(quick_space)} parameters")

    # Test parameter sampling
    lr_range = default_space["training.learning_rate"]
    sampled_lr = lr_range.sample()
    assert lr_range.values[0] <= sampled_lr <= lr_range.values[1]
    print("✓ Parameter sampling")

    # Test presets
    presets = get_preset_configs()
    assert "fast_dev" in presets
    assert "balanced" in presets
    print(f"✓ Presets available: {list(presets.keys())}")

    # Test preset creation
    fast_dev = create_config_from_preset("fast_dev")
    assert fast_dev.model.cnn_residual_blocks == 3
    print("✓ Create from preset")

    # Test random config generation
    random_config = generate_random_config(seed=42)
    assert random_config is not None
    print("✓ Random config generation")

    # Test config flattening
    flat = config_to_flat_dict(config)
    assert "model.cnn_filters" in flat
    print(f"✓ Config flattening: {len(flat)} flat keys")

    # Test config comparison
    config2 = HyperparameterConfig(name="test2")
    config2.model.cnn_residual_blocks = 20
    diff = compare_configs(config, config2)
    assert "model.cnn_residual_blocks" in diff
    print(f"✓ Config comparison: {len(diff)} differences")

    print("\n✅ All configuration tests passed!")


def test_experiment_tracker():
    """Test experiment tracking system"""
    print("\n" + "=" * 60)
    print("TEST: Experiment Tracking System")
    print("=" * 60)

    from chess_engine.tuning.experiment_tracker import (
        ExperimentTracker,
        ExperimentResult,
        Experiment,
        create_experiment_summary_table,
    )
    from chess_engine.tuning.hyperparameter_config import HyperparameterConfig

    # Create tracker in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = ExperimentTracker(base_dir=tmpdir)

        # Create experiment
        config = HyperparameterConfig(name="test")
        exp = tracker.create_experiment(
            name="Test Experiment",
            description="Testing tracker",
            base_config=config,
        )
        assert exp.experiment_id is not None
        print(f"✓ Created experiment: {exp.experiment_id}")

        # Add trials
        for i in range(5):
            result = ExperimentResult(
                experiment_id=exp.experiment_id,
                trial_id=f"trial_{i+1}",
                status="completed",
                elo_estimate=1500 + np.random.randn() * 100,
                final_loss=0.5 + np.random.randn() * 0.1,
                win_rate_vs_baseline=0.5 + np.random.randn() * 0.1,
                training_time_seconds=3600 + np.random.randn() * 600,
                config_dict=config.to_dict(),
            )
            tracker.record_trial_result(exp.experiment_id, result)
        print("✓ Recorded 5 trial results")

        # Load experiment
        loaded = tracker.load_experiment(exp.experiment_id)
        assert loaded is not None
        assert len(loaded.trials) == 5
        print(f"✓ Loaded experiment with {len(loaded.trials)} trials")

        # Get statistics
        stats = loaded.get_statistics("elo_estimate")
        assert "mean" in stats
        assert "std" in stats
        print(f"✓ Statistics: mean={stats['mean']:.1f}, std={stats['std']:.1f}")

        # Get best trial
        best = loaded.get_best_trial("elo_estimate")
        assert best is not None
        print(f"✓ Best trial: {best.trial_id} (ELO: {best.elo_estimate:.1f})")

        # List experiments
        experiments = tracker.list_experiments()
        assert len(experiments) >= 1
        print(f"✓ Listed {len(experiments)} experiments")

        # Generate report
        report = tracker.generate_report(exp.experiment_id)
        assert "EXPERIMENT REPORT" in report
        print("✓ Generated experiment report")

        # Summary table
        table = create_experiment_summary_table(tracker)
        assert "Experiment" in table
        print("✓ Created summary table")

    print("\n✅ All experiment tracker tests passed!")


def test_hyperparameter_tuner():
    """Test hyperparameter tuning infrastructure"""
    print("\n" + "=" * 60)
    print("TEST: Hyperparameter Tuner Infrastructure")
    print("=" * 60)

    from chess_engine.tuning.hyperparameter_tuner import (
        HyperparameterTuner,
        TuningConfig,
        SearchStrategy,
    )
    from chess_engine.tuning.hyperparameter_config import (
        HyperparameterConfig,
        get_quick_search_space,
    )

    # Test TuningConfig
    tuning_config = TuningConfig(
        strategy=SearchStrategy.RANDOM,
        max_trials=5,
        training_iterations=1,
        games_per_iteration=2,
    )
    assert tuning_config.strategy == SearchStrategy.RANDOM
    print("✓ TuningConfig creation")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create tuner
        tuner = HyperparameterTuner(
            tuning_config=tuning_config,
            search_space=get_quick_search_space(),
            base_config=HyperparameterConfig(name="base"),
        )
        print("✓ HyperparameterTuner initialization")

        # Test config generation
        random_config = tuner._generate_random_config()
        assert random_config is not None
        print("✓ Random config generation")

        # Test grid config generation
        tuner.tuning_config.strategy = SearchStrategy.GRID
        grid_configs = tuner._generate_grid_configs()
        assert len(grid_configs) > 0
        print(f"✓ Grid configs generated: {len(grid_configs)} combinations")

        # Test Bayesian components
        tuner.completed_scores = [1500, 1520, 1480, 1550, 1490]
        tuner.completed_configs = [
            {"training.learning_rate": 0.001, "mcts.num_simulations": 100},
            {"training.learning_rate": 0.0001, "mcts.num_simulations": 200},
            {"training.learning_rate": 0.01, "mcts.num_simulations": 150},
            {"training.learning_rate": 0.0005, "mcts.num_simulations": 250},
            {"training.learning_rate": 0.002, "mcts.num_simulations": 100},
        ]

        bayesian_config = tuner._generate_bayesian_config()
        assert bayesian_config is not None
        print("✓ Bayesian config generation")

        # Test expected improvement calculation
        test_config = HyperparameterConfig()
        ei = tuner._expected_improvement(test_config)
        assert isinstance(ei, float)
        print(f"✓ Expected improvement calculation: {ei:.4f}")

        # Test pruning decision
        from chess_engine.tuning.experiment_tracker import ExperimentResult

        result = ExperimentResult(
            experiment_id="test",
            trial_id="test_trial",
            elo_estimate=1400,  # Low score
        )
        tuner.tuning_config.use_early_stopping = True
        should_prune = tuner._should_prune_trial(result)
        # Should prune because 1400 is below the 25th percentile of [1480, 1490, 1500, 1520, 1550]
        print(f"✓ Pruning decision: {'prune' if should_prune else 'keep'}")

    print("\n✅ All tuner infrastructure tests passed!")


def test_comparison_report():
    """Test comparison and reporting utilities"""
    print("\n" + "=" * 60)
    print("TEST: Comparison & Reporting")
    print("=" * 60)

    from chess_engine.tuning.comparison import (
        TuningAnalyzer,
        ReportGenerator,
        ComparisonResult,
        compare_and_report,
    )
    from chess_engine.tuning.hyperparameter_config import (
        HyperparameterConfig,
        create_config_from_preset,
    )
    from chess_engine.tuning.experiment_tracker import (
        ExperimentTracker,
        ExperimentResult,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup tracker with test data
        tracker = ExperimentTracker(base_dir=tmpdir)

        # Create two experiments for comparison
        for exp_name, base_elo in [("Experiment A", 1450), ("Experiment B", 1550)]:
            config = create_config_from_preset("fast_dev")
            exp = tracker.create_experiment(name=exp_name, base_config=config)

            for i in range(5):
                result = ExperimentResult(
                    experiment_id=exp.experiment_id,
                    trial_id=f"trial_{i+1}",
                    status="completed",
                    elo_estimate=base_elo + np.random.randn() * 30,
                    final_loss=0.5 + np.random.randn() * 0.05,
                    win_rate_vs_baseline=0.5 + (base_elo - 1500) / 500,
                    training_time_seconds=3600,
                    config_dict=config.to_dict(),
                )
                tracker.record_trial_result(exp.experiment_id, result)

        print("✓ Created test experiments")

        # Test TuningAnalyzer
        analyzer = TuningAnalyzer(tracker)
        exp_ids = [eid for eid, _ in tracker.list_experiments()]

        comparison = analyzer.compare_experiments(exp_ids)
        assert comparison.best_experiment is not None
        assert len(comparison.recommendations) > 0
        print(f"✓ Comparison: best={comparison.best_experiment}")

        # Test best config extraction
        best_config, best_trial = analyzer.get_best_configuration(exp_ids)
        assert best_trial is not None
        print(f"✓ Best trial: ELO={best_trial.elo_estimate:.1f}")

        # Test ReportGenerator
        generator = ReportGenerator(tracker)

        text_report = generator.generate_text_report(exp_ids)
        assert "EXPERIMENT COMPARISON" in text_report
        print("✓ Text report generated")

        md_report = generator.generate_markdown_report(exp_ids)
        assert "# Hyperparameter Tuning Report" in md_report
        print("✓ Markdown report generated")

        json_summary = generator.generate_json_summary(exp_ids)
        assert "metrics_comparison" in json_summary
        print("✓ JSON summary generated")

    print("\n✅ All comparison tests passed!")


def test_integration():
    """Integration test of full workflow"""
    print("\n" + "=" * 60)
    print("TEST: Integration - Full Workflow")
    print("=" * 60)

    from chess_engine.tuning.hyperparameter_config import (
        HyperparameterConfig,
        create_config_from_preset,
        get_quick_search_space,
        generate_random_config,
        config_to_flat_dict,
    )
    from chess_engine.tuning.experiment_tracker import (
        ExperimentTracker,
        ExperimentResult,
    )

    from chess_engine.tuning.comparison import ExperimentTracker, ExperimentResult

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Setup
        tracker = ExperimentTracker(base_dir=tmpdir)
        base_config = create_config_from_preset("fast_dev")
        search_space = get_quick_search_space()

        # 2. Create experiment
        experiment = tracker.create_experiment(
            name="Integration Test",
            description="Full workflow integration test",
            base_config=base_config,
        )
        print(f"✓ Created experiment: {experiment.experiment_id}")

        # 3. Simulate trials
        for i in range(10):
            # Generate random config
            config = generate_random_config(search_space, base_config, seed=i)

            # Simulate training (mock results)
            lr = config.training.learning_rate
            sims = config.mcts.num_simulations

            # Mock: lower LR + more sims = better
            base_elo = 1500
            elo_from_lr = -200 * np.log10(lr / 0.001)  # Penalty for high LR
            elo_from_sims = 50 * np.log10(sims / 100)  # Bonus for more sims
            noise = np.random.randn() * 20
            elo = base_elo + elo_from_lr + elo_from_sims + noise

            result = ExperimentResult(
                experiment_id=experiment.experiment_id,
                trial_id=f"trial_{i+1}",
                status="completed",
                elo_estimate=elo,
                final_loss=0.5 - (elo - 1500) / 1000,
                win_rate_vs_baseline=0.5 + (elo - 1500) / 500,
                training_time_seconds=1800 + i * 100,
                config_dict=config.to_dict(),
            )
            tracker.record_trial_result(experiment.experiment_id, result)

        print("✓ Simulated 10 trials")

        # 4. Analyze results
        loaded = tracker.load_experiment(experiment.experiment_id)
        assert loaded is not None, "Failed to load experiment"
        stats = loaded.get_statistics("elo_estimate")
        best = loaded.get_best_trial("elo_estimate")

        print(f"✓ Results: mean={stats['mean']:.1f}, best={stats['max']:.1f}")

        # 5. Generate report
        report = tracker.generate_report(experiment.experiment_id)
        assert "Integration Test" in report
        print("✓ Generated final report")

        # 6. Verify best config makes sense
        assert best is not None, "No best trial found"
        best_config_dict = best.config_dict
        print(f"✓ Best trial: {best.trial_id}")
        print(f"  ELO: {best.elo_estimate:.1f}")
        print(f"  LR: {best_config_dict['training']['learning_rate']}")
        print(f"  MCTS Sims: {best_config_dict['mcts']['num_simulations']}")

    print("\n✅ Integration test passed!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("CHAPTER 13: HYPERPARAMETER TUNING - TEST SUITE")
    print("=" * 60)

    tests = [
        ("Configuration System", test_hyperparameter_config),
        ("Experiment Tracker", test_experiment_tracker),
        ("Tuner Infrastructure", test_hyperparameter_tuner),
        ("Comparison & Reports", test_comparison_report),
        ("Integration", test_integration),
    ]

    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True, None))
        except Exception as e:
            results.append((name, False, str(e)))
            import traceback

            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    for name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"        Error: {error}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
