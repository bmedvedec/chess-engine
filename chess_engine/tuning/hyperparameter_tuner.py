"""
HYPERPARAMETER TUNER

Systematic search over the hyperparameter space: grid search, random search,
and Bayesian optimization with early stopping and experiment tracking.
"""

import os
import time
import json
import math
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import itertools

import numpy as np
import torch

from chess_engine.tuning.hyperparameter_config import (
    HyperparameterConfig,
    ParameterRange,
    SearchSpace,
    get_default_search_space,
    get_quick_search_space,
    generate_random_config,
    config_to_flat_dict,
)
from chess_engine.tuning.experiment_tracker import (
    ExperimentTracker,
    ExperimentResult,
    Experiment,
)


class SearchStrategy(Enum):
    """Hyperparameter search strategies"""

    GRID = "grid"
    RANDOM = "random"
    BAYESIAN = "bayesian"


@dataclass
class TuningConfig:
    """Configuration for the hyperparameter tuning process"""

    # Search settings
    strategy: SearchStrategy = SearchStrategy.RANDOM
    max_trials: int = 50

    # Early stopping
    use_early_stopping: bool = True
    min_trials_before_pruning: int = 5
    pruning_percentile: float = 25.0

    # Time limits
    max_time_per_trial: float = 3600.0
    max_total_time: Optional[float] = None

    # Evaluation settings
    quick_eval_games: int = 10
    full_eval_games: int = 50

    # Training per trial
    training_iterations: int = 10
    games_per_iteration: int = 50

    # Metric to optimize
    primary_metric: str = "elo_estimate"
    higher_is_better: bool = True

    # Reproducibility
    seed: Optional[int] = None

    # Output
    output_dir: str = "data/tuning_results"
    save_top_k_models: int = 3


class HyperparameterTuner:
    """Systematic hyperparameter tuning: grid search, random search, or Bayesian optimization."""

    def __init__(
        self,
        tuning_config: TuningConfig,
        search_space: Optional[Dict[str, ParameterRange]] = None,
        base_config: Optional[HyperparameterConfig] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
    ):
        self.tuning_config = tuning_config
        self.search_space = search_space or get_quick_search_space()
        self.base_config = base_config or HyperparameterConfig()

        self.output_dir = Path(tuning_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if experiment_tracker is None:
            experiment_tracker = ExperimentTracker(
                base_dir=str(self.output_dir / "experiments")
            )
        self.tracker = experiment_tracker

        self.rng = np.random.RandomState(tuning_config.seed)

        self.completed_configs: List[Dict[str, Any]] = []
        self.completed_scores: List[float] = []

        self.experiment: Optional[Experiment] = None

        self.start_time: Optional[float] = None
        self.best_score: float = (
            float("-inf") if tuning_config.higher_is_better else float("inf")
        )
        self.best_config: Optional[HyperparameterConfig] = None

    def _generate_grid_configs(self) -> List[HyperparameterConfig]:
        """Generate all configurations for grid search"""
        param_grids = {}

        for name, param_range in self.search_space.items():
            if param_range.space_type == SearchSpace.CATEGORICAL:
                param_grids[name] = param_range.values
            elif param_range.space_type in [
                SearchSpace.INT_LINEAR,
                SearchSpace.INT_LOG,
            ]:
                low, high = param_range.values
                if param_range.space_type == SearchSpace.INT_LOG:
                    values = np.geomspace(low, high, num=5).astype(int)
                else:
                    values = np.linspace(low, high, num=5).astype(int)
                param_grids[name] = list(set(values))
            elif param_range.space_type in [
                SearchSpace.FLOAT_LINEAR,
                SearchSpace.FLOAT_LOG,
            ]:
                low, high = param_range.values
                if param_range.space_type == SearchSpace.FLOAT_LOG:
                    values = np.geomspace(low, high, num=5)
                else:
                    values = np.linspace(low, high, num=5)
                param_grids[name] = list(values)

        param_names = list(param_grids.keys())
        param_values = [param_grids[name] for name in param_names]

        configs = []
        for combination in itertools.product(*param_values):
            config = self.base_config.copy()
            for name, value in zip(param_names, combination):
                config.set_param(name, value)
            configs.append(config)

        return configs

    def _generate_random_config(self) -> HyperparameterConfig:
        """Generate a random configuration"""
        return generate_random_config(
            search_space=self.search_space,
            base_config=self.base_config,
            seed=self.rng.randint(0, 2**31),
        )

    def _generate_bayesian_config(self) -> HyperparameterConfig:
        """Generate configuration using Bayesian optimization"""
        if len(self.completed_scores) < 5:
            return self._generate_random_config()

        n_candidates = 100
        best_candidate = None
        best_ei = float("-inf")

        for _ in range(n_candidates):
            candidate = self._generate_random_config()
            ei = self._expected_improvement(candidate)
            if ei > best_ei:
                best_ei = ei
                best_candidate = candidate

        return best_candidate or self._generate_random_config()

    def _expected_improvement(self, config: HyperparameterConfig) -> float:
        """Calculate expected improvement for Bayesian optimization"""
        if len(self.completed_scores) < 3:
            return float("inf")

        config_flat = config_to_flat_dict(config)
        config_vector = self._config_to_vector(config_flat)

        distances = []
        for completed_flat in self.completed_configs:
            completed_vector = self._config_to_vector(completed_flat)
            dist = np.linalg.norm(config_vector - completed_vector)
            distances.append(max(dist, 0.001))

        distances = np.array(distances)
        scores = np.array(self.completed_scores)

        weights = 1.0 / distances
        weights = weights / weights.sum()

        predicted_mean = np.dot(weights, scores)
        predicted_std = np.sqrt(np.dot(weights, (scores - predicted_mean) ** 2))

        best_so_far = (
            max(self.completed_scores)
            if self.tuning_config.higher_is_better
            else min(self.completed_scores)
        )

        if predicted_std < 0.001:
            return 0.0

        if self.tuning_config.higher_is_better:
            z = (predicted_mean - best_so_far) / predicted_std
        else:
            z = (best_so_far - predicted_mean) / predicted_std

        ei = predicted_std * (z * self._normal_cdf(z) + self._normal_pdf(z))
        return ei

    def _config_to_vector(self, config_flat: Dict[str, Any]) -> np.ndarray:
        """Convert flattened config to numerical vector"""
        vector = []
        for name, param_range in self.search_space.items():
            value = config_flat.get(name, param_range.default)

            if param_range.space_type == SearchSpace.CATEGORICAL:
                if value in param_range.values:
                    idx = param_range.values.index(value)
                else:
                    idx = 0
                vector.append(idx / max(len(param_range.values) - 1, 1))
            else:
                low, high = param_range.values
                if param_range.space_type in [
                    SearchSpace.INT_LOG,
                    SearchSpace.FLOAT_LOG,
                ]:
                    value = math.log(max(value, low))
                    low, high = math.log(low), math.log(high)
                normalized = (value - low) / max(high - low, 0.001)
                vector.append(normalized)

        return np.array(vector)

    @staticmethod
    def _normal_pdf(x: float) -> float:
        """Standard normal PDF"""
        return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Standard normal CDF (approximation)"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _should_prune_trial(self, trial_result: ExperimentResult) -> bool:
        """Check if trial should be pruned"""
        if not self.tuning_config.use_early_stopping:
            return False

        if len(self.completed_scores) < self.tuning_config.min_trials_before_pruning:
            return False

        current_score = trial_result.get_primary_metric(
            self.tuning_config.primary_metric
        )
        percentile = float(
            np.percentile(self.completed_scores, self.tuning_config.pruning_percentile)
        )

        if self.tuning_config.higher_is_better:
            return bool(current_score < percentile)
        else:
            return bool(current_score > percentile)

    def _run_single_trial(
        self,
        config: HyperparameterConfig,
        trial_id: str,
        quick_eval: bool = True,
    ) -> ExperimentResult:
        """Run a single hyperparameter trial"""
        result = ExperimentResult(
            experiment_id=(
                self.experiment.experiment_id if self.experiment else "unknown"
            ),
            trial_id=trial_id,
            config_dict=config.to_dict(),
            status="running",
        )

        start_time = time.time()

        try:
            from chess_engine.models.hybrid_model import HybridChessNet
            from chess_engine.training.rl_trainer import RLTrainer, RLTrainingConfig

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = HybridChessNet(
                cnn_input_channels=config.model.cnn_input_channels,
                cnn_filters=config.model.cnn_filters,
                cnn_residual_blocks=config.model.cnn_residual_blocks,
                cnn_dropout=config.model.cnn_dropout,
                rnn_hidden_size=config.model.rnn_hidden_size,
                rnn_num_layers=config.model.rnn_num_layers,
                rnn_dropout=config.model.rnn_dropout,
                rnn_use_attention=config.model.rnn_use_attention,
                fusion_type=config.model.fusion_type,
                use_rnn=config.model.use_rnn,
            )

            rl_config = RLTrainingConfig(
                num_iterations=self.tuning_config.training_iterations,
                games_per_iteration=self.tuning_config.games_per_iteration,
                training_steps_per_iteration=int(
                    config.training.training_steps_per_iteration
                ),
                num_simulations=int(config.mcts.num_simulations),
                c_puct=float(config.mcts.c_puct),
                temperature=float(config.mcts.temperature),
                batch_size=int(config.training.batch_size),
                learning_rate=float(config.training.learning_rate),
                weight_decay=float(config.training.weight_decay),
                policy_loss_weight=float(config.training.policy_loss_weight),
                value_loss_weight=float(config.training.value_loss_weight),
                buffer_size=int(config.buffer.buffer_size),
                # Pass architecture to RLTrainer so best_model matches
                cnn_blocks=int(config.model.cnn_residual_blocks),
                use_rnn=config.model.use_rnn,
                eval_frequency=2,
                eval_games=(
                    self.tuning_config.quick_eval_games
                    if quick_eval
                    else self.tuning_config.full_eval_games
                ),
                checkpoint_dir=str(self.output_dir / "trial_checkpoints" / trial_id),
                log_dir=str(self.output_dir / "logs" / trial_id),
                device=str(device),
            )

            trainer = RLTrainer(config=rl_config, model=model)
            trainer.train()

            history = trainer.history

            # Extract training metrics (filter out None values)
            train_losses = [x for x in history.get("train_loss", []) if x is not None]
            if train_losses:
                result.final_loss = train_losses[-1]
                result.best_loss = min(train_losses)

            policy_losses = [x for x in history.get("policy_loss", []) if x is not None]
            if policy_losses:
                result.final_policy_loss = policy_losses[-1]

            value_losses = [x for x in history.get("value_loss", []) if x is not None]
            if value_losses:
                result.final_value_loss = value_losses[-1]

            # Extract evaluation metrics (filter out None values)
            eval_win_rates = [
                x for x in history.get("eval_win_rate", []) if x is not None
            ]
            if eval_win_rates:
                result.win_rate_vs_baseline = max(eval_win_rates)

            # Calculate ELO estimate
            win_rate = result.win_rate_vs_baseline
            if win_rate is not None and 0.01 < win_rate < 0.99:
                elo_diff = 400 * math.log10(win_rate / (1 - win_rate))
                result.elo_estimate = 1500 + elo_diff
            elif win_rate is not None:
                result.elo_estimate = 1500 + (500 if win_rate > 0.5 else -500)
            else:
                # No evaluation data - use loss as proxy if available
                if result.final_loss is not None and result.final_loss > 0:
                    # Lower loss = better, map to rough ELO estimate
                    result.elo_estimate = 1500 - (result.final_loss - 0.5) * 200
                else:
                    result.elo_estimate = 1500  # Default

            result.total_games_played = trainer.total_games_played
            result.total_training_steps = trainer.total_training_steps
            result.status = "completed"

        except Exception as e:
            result.status = "failed"
            result.error_message = str(e)
            traceback.print_exc()

        result.training_time_seconds = time.time() - start_time
        return result

    def run(
        self,
        experiment_name: str = "hyperparameter_tuning",
        description: str = "",
    ) -> Experiment:
        """Run the hyperparameter tuning process"""
        self.start_time = time.time()

        search_space_dict = {
            name: {
                "space_type": pr.space_type.value,
                "values": pr.values if isinstance(pr.values, list) else list(pr.values),
                "default": pr.default,
            }
            for name, pr in self.search_space.items()
        }

        self.experiment = self.tracker.create_experiment(
            name=experiment_name,
            description=description or f"{self.tuning_config.strategy.value} search",
            base_config=self.base_config,
            search_space=search_space_dict,
        )

        print(f"\n{'='*80}")
        print(f"HYPERPARAMETER TUNING: {experiment_name}")
        print(f"{'='*80}")
        print(f"Strategy: {self.tuning_config.strategy.value}")
        print(f"Max trials: {self.tuning_config.max_trials}")
        print(f"Search space: {len(self.search_space)} parameters")
        print(f"{'='*80}\n")

        # Initialize grid configs if using grid search
        grid_configs: List[HyperparameterConfig] = []
        if self.tuning_config.strategy == SearchStrategy.GRID:
            grid_configs = self._generate_grid_configs()
            print(f"Grid search: {len(grid_configs)} configurations")
            grid_configs = grid_configs[: self.tuning_config.max_trials]

        trial_num = 0

        while trial_num < self.tuning_config.max_trials:
            if self.tuning_config.max_total_time and self.start_time is not None:
                elapsed = time.time() - self.start_time
                if elapsed > self.tuning_config.max_total_time:
                    print(f"\nTime limit reached ({elapsed/3600:.1f}h)")
                    break

            if self.tuning_config.strategy == SearchStrategy.GRID:
                if trial_num >= len(grid_configs):
                    break
                config = grid_configs[trial_num]
            elif self.tuning_config.strategy == SearchStrategy.RANDOM:
                config = self._generate_random_config()
            else:
                config = self._generate_bayesian_config()

            trial_id = f"trial_{trial_num + 1}_{int(time.time())}"

            print(f"\n{'─'*60}")
            print(f"TRIAL {trial_num + 1}/{self.tuning_config.max_trials}")
            print(f"{'─'*60}")

            print("Configuration:")
            for name in list(self.search_space.keys())[:5]:
                value = config.get_param(name)
                print(f"  {name}: {value}")

            if self.experiment is None:
                raise RuntimeError("Experiment not initialized")

            result = self._run_single_trial(config, trial_id)
            self.tracker.record_trial_result(self.experiment.experiment_id, result)

            if result.status == "completed":
                score = result.get_primary_metric(self.tuning_config.primary_metric)

                self.completed_configs.append(config_to_flat_dict(config))
                self.completed_scores.append(score)

                is_best = (
                    self.tuning_config.higher_is_better and score > self.best_score
                ) or (
                    not self.tuning_config.higher_is_better and score < self.best_score
                )

                if is_best:
                    self.best_score = score
                    self.best_config = config.copy()
                    print(
                        f"\nNew best: {self.tuning_config.primary_metric}: {score:.4f}"
                    )

                print(f"\nTrial {trial_num + 1} Results:")
                print(f"  {self.tuning_config.primary_metric}: {score:.4f}")
                print(f"  Loss: {result.final_loss:.4f}")
                print(f"  Time: {result.training_time_seconds/60:.1f}m")
            else:
                print(f"\nTrial failed: {result.error_message}")

            trial_num += 1

        self.experiment.status = "completed"
        self.tracker.save_experiment(self.experiment)
        self._generate_final_report()

        return self.experiment

    def _generate_final_report(self) -> None:
        """Generate final report"""
        if self.start_time is None:
            total_time = 0.0
        else:
            total_time = time.time() - self.start_time

        print(f"\n{'='*80}")
        print("TUNING COMPLETE")
        print(f"{'='*80}")

        completed = len(self.completed_scores)
        print(f"\nSummary:")
        print(f"  Total trials: {completed}")
        print(f"  Total time: {total_time/3600:.2f} hours")

        if self.completed_scores:
            print(f"\nPerformance ({self.tuning_config.primary_metric}):")
            print(f"  Best:   {self.best_score:.4f}")
            print(f"  Mean:   {np.mean(self.completed_scores):.4f}")
            print(f"  Std:    {np.std(self.completed_scores):.4f}")

        if self.best_config:
            print(f"\nBest Configuration:")
            for name in self.search_space.keys():
                value = self.best_config.get_param(name)
                print(f"  {name}: {value}")

            best_config_path = self.output_dir / "best_config.json"
            self.best_config.save(best_config_path)
            print(f"\nBest config saved: {best_config_path}")

        if self.experiment is not None:
            report_path = self.output_dir / "tuning_report.txt"
            self.tracker.generate_report(
                self.experiment.experiment_id, str(report_path)
            )


def run_grid_search(
    search_space: Optional[Dict[str, ParameterRange]] = None,
    base_config: Optional[HyperparameterConfig] = None,
    max_trials: int = 50,
    output_dir: str = "data/tuning_results/grid_search",
) -> Experiment:
    """Convenience function to run grid search"""
    tuning_config = TuningConfig(
        strategy=SearchStrategy.GRID,
        max_trials=max_trials,
        output_dir=output_dir,
    )

    tuner = HyperparameterTuner(
        tuning_config=tuning_config,
        search_space=search_space,
        base_config=base_config,
    )

    return tuner.run(experiment_name="grid_search")


def run_random_search(
    search_space: Optional[Dict[str, ParameterRange]] = None,
    base_config: Optional[HyperparameterConfig] = None,
    max_trials: int = 50,
    output_dir: str = "data/tuning_results/random_search",
    seed: Optional[int] = None,
) -> Experiment:
    """Convenience function to run random search"""
    tuning_config = TuningConfig(
        strategy=SearchStrategy.RANDOM,
        max_trials=max_trials,
        output_dir=output_dir,
        seed=seed,
    )

    tuner = HyperparameterTuner(
        tuning_config=tuning_config,
        search_space=search_space,
        base_config=base_config,
    )

    return tuner.run(experiment_name="random_search")


def run_bayesian_search(
    search_space: Optional[Dict[str, ParameterRange]] = None,
    base_config: Optional[HyperparameterConfig] = None,
    max_trials: int = 50,
    output_dir: str = "data/tuning_results/bayesian_search",
    seed: Optional[int] = None,
) -> Experiment:
    """Convenience function to run Bayesian optimization"""
    tuning_config = TuningConfig(
        strategy=SearchStrategy.BAYESIAN,
        max_trials=max_trials,
        output_dir=output_dir,
        seed=seed,
    )

    tuner = HyperparameterTuner(
        tuning_config=tuning_config,
        search_space=search_space,
        base_config=base_config,
    )

    return tuner.run(experiment_name="bayesian_optimization")


def quick_tune(
    base_config: Optional[HyperparameterConfig] = None,
    max_trials: int = 10,
    output_dir: str = "data/tuning_results/quick_tune",
) -> Optional[HyperparameterConfig]:
    """Quick tuning for development"""
    tuning_config = TuningConfig(
        strategy=SearchStrategy.RANDOM,
        max_trials=max_trials,
        output_dir=output_dir,
        training_iterations=3,
        games_per_iteration=20,
        quick_eval_games=5,
    )

    search_space = get_quick_search_space()

    tuner = HyperparameterTuner(
        tuning_config=tuning_config,
        search_space=search_space,
        base_config=base_config,
    )

    tuner.run(experiment_name="quick_tune")
    return tuner.best_config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuning for Chess Engine"
    )
    parser.add_argument(
        "--strategy",
        choices=["grid", "random", "bayesian"],
        default="random",
        help="Search strategy to use",
    )
    parser.add_argument(
        "--max-trials", type=int, default=20, help="Maximum number of trials to run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tuning_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--training-iterations",
        type=int,
        default=10,
        help="Training iterations per trial",
    )
    parser.add_argument(
        "--games-per-iteration",
        type=int,
        default=50,
        help="Self-play games per iteration",
    )
    parser.add_argument(
        "--search-space",
        choices=["quick", "full", "architecture", "training", "mcts"],
        default="quick",
        help="Which search space to use",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["fast_dev", "balanced", "strong", "alphazero", "hybrid"],
        help="Base configuration preset",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Select search space
    search_space_map = {
        "quick": get_quick_search_space,
        "full": get_default_search_space,
        "architecture": lambda: {
            k: v
            for k, v in get_default_search_space().items()
            if k.startswith("model.")
        },
        "training": lambda: {
            k: v
            for k, v in get_default_search_space().items()
            if k.startswith("training.")
        },
        "mcts": lambda: {
            k: v for k, v in get_default_search_space().items() if k.startswith("mcts.")
        },
    }
    search_space = search_space_map[args.search_space]()

    # Get base config from preset if specified
    base_config = None
    if args.preset:
        from chess_engine.tuning.hyperparameter_config import create_config_from_preset

        base_config = create_config_from_preset(args.preset)
        print(f"Using preset: {args.preset}")

    tuning_config = TuningConfig(
        strategy=SearchStrategy[args.strategy.upper()],
        max_trials=args.max_trials,
        output_dir=args.output_dir,
        training_iterations=args.training_iterations,
        games_per_iteration=args.games_per_iteration,
        seed=args.seed,
    )

    tuner = HyperparameterTuner(
        tuning_config=tuning_config,
        search_space=search_space,
        base_config=base_config,
    )

    experiment = tuner.run(experiment_name=f"{args.strategy}_search")
    print(f"\nTuning complete. Experiment: {experiment.experiment_id}")
