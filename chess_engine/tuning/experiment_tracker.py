"""
EXPERIMENT TRACKING

Experiment logging with full configuration capture, result persistence,
cross-experiment comparison, visualization, and report generation.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import hashlib

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from chess_engine.tuning.hyperparameter_config import (
    HyperparameterConfig,
    config_to_flat_dict,
    compare_configs,
)


@dataclass
class ExperimentResult:
    """
    Results from a single experiment/trial.

    Captures all metrics and metadata for analysis.
    """

    # Identification
    experiment_id: str
    trial_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Configuration
    config_hash: str = ""
    config_dict: Dict[str, Any] = field(default_factory=dict)

    # Training metrics
    final_loss: float = float("inf")
    final_policy_loss: float = float("inf")
    final_value_loss: float = float("inf")
    best_loss: float = float("inf")

    # Evaluation metrics
    elo_estimate: float = 0.0
    win_rate_vs_baseline: float = 0.0
    win_rate_vs_random: float = 0.0
    win_rate_vs_stockfish: Dict[int, float] = field(default_factory=dict)

    # Accuracy metrics
    policy_accuracy_top1: float = 0.0
    policy_accuracy_top5: float = 0.0
    value_mae: float = float("inf")

    # Resource usage
    training_time_seconds: float = 0.0
    total_games_played: int = 0
    total_training_steps: int = 0
    peak_memory_mb: float = 0.0

    # Convergence metrics
    epochs_to_converge: int = 0
    training_stability: float = 0.0  # Variance of loss over last N steps

    # Additional metrics
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: str = "pending"  # 'pending', 'running', 'completed', 'failed'
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (JSON-safe)"""
        from chess_engine.tuning.hyperparameter_config import _make_json_safe

        return _make_json_safe(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary"""
        return cls(**data)

    def get_primary_metric(self, metric_name: str = "elo_estimate") -> float:
        """Get the primary metric for comparison"""
        return getattr(self, metric_name, 0.0)


@dataclass
class Experiment:
    """
    Container for a full experiment with multiple trials.
    """

    experiment_id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Base configuration
    base_config: Dict[str, Any] = field(default_factory=dict)

    # Search space used
    search_space: Dict[str, Any] = field(default_factory=dict)

    # Trials
    trials: List[ExperimentResult] = field(default_factory=list)

    # Status
    status: str = "active"  # 'active', 'completed', 'paused'

    def add_trial(self, result: ExperimentResult) -> None:
        """Add a trial result"""
        self.trials.append(result)

    def get_best_trial(
        self, metric: str = "elo_estimate", higher_is_better: bool = True
    ) -> Optional[ExperimentResult]:
        """Get the best trial by a metric"""
        if not self.trials:
            return None

        completed = [t for t in self.trials if t.status == "completed"]
        if not completed:
            return None

        if higher_is_better:
            return max(completed, key=lambda t: t.get_primary_metric(metric))
        else:
            return min(completed, key=lambda t: t.get_primary_metric(metric))

    def get_statistics(self, metric: str = "elo_estimate") -> Dict[str, float]:
        """Get statistics across all completed trials"""
        completed = [t for t in self.trials if t.status == "completed"]
        if not completed:
            return {}

        values = [t.get_primary_metric(metric) for t in completed]
        return {
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "base_config": self.base_config,
            "search_space": self.search_space,
            "trials": [t.to_dict() for t in self.trials],
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Create from dictionary"""
        trials = [ExperimentResult.from_dict(t) for t in data.pop("trials", [])]
        return cls(**data, trials=trials)


class ExperimentTracker:
    """Tracks and manages hyperparameter tuning experiments with persistent storage."""

    def __init__(self, base_dir: str = "data/experiments"):
        """
        Initialize experiment tracker.

        Args:
            base_dir: Base directory for experiment storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Index file for quick lookup
        self.index_file = self.base_dir / "experiment_index.json"
        self.index = self._load_index()

    def _load_index(self) -> Dict[str, str]:
        """Load experiment index"""
        if self.index_file.exists():
            with open(self.index_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """Save experiment index"""
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2)

    def _get_experiment_dir(self, experiment_id: str) -> Path:
        """Get directory for an experiment"""
        return self.base_dir / experiment_id

    def create_experiment(
        self,
        name: str,
        description: str = "",
        base_config: Optional[HyperparameterConfig] = None,
        search_space: Optional[Dict[str, Any]] = None,
    ) -> Experiment:
        """
        Create a new experiment.

        Args:
            name: Human-readable experiment name
            description: Detailed description
            base_config: Base configuration for the experiment
            search_space: Search space definition

        Returns:
            New Experiment instance
        """
        # Generate unique ID
        experiment_id = f"{name.lower().replace(' ', '_')}_{int(time.time())}"

        # Create experiment
        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            base_config=base_config.to_dict() if base_config else {},
            search_space=search_space or {},
        )

        # Create directory
        exp_dir = self._get_experiment_dir(experiment_id)
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save experiment
        self.save_experiment(experiment)

        # Update index
        self.index[experiment_id] = name
        self._save_index()

        print(f"Created experiment: {name} ({experiment_id})")
        return experiment

    def save_experiment(self, experiment: Experiment) -> None:
        """Save experiment to disk"""
        exp_dir = self._get_experiment_dir(experiment.experiment_id)
        exp_dir.mkdir(parents=True, exist_ok=True)

        filepath = exp_dir / "experiment.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(experiment.to_dict(), f, indent=2)

    def load_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Load experiment from disk"""
        exp_dir = self._get_experiment_dir(experiment_id)
        filepath = exp_dir / "experiment.json"

        if not filepath.exists():
            return None

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Experiment.from_dict(data)

    def list_experiments(self) -> List[Tuple[str, str]]:
        """List all experiments as (id, name) tuples"""
        return list(self.index.items())

    def create_trial(
        self,
        experiment_id: str,
        config: HyperparameterConfig,
    ) -> ExperimentResult:
        """
        Create a new trial within an experiment.

        Args:
            experiment_id: Parent experiment ID
            config: Configuration for this trial

        Returns:
            New ExperimentResult (pending status)
        """
        # Generate trial ID
        experiment = self.load_experiment(experiment_id)
        trial_count = len(experiment.trials) if experiment else 0
        trial_id = f"trial_{trial_count + 1}_{int(time.time())}"

        # Create config hash for deduplication
        config_flat = config_to_flat_dict(config)
        config_hash = hashlib.md5(
            json.dumps(config_flat, sort_keys=True).encode()
        ).hexdigest()[:8]

        result = ExperimentResult(
            experiment_id=experiment_id,
            trial_id=trial_id,
            config_hash=config_hash,
            config_dict=config.to_dict(),
            status="pending",
        )

        return result

    def record_trial_result(
        self,
        experiment_id: str,
        result: ExperimentResult,
    ) -> None:
        """
        Record a trial result.

        Args:
            experiment_id: Parent experiment ID
            result: Trial result to record
        """
        experiment = self.load_experiment(experiment_id)
        if experiment is None:
            raise ValueError(f"Experiment not found: {experiment_id}")

        # Check if trial already exists (update) or new (append)
        existing_idx = None
        for idx, trial in enumerate(experiment.trials):
            if trial.trial_id == result.trial_id:
                existing_idx = idx
                break

        if existing_idx is not None:
            experiment.trials[existing_idx] = result
        else:
            experiment.trials.append(result)

        self.save_experiment(experiment)

    def get_best_config(
        self,
        experiment_id: str,
        metric: str = "elo_estimate",
        higher_is_better: bool = True,
    ) -> Optional[HyperparameterConfig]:
        """
        Get the best configuration from an experiment.

        Args:
            experiment_id: Experiment ID
            metric: Metric to optimize
            higher_is_better: Whether higher values are better

        Returns:
            Best HyperparameterConfig or None
        """
        experiment = self.load_experiment(experiment_id)
        if experiment is None:
            return None

        best_trial = experiment.get_best_trial(metric, higher_is_better)
        if best_trial is None:
            return None

        # Reconstruct config from dict
        if best_trial.config_dict:
            config = HyperparameterConfig()
            for section in [
                "model",
                "training",
                "mcts",
                "self_play",
                "buffer",
                "evaluation",
            ]:
                if section in best_trial.config_dict:
                    section_config = getattr(config, section)
                    for key, value in best_trial.config_dict[section].items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)
            return config
        return None

    def compare_experiments(
        self,
        experiment_ids: List[str],
        metric: str = "elo_estimate",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare
            metric: Metric for comparison

        Returns:
            Comparison dictionary
        """
        comparison = {}

        for exp_id in experiment_ids:
            experiment = self.load_experiment(exp_id)
            if experiment is None:
                continue

            stats = experiment.get_statistics(metric)
            best = experiment.get_best_trial(metric)

            comparison[exp_id] = {
                "name": experiment.name,
                "num_trials": len(experiment.trials),
                "completed_trials": len(
                    [t for t in experiment.trials if t.status == "completed"]
                ),
                "statistics": stats,
                "best_trial_id": best.trial_id if best else None,
                "best_value": best.get_primary_metric(metric) if best else None,
            }

        return comparison

    def generate_report(
        self,
        experiment_id: str,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate a text report for an experiment.

        Args:
            experiment_id: Experiment ID
            output_path: Optional path to save report

        Returns:
            Report text
        """
        experiment = self.load_experiment(experiment_id)
        if experiment is None:
            return f"Experiment not found: {experiment_id}"

        lines = []
        lines.append("=" * 80)
        lines.append(f"EXPERIMENT REPORT: {experiment.name}")
        lines.append("=" * 80)
        lines.append(f"\nExperiment ID: {experiment.experiment_id}")
        lines.append(f"Created: {experiment.created_at}")
        lines.append(f"Description: {experiment.description}")
        lines.append(f"Status: {experiment.status}")

        # Trial summary
        completed = [t for t in experiment.trials if t.status == "completed"]
        failed = [t for t in experiment.trials if t.status == "failed"]

        lines.append(f"\n{'─' * 40}")
        lines.append("TRIAL SUMMARY")
        lines.append(f"{'─' * 40}")
        lines.append(f"Total trials: {len(experiment.trials)}")
        lines.append(f"Completed: {len(completed)}")
        lines.append(f"Failed: {len(failed)}")

        if completed:
            # Statistics
            lines.append(f"\n{'─' * 40}")
            lines.append("PERFORMANCE STATISTICS")
            lines.append(f"{'─' * 40}")

            for metric in ["elo_estimate", "win_rate_vs_baseline", "final_loss"]:
                stats = experiment.get_statistics(metric)
                if stats:
                    lines.append(f"\n{metric}:")
                    lines.append(f"  Mean: {stats['mean']:.4f}")
                    lines.append(f"  Std:  {stats['std']:.4f}")
                    lines.append(f"  Min:  {stats['min']:.4f}")
                    lines.append(f"  Max:  {stats['max']:.4f}")

            # Best trial
            best = experiment.get_best_trial("elo_estimate")
            if best:
                lines.append(f"\n{'─' * 40}")
                lines.append("BEST TRIAL")
                lines.append(f"{'─' * 40}")
                lines.append(f"Trial ID: {best.trial_id}")
                lines.append(f"ELO Estimate: {best.elo_estimate:.1f}")
                lines.append(f"Win Rate vs Baseline: {best.win_rate_vs_baseline:.1%}")
                lines.append(f"Final Loss: {best.final_loss:.4f}")
                lines.append(
                    f"Training Time: {best.training_time_seconds/3600:.2f} hours"
                )

                # Best config parameters
                if best.config_dict:
                    lines.append("\nKey Configuration:")
                    config = best.config_dict
                    if "model" in config:
                        lines.append(
                            f"  CNN Blocks: {config['model'].get('cnn_residual_blocks', 'N/A')}"
                        )
                        lines.append(
                            f"  CNN Filters: {config['model'].get('cnn_filters', 'N/A')}"
                        )
                    if "training" in config:
                        lines.append(
                            f"  Learning Rate: {config['training'].get('learning_rate', 'N/A')}"
                        )
                        lines.append(
                            f"  Batch Size: {config['training'].get('batch_size', 'N/A')}"
                        )
                    if "mcts" in config:
                        lines.append(
                            f"  MCTS Simulations: {config['mcts'].get('num_simulations', 'N/A')}"
                        )
                        lines.append(f"  C_PUCT: {config['mcts'].get('c_puct', 'N/A')}")

        # Top 5 trials
        if len(completed) > 1:
            lines.append(f"\n{'─' * 40}")
            lines.append("TOP 5 TRIALS BY ELO")
            lines.append(f"{'─' * 40}")

            sorted_trials = sorted(
                completed, key=lambda t: t.elo_estimate, reverse=True
            )[:5]
            for i, trial in enumerate(sorted_trials, 1):
                lines.append(f"\n{i}. {trial.trial_id}")
                lines.append(
                    f"   ELO: {trial.elo_estimate:.1f}, Loss: {trial.final_loss:.4f}"
                )
                lines.append(f"   Win Rate: {trial.win_rate_vs_baseline:.1%}")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"Report saved: {output_path}")

        return report

    def plot_experiment_results(
        self,
        experiment_id: str,
        metric: str = "elo_estimate",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Create visualization plots for experiment results.

        Args:
            experiment_id: Experiment ID
            metric: Metric to visualize
            save_path: Optional path to save plot
        """
        if not HAS_MATPLOTLIB:
            print("matplotlib not available for plotting")
            return

        experiment = self.load_experiment(experiment_id)
        if experiment is None:
            print(f"Experiment not found: {experiment_id}")
            return

        completed = [t for t in experiment.trials if t.status == "completed"]
        if not completed:
            print("No completed trials to plot")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Experiment: {experiment.name}", fontsize=14, fontweight="bold")

        # 1. Metric distribution
        ax1 = axes[0, 0]
        values = [t.get_primary_metric(metric) for t in completed]
        ax1.hist(values, bins=min(20, len(values)), edgecolor="black", alpha=0.7)
        ax1.axvline(
            np.mean(values),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(values):.2f}",
        )
        ax1.axvline(
            np.max(values),
            color="green",
            linestyle="--",
            label=f"Best: {np.max(values):.2f}",
        )
        ax1.set_xlabel(metric)
        ax1.set_ylabel("Count")
        ax1.set_title(f"{metric} Distribution")
        ax1.legend()

        # 2. Trial progression
        ax2 = axes[0, 1]
        trial_nums = range(1, len(completed) + 1)
        values = [t.get_primary_metric(metric) for t in completed]
        ax2.plot(trial_nums, values, "b-o", alpha=0.6)
        ax2.axhline(np.mean(values), color="red", linestyle="--", alpha=0.5)

        # Running best
        running_best = np.maximum.accumulate(values)
        ax2.plot(trial_nums, running_best, "g-", linewidth=2, label="Running Best")

        ax2.set_xlabel("Trial Number")
        ax2.set_ylabel(metric)
        ax2.set_title("Trial Progression")
        ax2.legend()

        # 3. Loss vs ELO scatter
        ax3 = axes[1, 0]
        losses = [t.final_loss for t in completed]
        elos = [t.elo_estimate for t in completed]
        ax3.scatter(losses, elos, alpha=0.6)
        ax3.set_xlabel("Final Loss")
        ax3.set_ylabel("ELO Estimate")
        ax3.set_title("Loss vs ELO Correlation")

        # 4. Training time vs performance
        ax4 = axes[1, 1]
        times = [t.training_time_seconds / 3600 for t in completed]
        values = [t.get_primary_metric(metric) for t in completed]
        ax4.scatter(times, values, alpha=0.6)
        ax4.set_xlabel("Training Time (hours)")
        ax4.set_ylabel(metric)
        ax4.set_title("Training Time vs Performance")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved: {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_hyperparameter_importance(
        self,
        experiment_id: str,
        metric: str = "elo_estimate",
        save_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Analyze and plot hyperparameter importance.

        Uses correlation between hyperparameter values and performance.

        Args:
            experiment_id: Experiment ID
            metric: Target metric
            save_path: Optional path to save plot

        Returns:
            Dictionary of parameter importance scores
        """
        experiment = self.load_experiment(experiment_id)
        if experiment is None:
            return {}

        completed = [t for t in experiment.trials if t.status == "completed"]
        if len(completed) < 5:
            print("Need at least 5 completed trials for importance analysis")
            return {}

        # Extract parameter values and target metric
        param_values: Dict[str, List[float]] = {}
        target_values = [t.get_primary_metric(metric) for t in completed]

        for trial in completed:
            flat_config = (
                config_to_flat_dict(
                    HyperparameterConfig(
                        **{
                            k: v
                            for k, v in trial.config_dict.items()
                            if not isinstance(v, dict)
                        }
                    )
                )
                if trial.config_dict
                else {}
            )

            # Also handle nested configs
            for section in ["model", "training", "mcts", "self_play", "buffer"]:
                if section in trial.config_dict:
                    for key, value in trial.config_dict[section].items():
                        if isinstance(value, (int, float)):
                            param_name = f"{section}.{key}"
                            if param_name not in param_values:
                                param_values[param_name] = []
                            param_values[param_name].append(value)

        # Calculate correlations
        importances = {}
        for param_name, values in param_values.items():
            if len(values) == len(target_values) and len(set(values)) > 1:
                correlation = np.corrcoef(values, target_values)[0, 1]
                if not np.isnan(correlation):
                    importances[param_name] = abs(correlation)

        # Sort by importance
        importances = dict(
            sorted(importances.items(), key=lambda x: x[1], reverse=True)
        )

        # Plot if matplotlib available
        if HAS_MATPLOTLIB and importances:
            fig, ax = plt.subplots(figsize=(12, max(6, len(importances) * 0.3)))

            params = list(importances.keys())[:20]  # Top 20
            scores = [importances[p] for p in params]

            y_pos = np.arange(len(params))
            bars = ax.barh(y_pos, scores, alpha=0.7)

            # Color bars by importance
            cmap = plt.get_cmap("RdYlGn")
            for bar, score in zip(bars, scores):
                bar.set_color(cmap(score))

            ax.set_yticks(y_pos)
            ax.set_yticklabels(params)
            ax.invert_yaxis()
            ax.set_xlabel("Absolute Correlation with " + metric)
            ax.set_title("Hyperparameter Importance")
            ax.set_xlim(0, 1)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"Importance plot saved: {save_path}")
            else:
                plt.show()

            plt.close()

        return importances


def create_experiment_summary_table(
    tracker: ExperimentTracker,
    experiment_ids: Optional[List[str]] = None,
) -> str:
    """
    Create a formatted summary table of experiments.

    Args:
        tracker: ExperimentTracker instance
        experiment_ids: Optional list of specific experiments (default: all)

    Returns:
        Formatted table string
    """
    if experiment_ids is None:
        experiment_ids = [eid for eid, _ in tracker.list_experiments()]

    lines = []
    lines.append("┌" + "─" * 98 + "┐")
    lines.append(
        f"│ {'Experiment':<30} │ {'Trials':>8} │ {'Best ELO':>10} │ {'Avg ELO':>10} │ {'Best Loss':>10} │ {'Status':<10} │"
    )
    lines.append("├" + "─" * 98 + "┤")

    for exp_id in experiment_ids:
        exp = tracker.load_experiment(exp_id)
        if exp is None:
            continue

        completed = [t for t in exp.trials if t.status == "completed"]
        best = exp.get_best_trial("elo_estimate") if completed else None
        stats = exp.get_statistics("elo_estimate")

        best_elo = f"{best.elo_estimate:.1f}" if best else "N/A"
        avg_elo = f"{stats.get('mean', 0):.1f}" if stats else "N/A"
        best_loss = f"{best.final_loss:.4f}" if best else "N/A"

        name = exp.name[:28] + ".." if len(exp.name) > 30 else exp.name
        lines.append(
            f"│ {name:<30} │ {len(completed):>8} │ {best_elo:>10} │ {avg_elo:>10} │ {best_loss:>10} │ {exp.status:<10} │"
        )

    lines.append("└" + "─" * 98 + "┘")

    return "\n".join(lines)


if __name__ == "__main__":
    # Demo usage
    print("=== Experiment Tracker Demo ===\n")

    tracker = ExperimentTracker(base_dir="/tmp/chess_experiments")

    # Create experiment
    from chess_engine.tuning.hyperparameter_config import HyperparameterConfig

    config = HyperparameterConfig(name="demo")
    exp = tracker.create_experiment(
        name="Demo Experiment",
        description="Testing the experiment tracker",
        base_config=config,
    )

    # Simulate some trials
    import random

    for i in range(5):
        result = tracker.create_trial(exp.experiment_id, config)
        result.status = "completed"
        result.elo_estimate = 1500 + random.gauss(0, 100)
        result.final_loss = 0.5 + random.gauss(0, 0.1)
        result.win_rate_vs_baseline = 0.5 + random.gauss(0, 0.1)
        result.training_time_seconds = 3600 + random.gauss(0, 600)

        tracker.record_trial_result(exp.experiment_id, result)

    # Generate report
    print(tracker.generate_report(exp.experiment_id))

    # Summary table
    print("\n" + create_experiment_summary_table(tracker))
