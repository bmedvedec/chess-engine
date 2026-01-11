"""
CHAPTER 13: COMPARISON AND REPORTING
Tools for Analyzing and Comparing Hyperparameter Experiments

This module provides:
- Side-by-side experiment comparison
- Statistical analysis of results
- Report generation in multiple formats
- Performance visualization
- Best configuration extraction
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from chess_engine.tuning.hyperparameter_config import (
    HyperparameterConfig,
    config_to_flat_dict,
    compare_configs,
)
from chess_engine.tuning.experiment_tracker import (
    ExperimentTracker,
    ExperimentResult,
    Experiment,
)


@dataclass
class ComparisonResult:
    """Results of comparing multiple experiments or configurations"""

    experiments: List[str]
    metrics_comparison: Dict[str, Dict[str, float]]
    best_experiment: str
    best_config: Optional[Dict[str, Any]]
    statistical_tests: Dict[str, Any]
    recommendations: List[str]


class TuningAnalyzer:
    """
    Analyzes and compares hyperparameter tuning experiments.

    Features:
    - Statistical comparison of experiments
    - Parameter importance analysis
    - Best configuration extraction
    - Visualization generation
    """

    def __init__(self, tracker: ExperimentTracker):
        """
        Initialize analyzer with experiment tracker.

        Args:
            tracker: ExperimentTracker with experiments to analyze
        """
        self.tracker = tracker

    def compare_experiments(
        self,
        experiment_ids: List[str],
        metric: str = "elo_estimate",
    ) -> ComparisonResult:
        """
        Compare multiple experiments statistically.

        Args:
            experiment_ids: List of experiment IDs to compare
            metric: Metric to compare

        Returns:
            ComparisonResult with comparison details
        """
        metrics_comparison = {}
        all_scores = {}

        for exp_id in experiment_ids:
            exp = self.tracker.load_experiment(exp_id)
            if exp is None:
                continue

            completed = [t for t in exp.trials if t.status == "completed"]
            if not completed:
                continue

            scores = [t.get_primary_metric(metric) for t in completed]
            all_scores[exp_id] = scores

            metrics_comparison[exp_id] = {
                "name": exp.name,
                "n_trials": len(completed),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "median": float(np.median(scores)),
                "q25": float(np.percentile(scores, 25)),
                "q75": float(np.percentile(scores, 75)),
            }

        # Determine best experiment
        best_exp = max(
            metrics_comparison.keys(), key=lambda x: metrics_comparison[x]["max"]
        )

        # Get best config from best experiment
        best_exp_data = self.tracker.load_experiment(best_exp)
        best_trial = best_exp_data.get_best_trial(metric) if best_exp_data else None
        best_config = best_trial.config_dict if best_trial else None

        # Statistical tests (simplified - Mann-Whitney U for pairwise comparison)
        statistical_tests = {}
        if len(experiment_ids) >= 2:
            for i, exp1 in enumerate(experiment_ids):
                for exp2 in experiment_ids[i + 1 :]:
                    if exp1 in all_scores and exp2 in all_scores:
                        # Simple comparison without scipy
                        scores1 = all_scores[exp1]
                        scores2 = all_scores[exp2]

                        mean_diff = np.mean(scores1) - np.mean(scores2)
                        pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)

                        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0

                        statistical_tests[f"{exp1}_vs_{exp2}"] = {
                            "mean_difference": float(mean_diff),
                            "effect_size": float(effect_size),
                            "exp1_better": mean_diff > 0,
                        }

        # Generate recommendations
        recommendations = self._generate_recommendations(
            metrics_comparison, statistical_tests, metric
        )

        return ComparisonResult(
            experiments=experiment_ids,
            metrics_comparison=metrics_comparison,
            best_experiment=best_exp,
            best_config=best_config,
            statistical_tests=statistical_tests,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        metrics: Dict[str, Dict[str, float]],
        tests: Dict[str, Any],
        metric: str,
    ) -> List[str]:
        """Generate actionable recommendations from comparison"""
        recommendations = []

        if not metrics:
            return ["No experiments to compare"]

        # Find best and worst
        sorted_exps = sorted(
            metrics.keys(), key=lambda x: metrics[x]["max"], reverse=True
        )
        best = sorted_exps[0]

        recommendations.append(
            f"Best experiment: {metrics[best]['name']} (max {metric}: {metrics[best]['max']:.2f})"
        )

        # Check for high variance
        for exp_id, data in metrics.items():
            if data["std"] > 0.2 * abs(data["mean"]):
                recommendations.append(
                    f"⚠️ {data['name']}: High variance (std={data['std']:.2f}). Consider more trials."
                )

        # Check for significant differences
        for test_name, test_data in tests.items():
            if abs(test_data["effect_size"]) > 0.5:
                exp1, exp2 = test_name.split("_vs_")
                better = exp1 if test_data["exp1_better"] else exp2
                recommendations.append(
                    f"Strong effect: {better} significantly outperforms (effect size: {test_data['effect_size']:.2f})"
                )

        return recommendations

    def analyze_parameter_sensitivity(
        self,
        experiment_id: str,
        metric: str = "elo_estimate",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze how sensitive performance is to each parameter.

        Args:
            experiment_id: Experiment to analyze
            metric: Target metric

        Returns:
            Dictionary mapping parameters to sensitivity analysis
        """
        exp = self.tracker.load_experiment(experiment_id)
        if exp is None:
            return {}

        completed = [t for t in exp.trials if t.status == "completed"]
        if len(completed) < 5:
            return {
                "_error": {
                    "message": "Need at least 5 completed trials",
                    "count": len(completed),
                }
            }

        # Extract parameter values and scores
        param_data: Dict[str, List[Tuple[float, float]]] = {}

        for trial in completed:
            score = trial.get_primary_metric(metric)

            # Get flattened config
            for section in ["model", "training", "mcts", "self_play", "buffer"]:
                if section in trial.config_dict:
                    for key, value in trial.config_dict[section].items():
                        if isinstance(value, (int, float)):
                            param_name = f"{section}.{key}"
                            if param_name not in param_data:
                                param_data[param_name] = []
                            param_data[param_name].append((value, score))

        # Analyze each parameter
        sensitivity = {}

        for param_name, data in param_data.items():
            if len(data) < 3:
                continue

            values, scores = zip(*data)
            values = np.array(values)
            scores = np.array(scores)

            # Check for sufficient variation
            if np.std(values) < 1e-10:
                continue

            # Correlation
            correlation = (
                np.corrcoef(values, scores)[0, 1] if len(set(values)) > 1 else 0
            )

            # Linear regression slope (normalized)
            if np.std(values) > 0:
                slope = np.cov(values, scores)[0, 1] / np.var(values)
                normalized_slope = slope * np.std(values) / (np.std(scores) + 1e-10)
            else:
                normalized_slope = 0

            # Value ranges and their average scores
            sorted_idx = np.argsort(values)
            n = len(values)

            low_scores = scores[sorted_idx[: n // 3]]
            mid_scores = scores[sorted_idx[n // 3 : 2 * n // 3]]
            high_scores = scores[sorted_idx[2 * n // 3 :]]

            sensitivity[param_name] = {
                "correlation": float(correlation) if not np.isnan(correlation) else 0,
                "abs_correlation": (
                    float(abs(correlation)) if not np.isnan(correlation) else 0
                ),
                "normalized_slope": float(normalized_slope),
                "value_range": (float(np.min(values)), float(np.max(values))),
                "low_value_avg_score": (
                    float(np.mean(low_scores)) if len(low_scores) > 0 else 0
                ),
                "mid_value_avg_score": (
                    float(np.mean(mid_scores)) if len(mid_scores) > 0 else 0
                ),
                "high_value_avg_score": (
                    float(np.mean(high_scores)) if len(high_scores) > 0 else 0
                ),
                "optimal_direction": (
                    "increase"
                    if correlation > 0.1
                    else ("decrease" if correlation < -0.1 else "neutral")
                ),
            }

        # Sort by absolute correlation
        sensitivity = dict(
            sorted(
                sensitivity.items(), key=lambda x: x[1]["abs_correlation"], reverse=True
            )
        )

        return sensitivity

    def get_best_configuration(
        self,
        experiment_ids: Optional[List[str]] = None,
        metric: str = "elo_estimate",
    ) -> Tuple[Optional[HyperparameterConfig], Optional[ExperimentResult]]:
        """
        Get the best configuration across experiments.

        Args:
            experiment_ids: Experiments to search (default: all)
            metric: Metric to optimize

        Returns:
            Tuple of (best_config, best_trial_result)
        """
        if experiment_ids is None:
            experiment_ids = [eid for eid, _ in self.tracker.list_experiments()]

        best_score = float("-inf")
        best_trial = None

        for exp_id in experiment_ids:
            exp = self.tracker.load_experiment(exp_id)
            if exp is None:
                continue

            trial = exp.get_best_trial(metric)
            if trial is None:
                continue

            score = trial.get_primary_metric(metric)
            if score > best_score:
                best_score = score
                best_trial = trial

        if best_trial is None:
            return None, None

        # Reconstruct config
        config = HyperparameterConfig()
        if best_trial.config_dict:
            for section in ["model", "training", "mcts", "self_play", "buffer"]:
                if section in best_trial.config_dict:
                    section_config = getattr(config, section)
                    for key, value in best_trial.config_dict[section].items():
                        if hasattr(section_config, key):
                            setattr(section_config, key, value)

        return config, best_trial


class ReportGenerator:
    """
    Generates comprehensive reports from tuning experiments.

    Supports multiple output formats:
    - Text report
    - Markdown
    - JSON summary
    - HTML (with charts)
    """

    def __init__(self, tracker: ExperimentTracker):
        """Initialize with experiment tracker"""
        self.tracker = tracker
        self.analyzer = TuningAnalyzer(tracker)

    def generate_text_report(
        self,
        experiment_ids: List[str],
        output_path: Optional[str] = None,
    ) -> str:
        """Generate detailed text report"""
        lines = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines.append("=" * 80)
        lines.append("HYPERPARAMETER TUNING REPORT")
        lines.append(f"Generated: {timestamp}")
        lines.append("=" * 80)

        # Comparison
        comparison = self.analyzer.compare_experiments(experiment_ids)

        lines.append("\n" + "─" * 40)
        lines.append("EXPERIMENT COMPARISON")
        lines.append("─" * 40)

        for exp_id, metrics in comparison.metrics_comparison.items():
            lines.append(f"\n{metrics['name']}:")
            lines.append(f"  Trials: {metrics['n_trials']}")
            lines.append(f"  Mean ELO: {metrics['mean']:.1f} ± {metrics['std']:.1f}")
            lines.append(f"  Best ELO: {metrics['max']:.1f}")
            lines.append(f"  Range: [{metrics['min']:.1f}, {metrics['max']:.1f}]")

        # Recommendations
        lines.append("\n" + "─" * 40)
        lines.append("RECOMMENDATIONS")
        lines.append("─" * 40)

        for rec in comparison.recommendations:
            lines.append(f"• {rec}")

        # Best configuration
        lines.append("\n" + "─" * 40)
        lines.append("BEST CONFIGURATION")
        lines.append("─" * 40)

        best_config, best_trial = self.analyzer.get_best_configuration(experiment_ids)

        if best_trial:
            lines.append(f"\nFrom: {best_trial.experiment_id}")
            lines.append(f"Trial: {best_trial.trial_id}")
            lines.append(f"ELO: {best_trial.elo_estimate:.1f}")
            lines.append(f"Loss: {best_trial.final_loss:.4f}")

            if best_trial.config_dict:
                lines.append("\nParameters:")
                for section in ["model", "training", "mcts"]:
                    if section in best_trial.config_dict:
                        lines.append(f"\n  {section.upper()}:")
                        for key, value in best_trial.config_dict[section].items():
                            lines.append(f"    {key}: {value}")

        # Parameter sensitivity (for best experiment)
        if comparison.best_experiment:
            lines.append("\n" + "─" * 40)
            lines.append("PARAMETER SENSITIVITY")
            lines.append("─" * 40)

            sensitivity = self.analyzer.analyze_parameter_sensitivity(
                comparison.best_experiment
            )

            for param, data in list(sensitivity.items())[:10]:  # Top 10
                lines.append(f"\n{param}:")
                lines.append(f"  Correlation: {data['correlation']:.3f}")
                lines.append(f"  Optimal: {data['optimal_direction']}")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"📄 Report saved: {output_path}")

        return report

    def generate_markdown_report(
        self,
        experiment_ids: List[str],
        output_path: Optional[str] = None,
    ) -> str:
        """Generate Markdown format report"""
        lines = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines.append("# Hyperparameter Tuning Report")
        lines.append(f"\n*Generated: {timestamp}*\n")

        # Comparison
        comparison = self.analyzer.compare_experiments(experiment_ids)

        lines.append("## Experiment Comparison\n")
        lines.append("| Experiment | Trials | Mean ELO | Std | Best ELO |")
        lines.append("|------------|--------|----------|-----|----------|")

        for exp_id, metrics in comparison.metrics_comparison.items():
            lines.append(
                f"| {metrics['name']} | {metrics['n_trials']} | "
                f"{metrics['mean']:.1f} | {metrics['std']:.1f} | {metrics['max']:.1f} |"
            )

        # Recommendations
        lines.append("\n## Recommendations\n")
        for rec in comparison.recommendations:
            lines.append(f"- {rec}")

        # Best configuration
        lines.append("\n## Best Configuration\n")

        best_config, best_trial = self.analyzer.get_best_configuration(experiment_ids)

        if best_trial:
            lines.append(f"**ELO:** {best_trial.elo_estimate:.1f}")
            lines.append(f"**Loss:** {best_trial.final_loss:.4f}\n")

            if best_trial.config_dict:
                lines.append("```json")
                lines.append(json.dumps(best_trial.config_dict, indent=2))
                lines.append("```")

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"📄 Markdown report saved: {output_path}")

        return report

    def generate_json_summary(
        self,
        experiment_ids: List[str],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate JSON summary for programmatic use"""
        comparison = self.analyzer.compare_experiments(experiment_ids)
        best_config, best_trial = self.analyzer.get_best_configuration(experiment_ids)

        summary = {
            "generated_at": datetime.now().isoformat(),
            "experiments": experiment_ids,
            "metrics_comparison": comparison.metrics_comparison,
            "best_experiment": comparison.best_experiment,
            "best_configuration": comparison.best_config,
            "statistical_tests": comparison.statistical_tests,
            "recommendations": comparison.recommendations,
            "best_trial": best_trial.to_dict() if best_trial else None,
        }

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"📄 JSON summary saved: {output_path}")

        return summary

    def generate_comparison_plot(
        self,
        experiment_ids: List[str],
        metric: str = "elo_estimate",
        output_path: Optional[str] = None,
    ) -> None:
        """Generate comparison visualization"""
        if not HAS_MATPLOTLIB:
            print("⚠️ matplotlib not available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Collect data
        exp_names = []
        exp_scores = []

        for exp_id in experiment_ids:
            exp = self.tracker.load_experiment(exp_id)
            if exp is None:
                continue

            completed = [t for t in exp.trials if t.status == "completed"]
            if not completed:
                continue

            scores = [t.get_primary_metric(metric) for t in completed]
            exp_names.append(exp.name[:20])
            exp_scores.append(scores)

        if not exp_scores:
            print("No data to plot")
            return

        # 1. Box plot comparison
        ax1 = axes[0, 0]
        bp = ax1.boxplot(exp_scores, labels=exp_names, patch_artist=True)
        cmap = plt.get_cmap("Set3")
        colors = [cmap(i) for i in np.linspace(0, 1, len(exp_scores))]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
        ax1.set_ylabel(metric)
        ax1.set_title("Distribution by Experiment")
        ax1.tick_params(axis="x", rotation=45)

        # 2. Violin plot
        ax2 = axes[0, 1]
        parts = ax2.violinplot(exp_scores, showmeans=True, showmedians=True)
        ax2.set_xticks(range(1, len(exp_names) + 1))
        ax2.set_xticklabels(exp_names, rotation=45)
        ax2.set_ylabel(metric)
        ax2.set_title("Score Distribution")

        # 3. Bar chart of best scores
        ax3 = axes[1, 0]
        best_scores = [max(s) for s in exp_scores]
        bars = ax3.bar(exp_names, best_scores, color=colors)
        ax3.set_ylabel(f"Best {metric}")
        ax3.set_title("Best Score by Experiment")
        ax3.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, score in zip(bars, best_scores):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{score:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # 4. Combined trial progression
        ax4 = axes[1, 1]
        for i, (name, scores) in enumerate(zip(exp_names, exp_scores)):
            trials = range(1, len(scores) + 1)
            running_best = np.maximum.accumulate(scores)
            ax4.plot(trials, running_best, "-", label=name, alpha=0.8)

        ax4.set_xlabel("Trial Number")
        ax4.set_ylabel(f"Running Best {metric}")
        ax4.set_title("Convergence Comparison")
        ax4.legend(loc="lower right", fontsize=8)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"📊 Plot saved: {output_path}")
        else:
            plt.show()

        plt.close()


def compare_and_report(
    tracker: ExperimentTracker,
    experiment_ids: Optional[List[str]] = None,
    output_dir: str = "data/tuning_reports",
) -> Dict[str, str]:
    """
    Convenience function to generate all reports.

    Args:
        tracker: ExperimentTracker instance
        experiment_ids: Experiments to compare (default: all)
        output_dir: Directory for output files

    Returns:
        Dictionary mapping report type to file path
    """
    if experiment_ids is None:
        experiment_ids = [eid for eid, _ in tracker.list_experiments()]

    if not experiment_ids:
        print("No experiments found")
        return {}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    generator = ReportGenerator(tracker)

    outputs: Dict[str, str] = {}

    # Text report
    text_path = output_path / "report.txt"
    generator.generate_text_report(experiment_ids, str(text_path))
    outputs["text"] = str(text_path)

    # Markdown report
    md_path = output_path / "report.md"
    generator.generate_markdown_report(experiment_ids, str(md_path))
    outputs["markdown"] = str(md_path)

    # JSON summary
    json_path = output_path / "summary.json"
    generator.generate_json_summary(experiment_ids, str(json_path))
    outputs["json"] = str(json_path)

    # Comparison plot
    if HAS_MATPLOTLIB:
        plot_path = output_path / "comparison.png"
        generator.generate_comparison_plot(experiment_ids, output_path=str(plot_path))
        outputs["plot"] = str(plot_path)

    print(f"\n✅ Generated {len(outputs)} reports in {output_path}")
    return outputs


if __name__ == "__main__":
    # Demo
    print("=== Comparison and Reporting Demo ===\n")

    tracker = ExperimentTracker(base_dir="data/experiments")

    experiments = tracker.list_experiments()
    if experiments:
        exp_ids = [eid for eid, _ in experiments]
        outputs = compare_and_report(tracker, exp_ids)
        print(f"\nGenerated: {list(outputs.keys())}")
    else:
        print("No experiments found. Run hyperparameter tuning first.")
