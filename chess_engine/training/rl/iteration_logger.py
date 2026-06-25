"""
RL TRAINER - Iteration Logger

Writes one row per training iteration to a persistent CSV and JSON file so
that every metric visible in the console is also available for offline analysis
(e.g. pandas DataFrames for a thesis).

Files written to checkpoint_dir:
    training_log.csv      -- append-mode CSV, one row per iteration
    training_log.json     -- same data as a JSON array (updated in-place)
    run_manifest.json     -- run metadata: phase, config snapshot, start time

The phase label is derived from the checkpoint_dir basename so no extra CLI
flag is needed (e.g. "data/rl_checkpoints/phase1" -> phase = "phase1").
"""

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# All columns in the CSV, in order.
CSV_COLUMNS = [
    # --- identity ---
    "iteration",
    "timestamp_utc",
    "phase",
    # --- losses ---
    "policy_loss",
    "value_loss",
    "total_loss",
    # --- value head diagnostics ---
    "value_correlation",
    "mean_mcts_value",
    "std_mcts_value",
    "std_pred_value",
    "min_mcts_value",
    "max_mcts_value",
    # --- fusion gate (gated fusion only; None otherwise) ---
    "gate_mean",
    # --- evaluation ---
    "eval_win_rate",
    # --- self-play game stats ---
    "white_win_pct",
    "black_win_pct",
    "draw_pct",
    "resign_pct",
    "avg_moves_per_game",
    "total_games",
    # --- training state ---
    "buffer_size",
    "games_played",
    "training_steps",
    "learning_rate",
    # --- timing (seconds) ---
    "selfplay_seconds",
    "train_seconds",
    "iteration_seconds",
]


class IterationLogger:
    """
    Appends one row to training_log.csv after every iteration.

    Designed to be crash-safe: the CSV is opened in append mode so partial
    runs accumulate correctly across resumes.  The JSON mirror is rewritten
    fully each iteration (it stays small enough that this is fine).
    """

    def __init__(self, checkpoint_dir: str, config_dict: Dict[str, Any]):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Derive phase name from the last component of the checkpoint path.
        # "data/rl_checkpoints/phase1" -> "phase1"
        self.phase = self.checkpoint_dir.name

        self.csv_path = self.checkpoint_dir / "training_log.csv"
        self.json_path = self.checkpoint_dir / "training_log.json"
        self.manifest_path = self.checkpoint_dir / "run_manifest.json"

        # In-memory list mirrors the JSON file.  Pre-load existing rows if
        # resuming so the JSON file stays complete.
        self._rows: list = []
        if self.json_path.exists():
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    self._rows = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._rows = []

        # Write manifest on every init (picks up resumed runs correctly).
        self._write_manifest(config_dict)

        # Write CSV header only if starting a fresh file.
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(
        self,
        iteration: int,
        train_metrics: Dict[str, Any],
        selfplay_stats,  # SelfPlayStatistics or None
        eval_metrics: Optional[Dict[str, Any]],
        buffer_size: int,
        games_played: int,
        training_steps: int,
        learning_rate: float,
        selfplay_seconds: float,
        train_seconds: float,
        iteration_seconds: float,
    ) -> None:
        """Write one row for the completed iteration."""

        # --- build row dict ---
        row: Dict[str, Any] = {
            "iteration": iteration,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "phase": self.phase,
            # losses
            "policy_loss": _fmt(train_metrics.get("policy_loss")),
            "value_loss": _fmt(train_metrics.get("value_loss")),
            "total_loss": _fmt(train_metrics.get("loss")),
            # value head diagnostics (all come from train_metrics)
            "value_correlation": _fmt(train_metrics.get("value_correlation")),
            "mean_mcts_value": _fmt(train_metrics.get("mean_mcts_value")),
            "std_mcts_value": _fmt(train_metrics.get("std_mcts_value")),
            "std_pred_value": _fmt(train_metrics.get("std_pred_value")),
            "min_mcts_value": _fmt(train_metrics.get("min_mcts_value")),
            "max_mcts_value": _fmt(train_metrics.get("max_mcts_value")),
            "gate_mean": _fmt(train_metrics.get("gate_mean")),
            # evaluation
            "eval_win_rate": _fmt(
                eval_metrics.get("win_rate") if eval_metrics else None
            ),
            # self-play game stats
            "white_win_pct": _fmt(
                selfplay_stats.white_win_rate if selfplay_stats else None
            ),
            "black_win_pct": _fmt(
                selfplay_stats.black_win_rate if selfplay_stats else None
            ),
            "draw_pct": _fmt(selfplay_stats.draw_rate if selfplay_stats else None),
            "resign_pct": _fmt(
                selfplay_stats.resignation_rate if selfplay_stats else None
            ),
            "avg_moves_per_game": _fmt(
                selfplay_stats.avg_moves_per_game if selfplay_stats else None
            ),
            "total_games": selfplay_stats.total_games if selfplay_stats else None,
            # training state
            "buffer_size": buffer_size,
            "games_played": games_played,
            "training_steps": training_steps,
            "learning_rate": _fmt(learning_rate, decimals=8),
            # timing
            "selfplay_seconds": round(selfplay_seconds, 1),
            "train_seconds": round(train_seconds, 1),
            "iteration_seconds": round(iteration_seconds, 1),
        }

        # Append to CSV
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(row)

        # Update JSON mirror (replace existing row for this iteration if any)
        existing_iters = {r["iteration"] for r in self._rows}
        if iteration in existing_iters:
            self._rows = [r for r in self._rows if r["iteration"] != iteration]
        self._rows.append(row)
        self._rows.sort(key=lambda r: r["iteration"])

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self._rows, f, indent=2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_manifest(self, config_dict: Dict[str, Any]) -> None:
        """Write or update the run manifest."""
        # Load existing manifest to preserve first_start_utc on resume.
        existing: Dict[str, Any] = {}
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        manifest = {
            "phase": self.phase,
            "first_start_utc": existing.get(
                "first_start_utc",
                datetime.now(timezone.utc).isoformat(),
            ),
            "last_resume_utc": datetime.now(timezone.utc).isoformat(),
            "config": config_dict,
        }

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)


def _fmt(value, decimals: int = 6) -> Any:
    """Round a float for storage, pass None through unchanged."""
    if value is None:
        return None
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return value
