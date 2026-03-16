import io
import os
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding="utf-8")

"""
Residual Block Count Experiment: blocks_10 vs blocks_12 vs blocks_15

Trains each preset for identical gradient-step budgets on the same synthetic
chess dataset and records:
  - Training throughput (steps/sec)
  - Policy loss, value loss, combined loss per step
  - Loss drop (first → last step)

Because self-play data collection is omitted, all three presets start from
the same pre-filled buffer of randomly-generated chess positions, isolating
the effect of block count on training dynamics.

By default the CNN filter width is scaled down (--filters 64) so the
experiment completes quickly.  Pass --filters 256 to measure the production
architecture at the cost of longer runtime.

Results are printed as a side-by-side table and saved to:
  logs/block_count_experiment/benchmark_results.json
  logs/block_count_experiment/<preset>_steps.csv

Usage:
  # Quick run (scaled-down filters, CPU)
  python scripts/experiment_block_count.py

  # Full-size model, GPU, more steps
  python scripts/experiment_block_count.py --filters 256 --steps 100 --device cuda

  # Custom output directory
  python scripts/experiment_block_count.py --output-dir results/blocks
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from chess_engine.config import RESIDUAL_BLOCK_PRESETS
from chess_engine.data.replay.buffer import ReplayBuffer
from chess_engine.data.replay.storage import GameExample
from chess_engine.models.hybrid.config import HybridModelConfig
from chess_engine.models.hybrid.hybrid_net import HybridChessNet
from chess_engine.training.rl.rl_config import RLTrainingConfig
from chess_engine.training.rl.train_step import execute_training_step
from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder


# -----------------------------------------------------------------------------
# Result container
# -----------------------------------------------------------------------------

@dataclass
class PresetResult:
    preset: str              # e.g. "blocks_10"
    num_blocks: int
    num_filters: int
    n_positions: int
    n_steps: int
    step_times_s: List[float]
    losses: List[float]
    policy_losses: List[float]
    value_losses: List[float]

    @property
    def mean_time(self) -> float:
        return float(np.mean(self.step_times_s))

    @property
    def steps_per_sec(self) -> float:
        return 1.0 / self.mean_time if self.mean_time > 0 else 0.0

    @property
    def loss_drop(self) -> float:
        return self.losses[0] - self.losses[-1]

    @property
    def final_loss(self) -> float:
        return self.losses[-1]


# -----------------------------------------------------------------------------
# Synthetic data generation (identical to benchmark_per.py for comparability)
# -----------------------------------------------------------------------------

def generate_synthetic_positions(n: int) -> List[GameExample]:
    """
    Generate n random chess positions with realistic-looking policy
    distributions and value targets drawn from [-1, 1].
    Seed is fixed so all presets train on exactly the same data.
    """
    rng = np.random.default_rng(seed=42)
    board = chess.Board()
    examples: List[GameExample] = []

    for i in range(n):
        legal_moves = list(board.legal_moves)

        if legal_moves:
            weights = rng.exponential(scale=1.0, size=len(legal_moves))
            weights /= weights.sum()
            policy = {mv.uci(): float(w) for mv, w in zip(legal_moves, weights)}
        else:
            policy = {}

        value = float(rng.uniform(-1.0, 1.0))
        examples.append(
            GameExample(fen=board.fen(), policy=policy, value=value, move_number=i)
        )

        if legal_moves and i % 60 != 59 and not board.is_game_over():
            board.push(rng.choice(legal_moves))  # type: ignore[arg-type]
        else:
            board = chess.Board()

    return examples


# -----------------------------------------------------------------------------
# Model + training setup
# -----------------------------------------------------------------------------

def build_model(num_blocks: int, num_filters: int, device: torch.device) -> HybridChessNet:
    """Build a CNN-only HybridChessNet with the given block and filter counts."""
    cfg = HybridModelConfig(
        cnn_input_channels=22,
        cnn_filters=num_filters,
        cnn_residual_blocks=num_blocks,
        use_rnn=False,
        num_actions=4672,
    )
    return HybridChessNet(cfg).to(device)


def build_train_config(batch_size: int) -> RLTrainingConfig:
    """Minimal RLTrainingConfig for single-step timing."""
    return RLTrainingConfig(
        batch_size=batch_size,
        training_steps_per_iteration=1,  # one step per call
        use_prioritized_replay=False,
        log_frequency=99999,             # suppress per-step tensorboard noise
        device="cpu",
        use_amp=False,
    )


# -----------------------------------------------------------------------------
# Benchmark runner for one preset
# -----------------------------------------------------------------------------

def run_preset(
    preset_name: str,
    num_blocks: int,
    num_filters: int,
    examples: List[GameExample],
    n_steps: int,
    batch_size: int,
    device: torch.device,
    output_dir: str,
) -> PresetResult:
    param_count = _count_params(num_blocks, num_filters, device)

    print(f"\n{'-'*60}")
    print(f"  Preset   : {preset_name}")
    print(f"  Blocks   : {num_blocks}  |  Filters: {num_filters}")
    print(f"  Params   : {param_count:,}")
    print(f"  Buffer   : {len(examples)} positions")
    print(f"  Steps    : {n_steps}  |  Batch: {batch_size}")
    print(f"{'-'*60}")

    model = build_model(num_blocks, num_filters, device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    value_criterion = nn.MSELoss()

    buffer = ReplayBuffer(max_size=len(examples) + 1)
    buffer.add_game_examples(examples)

    config = build_train_config(batch_size=batch_size)

    board_encoder = BoardEncoder()
    move_encoder = MoveEncoder()

    log_path = os.path.join(output_dir, f"tb_{preset_name}")
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)

    step_times, losses, policy_losses, value_losses = [], [], [], []
    total_steps = 0

    for step in range(1, n_steps + 1):
        t0 = time.perf_counter()
        metrics, total_steps, _ = execute_training_step(
            model=model,
            replay_buffer=buffer,
            optimizer=optimizer,
            config=config,
            board_encoder=board_encoder,
            move_encoder=move_encoder,
            device=device,
            use_amp=False,
            scaler=None,
            value_criterion=value_criterion,
            writer=writer,
            current_iteration=total_steps,
            total_training_steps=total_steps,
            per_beta=None,
        )
        elapsed = time.perf_counter() - t0

        step_times.append(elapsed)
        losses.append(metrics["loss"])
        policy_losses.append(metrics["policy_loss"])
        value_losses.append(metrics["value_loss"])

        print(
            f"  step {step:>3}/{n_steps}  "
            f"loss={metrics['loss']:.4f}  "
            f"(policy={metrics['policy_loss']:.4f}  value={metrics['value_loss']:.4f})  "
            f"time={elapsed*1000:.1f}ms"
        )

    writer.close()

    return PresetResult(
        preset=preset_name,
        num_blocks=num_blocks,
        num_filters=num_filters,
        n_positions=len(examples),
        n_steps=n_steps,
        step_times_s=step_times,
        losses=losses,
        policy_losses=policy_losses,
        value_losses=value_losses,
    )


def _count_params(num_blocks: int, num_filters: int, device: torch.device) -> int:
    """Return total trainable parameter count for a given config."""
    model = build_model(num_blocks, num_filters, device)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def save_csv(result: PresetResult, output_dir: str) -> None:
    path = os.path.join(output_dir, f"{result.preset}_steps.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time_s", "loss", "policy_loss", "value_loss"])
        for i, (t, l, pl, vl) in enumerate(
            zip(result.step_times_s, result.losses, result.policy_losses, result.value_losses),
            start=1,
        ):
            writer.writerow([i, f"{t:.6f}", f"{l:.6f}", f"{pl:.6f}", f"{vl:.6f}"])
    print(f"  Saved: {path}")


def print_summary(results: List[PresetResult]) -> None:
    print("\n" + "=" * 75)
    print("  BLOCK COUNT EXPERIMENT SUMMARY")
    print("=" * 75)

    # Header
    col = 20
    header = f"  {'Metric':<28}"
    for r in results:
        header += f"  {r.preset:>{col}}"
    print(header)
    print("  " + "-" * (28 + (col + 2) * len(results)))

    def row(label: str, values: List[str]) -> str:
        line = f"  {label:<28}"
        for v in values:
            line += f"  {v:>{col}}"
        return line

    rows_data = [
        ("Blocks", [str(r.num_blocks) for r in results]),
        ("Filters", [str(r.num_filters) for r in results]),
        ("Params", [f"{_count_params(r.num_blocks, r.num_filters, torch.device('cpu')):,}" for r in results]),
        ("Mean time/step (ms)", [f"{r.mean_time*1000:.1f}" for r in results]),
        ("Steps/sec", [f"{r.steps_per_sec:.2f}" for r in results]),
        ("First-step loss", [f"{r.losses[0]:.4f}" for r in results]),
        ("Final loss", [f"{r.final_loss:.4f}" for r in results]),
        ("Loss drop (first→last)", [f"{r.loss_drop:.4f}" for r in results]),
        ("Mean policy loss", [f"{float(np.mean(r.policy_losses)):.4f}" for r in results]),
        ("Mean value loss", [f"{float(np.mean(r.value_losses)):.4f}" for r in results]),
    ]

    for label, values in rows_data:
        print(row(label, values))

    # Relative speed vs baseline (blocks_10)
    if len(results) > 1:
        base_time = results[0].mean_time
        print(f"\n  Speed relative to {results[0].preset}:")
        for r in results:
            rel = r.mean_time / base_time
            print(f"    {r.preset:<14}: {rel:+.2f}x  ({(rel-1)*100:+.1f}%)")

    print("=" * 75)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark residual block count presets: blocks_10, blocks_12, blocks_15"
    )
    parser.add_argument(
        "--n-positions", type=int, default=5000,
        help="Number of synthetic positions in the replay buffer (default: 5000)",
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Number of training steps per preset (default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Training batch size (default: 64)",
    )
    parser.add_argument(
        "--filters", type=int, default=64,
        help=(
            "Override CNN filter count for all presets (default: 64 for quick runs). "
            "Pass 256 to benchmark the production architecture."
        ),
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device: 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="logs/block_count_experiment",
        help="Directory for CSV and JSON output (default: logs/block_count_experiment)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  RESIDUAL BLOCK COUNT EXPERIMENT")
    print("=" * 60)
    print(f"  Positions  : {args.n_positions}")
    print(f"  Steps      : {args.steps} per preset")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Filters    : {args.filters}  (use --filters 256 for full model)")
    print(f"  Device     : {device}")
    print(f"  Output     : {args.output_dir}")

    print(f"\n  Generating {args.n_positions} synthetic positions...")
    t0 = time.perf_counter()
    examples = generate_synthetic_positions(args.n_positions)
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    # Run each preset
    results: List[PresetResult] = []
    for preset_name, preset_cfg in RESIDUAL_BLOCK_PRESETS.items():
        num_blocks = preset_cfg["num_residual_blocks"]
        # Allow --filters to override the preset's filter count for quick runs
        num_filters = args.filters if args.filters != preset_cfg["num_filters"] else preset_cfg["num_filters"]

        result = run_preset(
            preset_name=preset_name,
            num_blocks=num_blocks,
            num_filters=num_filters,
            examples=examples,
            n_steps=args.steps,
            batch_size=args.batch_size,
            device=device,
            output_dir=args.output_dir,
        )
        results.append(result)

    # Print summary table
    print_summary(results)

    # Save per-preset CSVs
    print()
    for r in results:
        save_csv(r, args.output_dir)

    # Save JSON summary
    summary = {
        "config": {
            "n_positions": args.n_positions,
            "n_steps": args.steps,
            "batch_size": args.batch_size,
            "filters_override": args.filters,
            "device": str(device),
        },
        "results": {
            r.preset: {
                "num_blocks": r.num_blocks,
                "num_filters": r.num_filters,
                "param_count": _count_params(r.num_blocks, r.num_filters, device),
                "mean_time_ms": round(r.mean_time * 1000, 2),
                "steps_per_sec": round(r.steps_per_sec, 3),
                "first_loss": round(r.losses[0], 6),
                "final_loss": round(r.final_loss, 6),
                "loss_drop": round(r.loss_drop, 6),
                "mean_policy_loss": round(float(np.mean(r.policy_losses)), 6),
                "mean_value_loss": round(float(np.mean(r.value_losses)), 6),
            }
            for r in results
        },
    }

    # Add relative speed vs baseline
    if results:
        base_time = results[0].mean_time
        for r in results:
            summary["results"][r.preset]["relative_speed_vs_blocks_10"] = round(
                r.mean_time / base_time, 4
            )

    json_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")
    print("\nExperiment complete.")
    print(
        "\nNext step: run with --filters 256 --device cuda --steps 100 after training"
        " reaches a useful checkpoint, then update RESIDUAL_BLOCK_PRESET in config.py"
        " with the best-performing value."
    )


if __name__ == "__main__":
    main()
