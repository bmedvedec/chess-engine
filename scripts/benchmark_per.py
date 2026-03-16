import io
import os
import sys

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")
if isinstance(sys.stderr, io.TextIOWrapper):
    sys.stderr.reconfigure(encoding="utf-8")

"""
PER vs Uniform Replay — Training-Step Benchmark

Measures the overhead and loss-convergence difference between
Prioritized Experience Replay (PER) and uniform replay over
50 training steps, using a small synthetic replay buffer.

Because self-play data collection is intentionally omitted, both
conditions start from the same pre-filled buffer of randomly-generated
chess positions.  This isolates the sampling and gradient-update
cost of each strategy.

Metrics collected per condition (50 steps each):
  • Wall-clock time per step (s)
  • Policy loss, value loss, combined loss per step
  • Loss drop: (step-1 loss) − (step-50 loss)

Results are printed as a side-by-side table and saved to:
  logs/per_benchmark/benchmark_results.json
  logs/per_benchmark/per_steps.csv
  logs/per_benchmark/uniform_steps.csv

Usage:
  # Quick smoke test (small model, CPU-only)
  python scripts/benchmark_per.py

  # Larger run, GPU
  python scripts/benchmark_per.py --n-positions 10000 --steps 50 --device cuda

  # Custom output directory
  python scripts/benchmark_per.py --output-dir results/per_bench
"""

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from chess_engine.data.replay.buffer import ReplayBuffer
from chess_engine.data.replay.prioritized import PrioritizedReplayBuffer
from chess_engine.data.replay.storage import GameExample
from chess_engine.models.hybrid.config import HybridModelConfig
from chess_engine.models.hybrid.hybrid_net import HybridChessNet
from chess_engine.training.rl.rl_config import RLTrainingConfig
from chess_engine.training.rl.train_step import execute_training_step
from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    condition: str          # "uniform" or "per"
    n_positions: int        # buffer size
    n_steps: int            # gradient steps
    step_times_s: List[float]
    losses: List[float]
    policy_losses: List[float]
    value_losses: List[float]

    @property
    def mean_time(self) -> float:
        return float(np.mean(self.step_times_s))

    @property
    def loss_drop(self) -> float:
        return self.losses[0] - self.losses[-1]

    @property
    def final_loss(self) -> float:
        return self.losses[-1]


# -----------------------------------------------------------------------------
# Synthetic data generation
# -----------------------------------------------------------------------------

def generate_synthetic_positions(n: int) -> List[GameExample]:
    """
    Generate n random chess positions with realistic-looking policy
    distributions and value targets drawn from [-1, 1].

    Positions are produced by playing random moves from the starting
    position (resetting every 60 plies to avoid trivial terminal states).
    """
    rng = np.random.default_rng(seed=42)
    board = chess.Board()
    examples: List[GameExample] = []

    for i in range(n):
        legal_moves = list(board.legal_moves)

        # Policy: sparse Dirichlet-like distribution over legal moves
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

        # Advance board; reset if game over or every 60 plies
        if legal_moves and i % 60 != 59 and not board.is_game_over():
            board.push(rng.choice(legal_moves))  # type: ignore[arg-type]
        else:
            board = chess.Board()

    return examples


# -----------------------------------------------------------------------------
# Model + training setup
# -----------------------------------------------------------------------------

def build_small_model(device: torch.device) -> HybridChessNet:
    """Small model (4 blocks × 64 filters) for fast benchmarking."""
    cfg = HybridModelConfig(
        cnn_input_channels=22,
        cnn_filters=64,
        cnn_residual_blocks=4,
        use_rnn=False,
        num_actions=4672,
    )
    return HybridChessNet(cfg).to(device)


def build_train_config(batch_size: int, n_steps: int) -> RLTrainingConfig:
    """Minimal RLTrainingConfig for a single training step at a time."""
    cfg = RLTrainingConfig(
        batch_size=batch_size,
        training_steps_per_iteration=n_steps,
        use_prioritized_replay=False,   # toggled per condition below
        per_alpha=0.6,
        per_beta=0.4,
        per_beta_end=1.0,
        per_epsilon=1e-6,
        log_frequency=99999,            # suppress per-step tensorboard noise
        device="cpu",
        use_amp=False,
    )
    return cfg


# -----------------------------------------------------------------------------
# Single-step timing helper
# -----------------------------------------------------------------------------

def run_single_step(
    model: HybridChessNet,
    buffer,
    config: RLTrainingConfig,
    board_encoder: BoardEncoder,
    move_encoder: MoveEncoder,
    device: torch.device,
    writer: SummaryWriter,
    per_beta: float | None,
    optimizer: optim.Optimizer,
    value_criterion: nn.Module,
    total_steps: int,
) -> Tuple[float, float, float, float]:
    """
    Run exactly one gradient step.  Returns (elapsed_s, loss, policy_loss, value_loss).
    """
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
        per_beta=per_beta,
    )
    elapsed = time.perf_counter() - t0
    return elapsed, metrics["loss"], metrics["policy_loss"], metrics["value_loss"]


# -----------------------------------------------------------------------------
# Benchmark runner
# -----------------------------------------------------------------------------

def run_condition(
    condition: str,
    examples: List[GameExample],
    n_steps: int,
    batch_size: int,
    device: torch.device,
    output_dir: str,
) -> BenchmarkResult:
    """
    Run `n_steps` training steps for one condition (uniform or per).
    A fresh model and optimizer are created so conditions are independent.
    """
    print(f"\n{'-'*60}")
    print(f"  Condition: {condition.upper()}")
    print(f"  Buffer   : {len(examples)} positions")
    print(f"  Steps    : {n_steps}  |  Batch: {batch_size}")
    print(f"{'-'*60}")

    use_per = condition == "per"

    # --- fresh model + optimizer ---
    model = build_small_model(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    value_criterion = nn.MSELoss()

    # --- replay buffer ---
    if use_per:
        buffer: ReplayBuffer = PrioritizedReplayBuffer(
            max_size=len(examples) + 1, alpha=0.6
        )
        for ex in examples:
            buffer.add_game_example(ex, priority=1.0)  # type: ignore[call-arg]
    else:
        buffer = ReplayBuffer(max_size=len(examples) + 1)
        buffer.add_game_examples(examples)

    # --- config: one step at a time so we can time each individually ---
    config = build_train_config(batch_size=batch_size, n_steps=1)

    board_encoder = BoardEncoder()
    move_encoder = MoveEncoder()

    log_path = os.path.join(output_dir, f"tb_{condition}")
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter(log_dir=log_path)

    step_times, losses, policy_losses, value_losses = [], [], [], []
    total_steps = 0

    for step in range(1, n_steps + 1):
        per_beta = 0.4 + (1.0 - 0.4) * ((step - 1) / max(1, n_steps - 1)) if use_per else None

        elapsed, loss, ploss, vloss = run_single_step(
            model=model,
            buffer=buffer,
            config=config,
            board_encoder=board_encoder,
            move_encoder=move_encoder,
            device=device,
            writer=writer,
            per_beta=per_beta,
            optimizer=optimizer,
            value_criterion=value_criterion,
            total_steps=total_steps,
        )
        total_steps += 1

        step_times.append(elapsed)
        losses.append(loss)
        policy_losses.append(ploss)
        value_losses.append(vloss)

        print(
            f"  step {step:>3}/{n_steps}  "
            f"loss={loss:.4f}  "
            f"(policy={ploss:.4f}  value={vloss:.4f})  "
            f"time={elapsed*1000:.1f}ms"
        )

        # After each PER step, update the priorities from value prediction
        # (simulated: use abs(value_loss) as a proxy priority for all samples)
        if use_per and isinstance(buffer, PrioritizedReplayBuffer):
            last_batch = buffer.sample(min(batch_size, len(buffer)), beta=per_beta or 0.4)
            new_pris = [abs(vloss) + 1e-6] * len(last_batch["indices"])
            buffer.update_priorities(last_batch["indices"], new_pris)

    writer.close()

    return BenchmarkResult(
        condition=condition,
        n_positions=len(examples),
        n_steps=n_steps,
        step_times_s=step_times,
        losses=losses,
        policy_losses=policy_losses,
        value_losses=value_losses,
    )


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

def save_csv(result: BenchmarkResult, output_dir: str) -> None:
    path = os.path.join(output_dir, f"{result.condition}_steps.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "time_s", "loss", "policy_loss", "value_loss"])
        for i, (t, l, pl, vl) in enumerate(
            zip(result.step_times_s, result.losses, result.policy_losses, result.value_losses),
            start=1,
        ):
            writer.writerow([i, f"{t:.6f}", f"{l:.6f}", f"{pl:.6f}", f"{vl:.6f}"])
    print(f"  Saved: {path}")


def print_summary(uniform: BenchmarkResult, per: BenchmarkResult) -> None:
    print("\n" + "=" * 60)
    print("  BENCHMARK SUMMARY")
    print("=" * 60)
    header = f"  {'Metric':<30}  {'Uniform':>10}  {'PER':>10}"
    print(header)
    print("  " + "-" * 56)

    rows = [
        ("Mean time per step (ms)", uniform.mean_time * 1000, per.mean_time * 1000, "{:.1f}"),
        ("First-step loss",         uniform.losses[0],       per.losses[0],        "{:.4f}"),
        ("Final loss",              uniform.final_loss,      per.final_loss,       "{:.4f}"),
        ("Loss drop (first→last)",  uniform.loss_drop,       per.loss_drop,        "{:.4f}"),
        ("Mean policy loss",        float(np.mean(uniform.policy_losses)),
                                    float(np.mean(per.policy_losses)),  "{:.4f}"),
        ("Mean value loss",         float(np.mean(uniform.value_losses)),
                                    float(np.mean(per.value_losses)),   "{:.4f}"),
    ]

    for label, u_val, p_val, fmt in rows:
        u_str = fmt.format(u_val)
        p_str = fmt.format(p_val)
        print(f"  {label:<30}  {u_str:>10}  {p_str:>10}")

    overhead_pct = (per.mean_time / uniform.mean_time - 1.0) * 100
    print(f"\n  PER overhead vs uniform: {overhead_pct:+.1f}%")
    print("=" * 60)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark PER vs uniform replay over N training steps"
    )
    parser.add_argument(
        "--n-positions", type=int, default=5000,
        help="Number of synthetic positions in the replay buffer (default: 5000)",
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Number of training steps per condition (default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Training batch size (default: 64)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device: 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="logs/per_benchmark",
        help="Directory for CSV and JSON output (default: logs/per_benchmark)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  PER vs UNIFORM REPLAY BENCHMARK")
    print("=" * 60)
    print(f"  Positions : {args.n_positions}")
    print(f"  Steps     : {args.steps} per condition")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device    : {device}")
    print(f"  Output    : {args.output_dir}")

    # Shared synthetic buffer (same positions for both conditions)
    print(f"\n  Generating {args.n_positions} synthetic positions...")
    t0 = time.perf_counter()
    examples = generate_synthetic_positions(args.n_positions)
    print(f"  Done in {time.perf_counter()-t0:.1f}s")

    # Run both conditions
    uniform_result = run_condition(
        "uniform", examples, args.steps, args.batch_size, device, args.output_dir
    )
    per_result = run_condition(
        "per", examples, args.steps, args.batch_size, device, args.output_dir
    )

    # Print summary
    print_summary(uniform_result, per_result)

    # Save CSVs
    print()
    save_csv(uniform_result, args.output_dir)
    save_csv(per_result, args.output_dir)

    # Save JSON summary
    overhead_pct = (per_result.mean_time / uniform_result.mean_time - 1.0) * 100
    summary = {
        "config": {
            "n_positions": args.n_positions,
            "n_steps": args.steps,
            "batch_size": args.batch_size,
            "device": str(device),
        },
        "uniform": {
            "mean_time_ms": round(uniform_result.mean_time * 1000, 2),
            "first_loss": round(uniform_result.losses[0], 6),
            "final_loss": round(uniform_result.final_loss, 6),
            "loss_drop": round(uniform_result.loss_drop, 6),
            "mean_policy_loss": round(float(np.mean(uniform_result.policy_losses)), 6),
            "mean_value_loss": round(float(np.mean(uniform_result.value_losses)), 6),
        },
        "per": {
            "mean_time_ms": round(per_result.mean_time * 1000, 2),
            "first_loss": round(per_result.losses[0], 6),
            "final_loss": round(per_result.final_loss, 6),
            "loss_drop": round(per_result.loss_drop, 6),
            "mean_policy_loss": round(float(np.mean(per_result.policy_losses)), 6),
            "mean_value_loss": round(float(np.mean(per_result.value_losses)), 6),
        },
        "per_overhead_pct": round(overhead_pct, 1),
    }

    json_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")
    print("\n✅ Benchmark complete.")


if __name__ == "__main__":
    main()
