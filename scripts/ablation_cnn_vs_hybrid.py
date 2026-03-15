"""
ABLATION STUDY: CNN-only vs Hybrid (CNN + LSTM) — Equal Parameter Budget

Trains both architectures in lockstep (same number of gradient steps at every
checkpoint) and periodically pauses to play head-to-head games.  This produces
three comparable time-series:

  1. Policy loss  (per step, both models)
  2. Value  loss  (per step, both models)
  3. Win rate     (Hybrid win-% vs CNN-only, sampled every --eval-interval steps)

For the Hybrid the gate activation mean is also recorded at every step so you
can see if the LSTM branch is being used.

The CNN-only model is automatically sized to match the Hybrid's parameter count
so the comparison is fair.

Output files (written to --output-dir):
  ablation_training.csv   — per-step policy/value/total loss + gate_mean
  ablation_winrate.csv    — win-rate snapshot at each eval checkpoint
  ablation_summary.json   — full summary (model configs, loss stats, head-to-head)

Usage:
    python scripts/ablation_cnn_vs_hybrid.py
    python scripts/ablation_cnn_vs_hybrid.py --steps 500 --eval-interval 100
    python scripts/ablation_cnn_vs_hybrid.py --steps 200 --eval-interval 50 --eval-games 4
    python scripts/ablation_cnn_vs_hybrid.py --eval-interval 0   # skip mid-training eval
    python scripts/ablation_cnn_vs_hybrid.py --games 0           # skip final head-to-head
"""

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_engine.models.hybrid.config import HybridModelConfig
from chess_engine.models.hybrid.hybrid_net import HybridChessNet
from chess_engine.search.mcts.search import MCTS
from chess_engine.models.hybrid.fusion import FeatureFusion
from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_hybrid(blocks: int, filters: int = 256) -> HybridChessNet:
    """Hybrid CNN + LSTM model with gated fusion and gate logging enabled."""
    cfg = HybridModelConfig(
        cnn_input_channels=22,
        cnn_filters=filters,
        cnn_residual_blocks=blocks,
        use_rnn=True,
        rnn_hidden_size=256,
        rnn_num_layers=2,
        fusion_type="gated",
    )
    model = HybridChessNet(cfg)
    if model.fusion is not None:
        setattr(model.fusion, "log_gates", True)
    return model


def build_cnn_only(blocks: int, filters: int = 256) -> HybridChessNet:
    """CNN-only model (LSTM branch disabled)."""
    cfg = HybridModelConfig(
        cnn_input_channels=22,
        cnn_filters=filters,
        cnn_residual_blocks=blocks,
        use_rnn=False,
    )
    return HybridChessNet(cfg)


def find_matching_cnn_blocks(hybrid: HybridChessNet, filters: int = 256) -> int:
    """CNN-only block count whose param total best matches the Hybrid's."""
    target = count_params(hybrid)
    best_blocks, best_diff = 1, float("inf")
    for b in range(1, 40):
        candidate = build_cnn_only(b, filters)
        diff = abs(count_params(candidate) - target)
        if diff < best_diff:
            best_diff = diff
            best_blocks = b
        if count_params(candidate) > target * 1.1:
            break
    return best_blocks


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic training data
# ─────────────────────────────────────────────────────────────────────────────


def make_batch(
    batch_size: int,
    num_actions: int,
    device: torch.device,
    use_rnn: bool = False,
    seq_len: int = 20,
) -> Tuple:
    """Random board tensors + sparse policy targets + uniform values."""
    boards = torch.randn(batch_size, 22, 8, 8, device=device)

    k = 30  # approximate legal move count
    indices = torch.stack(
        [torch.randperm(num_actions, device=device)[:k] for _ in range(batch_size)]
    )
    weights = torch.rand(batch_size, k, device=device)
    weights = weights / weights.sum(dim=1, keepdim=True)
    policies = torch.zeros(batch_size, num_actions, device=device)
    policies.scatter_(1, indices, weights)

    values = torch.empty(batch_size, 1, device=device).uniform_(-1.0, 1.0)

    if use_rnn:
        move_history = torch.randint(
            0, num_actions, (batch_size, seq_len), device=device
        )
        history_lengths = torch.full((batch_size,), seq_len, device=device)
    else:
        move_history = None
        history_lengths = None

    return boards, policies, values, move_history, history_lengths


# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class StepRecord:
    """One gradient step for one model."""

    step: int
    model: str  # "hybrid" or "cnn_only"
    policy_loss: float
    value_loss: float
    total_loss: float
    gate_mean: Optional[float]  # None for cnn_only
    step_time_ms: float


def train_one_step(
    model: HybridChessNet,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    num_actions: int,
    device: torch.device,
    use_rnn: bool,
    value_criterion: nn.Module,
) -> Tuple[float, float, float, Optional[float], float]:
    """
    One gradient step on synthetic data.

    Returns:
        (policy_loss, value_loss, total_loss, gate_mean_or_None, step_time_ms)
    """
    model.train()
    t0 = time.perf_counter()

    boards, policies, values, move_history, history_lengths = make_batch(
        batch_size, num_actions, device, use_rnn
    )

    optimizer.zero_grad()
    policy_logits, value_pred, _ = model(boards, move_history, history_lengths)

    # Soft cross-entropy — same formulation as train_step.py
    policies_safe = policies + 1e-8
    policies_safe = policies_safe / policies_safe.sum(dim=1, keepdim=True)
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = torch.mean(torch.sum(-policies_safe * log_probs, dim=1))

    value_loss = value_criterion(value_pred, values)
    loss = policy_loss + value_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    step_time_ms = (time.perf_counter() - t0) * 1000.0

    gate_mean: Optional[float] = None
    if use_rnn and isinstance(model.fusion, FeatureFusion):
        gate_mean = getattr(model.fusion, "last_gate_mean", None)

    return policy_loss.item(), value_loss.item(), loss.item(), gate_mean, step_time_ms


# ─────────────────────────────────────────────────────────────────────────────
# Head-to-head game (single game)
# ─────────────────────────────────────────────────────────────────────────────


def play_head_to_head_game(
    hybrid: HybridChessNet,
    cnn_only: HybridChessNet,
    board_encoder: BoardEncoder,
    move_encoder: MoveEncoder,
    device: torch.device,
    hybrid_plays_white: bool,
    num_simulations: int = 50,
    c_puct: float = 2.0,
    max_moves: int = 200,
) -> str:
    """
    Play one game between Hybrid and CNN-only with separate MCTS instances.

    Each model gets its own MCTS so the Hybrid uses use_rnn=True and the
    CNN-only model uses use_rnn=False independently.

    Returns: "hybrid_win", "cnn_only_win", or "draw"
    """
    board = chess.Board()

    hybrid_mcts = MCTS(
        model=hybrid,
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        num_simulations=num_simulations,
        c_puct=c_puct,
        use_rnn=True,
        temperature=0.1,  # Near-greedy for evaluation
    )
    cnn_mcts = MCTS(
        model=cnn_only,
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        num_simulations=num_simulations,
        c_puct=c_puct,
        use_rnn=False,
        temperature=0.1,
    )

    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        if board.turn == chess.WHITE:
            mcts = hybrid_mcts if hybrid_plays_white else cnn_mcts
        else:
            mcts = cnn_mcts if hybrid_plays_white else hybrid_mcts

        move, _ = mcts.search(board)
        if move is None:
            break
        board.push(move)
        move_count += 1

    if board.is_checkmate():
        white_won = not board.turn
        hybrid_won = (white_won and hybrid_plays_white) or (
            not white_won and not hybrid_plays_white
        )
        return "hybrid_win" if hybrid_won else "cnn_only_win"
    return "draw"


# ─────────────────────────────────────────────────────────────────────────────
# Multi-game evaluation block
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class WinRateRecord:
    """Win-rate snapshot at a given training step."""

    at_step: int
    hybrid_wins: int
    cnn_only_wins: int
    draws: int
    games_played: int
    hybrid_win_pct: float


def run_eval_block(
    hybrid: HybridChessNet,
    cnn_only: HybridChessNet,
    device: torch.device,
    board_encoder: BoardEncoder,
    move_encoder: MoveEncoder,
    at_step: int,
    num_games: int,
    num_simulations: int,
) -> WinRateRecord:
    """Play num_games games and return a WinRateRecord."""
    hybrid.eval()
    cnn_only.eval()

    hw = co = dr = 0
    for g in range(num_games):
        outcome = play_head_to_head_game(
            hybrid,
            cnn_only,
            board_encoder,
            move_encoder,
            device,
            hybrid_plays_white=(g % 2 == 0),
            num_simulations=num_simulations,
        )
        if outcome == "hybrid_win":
            hw += 1
        elif outcome == "cnn_only_win":
            co += 1
        else:
            dr += 1

    total = num_games
    win_pct = round(hw / total * 100, 1)
    return WinRateRecord(
        at_step=at_step,
        hybrid_wins=hw,
        cnn_only_wins=co,
        draws=dr,
        games_played=total,
        hybrid_win_pct=win_pct,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Final head-to-head (larger, dedicated run)
# ─────────────────────────────────────────────────────────────────────────────


def run_head_to_head(
    hybrid: HybridChessNet,
    cnn_only: HybridChessNet,
    device: torch.device,
    board_encoder: BoardEncoder,
    move_encoder: MoveEncoder,
    num_games: int = 10,
    num_simulations: int = 50,
) -> Dict:
    """Full final head-to-head match with per-game log."""
    hybrid.eval()
    cnn_only.eval()

    results: Dict = {
        "hybrid_wins": 0,
        "cnn_only_wins": 0,
        "draws": 0,
        "games": [],
    }

    print(f"\n{'─' * 70}")
    print(
        f"⚔  Final head-to-head: Hybrid vs CNN-only  "
        f"({num_games} games, {num_simulations} sims/move)"
    )
    print(f"{'─' * 70}")

    for g in range(num_games):
        hybrid_plays_white = g % 2 == 0
        color_str = "White" if hybrid_plays_white else "Black"

        t0 = time.perf_counter()
        outcome = play_head_to_head_game(
            hybrid,
            cnn_only,
            board_encoder,
            move_encoder,
            device,
            hybrid_plays_white,
            num_simulations,
        )
        elapsed = time.perf_counter() - t0

        key = outcome + "s"  # e.g. "hybrid_wins"
        results[key] = results.get(key, 0) + 1
        results["games"].append(
            {
                "game": g + 1,
                "hybrid_color": color_str,
                "outcome": outcome,
                "elapsed_s": round(elapsed, 1),
            }
        )

        icon = {"hybrid_win": "H", "cnn_only_win": "C", "draw": "="}[outcome]
        print(
            f"  Game {g+1:2d}/{num_games}  Hybrid={color_str:5s}  "
            f"result={outcome:12s} [{icon}]  ({elapsed:.1f}s)"
        )

    hw = results["hybrid_wins"]
    co = results["cnn_only_wins"]
    dr = results["draws"]
    total = num_games

    print(f"\n  Hybrid   : {hw:2d} wins  ({hw/total*100:.0f}%)")
    print(f"  CNN-only : {co:2d} wins  ({co/total*100:.0f}%)")
    print(f"  Draws    : {dr:2d}        ({dr/total*100:.0f}%)")

    results["summary"] = {
        "hybrid_win_rate": round(hw / total * 100, 1),
        "cnn_only_win_rate": round(co / total * 100, 1),
        "draw_rate": round(dr / total * 100, 1),
        "total_games": total,
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────


def window_avg(values: list, n: int = 10) -> Optional[float]:
    if not values:
        return None
    w = values[:n] if len(values) >= n else values
    return sum(w) / len(w)


def summarize_field(records: List[StepRecord], model_label: str, field: str) -> dict:
    vals = [
        getattr(r, field)
        for r in records
        if r.model == model_label and getattr(r, field) is not None
    ]
    if not vals:
        return {}
    return {
        "first_10_avg": window_avg(vals[:10]),
        "last_10_avg": window_avg(vals[-10:]),
        "min": min(vals),
        "max": max(vals),
    }


def gate_verdict(gate_last_10: Optional[float]) -> str:
    if gate_last_10 is None:
        return "N/A"
    if gate_last_10 > 0.8:
        return "⚠  Gate near 1.0 → LSTM barely contributing (consider disabling RNN)"
    if gate_last_10 < 0.2:
        return "⚠  Gate near 0.0 → CNN barely contributing (unusual)"
    return "✓  Gate in (0.2, 0.8) → both branches actively used"


def win_rate_verdict(hybrid_win_pct: Optional[float]) -> str:
    if hybrid_win_pct is None:
        return "N/A — no head-to-head games played"
    if hybrid_win_pct > 55:
        return "✓  Hybrid outperforms CNN-only → LSTM branch is contributing"
    if hybrid_win_pct < 45:
        return "⚠  CNN-only outperforms Hybrid → LSTM may be hurting performance"
    return "~  Results inconclusive (within noise for this game count)"


# ─────────────────────────────────────────────────────────────────────────────
# Gate value analysis
# ─────────────────────────────────────────────────────────────────────────────

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _make_sparkline(values: list, width: int = 40) -> str:
    """
    Downsample values to `width` points and render as a Unicode block sparkline.
    Useful for visualising the gate trajectory at a glance in the terminal.
    """
    if not values:
        return ""
    step = max(1, len(values) // width)
    sampled = [values[i] for i in range(0, len(values), step)][:width]
    lo, hi = min(sampled), max(sampled)
    if hi == lo:
        return _SPARK_CHARS[3] * len(sampled)
    return "".join(_SPARK_CHARS[min(7, int((v - lo) / (hi - lo) * 8))] for v in sampled)


def analyze_gate_values(step_records: List[StepRecord]) -> Dict:
    """
    Compute gate activation statistics from training records to determine
    whether the LSTM branch is contributing, dominating, or being ignored.

    Returns a dict with:
        available       — False if no gate data was collected
        n_steps         — number of gate samples
        overall_mean    — mean gate value across all steps (0–1)
        overall_std     — standard deviation
        early_mean      — mean of first third of training
        late_mean       — mean of last third of training
        trend           — late_mean − early_mean (+ → CNN, − → LSTM over time)
        pct_collapsed_high — % of steps with gate > 0.85 (near-CNN-only)
        pct_collapsed_low  — % of steps with gate < 0.15 (near-RNN-only)
        pct_balanced       — % of steps with gate in [0.2, 0.8]
        sparkline       — ASCII trend visualisation (low = RNN, high = CNN)
        verdict         — one of HEALTHY_BALANCED / COLLAPSED_HIGH /
                          COLLAPSED_LOW / TRENDING_CNN / TRENDING_RNN / MIXED
        interpretation  — human-readable explanation
    """
    gate_vals = [
        r.gate_mean
        for r in step_records
        if r.model == "hybrid" and r.gate_mean is not None
    ]

    if not gate_vals:
        return {"available": False}

    n = len(gate_vals)
    mean_val = sum(gate_vals) / n
    variance = sum((v - mean_val) ** 2 for v in gate_vals) / n
    std_val = variance**0.5

    # Trend: compare first-third vs last-third averages
    third = max(1, n // 3)
    early_mean = sum(gate_vals[:third]) / third
    late_mean = sum(gate_vals[-third:]) / third
    trend = late_mean - early_mean  # + → gate rising (more CNN), − → more RNN

    # Collapse / balance fractions
    pct_high = sum(1 for v in gate_vals if v > 0.85) / n
    pct_low = sum(1 for v in gate_vals if v < 0.15) / n
    pct_balanced = sum(1 for v in gate_vals if 0.2 <= v <= 0.8) / n

    # Classify behaviour
    if pct_high > 0.5:
        verdict = "COLLAPSED_HIGH"
        interpretation = (
            f"Gate > 0.85 for {pct_high*100:.0f}% of steps — "
            "LSTM output is being suppressed; CNN dominates the fusion. "
            "Consider removing the RNN branch or reducing its hidden size."
        )
    elif pct_low > 0.5:
        verdict = "COLLAPSED_LOW"
        interpretation = (
            f"Gate < 0.15 for {pct_low*100:.0f}% of steps — "
            "CNN features are being suppressed; RNN dominates. "
            "Check that the CNN backbone is training correctly."
        )
    elif pct_balanced > 0.6 and abs(trend) < 0.1:
        verdict = "HEALTHY_BALANCED"
        interpretation = (
            f"Gate stable in [0.2, 0.8] for {pct_balanced*100:.0f}% of steps — "
            "both CNN and LSTM branches are actively contributing."
        )
    elif trend > 0.1:
        verdict = "TRENDING_CNN"
        interpretation = (
            f"Gate rising by {trend:.3f} (early={early_mean:.3f} → "
            f"late={late_mean:.3f}) — model is learning to rely more on CNN "
            "as training progresses."
        )
    elif trend < -0.1:
        verdict = "TRENDING_RNN"
        interpretation = (
            f"Gate falling by {abs(trend):.3f} (early={early_mean:.3f} → "
            f"late={late_mean:.3f}) — model is learning to rely more on LSTM "
            "as training progresses."
        )
    else:
        verdict = "MIXED"
        interpretation = (
            "Gate behaviour is mixed or noisy — "
            "consider training for more steps to establish a clear trend."
        )

    return {
        "available": True,
        "n_steps": n,
        "overall_mean": round(mean_val, 4),
        "overall_std": round(std_val, 4),
        "early_mean": round(early_mean, 4),
        "late_mean": round(late_mean, 4),
        "trend": round(trend, 4),
        "pct_collapsed_high": round(pct_high * 100, 1),
        "pct_collapsed_low": round(pct_low * 100, 1),
        "pct_balanced": round(pct_balanced * 100, 1),
        "sparkline": _make_sparkline(gate_vals),
        "verdict": verdict,
        "interpretation": interpretation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CNN-only vs Hybrid ablation study (equal parameter budget)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Gradient steps (applied to BOTH models in lockstep, default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)",
    )
    parser.add_argument(
        "--hybrid-blocks",
        type=int,
        default=10,
        help="Residual blocks in the Hybrid CNN (default: 10)",
    )
    parser.add_argument(
        "--filters",
        type=int,
        default=256,
        help="CNN filters in both models (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50,
        help="Play --eval-games games every N training steps (default: 50; 0 = off)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=4,
        help="Games per mid-training evaluation checkpoint (default: 4)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Final head-to-head games after training (default: 10; 0 = off)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=50,
        help="MCTS simulations per move during evaluation (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/ablation",
        help="Directory for output files (default: logs/ablation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_actions: int = MoveEncoder().num_moves  # 4672
    value_criterion = nn.MSELoss()
    board_encoder = BoardEncoder()
    move_encoder = MoveEncoder()

    # ── Build models ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("ABLATION: CNN-only vs Hybrid (CNN + LSTM)")
    print("=" * 70)
    print(f"\nDevice: {device}")

    hybrid = build_hybrid(args.hybrid_blocks, args.filters).to(device)
    cnn_blocks_matched = find_matching_cnn_blocks(hybrid, args.filters)
    cnn_only = build_cnn_only(cnn_blocks_matched, args.filters).to(device)

    hybrid_params = count_params(hybrid)
    cnn_params = count_params(cnn_only)

    print(f"\nModel configurations (equal parameter budget):")
    print(
        f"  Hybrid   — {args.hybrid_blocks:2d} CNN blocks + LSTM  "
        f"→ {hybrid_params:>12,} params"
    )
    print(
        f"  CNN-only — {cnn_blocks_matched:2d} CNN blocks          "
        f"→ {cnn_params:>12,} params  "
        f"(Δ = {abs(hybrid_params - cnn_params):,})"
    )
    print(
        f"\nTraining  : {args.steps} lockstep steps × batch_size={args.batch_size} "
        f" lr={args.lr}"
    )
    if args.eval_interval > 0:
        n_checkpoints = args.steps // args.eval_interval
        print(
            f"Evaluation: {args.eval_games} games every {args.eval_interval} steps "
            f"({n_checkpoints} checkpoints)"
        )
    if args.games > 0:
        print(f"Final H2H : {args.games} games, {args.simulations} sims/move")

    # ── Optimizers ────────────────────────────────────────────────────────────
    opt_hybrid = torch.optim.Adam(hybrid.parameters(), lr=args.lr)
    opt_cnn = torch.optim.Adam(cnn_only.parameters(), lr=args.lr)

    # ── Lockstep training loop ────────────────────────────────────────────────
    step_records: List[StepRecord] = []
    win_rate_records: List[WinRateRecord] = []

    print(f"\n{'─' * 70}")
    print("▶  Training both models in lockstep ...")
    print(f"{'─' * 70}")

    for step in range(1, args.steps + 1):
        # Train Hybrid
        pl_h, vl_h, tl_h, gm_h, ms_h = train_one_step(
            hybrid,
            opt_hybrid,
            args.batch_size,
            num_actions,
            device,
            True,
            value_criterion,
        )
        step_records.append(
            StepRecord(
                step=step,
                model="hybrid",
                policy_loss=pl_h,
                value_loss=vl_h,
                total_loss=tl_h,
                gate_mean=gm_h,
                step_time_ms=ms_h,
            )
        )

        # Train CNN-only
        pl_c, vl_c, tl_c, _, ms_c = train_one_step(
            cnn_only,
            opt_cnn,
            args.batch_size,
            num_actions,
            device,
            False,
            value_criterion,
        )
        step_records.append(
            StepRecord(
                step=step,
                model="cnn_only",
                policy_loss=pl_c,
                value_loss=vl_c,
                total_loss=tl_c,
                gate_mean=None,
                step_time_ms=ms_c,
            )
        )

        # Progress print
        if step == 1 or step % 50 == 0:
            gate_str = f"  gate={gm_h:.4f}" if gm_h is not None else ""
            print(
                f"  step {step:4d}/{args.steps}  "
                f"Hybrid  policy={pl_h:.4f} value={vl_h:.4f}{gate_str}\n"
                f"            "
                f"CNN-only policy={pl_c:.4f} value={vl_c:.4f}"
            )

        # Mid-training evaluation checkpoint
        if args.eval_interval > 0 and step % args.eval_interval == 0:
            print(
                f"\n  [step {step}] Running {args.eval_games}-game eval checkpoint ..."
            )
            wr = run_eval_block(
                hybrid,
                cnn_only,
                device,
                board_encoder,
                move_encoder,
                at_step=step,
                num_games=args.eval_games,
                num_simulations=args.simulations,
            )
            win_rate_records.append(wr)
            print(
                f"  [step {step}] Hybrid {wr.hybrid_wins}W / "
                f"CNN-only {wr.cnn_only_wins}W / {wr.draws}D  "
                f"→ Hybrid win-rate: {wr.hybrid_win_pct:.0f}%\n"
            )
            # Resume training
            hybrid.train()
            cnn_only.train()

    # ── Final head-to-head ────────────────────────────────────────────────────
    final_h2h: Optional[Dict] = None
    if args.games > 0:
        final_h2h = run_head_to_head(
            hybrid,
            cnn_only,
            device,
            board_encoder,
            move_encoder,
            num_games=args.games,
            num_simulations=args.simulations,
        )

    # ── Gate analysis ─────────────────────────────────────────────────────────
    gate_analysis = analyze_gate_values(step_records)

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    train_csv_path = os.path.join(args.output_dir, "ablation_training.csv")
    winrate_csv_path = os.path.join(args.output_dir, "ablation_winrate.csv")
    json_path = os.path.join(args.output_dir, "ablation_summary.json")

    # Training CSV — per-step policy/value/total loss + gate_mean
    with open(train_csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(step_records[0]).keys()))
        writer.writeheader()
        for r in step_records:
            writer.writerow(asdict(r))

    # Win-rate CSV — checkpoint snapshots
    if win_rate_records:
        with open(winrate_csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=list(asdict(win_rate_records[0]).keys())
            )
            writer.writeheader()
            for r in win_rate_records:
                writer.writerow(asdict(r))

    # JSON summary
    summary: Dict = {
        "config": vars(args),
        "models": {
            "hybrid": {"params": hybrid_params, "blocks": args.hybrid_blocks},
            "cnn_only": {"params": cnn_params, "blocks": cnn_blocks_matched},
        },
        "training": {
            lbl: {
                field: summarize_field(step_records, lbl, field)
                for field in (
                    "policy_loss",
                    "value_loss",
                    "total_loss",
                    "gate_mean",
                    "step_time_ms",
                )
            }
            for lbl in ("hybrid", "cnn_only")
        },
        "win_rate_over_training": [asdict(r) for r in win_rate_records],
        "final_head_to_head": final_h2h,
        "gate_analysis": gate_analysis,
    }

    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 70}")

    for label in ("hybrid", "cnn_only"):
        r = summary["training"][label]
        m = summary["models"][label]
        print(f"\n{label.upper()}  ({m['params']:,} params, {m['blocks']} blocks):")
        for metric_name, key in [
            ("Policy loss", "policy_loss"),
            ("Value  loss", "value_loss"),
            ("Total  loss", "total_loss"),
        ]:
            d = r[key]
            print(
                f"  {metric_name} : "
                f"first-10 avg = {d.get('first_10_avg', 0):.4f}  "
                f"last-10 avg = {d.get('last_10_avg', 0):.4f}"
            )
        st = r["step_time_ms"]
        print(f"  Step time   : last-10 avg = {st.get('last_10_avg', 0):.1f} ms/step")

        if label == "hybrid":
            gm = r.get("gate_mean", {})
            if gm:
                last_gate = gm.get("last_10_avg")
                print(
                    f"  Gate mean   : "
                    f"first-10 avg = {gm.get('first_10_avg', 0):.4f}  "
                    f"last-10 avg = {last_gate:.4f}"
                )
                print(f"  Gate verdict: {gate_verdict(last_gate)}")

    # Loss delta
    hy_last = summary["training"]["hybrid"]["total_loss"].get("last_10_avg", 0)
    cn_last = summary["training"]["cnn_only"]["total_loss"].get("last_10_avg", 0)
    delta = hy_last - cn_last
    sign = "lower" if delta < 0 else "higher"
    print(f"\n  ▶ Hybrid final loss is {abs(delta):.4f} {sign} than CNN-only")

    hy_ms = summary["training"]["hybrid"]["step_time_ms"].get("last_10_avg", 1)
    cn_ms = summary["training"]["cnn_only"]["step_time_ms"].get("last_10_avg", 1)
    overhead = (hy_ms - cn_ms) / cn_ms * 100
    print(f"  ▶ Hybrid per-step overhead: {overhead:+.1f}% vs CNN-only")

    # ── Gate analysis ─────────────────────────────────────────────────────────
    if gate_analysis.get("available"):
        ga = gate_analysis
        print(f"\n{'=' * 70}")
        print("GATE ANALYSIS  (Hybrid fusion — CNN vs LSTM contribution)")
        print(f"  (gate ≈ 1.0 → CNN dominates · gate ≈ 0.0 → LSTM dominates)")
        print(f"{'=' * 70}")
        print(
            f"  Samples     : {ga['n_steps']} steps\n"
            f"  Mean ± std  : {ga['overall_mean']:.4f} ± {ga['overall_std']:.4f}\n"
            f"  Early mean  : {ga['early_mean']:.4f}  →  "
            f"Late mean: {ga['late_mean']:.4f}  "
            f"(trend: {ga['trend']:+.4f})"
        )
        print(
            f"  High  >0.85 : {ga['pct_collapsed_high']:5.1f}%  "
            f"(CNN dominates)\n"
            f"  Low   <0.15 : {ga['pct_collapsed_low']:5.1f}%  "
            f"(LSTM dominates)\n"
            f"  Balanced    : {ga['pct_balanced']:5.1f}%  "
            f"[0.2 – 0.8]"
        )
        print(f"  Sparkline   : {ga['sparkline']}")
        print(f"  Verdict     : {ga['verdict']}")
        print(f"  {ga['interpretation']}")

    # Win rate over training
    if win_rate_records:
        print(f"\n{'=' * 70}")
        print("WIN RATE OVER TRAINING  (Hybrid win-%)")
        print(f"{'=' * 70}")
        print(
            f"  {'Step':>6}  {'H wins':>7}  {'C wins':>7}  {'Draws':>6}  {'H win-%':>8}"
        )
        for wr in win_rate_records:
            print(
                f"  {wr.at_step:6d}  {wr.hybrid_wins:7d}  "
                f"{wr.cnn_only_wins:7d}  {wr.draws:6d}  "
                f"{wr.hybrid_win_pct:7.0f}%"
            )

    # Final head-to-head
    if final_h2h:
        s = final_h2h["summary"]
        print(f"\n{'=' * 70}")
        print("FINAL HEAD-TO-HEAD")
        print(f"{'=' * 70}")
        print(f"  Hybrid   win rate : {s['hybrid_win_rate']:.0f}%")
        print(f"  CNN-only win rate : {s['cnn_only_win_rate']:.0f}%")
        print(f"  Draw rate         : {s['draw_rate']:.0f}%")
        print(f"  Verdict           : {win_rate_verdict(s['hybrid_win_rate'])}")

    print(f"\nResults saved to:")
    print(f"  {train_csv_path}")
    if win_rate_records:
        print(f"  {winrate_csv_path}")
    print(f"  {json_path}")
    print()


if __name__ == "__main__":
    main()
