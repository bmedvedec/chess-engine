"""
c_puct Sweep — Find the best exploration constant for the Hybrid model

For each candidate c_puct value in C_PUCT_SWEEP (defined in chess_engine/config.py)
this script pits a Challenger MCTS (sweep c_puct) against a Baseline MCTS
(fixed c_puct, default 1.5) using the SAME model weights.  Because the model
weights are identical, any consistent win-rate difference is purely a function
of how well each c_puct value directs the MCTS search.

NOTE: Results are only meaningful with a trained model checkpoint (--checkpoint).
A randomly-initialised model produces near-random play and the win rates will
all hover around 50 % regardless of c_puct.

Output files (--output-dir, default logs/cpuct_sweep/):
    cpuct_sweep.csv   — per-game log (c_puct, game, challenger_color, outcome)
    cpuct_sweep.json  — per-value win rates + overall best value

Usage:
    # Sweep all values defined in config.py against baseline=1.5
    python scripts/tune_cpuct.py --checkpoint path/to/model.pt

    # Quick smoke test (no checkpoint, random model — not for real tuning)
    python scripts/tune_cpuct.py --games 4 --simulations 20

    # Custom sweep values
    python scripts/tune_cpuct.py --sweep 1.5 2.0 2.5 --games 20 --checkpoint path/to/model.pt

    # Different baseline
    python scripts/tune_cpuct.py --baseline 2.0 --checkpoint path/to/model.pt
"""

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import chess
import torch

from chess_engine.config import C_PUCT, C_PUCT_SWEEP
from chess_engine.models.cnn.chess_net import ChessNet
from chess_engine.models.hybrid.config import HybridModelConfig
from chess_engine.models.hybrid.hybrid_net import HybridChessNet
import torch.nn as nn
from chess_engine.search.mcts.search import MCTS
from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────


def build_model(checkpoint: Optional[str], device: torch.device) -> nn.Module:
    """
    Load a model from checkpoint, inferring architecture (CNN vs Hybrid) from
    the saved model_config.  Falls back to CNN-only when use_rnn is absent or False.
    For real c_puct tuning always provide --checkpoint; a random model produces
    near-random play and the sweep results are meaningless.
    """
    if checkpoint:
        raw = torch.load(checkpoint, map_location=device)

        # Resolve state dict
        if isinstance(raw, dict) and "model_state_dict" in raw:
            state = raw["model_state_dict"]
        elif isinstance(raw, dict) and "state_dict" in raw:
            state = raw["state_dict"]
        else:
            state = raw

        # Detect architecture from saved metadata
        model_cfg = raw.get("model_config", {}) if isinstance(raw, dict) else {}
        use_rnn = model_cfg.get("use_rnn", False)
        cnn_blocks = model_cfg.get("cnn_residual_blocks", 10)

        if use_rnn:
            cfg = HybridModelConfig(
                cnn_input_channels=22,
                cnn_filters=256,
                cnn_residual_blocks=cnn_blocks,
                use_rnn=True,
                rnn_hidden_size=256,
                rnn_num_layers=2,
                fusion_type="gated",
            )
            model = HybridChessNet(cfg).to(device)
        else:
            model = ChessNet(
                input_channels=22,
                num_filters=256,
                num_residual_blocks=cnn_blocks,
            ).to(device)

        # Remap keys if checkpoint was saved from the RL trainer's internal model
        # (uses "cnn." prefix) but ChessNet expects "backbone." prefix.
        if not use_rnn and any(k.startswith("cnn.") for k in state):
            state = {
                k.replace("cnn.", "backbone.", 1) if k.startswith("cnn.") else k: v
                for k, v in state.items()
            }

        model.load_state_dict(state)
        arch = "Hybrid (CNN+RNN)" if use_rnn else "CNN-only"
        print(f"  Loaded checkpoint: {checkpoint}  [{arch}, {cnn_blocks} blocks]")
    else:
        print("  ⚠  No checkpoint provided — using random weights.")
        print(
            "     Results will be ~50% for all c_puct values (not useful for tuning)."
        )
        model = ChessNet(input_channels=22, num_filters=256, num_residual_blocks=10).to(device)

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Single game
# ─────────────────────────────────────────────────────────────────────────────


def play_game(
    model: nn.Module,
    board_encoder: BoardEncoder,
    move_encoder: MoveEncoder,
    device: torch.device,
    challenger_c_puct: float,
    baseline_c_puct: float,
    challenger_plays_white: bool,
    num_simulations: int,
    max_moves: int,
    use_rnn: bool = False,
) -> str:
    """
    Play one game between Challenger (sweep c_puct) and Baseline (fixed c_puct).
    Both sides use the same model weights — only c_puct differs.

    Returns: "challenger_win", "baseline_win", or "draw"
    """
    board = chess.Board()

    challenger_mcts = MCTS(
        model=model,
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        num_simulations=num_simulations,
        c_puct=challenger_c_puct,
        use_rnn=use_rnn,
        temperature=0.1,  # Near-deterministic evaluation play
    )
    baseline_mcts = MCTS(
        model=model,
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        num_simulations=num_simulations,
        c_puct=baseline_c_puct,
        use_rnn=use_rnn,
        temperature=0.1,
    )

    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        if board.turn == chess.WHITE:
            mcts = challenger_mcts if challenger_plays_white else baseline_mcts
        else:
            mcts = baseline_mcts if challenger_plays_white else challenger_mcts

        move, _ = mcts.search(board)
        if move is None:
            break
        board.push(move)
        move_count += 1

    if board.is_checkmate():
        white_won = not board.turn  # the side that just moved delivered checkmate
        challenger_won = (white_won and challenger_plays_white) or (
            not white_won and not challenger_plays_white
        )
        return "challenger_win" if challenger_won else "baseline_win"
    return "draw"


# ─────────────────────────────────────────────────────────────────────────────
# Per-value evaluation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GameRecord:
    c_puct: float
    game: int
    challenger_color: str
    outcome: str
    elapsed_s: float


@dataclass
class ValueResult:
    c_puct: float
    games: int
    challenger_wins: int
    baseline_wins: int
    draws: int
    win_pct: float  # challenger wins / total
    score_pct: float  # (wins + 0.5*draws) / total  — chess scoring


def evaluate_c_puct(
    c_puct_value: float,
    baseline: float,
    model: nn.Module,
    board_encoder: BoardEncoder,
    move_encoder: MoveEncoder,
    device: torch.device,
    num_games: int,
    num_simulations: int,
    max_moves: int,
    use_rnn: bool = False,
) -> tuple[ValueResult, List[GameRecord]]:
    """Play num_games games and return the result + per-game log."""
    cw = bw = dr = 0
    records: List[GameRecord] = []

    for g in range(num_games):
        challenger_white = g % 2 == 0
        color_str = "White" if challenger_white else "Black"

        t0 = time.perf_counter()
        outcome = play_game(
            model,
            board_encoder,
            move_encoder,
            device,
            challenger_c_puct=c_puct_value,
            baseline_c_puct=baseline,
            challenger_plays_white=challenger_white,
            num_simulations=num_simulations,
            max_moves=max_moves,
            use_rnn=use_rnn,
        )
        elapsed = round(time.perf_counter() - t0, 1)

        if outcome == "challenger_win":
            cw += 1
        elif outcome == "baseline_win":
            bw += 1
        else:
            dr += 1

        icon = {"challenger_win": "C", "baseline_win": "B", "draw": "="}[outcome]
        print(
            f"    Game {g+1:2d}/{num_games}  Challenger={'White' if challenger_white else 'Black '}  "
            f"result={outcome:15s} [{icon}]  ({elapsed}s)"
        )
        records.append(
            GameRecord(
                c_puct=c_puct_value,
                game=g + 1,
                challenger_color=color_str,
                outcome=outcome,
                elapsed_s=elapsed,
            )
        )

    total = num_games
    win_pct = round(cw / total * 100, 1)
    score_pct = round((cw + 0.5 * dr) / total * 100, 1)

    result = ValueResult(
        c_puct=c_puct_value,
        games=total,
        challenger_wins=cw,
        baseline_wins=bw,
        draws=dr,
        win_pct=win_pct,
        score_pct=score_pct,
    )
    return result, records


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep c_puct values for the Hybrid model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trained model checkpoint (.pt). "
        "Required for meaningful results (default: None → random init)",
    )
    parser.add_argument(
        "--sweep",
        type=float,
        nargs="+",
        default=C_PUCT_SWEEP,
        help=f"c_puct values to evaluate (default: C_PUCT_SWEEP from config = {C_PUCT_SWEEP})",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=C_PUCT,
        help=f"Fixed baseline c_puct to compare against (default: C_PUCT = {C_PUCT})",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=20,
        help="Games per c_puct value (default: 20; use ≥20 for reliable results)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="MCTS simulations per move (default: 100)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Maximum moves per game before declaring draw (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/cpuct_sweep",
        help="Output directory for results (default: logs/cpuct_sweep)",
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
    board_encoder = BoardEncoder()
    move_encoder = MoveEncoder()

    print("=" * 70)
    print("c_puct SWEEP — Hybrid Model MCTS Tuning")
    print("=" * 70)
    print(f"\n  Device      : {device}")
    print(f"  Checkpoint  : {args.checkpoint or '(none — random init)'}")
    print(f"  Baseline    : c_puct = {args.baseline}")
    print(f"  Sweep       : {args.sweep}")
    print(f"  Games/value : {args.games}")
    print(f"  Simulations : {args.simulations} per move")

    # Skip baseline vs itself (win rate would be exactly 50 % by symmetry)
    sweep_values = [v for v in args.sweep if v != args.baseline]
    skipped = [v for v in args.sweep if v == args.baseline]
    if skipped:
        print(
            f"  Note        : {skipped} == baseline, will be recorded as 50% without playing"
        )
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model ...")
    model = build_model(args.checkpoint, device)
    print()

    # ── Run sweep ─────────────────────────────────────────────────────────────
    all_results: List[ValueResult] = []
    all_game_records: List[GameRecord] = []

    # Insert the baseline entry (no games needed)
    baseline_entry = ValueResult(
        c_puct=args.baseline,
        games=0,
        challenger_wins=0,
        baseline_wins=0,
        draws=0,
        win_pct=50.0,
        score_pct=50.0,
    )
    all_results.append(baseline_entry)

    for c_puct_val in sweep_values:
        print(f"{'─' * 70}")
        print(
            f"c_puct = {c_puct_val}  vs  baseline = {args.baseline}  "
            f"({args.games} games, {args.simulations} sims/move)"
        )
        print(f"{'─' * 70}")

        result, game_records = evaluate_c_puct(
            c_puct_value=c_puct_val,
            baseline=args.baseline,
            model=model,
            board_encoder=board_encoder,
            move_encoder=move_encoder,
            device=device,
            num_games=args.games,
            num_simulations=args.simulations,
            max_moves=args.max_moves,
            use_rnn=isinstance(model, HybridChessNet),
        )
        all_results.append(result)
        all_game_records.extend(game_records)

        print(
            f"\n  c_puct={c_puct_val}:  "
            f"Challenger {result.challenger_wins}W / "
            f"Baseline {result.baseline_wins}W / {result.draws}D  "
            f"→ win% = {result.win_pct:.1f}%  score% = {result.score_pct:.1f}%\n"
        )

    # ── Determine best value ───────────────────────────────────────────────────
    # Use score_pct (chess scoring: win=1, draw=0.5, loss=0) to rank
    best = max(all_results, key=lambda r: r.score_pct)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("RESULTS SUMMARY  (Challenger score vs Baseline)")
    print("=" * 70)
    col = "  {:<10}  {:>8}  {:>10}  {:>10}  {:>8}  {:>10}  {}"
    print(col.format("c_puct", "Games", "C wins", "B wins", "Draws", "Score %", ""))
    print("  " + "─" * 66)
    for r in sorted(all_results, key=lambda r: r.c_puct):
        tag = ""
        if r.c_puct == args.baseline:
            tag = " ← baseline"
        if r.c_puct == best.c_puct and r.c_puct != args.baseline:
            tag = " ← BEST"
        games_str = str(r.games) if r.games > 0 else "(ref)"
        print(
            col.format(
                r.c_puct,
                games_str,
                r.challenger_wins,
                r.baseline_wins,
                r.draws,
                f"{r.score_pct:.1f}%",
                tag,
            )
        )

    print(f"\n  Best c_puct: {best.c_puct}  (score = {best.score_pct:.1f}%)")
    if best.c_puct == args.baseline:
        print(
            f"  → Current default ({args.baseline}) is already optimal in this sweep."
        )
    else:
        print(
            f"  → Consider updating C_PUCT in chess_engine/config.py to {best.c_puct}."
        )

    note_games = args.games
    if note_games < 50:
        ci = round(50 / (note_games**0.5), 1)
        print(
            f"\n  ⚠  Statistical note: at {note_games} games/value the confidence interval is"
            f" ≈ ±{ci} percentage points.  Run with --games 50 or more for reliable results."
        )

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "cpuct_sweep.csv")
    json_path = os.path.join(args.output_dir, "cpuct_sweep.json")

    # Per-game CSV
    if all_game_records:
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=list(asdict(all_game_records[0]).keys())
            )
            writer.writeheader()
            for rec in all_game_records:
                writer.writerow(asdict(rec))

    # JSON summary
    summary = {
        "config": {
            "checkpoint": args.checkpoint,
            "baseline_cpuct": args.baseline,
            "sweep": args.sweep,
            "games_per_value": args.games,
            "simulations": args.simulations,
            "max_moves": args.max_moves,
            "seed": args.seed,
        },
        "results": [asdict(r) for r in sorted(all_results, key=lambda r: r.c_puct)],
        "best_c_puct": best.c_puct,
        "best_score_pct": best.score_pct,
    }
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nResults saved to:")
    if all_game_records:
        print(f"  {csv_path}")
    print(f"  {json_path}")
    print()


if __name__ == "__main__":
    main()
