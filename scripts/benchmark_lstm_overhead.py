"""
LSTM Overhead Benchmark: Hybrid (CNN + LSTM) vs CNN-only — GPU timing

Measures per-step wall time for both architectures across multiple LSTM hidden
sizes, using proper GPU synchronisation so CUDA kernel latency is counted.

Two timing modes are reported for each configuration:
  • inference  — forward pass only  (model.eval(), no grad)
  • train_step — forward + backward + optimizer step  (model.train())

Usage:
    # Default: test rnn_hidden_size 256 and 128 on whatever device is available
    python scripts/benchmark_lstm_overhead.py

    # Explicit hidden sizes
    python scripts/benchmark_lstm_overhead.py --rnn-hidden 64 128 256 512

    # Longer run for tighter confidence intervals
    python scripts/benchmark_lstm_overhead.py --iters 200 --warmup 20

    # Larger model
    python scripts/benchmark_lstm_overhead.py --hybrid-blocks 10 --filters 256

Output:
    Prints a comparison table to stdout.
    Saves results to --output (default: logs/benchmark_lstm_overhead.json).
"""

import argparse
import json
import os
import statistics
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_engine.models.hybrid.config import HybridModelConfig
from chess_engine.models.hybrid.hybrid_net import HybridChessNet


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_hybrid(blocks: int, filters: int, rnn_hidden: int) -> HybridChessNet:
    cfg = HybridModelConfig(
        cnn_input_channels=22,
        cnn_filters=filters,
        cnn_residual_blocks=blocks,
        use_rnn=True,
        rnn_hidden_size=rnn_hidden,
        rnn_num_layers=2,
        fusion_type="gated",
    )
    return HybridChessNet(cfg)


def build_cnn_only(blocks: int, filters: int) -> HybridChessNet:
    cfg = HybridModelConfig(
        cnn_input_channels=22,
        cnn_filters=filters,
        cnn_residual_blocks=blocks,
        use_rnn=False,
    )
    return HybridChessNet(cfg)


def find_matching_cnn_blocks(hybrid: HybridChessNet, filters: int) -> int:
    """Return CNN-only block count whose param total best matches hybrid."""
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
# Batch helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_batch(
    batch_size: int,
    num_actions: int,
    device: torch.device,
    use_rnn: bool,
    seq_len: int = 20,
) -> Tuple:
    boards = torch.randn(batch_size, 22, 8, 8, device=device)
    if use_rnn:
        move_history = torch.randint(
            0, num_actions, (batch_size, seq_len), device=device
        )
        history_lengths = torch.full((batch_size,), seq_len, device=device)
    else:
        move_history = None
        history_lengths = None
    return boards, move_history, history_lengths


# ─────────────────────────────────────────────────────────────────────────────
# GPU-safe timer
# ─────────────────────────────────────────────────────────────────────────────


def sync() -> None:
    """Block until all CUDA kernels have finished (no-op on CPU)."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_forward(
    model: HybridChessNet,
    boards: torch.Tensor,
    move_history: Optional[torch.Tensor],
    history_lengths: Optional[torch.Tensor],
) -> float:
    """Single inference forward pass. Returns wall time in ms."""
    sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(boards, move_history, history_lengths)
    sync()
    return (time.perf_counter() - t0) * 1000.0


def timed_train_step(
    model: HybridChessNet,
    optimizer: torch.optim.Optimizer,
    boards: torch.Tensor,
    move_history: Optional[torch.Tensor],
    history_lengths: Optional[torch.Tensor],
    num_actions: int,
) -> float:
    """Single train step (forward + backward + opt). Returns wall time in ms."""
    batch_size = boards.shape[0]
    # Minimal targets — same as ablation script
    policies = torch.zeros(batch_size, num_actions, device=boards.device)
    policies[:, 0] = 1.0
    values = torch.zeros(batch_size, 1, device=boards.device)

    sync()
    t0 = time.perf_counter()

    optimizer.zero_grad()
    policy_logits, value_pred, _ = model(boards, move_history, history_lengths)
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = torch.mean(torch.sum(-(policies + 1e-8) * log_probs, dim=1))
    value_loss = F.mse_loss(value_pred, values)
    (policy_loss + value_loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    sync()
    return (time.perf_counter() - t0) * 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# Single configuration benchmark
# ─────────────────────────────────────────────────────────────────────────────


def benchmark_model(
    model: HybridChessNet,
    device: torch.device,
    batch_size: int,
    num_actions: int,
    warmup: int,
    iters: int,
    use_rnn: bool,
    seq_len: int,
) -> Dict:
    """Return {'inference_ms': {...}, 'train_ms': {...}} with mean/std/min/max."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ── Inference ──────────────────────────────────────────────────────────
    model.eval()
    boards, mh, hl = make_batch(batch_size, num_actions, device, use_rnn, seq_len)

    # Warm-up
    for _ in range(warmup):
        timed_forward(model, boards, mh, hl)

    inf_times: List[float] = []
    for _ in range(iters):
        # Refresh batch each iter to avoid caching artefacts
        boards, mh, hl = make_batch(batch_size, num_actions, device, use_rnn, seq_len)
        inf_times.append(timed_forward(model, boards, mh, hl))

    # ── Training step ──────────────────────────────────────────────────────
    model.train()
    boards, mh, hl = make_batch(batch_size, num_actions, device, use_rnn, seq_len)

    # Warm-up
    for _ in range(warmup):
        boards, mh, hl = make_batch(batch_size, num_actions, device, use_rnn, seq_len)
        timed_train_step(model, optimizer, boards, mh, hl, num_actions)

    train_times: List[float] = []
    for _ in range(iters):
        boards, mh, hl = make_batch(batch_size, num_actions, device, use_rnn, seq_len)
        train_times.append(
            timed_train_step(model, optimizer, boards, mh, hl, num_actions)
        )

    def _stats(ts: List[float]) -> Dict:
        return {
            "mean_ms": round(statistics.mean(ts), 3),
            "std_ms": round(statistics.stdev(ts) if len(ts) > 1 else 0.0, 3),
            "min_ms": round(min(ts), 3),
            "max_ms": round(max(ts), 3),
            "p50_ms": round(statistics.median(ts), 3),
            "p95_ms": round(sorted(ts)[int(len(ts) * 0.95)], 3),
        }

    return {
        "inference": _stats(inf_times),
        "train_step": _stats(train_times),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark LSTM overhead: Hybrid vs CNN-only across hidden sizes"
    )
    parser.add_argument(
        "--rnn-hidden",
        type=int,
        nargs="+",
        default=[256, 128],
        help="LSTM hidden sizes to test (default: 256 128)",
    )
    parser.add_argument(
        "--hybrid-blocks",
        type=int,
        default=10,
        help="Residual blocks in Hybrid CNN (default: 10)",
    )
    parser.add_argument(
        "--filters",
        type=int,
        default=256,
        help="CNN filter width (default: 256)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for timing (default: 64)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=20,
        help="Move-history sequence length fed to the LSTM (default: 20)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warm-up iterations before timing (default: 10)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Timed iterations per configuration (default: 50)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/benchmark_lstm_overhead.json",
        help="Path for JSON results (default: logs/benchmark_lstm_overhead.json)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_actions = 4672

    print("=" * 72)
    print("LSTM OVERHEAD BENCHMARK")
    print("=" * 72)
    print(
        f"  Device      : {device}"
        + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    )
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Seq length  : {args.seq_len}")
    print(f"  Warmup/iters: {args.warmup} / {args.iters}")
    print(f"  Hybrid CNN  : {args.hybrid_blocks} blocks, {args.filters} filters")
    print(f"  RNN hidden  : {args.rnn_hidden}")
    print()

    results: Dict = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        ),
        "config": vars(args),
        "runs": [],
    }

    # ── Build CNN-only baseline (shared across all hidden sizes) ─────────────
    # Use the matched block count for the first hidden size; all hybrids share
    # the same CNN backbone so CNN-only block count is effectively the same.
    hybrid_ref = build_hybrid(args.hybrid_blocks, args.filters, args.rnn_hidden[0])
    cnn_blocks = find_matching_cnn_blocks(hybrid_ref, args.filters)
    cnn_only = build_cnn_only(cnn_blocks, args.filters)
    cnn_params = count_params(cnn_only)

    print(f"CNN-only baseline: {cnn_blocks} blocks, {cnn_params:,} params")
    print("Benchmarking CNN-only ...")
    cnn_timing = benchmark_model(
        cnn_only,
        device,
        args.batch_size,
        num_actions,
        args.warmup,
        args.iters,
        use_rnn=False,
        seq_len=args.seq_len,
    )
    print(
        f"  inference : {cnn_timing['inference']['mean_ms']:.2f} ms  "
        f"± {cnn_timing['inference']['std_ms']:.2f}\n"
        f"  train_step: {cnn_timing['train_step']['mean_ms']:.2f} ms  "
        f"± {cnn_timing['train_step']['std_ms']:.2f}"
    )

    results["cnn_only"] = {
        "blocks": cnn_blocks,
        "params": cnn_params,
        "timing": cnn_timing,
    }

    # ── Hybrid runs across hidden sizes ──────────────────────────────────────
    hybrid_results = []
    for hidden in args.rnn_hidden:
        hybrid = build_hybrid(args.hybrid_blocks, args.filters, hidden)
        h_params = count_params(hybrid)
        param_delta = h_params - cnn_params

        print(
            f"\nHybrid rnn_hidden_size={hidden}: {h_params:,} params  "
            f"(+{param_delta:,} vs CNN-only)"
        )
        print(f"Benchmarking Hybrid (hidden={hidden}) ...")
        h_timing = benchmark_model(
            hybrid,
            device,
            args.batch_size,
            num_actions,
            args.warmup,
            args.iters,
            use_rnn=True,
            seq_len=args.seq_len,
        )

        inf_overhead = (
            (h_timing["inference"]["mean_ms"] - cnn_timing["inference"]["mean_ms"])
            / cnn_timing["inference"]["mean_ms"]
            * 100
        )
        train_overhead = (
            (h_timing["train_step"]["mean_ms"] - cnn_timing["train_step"]["mean_ms"])
            / cnn_timing["train_step"]["mean_ms"]
            * 100
        )

        print(
            f"  inference : {h_timing['inference']['mean_ms']:.2f} ms  "
            f"± {h_timing['inference']['std_ms']:.2f}  "
            f"(overhead: {inf_overhead:+.1f}%)\n"
            f"  train_step: {h_timing['train_step']['mean_ms']:.2f} ms  "
            f"± {h_timing['train_step']['std_ms']:.2f}  "
            f"(overhead: {train_overhead:+.1f}%)"
        )

        run = {
            "rnn_hidden_size": hidden,
            "params": h_params,
            "param_delta_vs_cnn": param_delta,
            "timing": h_timing,
            "overhead_pct": {
                "inference": round(inf_overhead, 1),
                "train_step": round(train_overhead, 1),
            },
        }
        hybrid_results.append(run)

    results["runs"] = hybrid_results

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("SUMMARY TABLE")
    print(f"{'=' * 72}")

    col = "{:<22}  {:>10}  {:>12}  {:>12}  {:>10}  {:>10}"
    print(
        col.format(
            "Model", "Params", "Inf (ms)", "Train (ms)", "Inf ovhd", "Train ovhd"
        )
    )
    print("─" * 72)
    print(
        col.format(
            f"CNN-only ({cnn_blocks} blocks)",
            f"{cnn_params:,}",
            f"{cnn_timing['inference']['mean_ms']:.2f} ± {cnn_timing['inference']['std_ms']:.2f}",
            f"{cnn_timing['train_step']['mean_ms']:.2f} ± {cnn_timing['train_step']['std_ms']:.2f}",
            "baseline",
            "baseline",
        )
    )
    for r in hybrid_results:
        print(
            col.format(
                f"Hybrid h={r['rnn_hidden_size']}",
                f"{r['params']:,}",
                f"{r['timing']['inference']['mean_ms']:.2f} ± {r['timing']['inference']['std_ms']:.2f}",
                f"{r['timing']['train_step']['mean_ms']:.2f} ± {r['timing']['train_step']['std_ms']:.2f}",
                f"{r['overhead_pct']['inference']:+.1f}%",
                f"{r['overhead_pct']['train_step']:+.1f}%",
            )
        )

    print(f"\nNote: overhead = (Hybrid − CNN-only) / CNN-only × 100")
    if device.type == "cpu":
        print(
            "⚠  Running on CPU — GPU overhead will be smaller due to "
            "better LSTM parallelism on CUDA cores."
        )

    # ── Save JSON ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nFull results saved to: {args.output}\n")


if __name__ == "__main__":
    main()
