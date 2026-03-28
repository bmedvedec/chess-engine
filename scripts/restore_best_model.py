"""
Restore best_model.pt and patch latest.pt to use a specific checkpoint's best model.

Usage:
    python scripts/restore_best_model.py \
        --checkpoint data/rl_checkpoints/phase1/checkpoint_iteration_105.pt \
        --latest    data/rl_checkpoints/phase1/latest.pt \
        --win-rate  0.60
"""

import argparse
import shutil
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Source checkpoint .pt file")
    parser.add_argument("--latest",     required=True, help="latest.pt to patch")
    parser.add_argument("--win-rate",   type=float, required=True,
                        help="Known win rate of this checkpoint (e.g. 0.60)")
    args = parser.parse_args()

    ckpt_path   = Path(args.checkpoint)
    latest_path = Path(args.latest)
    best_path   = ckpt_path.parent / "best_model.pt"

    # ── 1. Load source checkpoint ────────────────────────────────────────────
    print(f"Loading {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    iteration    = ckpt["iteration"]          # 0-based index stored in file
    win_rate     = args.win_rate

    # best_model_state_dict is the promoted model saved at this iteration
    best_state = ckpt.get("best_model_state_dict") or ckpt["model_state_dict"]

    # ── 2. Build best_model.pt ───────────────────────────────────────────────
    # Back up existing best_model.pt first
    if best_path.exists():
        backup = best_path.with_suffix(".pt.bak")
        shutil.copy2(best_path, backup)
        print(f"Backed up existing best_model.pt → {backup.name}")

    best_ckpt = {
        "iteration":        iteration,
        "model_state_dict": best_state,
        "win_rate":         win_rate,
        "config":           ckpt.get("config", {}),
        "model_config": {
            "cnn_residual_blocks": ckpt.get("config", {}).get("cnn_blocks", 10),
            "use_rnn":             ckpt.get("config", {}).get("use_rnn", False),
        },
    }
    torch.save(best_ckpt, best_path)
    print(f"Saved best_model.pt  (iteration={iteration + 1}, win_rate={win_rate:.1%})")

    # ── 3. Patch latest.pt ───────────────────────────────────────────────────
    print(f"\nPatching {latest_path} ...")
    latest = torch.load(latest_path, map_location="cpu")

    old_bi  = latest.get("best_iteration", "?")
    old_wr  = latest.get("best_win_rate",  "?")

    latest["best_iteration"]        = iteration
    latest["best_win_rate"]         = win_rate
    latest["best_model_state_dict"] = best_state

    torch.save(latest, latest_path)
    print(f"  best_iteration : {old_bi} → {iteration}  (iter {iteration + 1})")
    print(f"  best_win_rate  : {old_wr} → {win_rate:.1%}")
    print(f"  best_model_state_dict updated")

    print("\nDone. Resume training and it will treat iteration 105 as the baseline.")


if __name__ == "__main__":
    main()
