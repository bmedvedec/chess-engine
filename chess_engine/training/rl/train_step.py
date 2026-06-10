"""
RL TRAINER - Training Step

Executes a single training iteration on the replay buffer with support for
both mixed precision (AMP) and standard FP32 training paths.

Includes soft cross-entropy loss for MCTS probability distributions and
comprehensive sanity metrics for value target verification.

When Prioritized Experience Replay (PER) is enabled (per_beta is not None),
each training step samples a fresh prioritized batch from the buffer, applies
importance-sampling (IS) weights to correct the sampling bias, computes new
priorities from the TD-error, and returns them so the trainer can push them
back into the buffer.
"""

from typing import Dict, List, Optional, Any, Union

import chess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from chess_engine.training.rl.rl_config import RLTrainingConfig
from chess_engine.training.rl.rl_dataset import RLDataset
from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder
from chess_engine.data.replay.buffer import ReplayBuffer
from chess_engine.data.replay.prioritized import PrioritizedReplayBuffer

AMP_DEVICE = "cuda"


def execute_training_step(
    model: nn.Module,
    replay_buffer: Union[ReplayBuffer, PrioritizedReplayBuffer],
    optimizer: torch.optim.Optimizer,
    config: RLTrainingConfig,
    board_encoder: BoardEncoder,
    move_encoder: MoveEncoder,
    device: torch.device,
    use_amp: bool,
    scaler: Optional[GradScaler],
    value_criterion: nn.Module,
    writer: SummaryWriter,
    current_iteration: int,
    total_training_steps: int,
    per_beta: Optional[float] = None,
) -> tuple:
    """
    Execute network training on replay buffer.

    Args:
        model: Neural network model
        replay_buffer: Replay buffer with training examples
        optimizer: Optimizer
        config: Training configuration
        board_encoder: Board encoding utility
        move_encoder: Move encoding utility
        device: Torch device
        use_amp: Whether to use automatic mixed precision
        scaler: GradScaler for AMP (None if not using AMP)
        value_criterion: Value loss function (MSELoss)
        writer: TensorBoard writer
        current_iteration: Current iteration number
        total_training_steps: Running count of total training steps
        per_beta: Importance-sampling exponent for PER (None = uniform replay).
                  When set, each step samples directly from the PrioritizedReplayBuffer,
                  applies IS weights to the loss, and returns updated priorities.

    Returns:
        Tuple of (metrics_dict, updated_total_training_steps, per_update).
        per_update is None when uniform replay is used; for PER it is a dict
        with 'indices' and 'priorities' for the trainer to push back to the buffer.
    """
    print(f"\n📚 Training: {config.training_steps_per_iteration} steps...")

    model.train()

    # Shared metric accumulators (populated by whichever training path runs)
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    # Sanity metrics for value target verification
    mcts_root_values: List[float] = []
    predicted_values: List[float] = []

    # Will be populated by the PER path; None signals uniform replay was used.
    per_update = None

    # =========================================================================
    # PER TRAINING PATH
    # =========================================================================
    if per_beta is not None:
        # Narrow the type: per_beta is only set when use_prioritized_replay=True,
        # which guarantees the buffer is a PrioritizedReplayBuffer.  The assertion
        # gives Pylance enough information to resolve .sample(beta=...) correctly.
        assert isinstance(replay_buffer, PrioritizedReplayBuffer), (
            "per_beta is set but replay_buffer is not a PrioritizedReplayBuffer. "
            "Ensure config.use_prioritized_replay=True when passing per_beta."
        )

        # Sample a fresh prioritized batch each step so higher-error positions
        # receive more gradient updates.  IS weights correct the resulting bias.
        pbar = tqdm(total=config.training_steps_per_iteration, desc="Training (PER)")

        all_indices: List[int] = []
        all_priorities: List[float] = []

        for _step in range(config.training_steps_per_iteration):
            # sample() on PrioritizedReplayBuffer returns boards as chess.Board
            # objects plus IS weights and the buffer indices of each sample.
            batch = replay_buffer.sample(config.batch_size, beta=per_beta)

            # --- encode boards ---
            board_tensors = [board_encoder.board_to_tensor(b) for b in batch["boards"]]
            boards = torch.stack(board_tensors).to(device)

            # --- encode policies (dict or pre-encoded tensor) ---
            policy_tensors = []
            for policy in batch["policies"]:
                p = torch.zeros(move_encoder.num_moves)
                if isinstance(policy, dict):
                    for move_uci, prob in policy.items():
                        try:
                            mv = chess.Move.from_uci(move_uci)
                            idx = move_encoder.encode_move(mv)
                            p[idx] = float(prob)
                        except Exception:
                            pass
                elif isinstance(policy, torch.Tensor):
                    p = policy.clone().float()
                if p.sum() > 0:
                    p = p / p.sum()
                policy_tensors.append(p)
            policies = torch.stack(policy_tensors).to(device)

            # --- values and importance-sampling weights ---
            values = (
                torch.tensor(batch["values"], dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )
            is_weights = torch.tensor(batch["weights"], dtype=torch.float32).to(device)

            optimizer.zero_grad()

            if use_amp:
                # MIXED PRECISION PATH
                assert (
                    scaler is not None
                ), "Scaler must be initialised when use_amp is True"
                with autocast(device_type=AMP_DEVICE):
                    policy_logits, value_pred, _ = model(boards)

                    policies_safe = policies + 1e-8
                    policies_safe = policies_safe / policies_safe.sum(
                        dim=1, keepdim=True
                    )
                    log_probs = torch.nn.functional.log_softmax(policy_logits, dim=1)

                    # Per-sample losses (keep batch dim for IS weighting)
                    per_sample_policy = torch.sum(-policies_safe * log_probs, dim=1)
                    per_sample_value = (value_pred.squeeze(1) - values.squeeze(1)).pow(
                        2
                    )

                    policy_loss = torch.mean(is_weights * per_sample_policy)
                    value_loss = torch.mean(is_weights * per_sample_value)
                    loss = (
                        config.policy_loss_weight * policy_loss
                        + config.value_loss_weight * value_loss
                    )

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            else:
                # STANDARD FP32 PATH
                policy_logits, value_pred, _ = model(boards)

                policies_safe = policies + 1e-8
                policies_safe = policies_safe / policies_safe.sum(dim=1, keepdim=True)
                log_probs = torch.nn.functional.log_softmax(policy_logits, dim=1)

                per_sample_policy = torch.sum(-policies_safe * log_probs, dim=1)
                per_sample_value = (value_pred.squeeze(1) - values.squeeze(1)).pow(2)

                policy_loss = torch.mean(is_weights * per_sample_policy)
                value_loss = torch.mean(is_weights * per_sample_value)
                loss = (
                    config.policy_loss_weight * policy_loss
                    + config.value_loss_weight * value_loss
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Compute new priorities: |TD-error| + epsilon so nothing hits zero
            with torch.no_grad():
                td_errors = (value_pred.squeeze(1) - values.squeeze(1)).abs()
                new_priorities = (td_errors + config.per_epsilon).cpu().tolist()

            all_indices.extend(batch["indices"])
            all_priorities.extend(new_priorities)

            # --- accumulate metrics ---
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
            total_training_steps += 1

            # Sanity metrics
            v_cpu = values.cpu().detach().squeeze()
            mcts_root_values.extend(
                v_cpu.tolist() if v_cpu.dim() > 0 else [v_cpu.item()]
            )
            p_cpu = value_pred.cpu().detach().squeeze()
            predicted_values.extend(
                p_cpu.tolist() if p_cpu.dim() > 0 else [p_cpu.item()]
            )

            # TensorBoard logging
            if total_training_steps % config.log_frequency == 0:
                writer.add_scalar("train/loss", loss.item(), total_training_steps)
                writer.add_scalar(
                    "train/policy_loss", policy_loss.item(), total_training_steps
                )
                writer.add_scalar(
                    "train/value_loss", value_loss.item(), total_training_steps
                )
                writer.add_scalar(
                    "train/learning_rate",
                    optimizer.param_groups[0]["lr"],
                    total_training_steps,
                )
                writer.add_scalar("train/epoch", 0, total_training_steps)

            pbar.update(1)

        pbar.close()
        per_update = {"indices": all_indices, "priorities": all_priorities}

    # =========================================================================
    # UNIFORM TRAINING PATH  (original DataLoader-based approach)
    # =========================================================================
    else:
        # Sample from replay buffer - get all examples
        buffer_examples = replay_buffer.get_all()

        # Determine sample size
        sample_size = min(
            len(buffer_examples),
            int(len(replay_buffer) * config.sample_ratio),
        )

        if sample_size < len(buffer_examples):
            indices = np.random.choice(len(buffer_examples), sample_size, replace=False)
            buffer_examples = [buffer_examples[i] for i in indices]

        # Create dataset and dataloader
        dataset = RLDataset(buffer_examples, board_encoder, move_encoder)
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        steps_per_epoch = len(dataloader)
        num_epochs = max(
            1, -(-config.training_steps_per_iteration // steps_per_epoch)
        )  # ceiling division

        pbar = tqdm(total=config.training_steps_per_iteration, desc="Training")

        for epoch in range(num_epochs):
            for boards, policies, values in dataloader:
                # Move to device
                boards = boards.to(device)
                policies = policies.to(device)
                values = values.to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass with optional mixed precision
                if use_amp:
                    # MIXED PRECISION PATH
                    assert (
                        scaler is not None
                    ), "Scaler should be initialized when use_amp is True"

                    autocast_context = autocast(device_type=AMP_DEVICE)

                    with autocast_context:
                        # Forward pass in fp16
                        policy_logits, value_pred, _ = model(boards)

                        # Policy loss: Soft cross-entropy for MCTS probability distributions
                        # Add small epsilon to avoid log(0) numerical issues
                        policies_safe = policies + 1e-8
                        policies_safe = policies_safe / policies_safe.sum(
                            dim=1, keepdim=True
                        )

                        log_probs = torch.nn.functional.log_softmax(
                            policy_logits, dim=1
                        )
                        policy_loss = torch.mean(
                            torch.sum(-policies_safe * log_probs, dim=1)
                        )

                        value_loss = value_criterion(value_pred, values)
                        loss = (
                            config.policy_loss_weight * policy_loss
                            + config.value_loss_weight * value_loss
                        )

                    # Backward pass with scaled gradients
                    scaler.scale(loss).backward()

                    # Unscale before gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Optimizer step with scaler
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    # STANDARD FP32 PATH (fallback for CPU or when AMP disabled)
                    policy_logits, value_pred, _ = model(boards)

                    # Policy loss: Soft cross-entropy for MCTS probability distributions
                    # Add small epsilon to avoid log(0) numerical issues
                    policies_safe = policies + 1e-8
                    policies_safe = policies_safe / policies_safe.sum(
                        dim=1, keepdim=True
                    )

                    log_probs = torch.nn.functional.log_softmax(policy_logits, dim=1)
                    policy_loss = torch.mean(
                        torch.sum(-policies_safe * log_probs, dim=1)
                    )

                    value_loss = value_criterion(value_pred, values)
                    loss = (
                        config.policy_loss_weight * policy_loss
                        + config.value_loss_weight * value_loss
                    )

                    # Backward pass
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Optimizer step
                    optimizer.step()

                # Track metrics
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1
                total_training_steps += 1

                # Collect sanity metrics (for value target verification)
                # Handle both scalar and tensor cases
                values_cpu = values.cpu().detach().squeeze()
                if values_cpu.dim() == 0:
                    # Scalar tensor
                    mcts_root_values.append(values_cpu.item())
                else:
                    # Tensor with multiple values
                    mcts_root_values.extend(values_cpu.tolist())

                pred_cpu = value_pred.cpu().detach().squeeze()
                if pred_cpu.dim() == 0:
                    # Scalar tensor
                    predicted_values.append(pred_cpu.item())
                else:
                    # Tensor with multiple values
                    predicted_values.extend(pred_cpu.tolist())

                # Log to tensorboard
                if total_training_steps % config.log_frequency == 0:
                    writer.add_scalar("train/loss", loss.item(), total_training_steps)
                    writer.add_scalar(
                        "train/policy_loss",
                        policy_loss.item(),
                        total_training_steps,
                    )
                    writer.add_scalar(
                        "train/value_loss",
                        value_loss.item(),
                        total_training_steps,
                    )
                    writer.add_scalar(
                        "train/learning_rate",
                        optimizer.param_groups[0]["lr"],
                        total_training_steps,
                    )
                    writer.add_scalar("train/epoch", epoch, total_training_steps)

                pbar.update(1)

                # Break if we've done enough steps
                if num_batches >= config.training_steps_per_iteration:
                    break

            if num_batches >= config.training_steps_per_iteration:
                break

        pbar.close()

    # =========================================================================
    # SHARED: compute and log summary metrics
    # =========================================================================
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0.0
    avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0.0

    # Compute sanity metrics for value target verification
    mean_mcts_value = np.mean(mcts_root_values) if mcts_root_values else 0.0
    mean_pred_value = np.mean(predicted_values) if predicted_values else 0.0
    std_mcts_value = np.std(mcts_root_values) if mcts_root_values else 0.0
    std_pred_value = np.std(predicted_values) if predicted_values else 0.0
    min_mcts_value = np.min(mcts_root_values) if mcts_root_values else 0.0
    max_mcts_value = np.max(mcts_root_values) if mcts_root_values else 0.0

    # Compute correlation (Pearson) between MCTS targets and network predictions
    correlation = 0.0
    if len(mcts_root_values) > 1 and len(predicted_values) > 1:
        mcts_arr = np.array(mcts_root_values)
        pred_arr = np.array(predicted_values)
        if np.std(mcts_arr) > 1e-6 and np.std(pred_arr) > 1e-6:
            correlation = np.corrcoef(mcts_arr, pred_arr)[0, 1]

    metrics = {
        "loss": avg_loss,
        "policy_loss": avg_policy_loss,
        "value_loss": avg_value_loss,
        "mean_mcts_value": mean_mcts_value,
        "mean_pred_value": mean_pred_value,
        "std_mcts_value": std_mcts_value,
        "std_pred_value": std_pred_value,
        "min_mcts_value": min_mcts_value,
        "max_mcts_value": max_mcts_value,
        "value_correlation": correlation,
    }

    # Log sanity metrics
    writer.add_scalar("sanity/mean_mcts_value", mean_mcts_value, current_iteration)
    writer.add_scalar("sanity/mean_pred_value", mean_pred_value, current_iteration)
    writer.add_scalar("sanity/std_mcts_value", std_mcts_value, current_iteration)
    writer.add_scalar("sanity/std_pred_value", std_pred_value, current_iteration)
    writer.add_scalar("sanity/min_mcts_value", min_mcts_value, current_iteration)
    writer.add_scalar("sanity/max_mcts_value", max_mcts_value, current_iteration)
    writer.add_scalar("sanity/value_correlation", correlation, current_iteration)

    print(f"\n✅ Training complete")
    print(
        f"   Loss: {avg_loss:.4f} (Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})"
    )
    print(f"   Total training steps: {total_training_steps}")
    print(f"\n   Value Sanity Metrics:")
    print(
        f"      MCTS targets: mean={mean_mcts_value:.4f}, std={std_mcts_value:.4f}, "
        f"range=[{min_mcts_value:.4f}, {max_mcts_value:.4f}]"
    )
    print(f"      Predictions: mean={mean_pred_value:.4f}, std={std_pred_value:.4f}")
    print(f"      Correlation: {correlation:.4f}")

    # Warning if value targets are collapsing
    if std_mcts_value < 0.01:
        print(
            f"      ⚠️  WARNING: Value targets have very low variance (std={std_mcts_value:.4f})"
        )
        print(
            f"         This suggests MCTS is producing similar values for all positions."
        )
        print(f"         Model may not be learning to distinguish positions.")

    return metrics, total_training_steps, per_update
