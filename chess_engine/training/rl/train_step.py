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

from typing import Dict, List, Optional, Any, Tuple, Union

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


def _encode_move_histories(
    move_histories: List[List[chess.Move]],
    move_encoder: MoveEncoder,
    max_history: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch of move history lists into padded (batch, max_history) index tensors."""
    tensors = []
    lengths = []
    for moves in move_histories:
        t = torch.zeros(max_history, dtype=torch.long)
        length = max(1, min(len(moves), max_history))
        for i, move in enumerate(moves[:length]):
            try:
                if isinstance(move, str):
                    move = chess.Move.from_uci(move)
                t[i] = move_encoder.encode_move(move)
            except Exception:
                pass
        tensors.append(t)
        lengths.append(length)
    return (
        torch.stack(tensors).to(device),
        torch.tensor(lengths, dtype=torch.long).to(device),
    )


def _read_gate_mean(model: nn.Module) -> Optional[float]:
    """Return last_gate_mean from model.fusion if available, else None."""
    fusion = getattr(model, "fusion", None)
    if fusion is None:
        return None
    return fusion.last_gate_mean


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
    print(f"\nTraining: {config.training_steps_per_iteration} steps...")

    model.train()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    mcts_root_values: List[float] = []
    predicted_values: List[float] = []

    # Tracks CNN vs RNN contribution per step; only populated with gated fusion.
    gate_values: List[float] = []

    per_update = None

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
            batch = replay_buffer.sample(config.batch_size, beta=per_beta)

            board_tensors = [board_encoder.board_to_tensor(b) for b in batch["boards"]]
            boards = torch.stack(board_tensors).to(device)

            move_histories_t, hist_lengths_t = _encode_move_histories(
                batch["move_histories"], move_encoder, config.rnn_max_history, device
            )

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

            values = (
                torch.tensor(batch["values"], dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )
            is_weights = torch.tensor(batch["weights"], dtype=torch.float32).to(device)

            optimizer.zero_grad()

            if use_amp:
                assert (
                    scaler is not None
                ), "Scaler must be initialised when use_amp is True"
                with autocast(device_type=AMP_DEVICE):
                    policy_logits, value_pred, _ = model(
                        boards, move_histories_t, hist_lengths_t
                    )

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
                policy_logits, value_pred, _ = model(
                    boards, move_histories_t, hist_lengths_t
                )

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

            gm = _read_gate_mean(model)
            if gm is not None:
                gate_values.append(gm)

            # Compute new priorities: |TD-error| + epsilon so nothing hits zero
            with torch.no_grad():
                td_errors = (value_pred.squeeze(1) - values.squeeze(1)).abs()
                new_priorities = (td_errors + config.per_epsilon).cpu().tolist()

            all_indices.extend(batch["indices"])
            all_priorities.extend(new_priorities)

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
            total_training_steps += 1

            v_cpu = values.cpu().detach().squeeze()
            mcts_root_values.extend(
                v_cpu.tolist() if v_cpu.dim() > 0 else [v_cpu.item()]
            )
            p_cpu = value_pred.cpu().detach().squeeze()
            predicted_values.extend(
                p_cpu.tolist() if p_cpu.dim() > 0 else [p_cpu.item()]
            )

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

    else:
        buffer_examples: List[Any] = replay_buffer.get_all()

        sample_size = min(
            len(buffer_examples),
            int(len(replay_buffer) * config.sample_ratio),
        )

        if sample_size < len(buffer_examples):
            indices = np.random.choice(len(buffer_examples), sample_size, replace=False)
            buffer_examples = [buffer_examples[i] for i in indices]

        # move history encoded once at dataset construction, not per batch step
        dataset = RLDataset(
            buffer_examples,
            board_encoder,
            move_encoder,
            rnn_max_history=config.rnn_max_history,
        )
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
            for boards, policies, values, move_histories, history_lengths in dataloader:
                boards = boards.to(device)
                policies = policies.to(device)
                values = values.to(device)
                move_histories = move_histories.to(device)
                history_lengths = history_lengths.to(device)

                optimizer.zero_grad()

                if use_amp:
                    assert (
                        scaler is not None
                    ), "Scaler should be initialized when use_amp is True"

                    with autocast(device_type=AMP_DEVICE):
                        policy_logits, value_pred, _ = model(
                            boards, move_histories, history_lengths
                        )

                        # Add small epsilon to avoid log(0)
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

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    policy_logits, value_pred, _ = model(
                        boards, move_histories, history_lengths
                    )

                    # Add small epsilon to avoid log(0)
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

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                gm = _read_gate_mean(model)
                if gm is not None:
                    gate_values.append(gm)

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1
                total_training_steps += 1

                values_cpu = values.cpu().detach().squeeze()
                if values_cpu.dim() == 0:
                    mcts_root_values.append(values_cpu.item())
                else:
                    mcts_root_values.extend(values_cpu.tolist())

                pred_cpu = value_pred.cpu().detach().squeeze()
                if pred_cpu.dim() == 0:
                    predicted_values.append(pred_cpu.item())
                else:
                    predicted_values.extend(pred_cpu.tolist())

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

                if num_batches >= config.training_steps_per_iteration:
                    break

            if num_batches >= config.training_steps_per_iteration:
                break

        pbar.close()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_policy_loss = total_policy_loss / num_batches if num_batches > 0 else 0.0
    avg_value_loss = total_value_loss / num_batches if num_batches > 0 else 0.0

    mean_mcts_value = np.mean(mcts_root_values) if mcts_root_values else 0.0
    mean_pred_value = np.mean(predicted_values) if predicted_values else 0.0
    std_mcts_value = np.std(mcts_root_values) if mcts_root_values else 0.0
    std_pred_value = np.std(predicted_values) if predicted_values else 0.0
    min_mcts_value = np.min(mcts_root_values) if mcts_root_values else 0.0
    max_mcts_value = np.max(mcts_root_values) if mcts_root_values else 0.0

    correlation = 0.0
    if len(mcts_root_values) > 1 and len(predicted_values) > 1:
        mcts_arr = np.array(mcts_root_values)
        pred_arr = np.array(predicted_values)
        if np.std(mcts_arr) > 1e-6 and np.std(pred_arr) > 1e-6:
            correlation = np.corrcoef(mcts_arr, pred_arr)[0, 1]

    # Average fusion gate mean across all training steps this iteration.
    # ~1.0 = CNN dominates; ~0.0 = RNN dominates; ~0.5 = balanced.
    # None when model uses concat/attention fusion or use_rnn=False.
    avg_gate_mean = float(np.mean(gate_values)) if gate_values else None

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
        "gate_mean": avg_gate_mean,
    }

    writer.add_scalar("sanity/mean_mcts_value", mean_mcts_value, current_iteration)
    writer.add_scalar("sanity/mean_pred_value", mean_pred_value, current_iteration)
    writer.add_scalar("sanity/std_mcts_value", std_mcts_value, current_iteration)
    writer.add_scalar("sanity/std_pred_value", std_pred_value, current_iteration)
    writer.add_scalar("sanity/min_mcts_value", min_mcts_value, current_iteration)
    writer.add_scalar("sanity/max_mcts_value", max_mcts_value, current_iteration)
    writer.add_scalar("sanity/value_correlation", correlation, current_iteration)
    if avg_gate_mean is not None:
        writer.add_scalar("fusion/gate_mean", avg_gate_mean, current_iteration)

    print("\nTraining complete")
    print(
        f"   Loss: {avg_loss:.4f} (Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f})"
    )
    print(f"   Total training steps: {total_training_steps}")
    if avg_gate_mean is not None:
        print(f"   Fusion gate mean: {avg_gate_mean:.4f} (1.0=CNN, 0.0=RNN)")
    print(f"\n   Value Sanity Metrics:")
    print(
        f"      MCTS targets: mean={mean_mcts_value:.4f}, std={std_mcts_value:.4f}, "
        f"range=[{min_mcts_value:.4f}, {max_mcts_value:.4f}]"
    )
    print(f"      Predictions: mean={mean_pred_value:.4f}, std={std_pred_value:.4f}")
    print(f"      Correlation: {correlation:.4f}")

    if std_mcts_value < 0.01:
        print(
            f"      WARNING: Value targets have very low variance (std={std_mcts_value:.4f})"
        )
        print(
            f"         This suggests MCTS is producing similar values for all positions."
        )
        print(f"         Model may not be learning to distinguish positions.")

    return metrics, total_training_steps, per_update
