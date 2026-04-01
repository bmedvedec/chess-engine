"""
RL TRAINER - Self-Play Runner

Generates self-play games using MCTS, supporting both sequential and
parallel execution modes.
"""

import os
import time

import torch
import torch.nn as nn

from chess_engine.training.rl.rl_config import RLTrainingConfig
from chess_engine.training.self_play.config import SelfPlayConfig
from chess_engine.training.self_play.game_runner import SelfPlayGameRunner
from chess_engine.training.self_play.parallel import ParallelSelfPlay
from chess_engine.data.replay.buffer import ReplayBuffer


def execute_self_play_step(
    model: nn.Module,
    config: RLTrainingConfig,
    replay_buffer: ReplayBuffer,
    device: torch.device,
    total_games_played: int,
) -> tuple:
    """
    Execute self-play game generation.

    Args:
        model: Neural network model
        config: RL training configuration
        replay_buffer: Replay buffer to store examples
        device: Torch device
        total_games_played: Running count of total games played

    Returns:
        Tuple of (examples, updated_total_games_played, stats, selfplay_seconds)
        where stats is a SelfPlayStatistics object and selfplay_seconds is the
        wall-clock time for this self-play phase.
    """
    print(f"\n🎮 Self-Play: Generating {config.games_per_iteration} games...")

    model.eval()

    # Create self-play configuration
    selfplay_config = SelfPlayConfig(
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        temperature=config.temperature,
        temperature_threshold=config.temperature_threshold,
        max_moves=config.max_moves_per_game,
        use_rnn=config.use_rnn,
        rnn_max_history=config.rnn_max_history,
        dirichlet_alpha=config.dirichlet_alpha,
        resign_threshold=config.resign_threshold,
    )

    # Generate games
    start_time = time.time()

    if config.use_parallel_selfplay:
        # Save current model for parallel workers
        temp_model_path = os.path.join(config.checkpoint_dir, "temp_model.pt")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config.to_dict(),
                "model_config": {
                    "cnn_residual_blocks": config.cnn_blocks,
                    "use_rnn": config.use_rnn,
                },
            },
            temp_model_path,
        )

        # Parallel self-play
        worker = ParallelSelfPlay(
            model_path=temp_model_path,
            config=selfplay_config,
            num_workers=config.num_workers,
            model_config={
                "cnn_filters": config.cnn_filters,
                "cnn_blocks": config.cnn_blocks,
                "num_actions": config.num_actions,
                "rnn_hidden_size": config.rnn_hidden_size,
                "rnn_layers": config.rnn_layers,
                "rnn_use_attention": config.rnn_use_attention,
                "rnn_bidirectional": config.rnn_bidirectional,
                "fusion_type": config.fusion_type,
            },
        )
        examples, stats = worker.play_games_parallel(
            num_games=config.games_per_iteration,
            buffer=replay_buffer,
        )
    else:
        # Sequential self-play
        worker = SelfPlayGameRunner(
            model=model,
            device=device,
            config=selfplay_config,
        )
        examples, stats = worker.play_games(
            num_games=config.games_per_iteration,
            buffer=replay_buffer,
        )

    elapsed = time.time() - start_time
    total_games_played += config.games_per_iteration

    print(f"\n✅ Generated {len(examples)} training examples")
    print(
        f"   Time: {elapsed:.1f}s ({elapsed/config.games_per_iteration:.1f}s per game)"
    )
    print(f"   Buffer size: {len(replay_buffer)}/{config.buffer_size}")
    print(f"   Total games played: {total_games_played}")

    return examples, total_games_played, stats, elapsed
