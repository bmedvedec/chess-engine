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
    print(f"\nSelf-Play: Generating {config.games_per_iteration} games...")

    model.eval()

    selfplay_config = SelfPlayConfig(
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        temperature=config.temperature,
        temperature_threshold=config.temperature_threshold,
        late_game_temperature=config.late_game_temperature,
        max_moves=config.max_moves_per_game,
        use_rnn=config.use_rnn,
        rnn_max_history=config.rnn_max_history,
        dirichlet_alpha=config.dirichlet_alpha,
        dirichlet_epsilon=config.dirichlet_epsilon,
        resign_threshold=config.resign_threshold,
        value_blend_alpha=config.value_blend_alpha,
        draw_value_penalty=config.draw_value_penalty,
        random_opening_moves=config.random_opening_moves,
    )

    start_time = time.time()

    if config.use_parallel_selfplay:
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

    print(f"\nGenerated {len(examples)} training examples")
    print(
        f"   Time: {elapsed:.1f}s ({elapsed/config.games_per_iteration:.1f}s per game)"
    )
    print(f"   Buffer size: {len(replay_buffer)}/{config.buffer_size}")
    print(f"   Total games played: {total_games_played}")

    return examples, total_games_played, stats, elapsed
