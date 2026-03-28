"""
RL TRAINER - Evaluation

Head-to-head evaluation between current model and best model using MCTS.
"""

from typing import Dict

import chess
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder
from chess_engine.search.mcts.evaluator import Evaluator
from chess_engine.search.mcts.cache import PositionCache
from chess_engine.search.mcts.search import MCTS
from chess_engine.training.self_play.config import SelfPlayConfig


def play_evaluation_game(
    current_model: nn.Module,
    best_model: nn.Module,
    board_encoder: BoardEncoder,
    move_encoder: MoveEncoder,
    device,
    current_plays_white: bool,
    config: SelfPlayConfig,
) -> str:
    """
    Play single evaluation game between current and best models.

    Args:
        current_model: Current training model
        best_model: Best model so far
        board_encoder: Board encoding utility
        move_encoder: Move encoding utility
        device: Torch device
        current_plays_white: Whether current model plays white
        config: Self-play configuration

    Returns:
        "current_win", "best_win", or "draw"
    """

    board = chess.Board()

    # Create MCTS for both models
    # --- Evaluators ---
    current_evaluator = Evaluator(
        model=current_model,
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        use_rnn=config.use_rnn,
        temperature=config.temperature,
        cache=PositionCache(max_size=50_000),
    )

    best_evaluator = Evaluator(
        model=best_model,
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        use_rnn=config.use_rnn,
        temperature=config.temperature,
        cache=PositionCache(max_size=50_000),
    )

    # --- MCTS instances ---
    current_mcts = MCTS(
        model=current_model,
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        use_rnn=config.use_rnn,
        temperature=config.temperature,
    )

    best_mcts = MCTS(
        model=best_model,
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        num_simulations=config.num_simulations,
        c_puct=config.c_puct,
        use_rnn=config.use_rnn,
        temperature=config.temperature,
    )

    # Play game
    move_count = 0
    while not board.is_game_over() and move_count < config.max_moves:
        # Determine which model's turn
        if board.turn == chess.WHITE:
            mcts = current_mcts if current_plays_white else best_mcts
        else:
            mcts = best_mcts if current_plays_white else current_mcts

        # Get move
        move, _ = mcts.search(board)

        if move is None:
            break

        board.push(move)
        move_count += 1

    # Determine result
    if board.is_checkmate():
        # Winner is opposite of whose turn it is
        white_won = not board.turn
        if (white_won and current_plays_white) or (
            not white_won and not current_plays_white
        ):
            return "current_win"
        else:
            return "best_win"
    else:
        # Draw (stalemate, repetition, 50-move, insufficient material, or max moves)
        return "draw"


def execute_evaluation_step(
    current_model: nn.Module,
    best_model: nn.Module,
    board_encoder: BoardEncoder,
    move_encoder: MoveEncoder,
    device,
    config,
    writer: SummaryWriter,
    current_iteration: int,
    best_iteration: int,
    best_win_rate: float,
) -> tuple:
    """
    Evaluate current model against best model via head-to-head matches.

    Args:
        current_model: Current training model
        best_model: Best model so far
        board_encoder: Board encoding utility
        move_encoder: Move encoding utility
        device: Torch device
        config: RLTrainingConfig
        writer: TensorBoard writer
        current_iteration: Current iteration number
        best_iteration: Best model iteration
        best_win_rate: Best win rate so far

    Returns:
        Tuple of (eval_metrics, updated_best_model_flag, updated_best_iteration, updated_best_win_rate)
    """
    print(f"\n⚔️  Evaluation: Playing {config.eval_games} games vs best model...")

    current_model.eval()
    best_model.eval()

    # Play evaluation games
    wins = 0
    losses = 0
    draws = 0

    eval_config = SelfPlayConfig(
        num_simulations=config.eval_simulations,
        c_puct=config.c_puct,
        temperature=0.1,  # Lower temperature for evaluation
        temperature_threshold=0,  # Deterministic play
        max_moves=config.max_moves_per_game,
        use_rnn=config.use_rnn,
        dirichlet_alpha=0.0,  # No exploration during evaluation
        resign_threshold=config.resign_threshold,
    )

    for game_num in tqdm(range(config.eval_games), desc="Evaluation games"):
        # Alternate colors
        if game_num % 2 == 0:
            current_model_plays_white = True
        else:
            current_model_plays_white = False

        result = play_evaluation_game(
            current_model=current_model,
            best_model=best_model,
            board_encoder=board_encoder,
            move_encoder=move_encoder,
            device=device,
            current_plays_white=current_model_plays_white,
            config=eval_config,
        )

        if result == "current_win":
            wins += 1
        elif result == "best_win":
            losses += 1
        else:
            draws += 1

    # Calculate win rate
    total_games = wins + losses + draws
    win_rate = (wins + 0.5 * draws) / total_games if total_games > 0 else 0.0

    metrics = {
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "draws": draws,
    }

    print(f"\n✅ Evaluation complete")
    print(f"   Win rate: {win_rate:.1%} ({wins}W / {losses}L / {draws}D)")

    # Determine if best model should be updated
    should_update_best = False
    updated_best_iteration = best_iteration
    updated_best_win_rate = best_win_rate

    if win_rate >= config.win_threshold and win_rate > best_win_rate:
        print(
            f"\n🏆 New best model! (win rate: {win_rate:.1%} >= {config.win_threshold:.1%}, beats previous best {best_win_rate:.1%})"
        )
        should_update_best = True
        updated_best_iteration = current_iteration
        updated_best_win_rate = win_rate
    elif win_rate >= config.win_threshold:
        print(
            f"\n   Passed threshold ({win_rate:.1%} >= {config.win_threshold:.1%}) but not better than current best ({best_win_rate:.1%})"
        )
    else:
        print(
            f"\n   Current model not better than best (need {config.win_threshold:.1%})"
        )

        if win_rate > best_win_rate:
            print(
                f"   However, this is better than previous best ({best_win_rate:.1%})"
            )
            should_update_best = True
            updated_best_iteration = current_iteration
            updated_best_win_rate = win_rate

    # Log to tensorboard
    writer.add_scalar("eval/win_rate", win_rate, current_iteration)
    writer.add_scalar("eval/wins", wins, current_iteration)
    writer.add_scalar("eval/losses", losses, current_iteration)
    writer.add_scalar("eval/draws", draws, current_iteration)

    return metrics, should_update_best, updated_best_iteration, updated_best_win_rate
