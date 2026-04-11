"""
RL TRAINER - Evaluation

Head-to-head evaluation between current model and best model using MCTS.
"""

import os
import signal
import tempfile
from multiprocessing import Pool, cpu_count
from typing import Dict, Optional

import chess
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from chess_engine.models.hybrid.config import HybridModelConfig
from chess_engine.models.hybrid.hybrid_net import HybridChessNet
from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder
from chess_engine.search.mcts.search import MCTS


# Per-worker process state — populated once by _init_eval_worker(), reused per game.
_eval_current_mcts: Optional[MCTS] = None
_eval_best_mcts: Optional[MCTS] = None
_eval_max_moves: int = 150


def _init_eval_worker(current_path: str, best_path: str, config_dict: dict) -> None:
    """
    Pool initializer — runs once per worker process.
    Loads both models from disk and builds MCTS instances into module globals.
    """
    global _eval_current_mcts, _eval_best_mcts, _eval_max_moves

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _eval_max_moves = config_dict["max_moves"]

    model_config = HybridModelConfig(
        cnn_input_channels=22,
        cnn_filters=config_dict["cnn_filters"],
        cnn_residual_blocks=config_dict["cnn_blocks"],
        cnn_dropout=config_dict.get("cnn_dropout", 0.0),
        use_rnn=config_dict["use_rnn"],
        rnn_hidden_size=config_dict.get("rnn_hidden_size", 256),
        rnn_num_layers=config_dict.get("rnn_layers", 2),
        rnn_dropout=config_dict.get("rnn_dropout", 0.0),
        rnn_use_attention=config_dict.get("rnn_use_attention", False),
        rnn_bidirectional=config_dict.get("rnn_bidirectional", False),
        fusion_type=config_dict.get("fusion_type", "gated"),
        num_actions=config_dict["num_actions"],
    )

    def _load_model(path: str) -> nn.Module:
        model = HybridChessNet(model_config).to(device)
        state = torch.load(path, map_location=device, weights_only=False)
        state_dict = state["model_state_dict"] if "model_state_dict" in state else state
        model.load_state_dict(state_dict)
        model.eval()
        return model

    board_encoder = BoardEncoder()
    move_encoder = MoveEncoder()

    _eval_current_mcts = MCTS(
        model=_load_model(current_path),
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        num_simulations=config_dict["eval_simulations"],
        c_puct=config_dict["c_puct"],
        temperature=0.1,
        use_rnn=config_dict["use_rnn"],
        rnn_max_history=config_dict["rnn_max_history"],
        dirichlet_epsilon=0.0,
        dirichlet_alpha=config_dict["dirichlet_alpha"],
    )

    _eval_best_mcts = MCTS(
        model=_load_model(best_path),
        board_encoder=board_encoder,
        move_encoder=move_encoder,
        device=device,
        num_simulations=config_dict["eval_simulations"],
        c_puct=config_dict["c_puct"],
        temperature=0.1,
        use_rnn=config_dict["use_rnn"],
        rnn_max_history=config_dict["rnn_max_history"],
        dirichlet_epsilon=0.0,
        dirichlet_alpha=config_dict["dirichlet_alpha"],
    )


def _play_eval_game_worker(game_num: int) -> str:
    """
    Lightweight per-game worker — MCTS instances already initialized by _init_eval_worker().
    Plays one evaluation game and returns the result string.
    """
    assert _eval_current_mcts is not None and _eval_best_mcts is not None, \
        "_init_eval_worker() was not called"
    return play_evaluation_game(
        current_mcts=_eval_current_mcts,
        best_mcts=_eval_best_mcts,
        current_plays_white=(game_num % 2 == 0),
        max_moves=_eval_max_moves,
    )


def play_evaluation_game(
    current_mcts: MCTS,
    best_mcts: MCTS,
    current_plays_white: bool,
    max_moves: int,
) -> str:
    """
    Play single evaluation game between current and best models.

    Args:
        current_mcts: Pre-built MCTS for the current model (reused across games)
        best_mcts: Pre-built MCTS for the best model (reused across games)
        current_plays_white: Whether current model plays white
        max_moves: Maximum moves before declaring draw

    Returns:
        "current_win", "best_win", or "draw"
    """

    board = chess.Board()

    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        if board.turn == chess.WHITE:
            mcts = current_mcts if current_plays_white else best_mcts
        else:
            mcts = current_mcts if not current_plays_white else best_mcts

        move, _ = mcts.search(board)

        if move is None:
            break

        board.push(move)
        move_count += 1

    if board.is_checkmate():
        white_won = not board.turn
        if (white_won and current_plays_white) or (
            not white_won and not current_plays_white
        ):
            return "current_win"
        else:
            return "best_win"
    else:
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
    num_workers = config.num_workers or 1
    print(f"\n⚔️  Evaluation: Playing {config.eval_games} games vs best model"
          f" ({num_workers} worker{'s' if num_workers > 1 else ''})...")

    current_model.eval()
    best_model.eval()

    wins = 0
    losses = 0
    draws = 0

    if num_workers > 1:
        # --- Parallel path ---
        # Save both model state dicts to temp files so worker processes can load them.
        current_tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        best_tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        try:
            torch.save(current_model.state_dict(), current_tmp.name)
            torch.save(best_model.state_dict(), best_tmp.name)
            current_tmp.close()
            best_tmp.close()

            config_dict = {
                "eval_simulations": config.eval_simulations,
                "c_puct": config.c_puct,
                "max_moves": config.max_moves_per_game,
                "use_rnn": config.use_rnn,
                "rnn_max_history": config.rnn_max_history,
                "dirichlet_alpha": config.dirichlet_alpha,
                # Model architecture — needed to reconstruct HybridChessNet in each worker.
                "cnn_filters": config.cnn_filters,
                "cnn_blocks": config.cnn_blocks,
                "use_rnn": config.use_rnn,
                "rnn_hidden_size": config.rnn_hidden_size,
                "rnn_layers": config.rnn_layers,
                "rnn_use_attention": config.rnn_use_attention,
                "rnn_bidirectional": config.rnn_bidirectional,
                "fusion_type": config.fusion_type,
                "num_actions": config.num_actions,
            }

            with Pool(
                processes=num_workers,
                initializer=_init_eval_worker,
                initargs=(current_tmp.name, best_tmp.name, config_dict),
            ) as pool:
                results = list(
                    tqdm(
                        pool.imap(_play_eval_game_worker, range(config.eval_games)),
                        total=config.eval_games,
                        desc="Evaluation games (parallel)",
                    )
                )

            for result in results:
                if result == "current_win":
                    wins += 1
                elif result == "best_win":
                    losses += 1
                else:
                    draws += 1

        finally:
            os.unlink(current_tmp.name)
            os.unlink(best_tmp.name)

    else:
        # --- Sequential path ---
        # Build MCTS instances once — reused across all eval games.
        # Cache stays warm between games (deterministic eval revisits same openings).
        current_mcts = MCTS(
            model=current_model,
            board_encoder=board_encoder,
            move_encoder=move_encoder,
            device=device,
            num_simulations=config.eval_simulations,
            c_puct=config.c_puct,
            temperature=0.1,
            use_rnn=config.use_rnn,
            rnn_max_history=config.rnn_max_history,
            dirichlet_epsilon=0.0,
            dirichlet_alpha=config.dirichlet_alpha,
        )

        best_mcts = MCTS(
            model=best_model,
            board_encoder=board_encoder,
            move_encoder=move_encoder,
            device=device,
            num_simulations=config.eval_simulations,
            c_puct=config.c_puct,
            temperature=0.1,
            use_rnn=config.use_rnn,
            rnn_max_history=config.rnn_max_history,
            dirichlet_epsilon=0.0,
            dirichlet_alpha=config.dirichlet_alpha,
        )

        for game_num in tqdm(range(config.eval_games), desc="Evaluation games"):
            result = play_evaluation_game(
                current_mcts=current_mcts,
                best_mcts=best_mcts,
                current_plays_white=(game_num % 2 == 0),
                max_moves=config.max_moves_per_game,
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
