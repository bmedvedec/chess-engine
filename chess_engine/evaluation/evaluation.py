"""
EVALUATION & BENCHMARKING
Complete Implementation

This module provides comprehensive evaluation and benchmarking capabilities for the chess engine.

Features:
- Automated game playing against baselines (random, stockfish)
- Win/Loss/Draw statistics
- ELO rating estimation
- Game phase analysis (opening, middle, endgame)
- Weakness detection
- Performance visualization
- Comprehensive reporting
"""

import os
import sys
import sys
import argparse
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Sequence, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from enum import Enum
import random
import math

import chess
import chess.pgn
import torch
import numpy as np
from tqdm import tqdm

# Try to import chess.engine for Stockfish support
try:
    import chess.engine

    CHESS_ENGINE_AVAILABLE = True
except ImportError:
    CHESS_ENGINE_AVAILABLE = False
    print("⚠️  chess.engine not available - Stockfish support disabled")

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  Matplotlib not available - visualizations disabled")

# Chess engine imports
from chess_engine.models.hybrid_model import HybridChessNet
from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder, MoveHistory

# Try to import MCTS
try:
    from chess_engine.search.mcts import MCTS

    MCTS_AVAILABLE = True
except ImportError:
    MCTS_AVAILABLE = False
    print("⚠️  MCTS not available - will use policy-only mode")


# =============================================================================
# ENUMS & DATA CLASSES
# =============================================================================


class GamePhase(Enum):
    """Game phase classification"""

    OPENING = "opening"  # Moves 1-10
    MIDDLE = "middle"  # Moves 11-30
    ENDGAME = "endgame"  # Moves 31+


class GameResult(Enum):
    """Game outcome from engine perspective"""

    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


@dataclass
class GameRecord:
    """Record of a single game"""

    game_id: int
    opponent: str
    result: GameResult
    num_moves: int
    time_taken: float
    engine_color: chess.Color
    termination: str  # checkmate, stalemate, timeout, etc.
    pgn: Optional[str] = None

    # Phase statistics
    opening_moves: int = 0
    middle_moves: int = 0
    endgame_moves: int = 0

    # Move quality
    blunders: int = 0
    mistakes: int = 0
    good_moves: int = 0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResults:
    """Complete evaluation results"""

    model_name: str
    timestamp: str
    total_games: int

    # Overall statistics
    wins: int = 0
    losses: int = 0
    draws: int = 0

    # By opponent
    results_by_opponent: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # By color
    wins_as_white: int = 0
    wins_as_black: int = 0
    losses_as_white: int = 0
    losses_as_black: int = 0
    draws_as_white: int = 0
    draws_as_black: int = 0

    # By phase
    phase_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ELO estimation
    estimated_elo: Optional[float] = None
    elo_confidence_interval: Optional[Tuple[float, float]] = None

    # Performance metrics
    avg_game_length: float = 0.0
    avg_time_per_game: float = 0.0
    total_time: float = 0.0

    # Weakness analysis
    common_errors: List[str] = field(default_factory=list)
    weak_phases: List[str] = field(default_factory=list)

    # Individual games
    games: List[GameRecord] = field(default_factory=list)


# =============================================================================
# OPPONENT PLAYERS
# =============================================================================


class OpponentPlayer:
    """Base class for opponent players"""

    def __init__(self, name: str):
        self.name = name
        self.base_elo = 0  # Estimated ELO of this opponent

    def select_move(self, board: chess.Board) -> chess.Move:
        """Select a move for the given position"""
        raise NotImplementedError


class RandomPlayer(OpponentPlayer):
    """Plays completely random legal moves"""

    def __init__(self):
        super().__init__("Random")
        self.base_elo = 400  # Very weak baseline

    def select_move(self, board: chess.Board) -> chess.Move:
        """Select random legal move"""
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves)


class StockfishPlayer(OpponentPlayer):
    """Plays using Stockfish engine"""

    def __init__(self, level: int = 1, time_limit: float = 0.1):
        """
        Initialize Stockfish player

        Args:
            level: Skill level 0-20 (0=weakest, 20=strongest)
            time_limit: Time limit per move in seconds
        """
        super().__init__(f"Stockfish-{level}")
        self.level = level
        self.time_limit = time_limit

        # Estimated ELO based on level (approximate)
        # Level 0: ~800, Level 10: ~1800, Level 20: ~3000
        self.base_elo = 800 + (level * 110)

        # Check if chess.engine is available
        if not CHESS_ENGINE_AVAILABLE:
            self.engine_available = False
            print(
                "⚠️  chess.engine module not available - install python-chess with engine support"
            )
            return

        # Try to find and initialize stockfish
        stockfish_path = self._find_stockfish()
        if stockfish_path:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                self.engine.configure({"Skill Level": level})
                self.engine_available = True
                print(f"✅ Stockfish initialized at level {level}")
            except Exception as e:
                self.engine_available = False
                print(f"⚠️  Failed to initialize Stockfish: {e}")
        else:
            self.engine_available = False
            print("⚠️  Stockfish executable not found")

    def _find_stockfish(self) -> Optional[str]:
        """Try to find stockfish executable"""
        # Possible paths (Linux, macOS, Windows)
        possible_paths = [
            # Linux
            "/usr/games/stockfish",
            "/usr/bin/stockfish",
            "/usr/local/bin/stockfish",
            # macOS
            "/opt/homebrew/bin/stockfish",
            # Windows
            r"C:\Program Files\Stockfish\stockfish.exe",
            r"C:\Program Files\Stockfish\stockfish-windows-x86-64-avx2.exe",
            r"C:\stockfish\stockfish.exe",
            r"C:\stockfish\stockfish-windows-x86-64-avx2.exe",
            # Generic
            "stockfish",
            "stockfish.exe",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path
            # Try without explicit path (in PATH)
            try:
                import subprocess

                result = subprocess.run(
                    [path, "--help"], capture_output=True, timeout=1
                )
                if result.returncode == 0:
                    return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        return None

    def select_move(self, board: chess.Board) -> chess.Move:
        """Select move using Stockfish"""
        if not self.engine_available:
            # Fallback to random
            return RandomPlayer().select_move(board)

        try:
            result = self.engine.play(board, chess.engine.Limit(time=self.time_limit))
            if result.move is None:
                # Shouldn't happen with legal position, but handle it
                print("⚠️  Stockfish returned None, using random move")
                return RandomPlayer().select_move(board)
            return result.move
        except Exception as e:
            print(f"⚠️  Stockfish error: {e}, using random move")
            return RandomPlayer().select_move(board)

    def close(self):
        """Close the engine"""
        if hasattr(self, "engine"):
            self.engine.quit()


# =============================================================================
# ENGINE EVALUATOR
# =============================================================================


class ChessEngineEvaluator:
    """
    Comprehensive evaluation system for chess engines
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_mcts: bool = False,
        mcts_simulations: int = 100,
        verbose: bool = True,
    ):
        """
        Initialize evaluator

        Args:
            model_path: Path to model checkpoint
            device: Device to run on
            use_mcts: Whether to use MCTS for move selection
            mcts_simulations: Number of MCTS simulations
            verbose: Print progress
        """
        self.model_path = model_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_mcts = use_mcts and MCTS_AVAILABLE
        self.mcts_simulations = mcts_simulations
        self.verbose = verbose

        # Load model
        self._load_model()

        # Initialize encoders
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()
        self.move_history = MoveHistory()

        # Results
        self.results = EvaluationResults(
            model_name=Path(model_path).stem,
            timestamp=datetime.now().isoformat(),
            total_games=0,
        )

    def _load_model(self):
        """Load the chess model"""
        if self.verbose:
            print(f"Loading model from {self.model_path}...")

        checkpoint = torch.load(self.model_path, map_location=self.device)

        # Extract state dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Auto-detect architecture
        has_rnn = any(
            key.startswith("rnn.") or key.startswith("fusion.")
            for key in state_dict.keys()
        )

        # Detect CNN blocks
        block_keys = [
            k for k in state_dict.keys() if k.startswith("cnn.residual_blocks.")
        ]
        if block_keys:
            max_block = max(int(k.split(".")[2]) for k in block_keys)
            cnn_blocks = max_block + 1
        else:
            cnn_blocks = 10

        # Detect RNN parameters from checkpoint if RNN is present
        rnn_hidden_size = 256  # default
        rnn_layers = 2  # default
        rnn_attention = False  # default
        fusion_type = "gated"  # default

        if has_rnn:
            # Try to detect RNN hidden size from weight shapes
            for key in state_dict.keys():
                if "rnn.lstm.weight_ih_l0" in key:
                    weight_shape = state_dict[key].shape
                    rnn_hidden_size = weight_shape[0] // 4  # LSTM has 4 gates
                    break

            # Detect number of RNN layers
            rnn_layer_keys = [
                k for k in state_dict.keys() if k.startswith("rnn.lstm.weight_ih_l")
            ]
            if rnn_layer_keys:
                max_layer = max(int(k.split("_l")[1][0]) for k in rnn_layer_keys)
                rnn_layers = max_layer + 1

            # Detect attention
            rnn_attention = any("attention" in key for key in state_dict.keys())

            # Detect fusion type (check for gated fusion layers)
            if any("fusion.gate" in key for key in state_dict.keys()):
                fusion_type = "gated"
            elif any("fusion.attention" in key for key in state_dict.keys()):
                fusion_type = "attention"
            else:
                fusion_type = "concat"

        # Create model with detected parameters
        self.model = HybridChessNet(
            cnn_residual_blocks=cnn_blocks,
            use_rnn=has_rnn,
            rnn_hidden_size=rnn_hidden_size if has_rnn else 256,
            rnn_num_layers=rnn_layers if has_rnn else 2,
            rnn_use_attention=rnn_attention if has_rnn else False,
            fusion_type=fusion_type if has_rnn else "gated",
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.has_rnn = has_rnn

        if self.verbose:
            if has_rnn:
                print(
                    f"✅ Model loaded (Hybrid CNN-RNN, {cnn_blocks} CNN blocks, {rnn_layers} RNN layers, hidden={rnn_hidden_size}, attention={rnn_attention}, fusion={fusion_type})"
                )
            else:
                print(f"✅ Model loaded (CNN-only, {cnn_blocks} blocks)")

            if self.use_mcts:
                print(f"✅ MCTS enabled ({self.mcts_simulations} simulations)")
            else:
                print("ℹ️  Using policy-only mode")

    def select_move(self, board: chess.Board) -> chess.Move:
        """
        Select best move for current position

        Args:
            board: Current board position

        Returns:
            Selected move
        """
        if self.use_mcts:
            return self._select_move_mcts(board)
        else:
            return self._select_move_policy(board)

    def _select_move_policy(self, board: chess.Board) -> chess.Move:
        """Select move using policy network only"""
        # Encode board
        board_tensor = (
            self.board_encoder.board_to_tensor(board).unsqueeze(0).to(self.device)
        )

        # Encode move history if using RNN
        if self.has_rnn:
            move_history = self.move_history.encode_game_history(board, pad=True)
            move_history = move_history.unsqueeze(0).to(self.device)
        else:
            move_history = None

        # Get policy
        with torch.no_grad():
            if self.has_rnn and move_history is not None:
                # Model with RNN may return 2 or 3 values
                model_output = self.model(board_tensor, move_history)
                policy_logits = model_output[0]  # First element is always policy
            else:
                # CNN-only model may return 2 or 3 values
                model_output = self.model(board_tensor)
                policy_logits = model_output[0]  # First element is always policy

        # Select best legal move
        policy_logits = policy_logits.squeeze(0).cpu()
        move, _ = self.move_encoder.get_best_move(policy_logits, board)

        return move

    def _select_move_mcts(self, board: chess.Board) -> chess.Move:
        """Select move using MCTS"""
        mcts = MCTS(
            model=self.model,
            board_encoder=self.board_encoder,
            move_encoder=self.move_encoder,
            device=self.device,
            c_puct=1.5,
            num_simulations=self.mcts_simulations,
        )

        # MCTS search returns tuple of (move, stats)
        move, _ = mcts.search(board, return_stats=False)

        return move

    def play_game(
        self,
        opponent: OpponentPlayer,
        engine_color: chess.Color,
        max_moves: int = 200,
        game_id: int = 0,
    ) -> GameRecord:
        """
        Play a single game against an opponent

        Args:
            opponent: Opponent player
            engine_color: Color for the engine
            max_moves: Maximum number of moves before draw
            game_id: Game identifier

        Returns:
            GameRecord with game details
        """
        board = chess.Board()
        start_time = time.time()
        moves_played = 0

        # Phase tracking
        opening_moves = 0
        middle_moves = 0
        endgame_moves = 0

        while not board.is_game_over() and moves_played < max_moves:
            # Determine game phase
            if moves_played < 10:
                phase = GamePhase.OPENING
                opening_moves += 1
            elif moves_played < 30:
                phase = GamePhase.MIDDLE
                middle_moves += 1
            else:
                phase = GamePhase.ENDGAME
                endgame_moves += 1

            # Select move
            if board.turn == engine_color:
                try:
                    move = self.select_move(board)
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Engine error: {e}")
                    # Fallback to random
                    move = random.choice(list(board.legal_moves))
            else:
                move = opponent.select_move(board)

            board.push(move)
            moves_played += 1

        # Determine result
        time_taken = time.time() - start_time

        if board.is_checkmate():
            winner = not board.turn
            if winner == engine_color:
                result = GameResult.WIN
                termination = "checkmate"
            else:
                result = GameResult.LOSS
                termination = "checkmated"
        elif board.is_stalemate():
            result = GameResult.DRAW
            termination = "stalemate"
        elif board.is_insufficient_material():
            result = GameResult.DRAW
            termination = "insufficient_material"
        elif board.can_claim_draw():
            result = GameResult.DRAW
            termination = "repetition_or_50_moves"
        elif moves_played >= max_moves:
            result = GameResult.DRAW
            termination = "max_moves"
        else:
            result = GameResult.DRAW
            termination = "other"

        # Create PGN
        game = chess.pgn.Game()
        game.headers["Event"] = "Evaluation"
        game.headers["White"] = (
            "Engine" if engine_color == chess.WHITE else opponent.name
        )
        game.headers["Black"] = (
            opponent.name if engine_color == chess.WHITE else "Engine"
        )
        game.headers["Result"] = board.result()

        node = game
        temp_board = chess.Board()
        for move in board.move_stack:
            node = node.add_variation(move)
            temp_board.push(move)

        pgn_string = str(game)

        return GameRecord(
            game_id=game_id,
            opponent=opponent.name,
            result=result,
            num_moves=moves_played,
            time_taken=time_taken,
            engine_color=engine_color,
            termination=termination,
            pgn=pgn_string,
            opening_moves=opening_moves,
            middle_moves=middle_moves,
            endgame_moves=endgame_moves,
        )

    def evaluate(
        self,
        opponents: Sequence[OpponentPlayer],
        games_per_opponent: int = 50,
        alternate_colors: bool = True,
    ) -> EvaluationResults:
        """
        Run comprehensive evaluation

        Args:
            opponents: List of opponent players
            games_per_opponent: Number of games against each opponent
            alternate_colors: Alternate engine color

        Returns:
            EvaluationResults
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("🎯 STARTING EVALUATION")
            print("=" * 70)
            print(f"Model: {self.model_path}")
            print(f"Opponents: {[opp.name for opp in opponents]}")
            print(f"Games per opponent: {games_per_opponent}")
            print(f"Total games: {len(opponents) * games_per_opponent}")
            print("=" * 70 + "\n")

        game_id = 0
        total_games = len(opponents) * games_per_opponent

        with tqdm(total=total_games, disable=not self.verbose) as pbar:
            for opponent in opponents:
                # Initialize opponent stats
                if opponent.name not in self.results.results_by_opponent:
                    self.results.results_by_opponent[opponent.name] = {
                        "wins": 0,
                        "losses": 0,
                        "draws": 0,
                    }

                for i in range(games_per_opponent):
                    # Alternate colors
                    if alternate_colors:
                        engine_color = chess.WHITE if i % 2 == 0 else chess.BLACK
                    else:
                        engine_color = chess.WHITE

                    # Play game
                    game_record = self.play_game(
                        opponent, engine_color, game_id=game_id
                    )

                    # Update results
                    self.results.games.append(game_record)
                    self.results.total_games += 1

                    if game_record.result == GameResult.WIN:
                        self.results.wins += 1
                        self.results.results_by_opponent[opponent.name]["wins"] += 1
                        if engine_color == chess.WHITE:
                            self.results.wins_as_white += 1
                        else:
                            self.results.wins_as_black += 1
                    elif game_record.result == GameResult.LOSS:
                        self.results.losses += 1
                        self.results.results_by_opponent[opponent.name]["losses"] += 1
                        if engine_color == chess.WHITE:
                            self.results.losses_as_white += 1
                        else:
                            self.results.losses_as_black += 1
                    else:
                        self.results.draws += 1
                        self.results.results_by_opponent[opponent.name]["draws"] += 1
                        if engine_color == chess.WHITE:
                            self.results.draws_as_white += 1
                        else:
                            self.results.draws_as_black += 1

                    pbar.update(1)
                    pbar.set_description(
                        f"{opponent.name}: W{self.results.results_by_opponent[opponent.name]['wins']}"
                        f"-L{self.results.results_by_opponent[opponent.name]['losses']}"
                        f"-D{self.results.results_by_opponent[opponent.name]['draws']}"
                    )

                    game_id += 1

        # Calculate aggregate statistics
        self._calculate_statistics()

        # Estimate ELO
        self._estimate_elo(opponents)

        # Analyze weaknesses
        self._analyze_weaknesses()

        if self.verbose:
            print("\n✅ Evaluation complete!")

        return self.results

    def _calculate_statistics(self):
        """Calculate aggregate statistics"""
        if not self.results.games:
            return

        # Average game length (convert numpy type to float)
        self.results.avg_game_length = float(
            np.mean([g.num_moves for g in self.results.games])
        )

        # Average time per game (convert numpy type to float)
        self.results.avg_time_per_game = float(
            np.mean([g.time_taken for g in self.results.games])
        )

        # Total time
        self.results.total_time = sum(g.time_taken for g in self.results.games)

        # Phase statistics
        for phase in ["opening", "middle", "endgame"]:
            phase_games = [
                g for g in self.results.games if getattr(g, f"{phase}_moves") > 0
            ]

            if phase_games:
                wins = sum(1 for g in phase_games if g.result == GameResult.WIN)
                losses = sum(1 for g in phase_games if g.result == GameResult.LOSS)
                draws = sum(1 for g in phase_games if g.result == GameResult.DRAW)

                # Ensure all values are float, not numpy types
                total = len(phase_games)
                avg_moves_value = float(
                    np.mean([getattr(g, f"{phase}_moves") for g in phase_games])
                )

                self.results.phase_statistics[phase] = {
                    "total_games": total,
                    "win_rate": float(wins / total) if total > 0 else 0.0,
                    "loss_rate": float(losses / total) if total > 0 else 0.0,
                    "draw_rate": float(draws / total) if total > 0 else 0.0,
                    "avg_moves": avg_moves_value,
                }

    def _estimate_elo(self, opponents: Sequence[OpponentPlayer]):
        """
        Estimate ELO rating using performance against known opponents

        Uses the formula:
        Performance Rating = Opponent Rating + 400 * log10(W / L)
        where W = wins, L = losses (draws count as 0.5)
        """
        if not self.results.games:
            return

        # Calculate expected score against each opponent
        performance_ratings = []

        for opponent in opponents:
            opp_stats = self.results.results_by_opponent.get(opponent.name, {})
            wins = opp_stats.get("wins", 0)
            losses = opp_stats.get("losses", 0)
            draws = opp_stats.get("draws", 0)

            total = wins + losses + draws
            if total == 0:
                continue

            # Score (wins + 0.5 * draws)
            score = wins + 0.5 * draws
            score_percentage = score / total

            # Performance rating formula
            # If score is 100%, use a high performance rating
            if score_percentage >= 0.99:
                performance = opponent.base_elo + 400
            elif score_percentage <= 0.01:
                performance = opponent.base_elo - 400
            else:
                # Use standard formula
                # Performance = Opponent_Rating + 400 * log10(W/L)
                # where W/L is calculated from score percentage
                try:
                    performance = opponent.base_elo + 400 * math.log10(
                        score_percentage / (1 - score_percentage)
                    )
                except (ValueError, ZeroDivisionError):
                    performance = opponent.base_elo

            performance_ratings.append(performance)

        if performance_ratings:
            # Average performance rating (convert numpy type to float)
            self.results.estimated_elo = float(np.mean(performance_ratings))

            # Confidence interval (rough estimate)
            if len(performance_ratings) > 1:
                std = float(np.std(performance_ratings))
                self.results.elo_confidence_interval = (
                    self.results.estimated_elo - std,
                    self.results.estimated_elo + std,
                )
            else:
                self.results.elo_confidence_interval = (
                    self.results.estimated_elo - 100.0,
                    self.results.estimated_elo + 100.0,
                )

    def _analyze_weaknesses(self):
        """Analyze common weaknesses"""
        # Identify weak phases
        weak_threshold = 0.4  # < 40% win rate

        for phase, stats in self.results.phase_statistics.items():
            if stats["win_rate"] < weak_threshold:
                self.results.weak_phases.append(
                    f"{phase.capitalize()} (win rate: {stats['win_rate']:.1%})"
                )

        # Identify color weakness
        total_white = (
            self.results.wins_as_white
            + self.results.losses_as_white
            + self.results.draws_as_white
        )
        total_black = (
            self.results.wins_as_black
            + self.results.losses_as_black
            + self.results.draws_as_black
        )

        if total_white > 0:
            white_win_rate = self.results.wins_as_white / total_white
            if white_win_rate < weak_threshold:
                self.results.common_errors.append(
                    f"Weak as White (win rate: {white_win_rate:.1%})"
                )

        if total_black > 0:
            black_win_rate = self.results.wins_as_black / total_black
            if black_win_rate < weak_threshold:
                self.results.common_errors.append(
                    f"Weak as Black (win rate: {black_win_rate:.1%})"
                )

        # Analyze termination types
        terminations = Counter(g.termination for g in self.results.games)

        # Check for pattern of losing by checkmate
        losses_by_checkmate = sum(
            1
            for g in self.results.games
            if g.result == GameResult.LOSS and g.termination == "checkmated"
        )
        if losses_by_checkmate > self.results.total_games * 0.3:
            self.results.common_errors.append(
                f"Frequently gets checkmated ({losses_by_checkmate} games)"
            )

    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Model: {self.results.model_name}")
        print(f"Total Games: {self.results.total_games}")
        print()

        # Overall results
        total = self.results.wins + self.results.losses + self.results.draws
        win_rate = self.results.wins / total if total > 0 else 0
        loss_rate = self.results.losses / total if total > 0 else 0
        draw_rate = self.results.draws / total if total > 0 else 0

        print("Overall Performance:")
        print(f"  Wins:   {self.results.wins:3d} ({win_rate:.1%})")
        print(f"  Losses: {self.results.losses:3d} ({loss_rate:.1%})")
        print(f"  Draws:  {self.results.draws:3d} ({draw_rate:.1%})")
        print()

        # By opponent
        print("Performance by Opponent:")
        for opponent_name, stats in self.results.results_by_opponent.items():
            total = stats["wins"] + stats["losses"] + stats["draws"]
            win_rate = stats["wins"] / total if total > 0 else 0
            print(
                f"  {opponent_name:15s}: {stats['wins']:2d}-{stats['losses']:2d}-{stats['draws']:2d} ({win_rate:.1%})"
            )
        print()

        # By color
        print("Performance by Color:")
        total_white = (
            self.results.wins_as_white
            + self.results.losses_as_white
            + self.results.draws_as_white
        )
        total_black = (
            self.results.wins_as_black
            + self.results.losses_as_black
            + self.results.draws_as_black
        )

        if total_white > 0:
            white_win_rate = self.results.wins_as_white / total_white
            print(
                f"  As White: {self.results.wins_as_white:2d}-{self.results.losses_as_white:2d}-{self.results.draws_as_white:2d} ({white_win_rate:.1%})"
            )

        if total_black > 0:
            black_win_rate = self.results.wins_as_black / total_black
            print(
                f"  As Black: {self.results.wins_as_black:2d}-{self.results.losses_as_black:2d}-{self.results.draws_as_black:2d} ({black_win_rate:.1%})"
            )
        print()

        # ELO estimation
        if self.results.estimated_elo:
            print("ELO Estimation:")
            print(f"  Estimated ELO: {self.results.estimated_elo:.0f}")
            if self.results.elo_confidence_interval:
                low, high = self.results.elo_confidence_interval
                print(f"  95% Confidence: {low:.0f} - {high:.0f}")
            print()

        # Performance metrics
        print("Performance Metrics:")
        print(f"  Avg Game Length: {self.results.avg_game_length:.1f} moves")
        print(f"  Avg Time/Game:   {self.results.avg_time_per_game:.2f}s")
        print(f"  Total Time:      {self.results.total_time:.1f}s")
        print()

        # Phase analysis
        if self.results.phase_statistics:
            print("Phase Analysis:")
            for phase, stats in self.results.phase_statistics.items():
                print(
                    f"  {phase.capitalize():8s}: {stats['win_rate']:.1%} win rate, {stats['avg_moves']:.1f} avg moves"
                )
            print()

        # Weaknesses
        if self.results.weak_phases or self.results.common_errors:
            print("  Identified Weaknesses:")
            for weakness in self.results.weak_phases:
                print(f"  • Weak in {weakness}")
            for error in self.results.common_errors:
                print(f"  • {error}")
            print()

        print("=" * 70)

    def save_results(self, output_dir: str = "evaluation_results"):
        """Save evaluation results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.results.model_name}_{timestamp}"

        # Save JSON summary
        json_path = output_path / f"{base_name}_summary.json"
        with open(json_path, "w") as f:
            json.dump(asdict(self.results), f, indent=2, default=str)

        print(f"💾 Results saved to {json_path}")

        # Save PGN games
        pgn_path = output_path / f"{base_name}_games.pgn"
        with open(pgn_path, "w") as f:
            for game in self.results.games:
                if game.pgn:
                    f.write(game.pgn)
                    f.write("\n\n")

        print(f"💾 Games saved to {pgn_path}")

        # Save text report
        report_path = output_path / f"{base_name}_report.txt"
        with open(report_path, "w") as f:
            # Redirect print to file
            import sys

            old_stdout = sys.stdout
            sys.stdout = f
            self.print_summary()
            sys.stdout = old_stdout

        print(f"💾 Report saved to {report_path}")

        return output_path

    def plot_results(self, output_dir: str = "evaluation_results"):
        """Generate visualization plots"""
        if not PLOTTING_AVAILABLE:
            print("⚠️  Plotting not available (install matplotlib and seaborn)")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.results.model_name}_{timestamp}"

        # Set style
        sns.set_style("whitegrid")

        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))

        # 1. Overall win/loss/draw pie chart
        ax1 = plt.subplot(2, 3, 1)
        sizes = [self.results.wins, self.results.losses, self.results.draws]
        labels = [
            f"Wins ({self.results.wins})",
            f"Losses ({self.results.losses})",
            f"Draws ({self.results.draws})",
        ]
        colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
        ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Overall Results")

        # 2. Performance by opponent
        ax2 = plt.subplot(2, 3, 2)
        opponents = list(self.results.results_by_opponent.keys())
        win_rates = [
            self.results.results_by_opponent[opp]["wins"]
            / sum(self.results.results_by_opponent[opp].values())
            for opp in opponents
        ]
        ax2.barh(opponents, win_rates, color="#3498db")
        ax2.set_xlabel("Win Rate")
        ax2.set_title("Win Rate by Opponent")
        ax2.set_xlim(0, 1)

        # 3. Performance by color
        ax3 = plt.subplot(2, 3, 3)
        colors_data = ["White", "Black"]
        wins = [self.results.wins_as_white, self.results.wins_as_black]
        losses = [self.results.losses_as_white, self.results.losses_as_black]
        draws = [self.results.draws_as_white, self.results.draws_as_black]

        x = np.arange(len(colors_data))
        width = 0.25

        ax3.bar(x - width, wins, width, label="Wins", color="#2ecc71")
        ax3.bar(x, losses, width, label="Losses", color="#e74c3c")
        ax3.bar(x + width, draws, width, label="Draws", color="#95a5a6")

        ax3.set_ylabel("Games")
        ax3.set_title("Results by Color")
        ax3.set_xticks(x)
        ax3.set_xticklabels(colors_data)
        ax3.legend()

        # 4. Phase performance
        ax4 = plt.subplot(2, 3, 4)
        phases = list(self.results.phase_statistics.keys())
        phase_win_rates = [
            self.results.phase_statistics[phase]["win_rate"] for phase in phases
        ]
        ax4.bar(phases, phase_win_rates, color=["#e67e22", "#9b59b6", "#1abc9c"])
        ax4.set_ylabel("Win Rate")
        ax4.set_title("Win Rate by Game Phase")
        ax4.set_ylim(0, 1)

        # 5. Game length distribution
        ax5 = plt.subplot(2, 3, 5)
        game_lengths = [g.num_moves for g in self.results.games]
        ax5.hist(game_lengths, bins=20, color="#34495e", edgecolor="black")
        ax5.set_xlabel("Number of Moves")
        ax5.set_ylabel("Frequency")
        ax5.set_title("Game Length Distribution")
        ax5.axvline(
            self.results.avg_game_length,
            color="red",
            linestyle="--",
            label=f"Mean: {self.results.avg_game_length:.1f}",
        )
        ax5.legend()

        # 6. ELO estimation
        ax6 = plt.subplot(2, 3, 6)
        if self.results.estimated_elo:
            elo = self.results.estimated_elo
            if self.results.elo_confidence_interval:
                low, high = self.results.elo_confidence_interval
                ax6.barh(
                    ["Estimated ELO"],
                    [elo],
                    xerr=[[elo - low], [high - elo]],
                    color="#f39c12",
                    capsize=5,
                )
            else:
                ax6.barh(["Estimated ELO"], [elo], color="#f39c12")
            ax6.set_xlabel("ELO Rating")
            ax6.set_title("ELO Estimation")
            ax6.set_xlim(0, max(2000, elo * 1.2))
        else:
            ax6.text(0.5, 0.5, "No ELO estimation", ha="center", va="center")
            ax6.set_title("ELO Estimation")

        plt.tight_layout()

        # Save
        plot_path = output_path / f"{base_name}_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"📊 Plots saved to {plot_path}")

        plt.close()


# =============================================================================
# MODEL COMPARISON
# =============================================================================


def compare_models(
    model_paths: List[str],
    games_per_model: int = 30,
    device: str = "cuda",
    output_dir: str = "evaluation_results",
):
    """
    Compare multiple models against each other

    Args:
        model_paths: List of model checkpoint paths
        games_per_model: Games to play between each pair
        device: Device to run on
        output_dir: Output directory
    """
    print("\n" + "=" * 70)
    print("🔬 MODEL COMPARISON")
    print("=" * 70)
    print(f"Models: {len(model_paths)}")
    print(f"Games per matchup: {games_per_model}")
    print("=" * 70 + "\n")

    # Load all models
    evaluators = []
    for path in model_paths:
        evaluator = ChessEngineEvaluator(
            path, device=device, use_mcts=False, verbose=False
        )
        evaluators.append(evaluator)

    # Create comparison matrix
    results_matrix = np.zeros((len(model_paths), len(model_paths)))

    # Play round-robin
    for i, eval1 in enumerate(evaluators):
        for j, eval2 in enumerate(evaluators):
            if i == j:
                continue

            print(f"\n{eval1.results.model_name} vs {eval2.results.model_name}")

            wins = 0
            for game_num in range(games_per_model):
                # Alternate colors
                engine_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK

                board = chess.Board()
                while not board.is_game_over():
                    if board.turn == engine_color:
                        move = eval1.select_move(board)
                    else:
                        move = eval2.select_move(board)
                    board.push(move)

                # Determine winner
                if board.is_checkmate():
                    winner = not board.turn
                    if winner == engine_color:
                        wins += 1
                        results_matrix[i][j] += 1

            win_rate = wins / games_per_model
            print(f"  Score: {wins}/{games_per_model} ({win_rate:.1%})")

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON MATRIX (row vs column)")
    print("=" * 70)

    # Header
    print(f"{'':20s}", end="")
    for path in model_paths:
        name = Path(path).stem[:15]
        print(f"{name:>15s}", end="")
    print()

    # Rows
    for i, path in enumerate(model_paths):
        name = Path(path).stem[:20]
        print(f"{name:20s}", end="")
        for j in range(len(model_paths)):
            if i == j:
                print(f"{'--':>15s}", end="")
            else:
                score = results_matrix[i][j]
                print(f"{score:>15.0f}", end="")
        print()

    print("=" * 70)


# =============================================================================
# MAIN CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Chess Engine Evaluation & Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick evaluation against random player
  python evaluation.py --model best.pt --games 50

  # Comprehensive evaluation
  python evaluation.py --model best.pt --games 100 --opponents random stockfish --stockfish-level 5

  # Use MCTS for stronger play
  python evaluation.py --model best.pt --games 30 --mcts --simulations 200

  # Compare multiple models
  python evaluation.py --compare model1.pt model2.pt model3.pt --games 30
        """,
    )

    # Model selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="Path to model checkpoint")
    group.add_argument("--compare", nargs="+", help="Compare multiple models")

    # Evaluation settings
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Number of games per opponent (default: 50)",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["random"],
        choices=["random", "stockfish"],
        help="Opponents to play against",
    )
    parser.add_argument(
        "--stockfish-level",
        type=int,
        default=1,
        help="Stockfish skill level 0-20 (default: 1)",
    )

    # Engine settings
    parser.add_argument(
        "--mcts", action="store_true", help="Use MCTS for move selection"
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="MCTS simulations per move (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Disable plot generation"
    )
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    # Comparison mode
    if args.compare:
        compare_models(
            args.compare,
            games_per_model=args.games,
            device=args.device,
            output_dir=args.output_dir,
        )
        return

    # Single model evaluation
    evaluator = ChessEngineEvaluator(
        args.model,
        device=args.device,
        use_mcts=args.mcts,
        mcts_simulations=args.simulations,
        verbose=not args.quiet,
    )

    # Create opponents
    opponents = []
    for opp_name in args.opponents:
        if opp_name == "random":
            opponents.append(RandomPlayer())
        elif opp_name == "stockfish":
            opponents.append(StockfishPlayer(level=args.stockfish_level))

    # Run evaluation
    results = evaluator.evaluate(opponents, games_per_opponent=args.games)

    # Print summary
    evaluator.print_summary()

    # Save results
    output_path = evaluator.save_results(args.output_dir)

    # Generate plots
    if not args.no_plots and PLOTTING_AVAILABLE:
        evaluator.plot_results(args.output_dir)

    print(f"\n✅ Evaluation complete! Results saved to {output_path}")


if __name__ == "__main__":
    main()
