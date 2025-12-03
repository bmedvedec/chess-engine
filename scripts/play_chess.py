"""
UNIFIED CHESS ENGINE

Features:
- Policy mode (fast, with sophisticated play styles)
- MCTS mode (strong, with time control)
- Auto-detection (RNN, architecture)
- Comprehensive UI/UX
- Integrated play_styles.py (4 styles, temperature progression, position evaluation)
- Integrated time_control.py (4 strategies, complexity analysis)
"""

import os
import sys
from typing import Optional, Tuple, Dict, List, Any
import argparse
import time
from dataclasses import dataclass, field
from collections import deque
from statistics import mean, stdev
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod

import torch
import chess
import chess.pgn

from chess_engine.models.hybrid_model import HybridChessNet
from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder, MoveHistory

# Try to import advanced modules (fallback if not available)
try:
    from chess_engine.engine.play_styles import PlayStyle, StyleType, get_style_names

    ADVANCED_STYLES_AVAILABLE = True
    print("✨ Advanced play styles module loaded")
except ImportError:
    ADVANCED_STYLES_AVAILABLE = False
    print(
        "ℹ️  Using basic play styles (place play_styles.py in chess_engine/engine/ for advanced features)"
    )

try:
    from chess_engine.engine.time_control import (
        TimeControl,
        TimeControlConfig,
        PositionComplexity,
    )

    TIME_CONTROL_AVAILABLE = True
except ImportError:
    TIME_CONTROL_AVAILABLE = False
    print("⚠️  Time control module not available")


# =============================================================================
# CONFIGURATION & DATA CLASSES
# =============================================================================


@dataclass
class EngineConfig:
    """Base configuration for chess engine"""

    cache_size: int = 1000
    show_thinking: bool = True
    show_performance_warnings: bool = True
    top_moves_count: int = 5
    performance_threshold_ms: float = 100.0


@dataclass
class PolicyConfig(EngineConfig):
    """Configuration for policy-based engine"""

    play_style: str = (
        "balanced"  # aggressive, balanced, defensive, positional (if advanced)
    )
    time_per_move: float = 5.0
    enforce_time_limit: bool = False


@dataclass
class MCTSConfig(EngineConfig):
    """Configuration for MCTS engine"""

    num_simulations: int = 100
    c_puct: float = 1.5
    use_batched: bool = False
    batch_size: int = 8
    enable_caching: bool = True
    progressive_widening: bool = False
    early_termination: bool = True
    play_style: str = "balanced"  # Also affects MCTS if advanced styles available
    # Time control
    time_control_enabled: bool = False
    total_time: float = 300.0
    increment: float = 0.0
    time_strategy: str = "adaptive"


@dataclass
class MoveInfo:
    """Information about a move"""

    move: chess.Move
    confidence: float
    evaluation: float
    time_taken: float
    timed_out: bool = False
    extra_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameStatistics:
    """Track game statistics"""

    move_times: deque = field(default_factory=lambda: deque(maxlen=100))
    evaluations: List[float] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0

    def add_move(self, move_info: MoveInfo, from_cache: bool = False):
        self.move_times.append(move_info.time_taken)
        self.evaluations.append(move_info.evaluation)
        if from_cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def get_summary(self) -> Dict[str, float]:
        times_ms = [t * 1000 for t in self.move_times]
        return {
            "avg_move_time_ms": mean(times_ms) if times_ms else 0,
            "std_move_time_ms": stdev(times_ms) if len(times_ms) > 1 else 0,
            "max_move_time_ms": max(times_ms) if times_ms else 0,
            "min_move_time_ms": min(times_ms) if times_ms else 0,
            "total_moves": len(self.move_times),
            "avg_evaluation": mean(self.evaluations) if self.evaluations else 0,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else 0
            ),
        }


# =============================================================================
# MODEL LOADING WITH AUTO-DETECTION
# =============================================================================


def load_model_safe(
    model_path: str,
    device: torch.device,
    use_rnn: Optional[bool] = None,
    cnn_blocks: Optional[int] = None,
) -> Tuple[HybridChessNet, bool, int]:
    """
    Load model with automatic architecture detection.

    Args:
        model_path: Path to checkpoint
        device: Device to load on
        use_rnn: Force RNN usage (None = auto-detect)
        cnn_blocks: Number of CNN blocks (None = auto-detect)

    Returns:
        Tuple of (model, actual_use_rnn, actual_cnn_blocks)
    """
    checkpoint = torch.load(model_path, map_location=device)

    # Extract state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        epoch = checkpoint.get("epoch", None)
    else:
        state_dict = checkpoint
        epoch = None

    # Auto-detect RNN usage
    has_rnn = any(
        key.startswith("rnn.") or key.startswith("fusion.") for key in state_dict.keys()
    )

    if use_rnn is None:
        use_rnn = has_rnn
        if has_rnn:
            print("ℹ️  Auto-detected: Model has RNN")
    elif use_rnn and not has_rnn:
        print("⚠️  Warning: --use-rnn specified but model has no RNN weights")
        print("    Loading as CNN-only model")
        use_rnn = False
    elif not use_rnn and has_rnn:
        print("ℹ️  Note: Model has RNN weights, enabling RNN support")
        use_rnn = True

    # Auto-detect CNN blocks
    if cnn_blocks is None:
        block_keys = [
            k for k in state_dict.keys() if k.startswith("cnn.residual_blocks.")
        ]
        if block_keys:
            max_block = max(int(k.split(".")[2]) for k in block_keys)
            cnn_blocks = max_block + 1
            print(f"ℹ️  Auto-detected: {cnn_blocks} CNN residual blocks")
        else:
            cnn_blocks = 10
            print(f"⚠️  Could not detect blocks, using default: {cnn_blocks}")

    # Create model
    model = HybridChessNet(cnn_residual_blocks=cnn_blocks, use_rnn=use_rnn)

    # Load weights
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Print info
    rnn_status = "with RNN" if use_rnn else "CNN-only"
    if epoch is not None:
        print(f"✅ Loaded model from epoch {epoch} ({rnn_status}, {cnn_blocks} blocks)")
    else:
        print(f"✅ Loaded model ({rnn_status}, {cnn_blocks} blocks)")

    return model, use_rnn, cnn_blocks


# =============================================================================
# BASE ENGINE INTERFACE
# =============================================================================


class BaseChessEngine(ABC):
    """Abstract base class for all chess engines"""

    # Fallback basic styles (used if advanced module not available)
    BASIC_STYLES = {
        "aggressive": {
            "temperature": 0.3,
            "value_weight": 0.7,
            "description": "Prefers tactical, attacking moves",
        },
        "balanced": {
            "temperature": 0.1,
            "value_weight": 1.0,
            "description": "Standard play with good balance",
        },
        "defensive": {
            "temperature": 0.05,
            "value_weight": 1.3,
            "description": "Cautious, solid positional play",
        },
    }

    def __init__(
        self,
        model_path: str,
        config: EngineConfig,
        device: str = "cuda",
        use_rnn: Optional[bool] = None,
    ):
        self.config = config
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize encoders
        self.board_encoder = BoardEncoder()
        self.move_encoder = MoveEncoder()

        # Load model with auto-detection
        self.model, self.use_rnn, self.cnn_blocks = load_model_safe(
            model_path, self.device, use_rnn
        )

        # Initialize move history for RNN
        self.move_history = MoveHistory(max_length=50) if self.use_rnn else None

        # Statistics
        self.stats = GameStatistics()

        print(f"🔧 Engine loaded on {self.device}")

    @abstractmethod
    def get_move(self, board: chess.Board, **kwargs) -> MoveInfo:
        """Get best move for position"""
        pass

    @abstractmethod
    def get_top_moves(
        self, board: chess.Board, n: int = 5
    ) -> List[Tuple[chess.Move, float]]:
        """Get top N moves"""
        pass

    def _annotate_move(self, move: chess.Move, board: chess.Board) -> str:
        """Provide natural language annotation"""
        annotations = []

        piece = board.piece_at(move.from_square)
        if piece:
            piece_name = chess.piece_name(piece.piece_type).capitalize()
            annotations.append(piece_name)

        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            if captured:
                annotations.append(f"x{chess.piece_name(captured.piece_type)}")

        if board.is_castling(move):
            if move.to_square > move.from_square:
                annotations.append("O-O")
            else:
                annotations.append("O-O-O")

        board.push(move)
        if board.is_check():
            annotations.append("check")
        if board.is_checkmate():
            annotations.append("mate!")
        board.pop()

        return " ".join(annotations) if annotations else ""

    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return self.stats.get_summary()


# =============================================================================
# POLICY ENGINE (FAST MODE)
# =============================================================================


class PolicyEngine(BaseChessEngine):
    """
    Fast policy-based engine with play styles.
    Uses neural network policy directly (no MCTS).

    Supports both:
    - Advanced styles (if play_styles.py available): 4 styles, temperature progression, position evaluation
    - Basic styles (fallback): 3 simple styles
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[PolicyConfig] = None,
        device: str = "cuda",
        use_rnn: Optional[bool] = None,
    ):
        config = config or PolicyConfig()
        super().__init__(model_path, config, device, use_rnn)

        # Initialize play style (advanced or basic)
        if ADVANCED_STYLES_AVAILABLE:
            self.play_style = PlayStyle.from_string(config.play_style)
            self.using_advanced_styles = True
            print(f"🎭 Play style: {self.play_style} (Advanced)")
        else:
            # Basic fallback
            if config.play_style not in self.BASIC_STYLES:
                print(f"⚠️  Unknown style '{config.play_style}', using 'balanced'")
                config.play_style = "balanced"
            self.style_config = self.BASIC_STYLES[config.play_style]
            self.using_advanced_styles = False
            print(
                f"🎭 Play style: {config.play_style} - {self.style_config['description']} (Basic)"
            )

        self.position_cache: Dict[str, Tuple[chess.Move, float, float]] = {}

        print(f"⚡ Policy mode: Fast inference (<100ms per move)")
        print(f"🎮 Engine ready!")

    def get_move(
        self,
        board: chess.Board,
        time_limit: Optional[float] = None,
        move_number: int = 1,
        **kwargs,
    ) -> MoveInfo:
        """Get move using policy network"""
        config = self.config
        assert isinstance(config, PolicyConfig)

        start_time = time.time()
        timeout = time_limit if time_limit is not None else config.time_per_move

        # Check cache
        fen = board.fen()
        if fen in self.position_cache:
            move, conf, val = self.position_cache[fen]
            elapsed = time.time() - start_time
            move_info = MoveInfo(
                move=move,
                confidence=conf,
                evaluation=val,
                time_taken=elapsed,
                timed_out=False,
            )
            self.stats.add_move(move_info, from_cache=True)
            return move_info

        # Compute move
        move, confidence, evaluation = self._compute_move(board, move_number)
        elapsed = time.time() - start_time

        # Check time
        timed_out = elapsed > timeout
        if timed_out and config.show_performance_warnings:
            print(f"⚠️  Time exceeded: {elapsed:.2f}s > {timeout:.2f}s")

        # Update cache
        if len(self.position_cache) >= config.cache_size:
            self.position_cache.pop(next(iter(self.position_cache)))
        self.position_cache[fen] = (move, confidence, evaluation)

        move_info = MoveInfo(
            move=move,
            confidence=confidence,
            evaluation=evaluation,
            time_taken=elapsed,
            timed_out=timed_out,
        )
        self.stats.add_move(move_info, from_cache=False)

        return move_info

    def _compute_move(
        self, board: chess.Board, move_number: int = 1
    ) -> Tuple[chess.Move, float, float]:
        """Internal computation with style integration"""
        config = self.config
        assert isinstance(config, PolicyConfig)

        # Get temperature (dynamic if advanced styles available)
        if self.using_advanced_styles:
            temperature = self.play_style.get_temperature(move_number)
        else:
            temperature = self.style_config["temperature"]

        with torch.no_grad():
            board_tensor = (
                self.board_encoder.board_to_tensor(board).unsqueeze(0).to(self.device)
            )

            if self.move_history is not None:
                history_tensor, lengths = self.move_history.batch_encode_histories(
                    [board]
                )
                history_tensor = history_tensor.to(self.device)
                lengths = lengths.to(self.device)
            else:
                history_tensor = None
                lengths = None

            policy_logits, value, _ = self.model(board_tensor, history_tensor, lengths)

            move_probs = self.move_encoder.policy_to_move_probs(
                policy_logits[0], board, temperature=temperature
            )

            # Apply style-based move filtering
            if self.using_advanced_styles:
                # Use advanced filtering from play_styles.py
                move_probs = self.play_style.filter_moves(board, move_probs)
            else:
                # Use basic filtering
                if config.play_style == "aggressive":
                    move_probs = self._basic_adjust_for_aggression(move_probs, board)
                elif config.play_style == "defensive":
                    move_probs = self._basic_adjust_for_defense(move_probs, board)

            best_move = max(move_probs.items(), key=lambda x: x[1])
            move, confidence = best_move
            position_value = value.item()

            # Add position evaluation bonus if using advanced styles
            if self.using_advanced_styles:
                bonus = self.play_style.evaluate_position_bonus(board)
                position_value += bonus * 0.1  # Small adjustment

        return move, confidence, position_value

    def _basic_adjust_for_aggression(
        self, move_probs: Dict[chess.Move, float], board: chess.Board
    ) -> Dict[chess.Move, float]:
        """Basic aggressive adjustment (fallback)"""
        adjusted = {}
        for move, prob in move_probs.items():
            multiplier = 1.0

            if board.is_capture(move):
                multiplier *= 1.3

            board.push(move)
            if board.is_check():
                multiplier *= 1.2
            board.pop()

            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN:
                from_rank = chess.square_rank(move.from_square)
                to_rank = chess.square_rank(move.to_square)
                if (piece.color == chess.WHITE and to_rank > from_rank) or (
                    piece.color == chess.BLACK and to_rank < from_rank
                ):
                    multiplier *= 1.1

            adjusted[move] = prob * multiplier

        total = sum(adjusted.values())
        return {move: prob / total for move, prob in adjusted.items()}

    def _basic_adjust_for_defense(
        self, move_probs: Dict[chess.Move, float], board: chess.Board
    ) -> Dict[chess.Move, float]:
        """Basic defensive adjustment (fallback)"""
        adjusted = {}
        for move, prob in move_probs.items():
            multiplier = 1.0

            if board.is_castling(move):
                multiplier *= 1.4

            if board.is_attacked_by(not board.turn, move.from_square):
                multiplier *= 1.2

            adjusted[move] = prob * multiplier

        total = sum(adjusted.values())
        return {move: prob / total for move, prob in adjusted.items()}

    def get_top_moves(
        self, board: chess.Board, n: int = 5
    ) -> List[Tuple[chess.Move, float]]:
        """Get top N moves"""
        with torch.no_grad():
            board_tensor = (
                self.board_encoder.board_to_tensor(board).unsqueeze(0).to(self.device)
            )

            if self.move_history is not None:
                history_tensor, lengths = self.move_history.batch_encode_histories(
                    [board]
                )
                history_tensor = history_tensor.to(self.device)
                lengths = lengths.to(self.device)
            else:
                history_tensor = None
                lengths = None

            policy_logits, _, _ = self.model(board_tensor, history_tensor, lengths)
            move_probs = self.move_encoder.policy_to_move_probs(
                policy_logits[0], board, temperature=1.0
            )

            sorted_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)
            return sorted_moves[:n]


# =============================================================================
# MCTS ENGINE (STRONG MODE)
# =============================================================================


class MCTSEngine(BaseChessEngine):
    """
    Strong MCTS-based engine with time control.
    Uses Monte Carlo Tree Search for stronger play.
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[MCTSConfig] = None,
        device: str = "cuda",
        use_rnn: Optional[bool] = None,
    ):
        config = config or MCTSConfig()
        super().__init__(model_path, config, device, use_rnn)

        # Adjust c_puct based on play style if advanced styles available
        if ADVANCED_STYLES_AVAILABLE:
            play_style = PlayStyle.from_string(config.play_style)
            mcts_params = play_style.get_mcts_params()
            config.c_puct = mcts_params["c_puct"]
            print(f"🎭 MCTS style: {play_style} (c_puct={config.c_puct})")

        # Import MCTS (may need time_control module)
        try:
            from chess_engine.search.mcts import MCTS

            # Create MCTS
            self.mcts = MCTS(
                model=self.model,
                board_encoder=self.board_encoder,
                move_encoder=self.move_encoder,
                device=self.device,
                num_simulations=config.num_simulations,
                c_puct=config.c_puct,
                use_rnn=self.use_rnn,
                enable_caching=config.enable_caching,
                eval_batch_size=config.batch_size if config.use_batched else 1,
                use_progressive_widening=config.progressive_widening,
                enable_early_termination=config.early_termination,
            )

            print(f"🌳 MCTS: {config.num_simulations} simulations/move")
            print(f"✨ Features:")
            print(f"  - Caching: {'ON' if config.enable_caching else 'OFF'}")
            print(f"  - Batched: {'ON' if config.use_batched else 'OFF'}")
            print(
                f"  - Progressive widening: {'ON' if config.progressive_widening else 'OFF'}"
            )
            print(
                f"  - Early termination: {'ON' if config.early_termination else 'OFF'}"
            )

        except ImportError as e:
            print(f"⚠️  Warning: MCTS module not found: {e}")
            print(f"    Falling back to policy-only mode")
            self.mcts = None

        # Time control
        if config.time_control_enabled and TIME_CONTROL_AVAILABLE:
            time_config = TimeControlConfig(
                total_time=config.total_time,
                increment=config.increment,
                strategy=config.time_strategy,
            )
            self.time_control = TimeControl(time_config)
            print(
                f"⏱️  Time Control: {config.total_time}s + {config.increment}s "
                f"({config.time_strategy})"
            )
        else:
            self.time_control = None
            if config.time_control_enabled and not TIME_CONTROL_AVAILABLE:
                print("⚠️  Time control requested but module not available")

        print(f"🎮 Engine ready!")

    def get_move(self, board: chess.Board, move_number: int = 1, **kwargs) -> MoveInfo:
        """Get move using MCTS"""
        config = self.config
        assert isinstance(config, MCTSConfig)

        if self.mcts is None:
            raise RuntimeError("MCTS not available - check import errors")

        start_time = time.time()

        if config.time_control_enabled and self.time_control is not None:
            # Time-controlled search
            time_for_move = self.time_control.get_move_time(board, move_number)
            self.time_control.start_move_timer()

            move, stats = self.mcts.search_with_time_limit(
                board, time_limit=time_for_move, return_stats=True
            )

            actual_time = self.time_control.end_move_timer(move_number)
            elapsed = actual_time

            # Add time info
            if stats is None:
                stats = {}
            stats["time_allocated"] = time_for_move
            stats["time_used"] = actual_time

            # Add complexity if available
            if TIME_CONTROL_AVAILABLE:
                stats["complexity"] = PositionComplexity.calculate(board)

        else:
            # Fixed simulations
            if config.use_batched:
                move, stats = self.mcts.search_batched(board, return_stats=True)
            else:
                move, stats = self.mcts.search(board, return_stats=True)

            elapsed = time.time() - start_time

        # Extract info
        value = stats.get("root_value", 0.0) if stats else 0.0
        confidence = (
            stats.get("top_moves", [{}])[0].get("visit_pct", 0.0) if stats else 0.0
        )

        move_info = MoveInfo(
            move=move,
            confidence=confidence,
            evaluation=value,
            time_taken=elapsed,
            timed_out=False,
            extra_stats=stats or {},
        )

        self.stats.add_move(move_info, from_cache=False)

        return move_info

    def get_top_moves(
        self, board: chess.Board, n: int = 5
    ) -> List[Tuple[chess.Move, float]]:
        """Get top N moves from MCTS"""
        if self.mcts is None:
            return []

        _, stats = self.mcts.search(board, return_stats=True)

        if stats and "top_moves" in stats:
            top = stats["top_moves"][:n]
            return [
                (chess.Move.from_uci(m["move"]), m.get("visit_pct", 0.0)) for m in top
            ]

        return []


# =============================================================================
# UNIFIED GAME INTERFACE
# =============================================================================


class ChessGame:
    """Interactive chess game with any engine"""

    def __init__(self, engine: BaseChessEngine, save_games: bool = True):
        self.engine = engine
        self.board = chess.Board()
        self.move_history: List[chess.Move] = []
        self.save_games = save_games
        self.player_color = chess.WHITE

    def print_board(self):
        """Print current board"""
        print("\n" + "=" * 70)
        print(self.board)
        print("=" * 70)
        print(f"FEN: {self.board.fen()}")
        print(f"Move: {len(self.move_history) + 1}")
        print(f"Turn: {'⚪ White' if self.board.turn else '⚫ Black'}")

        if self.board.is_check():
            print("⚠️  CHECK!")
        if self.board.is_checkmate():
            print("👑 CHECKMATE!")
        elif self.board.is_stalemate():
            print("🤝 STALEMATE!")

        # Time control info for MCTS engine
        if isinstance(self.engine, MCTSEngine):
            config = self.engine.config
            assert isinstance(config, MCTSConfig)
            if config.time_control_enabled and self.engine.time_control:
                stats = self.engine.time_control.get_statistics()
                remaining = stats.get("remaining_time", 0)
                print(f"⏱️  Time: {remaining:.1f}s remaining")

        print()

    def get_player_move(self) -> Optional[chess.Move]:
        """Get move from player"""
        while True:
            try:
                move_str = input("Your move (e.g., 'e2e4' or 'help'): ").strip().lower()

                if move_str in ["quit", "q", "exit"]:
                    return None

                if move_str in ["help", "h", "?"]:
                    self.print_help()
                    continue

                if move_str in ["analyze", "a"]:
                    self.analyze_position()
                    continue

                if move_str in ["hint", "suggest"]:
                    move_info = self.engine.get_move(
                        self.board, move_number=len(self.move_history) + 1
                    )
                    print(
                        f"💡 Suggested: {move_info.move.uci()} ({move_info.confidence:.1%})"
                    )
                    print(
                        f"   {self.engine._annotate_move(move_info.move, self.board)}"
                    )
                    continue

                if move_str in ["legal", "moves"]:
                    self.print_legal_moves()
                    continue

                if move_str in ["stats", "performance"]:
                    self.print_stats()
                    continue

                if move_str.startswith("save"):
                    filename = (
                        move_str.split()[1] if len(move_str.split()) > 1 else None
                    )
                    self.save_game(filename)
                    continue

                move = chess.Move.from_uci(move_str)

                if move in self.board.legal_moves:
                    return move
                else:
                    print("❌ Illegal move!")

            except ValueError as e:
                print(f"❌ Invalid format: {e}")
                print("   Use UCI format like 'e2e4'")

    def print_help(self):
        """Print help"""
        print("\n" + "=" * 70)
        print("📖 COMMANDS")
        print("=" * 70)
        print("  e2e4         - Make a move (UCI format)")
        print("  analyze      - Analyze current position")
        print("  hint         - Get move suggestion")
        print("  legal        - Show all legal moves")
        print("  stats        - Show performance statistics")
        print("  save [name]  - Save game to PGN")
        print("  help         - Show this help")
        print("  quit         - Exit game")
        print("=" * 70)

    def print_legal_moves(self):
        """Print legal moves grouped by piece"""
        legal = list(self.board.legal_moves)
        print(f"\n📋 Legal moves ({len(legal)}):")

        by_piece: Dict[str, List[chess.Move]] = {}
        for move in legal:
            piece = self.board.piece_at(move.from_square)
            piece_name = (
                chess.piece_name(piece.piece_type).capitalize() if piece else "?"
            )
            if piece_name not in by_piece:
                by_piece[piece_name] = []
            by_piece[piece_name].append(move)

        for piece_name in sorted(by_piece.keys()):
            moves = by_piece[piece_name]
            moves_str = ", ".join([m.uci() for m in moves[:10]])
            if len(moves) > 10:
                moves_str += f" ... ({len(moves) - 10} more)"
            print(f"  {piece_name}: {moves_str}")

    def analyze_position(self):
        """Analyze position"""
        print("\n📊 POSITION ANALYSIS")
        print("=" * 70)

        start = time.time()
        top_moves = self.engine.get_top_moves(self.board, n=5)
        elapsed = time.time() - start

        move_info = self.engine.get_move(
            self.board, move_number=len(self.move_history) + 1
        )

        print(f"\nEvaluation: {move_info.evaluation:+.3f}")
        if move_info.evaluation > 0.5:
            print("  → White is winning")
        elif move_info.evaluation < -0.5:
            print("  → Black is winning")
        else:
            print("  → Position is roughly equal")

        print(f"\nTop 5 moves:")
        for i, (move, prob) in enumerate(top_moves, 1):
            bar = "█" * int(prob * 40)
            annotation = self.engine._annotate_move(move, self.board)
            print(f"  {i}. {move.uci():6s} {prob:6.2%} {bar:40s} {annotation}")

        print(f"\nAnalysis time: {elapsed:.2f}s")

    def print_stats(self):
        """Print performance statistics"""
        stats = self.engine.get_performance_stats()

        print("\n" + "=" * 70)
        print("📈 ENGINE PERFORMANCE")
        print("=" * 70)
        print(f"Total moves: {stats['total_moves']}")
        print(f"Average time: {stats['avg_move_time_ms']:.2f}ms")
        print(
            f"Min/Max: {stats['min_move_time_ms']:.2f}ms / {stats['max_move_time_ms']:.2f}ms"
        )

        if isinstance(self.engine, PolicyEngine):
            print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
            if stats["avg_move_time_ms"] < 100:
                print("✅ Meets <100ms requirement")

        print("=" * 70)

    def save_game(self, filename: Optional[str] = None):
        """Save game to PGN"""
        if not self.move_history:
            print("⚠️  No moves to save")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"game_{timestamp}.pgn"

        if not filename.endswith(".pgn"):
            filename += ".pgn"

        game = chess.pgn.Game()
        game.headers["Event"] = "Claude Chess Engine Game"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = (
            "Player" if self.player_color == chess.WHITE else "Engine"
        )
        game.headers["Black"] = (
            "Engine" if self.player_color == chess.WHITE else "Player"
        )
        game.headers["Result"] = self.board.result()

        node = game
        board = chess.Board()
        for move in self.move_history:
            node = node.add_variation(move)
            board.push(move)

        save_dir = Path("saved_games")
        save_dir.mkdir(exist_ok=True)
        filepath = save_dir / filename

        with open(filepath, "w") as f:
            print(game, file=f)

        print(f"💾 Saved to: {filepath}")

    def play(self, player_color: chess.Color = chess.WHITE):
        """Play game"""
        self.player_color = player_color

        print("\n" + "🎮" * 35)
        print("  CHESS GAME - UNIFIED ENGINE")
        print("🎮" * 35)
        print(f"\nYou: {'⚪ White' if player_color else '⚫ Black'}")

        if isinstance(self.engine, PolicyEngine):
            config = self.engine.config
            assert isinstance(config, PolicyConfig)
            mode_desc = "Advanced" if self.engine.using_advanced_styles else "Basic"
            print(f"Engine: Policy mode ({config.play_style}, {mode_desc})")
        elif isinstance(self.engine, MCTSEngine):
            config = self.engine.config
            assert isinstance(config, MCTSConfig)
            print(f"Engine: MCTS mode ({config.num_simulations} sims)")
            if config.time_control_enabled:
                print(f"Time Control: {config.total_time}s + {config.increment}s")

        print("\nType 'help' for commands\n")

        move_number = 0

        while not self.board.is_game_over():
            move_number += 1
            self.print_board()

            if self.board.turn == player_color:
                # Player
                print("👤 Your turn:")
                move = self.get_player_move()

                if move is None:
                    print("\n👋 Game ended")
                    if self.save_games:
                        self.save_game()
                    return

                self.board.push(move)
                self.move_history.append(move)

            else:
                # Engine
                print("🤖 Engine thinking...")

                if isinstance(self.engine, MCTSEngine):
                    move_info = self.engine.get_move(
                        self.board, move_number=move_number
                    )
                else:
                    move_info = self.engine.get_move(
                        self.board, move_number=move_number
                    )

                print(f"\n🤖 Engine plays: {move_info.move.uci()}")
                print(f"   {self.engine._annotate_move(move_info.move, self.board)}")
                print(f"   Confidence: {move_info.confidence:.1%}")
                print(f"   Evaluation: {move_info.evaluation:+.3f}")
                print(f"   Time: {move_info.time_taken:.2f}s")

                # MCTS-specific stats
                if isinstance(self.engine, MCTSEngine) and move_info.extra_stats:
                    stats = move_info.extra_stats
                    sims = stats.get(
                        "simulations_completed", stats.get("simulations", "?")
                    )
                    print(f"   Simulations: {sims}")

                    if "top_moves" in stats and len(stats["top_moves"]) > 0:
                        print(f"\n   💡 Top 3 moves:")
                        for i, m in enumerate(stats["top_moves"][:3], 1):
                            print(
                                f"   {i}. {m['move']:5s} {m['visits']:3d} visits "
                                f"({m.get('visit_pct', 0):.0%}) value={m.get('value', 0):+.3f}"
                            )

                self.board.push(move_info.move)
                self.move_history.append(move_info.move)

        # Game over
        self.print_board()
        self.show_game_summary()

    def show_game_summary(self):
        """Show game summary"""
        print("\n" + "=" * 70)
        print("🏁 GAME OVER")
        print("=" * 70)

        result = self.board.result()
        print(f"Result: {result}")

        if result == "1-0":
            winner = "You" if self.player_color == chess.WHITE else "Engine"
            print(f"⚪ {winner} win!")
        elif result == "0-1":
            winner = "You" if self.player_color == chess.BLACK else "Engine"
            print(f"⚫ {winner} win!")
        else:
            print("🤝 Draw!")

        if self.board.is_checkmate():
            print("Victory by checkmate")
        elif self.board.is_stalemate():
            print("Draw by stalemate")

        print(f"\nTotal moves: {len(self.move_history)}")

        # Final stats
        self.print_stats()

        if self.save_games:
            save = input("\nSave game? (y/n): ").strip().lower()
            if save == "y":
                self.save_game()


# =============================================================================
# MAIN & CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Unified Chess Engine - Policy or MCTS mode with advanced modules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast policy mode (basic)
  python play_chess_unified.py --model best.pt --mode policy --style aggressive

  # Fast policy mode (advanced - requires play_styles.py)
  python play_chess_unified.py --model best.pt --mode policy --style positional

  # Strong MCTS mode
  python play_chess_unified.py --model best.pt --mode mcts --simulations 200

  # MCTS with time control (requires time_control.py)
  python play_chess_unified.py --model best.pt --mode mcts --time 300 --increment 2
        """,
    )

    # Required
    parser.add_argument(
        "--model", type=str, required=True, help="Model checkpoint path"
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="policy",
        choices=["policy", "mcts"],
        help="Engine mode: policy (fast) or mcts (strong)",
    )

    # Game settings
    parser.add_argument(
        "--color",
        type=str,
        default="white",
        choices=["white", "black"],
        help="Your color",
    )
    parser.add_argument("--use-rnn", action="store_true", help="Force RNN usage")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")

    # Play style (works for both modes if advanced styles available)
    style_choices = ["aggressive", "balanced", "defensive"]
    if ADVANCED_STYLES_AVAILABLE:
        style_choices.append("positional")

    parser.add_argument(
        "--style",
        type=str,
        default="balanced",
        choices=style_choices,
        help="Play style (positional requires play_styles.py)",
    )

    # Policy mode settings
    parser.add_argument(
        "--policy-time", type=float, default=5.0, help="Time per move for policy mode"
    )

    # MCTS mode settings
    parser.add_argument(
        "--simulations", type=int, default=100, help="MCTS simulations per move"
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.5,
        help="MCTS exploration constant (overridden by style if advanced)",
    )
    parser.add_argument(
        "--batched", action="store_true", help="Use batched MCTS evaluation"
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable MCTS caching")

    # Time control (MCTS mode)
    parser.add_argument(
        "--time",
        type=float,
        help="Total time in seconds (enables time control, requires time_control.py)",
    )
    parser.add_argument("--increment", type=float, default=0, help="Increment per move")
    parser.add_argument(
        "--strategy",
        type=str,
        default="adaptive",
        choices=["fixed", "proportional", "incremental", "adaptive"],
        help="Time allocation strategy",
    )

    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"
    use_rnn = args.use_rnn if args.use_rnn else None

    print("\n" + "=" * 70)
    print("🎮 UNIFIED CHESS ENGINE")
    print("=" * 70)

    if ADVANCED_STYLES_AVAILABLE:
        print("✨ Advanced play styles: ENABLED")
    else:
        print("ℹ️  Advanced play styles: Not available (basic styles only)")

    if TIME_CONTROL_AVAILABLE:
        print("⏱️  Time control: AVAILABLE")
    else:
        print("ℹ️  Time control: Not available")

    print()

    # Create appropriate engine
    if args.mode == "policy":
        config = PolicyConfig(
            play_style=args.style, time_per_move=args.policy_time, cache_size=1000
        )
        engine = PolicyEngine(args.model, config, device, use_rnn)
    else:  # mcts
        config = MCTSConfig(
            num_simulations=args.simulations,
            c_puct=args.c_puct,
            use_batched=args.batched,
            enable_caching=not args.no_cache,
            play_style=args.style,
            time_control_enabled=args.time is not None,
            total_time=args.time if args.time else 300,
            increment=args.increment,
            time_strategy=args.strategy,
        )
        engine = MCTSEngine(args.model, config, device, use_rnn)

    # Play
    game = ChessGame(engine)
    player_color = chess.WHITE if args.color == "white" else chess.BLACK
    game.play(player_color=player_color)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\n" + "=" * 70)
        print("🎮 UNIFIED CHESS ENGINE")
        print("=" * 70)
        print("\n✨ FEATURES:")
        print("  • Two modes: Policy (fast) and MCTS (strong)")
        print("  • Advanced play styles (if play_styles.py installed)")
        print("  • Time control (if time_control.py installed)")
        print("  • Auto-detection (RNN, architecture)")
        print("  • Comprehensive UI/UX")
        print("\n📖 Usage:")
        print(
            "  python play_chess_unified.py --model <path> --mode <policy|mcts> [options]"
        )
        print("\n🎯 Examples:")
        print("\n  Policy mode (fast):")
        print(
            "    python play_chess_unified.py --model best.pt --mode policy --style aggressive"
        )
        print("\n  MCTS mode (strong):")
        print(
            "    python play_chess_unified.py --model best.pt --mode mcts --simulations 200"
        )
        print("\n  MCTS with time control:")
        print(
            "    python play_chess_unified.py --model best.pt --mode mcts --time 300 --increment 2"
        )
        print("\n  Advanced positional style (requires play_styles.py):")
        print(
            "    python play_chess_unified.py --model best.pt --mode policy --style positional"
        )
        print("\n⚙️  Setup:")
        print("  For full features, place these files in chess_engine/engine/:")
        print("    • play_styles.py  (4 styles, temperature progression, evaluation)")
        print("    • time_control.py (4 strategies, complexity analysis)")
        print("\n  Basic features work without these modules!")
        print("=" * 70)
    else:
        main()
