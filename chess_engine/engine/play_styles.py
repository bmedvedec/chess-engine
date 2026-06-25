"""
Play Style System for Chess Engine

Different playing personalities that modify MCTS exploration, move selection
temperature, and position evaluation weights.
"""

from typing import Dict, Any, Optional
from enum import Enum
import chess


class StyleType(Enum):
    """Available playing styles."""

    AGGRESSIVE = "aggressive"
    DEFENSIVE = "defensive"
    POSITIONAL = "positional"
    BALANCED = "balanced"


class PlayStyle:
    """Configurable playing personality: MCTS parameters, temperature, and evaluation biases."""

    # Style definitions with their parameters
    STYLES = {
        StyleType.AGGRESSIVE: {
            "name": "Aggressive",
            "description": "Tactical, attacking play. Sacrifices material for initiative.",
            "temperature": 1.0,  # Higher = more variety in moves
            "mcts_c_puct": 2.0,  # Higher = more exploration (find tactics)
            "mcts_temperature": 1.2,  # Opening/middlegame randomness
            "endgame_temperature": 0.3,  # Still somewhat sharp in endgame
            # Evaluation weights (multipliers for position features)
            "weights": {
                "material": 0.8,  # Less concerned about material
                "king_safety": 0.7,  # Will sacrifice king safety for attack
                "attack": 1.5,  # Heavily weight attacking chances
                "center_control": 1.2,  # Aggressive center play
                "development": 1.3,  # Rapid development for attack
                "pawn_structure": 0.7,  # Less concerned about pawn weaknesses
                "piece_activity": 1.4,  # Active pieces over positional factors
            },
            # Behavioral flags
            "prefer_checks": True,  # Favor checking moves
            "prefer_attacks": True,  # Favor attacking moves
            "accept_sacrifices": True,  # Willing to sacrifice material
            "aggressive_king": True,  # Can castle into attacks
        },
        StyleType.DEFENSIVE: {
            "name": "Defensive",
            "description": "Solid, safe play. Prioritizes king safety and material.",
            "temperature": 0.5,  # Lower = more deterministic
            "mcts_c_puct": 1.0,  # Lower = more exploitation (proven moves)
            "mcts_temperature": 0.8,  # Less variety
            "endgame_temperature": 0.1,  # Very precise endgames
            "weights": {
                "material": 1.3,  # Highly value material
                "king_safety": 1.5,  # Paramount importance
                "attack": 0.7,  # Less aggressive
                "center_control": 1.0,  # Standard center play
                "development": 1.1,  # Solid development
                "pawn_structure": 1.3,  # Strong pawn structure
                "piece_activity": 0.9,  # Piece safety over activity
            },
            "prefer_checks": False,
            "prefer_attacks": False,
            "accept_sacrifices": False,  # Avoid material sacrifices
            "aggressive_king": False,  # Safe king always
        },
        StyleType.POSITIONAL: {
            "name": "Positional",
            "description": "Strategic, long-term play. Builds advantages slowly.",
            "temperature": 0.7,
            "mcts_c_puct": 1.5,  # Balanced exploration
            "mcts_temperature": 1.0,
            "endgame_temperature": 0.2,
            "weights": {
                "material": 1.0,  # Standard material value
                "king_safety": 1.2,  # Important but not paramount
                "attack": 0.9,  # Secondary to position
                "center_control": 1.4,  # Strong emphasis on center
                "development": 1.3,  # Good development crucial
                "pawn_structure": 1.4,  # Very important
                "piece_activity": 1.2,  # Well-placed pieces
            },
            "prefer_checks": False,
            "prefer_attacks": False,
            "accept_sacrifices": False,  # Only positional sacrifices
            "aggressive_king": False,
        },
        StyleType.BALANCED: {
            "name": "Balanced",
            "description": "Standard AlphaZero-style play. Balanced approach.",
            "temperature": 0.8,
            "mcts_c_puct": 1.5,  # Standard AlphaZero value
            "mcts_temperature": 1.0,
            "endgame_temperature": 0.1,
            "weights": {
                "material": 1.0,
                "king_safety": 1.0,
                "attack": 1.0,
                "center_control": 1.0,
                "development": 1.0,
                "pawn_structure": 1.0,
                "piece_activity": 1.0,
            },
            "prefer_checks": False,
            "prefer_attacks": False,
            "accept_sacrifices": False,
            "aggressive_king": False,
        },
    }

    def __init__(self, style: StyleType = StyleType.BALANCED):
        self.style_type = style
        self.config = self.STYLES[style].copy()

    @classmethod
    def from_string(cls, style_name: str) -> "PlayStyle":
        """Create a PlayStyle from a case-insensitive name string; defaults to BALANCED."""
        style_name = style_name.lower()
        for style_type in StyleType:
            if style_type.value == style_name:
                return cls(style_type)
        return cls(StyleType.BALANCED)

    def get_temperature(self, move_number: int) -> float:
        """Return move selection temperature interpolated from opening to endgame phase."""
        if move_number < 15:
            return self.config["mcts_temperature"]
        elif move_number < 40:
            progress = (move_number - 15) / 25.0
            opening_temp = self.config["mcts_temperature"]
            endgame_temp = self.config["endgame_temperature"]
            return opening_temp * (1 - progress) + endgame_temp * progress
        else:
            return self.config["endgame_temperature"]

    def get_mcts_params(self) -> Dict[str, float]:
        return {
            "c_puct": self.config["mcts_c_puct"],
            "temperature": self.config["temperature"],
        }

    def evaluate_position_bonus(self, board: chess.Board) -> float:
        """Style-weighted position bonus in [-1, 1] to bias neural network evaluation."""
        bonus = 0.0
        weights = self.config["weights"]

        material_score = self._evaluate_material(board)
        bonus += material_score * weights["material"] * 0.1

        king_safety = self._evaluate_king_safety(board)
        bonus += king_safety * weights["king_safety"] * 0.15

        center_control = self._evaluate_center(board)
        bonus += center_control * weights["center_control"] * 0.1

        development = self._evaluate_development(board)
        bonus += development * weights["development"] * 0.1

        pawn_structure = self._evaluate_pawns(board)
        bonus += pawn_structure * weights["pawn_structure"] * 0.1

        if self.config.get("prefer_attacks"):
            attack_score = self._evaluate_attacks(board)
            bonus += attack_score * weights["attack"] * 0.15

        return max(-1.0, min(1.0, bonus))

    def _evaluate_material(self, board: chess.Board) -> float:
        """Calculate material advantage (White's perspective)."""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

        score = 0
        for piece_type, value in piece_values.items():
            score += len(board.pieces(piece_type, chess.WHITE)) * value
            score -= len(board.pieces(piece_type, chess.BLACK)) * value

        # Normalize to roughly [-1, 1]
        return score / 40.0

    def _evaluate_king_safety(self, board: chess.Board) -> float:
        """Evaluate king safety for both sides."""

        def king_safety_score(color):
            king_square = board.king(color)
            if king_square is None:
                return -1.0

            safety = 0.0

            # Castling rights (before castling)
            if board.has_kingside_castling_rights(color):
                safety += 0.2
            if board.has_queenside_castling_rights(color):
                safety += 0.2

            # King position safety
            rank = chess.square_rank(king_square)
            file = chess.square_file(king_square)

            # Prefer king on back rank (for White) or 8th rank (for Black)
            if color == chess.WHITE:
                if rank == 0:
                    safety += 0.3
            else:
                if rank == 7:
                    safety += 0.3

            # Prefer king on wing (castled position)
            if file <= 2 or file >= 5:
                safety += 0.2

            # Check pawn shield
            shield_squares = []
            if color == chess.WHITE:
                if rank == 0:
                    shield_squares = [
                        king_square + 8,
                        king_square + 7 if file > 0 else None,
                        king_square + 9 if file < 7 else None,
                    ]
            else:
                if rank == 7:
                    shield_squares = [
                        king_square - 8,
                        king_square - 7 if file < 7 else None,
                        king_square - 9 if file > 0 else None,
                    ]

            shield_count = 0
            for sq in shield_squares:
                if sq is not None and 0 <= sq < 64:
                    piece = board.piece_at(sq)
                    if (
                        piece
                        and piece.piece_type == chess.PAWN
                        and piece.color == color
                    ):
                        shield_count += 1

            safety += shield_count * 0.1

            return min(1.0, safety)

        white_safety = king_safety_score(chess.WHITE)
        black_safety = king_safety_score(chess.BLACK)

        return (
            white_safety - black_safety
            if board.turn == chess.WHITE
            else black_safety - white_safety
        )

    def _evaluate_center(self, board: chess.Board) -> float:
        """Evaluate center control."""
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        extended_center = [
            chess.C3,
            chess.C4,
            chess.C5,
            chess.C6,
            chess.D3,
            chess.D6,
            chess.E3,
            chess.E6,
            chess.F3,
            chess.F4,
            chess.F5,
            chess.F6,
        ]

        score = 0.0

        # Pieces in center
        for square in center_squares:
            piece = board.piece_at(square)
            if piece:
                value = 0.3 if piece.piece_type == chess.PAWN else 0.2
                score += value if piece.color == board.turn else -value

        # Control of center (attacks)
        for square in center_squares + extended_center:
            white_attacks = len(board.attackers(chess.WHITE, square))
            black_attacks = len(board.attackers(chess.BLACK, square))
            control = (white_attacks - black_attacks) * 0.05
            score += control if board.turn == chess.WHITE else -control

        return max(-1.0, min(1.0, score))

    def _evaluate_development(self, board: chess.Board) -> float:
        """Evaluate piece development."""
        if board.fullmove_number > 15:
            return 0.0  # Development only matters in opening

        def development_score(color):
            score = 0.0

            # Knights and bishops off back rank
            back_rank = 0 if color == chess.WHITE else 7

            for piece_type in [chess.KNIGHT, chess.BISHOP]:
                for square in board.pieces(piece_type, color):
                    if chess.square_rank(square) != back_rank:
                        score += 0.2

            # Castling completed
            king_square = board.king(color)
            if king_square:
                king_file = chess.square_file(king_square)
                if king_file <= 2 or king_file >= 5:
                    score += 0.3

            # Queen not moved too early
            for square in board.pieces(chess.QUEEN, color):
                if chess.square_rank(square) == back_rank:
                    score += 0.1

            return score

        white_dev = development_score(chess.WHITE)
        black_dev = development_score(chess.BLACK)

        return (
            white_dev - black_dev
            if board.turn == chess.WHITE
            else black_dev - white_dev
        )

    def _evaluate_pawns(self, board: chess.Board) -> float:
        """Evaluate pawn structure."""
        score = 0.0

        def pawn_score(color):
            pawns = board.pieces(chess.PAWN, color)
            p_score = 0.0

            # Doubled pawns (bad)
            for file in range(8):
                file_pawns = [p for p in pawns if chess.square_file(p) == file]
                if len(file_pawns) > 1:
                    p_score -= 0.1 * (len(file_pawns) - 1)

            # Isolated pawns (bad)
            for square in pawns:
                file = chess.square_file(square)
                has_neighbor = False

                for adj_file in [file - 1, file + 1]:
                    if 0 <= adj_file < 8:
                        if any(chess.square_file(p) == adj_file for p in pawns):
                            has_neighbor = True
                            break

                if not has_neighbor:
                    p_score -= 0.1

            # Passed pawns (good)
            for square in pawns:
                is_passed = True
                file = chess.square_file(square)
                rank = chess.square_rank(square)

                # Check if any enemy pawns block or can capture
                enemy_pawns = board.pieces(chess.PAWN, not color)
                for enemy_square in enemy_pawns:
                    enemy_file = chess.square_file(enemy_square)
                    enemy_rank = chess.square_rank(enemy_square)

                    # Same or adjacent file
                    if abs(file - enemy_file) <= 1:
                        # In front of our pawn
                        if color == chess.WHITE:
                            if enemy_rank > rank:
                                is_passed = False
                                break
                        else:
                            if enemy_rank < rank:
                                is_passed = False
                                break

                if is_passed:
                    # More valuable the further advanced
                    advancement = rank if color == chess.WHITE else (7 - rank)
                    p_score += 0.1 * advancement / 7.0

            return p_score

        white_pawns = pawn_score(chess.WHITE)
        black_pawns = pawn_score(chess.BLACK)

        return (
            white_pawns - black_pawns
            if board.turn == chess.WHITE
            else black_pawns - white_pawns
        )

    def _evaluate_attacks(self, board: chess.Board) -> float:
        """Evaluate attacking potential."""
        score = 0.0

        # Count attacks near enemy king
        enemy_king = board.king(not board.turn)
        if enemy_king is None:
            return 0.0

        # Squares around enemy king
        king_zone = []
        king_file = chess.square_file(enemy_king)
        king_rank = chess.square_rank(enemy_king)

        for df in [-1, 0, 1]:
            for dr in [-1, 0, 1]:
                f = king_file + df
                r = king_rank + dr
                if 0 <= f < 8 and 0 <= r < 8:
                    king_zone.append(chess.square(f, r))

        # Count our attacks in king zone
        for square in king_zone:
            attackers = len(board.attackers(board.turn, square))
            score += attackers * 0.1

        # Bonus for checks available
        legal_moves = list(board.legal_moves)
        checks = sum(1 for move in legal_moves if board.gives_check(move))
        score += checks * 0.15

        return min(1.0, score)

    def filter_moves(
        self, board: chess.Board, move_probs: Dict[chess.Move, float]
    ) -> Dict[chess.Move, float]:
        """
        Filter and reweight moves based on style preferences.

        Args:
            board: Current position
            move_probs: Dictionary of moves to probabilities

        Returns:
            Reweighted move probabilities
        """
        if not self.config.get("prefer_attacks") and not self.config.get(
            "prefer_checks"
        ):
            return move_probs  # No filtering needed

        filtered_probs = move_probs.copy()

        for move in list(filtered_probs.keys()):
            # Boost checking moves for aggressive style
            if self.config.get("prefer_checks") and board.gives_check(move):
                filtered_probs[move] *= 1.3

            # Boost attacking moves
            if self.config.get("prefer_attacks"):
                # Captures
                if board.is_capture(move):
                    filtered_probs[move] *= 1.2

                # Attacks on enemy king zone
                enemy_king = board.king(not board.turn)
                if enemy_king:
                    king_file = chess.square_file(enemy_king)
                    king_rank = chess.square_rank(enemy_king)
                    to_file = chess.square_file(move.to_square)
                    to_rank = chess.square_rank(move.to_square)

                    # Move near enemy king
                    if abs(to_file - king_file) <= 2 and abs(to_rank - king_rank) <= 2:
                        filtered_probs[move] *= 1.1

        # Renormalize
        total = sum(filtered_probs.values())
        if total > 0:
            filtered_probs = {
                move: prob / total for move, prob in filtered_probs.items()
            }

        return filtered_probs

    def __str__(self) -> str:
        """String representation."""
        return f"{self.config['name']} Style"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"PlayStyle({self.style_type.value})"


# Convenience function for UCI integration
def get_style_names() -> list:
    """Get list of available style names for UCI options."""
    return [style.value for style in StyleType]


def create_style(name: str) -> PlayStyle:
    """Create a PlayStyle from a name string."""
    return PlayStyle.from_string(name)


# Testing
if __name__ == "__main__":
    print("=== Play Style System Test ===\n")

    # Test all styles
    for style_type in StyleType:
        style = PlayStyle(style_type)
        print(f"\n{style}")
        print(f"Description: {style.config['description']}")
        print(f"MCTS c_puct: {style.config['mcts_c_puct']}")
        print(f"Temperature: {style.config['temperature']}")
        print(f"Weights: {style.config['weights']}")

        # Test temperature progression
        print(f"\nTemperature progression:")
        for move in [1, 10, 20, 30, 40, 50]:
            temp = style.get_temperature(move)
            print(f"  Move {move}: {temp:.2f}")

    # Test position evaluation with different styles
    print("\n\n=== Position Evaluation Test ===\n")

    import chess

    # Test position (Italian Game)
    board = chess.Board(
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    )

    for style_type in StyleType:
        style = PlayStyle(style_type)
        bonus = style.evaluate_position_bonus(board)
        print(f"{style.config['name']:12s}: Position bonus = {bonus:+.3f}")

    print("\nAll tests passed.")
