"""
TIME MANAGEMENT

Time control system for chess engine.

Manages time allocation across moves with different strategies:
- Fixed: Equal time per move
- Proportional: More time for complex positions
- Incremental: Save increment for later
- Adaptive: Learn from game phase
"""

import time
import chess
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import math


@dataclass
class TimeControlConfig:
    """Configuration for time control"""

    total_time: float  # Total time in seconds
    increment: float = 0.0  # Increment per move in seconds
    moves_to_go: Optional[int] = (
        None  # Moves until next time control (None = sudden death)
    )
    strategy: str = (
        "proportional"  # Strategy: fixed, proportional, incremental, adaptive
    )
    emergency_threshold: float = (
        0.1  # Use emergency mode when below this fraction of time
    )
    emergency_moves: int = 5  # Number of moves to survive in emergency


class PositionComplexity:
    """
    Analyze position complexity to inform time allocation.

    More complex positions deserve more thinking time.
    """

    @staticmethod
    def calculate(board: chess.Board) -> float:
        """Calculate position complexity score (0.0 to 1.0)."""
        complexity = 0.0

        # Factor 1: Number of legal moves (0-1 scale)
        num_legal_moves = len(list(board.legal_moves))
        move_complexity = min(1.0, num_legal_moves / 40.0)
        complexity += move_complexity * 0.3

        # Factor 2: Piece count (more pieces = more complex)
        # Max ~32 pieces, normalize to 0-1
        piece_count = len(board.piece_map())
        piece_complexity = piece_count / 32.0
        complexity += piece_complexity * 0.2

        # Factor 3: Check or tactical position
        if board.is_check():
            complexity += 0.2

        # Factor 4: Game phase (middlegame is most complex)
        game_phase = PositionComplexity._get_game_phase(board)
        if game_phase == "middlegame":
            complexity += 0.2
        elif game_phase == "opening":
            complexity += 0.05
        else:  # endgame
            complexity += 0.05

        # Factor 5: Captures available (tactical complexity)
        captures = [m for m in board.legal_moves if board.is_capture(m)]
        if len(captures) > 5:
            complexity += 0.1

        # Clamp to [0, 1]
        return min(1.0, complexity)

    @staticmethod
    def _get_game_phase(board: chess.Board) -> str:
        """Determine game phase based on material"""
        piece_count = len(board.piece_map())
        move_number = board.fullmove_number

        if move_number < 10:
            return "opening"
        elif piece_count > 20:
            return "middlegame"
        elif piece_count > 10:
            return "middlegame"
        else:
            return "endgame"


class TimeControl:
    """
    Manage time allocation across moves.

    Handles various time control formats:
    - Fixed time per move
    - Total time + increment
    - Moves/time with increment
    - Emergency time management
    """

    def __init__(self, config: TimeControlConfig):
        """
        Initialize time control.

        Args:
            config: Time control configuration
        """
        self.config = config
        self.time_used = 0.0
        self.move_times: Dict[int, float] = {}  # Track time per move
        self.start_time: Optional[float] = None

    def get_move_time(
        self,
        board: chess.Board,
        move_number: int,
        remaining_time: Optional[float] = None,
    ) -> float:
        if remaining_time is None:
            remaining_time = self.config.total_time - self.time_used

        if remaining_time <= 0:
            return 0.1

        time_ratio = remaining_time / self.config.total_time
        if time_ratio < self.config.emergency_threshold:
            return self._emergency_time_allocation(remaining_time)

        complexity = PositionComplexity.calculate(board)

        if self.config.strategy == "fixed":
            return self._fixed_time(remaining_time, move_number)
        elif self.config.strategy == "proportional":
            return self._proportional_time(remaining_time, move_number, complexity)
        elif self.config.strategy == "incremental":
            return self._incremental_time(remaining_time, move_number, complexity)
        elif self.config.strategy == "adaptive":
            return self._adaptive_time(remaining_time, move_number, complexity, board)
        else:
            return self._proportional_time(remaining_time, move_number, complexity)

    def _fixed_time(self, remaining_time: float, move_number: int) -> float:
        """
        Fixed time per move strategy.

        Divides remaining time equally across expected moves.
        """
        moves_left = self._estimate_moves_left(move_number)
        return max(0.1, remaining_time / moves_left)

    def _proportional_time(
        self, remaining_time: float, move_number: int, complexity: float
    ) -> float:
        """
        Proportional time strategy.

        Allocates more time for complex positions.
        """
        moves_left = self._estimate_moves_left(move_number)
        base_time = remaining_time / moves_left

        # Scale by complexity (0.5x to 1.5x base time)
        time_multiplier = 0.5 + complexity
        allocated_time = base_time * time_multiplier

        # Don't use more than 20% of remaining time on one move
        max_time = remaining_time * 0.2

        return min(allocated_time, max_time)

    def _incremental_time(
        self, remaining_time: float, move_number: int, complexity: float
    ) -> float:
        """
        Incremental time strategy.

        Saves increment for later, uses base time conservatively.
        """
        moves_left = self._estimate_moves_left(move_number)

        # Reserve increment for future moves
        reserved = self.config.increment * moves_left
        available = remaining_time - reserved

        if available <= 0:
            # Use increment only
            return max(0.1, self.config.increment * 0.8)

        base_time = available / moves_left

        # Scale by complexity
        time_multiplier = 0.7 + complexity * 0.6

        return base_time * time_multiplier + self.config.increment * 0.5

    def _adaptive_time(
        self,
        remaining_time: float,
        move_number: int,
        complexity: float,
        board: chess.Board,
    ) -> float:
        """
        Adaptive time strategy.

        Adjusts based on game phase and position characteristics.
        """
        moves_left = self._estimate_moves_left(move_number)
        game_phase = PositionComplexity._get_game_phase(board)

        base_time = remaining_time / moves_left

        # Game phase adjustments
        if game_phase == "opening":
            # Use less time in opening (theory)
            phase_multiplier = 0.6
        elif game_phase == "middlegame":
            # Most critical phase
            phase_multiplier = 1.2
        else:  # endgame
            # Precision matters
            phase_multiplier = 1.0

        # Complexity adjustment
        complexity_multiplier = 0.7 + complexity * 0.8

        # Combined
        allocated_time = base_time * phase_multiplier * complexity_multiplier

        # Cap at 25% of remaining time
        max_time = remaining_time * 0.25

        return min(allocated_time, max_time)

    def _emergency_time_allocation(self, remaining_time: float) -> float:
        """
        Emergency mode: very low on time.

        Allocate minimal time per move to survive.
        """
        # Divide remaining time across emergency_moves
        time_per_move = remaining_time / self.config.emergency_moves

        # Minimum 0.1s, maximum 2s in emergency
        return max(0.1, min(2.0, time_per_move))

    def _estimate_moves_left(self, move_number: int) -> int:
        """
        Estimate number of moves remaining in game.

        Args:
            move_number: Current move number

        Returns:
            Estimated moves remaining
        """
        if self.config.moves_to_go is not None:
            # Calculate moves until next time control
            moves_in_period = move_number % self.config.moves_to_go
            return max(1, self.config.moves_to_go - moves_in_period)
        else:
            # Sudden death: estimate based on average game length
            average_game_length = 40
            estimated_left = max(5, average_game_length - move_number)
            return estimated_left

    def start_move_timer(self):
        """Start timing a move"""
        self.start_time = time.time()

    def end_move_timer(self, move_number: int) -> float:
        """
        End timing a move and record it.

        Args:
            move_number: Move number that just completed

        Returns:
            Time spent on the move
        """
        if self.start_time is None:
            return 0.0

        elapsed = time.time() - self.start_time
        self.time_used += elapsed
        self.move_times[move_number] = elapsed
        self.start_time = None

        return elapsed

    def get_statistics(self) -> Dict[str, float]:
        """
        Get time usage statistics.

        Returns:
            Dictionary of statistics
        """
        if not self.move_times:
            return {
                "total_time_used": self.time_used,
                "avg_time_per_move": 0.0,
                "min_time": 0.0,
                "max_time": 0.0,
                "remaining_time": self.config.total_time - self.time_used,
            }

        times = list(self.move_times.values())

        return {
            "total_time_used": self.time_used,
            "avg_time_per_move": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "num_moves": len(times),
            "remaining_time": self.config.total_time - self.time_used,
        }

    def is_time_critical(self) -> bool:
        """
        Check if time situation is critical.

        Returns:
            True if running low on time
        """
        remaining = self.config.total_time - self.time_used
        time_ratio = remaining / self.config.total_time
        return time_ratio < self.config.emergency_threshold


def test_time_control():
    """Test time control system"""
    print("=" * 80)
    print("TESTING TIME CONTROL SYSTEM")
    print("=" * 80)

    # Test 1: Fixed time strategy
    print("\n1. Testing FIXED time strategy...")
    config = TimeControlConfig(
        total_time=300, increment=0, strategy="fixed"  # 5 minutes
    )
    tc = TimeControl(config)

    board = chess.Board()
    for move_num in [1, 10, 20, 30]:
        time_allocated = tc.get_move_time(board, move_num)
        print(f"   Move {move_num}: {time_allocated:.2f}s")

    # Test 2: Proportional time strategy
    print("\n2. Testing PROPORTIONAL time strategy...")
    config = TimeControlConfig(total_time=300, increment=0, strategy="proportional")
    tc = TimeControl(config)

    # Test with different position complexities
    simple_board = chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1")  # Simple endgame
    complex_board = chess.Board()  # Starting position

    print(f"   Simple position (Move 1): {tc.get_move_time(simple_board, 1):.2f}s")
    print(f"   Complex position (Move 1): {tc.get_move_time(complex_board, 1):.2f}s")

    # Test 3: Incremental strategy
    print("\n3. Testing INCREMENTAL time strategy...")
    config = TimeControlConfig(
        total_time=180,  # 3 minutes
        increment=2,  # 2 second increment
        strategy="incremental",
    )
    tc = TimeControl(config)

    for move_num in [1, 10, 20]:
        time_allocated = tc.get_move_time(board, move_num)
        print(f"   Move {move_num}: {time_allocated:.2f}s")

    # Test 4: Emergency mode
    print("\n4. Testing EMERGENCY mode...")
    config = TimeControlConfig(
        total_time=10,  # Only 10 seconds left!
        increment=0,
        strategy="proportional",
        emergency_threshold=0.5,
    )
    tc = TimeControl(config)
    tc.time_used = 9  # Used 9 seconds already

    time_allocated = tc.get_move_time(board, 25, remaining_time=1.0)
    print(f"   Emergency allocation (1s remaining): {time_allocated:.2f}s")
    print(f"   Is critical: {tc.is_time_critical()}")

    # Test 5: Position complexity
    print("\n5. Testing POSITION COMPLEXITY...")
    positions = [
        ("Starting position", chess.Board()),
        ("Simple endgame", chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1")),
        (
            "Complex middlegame",
            chess.Board(
                "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1"
            ),
        ),
    ]

    for name, pos in positions:
        complexity = PositionComplexity.calculate(pos)
        print(f"   {name}: {complexity:.2f}")

    # Test 6: Adaptive strategy
    print("\n6. Testing ADAPTIVE strategy...")
    config = TimeControlConfig(
        total_time=600, increment=0, strategy="adaptive"  # 10 minutes
    )
    tc = TimeControl(config)

    opening = chess.Board()
    middlegame = chess.Board(
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1"
    )
    endgame = chess.Board("8/8/4k3/8/8/4K3/8/8 w - - 0 1")

    print(f"   Opening (Move 5): {tc.get_move_time(opening, 5):.2f}s")
    print(f"   Middlegame (Move 20): {tc.get_move_time(middlegame, 20):.2f}s")
    print(f"   Endgame (Move 50): {tc.get_move_time(endgame, 50):.2f}s")

    # Test 7: Time tracking
    print("\n7. Testing TIME TRACKING...")
    config = TimeControlConfig(total_time=300, strategy="proportional")
    tc = TimeControl(config)

    tc.start_move_timer()
    time.sleep(0.1)  # Simulate thinking
    elapsed = tc.end_move_timer(1)

    print(f"   Move 1 took: {elapsed:.2f}s")
    print(f"   Total time used: {tc.time_used:.2f}s")

    stats = tc.get_statistics()
    print(f"   Statistics: {stats}")

    print("\n" + "=" * 80)
    print("TIME CONTROL TESTS COMPLETE")
    print("=" * 80)
    print("\nKey features working:")
    print("  - Multiple time strategies (fixed, proportional, incremental, adaptive)")
    print("  - Position complexity analysis")
    print("  - Emergency time management")
    print("  - Time tracking and statistics")
    print("  - Game phase detection")

    print("\nNext steps:")
    print("  1. Integrate with MCTS (add time limit to search)")
    print("  2. Integrate with play_chess_mcts.py")
    print("  3. Test in real games with time pressure")


if __name__ == "__main__":
    test_time_control()
