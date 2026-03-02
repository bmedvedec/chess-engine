"""
SELF-PLAY - Statistics
"""

from dataclasses import dataclass


@dataclass
class SelfPlayStatistics:
    """Statistics for self-play games."""

    total_games: int = 0
    white_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    resignations: int = 0
    total_examples: int = 0
    total_moves: int = 0
    errors: int = 0

    @property
    def avg_moves_per_game(self) -> float:
        """Average moves per game."""
        return self.total_moves / self.total_games if self.total_games > 0 else 0.0

    @property
    def white_win_rate(self) -> float:
        """White win percentage."""
        return self.white_wins / self.total_games * 100 if self.total_games > 0 else 0.0

    @property
    def black_win_rate(self) -> float:
        """Black win percentage."""
        return self.black_wins / self.total_games * 100 if self.total_games > 0 else 0.0

    @property
    def draw_rate(self) -> float:
        """Draw percentage."""
        return self.draws / self.total_games * 100 if self.total_games > 0 else 0.0

    @property
    def resignation_rate(self) -> float:
        """Resignation percentage."""
        return (
            self.resignations / self.total_games * 100 if self.total_games > 0 else 0.0
        )

    def update(
        self, result: str, num_examples: int, num_moves: int, resigned: bool = False
    ):
        """Update statistics with game result."""
        self.total_games += 1
        self.total_examples += num_examples
        self.total_moves += num_moves

        if resigned:
            self.resignations += 1

        if result == "1-0":
            self.white_wins += 1
        elif result == "0-1":
            self.black_wins += 1
        else:
            self.draws += 1

    def print_summary(self):
        """Print statistics summary."""
        print(f"\nSelf-Play Statistics:")
        print(f"   Total games: {self.total_games}")
        print(f"   Total examples: {self.total_examples}")
        print(f"   Total moves: {self.total_moves}")
        print(f"   Avg moves/game: {self.avg_moves_per_game:.1f}")
        print(f"\n   Results:")
        print(f"      White wins: {self.white_wins} ({self.white_win_rate:.1f}%)")
        print(f"      Black wins: {self.black_wins} ({self.black_win_rate:.1f}%)")
        print(f"      Draws: {self.draws} ({self.draw_rate:.1f}%)")
        if self.resignations > 0:
            print(
                f"      Resignations: {self.resignations} ({self.resignation_rate:.1f}%)"
            )
        if self.errors > 0:
            print(f"   Errors: {self.errors}")
