"""
DATA LOADING & PREPROCESSING - PGN Parser

Parses PGN files and extracts training positions with ELO filtering,
time control filtering, streaming mode, and progress callbacks.
"""

import chess
import chess.pgn
from typing import Callable, Iterator, List, Dict, Optional
from tqdm import tqdm

# Constants
SKIP_OPENING_MOVES = 5  # Skip first N moves (usually book moves)
DEFAULT_MIN_ELO = 1500
DEFAULT_MAX_ELO = 3000
DEFAULT_MAX_POSITIONS_PER_GAME = 40
DEFAULT_STREAMING_CHUNK_SIZE = 10000  # Positions per chunk in streaming mode


class ChessGameParser:
    """
    Parse PGN files and extract training positions.

    Features:
    - Normal mode: Load all positions into memory
    - Streaming mode: Process in chunks for large datasets
    - Progress callbacks: Custom progress tracking

    Each position includes:
    - Board state
    - Move played (ground truth)
    - Game outcome (for value target)
    - Ply count (half-move number)
    """

    def __init__(
        self,
        min_elo: int = DEFAULT_MIN_ELO,
        max_elo: int = DEFAULT_MAX_ELO,
        time_controls: Optional[set] = None,
        max_positions_per_game: int = DEFAULT_MAX_POSITIONS_PER_GAME,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Initialize parser.

        Args:
            min_elo: Minimum ELO rating to include games (default: 1500)
            max_elo: Maximum ELO rating to include games (default: 3000)
            time_controls: Optional set of time control strings to filter
            max_positions_per_game: Max positions to extract per game (default: 40)
            progress_callback: Optional callback(games_parsed, positions_extracted)
        """
        self.min_elo = min_elo
        self.max_elo = max_elo
        self.time_controls = time_controls
        self.max_positions_per_game = max_positions_per_game
        self.progress_callback = progress_callback

    def parse_pgn_file(
        self, pgn_path: str, max_games: Optional[int] = None
    ) -> List[Dict]:
        """
        Parse PGN file and extract training examples.

        For large files, consider using parse_pgn_file_streaming() instead.

        Args:
            pgn_path: Path to PGN file
            max_games: Maximum number of games to parse (default: all)

        Returns:
            List of training examples, each with:
            - 'board': chess.Board object
            - 'move': chess.Move object (what was played)
            - 'outcome': Game result (1.0, 0.0, -1.0)
            - 'ply': Half-move number
        """
        examples = []

        try:
            with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn_file:
                game_count = 0

                with tqdm(desc="Parsing games", unit=" games") as pbar:
                    while True:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break

                        if max_games is not None and game_count >= max_games:
                            break

                        if not self._is_valid_game(game):
                            continue

                        game_examples = self._extract_positions(game)
                        examples.extend(game_examples)

                        game_count += 1
                        pbar.update(1)
                        pbar.set_postfix(
                            {"games": game_count, "positions": len(examples)}
                        )

                        if self.progress_callback is not None:
                            self.progress_callback(game_count, len(examples))

        except FileNotFoundError:
            print(f"Error: PGN file not found: {pgn_path}")
            raise
        except Exception as e:
            print(f"Error parsing PGN file: {e}")
            raise

        return examples

    def parse_pgn_file_streaming(
        self,
        pgn_path: str,
        chunk_size: int = DEFAULT_STREAMING_CHUNK_SIZE,
        max_games: Optional[int] = None,
    ) -> Iterator[List[Dict]]:
        """
        Parse PGN file in streaming mode (memory efficient).

        Yields chunks of examples instead of loading all into memory.

        Args:
            pgn_path: Path to PGN file
            chunk_size: Number of positions per chunk (default: 10000)
            max_games: Maximum number of games to parse (None = all games)

        Yields:
            Chunks of training examples
        """
        try:
            with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn_file:
                game_count = 0
                total_positions = 0
                current_chunk = []

                with tqdm(desc="Streaming games", unit=" games") as pbar:
                    while True:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            if current_chunk:
                                yield current_chunk
                            break

                        if max_games is not None and game_count >= max_games:
                            if current_chunk:
                                yield current_chunk
                            break

                        if not self._is_valid_game(game):
                            continue

                        game_examples = self._extract_positions(game)
                        current_chunk.extend(game_examples)
                        total_positions += len(game_examples)

                        game_count += 1
                        pbar.update(1)
                        pbar.set_postfix(
                            {
                                "games": game_count,
                                "positions": total_positions,
                                "chunk": len(current_chunk),
                            }
                        )

                        if self.progress_callback is not None:
                            self.progress_callback(game_count, total_positions)

                        if len(current_chunk) >= chunk_size:
                            yield current_chunk
                            current_chunk = []

        except FileNotFoundError:
            print(f"Error: PGN file not found: {pgn_path}")
            raise
        except Exception as e:
            print(f"Error parsing PGN file: {e}")
            raise

    def _is_valid_game(self, game: chess.pgn.Game) -> bool:
        """Check if game meets quality criteria."""
        headers = game.headers

        try:
            white_elo = int(headers.get("WhiteElo", 0))
            black_elo = int(headers.get("BlackElo", 0))

            if not (
                self.min_elo <= white_elo <= self.max_elo
                and self.min_elo <= black_elo <= self.max_elo
            ):
                return False
        except (ValueError, TypeError):
            return False

        # Time control filter (Lichess-specific)
        if self.time_controls is not None:
            event = headers.get("Event", "").lower()
            if not any(tc in event for tc in self.time_controls):
                return False

        # Must have a result
        result = headers.get("Result", "*")
        if result == "*":
            return False

        return True

    def _extract_positions(self, game: chess.pgn.Game) -> List[Dict]:
        """Extract training positions from a game."""
        examples = []

        result = game.headers.get("Result", "*")
        outcome = self._parse_outcome(result)

        board = game.board()
        positions_extracted = 0

        for move in game.mainline_moves():
            if board.fullmove_number <= SKIP_OPENING_MOVES:
                board.push(move)
                continue

            if positions_extracted >= self.max_positions_per_game:
                break

            examples.append(
                {
                    "board": board.copy(),
                    "move": move,
                    "outcome": outcome if board.turn == chess.WHITE else -outcome,
                    "ply": board.ply(),
                }
            )

            board.push(move)
            positions_extracted += 1

        return examples

    def _parse_outcome(self, result: str) -> float:
        """
        Parse game outcome.

        Returns:
            1.0 for White win, 0.0 for draw, -1.0 for Black win
        """
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        elif result == "1/2-1/2":
            return 0.0
        else:
            return 0.0  # Unknown/abandoned
