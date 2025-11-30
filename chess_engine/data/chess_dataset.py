"""
DATA LOADING & PREPROCESSING
Fast Track Implementation

This module handles loading chess games and preparing them for training.

Pipeline:
Game (PGN) → Parse → Extract Positions → Encode → Batch → Train
"""

import os
import pickle
import random
import chess
from typing import Callable, Iterator, List, Tuple, Optional, Dict
import chess
import chess.pgn
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from chess_engine.utils.board_encoder import DataAugmentation

# Constants
SKIP_OPENING_MOVES = 5  # Skip first N moves (usually book moves)
DEFAULT_MIN_ELO = 1500
DEFAULT_MAX_POSITIONS_PER_GAME = 40
DEFAULT_BATCH_SIZE = 256
DEFAULT_TRAIN_RATIO = 0.9
DEFAULT_NUM_WORKERS = 4  # Parallel data loading workers
AUGMENTATION_PROBABILITY = 0.5  # 50% chance to apply augmentation
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
        max_positions_per_game: int = DEFAULT_MAX_POSITIONS_PER_GAME,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Initialize parser.

        Args:
            min_elo: Minimum ELO rating to include games (default: 1500)
            max_positions_per_game: Max positions to extract per game (default: 40)
            progress_callback: Optional callback(games_parsed, positions_extracted)
                            Called after each game is processed
        """
        self.min_elo = min_elo
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
                        # Read next game
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break

                        # Check if we've reached max games
                        if max_games is not None and game_count >= max_games:
                            break

                        # Filter by ELO if available
                        if not self._is_valid_game(game):
                            continue

                        # Extract positions from this game
                        game_examples = self._extract_positions(game)
                        examples.extend(game_examples)

                        game_count += 1
                        pbar.update(1)
                        pbar.set_postfix(
                            {"games": game_count, "positions": len(examples)}
                        )

                        # Call progress callback if provided
                        if self.progress_callback is not None:
                            self.progress_callback(game_count, len(examples))

        except FileNotFoundError:
            print(f"❌ Error: PGN file not found: {pgn_path}")
            raise
        except Exception as e:
            print(f"❌ Error parsing PGN file: {e}")
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
        Ideal for very large datasets (millions of games).

        Args:
            pgn_path: Path to PGN file
            chunk_size: Number of positions per chunk (default: 10000)
            max_games: Maximum number of games to parse (None = all games)

        Yields:
            Chunks of training examples (each chunk is a list of dicts)

        Example:
            parser = ChessGameParser()
            for chunk in parser.parse_pgn_file_streaming("large_database.pgn"):
                # Process chunk (e.g., save to disk, train on it, etc.)
                save_dataset(chunk, f"chunk_{i}.pkl")
        """
        try:
            with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn_file:
                game_count = 0
                total_positions = 0
                current_chunk = []

                with tqdm(desc="Streaming games", unit=" games") as pbar:
                    while True:
                        # Read next game
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            # Yield remaining chunk if any
                            if current_chunk:
                                yield current_chunk
                            break

                        # Check if we've reached max games
                        if max_games is not None and game_count >= max_games:
                            # Yield remaining chunk if any
                            if current_chunk:
                                yield current_chunk
                            break

                        # Filter by ELO if available
                        if not self._is_valid_game(game):
                            continue

                        # Extract positions from this game
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

                        # Call progress callback if provided
                        if self.progress_callback is not None:
                            self.progress_callback(game_count, total_positions)

                        # Yield chunk if it reaches chunk_size
                        if len(current_chunk) >= chunk_size:
                            yield current_chunk
                            current_chunk = []

        except FileNotFoundError:
            print(f"❌ Error: PGN file not found: {pgn_path}")
            raise
        except Exception as e:
            print(f"❌ Error parsing PGN file: {e}")
            raise

    def _is_valid_game(self, game: chess.pgn.Game) -> bool:
        """Check if game meets quality criteria"""
        headers = game.headers

        # Check for ELO ratings
        try:
            white_elo = int(headers.get("WhiteElo", 0))
            black_elo = int(headers.get("BlackElo", 0))

            if white_elo < self.min_elo or black_elo < self.min_elo:
                return False
        except (ValueError, TypeError):
            # No valid ELO, skip
            return False

        # Must have a result
        result = headers.get("Result", "*")
        if result == "*":
            return False

        return True

    def _extract_positions(self, game: chess.pgn.Game) -> List[Dict]:
        """Extract training positions from a game"""
        examples = []

        # Get game outcome
        result = game.headers.get("Result", "*")
        outcome = self._parse_outcome(result)

        board = game.board()
        positions_extracted = 0

        for move in game.mainline_moves():
            # Skip opening moves (usually book moves)
            if board.fullmove_number <= SKIP_OPENING_MOVES:
                board.push(move)
                continue

            # Skip if too many positions already
            if positions_extracted >= self.max_positions_per_game:
                break

            # Store position before move
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
            1.0 for White win
            0.0 for draw
            -1.0 for Black win
        """
        if result == "1-0":
            return 1.0
        elif result == "0-1":
            return -1.0
        elif result == "1/2-1/2":
            return 0.0
        else:
            return 0.0  # Unknown/abandoned


class ChessDataset(Dataset):
    """
    PyTorch Dataset for chess positions with full data augmenatation and tensor caching.

    Features:
    - Data augmentation (uses DataAugmentation class from board_encoder.py)
    - Tensor caching for faster iteration
    - Memory-efficient operation

    Usage:
        dataset = ChessDataset(
            examples,
            board_encoder,
            move_encoder,
            augment=True,
            cache_tensors=True  # Enable caching for faster epochs
        )
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    """

    def __init__(
        self,
        examples: List[Dict],
        board_encoder,
        move_encoder,
        augment: bool = True,
        cache_tensors: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            examples: List of training examples from parser
            board_encoder: BoardEncoder instance
            move_encoder: MoveEncoder instance
            augment: Whether to apply data augmentation (default: True)
            cache_tensors: Whether to cache encoded tensors (faster but uses more memory)
                          Recommended: True for small datasets (<100k positions)
                          False for large datasets to save memory
        """
        self.examples = examples
        self.board_encoder = board_encoder
        self.move_encoder = move_encoder
        self.augment = augment
        self.cache_tensors = cache_tensors

        # Initialize cache if enabled
        self._tensor_cache: Optional[Dict[int, Tuple[torch.Tensor, int, float]]] = (
            {} if cache_tensors else None
        )

        # Pre-compute and cache all tensors if caching is enabled
        if self.cache_tensors:
            self._build_cache()

    def _build_cache(self) -> None:
        """
        Pre-compute and cache all tensor conversions.

        This speeds up dataset iteration at the cost of memory.
        Only called if cache_tensors=True.
        """
        assert self._tensor_cache is not None, "Cache should be initialized"

        print(f"Building tensor cache for {len(self.examples)} examples...")

        for idx in tqdm(range(len(self.examples)), desc="Caching tensors"):
            example = self.examples[idx]

            # Encode without augmentation for cache
            board_tensor = self.board_encoder.board_to_tensor(example["board"])
            move_index = self.move_encoder.encode_move(example["move"])
            outcome = example["outcome"]

            # Store in cache
            self._tensor_cache[idx] = (board_tensor, move_index, outcome)

        print(f"✅ Tensor cache built! ({len(self._tensor_cache)} entries)")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single training example.

        Returns:
            Tuple of:
            - board_tensor: (20, 8, 8)
            - move_index: scalar
            - outcome: scalar
        """
        # If caching enabled, get from cache
        if self._tensor_cache is not None:
            board_tensor, move_index, outcome = self._tensor_cache[idx]

            # Apply augmentation if enabled (even with caching)
            if self.augment and random.random() < AUGMENTATION_PROBABILITY:
                board_tensor = DataAugmentation.flip_tensor(board_tensor)
                move_index = DataAugmentation.flip_move_index(move_index)

            return (
                board_tensor,
                torch.tensor(move_index, dtype=torch.long),
                torch.tensor(outcome, dtype=torch.float32),
            )

        # Otherwise, compute on-the-fly (no caching)
        example = self.examples[idx]

        board = example["board"].copy()  # Copy to avoid modifying original
        move = example["move"]
        outcome = example["outcome"]

        # Apply data augmentation if enabled (50% chance)
        if self.augment and torch.rand(1).item() < AUGMENTATION_PROBABILITY:
            # board, move = self._apply_horizontal_flip(board, move)
            board = DataAugmentation.horizontal_flip(board)
            move = DataAugmentation.flip_move(move)

        # Encode board
        board_tensor = self.board_encoder.board_to_tensor(board)

        # Encode move
        move_index = self.move_encoder.encode_move(move)

        return (
            board_tensor,
            torch.tensor(move_index, dtype=torch.long),
            torch.tensor(outcome, dtype=torch.float32),
        )

    def clear_cache(self) -> None:
        """Clear tensor cache to free memory"""
        if self._tensor_cache is not None:
            self._tensor_cache.clear()
            print("✅ Tensor cache cleared")


def create_dataloader(
    dataset: ChessDataset,
    batch_size: int = DEFAULT_BATCH_SIZE,
    shuffle: bool = True,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> DataLoader:
    """
    Create DataLoader for training.

    Args:
        dataset: ChessDataset instance
        batch_size: Batch size (default: 256)
        shuffle: Whether to shuffle (default: True)
        num_workers: Number of worker processes (default: 0 = main process only)

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),  # Speed up GPU transfer
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
    )


def split_examples(
    examples: List[Dict],
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split examples into train and validation sets.

    Args:
        examples: List of training examples
        train_ratio: Ratio of examples for training (default: 0.9)
        shuffle: Whether to shuffle before splitting (default: True)
        seed: Random seed for reproducibility (optional)

    Returns:
        Tuple of (train_examples, val_examples)
    """
    if seed is not None:
        random.seed(seed)

    if shuffle:
        examples = examples.copy()
        random.shuffle(examples)

    split_idx = int(len(examples) * train_ratio)
    return examples[:split_idx], examples[split_idx:]


def compute_dataset_statistics(examples: List[Dict]) -> Dict[str, float]:
    """
    Compute comprehensive statistics about the dataset.

    Args:
        examples: List of training examples

    Returns:
        Dictionary with statistics including:
        - total_positions: Total number of positions
        - white_wins: Number of positions from white-winning games
        - black_wins: Number of positions from black-winning games
        - draws: Number of positions from drawn games
        - avg_ply: Average ply count
        - min_ply: Minimum ply count
        - max_ply: Maximum ply count
        - white_win_rate: Percentage of white wins
        - draw_rate: Percentage of draws
    """
    if not examples:
        return {}

    outcomes = [ex["outcome"] for ex in examples]
    plies = [ex["ply"] for ex in examples]

    white_wins = sum(1 for o in outcomes if o > 0.5)
    black_wins = sum(1 for o in outcomes if o < -0.5)
    draws = sum(1 for o in outcomes if abs(o) <= 0.5)
    total = len(examples)

    stats = {
        "total_positions": total,
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "white_win_rate": (white_wins / total * 100) if total > 0 else 0,
        "black_win_rate": (black_wins / total * 100) if total > 0 else 0,
        "draw_rate": (draws / total * 100) if total > 0 else 0,
        "avg_ply": sum(plies) / len(plies) if plies else 0,
        "min_ply": min(plies) if plies else 0,
        "max_ply": max(plies) if plies else 0,
    }

    return stats


def print_dataset_statistics(stats: Dict[str, float]) -> None:
    """
    Pretty print dataset statistics.

    Args:
        stats: Statistics dictionary from compute_dataset_statistics
    """
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total Positions:    {stats['total_positions']:,}")
    print(f"\nOutcome Distribution:")
    print(
        f"  White Wins:       {stats['white_wins']:,} ({stats['white_win_rate']:.1f}%)"
    )
    print(
        f"  Black Wins:       {stats['black_wins']:,} ({stats['black_win_rate']:.1f}%)"
    )
    print(f"  Draws:            {stats['draws']:,} ({stats['draw_rate']:.1f}%)")
    print(f"\nPly Distribution:")
    print(f"  Average:          {stats['avg_ply']:.1f}")
    print(f"  Min:              {stats['min_ply']:.0f}")
    print(f"  Max:              {stats['max_ply']:.0f}")
    print("=" * 60)


def save_dataset(examples: List[Dict], filepath: str) -> None:
    """
    Save parsed examples to disk with error handling.

    Args:
        examples: List of training examples
        filepath: Path where to save the dataset

    Raises:
        Exception: If save operation fails
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print(f"Saving {len(examples):,} examples to {filepath}")

    try:
        with open(filepath, "wb") as f:
            pickle.dump(examples, f)
        print(f"✅ Saved successfully!")
    except Exception as e:
        print(f"❌ Error saving dataset: {e}")
        raise


def load_dataset(filepath: str) -> List[Dict]:
    """
    Load parsed examples from disk with error handling.

    Args:
        filepath: Path to the saved dataset

    Returns:
        List of training examples

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If load operation fails
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    print(f"Loading dataset from {filepath}")

    try:
        with open(filepath, "rb") as f:
            examples = pickle.load(f)
        print(f"✅ Loaded {len(examples):,} examples successfully!")
        return examples
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise
