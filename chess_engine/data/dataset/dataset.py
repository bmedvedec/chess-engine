"""
DATA LOADING & PREPROCESSING - Chess Dataset

PyTorch Dataset with data augmentation and tensor caching.
"""

import random
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from chess_engine.utils.augmentations import BoardAugmentations

# Constants
AUGMENTATION_PROBABILITY = 0.5  # 50% chance to apply augmentation


class ChessDataset(Dataset):
    """PyTorch Dataset for chess positions with optional augmentation and tensor caching."""

    def __init__(
        self,
        examples: List[Dict],
        board_encoder,
        move_encoder,
        augment: bool = True,
        cache_tensors: bool = False,
    ):
        self.examples = examples
        self.board_encoder = board_encoder
        self.move_encoder = move_encoder
        self.augment = augment
        self.cache_tensors = cache_tensors

        self._tensor_cache: Optional[Dict[int, Tuple[torch.Tensor, int, float]]] = (
            {} if cache_tensors else None
        )

        if self.cache_tensors:
            self._build_cache()

    def _build_cache(self) -> None:
        """Pre-compute and cache all tensor conversions."""
        assert self._tensor_cache is not None, "Cache should be initialized"

        print(f"Building tensor cache for {len(self.examples)} examples...")

        for idx in tqdm(range(len(self.examples)), desc="Caching tensors"):
            example = self.examples[idx]

            board_tensor = self.board_encoder.board_to_tensor(example["board"])
            move_index = self.move_encoder.encode_move(example["move"])
            outcome = example["outcome"]

            self._tensor_cache[idx] = (board_tensor, move_index, outcome)

        print(f"Tensor cache built! ({len(self._tensor_cache)} entries)")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._tensor_cache is not None:
            board_tensor, move_index, outcome = self._tensor_cache[idx]

            # Apply augmentation if enabled (even with caching)
            if self.augment and random.random() < AUGMENTATION_PROBABILITY:
                board_tensor = BoardAugmentations.flip_tensor_horizontal(board_tensor)
                move_index = BoardAugmentations.flip_move_index(move_index)

            return (
                board_tensor,
                torch.tensor(move_index, dtype=torch.long),
                torch.tensor(outcome, dtype=torch.float32),
            )

        example = self.examples[idx]

        board = example["board"].copy()  # avoid modifying original
        move = example["move"]
        outcome = example["outcome"]

        if self.augment and torch.rand(1).item() < AUGMENTATION_PROBABILITY:
            board = BoardAugmentations.horizontal_flip(board)
            move = BoardAugmentations.flip_move_horizontal(move)

        board_tensor = self.board_encoder.board_to_tensor(board)
        move_index = self.move_encoder.encode_move(move)

        return (
            board_tensor,
            torch.tensor(move_index, dtype=torch.long),
            torch.tensor(outcome, dtype=torch.float32),
        )

    def clear_cache(self) -> None:
        """Clear tensor cache to free memory."""
        if self._tensor_cache is not None:
            self._tensor_cache.clear()
            print("Tensor cache cleared")
