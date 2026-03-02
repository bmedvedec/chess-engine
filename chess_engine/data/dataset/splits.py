"""
DATA LOADING & PREPROCESSING - Train/Val Split

Splits examples into train and validation sets with shuffling and reproducibility.
"""

import random
from typing import List, Dict, Tuple, Optional

# Constants
DEFAULT_TRAIN_RATIO = 0.9


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
