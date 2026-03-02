"""
REPLAY BUFFER - Sampling Utilities

This module is a thin helper; main sampling logic lives in buffer.py and prioritized.py.
Kept for compatibility with ChatGPT's import structure.
"""

from typing import List

import numpy as np
import torch


def prioritized_sample(
    priorities: List[float],
    batch_size: int,
    alpha: float = 0.6,
) -> List[int]:
    """
    Sample indices based on priorities.

    Args:
        priorities: List of priority values
        batch_size: Number of indices to sample
        alpha: Priority exponent (0 = uniform, 1 = fully prioritized)

    Returns:
        List of sampled indices
    """
    priorities_array = np.array(priorities, dtype=np.float64)
    probs = priorities_array**alpha
    probs /= probs.sum()

    indices = np.random.choice(len(priorities), size=batch_size, replace=False, p=probs)
    return indices.tolist()
