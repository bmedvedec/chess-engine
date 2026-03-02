"""
DATA LOADING & PREPROCESSING - Dataset Storage

Pickle-based dataset serialization for training examples.
"""

import os
import pickle
from typing import List, Dict


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
        print(f"Saved successfully!")
    except Exception as e:
        print(f"Error saving dataset: {e}")
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
        print(f"Loaded {len(examples):,} examples successfully!")
        return examples
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
