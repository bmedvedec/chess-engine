import numpy as np
import torch
from typing import Optional, Union


def visualize_tensor(
    self, tensor: Union[torch.Tensor, np.ndarray], channel: Optional[int] = None
):
    """
    Print a visual representation of the tensor.

    Args:
        tensor: torch.Tensor of shape (22, 8, 8)
        channel: If specified, show only this channel. Otherwise show summary.
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()

    if channel is not None:
        print(f"\nChannel {channel}:")
        print("-" * 40)
        for row in range(7, -1, -1):  # Print from rank 8 to rank 1
            print(f"Rank {row + 1}: ", end="")
            for col in range(8):
                val = tensor[channel, row, col]
                print(f"{val:5.2f} ", end="")
            print()
        print("       ", end="")
        for col in range(8):
            print(f"  {chr(ord('a') + col)}   ", end="")
        print()
    else:
        # Show summary of all channels
        print("\nTensor Summary:")
        print("=" * 60)
        channel_names = [
            "Own Pawns",
            "Own Knights",
            "Own Bishops",
            "Own Rooks",
            "Own Queens",
            "Own King",
            "Opp Pawns",
            "Opp Knights",
            "Opp Bishops",
            "Opp Rooks",
            "Opp Queens",
            "Opp King",
            "Color to Move",
            "Move Count",
            "White K-Castle",
            "White Q-Castle",
            "Black K-Castle",
            "Black Q-Castle",
            "En Passant",
            "Halfmove Clock",
        ]

        for ch, name in enumerate(channel_names):
            non_zero = np.sum(tensor[ch] > 0)
            max_val = np.max(tensor[ch])
            print(
                f"Ch {ch:2d} [{name:16s}]: {non_zero:2.0f} active squares, max={max_val:.2f}"
            )
