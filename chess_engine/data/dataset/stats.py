"""
DATA LOADING & PREPROCESSING - Dataset Statistics

Computes and displays dataset statistics (outcome distribution, ply distribution).
"""

from typing import Dict, List


def compute_dataset_statistics(examples: List[Dict]) -> Dict[str, float]:
    """
    Compute comprehensive statistics about the dataset.

    Args:
        examples: List of training examples

    Returns:
        Dictionary with statistics including:
        - total_positions, white_wins, black_wins, draws
        - avg_ply, min_ply, max_ply
        - white_win_rate, black_win_rate, draw_rate
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
