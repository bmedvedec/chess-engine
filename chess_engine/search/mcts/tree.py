"""
MCTS - Tree Operations

Tree traversal utilities: backpropagation, early termination check, and statistics.
"""

from typing import Dict, List

from chess_engine.search.mcts.node import MCTSNode


def backpropagate(search_path: List[MCTSNode], value: float) -> None:
    """
    Backpropagate value through search path.

    Args:
        search_path: Path from root to leaf
        value: Value from leaf node
    """
    # Alternate value sign for each level (player perspective)
    for node in reversed(search_path):
        node.update(value)
        value = -value  # Flip for opponent


def should_terminate_early(
    root: MCTSNode,
    simulations_done: int,
    threshold: float = 0.7,
) -> bool:
    """
    Check if we should terminate early when one move is clearly dominant.

    Args:
        root: Root node
        simulations_done: Number of simulations completed
        threshold: Visit ratio threshold for early termination (default: 0.7)

    Returns:
        True if should terminate early
    """
    if simulations_done < 30:  # Need minimum samples
        return False

    visit_counts = [child.visit_count for child in root.children.values()]
    if len(visit_counts) < 2:
        return False

    visit_counts_sorted = sorted(visit_counts, reverse=True)

    # If best move has >threshold% of visits and 2x more than second best
    best_ratio = visit_counts_sorted[0] / sum(visit_counts)
    if best_ratio > threshold:
        if visit_counts_sorted[0] > 2 * visit_counts_sorted[1]:
            return True

    return False


def get_search_stats(root: MCTSNode, cache_size: int = 0) -> Dict:
    """
    Get search statistics for analysis.

    Args:
        root: Root node after search
        cache_size: Current cache size

    Returns:
        Dictionary of statistics
    """
    stats = {
        "total_visits": root.visit_count,
        "root_value": root.value(),
        "top_moves": [],
        "cache_size": cache_size,
        "num_children": len(root.children),
    }

    # Sort children by visit count
    sorted_children = sorted(
        root.children.items(), key=lambda x: x[1].visit_count, reverse=True
    )

    # Get top 5 moves
    for move, child in sorted_children[:5]:
        stats["top_moves"].append(
            {
                "move": move.uci(),
                "visits": child.visit_count,
                "value": child.value(),
                "prior": child.prior,
                "visit_pct": (
                    child.visit_count / root.visit_count if root.visit_count > 0 else 0
                ),
            }
        )

    return stats
