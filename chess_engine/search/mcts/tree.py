"""
MCTS - Tree Operations

Tree traversal utilities: backpropagation, early termination check, and statistics.
"""

import heapq
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

    # heapq.nlargest(2) is O(N) vs O(N log N) for a full sort
    top2 = heapq.nlargest(2, visit_counts)
    total = sum(visit_counts)

    if top2[0] / total > threshold and top2[0] > 2 * top2[1]:
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
    # Full visit counts over ALL children — used by extract_policy() to build
    # accurate policy targets. Must cover every legal move, not just top-N.
    visit_counts = {
        move.uci(): child.visit_count for move, child in root.children.items()
    }

    stats = {
        "total_visits": root.visit_count,
        "root_value": root.value(),
        "visit_counts": visit_counts,
        "top_moves": [],
        "cache_size": cache_size,
        "num_children": len(root.children),
    }

    # Sort children by visit count
    sorted_children = sorted(
        root.children.items(), key=lambda x: x[1].visit_count, reverse=True
    )

    # Get top 5 moves (kept for logging/debugging purposes)
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
