"""
SELF-PLAY - Value Target Utilities

Provides resignation checking for self-play games.
"""

from typing import Dict, Optional


def should_resign(stats: Optional[Dict], threshold: float = -0.45) -> bool:
    """
    Check if the position is hopeless and should resign.

    Args:
        stats: MCTS statistics including root value
        threshold: Value threshold below which to resign (default: -0.9)

    Returns:
        True if should resign
    """
    if stats is None or "root_value" not in stats:
        return False

    root_value = stats["root_value"]
    return root_value < threshold
