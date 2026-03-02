"""
Self-Play Package

Re-exports all public classes and functions from the self-play module.
"""

from chess_engine.training.self_play.config import SelfPlayConfig
from chess_engine.training.self_play.stats import SelfPlayStatistics
from chess_engine.training.self_play.policy_extraction import extract_policy
from chess_engine.training.self_play.value_targets import should_resign
from chess_engine.training.self_play.game_runner import (
    SelfPlayGameRunner,
    SelfPlayWorker,
)
from chess_engine.training.self_play.parallel import (
    ParallelSelfPlay,
    ParallelSelfPlayWorker,
)

__all__ = [
    "SelfPlayConfig",
    "SelfPlayStatistics",
    "extract_policy",
    "should_resign",
    "SelfPlayGameRunner",
    "SelfPlayWorker",
    "ParallelSelfPlay",
    "ParallelSelfPlayWorker",
]
