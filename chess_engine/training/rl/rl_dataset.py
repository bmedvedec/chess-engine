"""
RL TRAINER - Dataset

PyTorch Dataset for RL training from replay buffer examples.

Both board and policy tensors are cached at module level so each unique
(position, policy) pair is only encoded once across all training calls.

Board cache:   FEN -> CPU tensor           (~5.6 KB each, 2.8 GB at 500k cap)
Policy cache:  (FEN, policy_key) ->        (~240 B each,  120 MB at 500k cap)
               (indices: List[int],
                probs:   List[float])

On a warm cache (all calls after the first), only examples added in the
latest self-play iteration need encoding — typically ~0.02% of the buffer.
"""

import itertools
from typing import Dict, List, Set, Tuple

import chess
import torch
from torch.utils.data import Dataset

from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder
from chess_engine.data.replay.storage import GameExample


# ---------------------------------------------------------------------------
# Module-level caches — persist across training calls for the process lifetime.
# ---------------------------------------------------------------------------
_board_tensor_cache: Dict[str, torch.Tensor] = {}
_policy_sparse_cache: Dict[Tuple, Tuple[List[int], List[float]]] = {}

# Hard ceiling: triggers bulk eviction down to _CACHE_TARGET.
# Set slightly above max buffer size so normal operation never evicts valid
# entries; only buffer overflow or sweep misses hit this path.
_CACHE_MAX = 520_000
_CACHE_TARGET = 500_000  # evict down to this on overflow

# Zombie sweep: every _CACHE_SWEEP_INTERVAL RLDataset constructions, purge
# cache entries whose FENs are no longer present in the replay buffer.
_cache_sweep_counter: int = 0
_CACHE_SWEEP_INTERVAL: int = 10


def _sweep_caches(valid_fens: Set[str]) -> None:
    """Remove cache entries for FENs no longer in the replay buffer.

    Eliminates zombie entries that accumulate after buffer eviction and waste
    memory while being unreachable from the current training set.

    Args:
        valid_fens: Set of all FENs currently in the replay buffer.
    """
    zombie_board_fens = [f for f in _board_tensor_cache if f not in valid_fens]
    for fen in zombie_board_fens:
        del _board_tensor_cache[fen]

    zombie_policy_keys = [k for k in _policy_sparse_cache if k[0] not in valid_fens]
    for key in zombie_policy_keys:
        del _policy_sparse_cache[key]


def _bulk_evict(cache: dict, target_size: int) -> None:
    """Evict oldest-inserted entries until len(cache) <= target_size.

    Uses itertools.islice for a single O(k) pass rather than k individual
    next(iter(cache)) calls, each of which is O(1) but creates iterator
    overhead and is called in a hot loop during buffer fill.

    Args:
        cache: The dict to evict from (insertion-ordered).
        target_size: Target size after eviction.
    """
    n_evict = len(cache) - target_size
    if n_evict <= 0:
        return
    keys_to_evict = list(itertools.islice(cache, n_evict))
    for k in keys_to_evict:
        del cache[k]


class RLDataset(Dataset):
    """PyTorch Dataset for RL training from replay buffer.

    Board and policy encodings are looked up from persistent module-level
    caches and only computed on cache miss.  __init__ stores lightweight
    references and sparse tuples — no full-buffer torch.stack().
    __getitem__ is O(1) board access + O(k) policy reconstruction (k ≈ 30
    legal moves), so the DataLoader collates only batch_size tensors per step.
    """

    def __init__(
        self,
        examples: List[GameExample],
        board_encoder: BoardEncoder,
        move_encoder: MoveEncoder,
    ):
        global _board_tensor_cache, _policy_sparse_cache, _cache_sweep_counter

        self._num_moves = move_encoder.num_moves

        # -------------------------------------------------------------------
        # Periodic zombie sweep — remove cache entries for FENs that have
        # been evicted from the replay buffer since the last sweep.
        # -------------------------------------------------------------------
        _cache_sweep_counter += 1
        if _cache_sweep_counter % _CACHE_SWEEP_INTERVAL == 0:
            valid_fens: Set[str] = {
                (ex["fen"] if isinstance(ex, dict) else ex.fen) for ex in examples
            }
            _sweep_caches(valid_fens)

        self._board_refs: List[torch.Tensor] = []
        self._policy_sparse: List[Tuple[List[int], List[float]]] = []
        value_list: List[float] = []

        for example in examples:
            if isinstance(example, dict):
                fen = example["fen"]
                policy_dict = example["policy"]
                value_target = example["value"]
            else:
                fen = example.fen
                policy_dict = example.policy
                value_target = example.value

            if fen not in _board_tensor_cache:
                _board_tensor_cache[fen] = board_encoder.board_to_tensor(
                    chess.Board(fen)
                )
                if len(_board_tensor_cache) > _CACHE_MAX:
                    _bulk_evict(_board_tensor_cache, _CACHE_TARGET)

            self._board_refs.append(_board_tensor_cache[fen])

            policy_key = (fen, frozenset(policy_dict.items()))
            if policy_key not in _policy_sparse_cache:
                idxs: List[int] = []
                probs: List[float] = []
                for move_uci, prob in policy_dict.items():
                    try:
                        idx = move_encoder.encode_move(chess.Move.from_uci(move_uci))
                        idxs.append(idx)
                        probs.append(float(prob))
                    except Exception:
                        pass
                _policy_sparse_cache[policy_key] = (idxs, probs)
                if len(_policy_sparse_cache) > _CACHE_MAX:
                    _bulk_evict(_policy_sparse_cache, _CACHE_TARGET)

            self._policy_sparse.append(_policy_sparse_cache[policy_key])

            value_list.append(float(value_target))

        self.values = torch.tensor(value_list, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self._board_refs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        board = self._board_refs[idx]

        idxs, probs = self._policy_sparse[idx]
        policy = torch.zeros(self._num_moves)
        if idxs:
            policy[idxs] = torch.tensor(probs, dtype=torch.float32)
            policy_sum = policy.sum()
            if policy_sum > 0:
                policy = policy / policy_sum

        return board, policy, self.values[idx]
