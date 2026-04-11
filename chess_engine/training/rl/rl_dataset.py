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

from typing import Dict, FrozenSet, List, Tuple

import chess
import torch
from torch.utils.data import Dataset

from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder
from chess_engine.data.replay.storage import GameExample


# Module-level caches — persist across training calls for the process lifetime.
# Bounded slightly above max buffer size to prevent unbounded growth.
_board_tensor_cache: Dict[str, torch.Tensor] = {}
_policy_sparse_cache: Dict[Tuple, Tuple[List[int], List[float]]] = {}
_CACHE_MAX = 520_000


class RLDataset(Dataset):
    """PyTorch Dataset for RL training from replay buffer.

    Both board and policy encodings are looked up from persistent module-level
    caches and only computed on cache miss. __getitem__ is O(1) tensor indexing.
    """

    def __init__(
        self,
        examples: List[GameExample],
        board_encoder: BoardEncoder,
        move_encoder: MoveEncoder,
    ):
        global _board_tensor_cache, _policy_sparse_cache

        boards = []
        policies = []
        values = []

        for example in examples:
            if isinstance(example, dict):
                fen = example["fen"]
                policy_dict = example["policy"]
                value_target = example["value"]
            else:
                fen = example.fen
                policy_dict = example.policy
                value_target = example.value

            # ------------------------------------------------------------------
            # Board tensor: keyed by FEN
            # ------------------------------------------------------------------
            if fen not in _board_tensor_cache:
                _board_tensor_cache[fen] = board_encoder.board_to_tensor(
                    chess.Board(fen)
                )
                if len(_board_tensor_cache) > _CACHE_MAX:
                    del _board_tensor_cache[next(iter(_board_tensor_cache))]

            boards.append(_board_tensor_cache[fen])

            # ------------------------------------------------------------------
            # Policy tensor: keyed by (FEN, frozenset of policy items).
            # Stored sparsely as (indices, probs) — ~240 B vs ~18 KB for a
            # full tensor — then reconstructed via vectorised scatter.
            # ------------------------------------------------------------------
            policy_key = (fen, frozenset(policy_dict.items()))
            if policy_key not in _policy_sparse_cache:
                idxs: List[int] = []
                probs: List[float] = []
                for move_uci, prob in policy_dict.items():
                    try:
                        idx = move_encoder.encode_move(
                            chess.Move.from_uci(move_uci)
                        )
                        idxs.append(idx)
                        probs.append(float(prob))
                    except Exception:
                        pass
                _policy_sparse_cache[policy_key] = (idxs, probs)
                if len(_policy_sparse_cache) > _CACHE_MAX:
                    del _policy_sparse_cache[next(iter(_policy_sparse_cache))]

            idxs, probs = _policy_sparse_cache[policy_key]
            policy_tensor = torch.zeros(move_encoder.num_moves)
            if idxs:
                policy_tensor[idxs] = torch.tensor(probs)
            if policy_tensor.sum() > 0:
                policy_tensor = policy_tensor / policy_tensor.sum()

            policies.append(policy_tensor)
            values.append(torch.tensor([value_target], dtype=torch.float32))

        self.boards = torch.stack(boards)
        self.policies = torch.stack(policies)
        self.values = torch.stack(values)

    def __len__(self) -> int:
        return len(self.boards)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.boards[idx], self.policies[idx], self.values[idx]
