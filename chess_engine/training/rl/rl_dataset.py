"""
RL TRAINER - Dataset

PyTorch Dataset for RL training from replay buffer examples.
"""

from typing import List, Tuple

import chess
import torch
from torch.utils.data import Dataset

from chess_engine.utils.board_encoder import BoardEncoder
from chess_engine.utils.move_encoder import MoveEncoder
from chess_engine.data.replay.storage import GameExample


class RLDataset(Dataset):
    """PyTorch Dataset for RL training from replay buffer"""

    def __init__(
        self,
        examples: List[GameExample],
        board_encoder: BoardEncoder,
        move_encoder: MoveEncoder,
    ):
        """
        Initialize dataset.

        Args:
            examples: List of training examples from replay buffer
            board_encoder: Board encoding utility
            move_encoder: Move encoding utility
        """
        self.examples = examples
        self.board_encoder = board_encoder
        self.move_encoder = move_encoder

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get single training example.

        Returns:
            Tuple of (board_tensor, policy_tensor, value)
        """
        example = self.examples[idx]

        # Handle both GameExample objects and dictionary format
        if isinstance(example, dict):
            fen = example["fen"]
            policy_dict = example["policy"]
            value_target = example["value"]
        else:
            # GameExample object
            fen = example.fen
            policy_dict = example.policy
            value_target = example.value

        # Parse board from FEN
        board = chess.Board(fen)

        # Encode board
        board_tensor = self.board_encoder.board_to_tensor(board)

        # Encode policy
        policy_tensor = torch.zeros(self.move_encoder.num_moves)
        for move_uci, prob in policy_dict.items():
            try:
                move = chess.Move.from_uci(move_uci)
                move_idx = self.move_encoder.encode_move(move)
                policy_tensor[move_idx] = prob
            except:
                # Skip invalid moves
                pass

        # Normalize policy (ensure it sums to 1)
        if policy_tensor.sum() > 0:
            policy_tensor = policy_tensor / policy_tensor.sum()

        # Get value
        value = torch.tensor([value_target], dtype=torch.float32)

        return board_tensor, policy_tensor, value
