"""
MCTS - Neural Network Evaluator

Handles single and batched neural network evaluation of chess positions,
including RNN support and position caching integration.
"""

import torch
import chess
from typing import Dict, List, Tuple, Optional

from chess_engine.utils.move_history import MoveHistory
from chess_engine.search.mcts.cache import PositionCache


class Evaluator:
    """
    Neural network evaluator for MCTS positions.

    Handles single and batch evaluation, caching, and both CNN and RNN models.
    """

    def __init__(
        self,
        model,
        board_encoder,
        move_encoder,
        device: torch.device,
        temperature: float = 1.0,
        use_rnn: bool = False,
        cache: Optional[PositionCache] = None,
        rnn_max_history: int = 15,
    ):
        """
        Initialize evaluator.

        Args:
            model: Neural network model
            board_encoder: BoardEncoder instance
            move_encoder: MoveEncoder instance
            device: Torch device (cuda/cpu)
            temperature: Temperature for policy output (default: 1.0)
            use_rnn: Whether model uses RNN (default: False)
            cache: Optional PositionCache for transposition table
        """
        self.model = model
        self.board_encoder = board_encoder
        self.move_encoder = move_encoder
        self.device = device
        self.temperature = temperature
        self.use_rnn = use_rnn
        self.cache = cache
        self.history_encoder: Optional[MoveHistory] = MoveHistory(max_length=rnn_max_history) if use_rnn else None

    def evaluate_position(
        self, board: chess.Board
    ) -> Tuple[Dict[chess.Move, float], float]:
        """
        Evaluate position using neural network with caching.

        Args:
            board: Board to evaluate

        Returns:
            Tuple of (policy_probs, value)
        """
        # Check cache first
        if self.cache is not None:
            fen = board.fen()
            cached = self.cache.get(fen)
            if cached is not None:
                return cached

        with torch.no_grad():
            # Encode board
            board_tensor = self.board_encoder.board_to_tensor(board).unsqueeze(0)
            board_tensor = board_tensor.to(self.device)

            # Get predictions
            if self.use_rnn:
                assert self.history_encoder is not None
                move_history, _ = self.history_encoder.encode_board(board, pad=True)
                move_history = move_history.unsqueeze(0).to(self.device)

                # Get actual sequence length (not padded length)
                # Ensure minimum length of 1 to avoid pack_padded_sequence error
                actual_length = max(1, min(len(board.move_stack), self.history_encoder.max_length))
                seq_length = torch.LongTensor([actual_length])

                policy_logits, value, _ = self.model(
                    board_tensor, move_history, seq_length
                )
            else:
                output = self.model(board_tensor)
                policy_logits, value = output[0], output[1]

            # Convert to move probabilities
            policy_probs = self.move_encoder.policy_to_move_probs(
                policy_logits[0], board, temperature=self.temperature
            )

            value = value.item()

        # Cache result if enabled
        if self.cache is not None:
            fen = board.fen()
            self.cache.put(fen, policy_probs, value)

        return policy_probs, value

    def evaluate_positions_batch(
        self, nodes: List
    ) -> Tuple[List[Dict[chess.Move, float]], List[float]]:
        """
        Evaluate multiple positions in a batch (more efficient on GPU).

        Args:
            nodes: List of MCTSNode objects to evaluate

        Returns:
            Tuple of (policies, values)
        """
        with torch.no_grad():
            # Encode all boards
            board_tensors = torch.stack(
                [self.board_encoder.board_to_tensor(node.board) for node in nodes]
            ).to(self.device)

            # Get predictions
            if self.use_rnn:
                assert self.history_encoder is not None
                # Encode all move histories
                move_histories = []
                seq_lengths = []

                for node in nodes:
                    move_history, _ = self.history_encoder.encode_board(node.board, pad=True)
                    move_histories.append(move_history)

                    # Ensure minimum length of 1 to avoid pack_padded_sequence error
                    actual_length = max(1, min(len(node.board.move_stack), self.history_encoder.max_length))
                    seq_lengths.append(actual_length)

                move_histories = torch.stack(move_histories).to(self.device)
                seq_lengths = torch.LongTensor(seq_lengths)

                policy_logits, values, _ = self.model(
                    board_tensors, move_histories, seq_lengths
                )
            else:
                output = self.model(board_tensors)
                policy_logits, values = output[0], output[1]

            # Convert to move probabilities
            policies = []
            for i, node in enumerate(nodes):
                policy_probs = self.move_encoder.policy_to_move_probs(
                    policy_logits[i], node.board, temperature=self.temperature
                )
                policies.append(policy_probs)

            values = values.squeeze().cpu().numpy()
            if len(nodes) == 1:
                values = [float(values)]
            else:
                values = values.tolist()

        return policies, values

    @staticmethod
    def get_game_result(board: chess.Board) -> float:
        """
        Get game result for terminal position.

        Args:
            board: Terminal board position

        Returns:
            Result from perspective of side that just moved
        """
        result = board.result()

        # The side that just moved is opposite to current turn
        # (because turn switches after a move)
        side_that_moved = not board.turn

        if result == "1-0":  # White wins
            return 1.0 if side_that_moved == chess.WHITE else -1.0
        elif result == "0-1":  # Black wins
            return -1.0 if side_that_moved == chess.WHITE else 1.0
        else:  # Draw
            return 0.0
