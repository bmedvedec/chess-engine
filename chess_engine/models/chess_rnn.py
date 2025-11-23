"""
RNN ARCHITECTURE IMPLEMENTATION
Complete Track Implementation

This module implements the Recurrent Neural Network (RNN) for processing
move sequences. The RNN captures temporal patterns and game flow context
that complement the CNN's spatial understanding.

Architecture Overview:
- LSTM layers to process move history
- Bidirectional option for richer context
- Attention mechanism (optional) for focusing on important moves
- Integration with CNN features
- Positional encoding for temporal awareness

Why RNN for Chess?
- Captures opening patterns (e.g., Sicilian Defense)
- Understands game phase transitions
- Recognizes recurring themes in move sequences
- Provides context for position evaluation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Tuple, Optional, Union, cast
from torch.utils.checkpoint import checkpoint
from torch.nn.utils.rnn import PackedSequence


class PositionalEncoding(nn.Module):
    """
    Positional encoding for move sequences.

    Adds information about move order to embeddings. This helps the model
    distinguish between:
    - Early game moves (e.g., move 1-10): Opening phase
    - Mid game moves (e.g., move 11-30): Tactical phase
    - Late game moves (e.g., move 31+): Endgame phase

    Uses sinusoidal encoding similar to Transformer models.
    """

    def __init__(self, embedding_dim: int, max_len: int = 512):
        """
        Initialize positional encoding.

        Args:
            embedding_dim: Dimension of move embeddings
            max_len: Maximum sequence length supported (default: 512 moves)
        """
        super().__init__()

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )

        pe = torch.zeros(max_len, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but saved with model)
        self.register_buffer("pe", pe)
        self.pe: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to embeddings.

        Args:
            x: Embedded moves (batch, seq_len, embedding_dim)

        Returns:
            Embeddings with positional information added
        """
        return x + self.pe[: x.size(1)]


class MoveEmbedding(nn.Module):
    """
    Embedding layer for chess moves.

    Converts move indices (0-4095) into dense vector representations.
    Similar to word embeddings in NLP, move embeddings capture:
    - Move similarity (e4 and d4 should be similar)
    - Piece movement patterns
    - Tactical relationships

    Example:
        e2e4 → [0.23, -0.45, 0.67, ..., 0.12]  (64-dim vector)
        e7e5 → [0.21, -0.43, 0.69, ..., 0.15]  (similar to e2e4)
        g1f3 → [-0.56, 0.34, -0.12, ..., 0.89] (different pattern)
    """

    def __init__(
        self,
        num_moves: int = 4096,
        embedding_dim: int = 64,
        padding_idx: int = 0,
        use_positional_encoding: bool = True,
        max_seq_len: int = 512,
    ):
        """
        Initialize move embedding layer.

        Args:
            num_moves: Total number of possible moves (default: 4096)
            embedding_dim: Dimension of embedding vectors (default: 64)
            padding_idx: Index used for padding (default: 0)
            use_positional_encoding: Whether to add positional encoding
            max_seq_len: Maximum sequence length for positional encoding
        """
        super().__init__()

        self.num_moves = num_moves
        self.embedding_dim = embedding_dim
        self.use_positional_encoding = use_positional_encoding

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=num_moves,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        # Use normal initialization (better for embeddings)
        # Xavier is designed for linear layers with tanh/sigmoid activations
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

        # Set padding embedding to zero
        if padding_idx is not None:
            with torch.no_grad():
                self.embedding.weight[padding_idx].fill_(0)

        if use_positional_encoding:
            self.pos_encoding = PositionalEncoding(embedding_dim, max_seq_len)

    def forward(self, move_indices: torch.Tensor) -> torch.Tensor:
        """
        Embed move indices into vectors.

        Args:
            move_indices: Tensor of shape (batch, sequence_length)

        Returns:
            Embedded moves of shape (batch, sequence_length, embedding_dim)
        """
        # Only during training
        if self.training:
            if move_indices.numel() > 0:  # Check tensor is not empty
                max_idx = move_indices.max().item()
                if max_idx >= self.num_moves:
                    raise ValueError(
                        f"Invalid move index: {max_idx} >= {self.num_moves}"
                    )

        # Embed moves
        embedded = self.embedding(move_indices)

        # Add positional encoding
        if self.use_positional_encoding:
            embedded = self.pos_encoding(embedded)

        return embedded


class ChessLSTM(nn.Module):
    """
    LSTM-based RNN for processing move sequences.

    Uses LSTM (Long Short-Term Memory) cells which are good at:
    - Remembering long-term patterns
    - Avoiding vanishing gradients
    - Capturing sequential dependencies

    Architecture:
        Move Indices → Embedding → LSTM Layers → Context Vector

    The context vector summarizes the game history and can be:
    - Combined with CNN features for decision making
    - Used alone for sequence-based predictions
    """

    def __init__(
        self,
        num_moves: int = 4096,
        embedding_dim: int = 64,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_layer_norm: bool = True,
        use_positional_encoding: bool = True,
        use_gradient_checkpointing: bool = False,
        gradient_checkpointing_threshold: int = 50,
        context_strategy: str = "last",  # "last", "max", "mean", or "multi"
    ):
        """
        Initialize LSTM network.

        Args:
            num_moves: Number of possible moves (default: 4096)
            embedding_dim: Dimension of move embeddings (default: 64)
            hidden_size: Size of LSTM hidden state (default: 256)
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout probability (default: 0.3)
            bidirectional: Use bidirectional LSTM (default: False)
            use_layer_norm: Apply layer normalization (default: True)
            use_positional_encoding: Add positional encoding (default: True)
            use_gradient_checkpointing: Enable gradient checkpointing (default: False)
            gradient_checkpointing_threshold: Min seq length for checkpointing (default: 50)
            context_strategy: How to extract context - "last", "max", "mean", or "multi"
        """
        super().__init__()

        self.num_moves = num_moves
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_layer_norm = use_layer_norm
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.gradient_checkpointing_threshold = gradient_checkpointing_threshold
        self.context_strategy = context_strategy

        # Move embedding layer with positional encoding
        self.embedding = MoveEmbedding(
            num_moves=num_moves,
            embedding_dim=embedding_dim,
            padding_idx=0,
            use_positional_encoding=use_positional_encoding,
        )

        # Dropout for embeddings (variational dropout)
        self.embedding_dropout = nn.Dropout(dropout)

        # LSTM layers
        # Note: PyTorch LSTM only applies dropout between layers,
        # so dropout=0 when num_layers=1 (no inter-layer connections)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,  # Only between layers
            bidirectional=bidirectional,
        )

        # Output projection size depends on context strategy
        lstm_output_size = hidden_size * self.num_directions
        if context_strategy == "multi":
            # Multi-strategy combines last, max and mean
            projection_input_size = lstm_output_size * 3
        else:
            projection_input_size = lstm_output_size

        self.output_projection = nn.Linear(projection_input_size, hidden_size)

        # Layer normalization for training stability
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)

        # Inference caching: Store hidden states for incremental processing
        self._cached_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._cache_enabled = False

    def enable_cache(self):
        """Enable caching for inference (10-100x speedup for incremental moves)"""
        self._cache_enabled = True
        self._cached_hidden = None

    def disable_cache(self):
        """Disable caching and clear cached states"""
        self._cache_enabled = False
        self._cached_hidden = None

    def reset_cache(self):
        """Clear cached hidden states"""
        self._cached_hidden = None

    def _lstm_forward(
        self,
        embedded: Union[torch.Tensor, PackedSequence],
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Union[torch.Tensor, PackedSequence], Tuple[torch.Tensor, torch.Tensor]]:
        """
        LSTM forward pass (wrapped for gradient checkpointing).

        Args:
            embedded: Embedded sequences
            hidden_state: Optional initial hidden state

        Returns:
            output, (hidden, cell)
        """
        if hidden_state is not None:
            return self.lstm(embedded, hidden_state)
        else:
            return self.lstm(embedded)

    def extract_context(
        self,
        output: torch.Tensor,
        hidden: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        CONTEXT EXTRACTION: Multiple strategies for better representation.

        Strategies:
        - "last": Use last hidden state
        - "max": Max pooling over sequence
        - "mean": Mean pooling over sequence
        - "multi": Combine all three strategies

        Args:
            output: LSTM outputs (batch, seq_len, hidden_size)
            hidden: Final hidden states (num_layers * num_directions, batch, hidden_size)
            lengths: Actual sequence lengths (batch,)

        Returns:
            Context vector (batch, hidden_size) or (batch, hidden_size*3) for multi
        """
        batch_size, seq_len, hidden_size = output.shape

        # Strategy 1: Last hidden state (from final layer)
        if self.bidirectional:
            forward_hidden = hidden[-2, :, :]
            backward_hidden = hidden[-1, :, :]
            last_context = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            last_context = hidden[-1, :, :]

        if self.context_strategy == "last":
            return last_context

        # Strategy 2: Max pooling over sequence
        max_context, _ = output.max(dim=1)  # (batch, hidden_size)

        if self.context_strategy == "max":
            return max_context

        # Strategy 3: Mean pooling over sequence (masked if lengths provided)
        if lengths is not None:
            # Create mask for valid positions
            mask = (
                torch.arange(seq_len, device=output.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            mask = (mask < lengths.unsqueeze(1)).unsqueeze(-1).float()

            # Masked mean
            mean_context = (output * mask).sum(dim=1) / lengths.unsqueeze(1).float()
        else:
            mean_context = output.mean(dim=1)

        if self.context_strategy == "mean":
            return mean_context

        # Strategy 4: Multi - combine all three
        if self.context_strategy == "multi":
            return torch.cat([last_context, max_context, mean_context], dim=-1)

        # Default to last
        return last_context

    def forward(
        self, move_indices: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSTM.

        FEATURES:
        - Gradient checkpointing for long sequences
        - Inference caching for incremental processing
        - Better context extraction

        Args:
            move_indices: Tensor of shape (batch, seq_len) with move indices
            lengths: Optional tensor of shape (batch,) with actual sequence lengths

        Returns:
            Tuple of:
            - output: LSTM outputs at each timestep (batch, seq_len, hidden_size)
            - context: Final context vector (batch, hidden_size)
        """
        batch_size, seq_len = move_indices.shape

        # INPUT VALIDATION: Only during training, with safety checks
        if self.training and move_indices.numel() > 0:
            max_idx = move_indices.max().item()
            if max_idx >= self.num_moves:
                raise ValueError(f"Invalid move index: {max_idx} >= {self.num_moves}")

            if lengths is not None:
                if not (lengths > 0).all():
                    raise ValueError("All lengths must be positive")
                if not (lengths <= seq_len).all():
                    raise ValueError(
                        f"Length {lengths.max()} exceeds sequence length {seq_len}"
                    )

        # Embed moves with positional encoding
        embedded = self.embedding(move_indices)  # (batch, seq_len, embedding_dim)

        # Apply dropout to embeddings
        embedded = self.embedding_dropout(embedded)

        # Inference caching: Use cached states if enabled
        initial_hidden = None
        if self._cache_enabled and self._cached_hidden is not None:
            initial_hidden = self._cached_hidden

        # Pack padded sequences if lengths provided
        if lengths is not None:
            # Sort by length (required by pack_padded_sequence)
            sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
            _, unsorted_idx = torch.sort(sorted_idx)

            # Sort sequences
            embedded = embedded[sorted_idx]

            # Pack
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, sorted_lengths, batch_first=True, enforce_sorted=True
            )

            if (
                self.use_gradient_checkpointing
                and self.training
                and seq_len > self.gradient_checkpointing_threshold
            ):
                # Use gradient checkpointing for memory efficiency
                # Checkpoint doesn't handle PackedSequence directly
                # Need to unpack, checkpoint, then repack
                def custom_forward(
                    data,
                    batch_sizes,
                    sorted_idx_packed,
                    unsorted_idx_packed,
                    hidden_tuple,
                ):
                    # Reconstruct PackedSequence
                    packed_input = PackedSequence(
                        data, batch_sizes, sorted_idx_packed, unsorted_idx_packed
                    )
                    if hidden_tuple is not None:
                        return self.lstm(packed_input, hidden_tuple)
                    else:
                        return self.lstm(packed_input)

                if initial_hidden is not None:
                    result = checkpoint(
                        custom_forward,
                        packed.data,
                        packed.batch_sizes,
                        packed.sorted_indices,
                        packed.unsorted_indices,
                        initial_hidden,
                        use_reentrant=False,
                    )
                else:
                    result = checkpoint(
                        custom_forward,
                        packed.data,
                        packed.batch_sizes,
                        packed.sorted_indices,
                        packed.unsorted_indices,
                        None,
                        use_reentrant=False,
                    )

                # Handle checkpoint result
                if result is not None:
                    # Extract output and hidden states
                    # Result is tuple: (PackedSequence, (hidden, cell))
                    lstm_output_packed = cast(PackedSequence, result[0])
                    hidden, cell = result[1]
                else:
                    # Fallback if checkpoint returns None
                    result = self._lstm_forward(packed, initial_hidden)
                    lstm_output_packed = cast(PackedSequence, result[0])
                    hidden, cell = result[1]
            else:
                lstm_output_packed: PackedSequence
                result = self._lstm_forward(packed, initial_hidden)
                lstm_output_packed = cast(PackedSequence, result[0])
                hidden, cell = result[1]

            # Unpack sequences - this always returns a Tensor
            # pad_packed_sequence: PackedSequence -> Tensor
            output: torch.Tensor
            output, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_output_packed, batch_first=True
            )

            # Unsort
            output = output[unsorted_idx]
            hidden = hidden[:, unsorted_idx, :]
            cell = cell[:, unsorted_idx, :]
        else:
            # Process without packing
            output: torch.Tensor

            if (
                self.use_gradient_checkpointing
                and self.training
                and seq_len > self.gradient_checkpointing_threshold
            ):

                def custom_forward_unpacked(emb, hidden_tuple):
                    if hidden_tuple is not None:
                        return self.lstm(emb, hidden_tuple)
                    else:
                        return self.lstm(emb)

                if initial_hidden is not None:
                    result = checkpoint(
                        custom_forward_unpacked,
                        embedded,
                        initial_hidden,
                        use_reentrant=False,
                    )
                else:
                    result = checkpoint(
                        custom_forward_unpacked, embedded, None, use_reentrant=False
                    )

                # Handle checkpoint result
                if result is not None:
                    output = cast(torch.Tensor, result[0])
                    hidden, cell = result[1]
                else:
                    # Fallback: run without checkpointing
                    result = self._lstm_forward(embedded, initial_hidden)
                    output = cast(torch.Tensor, result[0])
                    hidden, cell = result[1]
            else:
                result = self._lstm_forward(embedded, initial_hidden)
                output = cast(torch.Tensor, result[0])
                hidden, cell = result[1]

        # Cache hidden states if enabled
        if self._cache_enabled:
            self._cached_hidden = (hidden.detach(), cell.detach())

        # Extract context using chosen strategy
        context = self.extract_context(output, hidden, lengths)

        # Project context to desired size
        context = self.output_projection(context)  # (batch, hidden_size)

        # Apply layer normalization if enabled
        if self.use_layer_norm:
            context = self.layer_norm(context)

        return output, context


class AttentionLayer(nn.Module):
    """
    Attention mechanism for focusing on important moves.

    Instead of treating all moves equally, attention learns to focus on
    moves that are more relevant for the current decision.

    Example:
        In a position with a knight fork threat, attention might focus on:
        - The move that placed the knight
        - Previous moves that created the vulnerability
        - Defensive moves attempted
    """

    def __init__(self, hidden_size: int = 256):
        """
        Initialize attention layer.

        Args:
            hidden_size: Size of hidden representations (default: 256)
        """
        super().__init__()

        self.hidden_size = hidden_size

        # Attention scoring
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(
        self, lstm_output: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention to LSTM outputs.

        Args:
            lstm_output: LSTM outputs (batch, seq_len, hidden_size)
            mask: Optional padding mask (batch, seq_len) - True for valid positions

        Returns:
            Tuple of:
            - attended_output: Weighted sum of outputs (batch, hidden_size)
            - attention_weights: Attention distribution (batch, seq_len)
        """
        # Compute attention scores
        scores = self.attention_weights(lstm_output)  # (batch, seq_len, 1)
        scores = scores.squeeze(-1)  # (batch, seq_len)

        # Apply mask if provided (set padding to very negative value)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Compute attention weights (softmax)
        attention_weights = F.softmax(scores, dim=1)  # (batch, seq_len)

        # Apply attention weights
        attention_weights_expanded = attention_weights.unsqueeze(
            -1
        )  # (batch, seq_len, 1)
        attended_output = torch.sum(
            lstm_output * attention_weights_expanded, dim=1
        )  # (batch, hidden_size)

        return attended_output, attention_weights


class ChessRNN(nn.Module):
    """
    Complete RNN module for chess move sequence processing.

    This is the main RNN component that can be integrated with the CNN.
    Combines embedding, LSTM, and optional attention.
    """

    def __init__(
        self,
        num_moves: int = 4096,
        embedding_dim: int = 64,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = False,
        use_attention: bool = False,
        use_layer_norm: bool = True,
        use_positional_encoding: bool = True,
        use_gradient_checkpointing: bool = False,
        gradient_checkpointing_threshold: int = 50,
        context_strategy: str = "last",
    ):
        """
        Initialize complete RNN module.

        Args:
            num_moves: Number of possible moves (default: 4096)
            embedding_dim: Dimension of move embeddings (default: 64)
            hidden_size: Size of LSTM hidden state (default: 256)
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout probability (default: 0.3)
            bidirectional: Use bidirectional LSTM (default: False)
            use_attention: Use attention mechanism (default: False)
            use_layer_norm: Apply layer normalization (default: True)
            use_positional_encoding: Add positional encoding (default: True)
            use_gradient_checkpointing: Enable gradient checkpointing (default: False)
            gradient_checkpointing_threshold: Min seq length for checkpointing (50)
            context_strategy: Context extraction - "last", "max", "mean", "multi"
        """
        super().__init__()

        self.use_attention = use_attention

        # LSTM component
        self.lstm = ChessLSTM(
            num_moves=num_moves,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            use_layer_norm=use_layer_norm,
            use_positional_encoding=use_positional_encoding,
            use_gradient_checkpointing=use_gradient_checkpointing,
            gradient_checkpointing_threshold=gradient_checkpointing_threshold,
            context_strategy=context_strategy,
        )

        # Optional attention
        if use_attention:
            self.attention = AttentionLayer(hidden_size=hidden_size)

    def enable_cache(self):
        """
        Enable inference caching for 10-100x speedup.

        Use this when processing moves incrementally (e.g., during a game).
        The model will cache hidden states and only process new moves.

        Example:
            rnn.enable_cache()
            for move in game_moves:
                context, _ = rnn(move_tensor)  # Only processes new move
            rnn.disable_cache()
        """
        self.lstm.enable_cache()

    def disable_cache(self):
        """Disable caching and clear cached states"""
        self.lstm.disable_cache()

    def reset_cache(self):
        """Clear cached hidden states without disabling caching"""
        self.lstm.reset_cache()

    def forward(
        self, move_indices: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through complete RNN.

        Args:
            move_indices: Move sequences (batch, seq_len)
            lengths: Actual sequence lengths (batch,)

        Returns:
            Tuple of:
            - context: Context vector summarizing sequence (batch, hidden_size)
            - attention_weights: Optional attention weights (batch, seq_len)
        """
        # Process through LSTM
        lstm_output, lstm_context = self.lstm(move_indices, lengths)

        # Apply attention if enabled
        if self.use_attention:
            # Create mask from lengths based on actual output size
            if lengths is not None:
                batch_size, seq_len = lstm_output.size(0), lstm_output.size(1)
                mask = torch.arange(seq_len, device=lstm_output.device).unsqueeze(0)
                mask = mask < lengths.unsqueeze(1)  # (batch, seq_len)
            else:
                mask = None

            attended_context, attention_weights = self.attention(lstm_output, mask)
            return attended_context, attention_weights
        else:
            return lstm_context, None

    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary for saving/loading"""
        return {
            "num_moves": self.lstm.num_moves,
            "embedding_dim": self.lstm.embedding_dim,
            "hidden_size": self.lstm.hidden_size,
            "num_layers": self.lstm.num_layers,
            "dropout": self.lstm.dropout,
            "bidirectional": self.lstm.bidirectional,
            "use_attention": self.use_attention,
            "use_layer_norm": self.lstm.use_layer_norm,
            "use_gradient_checkpointing": self.lstm.use_gradient_checkpointing,
            "gradient_checkpointing_threshold": self.lstm.gradient_checkpointing_threshold,
            "context_strategy": self.lstm.context_strategy,
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
