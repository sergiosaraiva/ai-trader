"""CNN-LSTM-Attention architecture.

A flexible architecture combining:
- CNN layers for local pattern extraction
- Bi-LSTM for sequential dependencies
- Multi-head attention for important time steps

Can be used at any timeframe with appropriate configuration.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    ArchitectureConfig,
    ArchitectureRegistry,
    BaseArchitecture,
    MultiHeadAttention,
)

logger = logging.getLogger(__name__)


@dataclass
class CNNLSTMConfig(ArchitectureConfig):
    """Configuration for CNN-LSTM-Attention architecture.

    Attributes:
        cnn_filters: List of filter sizes for CNN layers.
        cnn_kernel_sizes: List of kernel sizes for CNN layers.
        cnn_dropout: Dropout after CNN layers.
        lstm_hidden_size: LSTM hidden state size.
        lstm_num_layers: Number of LSTM layers.
        lstm_dropout: Dropout between LSTM layers.
        lstm_bidirectional: Use bidirectional LSTM.
        attention_heads: Number of attention heads.
        attention_dropout: Attention dropout rate.
        use_residual: Use residual connections.
    """

    cnn_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    cnn_dropout: float = 0.2
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_bidirectional: bool = True
    attention_heads: int = 8
    attention_dropout: float = 0.1
    use_residual: bool = True


class CNNBlock(nn.Module):
    """CNN block with batch norm and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.2,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.batch_norm = nn.BatchNorm1d(out_channels) if use_batch_norm else None
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input shape: (batch, channels, seq_len)."""
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class CNNLSTMAttention(BaseArchitecture):
    """CNN-LSTM-Attention architecture.

    Combines convolutional feature extraction, bidirectional LSTM for
    temporal modeling, and multi-head attention for focusing on
    important time steps.

    Architecture flow:
        Input (batch, seq, features)
        -> CNN layers (local patterns)
        -> Bi-LSTM (sequential dependencies)
        -> Multi-head attention (importance weighting)
        -> Output heads (price, direction, confidence)

    Example:
        ```python
        config = CNNLSTMConfig(
            input_dim=50,
            sequence_length=168,
            hidden_dim=256,
            prediction_horizons=[1, 4, 12, 24],
        )
        model = CNNLSTMAttention(config)
        outputs = model(x)  # x: (batch, 168, 50)
        ```
    """

    name = "cnn_lstm_attention"
    description = "CNN-LSTM with Multi-Head Attention for time series"
    supported_output_types = ["regression", "classification", "multi"]

    def __init__(self, config: Optional[Union[CNNLSTMConfig, Dict]] = None):
        """Initialize CNN-LSTM-Attention.

        Args:
            config: Architecture configuration.
        """
        if config is None:
            config = CNNLSTMConfig()
        elif isinstance(config, dict):
            config = CNNLSTMConfig(**{
                k: v for k, v in config.items()
                if k in CNNLSTMConfig.__dataclass_fields__
            })

        super().__init__(config)
        self.config: CNNLSTMConfig = config

        # Build architecture
        self._build_cnn()
        self._build_lstm()
        self._build_attention()
        self._build_output_heads()

    def _build_cnn(self) -> None:
        """Build CNN layers."""
        cnn_layers = []
        in_channels = self.config.input_dim

        for i, (filters, kernel_size) in enumerate(
            zip(self.config.cnn_filters, self.config.cnn_kernel_sizes)
        ):
            cnn_layers.append(
                CNNBlock(
                    in_channels,
                    filters,
                    kernel_size,
                    dropout=self.config.cnn_dropout,
                    use_batch_norm=self.config.use_batch_norm,
                )
            )
            in_channels = filters

        self.cnn = nn.Sequential(*cnn_layers)
        self.cnn_output_dim = self.config.cnn_filters[-1] if self.config.cnn_filters else self.config.input_dim

    def _build_lstm(self) -> None:
        """Build LSTM layers."""
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_dim,
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
            batch_first=True,
            dropout=self.config.lstm_dropout if self.config.lstm_num_layers > 1 else 0,
            bidirectional=self.config.lstm_bidirectional,
        )

        self.lstm_output_dim = self.config.lstm_hidden_size * (
            2 if self.config.lstm_bidirectional else 1
        )

        # Layer norm after LSTM
        self.lstm_norm = nn.LayerNorm(self.lstm_output_dim)

    def _build_attention(self) -> None:
        """Build attention mechanism."""
        self.attention = MultiHeadAttention(
            d_model=self.lstm_output_dim,
            num_heads=self.config.attention_heads,
            dropout=self.config.attention_dropout,
        )

        self.attention_norm = nn.LayerNorm(self.lstm_output_dim)

        # Temporal aggregation
        self.temporal_fc = nn.Linear(self.lstm_output_dim, self.config.hidden_dim)

    def _build_output_heads(self) -> None:
        """Build output heads for predictions."""
        hidden_dim = self.config.hidden_dim
        num_horizons = len(self.config.prediction_horizons)

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        feature_dim = hidden_dim // 2

        # Price prediction head (multi-horizon)
        self.price_head = nn.Linear(feature_dim, num_horizons)

        # Direction prediction head (3 classes: down, neutral, up)
        self.direction_head = nn.Linear(feature_dim, self.config.num_classes * num_horizons)

        # Confidence head (learned uncertainty via concentration)
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, num_horizons * 2),  # alpha and beta for Beta distribution
            nn.Softplus(),  # Ensure positive
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim).

        Returns:
            Dictionary with keys:
                - price: Price predictions (batch, num_horizons)
                - direction_logits: Direction logits (batch, num_horizons, num_classes)
                - alpha: Beta distribution alpha (batch, num_horizons)
                - beta: Beta distribution beta (batch, num_horizons)
                - attention_weights: Attention weights (batch, heads, seq, seq)
        """
        batch_size = x.size(0)

        # CNN expects (batch, channels, seq)
        x_cnn = x.transpose(1, 2)
        x_cnn = self.cnn(x_cnn)
        x_cnn = x_cnn.transpose(1, 2)  # Back to (batch, seq, features)

        # LSTM
        lstm_out, _ = self.lstm(x_cnn)
        lstm_out = self.lstm_norm(lstm_out)

        # Self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Residual connection
        if self.config.use_residual:
            attn_out = self.attention_norm(attn_out + lstm_out)
        else:
            attn_out = self.attention_norm(attn_out)

        # Temporal aggregation (use last timestep + weighted average)
        last_hidden = attn_out[:, -1, :]
        weighted_avg = torch.mean(attn_out * F.softmax(attn_out.sum(dim=-1, keepdim=True), dim=1), dim=1)
        temporal_features = self.temporal_fc(last_hidden + weighted_avg)

        # Shared features
        features = self.feature_extractor(temporal_features)

        # Output heads
        price_pred = self.price_head(features)

        direction_logits = self.direction_head(features)
        direction_logits = direction_logits.view(
            batch_size, len(self.config.prediction_horizons), self.config.num_classes
        )

        confidence_params = self.confidence_head(features)
        confidence_params = confidence_params.view(
            batch_size, len(self.config.prediction_horizons), 2
        )
        alpha = confidence_params[:, :, 0] + 1  # Ensure > 1
        beta = confidence_params[:, :, 1] + 1

        return {
            "price": price_pred,
            "direction_logits": direction_logits,
            "alpha": alpha,
            "beta": beta,
            "attention_weights": attn_weights,
        }

    def get_output_info(self) -> Dict[str, Tuple[str, Tuple]]:
        """Get output information."""
        num_horizons = len(self.config.prediction_horizons)
        return {
            "price": ("regression", (num_horizons,)),
            "direction_logits": ("classification", (num_horizons, self.config.num_classes)),
            "alpha": ("probability", (num_horizons,)),
            "beta": ("probability", (num_horizons,)),
            "attention_weights": ("attention", ("heads", "seq", "seq")),
        }


# Register the architecture
ArchitectureRegistry.register(
    "cnn_lstm_attention",
    CNNLSTMAttention,
    aliases=["cnn_lstm", "short_term"],
)
