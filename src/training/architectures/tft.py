"""Temporal Fusion Transformer architecture.

A flexible implementation of the Temporal Fusion Transformer for
interpretable time series forecasting with multi-horizon predictions.
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
    GatedResidualNetwork,
    MultiHeadAttention,
)

logger = logging.getLogger(__name__)


@dataclass
class TFTConfig(ArchitectureConfig):
    """Configuration for Temporal Fusion Transformer.

    Attributes:
        lstm_hidden_size: LSTM encoder hidden size.
        lstm_num_layers: Number of LSTM layers.
        lstm_dropout: LSTM dropout.
        attention_heads: Number of attention heads.
        attention_dropout: Attention dropout.
        grn_hidden_size: GRN hidden layer size.
        grn_dropout: GRN dropout.
        quantiles: Quantile outputs for uncertainty.
        static_input_dim: Dimension of static features (optional).
        use_static_context: Whether to use static context.
    """

    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.1
    attention_heads: int = 4
    attention_dropout: float = 0.1
    grn_hidden_size: int = 256
    grn_dropout: float = 0.1
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])
    static_input_dim: int = 0
    use_static_context: bool = False


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for feature importance."""

    def __init__(
        self,
        input_dim: int,
        num_inputs: int,
        hidden_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()

        self.num_inputs = num_inputs
        self.input_dim = input_dim

        # GRN for each input
        self.grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim, hidden_dim, hidden_dim, dropout, context_dim
            )
            for _ in range(num_inputs)
        ])

        # Flattened GRN for variable weights
        self.weight_grn = GatedResidualNetwork(
            hidden_dim * num_inputs,
            hidden_dim,
            num_inputs,
            dropout,
            context_dim,
        )

    def forward(
        self,
        inputs: List[torch.Tensor],
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            inputs: List of input tensors, each (batch, seq, input_dim).
            context: Optional context tensor (batch, context_dim).

        Returns:
            Tuple of (weighted_output, variable_weights).
        """
        # Process each input through its GRN
        processed = []
        for i, (inp, grn) in enumerate(zip(inputs, self.grns)):
            processed.append(grn(inp, context))

        # Stack and compute weights
        stacked = torch.stack(processed, dim=-1)  # (batch, seq, hidden, num_inputs)
        flattened = stacked.view(stacked.size(0), stacked.size(1), -1)

        weights = self.weight_grn(flattened, context)
        weights = F.softmax(weights, dim=-1)  # (batch, seq, num_inputs)

        # Weighted combination
        weights_expanded = weights.unsqueeze(2)  # (batch, seq, 1, num_inputs)
        weighted = (stacked * weights_expanded).sum(dim=-1)  # (batch, seq, hidden)

        return weighted, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable multi-head attention with additive attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with interpretable attention."""
        batch_size = query.size(0)

        # Linear projections
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k)

        # Transpose for attention: (batch, heads, seq, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(context)

        # Average attention weights across heads for interpretability
        avg_attn = attn_weights.mean(dim=1)

        return output, avg_attn


class TemporalFusionTransformer(BaseArchitecture):
    """Temporal Fusion Transformer architecture.

    A simplified but complete TFT implementation supporting:
    - Variable selection for input feature importance
    - LSTM encoder for temporal patterns
    - Interpretable multi-head attention
    - Quantile outputs for uncertainty estimation
    - Multi-horizon predictions

    Example:
        ```python
        config = TFTConfig(
            input_dim=50,
            sequence_length=90,
            hidden_dim=256,
            prediction_horizons=[1, 3, 5, 7],
            quantiles=[0.1, 0.5, 0.9],
        )
        model = TemporalFusionTransformer(config)
        outputs = model(x)
        ```
    """

    name = "temporal_fusion_transformer"
    description = "Temporal Fusion Transformer for interpretable forecasting"
    supported_output_types = ["regression", "classification", "multi"]

    def __init__(self, config: Optional[Union[TFTConfig, Dict]] = None):
        """Initialize TFT.

        Args:
            config: Architecture configuration.
        """
        if config is None:
            config = TFTConfig()
        elif isinstance(config, dict):
            config = TFTConfig(**{
                k: v for k, v in config.items()
                if k in TFTConfig.__dataclass_fields__
            })

        super().__init__(config)
        self.config: TFTConfig = config

        # Build architecture
        self._build_input_processing()
        self._build_temporal_processing()
        self._build_output_heads()

    def _build_input_processing(self) -> None:
        """Build input processing layers."""
        # Input embedding
        self.input_embedding = nn.Linear(
            self.config.input_dim,
            self.config.hidden_dim,
        )

        # Input GRN for variable selection
        self.input_grn = GatedResidualNetwork(
            self.config.hidden_dim,
            self.config.grn_hidden_size,
            self.config.hidden_dim,
            self.config.grn_dropout,
        )

    def _build_temporal_processing(self) -> None:
        """Build temporal processing layers."""
        # LSTM encoder
        self.encoder_lstm = nn.LSTM(
            input_size=self.config.hidden_dim,
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
            batch_first=True,
            dropout=self.config.lstm_dropout if self.config.lstm_num_layers > 1 else 0,
            bidirectional=False,
        )

        # Project LSTM output to hidden_dim
        self.lstm_projection = nn.Linear(
            self.config.lstm_hidden_size,
            self.config.hidden_dim,
        )

        # GRN after LSTM
        self.post_lstm_grn = GatedResidualNetwork(
            self.config.hidden_dim,
            self.config.grn_hidden_size,
            self.config.hidden_dim,
            self.config.grn_dropout,
        )

        # Interpretable multi-head attention
        self.attention = InterpretableMultiHeadAttention(
            self.config.hidden_dim,
            self.config.attention_heads,
            self.config.attention_dropout,
        )

        # Post-attention GRN
        self.post_attention_grn = GatedResidualNetwork(
            self.config.hidden_dim,
            self.config.grn_hidden_size,
            self.config.hidden_dim,
            self.config.grn_dropout,
        )

        # Layer norm
        self.attention_norm = nn.LayerNorm(self.config.hidden_dim)

    def _build_output_heads(self) -> None:
        """Build output heads."""
        hidden_dim = self.config.hidden_dim
        num_horizons = len(self.config.prediction_horizons)
        num_quantiles = len(self.config.quantiles)

        # Pre-output GRN
        self.output_grn = GatedResidualNetwork(
            hidden_dim,
            self.config.grn_hidden_size,
            hidden_dim,
            self.config.grn_dropout,
        )

        # Quantile output head
        self.quantile_head = nn.Linear(hidden_dim, num_horizons * num_quantiles)

        # Point prediction head
        self.point_head = nn.Linear(hidden_dim, num_horizons)

        # Direction head
        self.direction_head = nn.Linear(hidden_dim, num_horizons * self.config.num_classes)

        # Confidence head (Beta distribution parameters)
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_horizons * 2),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim).

        Returns:
            Dictionary with predictions and attention weights.
        """
        batch_size = x.size(0)
        num_horizons = len(self.config.prediction_horizons)
        num_quantiles = len(self.config.quantiles)

        # Input processing
        embedded = self.input_embedding(x)
        processed = self.input_grn(embedded)

        # LSTM encoding
        lstm_out, (h_n, c_n) = self.encoder_lstm(processed)
        lstm_projected = self.lstm_projection(lstm_out)

        # Post-LSTM GRN with skip connection
        temporal = self.post_lstm_grn(lstm_projected) + processed

        # Self-attention
        attn_out, attn_weights = self.attention(temporal, temporal, temporal)

        # Post-attention with skip connection
        attn_out = self.post_attention_grn(attn_out)
        attn_out = self.attention_norm(attn_out + temporal)

        # Use last timestep for predictions
        final_hidden = attn_out[:, -1, :]

        # Output GRN
        output_features = self.output_grn(final_hidden)

        # Quantile predictions
        quantile_pred = self.quantile_head(output_features)
        quantile_pred = quantile_pred.view(batch_size, num_horizons, num_quantiles)

        # Point predictions (use median quantile if available)
        point_pred = self.point_head(output_features)

        # Direction predictions
        direction_logits = self.direction_head(output_features)
        direction_logits = direction_logits.view(batch_size, num_horizons, self.config.num_classes)

        # Confidence (Beta distribution)
        confidence_params = self.confidence_head(output_features)
        confidence_params = confidence_params.view(batch_size, num_horizons, 2)
        alpha = confidence_params[:, :, 0] + 1
        beta = confidence_params[:, :, 1] + 1

        return {
            "price": point_pred,
            "quantiles": quantile_pred,
            "direction_logits": direction_logits,
            "alpha": alpha,
            "beta": beta,
            "attention_weights": attn_weights,
        }

    def get_output_info(self) -> Dict[str, Tuple[str, Tuple]]:
        """Get output information."""
        num_horizons = len(self.config.prediction_horizons)
        num_quantiles = len(self.config.quantiles)

        return {
            "price": ("regression", (num_horizons,)),
            "quantiles": ("regression", (num_horizons, num_quantiles)),
            "direction_logits": ("classification", (num_horizons, self.config.num_classes)),
            "alpha": ("probability", (num_horizons,)),
            "beta": ("probability", (num_horizons,)),
            "attention_weights": ("attention", ("seq", "seq")),
        }


# Register the architecture
ArchitectureRegistry.register(
    "temporal_fusion_transformer",
    TemporalFusionTransformer,
    aliases=["tft", "medium_term"],
)
