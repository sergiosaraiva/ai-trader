"""N-BEATS + Transformer hybrid architecture.

Combines N-BEATS (Neural Basis Expansion Analysis for Time Series)
with Transformer attention for long-term forecasting.
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
    PositionalEncoding,
)

logger = logging.getLogger(__name__)


@dataclass
class NBEATSConfig(ArchitectureConfig):
    """Configuration for N-BEATS + Transformer architecture.

    Attributes:
        nbeats_stacks: Number of N-BEATS stacks.
        nbeats_blocks_per_stack: Blocks per stack.
        nbeats_hidden_layers: Hidden layers per block.
        nbeats_layer_width: Width of hidden layers.
        nbeats_expansion_coef: Expansion coefficient size.
        transformer_heads: Transformer attention heads.
        transformer_layers: Transformer encoder layers.
        transformer_dim: Transformer model dimension.
        transformer_ff_dim: Transformer feedforward dimension.
        transformer_dropout: Transformer dropout.
        fusion_method: How to combine N-BEATS and Transformer ('concat', 'add', 'gate').
        regime_classes: Classes for regime classification.
    """

    nbeats_stacks: int = 3
    nbeats_blocks_per_stack: int = 3
    nbeats_hidden_layers: int = 4
    nbeats_layer_width: int = 256
    nbeats_expansion_coef: int = 5
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_dim: int = 256
    transformer_ff_dim: int = 512
    transformer_dropout: float = 0.1
    fusion_method: str = "gate"
    regime_classes: List[str] = field(
        default_factory=lambda: ["trending_up", "trending_down", "ranging", "volatile"]
    )


class NBEATSBlock(nn.Module):
    """Single N-BEATS block with basis expansion."""

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        hidden_layers: int,
        hidden_size: int,
        backcast_length: int,
        forecast_length: int,
    ):
        super().__init__()

        self.input_size = input_size
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length

        # FC stack
        layers = []
        current_size = input_size

        for i in range(hidden_layers):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            current_size = hidden_size

        self.fc_stack = nn.Sequential(*layers)

        # Theta layers for backcast and forecast
        self.theta_b = nn.Linear(hidden_size, theta_size)
        self.theta_f = nn.Linear(hidden_size, theta_size)

        # Basis expansion layers
        self.basis_b = nn.Linear(theta_size, backcast_length, bias=False)
        self.basis_f = nn.Linear(theta_size, forecast_length, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (batch, input_size).

        Returns:
            Tuple of (backcast, forecast) tensors.
        """
        # FC stack
        hidden = self.fc_stack(x)

        # Theta computation
        theta_b = self.theta_b(hidden)
        theta_f = self.theta_f(hidden)

        # Basis expansion
        backcast = self.basis_b(theta_b)
        forecast = self.basis_f(theta_f)

        return backcast, forecast


class NBEATSStack(nn.Module):
    """N-BEATS stack containing multiple blocks."""

    def __init__(
        self,
        num_blocks: int,
        input_size: int,
        theta_size: int,
        hidden_layers: int,
        hidden_size: int,
        backcast_length: int,
        forecast_length: int,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            NBEATSBlock(
                input_size,
                theta_size,
                hidden_layers,
                hidden_size,
                backcast_length,
                forecast_length,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with residual stacking.

        Args:
            x: Input tensor (batch, seq_len, features).

        Returns:
            Tuple of (total_backcast, total_forecast).
        """
        # Flatten for blocks
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        residual = x_flat
        total_forecast = torch.zeros(batch_size, self.blocks[0].forecast_length, device=x.device)

        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            total_forecast = total_forecast + forecast

        # Reshape backcast to match input shape
        total_backcast = x_flat - residual

        return total_backcast, total_forecast


class NBEATSTransformer(BaseArchitecture):
    """N-BEATS + Transformer hybrid architecture.

    Combines the decomposition capabilities of N-BEATS with the attention
    mechanism of Transformers for powerful long-term forecasting.

    Architecture flow:
        Input (batch, seq, features)
        -> N-BEATS stacks (basis decomposition)
        -> Transformer encoder (attention on residuals)
        -> Fusion (combine both pathways)
        -> Output heads (price, regime, trend_strength)

    Example:
        ```python
        config = NBEATSConfig(
            input_dim=50,
            sequence_length=52,
            hidden_dim=256,
            prediction_horizons=[1, 2, 4],
        )
        model = NBEATSTransformer(config)
        outputs = model(x)
        ```
    """

    name = "nbeats_transformer"
    description = "N-BEATS + Transformer hybrid for long-term forecasting"
    supported_output_types = ["regression", "classification", "multi"]

    def __init__(self, config: Optional[Union[NBEATSConfig, Dict]] = None):
        """Initialize N-BEATS + Transformer.

        Args:
            config: Architecture configuration.
        """
        if config is None:
            config = NBEATSConfig()
        elif isinstance(config, dict):
            config = NBEATSConfig(**{
                k: v for k, v in config.items()
                if k in NBEATSConfig.__dataclass_fields__
            })

        super().__init__(config)
        self.config: NBEATSConfig = config

        # Build architecture
        self._build_nbeats()
        self._build_transformer()
        self._build_fusion()
        self._build_output_heads()

    def _build_nbeats(self) -> None:
        """Build N-BEATS pathway."""
        input_size = self.config.sequence_length * self.config.input_dim
        num_horizons = len(self.config.prediction_horizons)

        self.nbeats_stacks = nn.ModuleList([
            NBEATSStack(
                num_blocks=self.config.nbeats_blocks_per_stack,
                input_size=input_size,
                theta_size=self.config.nbeats_expansion_coef,
                hidden_layers=self.config.nbeats_hidden_layers,
                hidden_size=self.config.nbeats_layer_width,
                backcast_length=input_size,
                forecast_length=num_horizons,
            )
            for _ in range(self.config.nbeats_stacks)
        ])

        # Project N-BEATS output
        self.nbeats_projection = nn.Linear(num_horizons, self.config.hidden_dim)

    def _build_transformer(self) -> None:
        """Build Transformer pathway."""
        # Input projection
        self.transformer_input = nn.Linear(
            self.config.input_dim,
            self.config.transformer_dim,
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            self.config.transformer_dim,
            max_len=self.config.sequence_length * 2,
            dropout=self.config.transformer_dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.transformer_dim,
            nhead=self.config.transformer_heads,
            dim_feedforward=self.config.transformer_ff_dim,
            dropout=self.config.transformer_dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.transformer_layers,
        )

        # Project transformer output
        self.transformer_projection = nn.Linear(
            self.config.transformer_dim,
            self.config.hidden_dim,
        )

    def _build_fusion(self) -> None:
        """Build fusion layer."""
        hidden_dim = self.config.hidden_dim

        if self.config.fusion_method == "gate":
            # Gated fusion
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            self.fusion_output = nn.Linear(hidden_dim, hidden_dim)
        elif self.config.fusion_method == "concat":
            # Concatenation fusion
            self.fusion_output = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            # Addition fusion
            self.fusion_output = nn.Identity()

        self.fusion_norm = nn.LayerNorm(hidden_dim)

    def _build_output_heads(self) -> None:
        """Build output heads."""
        hidden_dim = self.config.hidden_dim
        num_horizons = len(self.config.prediction_horizons)
        num_regimes = len(self.config.regime_classes)

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        feature_dim = hidden_dim // 2

        # Price prediction head
        self.price_head = nn.Linear(feature_dim, num_horizons)

        # Regime classification head
        self.regime_head = nn.Linear(feature_dim, num_regimes)

        # Trend strength head (continuous 0-1)
        self.trend_head = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )

        # Direction head
        self.direction_head = nn.Linear(feature_dim, num_horizons * self.config.num_classes)

        # Confidence head (Beta distribution)
        self.confidence_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, num_horizons * 2),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim).

        Returns:
            Dictionary with predictions.
        """
        batch_size = x.size(0)
        num_horizons = len(self.config.prediction_horizons)

        # N-BEATS pathway
        nbeats_residual = x.view(batch_size, -1)
        nbeats_forecast = torch.zeros(
            batch_size, num_horizons, device=x.device
        )

        for stack in self.nbeats_stacks:
            backcast, forecast = stack(nbeats_residual.view(batch_size, self.config.sequence_length, -1))
            nbeats_residual = nbeats_residual - backcast
            nbeats_forecast = nbeats_forecast + forecast

        nbeats_features = self.nbeats_projection(nbeats_forecast)

        # Transformer pathway
        transformer_input = self.transformer_input(x)
        transformer_input = self.pos_encoder(transformer_input)
        transformer_out = self.transformer_encoder(transformer_input)
        transformer_features = self.transformer_projection(transformer_out[:, -1, :])

        # Fusion
        if self.config.fusion_method == "gate":
            combined = torch.cat([nbeats_features, transformer_features], dim=-1)
            gate = self.gate(combined)
            fused = gate * nbeats_features + (1 - gate) * transformer_features
            fused = self.fusion_output(fused)
        elif self.config.fusion_method == "concat":
            combined = torch.cat([nbeats_features, transformer_features], dim=-1)
            fused = self.fusion_output(combined)
        else:
            fused = nbeats_features + transformer_features

        fused = self.fusion_norm(fused)

        # Features for output heads
        features = self.feature_extractor(fused)

        # Outputs
        price_pred = self.price_head(features)
        regime_logits = self.regime_head(features)
        trend_strength = self.trend_head(features).squeeze(-1)

        direction_logits = self.direction_head(features)
        direction_logits = direction_logits.view(batch_size, num_horizons, self.config.num_classes)

        confidence_params = self.confidence_head(features)
        confidence_params = confidence_params.view(batch_size, num_horizons, 2)
        alpha = confidence_params[:, :, 0] + 1
        beta = confidence_params[:, :, 1] + 1

        return {
            "price": price_pred,
            "nbeats_forecast": nbeats_forecast,
            "direction_logits": direction_logits,
            "regime_logits": regime_logits,
            "trend_strength": trend_strength,
            "alpha": alpha,
            "beta": beta,
        }

    def get_output_info(self) -> Dict[str, Tuple[str, Tuple]]:
        """Get output information."""
        num_horizons = len(self.config.prediction_horizons)
        num_regimes = len(self.config.regime_classes)

        return {
            "price": ("regression", (num_horizons,)),
            "nbeats_forecast": ("regression", (num_horizons,)),
            "direction_logits": ("classification", (num_horizons, self.config.num_classes)),
            "regime_logits": ("classification", (num_regimes,)),
            "trend_strength": ("regression", (1,)),
            "alpha": ("probability", (num_horizons,)),
            "beta": ("probability", (num_horizons,)),
        }


# Register the architecture
ArchitectureRegistry.register(
    "nbeats_transformer",
    NBEATSTransformer,
    aliases=["nbeats", "long_term"],
)
