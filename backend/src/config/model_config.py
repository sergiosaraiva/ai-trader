"""Model configuration classes."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ModelConfig:
    """Base configuration for all models."""

    name: str
    version: str = "1.0.0"

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    early_stopping_patience: int = 15
    gradient_clip: float = 1.0

    # Data
    sequence_length: int = 100
    prediction_horizon: List[int] = field(default_factory=lambda: [1])
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # Regularization
    dropout: float = 0.3
    label_smoothing: float = 0.1

    # Device
    device: str = "auto"  # auto, cpu, cuda, mps


@dataclass
class ShortTermConfig(ModelConfig):
    """Configuration for Short-Term model (1H-4H candles)."""

    name: str = "short_term"
    sequence_length: int = 168  # 7 days of hourly data
    prediction_horizon: List[int] = field(default_factory=lambda: [1, 4, 12, 24])

    # CNN Configuration
    cnn_filters: List[int] = field(default_factory=lambda: [64, 128, 256])
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    cnn_dropout: float = 0.2

    # LSTM Configuration
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_bidirectional: bool = True

    # Attention Configuration
    attention_heads: int = 8
    attention_dim: int = 256


@dataclass
class MediumTermConfig(ModelConfig):
    """Configuration for Medium-Term model (Daily candles)."""

    name: str = "medium_term"
    sequence_length: int = 90  # 90 days history
    prediction_horizon: List[int] = field(default_factory=lambda: [1, 3, 5, 7])

    # TFT Configuration
    hidden_size: int = 256
    attention_heads: int = 4
    hidden_continuous_size: int = 64
    num_lstm_layers: int = 2

    # Quantile outputs for uncertainty
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 0.9])


@dataclass
class LongTermConfig(ModelConfig):
    """Configuration for Long-Term model (Weekly candles)."""

    name: str = "long_term"
    sequence_length: int = 52  # 52 weeks history
    prediction_horizon: List[int] = field(default_factory=lambda: [1, 2, 4])

    # N-BEATS Configuration
    nbeats_stacks: int = 30
    nbeats_blocks: int = 3
    nbeats_layers: int = 4
    nbeats_width: int = 256

    # Transformer Configuration
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_dim: int = 512

    # Regime classification
    regime_classes: List[str] = field(
        default_factory=lambda: ["trending_up", "trending_down", "ranging", "volatile"]
    )


@dataclass
class EnsembleConfig:
    """Configuration for ensemble model."""

    name: str = "technical_ensemble"
    version: str = "1.0.0"

    # Component models
    short_term_weight: float = 0.3
    medium_term_weight: float = 0.4
    long_term_weight: float = 0.3

    # Dynamic weighting
    use_dynamic_weights: bool = True
    weight_lookback_period: int = 30  # Days to calculate performance

    # Combination method: weighted_avg, stacking, attention
    combination_method: str = "stacking"

    # Meta-model configuration (for stacking)
    meta_hidden_size: int = 128
    meta_layers: int = 2

    # Confidence thresholds
    min_confidence: float = 0.6
    buy_threshold: float = 0.3
    sell_threshold: float = -0.3
