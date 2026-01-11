"""Neural network architectures for trading models.

This module provides a plugin-based architecture system where any architecture
can be used at any timeframe. Architectures are registered and created via
the ArchitectureRegistry.

Example:
    ```python
    from src.training.architectures import ArchitectureRegistry, CNNLSTMAttention

    # Register a custom architecture
    ArchitectureRegistry.register("my_arch", MyArchitecture)

    # Create architecture from registry
    model = ArchitectureRegistry.create(
        "cnn_lstm_attention",
        input_dim=50,
        sequence_length=168,
        output_dim=4,
    )

    # List available architectures
    print(ArchitectureRegistry.available())
    ```
"""

from .base import (
    ArchitectureConfig,
    BaseArchitecture,
    ArchitectureRegistry,
    PositionalEncoding,
    GatedResidualNetwork,
    MultiHeadAttention,
)
from .cnn_lstm import CNNLSTMAttention
from .tft import TemporalFusionTransformer
from .nbeats import NBEATSTransformer

__all__ = [
    "ArchitectureConfig",
    "BaseArchitecture",
    "ArchitectureRegistry",
    "PositionalEncoding",
    "GatedResidualNetwork",
    "MultiHeadAttention",
    "CNNLSTMAttention",
    "TemporalFusionTransformer",
    "NBEATSTransformer",
]
