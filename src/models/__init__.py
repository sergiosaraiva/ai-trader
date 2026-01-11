"""Machine learning models for price prediction.

This module provides two approaches:

1. **Legacy Models** (backward compatibility):
   - ShortTermModel, MediumTermModel, LongTermModel
   - Fixed architectures per timeframe

2. **Plugin Architecture** (recommended for new development):
   - PluginModel: Wraps any architecture in BaseModel interface
   - create_model(): Factory for any architecture at any timeframe
   - Architectures registered in training.architectures

Example:
    ```python
    # Legacy approach
    from src.models import ShortTermModel
    model = ShortTermModel(config)

    # Plugin approach (recommended)
    from src.models import create_model
    model = create_model("cnn_lstm_attention", timeframe="hourly")

    # Or use architecture at any timeframe
    model = create_model("tft", timeframe="hourly")  # TFT at hourly!
    ```
"""

from .base import BaseModel, Prediction, ModelRegistry
from .technical import ShortTermModel, MediumTermModel, LongTermModel
from .technical.plugin_bridge import PluginModel, create_model
from .ensemble import EnsembleModel, TechnicalEnsemble

__all__ = [
    # Base
    "BaseModel",
    "Prediction",
    "ModelRegistry",
    # Legacy technical models
    "ShortTermModel",
    "MediumTermModel",
    "LongTermModel",
    # Plugin architecture (recommended)
    "PluginModel",
    "create_model",
    # Ensemble
    "EnsembleModel",
    "TechnicalEnsemble",
]
