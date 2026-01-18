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

# Lazy imports to avoid torch dependency at package initialization
# Import these directly when needed:
#   from src.models.base import BaseModel, Prediction, ModelRegistry
#   from src.models.technical import ShortTermModel, MediumTermModel, LongTermModel
#   from src.models.technical.plugin_bridge import PluginModel, create_model
#   from src.models.ensemble import EnsembleModel, TechnicalEnsemble

__all__ = [
    # Base (import from src.models.base)
    "BaseModel",
    "Prediction",
    "ModelRegistry",
    # Legacy technical models (import from src.models.technical)
    "ShortTermModel",
    "MediumTermModel",
    "LongTermModel",
    # Plugin architecture (import from src.models.technical.plugin_bridge)
    "PluginModel",
    "create_model",
    # Ensemble (import from src.models.ensemble)
    "EnsembleModel",
    "TechnicalEnsemble",
]


def __getattr__(name):
    """Lazy import to avoid torch dependency."""
    if name in ("BaseModel", "Prediction", "ModelRegistry"):
        from .base import BaseModel, Prediction, ModelRegistry
        return {"BaseModel": BaseModel, "Prediction": Prediction, "ModelRegistry": ModelRegistry}[name]
    elif name in ("ShortTermModel", "MediumTermModel", "LongTermModel"):
        from .technical import ShortTermModel, MediumTermModel, LongTermModel
        return {"ShortTermModel": ShortTermModel, "MediumTermModel": MediumTermModel, "LongTermModel": LongTermModel}[name]
    elif name in ("PluginModel", "create_model"):
        from .technical.plugin_bridge import PluginModel, create_model
        return {"PluginModel": PluginModel, "create_model": create_model}[name]
    elif name in ("EnsembleModel", "TechnicalEnsemble"):
        from .ensemble import EnsembleModel, TechnicalEnsemble
        return {"EnsembleModel": EnsembleModel, "TechnicalEnsemble": TechnicalEnsemble}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
