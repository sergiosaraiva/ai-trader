"""Ensemble models for combining predictions.

This module provides:
- EnsembleModel: Base class for ensemble models
- TechnicalEnsemble: Ensemble for technical analysis models
- MetaModel: Meta-model for stacking ensemble
- EnsemblePredictor: Production-ready ensemble predictor
- DynamicWeightCalculator: Dynamic weight adjustment
- Loader utilities for discovering and loading trained models
"""

from .combiner import EnsembleModel, TechnicalEnsemble, EnsemblePrediction
from .meta_model import MetaModel
from .weights import (
    DynamicWeightCalculator,
    WeightConfig,
    MarketRegime,
    VolatilityLevel,
    TradeResult,
    detect_market_regime,
)
from .predictor import (
    EnsemblePredictor,
    EnsembleConfig,
    ModelPrediction,
    EnsemblePrediction as EnsemblePredictionNew,
)
from .loader import (
    discover_trained_models,
    get_latest_models,
    load_ensemble,
    load_single_model,
    get_model_info,
    validate_ensemble_models,
)

__all__ = [
    # Combiner
    "EnsembleModel",
    "TechnicalEnsemble",
    "EnsemblePrediction",
    # Meta-model
    "MetaModel",
    # Weights
    "DynamicWeightCalculator",
    "WeightConfig",
    "MarketRegime",
    "VolatilityLevel",
    "TradeResult",
    "detect_market_regime",
    # Predictor
    "EnsemblePredictor",
    "EnsembleConfig",
    "ModelPrediction",
    # Loader
    "discover_trained_models",
    "get_latest_models",
    "load_ensemble",
    "load_single_model",
    "get_model_info",
    "validate_ensemble_models",
]
