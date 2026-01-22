"""Multi-timeframe trading model system.

This module provides trading systems that use multiple timeframes:

V1 (Original - Scalper):
- 5min/15min/30min timeframes
- Next-bar prediction (less effective)

V2 (Improved):
- 1H/4H timeframes with longer horizons
- Triple barrier labeling
- Enhanced features (time, patterns, cross-TF)
- Better accuracy through meaningful predictions
"""

from .mtf_model import (
    TimeframeConfig,
    MultiTimeframeModel,
    TimeframePrediction,
)
from .mtf_predictor import (
    MTFPredictor,
    AggregatedPrediction,
    SignalAlignment,
)
from .mtf_signals import (
    MTFSignalGenerator,
    ScalperSignal,
    SignalStrength,
    ScalperConfig,
    TradeAction,
)
from .labeling import (
    AdvancedLabeler,
    LabelingConfig,
    LabelMethod,
    create_triple_barrier_labels,
    create_multi_bar_labels,
)
from .enhanced_features import (
    EnhancedFeatureEngine,
    add_enhanced_features,
)
from .improved_model import (
    ImprovedModelConfig,
    ImprovedTimeframeModel,
    ImprovedMultiTimeframeModel,
)
# Lazy imports for torch-dependent modules (optional, not needed for API)
# Import these directly when needed:
#   from src.models.multi_timeframe.sequence_model import ...
#   from src.models.multi_timeframe.hybrid_ensemble import ...
from .mtf_ensemble import (
    MTFEnsembleConfig,
    MTFEnsemble,
    MTFPrediction,
)
from .stacking_meta_learner import (
    StackingConfig,
    StackingMetaLearner,
    StackingMetaFeatures,
)
from .shap_analyzer import (
    SHAPAnalyzer,
)

__all__ = [
    # Original model
    "TimeframeConfig",
    "MultiTimeframeModel",
    "TimeframePrediction",
    "MTFPredictor",
    "AggregatedPrediction",
    "SignalAlignment",
    "MTFSignalGenerator",
    "ScalperSignal",
    "SignalStrength",
    "ScalperConfig",
    "TradeAction",
    # Improved labeling
    "AdvancedLabeler",
    "LabelingConfig",
    "LabelMethod",
    "create_triple_barrier_labels",
    "create_multi_bar_labels",
    # Enhanced features
    "EnhancedFeatureEngine",
    "add_enhanced_features",
    # Improved model
    "ImprovedModelConfig",
    "ImprovedTimeframeModel",
    "ImprovedMultiTimeframeModel",
    # Sequence model (torch-dependent, import directly when needed)
    # Hybrid ensemble (torch-dependent, import directly when needed)
    # MTF Ensemble (3-timeframe weighted)
    "MTFEnsembleConfig",
    "MTFEnsemble",
    "MTFPrediction",
    # Stacking Meta-Learner
    "StackingConfig",
    "StackingMetaLearner",
    "StackingMetaFeatures",
    # SHAP Analysis
    "SHAPAnalyzer",
]
