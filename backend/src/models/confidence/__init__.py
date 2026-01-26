# Confidence estimation and calibration module

# Calibration methods (post-hoc)
from .calibration import (
    TemperatureScaling,
    PlattScaling,
    IsotonicCalibration,
    CalibrationMetrics,
)

# Uncertainty estimation (inference-time)
from .uncertainty import (
    MCDropoutUncertainty,
    EnsembleUncertainty,
    ConfidenceEstimator,
)

# Integration with trading pipeline
from .integration import ConfidenceAwarePredictor

# Learned uncertainty (model architecture) - THE BEST APPROACH
from .learned_uncertainty import (
    # Output structures
    GaussianPrediction,
    BetaPrediction,
    DirichletPrediction,
    QuantilePrediction,
    # Output layers
    GaussianOutputLayer,
    BetaOutputLayer,
    DirichletOutputLayer,
    QuantileOutputLayer,
    # Loss functions
    GaussianNLLLoss,
    BetaNLLLoss,
    EvidentialLoss,
    QuantileLoss,
    # Complete model
    TradingModelWithUncertainty,
)

__all__ = [
    # Calibration
    'TemperatureScaling',
    'PlattScaling',
    'IsotonicCalibration',
    'CalibrationMetrics',
    # Uncertainty
    'MCDropoutUncertainty',
    'EnsembleUncertainty',
    'ConfidenceEstimator',
    # Integration
    'ConfidenceAwarePredictor',
    # Learned uncertainty (recommended)
    'GaussianPrediction',
    'BetaPrediction',
    'DirichletPrediction',
    'QuantilePrediction',
    'GaussianOutputLayer',
    'BetaOutputLayer',
    'DirichletOutputLayer',
    'QuantileOutputLayer',
    'GaussianNLLLoss',
    'BetaNLLLoss',
    'EvidentialLoss',
    'QuantileLoss',
    'TradingModelWithUncertainty',
]
