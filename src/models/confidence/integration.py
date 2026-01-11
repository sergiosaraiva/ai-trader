"""
Integration of confidence estimation with the trading pipeline.

This module provides the main interface for confidence-aware predictions
in the multi-timeframe trading system.

Key features:
1. Calibrated confidence scores
2. Position sizing based on confidence
3. Trade filtering based on minimum confidence
4. Integration with ensemble predictions
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .calibration import (
    TemperatureScaling,
    PlattScaling,
    IsotonicCalibration,
    CalibrationMetrics,
    select_best_calibrator,
)
from .uncertainty import (
    MCDropoutUncertainty,
    EnsembleUncertainty,
    ConfidenceEstimator,
    UncertaintyEstimate,
)


@dataclass
class TradingPrediction:
    """
    Complete prediction with confidence for trading.

    This is the final output of the prediction system,
    containing all information needed for trading decisions.
    """

    # Core prediction
    direction: int  # 1 (long/up), -1 (short/down), 0 (no trade)
    raw_probability: float  # Raw sigmoid output [0, 1]
    calibrated_probability: float  # Calibrated probability [0, 1]

    # Confidence measures
    confidence: float  # Calibrated confidence [0.5, 1.0]
    confidence_level: str  # 'very_high', 'high', 'moderate', 'low', 'skip'

    # Uncertainty decomposition
    epistemic_uncertainty: float  # Model uncertainty
    aleatoric_uncertainty: Optional[float] = None  # Data uncertainty

    # Trading signals
    position_size: float = 0.0  # Suggested position size [0, 1]
    should_trade: bool = False  # Whether to execute trade

    # Model contributions (for ensemble)
    model_predictions: Dict[str, float] = field(default_factory=dict)
    model_confidences: Dict[str, float] = field(default_factory=dict)

    # Metadata
    timestamp: Optional[str] = None
    symbol: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            'direction': self.direction,
            'raw_probability': self.raw_probability,
            'calibrated_probability': self.calibrated_probability,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level,
            'epistemic_uncertainty': self.epistemic_uncertainty,
            'aleatoric_uncertainty': self.aleatoric_uncertainty,
            'position_size': self.position_size,
            'should_trade': self.should_trade,
            'model_predictions': self.model_predictions,
            'model_confidences': self.model_confidences,
            'timestamp': self.timestamp,
            'symbol': self.symbol,
        }


class ConfidenceAwarePredictor:
    """
    Main interface for confidence-aware predictions.

    Integrates:
    1. Model predictions from multiple timeframes
    2. Calibration for accurate confidence
    3. Uncertainty estimation (MC Dropout + Ensemble)
    4. Position sizing based on confidence
    5. Trade filtering

    Usage:
        predictor = ConfidenceAwarePredictor(
            models={'short': short_model, 'medium': medium_model, 'long': long_model},
            model_weights={'short': 0.5, 'medium': 0.3, 'long': 0.2}
        )

        # Calibrate on validation data
        predictor.calibrate(val_logits, val_labels)

        # Make prediction
        prediction = predictor.predict(features)

        if prediction.should_trade:
            execute_trade(
                direction=prediction.direction,
                size=prediction.position_size * base_position
            )
    """

    def __init__(
        self,
        models: Optional[Dict[str, nn.Module]] = None,
        model_weights: Optional[Dict[str, float]] = None,
        min_confidence: float = 0.60,
        use_calibration: bool = True,
        use_mc_dropout: bool = True,
        mc_samples: int = 30,
        calibration_method: str = 'auto',  # 'temperature', 'platt', 'isotonic', 'auto'
        confidence_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize confidence-aware predictor.

        Args:
            models: Dictionary of models by timeframe
            model_weights: Weights for ensemble combination
            min_confidence: Minimum confidence to trade
            use_calibration: Whether to calibrate outputs
            use_mc_dropout: Whether to use MC Dropout
            mc_samples: Number of MC Dropout samples
            calibration_method: Calibration method to use
            confidence_thresholds: Custom confidence thresholds
        """
        self.models = models or {}
        self.model_weights = model_weights or {}
        self.min_confidence = min_confidence
        self.use_calibration = use_calibration
        self.use_mc_dropout = use_mc_dropout
        self.mc_samples = mc_samples
        self.calibration_method = calibration_method

        # Initialize components
        self.calibrators: Dict[str, Any] = {}
        self.confidence_estimator = ConfidenceEstimator(
            use_mc_dropout=use_mc_dropout,
            use_ensemble=True,
            mc_samples=mc_samples,
            confidence_thresholds=confidence_thresholds
        )

        self.ensemble_uncertainty = EnsembleUncertainty()

        # Track calibration status
        self._calibrated = False

    def add_model(
        self,
        name: str,
        model: nn.Module,
        weight: float = 1.0
    ) -> None:
        """Add a model to the ensemble."""
        self.models[name] = model
        self.model_weights[name] = weight

    def calibrate(
        self,
        val_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        verbose: bool = True
    ) -> Dict[str, CalibrationMetrics]:
        """
        Calibrate all models on validation data.

        Args:
            val_data: Dictionary of {model_name: (logits, labels)}
            verbose: Whether to print calibration results

        Returns:
            Dictionary of calibration metrics per model
        """
        metrics = {}

        for name, (logits, labels) in val_data.items():
            if name not in self.models:
                continue

            if isinstance(logits, torch.Tensor):
                logits_np = logits.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
            else:
                logits_np = logits
                labels_np = labels

            if self.calibration_method == 'auto':
                calibrator, method, cal_metrics = select_best_calibrator(
                    logits_np, labels_np
                )
                self.calibrators[name] = (calibrator, method)
            elif self.calibration_method == 'temperature':
                calibrator = TemperatureScaling()
                calibrator.fit(torch.tensor(logits_np), torch.tensor(labels_np))
                self.calibrators[name] = (calibrator, 'temperature')
            elif self.calibration_method == 'platt':
                calibrator = PlattScaling()
                calibrator.fit(logits_np, labels_np)
                self.calibrators[name] = (calibrator, 'platt')
            elif self.calibration_method == 'isotonic':
                probs = 1 / (1 + np.exp(-logits_np))
                calibrator = IsotonicCalibration()
                calibrator.fit(probs, labels_np)
                self.calibrators[name] = (calibrator, 'isotonic')

            # Compute final metrics
            if self.calibration_method == 'auto':
                metrics[name] = cal_metrics
            else:
                calibrator, method = self.calibrators[name]
                if method == 'temperature':
                    cal_probs = calibrator(torch.tensor(logits_np)).detach().numpy()
                elif method == 'platt':
                    cal_probs = calibrator.predict_proba(logits_np)
                else:
                    probs = 1 / (1 + np.exp(-logits_np))
                    cal_probs = calibrator.predict(probs)

                cal_preds = (cal_probs >= 0.5).astype(int)
                cal_conf = np.where(cal_preds == 1, cal_probs, 1 - cal_probs)
                metrics[name] = CalibrationMetrics.compute(
                    cal_conf, cal_preds, labels_np
                )

            if verbose:
                print(f"Calibration for {name}: {metrics[name]}")

        self._calibrated = True
        return metrics

    def _get_calibrated_probability(
        self,
        name: str,
        logits: torch.Tensor
    ) -> np.ndarray:
        """Get calibrated probability for a model's output."""
        if not self.use_calibration or name not in self.calibrators:
            return torch.sigmoid(logits).detach().cpu().numpy()

        calibrator, method = self.calibrators[name]

        if method == 'temperature':
            return calibrator(logits).detach().cpu().numpy()
        elif method == 'platt':
            logits_np = logits.detach().cpu().numpy()
            return calibrator.predict_proba(logits_np)
        else:  # isotonic
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            return calibrator.predict(probs)

    def predict_single(
        self,
        model_name: str,
        features: torch.Tensor
    ) -> TradingPrediction:
        """
        Make prediction with a single model.

        Args:
            model_name: Name of the model to use
            features: Input features

        Returns:
            TradingPrediction with confidence
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]
        model.eval()

        # Get raw prediction
        with torch.no_grad():
            logits = model(features)

        raw_prob = torch.sigmoid(logits).item()

        # Get calibrated probability
        cal_prob = self._get_calibrated_probability(model_name, logits)
        if isinstance(cal_prob, np.ndarray):
            cal_prob = cal_prob.item()

        # Estimate uncertainty
        if self.use_mc_dropout:
            mc_estimator = MCDropoutUncertainty(n_samples=self.mc_samples)
            uncertainty = mc_estimator.estimate(model, features)
            epistemic = uncertainty.epistemic.item() if isinstance(
                uncertainty.epistemic, np.ndarray
            ) else uncertainty.epistemic
        else:
            epistemic = 0.0

        # Compute confidence
        direction = 1 if cal_prob >= 0.5 else -1
        base_confidence = cal_prob if direction == 1 else (1 - cal_prob)

        # Adjust for uncertainty
        confidence = base_confidence - min(epistemic, 0.5)
        confidence = max(0.5, min(1.0, confidence))

        # Determine confidence level and position size
        signal = self.confidence_estimator.get_trading_signal(
            UncertaintyEstimate(
                prediction=np.array([1 if direction == 1 else 0]),
                confidence=np.array([confidence]),
                epistemic=np.array([epistemic])
            )
        )

        position_size = signal['position_size'][0]
        confidence_level = signal['confidence_level'][0]
        should_trade = confidence >= self.min_confidence

        if not should_trade:
            direction = 0
            position_size = 0.0

        return TradingPrediction(
            direction=direction,
            raw_probability=raw_prob,
            calibrated_probability=cal_prob,
            confidence=confidence,
            confidence_level=confidence_level,
            epistemic_uncertainty=epistemic,
            position_size=position_size,
            should_trade=should_trade,
            model_predictions={model_name: raw_prob},
            model_confidences={model_name: confidence}
        )

    def predict_ensemble(
        self,
        features: Dict[str, torch.Tensor],
        dynamic_weights: Optional[Dict[str, float]] = None
    ) -> TradingPrediction:
        """
        Make prediction using all models in ensemble.

        Args:
            features: Dictionary of {model_name: features} for each model
            dynamic_weights: Optional dynamic weights (e.g., based on regime)

        Returns:
            TradingPrediction with combined confidence
        """
        weights = dynamic_weights or self.model_weights

        model_probs = {}
        model_confidences = {}
        epistemic_values = []

        for name, model in self.models.items():
            if name not in features:
                continue

            model.eval()

            with torch.no_grad():
                logits = model(features[name])

            raw_prob = torch.sigmoid(logits).item()
            cal_prob = self._get_calibrated_probability(name, logits)
            if isinstance(cal_prob, np.ndarray):
                cal_prob = cal_prob.item()

            model_probs[name] = cal_prob

            # Estimate uncertainty per model
            if self.use_mc_dropout:
                mc_estimator = MCDropoutUncertainty(n_samples=self.mc_samples)
                uncertainty = mc_estimator.estimate(model, features[name])
                epistemic = uncertainty.epistemic.item() if isinstance(
                    uncertainty.epistemic, np.ndarray
                ) else uncertainty.epistemic
            else:
                epistemic = 0.0

            epistemic_values.append(epistemic)

            # Per-model confidence
            direction = 1 if cal_prob >= 0.5 else -1
            conf = cal_prob if direction == 1 else (1 - cal_prob)
            model_confidences[name] = conf

        # Weighted ensemble
        total_weight = sum(weights.get(name, 1.0) for name in model_probs)
        ensemble_prob = sum(
            model_probs[name] * weights.get(name, 1.0) / total_weight
            for name in model_probs
        )

        # Ensemble uncertainty (disagreement)
        prob_values = list(model_probs.values())
        ensemble_disagreement = np.std(prob_values)

        # Mean epistemic uncertainty
        mean_epistemic = np.mean(epistemic_values) if epistemic_values else 0.0

        # Total epistemic = sqrt(disagreement^2 + mean_mc^2)
        total_epistemic = np.sqrt(ensemble_disagreement ** 2 + mean_epistemic ** 2)

        # Final prediction
        direction = 1 if ensemble_prob >= 0.5 else -1
        base_confidence = ensemble_prob if direction == 1 else (1 - ensemble_prob)

        # Adjust for total uncertainty
        confidence = base_confidence - min(total_epistemic, 0.5)
        confidence = max(0.5, min(1.0, confidence))

        # Trading signal
        signal = self.confidence_estimator.get_trading_signal(
            UncertaintyEstimate(
                prediction=np.array([1 if direction == 1 else 0]),
                confidence=np.array([confidence]),
                epistemic=np.array([total_epistemic])
            )
        )

        position_size = signal['position_size'][0]
        confidence_level = signal['confidence_level'][0]
        should_trade = confidence >= self.min_confidence

        if not should_trade:
            direction = 0
            position_size = 0.0

        return TradingPrediction(
            direction=direction,
            raw_probability=ensemble_prob,
            calibrated_probability=ensemble_prob,  # Already calibrated per model
            confidence=confidence,
            confidence_level=confidence_level,
            epistemic_uncertainty=total_epistemic,
            position_size=position_size,
            should_trade=should_trade,
            model_predictions=model_probs,
            model_confidences=model_confidences
        )

    def predict(
        self,
        features: Dict[str, torch.Tensor],
        symbol: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> TradingPrediction:
        """
        Main prediction interface.

        Args:
            features: Input features per model
            symbol: Trading symbol
            timestamp: Prediction timestamp

        Returns:
            TradingPrediction with full confidence analysis
        """
        prediction = self.predict_ensemble(features)
        prediction.symbol = symbol
        prediction.timestamp = timestamp
        return prediction


def create_confidence_aware_predictor(
    short_term_model: nn.Module,
    medium_term_model: nn.Module,
    long_term_model: nn.Module,
    profile: str = 'trader',
    min_confidence: float = 0.60
) -> ConfidenceAwarePredictor:
    """
    Factory function to create predictor with standard configuration.

    Args:
        short_term_model: Short-term prediction model
        medium_term_model: Medium-term prediction model
        long_term_model: Long-term prediction model
        profile: Trading profile ('scalper', 'trader', 'investor')
        min_confidence: Minimum confidence threshold

    Returns:
        Configured ConfidenceAwarePredictor
    """
    # Default weights by profile
    weight_profiles = {
        'scalper': {'short': 0.60, 'medium': 0.30, 'long': 0.10},
        'trader': {'short': 0.50, 'medium': 0.35, 'long': 0.15},
        'investor': {'short': 0.20, 'medium': 0.35, 'long': 0.45},
    }

    weights = weight_profiles.get(profile, weight_profiles['trader'])

    predictor = ConfidenceAwarePredictor(
        models={
            'short': short_term_model,
            'medium': medium_term_model,
            'long': long_term_model,
        },
        model_weights=weights,
        min_confidence=min_confidence,
        use_calibration=True,
        use_mc_dropout=True,
        mc_samples=30
    )

    return predictor
