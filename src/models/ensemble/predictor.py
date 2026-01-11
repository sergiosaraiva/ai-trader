"""Production-ready ensemble predictor.

Provides end-to-end prediction pipeline combining multiple models
with dynamic weighting and confidence aggregation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .weights import (
    DynamicWeightCalculator,
    MarketRegime,
    VolatilityLevel,
    WeightConfig,
    TradeResult,
    detect_market_regime,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Prediction from a single model.

    Attributes:
        model_name: Name of the model.
        direction: Predicted direction (0=down, 1=up).
        direction_probs: Probabilities for each class.
        confidence: Model confidence (from Beta distribution).
        alpha: Beta distribution alpha parameter.
        beta: Beta distribution beta parameter.
        price_prediction: Price prediction (if available).
        raw_output: Full model output dictionary.
    """

    model_name: str
    direction: int
    direction_probs: np.ndarray
    confidence: float
    alpha: float
    beta: float
    price_prediction: Optional[float] = None
    raw_output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsemblePrediction:
    """Combined prediction from the ensemble.

    Attributes:
        timestamp: Prediction timestamp.
        symbol: Trading symbol.
        direction: Final direction prediction (0=down, 1=up).
        direction_probability: Probability of predicted direction.
        confidence: Ensemble confidence (confidence-weighted).
        agreement_score: How much models agree (0-1).
        should_trade: Whether confidence meets trading threshold.
        position_size_factor: Suggested position size (0-1).
        component_predictions: Individual model predictions.
        component_weights: Weights used for combination.
        market_regime: Detected market regime.
        volatility_level: Detected volatility level.
    """

    timestamp: datetime
    symbol: str
    direction: int
    direction_probability: float
    confidence: float
    agreement_score: float
    should_trade: bool
    position_size_factor: float
    component_predictions: Dict[str, ModelPrediction]
    component_weights: Dict[str, float]
    market_regime: str
    volatility_level: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "direction": self.direction,
            "direction_probability": self.direction_probability,
            "confidence": self.confidence,
            "agreement_score": self.agreement_score,
            "should_trade": self.should_trade,
            "position_size_factor": self.position_size_factor,
            "component_weights": self.component_weights,
            "market_regime": self.market_regime,
            "volatility_level": self.volatility_level,
            "component_predictions": {
                name: {
                    "direction": pred.direction,
                    "confidence": pred.confidence,
                }
                for name, pred in self.component_predictions.items()
            },
        }


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble predictor.

    Attributes:
        min_confidence: Minimum confidence to generate trade signal.
        min_agreement: Minimum agreement score to generate trade signal.
        disagreement_penalty: How much to reduce confidence when models disagree.
        confidence_position_scaling: Scale position size by confidence.
        use_dynamic_weights: Whether to adjust weights dynamically.
        weight_config: Configuration for dynamic weight calculator.
    """

    min_confidence: float = 0.60
    min_agreement: float = 0.50
    disagreement_penalty: float = 0.2
    confidence_position_scaling: bool = True
    use_dynamic_weights: bool = True
    weight_config: Optional[WeightConfig] = None

    def __post_init__(self):
        if self.weight_config is None:
            self.weight_config = WeightConfig()


class EnsemblePredictor:
    """Production-ready ensemble predictor for trading.

    Combines multiple trained models into a unified prediction system
    with dynamic weighting, confidence aggregation, and trading signals.

    Example:
        ```python
        # Load trained models
        predictor = EnsemblePredictor.from_trained_models(
            model_paths={
                "short_term": "models/trained/short_term_EURUSD_...",
                "medium_term": "models/trained/medium_term_EURUSD_...",
                "long_term": "models/trained/long_term_EURUSD_...",
            },
            device="cuda",
        )

        # Make prediction
        prediction = predictor.predict(
            features={
                "short_term": short_term_features,
                "medium_term": medium_term_features,
                "long_term": long_term_features,
            },
            symbol="EURUSD",
            prices=recent_prices,
        )

        if prediction.should_trade:
            size = base_position * prediction.position_size_factor
            if prediction.direction == 1:
                execute_buy(size)
            else:
                execute_sell(size)
        ```
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        config: Optional[EnsembleConfig] = None,
        device: str = "cpu",
    ):
        """Initialize ensemble predictor.

        Args:
            models: Dictionary mapping model names to PyTorch models.
            config: Ensemble configuration.
            device: Device for inference.
        """
        self.models = models
        self.config = config or EnsembleConfig()
        self.device = torch.device(device)

        # Move models to device and set to eval mode
        for model in self.models.values():
            model.to(self.device)
            model.eval()

        # Initialize weight calculator
        self.weight_calculator = DynamicWeightCalculator(
            self.config.weight_config
        )

        # Current weights
        self._current_weights: Dict[str, float] = {
            name: 1.0 / len(models) for name in models
        }

        # Initialize with base weights if available
        if self.config.weight_config:
            self._current_weights = self.config.weight_config.base_weights.copy()
            total = sum(self._current_weights.values())
            self._current_weights = {
                k: v / total for k, v in self._current_weights.items()
            }

        logger.info(
            f"Initialized EnsemblePredictor with {len(models)} models: "
            f"{list(models.keys())}"
        )

    @classmethod
    def from_trained_models(
        cls,
        model_paths: Dict[str, Union[str, Path]],
        config: Optional[EnsembleConfig] = None,
        device: str = "cpu",
    ) -> "EnsemblePredictor":
        """Create predictor from saved model paths.

        Args:
            model_paths: Dictionary mapping model names to save paths.
            config: Ensemble configuration.
            device: Device for inference.

        Returns:
            Configured EnsemblePredictor.
        """
        from src.training.trainer import Trainer

        models = {}
        for name, path in model_paths.items():
            logger.info(f"Loading model '{name}' from {path}")
            trainer = Trainer.load(path, device=device)
            models[name] = trainer.model

        return cls(models=models, config=config, device=device)

    def predict(
        self,
        features: Dict[str, torch.Tensor],
        symbol: str = "",
        prices: Optional[np.ndarray] = None,
    ) -> EnsemblePrediction:
        """Make ensemble prediction.

        Args:
            features: Dictionary mapping model names to input tensors.
                Each tensor should be shape (batch=1, seq_len, n_features).
            symbol: Trading symbol.
            prices: Recent prices for regime detection.

        Returns:
            EnsemblePrediction with combined result.
        """
        # Detect market conditions if prices provided
        if prices is not None and len(prices) > 20:
            regime, vol_level = detect_market_regime(prices)
        else:
            regime = MarketRegime.UNKNOWN
            vol_level = VolatilityLevel.NORMAL

        # Update weights if using dynamic weighting
        if self.config.use_dynamic_weights:
            self._current_weights = self.weight_calculator.calculate_weights(
                market_regime=regime,
                volatility_level=vol_level,
            )

        # Get predictions from each model
        component_predictions = {}
        for name, model in self.models.items():
            if name not in features:
                logger.warning(f"No features provided for model '{name}'")
                continue

            pred = self._get_model_prediction(
                model=model,
                features=features[name],
                model_name=name,
            )
            component_predictions[name] = pred

        # Combine predictions
        ensemble_pred = self._combine_predictions(
            component_predictions=component_predictions,
            symbol=symbol,
            regime=regime,
            vol_level=vol_level,
        )

        return ensemble_pred

    def _get_model_prediction(
        self,
        model: nn.Module,
        features: torch.Tensor,
        model_name: str,
    ) -> ModelPrediction:
        """Get prediction from a single model.

        Args:
            model: PyTorch model.
            features: Input features tensor.
            model_name: Name of the model.

        Returns:
            ModelPrediction with model output.
        """
        # Ensure correct device and shape
        if not isinstance(features, torch.Tensor):
            features = torch.FloatTensor(features)
        features = features.to(self.device)

        if features.dim() == 2:
            features = features.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(features)

        # Extract predictions
        direction_logits = output.get("direction_logits")
        alpha = output.get("alpha")
        beta = output.get("beta")
        price = output.get("price")

        # Process direction
        if direction_logits is not None:
            # Handle multi-horizon output (take first horizon)
            if direction_logits.dim() == 3:
                direction_logits = direction_logits[:, 0, :]  # First horizon

            direction_probs = F.softmax(direction_logits, dim=-1).cpu().numpy()[0]
            direction = int(np.argmax(direction_probs))

            # For binary, map class 0->down, class 1/2->up
            if len(direction_probs) > 2:
                # 3-class: combine neutral and up as "up"
                direction = 0 if direction == 0 else 1
        else:
            direction_probs = np.array([0.5, 0.5])
            direction = 1

        # Process confidence from Beta distribution
        if alpha is not None and beta is not None:
            # Take first horizon if multi-horizon
            if alpha.dim() > 1:
                alpha_val = alpha[0, 0].item()
                beta_val = beta[0, 0].item()
            else:
                alpha_val = alpha[0].item()
                beta_val = beta[0].item()

            # Confidence from concentration
            concentration = alpha_val + beta_val
            # Map concentration to confidence (higher = more confident)
            confidence = min(1.0, (concentration - 2) / 10)  # Normalize
            confidence = max(0.0, confidence)
        else:
            alpha_val = 1.0
            beta_val = 1.0
            confidence = 0.5

        # Price prediction
        price_val = None
        if price is not None:
            price_val = price[0, 0].item() if price.dim() > 1 else price[0].item()

        return ModelPrediction(
            model_name=model_name,
            direction=direction,
            direction_probs=direction_probs,
            confidence=confidence,
            alpha=alpha_val,
            beta=beta_val,
            price_prediction=price_val,
            raw_output={k: v.cpu().numpy() for k, v in output.items()},
        )

    def _combine_predictions(
        self,
        component_predictions: Dict[str, ModelPrediction],
        symbol: str,
        regime: MarketRegime,
        vol_level: VolatilityLevel,
    ) -> EnsemblePrediction:
        """Combine component predictions into ensemble prediction.

        Args:
            component_predictions: Predictions from each model.
            symbol: Trading symbol.
            regime: Market regime.
            vol_level: Volatility level.

        Returns:
            Combined EnsemblePrediction.
        """
        if not component_predictions:
            return self._empty_prediction(symbol, regime, vol_level)

        # Calculate weighted direction probability
        weighted_up_prob = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0

        for name, pred in component_predictions.items():
            weight = self._current_weights.get(name, 1.0 / len(component_predictions))

            # Direction probability (probability of "up")
            if len(pred.direction_probs) > 2:
                # 3-class: sum of neutral and up
                up_prob = pred.direction_probs[1] + pred.direction_probs[2]
            else:
                up_prob = pred.direction_probs[1] if len(pred.direction_probs) > 1 else 0.5

            weighted_up_prob += weight * up_prob
            weighted_confidence += weight * pred.confidence
            total_weight += weight

        # Normalize
        if total_weight > 0:
            weighted_up_prob /= total_weight
            weighted_confidence /= total_weight

        # Final direction
        direction = 1 if weighted_up_prob >= 0.5 else 0
        direction_probability = weighted_up_prob if direction == 1 else (1 - weighted_up_prob)

        # Agreement score
        directions = [pred.direction for pred in component_predictions.values()]
        agreement = directions.count(direction) / len(directions) if directions else 0.0

        # Apply disagreement penalty
        if agreement < 1.0:
            confidence_penalty = (1.0 - agreement) * self.config.disagreement_penalty
            weighted_confidence = max(0.0, weighted_confidence - confidence_penalty)

        # Determine if should trade
        should_trade = (
            weighted_confidence >= self.config.min_confidence
            and agreement >= self.config.min_agreement
        )

        # Position size factor
        if self.config.confidence_position_scaling and should_trade:
            # Scale from 0.25 at min_confidence to 1.0 at max confidence
            conf_range = 1.0 - self.config.min_confidence
            if conf_range > 0:
                position_factor = 0.25 + 0.75 * (
                    (weighted_confidence - self.config.min_confidence) / conf_range
                )
            else:
                position_factor = 1.0
            position_factor = min(1.0, max(0.25, position_factor))
        elif should_trade:
            position_factor = 1.0
        else:
            position_factor = 0.0

        return EnsemblePrediction(
            timestamp=datetime.now(),
            symbol=symbol,
            direction=direction,
            direction_probability=direction_probability,
            confidence=weighted_confidence,
            agreement_score=agreement,
            should_trade=should_trade,
            position_size_factor=position_factor,
            component_predictions=component_predictions,
            component_weights=self._current_weights.copy(),
            market_regime=regime.value,
            volatility_level=vol_level.value,
        )

    def _empty_prediction(
        self,
        symbol: str,
        regime: MarketRegime,
        vol_level: VolatilityLevel,
    ) -> EnsemblePrediction:
        """Create empty prediction when no models available."""
        return EnsemblePrediction(
            timestamp=datetime.now(),
            symbol=symbol,
            direction=1,
            direction_probability=0.5,
            confidence=0.0,
            agreement_score=0.0,
            should_trade=False,
            position_size_factor=0.0,
            component_predictions={},
            component_weights={},
            market_regime=regime.value,
            volatility_level=vol_level.value,
        )

    def update_performance(self, result: TradeResult) -> None:
        """Update performance tracking with trade result.

        Args:
            result: Result of a trade.
        """
        self.weight_calculator.add_trade_result(result)

    def get_current_weights(self) -> Dict[str, float]:
        """Get current model weights.

        Returns:
            Dictionary of model name to weight.
        """
        return self._current_weights.copy()

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Manually set model weights.

        Args:
            weights: Dictionary of model name to weight.
        """
        total = sum(weights.values())
        self._current_weights = {k: v / total for k, v in weights.items()}

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models.

        Returns:
            Dictionary with model information.
        """
        info = {
            "num_models": len(self.models),
            "model_names": list(self.models.keys()),
            "current_weights": self._current_weights,
            "device": str(self.device),
            "config": {
                "min_confidence": self.config.min_confidence,
                "min_agreement": self.config.min_agreement,
                "use_dynamic_weights": self.config.use_dynamic_weights,
            },
        }

        # Add parameter counts
        for name, model in self.models.items():
            info[f"{name}_parameters"] = sum(
                p.numel() for p in model.parameters()
            )

        return info
