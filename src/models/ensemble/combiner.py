"""Ensemble combiner for multiple model predictions."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np

from ..base import BaseModel, Prediction


@dataclass
class EnsemblePrediction(Prediction):
    """Extended prediction with ensemble-specific fields."""

    component_predictions: Dict[str, Prediction] = None
    component_weights: Dict[str, float] = None
    agreement_score: float = 0.0
    market_regime: str = "unknown"

    def __post_init__(self):
        if self.component_predictions is None:
            self.component_predictions = {}
        if self.component_weights is None:
            self.component_weights = {}


class EnsembleModel(BaseModel):
    """Base ensemble model for combining multiple predictions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ensemble model."""
        super().__init__(config)
        self.component_models: Dict[str, BaseModel] = {}
        self.weights: Dict[str, float] = {}

    def add_model(self, name: str, model: BaseModel, weight: float = 1.0) -> None:
        """Add a component model to the ensemble."""
        self.component_models[name] = model
        self.weights[name] = weight
        self._normalize_weights()

    def remove_model(self, name: str) -> None:
        """Remove a component model."""
        if name in self.component_models:
            del self.component_models[name]
            del self.weights[name]
            self._normalize_weights()

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set model weights."""
        for name, weight in weights.items():
            if name in self.weights:
                self.weights[name] = weight
        self._normalize_weights()

    def build(self) -> None:
        """Build all component models."""
        for model in self.component_models.values():
            if not model.is_trained:
                model.build()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train all component models."""
        histories = {}
        for name, model in self.component_models.items():
            print(f"Training {name}...")
            history = model.train(X_train, y_train, X_val, y_val)
            histories[name] = history
        self.is_trained = True
        return histories

    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """Make ensemble prediction."""
        component_preds = {}
        for name, model in self.component_models.items():
            component_preds[name] = model.predict(X)

        return self._combine_predictions(component_preds)

    def predict_batch(self, X: np.ndarray) -> List[EnsemblePrediction]:
        """Make batch ensemble predictions."""
        all_component_preds = {name: [] for name in self.component_models}

        for name, model in self.component_models.items():
            preds = model.predict_batch(X)
            all_component_preds[name] = preds

        results = []
        for i in range(len(X)):
            component_preds = {
                name: preds[i] for name, preds in all_component_preds.items()
            }
            results.append(self._combine_predictions(component_preds))

        return results

    def _combine_predictions(
        self, component_preds: Dict[str, Prediction]
    ) -> EnsemblePrediction:
        """Combine component predictions using weighted average."""
        # Weighted price prediction
        price_pred = sum(
            self.weights[name] * pred.price_prediction
            for name, pred in component_preds.items()
        )

        # Weighted direction probability
        direction_scores = {"bullish": 0.0, "bearish": 0.0, "neutral": 0.0}
        for name, pred in component_preds.items():
            direction_scores[pred.direction] += (
                self.weights[name] * pred.direction_probability
            )

        direction = max(direction_scores, key=direction_scores.get)
        direction_prob = direction_scores[direction]

        # Weighted confidence
        confidence = sum(
            self.weights[name] * pred.confidence
            for name, pred in component_preds.items()
        )

        # Agreement score (how much models agree)
        directions = [pred.direction for pred in component_preds.values()]
        agreement = directions.count(direction) / len(directions)

        return EnsemblePrediction(
            timestamp=datetime.now(),
            symbol=next(iter(component_preds.values())).symbol,
            price_prediction=price_pred,
            direction=direction,
            direction_probability=direction_prob,
            confidence=confidence * agreement,
            model_name=self.name,
            model_version=self.version,
            component_predictions=component_preds,
            component_weights=self.weights.copy(),
            agreement_score=agreement,
        )


class TechnicalEnsemble(EnsembleModel):
    """
    Ensemble specifically for technical analysis models.

    Combines Short-term, Medium-term, and Long-term models
    with dynamic weight adjustment based on market conditions.
    """

    DEFAULT_CONFIG = {
        "name": "technical_ensemble",
        "version": "1.0.0",
        "default_weights": {
            "short_term": 0.3,
            "medium_term": 0.4,
            "long_term": 0.3,
        },
        "use_dynamic_weights": True,
        "combination_method": "weighted_avg",  # weighted_avg, stacking, attention
        "min_confidence": 0.6,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize technical ensemble."""
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)
        self.weights = self.config["default_weights"].copy()
        self._normalize_weights()

    def update_weights_for_regime(self, market_regime: str) -> None:
        """
        Update weights based on detected market regime.

        In trending markets: favor medium/long-term
        In ranging markets: favor short-term
        In volatile markets: reduce long-term weight
        """
        regime_adjustments = {
            "trending_up": {"short_term": 0.8, "medium_term": 1.2, "long_term": 1.3},
            "trending_down": {"short_term": 0.8, "medium_term": 1.2, "long_term": 1.3},
            "ranging": {"short_term": 1.3, "medium_term": 1.0, "long_term": 0.7},
            "volatile": {"short_term": 1.0, "medium_term": 0.9, "long_term": 0.6},
        }

        adjustments = regime_adjustments.get(
            market_regime, {"short_term": 1.0, "medium_term": 1.0, "long_term": 1.0}
        )

        base_weights = self.config["default_weights"]
        self.weights = {
            name: base_weights[name] * adjustments.get(name, 1.0)
            for name in base_weights
        }
        self._normalize_weights()

    def update_weights_from_performance(
        self,
        recent_performance: Dict[str, float],
        lookback_weight: float = 0.3,
    ) -> None:
        """
        Update weights based on recent model performance.

        Args:
            recent_performance: Dict mapping model name to accuracy/score
            lookback_weight: How much to weight recent performance vs defaults
        """
        # Normalize performance scores
        total_perf = sum(recent_performance.values())
        if total_perf > 0:
            perf_weights = {
                name: score / total_perf
                for name, score in recent_performance.items()
            }

            # Blend with default weights
            base_weights = self.config["default_weights"]
            self.weights = {
                name: (1 - lookback_weight) * base_weights.get(name, 0)
                + lookback_weight * perf_weights.get(name, 0)
                for name in set(base_weights) | set(perf_weights)
            }
            self._normalize_weights()

    def _combine_predictions(
        self, component_preds: Dict[str, Prediction]
    ) -> EnsemblePrediction:
        """Combine predictions with technical-specific logic."""
        method = self.config.get("combination_method", "weighted_avg")

        if method == "weighted_avg":
            return super()._combine_predictions(component_preds)

        elif method == "stacking":
            return self._stacking_combine(component_preds)

        elif method == "attention":
            return self._attention_combine(component_preds)

        else:
            return super()._combine_predictions(component_preds)

    def _stacking_combine(
        self, component_preds: Dict[str, Prediction]
    ) -> EnsemblePrediction:
        """Combine using meta-model stacking."""
        # For now, fall back to weighted average
        # Full implementation would use trained meta-model
        return super()._combine_predictions(component_preds)

    def _attention_combine(
        self, component_preds: Dict[str, Prediction]
    ) -> EnsemblePrediction:
        """Combine using attention-based weighting."""
        # For now, fall back to weighted average
        # Full implementation would use attention mechanism
        return super()._combine_predictions(component_preds)

    def get_signal(self, prediction: EnsemblePrediction) -> Dict[str, Any]:
        """
        Convert ensemble prediction to trading signal.

        Returns:
            Dictionary with action, strength, confidence, and reasoning
        """
        min_confidence = self.config.get("min_confidence", 0.6)

        if prediction.confidence < min_confidence:
            return {
                "action": "HOLD",
                "strength": 0.0,
                "confidence": prediction.confidence,
                "reason": f"Low confidence ({prediction.confidence:.2f} < {min_confidence})",
            }

        if prediction.agreement_score < 0.5:
            return {
                "action": "HOLD",
                "strength": 0.0,
                "confidence": prediction.confidence,
                "reason": f"Low model agreement ({prediction.agreement_score:.2f})",
            }

        if prediction.direction == "bullish":
            action = "BUY"
            strength = prediction.direction_probability
        elif prediction.direction == "bearish":
            action = "SELL"
            strength = prediction.direction_probability
        else:
            action = "HOLD"
            strength = 0.0

        return {
            "action": action,
            "strength": strength,
            "confidence": prediction.confidence,
            "agreement": prediction.agreement_score,
            "reason": f"Direction: {prediction.direction} with {prediction.direction_probability:.1%} probability",
        }
