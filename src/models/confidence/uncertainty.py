"""
Uncertainty estimation methods for trading predictions.

Two types of uncertainty:
1. Aleatoric (data uncertainty) - inherent noise in the data
2. Epistemic (model uncertainty) - uncertainty due to limited knowledge

Methods implemented:
1. MC Dropout - Epistemic uncertainty via dropout at inference
2. Ensemble Disagreement - Epistemic uncertainty from model ensemble
3. Prediction Entropy - Combined uncertainty measure
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Callable
from dataclasses import dataclass


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty estimates."""

    prediction: np.ndarray  # Mean prediction
    confidence: np.ndarray  # Calibrated confidence [0.5, 1.0]
    epistemic: np.ndarray  # Model uncertainty
    aleatoric: Optional[np.ndarray] = None  # Data uncertainty (if available)
    total: Optional[np.ndarray] = None  # Combined uncertainty

    def __post_init__(self):
        """Compute total uncertainty if not provided."""
        if self.total is None and self.aleatoric is not None:
            # Total uncertainty = sqrt(epistemic^2 + aleatoric^2)
            self.total = np.sqrt(self.epistemic ** 2 + self.aleatoric ** 2)
        elif self.total is None:
            self.total = self.epistemic


class MCDropoutUncertainty:
    """
    Monte Carlo Dropout for uncertainty estimation.

    At inference time, run the model multiple times with dropout enabled.
    The variance of predictions indicates epistemic uncertainty.

    Key insight: Models are uncertain when:
    - They haven't seen similar data during training
    - The pattern is ambiguous
    - Market conditions are unusual

    Reference: Gal & Ghahramani "Dropout as a Bayesian Approximation" (2016)
    """

    def __init__(
        self,
        n_samples: int = 50,
        dropout_rate: float = 0.1
    ):
        """
        Initialize MC Dropout uncertainty estimator.

        Args:
            n_samples: Number of forward passes for uncertainty estimation
            dropout_rate: Dropout probability (if not set in model)
        """
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

    def enable_dropout(self, model: nn.Module) -> None:
        """Enable dropout layers for inference."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def estimate(
        self,
        model: nn.Module,
        x: torch.Tensor,
        return_samples: bool = False
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty using MC Dropout.

        Args:
            model: Neural network model (must have dropout layers)
            x: Input tensor
            return_samples: If True, also return individual samples

        Returns:
            UncertaintyEstimate with prediction, confidence, and uncertainty
        """
        model.eval()
        self.enable_dropout(model)

        samples = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                logits = model(x)
                probs = torch.sigmoid(logits)
                samples.append(probs.cpu().numpy())

        samples = np.array(samples)  # Shape: [n_samples, batch_size, 1]

        # Mean prediction
        mean_prob = np.mean(samples, axis=0).squeeze()

        # Epistemic uncertainty = std of predictions
        epistemic = np.std(samples, axis=0).squeeze()

        # Prediction (binary)
        prediction = (mean_prob >= 0.5).astype(int)

        # Confidence: adjusted by uncertainty
        # Base confidence from probability
        base_confidence = np.where(
            prediction == 1,
            mean_prob,
            1 - mean_prob
        )

        # Adjust confidence by uncertainty
        # High epistemic uncertainty → lower confidence
        # Scale epistemic to [0, 0.5] range for adjustment
        uncertainty_penalty = np.clip(epistemic * 2, 0, 0.5)
        confidence = base_confidence - uncertainty_penalty

        # Ensure confidence is in [0.5, 1.0] range
        confidence = np.clip(confidence, 0.5, 1.0)

        result = UncertaintyEstimate(
            prediction=prediction,
            confidence=confidence,
            epistemic=epistemic
        )

        if return_samples:
            result.samples = samples

        return result


class EnsembleUncertainty:
    """
    Ensemble-based uncertainty estimation.

    Uses disagreement between ensemble members as uncertainty measure.
    When models disagree, we're less confident.

    This naturally integrates with the multi-timeframe approach:
    - Short-term model says "up"
    - Medium-term model says "down"
    → High uncertainty, low confidence

    Best used with diverse models (different architectures/timeframes).
    """

    def __init__(
        self,
        aggregation: str = 'mean',
        calibrated: bool = True
    ):
        """
        Initialize ensemble uncertainty estimator.

        Args:
            aggregation: How to combine predictions ('mean', 'median', 'voting')
            calibrated: Whether to use calibrated outputs
        """
        self.aggregation = aggregation
        self.calibrated = calibrated

    def estimate(
        self,
        predictions: List[np.ndarray],
        weights: Optional[np.ndarray] = None
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty from ensemble predictions.

        Args:
            predictions: List of probability predictions from each model
                        Each array shape: [batch_size] or [batch_size, 1]
            weights: Optional weights for each model

        Returns:
            UncertaintyEstimate with ensemble prediction and uncertainty
        """
        # Stack predictions: [n_models, batch_size]
        preds = np.array([p.flatten() for p in predictions])
        n_models = len(predictions)

        if weights is None:
            weights = np.ones(n_models) / n_models
        else:
            weights = np.array(weights) / np.sum(weights)

        # Aggregate predictions
        if self.aggregation == 'mean':
            mean_prob = np.average(preds, axis=0, weights=weights)
        elif self.aggregation == 'median':
            mean_prob = np.median(preds, axis=0)
        elif self.aggregation == 'voting':
            votes = (preds >= 0.5).astype(float)
            mean_prob = np.average(votes, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Epistemic uncertainty = weighted std of predictions
        # Measures disagreement between models
        weighted_var = np.average((preds - mean_prob) ** 2, axis=0, weights=weights)
        epistemic = np.sqrt(weighted_var)

        # Binary prediction
        prediction = (mean_prob >= 0.5).astype(int)

        # Base confidence
        base_confidence = np.where(
            prediction == 1,
            mean_prob,
            1 - mean_prob
        )

        # Adjust for disagreement
        # Max disagreement when half say up, half say down → std ≈ 0.5
        disagreement_penalty = np.clip(epistemic, 0, 0.5)
        confidence = base_confidence - disagreement_penalty
        confidence = np.clip(confidence, 0.5, 1.0)

        return UncertaintyEstimate(
            prediction=prediction,
            confidence=confidence,
            epistemic=epistemic
        )

    def estimate_from_logits(
        self,
        logits_list: List[torch.Tensor],
        weights: Optional[np.ndarray] = None
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty from model logits.

        Args:
            logits_list: List of logits from each model
            weights: Optional weights for each model

        Returns:
            UncertaintyEstimate
        """
        probs = [
            torch.sigmoid(logits).detach().cpu().numpy()
            for logits in logits_list
        ]
        return self.estimate(probs, weights)


class ConfidenceEstimator:
    """
    Unified confidence estimation combining multiple methods.

    This is the main interface for the trading system.

    Confidence levels for trading decisions:
    - 0.90+: Very high confidence → Full position size
    - 0.80-0.90: High confidence → 75% position
    - 0.70-0.80: Moderate confidence → 50% position
    - 0.60-0.70: Low confidence → 25% position
    - 0.50-0.60: Very low confidence → Skip trade or minimal position
    """

    def __init__(
        self,
        use_mc_dropout: bool = True,
        use_ensemble: bool = True,
        mc_samples: int = 30,
        confidence_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize confidence estimator.

        Args:
            use_mc_dropout: Whether to use MC Dropout for uncertainty
            use_ensemble: Whether to use ensemble disagreement
            mc_samples: Number of MC Dropout samples
            confidence_thresholds: Thresholds for trading decisions
        """
        self.use_mc_dropout = use_mc_dropout
        self.use_ensemble = use_ensemble

        self.mc_dropout = MCDropoutUncertainty(n_samples=mc_samples)
        self.ensemble_uncertainty = EnsembleUncertainty()

        self.confidence_thresholds = confidence_thresholds or {
            'very_high': 0.90,
            'high': 0.80,
            'moderate': 0.70,
            'low': 0.60,
            'minimum': 0.50
        }

    def estimate_single_model(
        self,
        model: nn.Module,
        x: torch.Tensor,
        use_mc: bool = True
    ) -> UncertaintyEstimate:
        """
        Estimate confidence for a single model.

        Args:
            model: Neural network model
            x: Input tensor
            use_mc: Whether to use MC Dropout

        Returns:
            UncertaintyEstimate
        """
        if use_mc and self.use_mc_dropout:
            return self.mc_dropout.estimate(model, x)
        else:
            # Simple forward pass
            model.eval()
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy().squeeze()

            prediction = (probs >= 0.5).astype(int)
            confidence = np.where(prediction == 1, probs, 1 - probs)

            return UncertaintyEstimate(
                prediction=prediction,
                confidence=confidence,
                epistemic=np.zeros_like(probs)  # No uncertainty estimate
            )

    def estimate_ensemble(
        self,
        models: List[nn.Module],
        x: torch.Tensor,
        weights: Optional[np.ndarray] = None,
        use_mc: bool = True
    ) -> UncertaintyEstimate:
        """
        Estimate confidence for model ensemble.

        Args:
            models: List of neural network models
            x: Input tensor
            weights: Optional model weights
            use_mc: Whether to use MC Dropout per model

        Returns:
            UncertaintyEstimate combining all uncertainty sources
        """
        individual_estimates = []
        individual_probs = []

        for model in models:
            if use_mc and self.use_mc_dropout:
                estimate = self.mc_dropout.estimate(model, x)
                # Use mean probability for ensemble
                prob = np.where(
                    estimate.prediction == 1,
                    estimate.confidence,
                    1 - estimate.confidence
                )
            else:
                model.eval()
                with torch.no_grad():
                    logits = model(x)
                    prob = torch.sigmoid(logits).cpu().numpy().squeeze()

            individual_probs.append(prob)
            individual_estimates.append(estimate if use_mc else None)

        # Ensemble uncertainty
        ensemble_estimate = self.ensemble_uncertainty.estimate(
            individual_probs, weights
        )

        # Combine MC Dropout uncertainty with ensemble uncertainty
        if use_mc and self.use_mc_dropout:
            mc_uncertainties = [e.epistemic for e in individual_estimates if e]
            mean_mc_uncertainty = np.mean(mc_uncertainties, axis=0)

            # Total epistemic = sqrt(ensemble^2 + mean_mc^2)
            total_epistemic = np.sqrt(
                ensemble_estimate.epistemic ** 2 + mean_mc_uncertainty ** 2
            )

            # Recompute confidence with total uncertainty
            base_confidence = np.where(
                ensemble_estimate.prediction == 1,
                np.mean(individual_probs, axis=0),
                1 - np.mean(individual_probs, axis=0)
            )

            confidence = base_confidence - np.clip(total_epistemic, 0, 0.5)
            confidence = np.clip(confidence, 0.5, 1.0)

            return UncertaintyEstimate(
                prediction=ensemble_estimate.prediction,
                confidence=confidence,
                epistemic=total_epistemic
            )

        return ensemble_estimate

    def get_trading_signal(
        self,
        estimate: UncertaintyEstimate
    ) -> Dict[str, np.ndarray]:
        """
        Convert uncertainty estimate to trading signals.

        Args:
            estimate: UncertaintyEstimate from estimation methods

        Returns:
            Dictionary with:
            - direction: 1 (up/long), -1 (down/short), 0 (no trade)
            - confidence: Calibrated confidence score
            - position_size: Suggested position size multiplier [0, 1]
            - confidence_level: Categorical level (very_high, high, etc.)
        """
        direction = np.where(estimate.prediction == 1, 1, -1)
        confidence = estimate.confidence

        # Position sizing based on confidence
        position_size = np.zeros_like(confidence)

        very_high = confidence >= self.confidence_thresholds['very_high']
        high = (confidence >= self.confidence_thresholds['high']) & ~very_high
        moderate = (confidence >= self.confidence_thresholds['moderate']) & ~high & ~very_high
        low = (confidence >= self.confidence_thresholds['low']) & ~moderate & ~high & ~very_high

        position_size[very_high] = 1.0
        position_size[high] = 0.75
        position_size[moderate] = 0.50
        position_size[low] = 0.25
        # Below 'low' threshold → 0 position size (no trade)

        # Create confidence level labels
        confidence_level = np.full(confidence.shape, 'skip', dtype=object)
        confidence_level[low] = 'low'
        confidence_level[moderate] = 'moderate'
        confidence_level[high] = 'high'
        confidence_level[very_high] = 'very_high'

        # Zero out direction for no-trade signals
        direction = direction * (position_size > 0).astype(int)

        return {
            'direction': direction,
            'confidence': confidence,
            'position_size': position_size,
            'confidence_level': confidence_level,
            'epistemic_uncertainty': estimate.epistemic
        }


def entropy_uncertainty(probs: np.ndarray) -> np.ndarray:
    """
    Compute prediction entropy as uncertainty measure.

    Maximum entropy at p=0.5 (most uncertain)
    Minimum entropy at p=0 or p=1 (most certain)

    Args:
        probs: Prediction probabilities [0, 1]

    Returns:
        Entropy values (higher = more uncertain)
    """
    # Avoid log(0)
    eps = 1e-10
    probs = np.clip(probs, eps, 1 - eps)

    # Binary entropy: -p*log(p) - (1-p)*log(1-p)
    entropy = -probs * np.log2(probs) - (1 - probs) * np.log2(1 - probs)

    return entropy
