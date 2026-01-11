"""
Calibration methods for confidence estimation.

The key insight: raw neural network outputs (sigmoid/softmax) are often
poorly calibrated. A model outputting 0.9 confidence might only be correct
70% of the time. Calibration fixes this.

Methods implemented:
1. Temperature Scaling - Simple, effective, preserves ranking
2. Platt Scaling - Logistic regression on logits
3. Isotonic Regression - Non-parametric, most flexible
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import warnings


@dataclass
class CalibrationMetrics:
    """Metrics for evaluating calibration quality."""

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier_score: float  # Brier score (lower is better)
    reliability_diagram: Dict[str, np.ndarray]  # For plotting

    @staticmethod
    def compute(
        confidences: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> 'CalibrationMetrics':
        """
        Compute calibration metrics.

        Args:
            confidences: Model confidence scores [0, 1]
            predictions: Binary predictions (0 or 1)
            labels: True labels (0 or 1)
            n_bins: Number of bins for ECE/MCE calculation

        Returns:
            CalibrationMetrics with ECE, MCE, Brier score, and reliability data
        """
        # Ensure numpy arrays
        confidences = np.asarray(confidences).flatten()
        predictions = np.asarray(predictions).flatten()
        labels = np.asarray(labels).flatten()

        # Binary case: use confidence for predicted class
        # If prediction is 0, confidence is (1 - raw_confidence)
        # If prediction is 1, confidence is raw_confidence

        # Brier score
        brier = np.mean((confidences - labels) ** 2)

        # Bin the confidences
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(confidences, bin_boundaries[1:-1])

        ece = 0.0
        mce = 0.0

        bin_confidences = []
        bin_accuracies = []
        bin_counts = []

        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if np.sum(mask) > 0:
                bin_conf = np.mean(confidences[mask])
                bin_acc = np.mean(predictions[mask] == labels[mask])
                bin_count = np.sum(mask)

                bin_confidences.append(bin_conf)
                bin_accuracies.append(bin_acc)
                bin_counts.append(bin_count)

                # ECE contribution
                ece += (bin_count / len(confidences)) * abs(bin_acc - bin_conf)

                # MCE
                mce = max(mce, abs(bin_acc - bin_conf))
            else:
                bin_confidences.append(np.nan)
                bin_accuracies.append(np.nan)
                bin_counts.append(0)

        reliability_diagram = {
            'bin_confidences': np.array(bin_confidences),
            'bin_accuracies': np.array(bin_accuracies),
            'bin_counts': np.array(bin_counts),
            'bin_edges': bin_boundaries,
        }

        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier_score=brier,
            reliability_diagram=reliability_diagram
        )

    def __repr__(self) -> str:
        return (
            f"CalibrationMetrics(ECE={self.ece:.4f}, MCE={self.mce:.4f}, "
            f"Brier={self.brier_score:.4f})"
        )


class TemperatureScaling(nn.Module):
    """
    Temperature Scaling for calibration.

    The simplest and often most effective calibration method.
    Learns a single temperature parameter T to scale logits:

        calibrated_prob = sigmoid(logit / T)

    - T > 1: Softens predictions (reduces overconfidence)
    - T < 1: Sharpens predictions (increases confidence)
    - T = 1: No change

    Reference: Guo et al. "On Calibration of Modern Neural Networks" (2017)
    """

    def __init__(self):
        super().__init__()
        # Initialize temperature to 1 (no scaling)
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Raw model outputs (before sigmoid)

        Returns:
            Calibrated probabilities
        """
        return torch.sigmoid(logits / self.temperature)

    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> 'TemperatureScaling':
        """
        Learn optimal temperature on validation set.

        Args:
            logits: Validation set logits
            labels: Validation set labels
            lr: Learning rate for optimization
            max_iter: Maximum optimization iterations

        Returns:
            self (fitted)
        """
        logits = logits.detach()
        labels = labels.float()

        # Use LBFGS optimizer (works well for single parameter)
        optimizer = LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        return self

    def get_confidence(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get calibrated predictions and confidence scores.

        Args:
            logits: Raw model outputs

        Returns:
            Tuple of (predictions, confidence_scores)
            - predictions: Binary (0 or 1)
            - confidence_scores: How confident [0.5, 1.0]
        """
        probs = self.forward(logits)
        predictions = (probs >= 0.5).long()

        # Confidence is distance from 0.5, scaled to [0.5, 1.0]
        # prob=0.5 → conf=0.5, prob=0 or 1 → conf=1.0
        confidence = torch.where(
            predictions == 1,
            probs,  # For "up" predictions, confidence = prob
            1 - probs  # For "down" predictions, confidence = 1 - prob
        )

        return predictions, confidence


class PlattScaling:
    """
    Platt Scaling for calibration.

    Fits a logistic regression on top of model outputs:
        P(y=1|f) = 1 / (1 + exp(A*f + B))

    More flexible than temperature scaling but can overfit
    on small calibration sets.

    Reference: Platt (1999) "Probabilistic Outputs for SVMs"
    """

    def __init__(self):
        self.calibrator = LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            C=1e10  # High C = low regularization
        )
        self._fitted = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray
    ) -> 'PlattScaling':
        """
        Fit Platt scaling on validation data.

        Args:
            logits: Validation logits (can be numpy or torch)
            labels: Validation labels

        Returns:
            self (fitted)
        """
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        logits = logits.reshape(-1, 1)
        labels = labels.flatten()

        self.calibrator.fit(logits, labels)
        self._fitted = True

        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """
        Get calibrated probabilities.

        Args:
            logits: Raw model logits

        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            raise RuntimeError("PlattScaling not fitted. Call fit() first.")

        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()

        logits = logits.reshape(-1, 1)
        probs = self.calibrator.predict_proba(logits)[:, 1]

        return probs

    def get_confidence(
        self,
        logits: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get calibrated predictions and confidence scores.

        Args:
            logits: Raw model outputs

        Returns:
            Tuple of (predictions, confidence_scores)
        """
        probs = self.predict_proba(logits)
        predictions = (probs >= 0.5).astype(int)

        confidence = np.where(
            predictions == 1,
            probs,
            1 - probs
        )

        return predictions, confidence


class IsotonicCalibration:
    """
    Isotonic Regression for calibration.

    Non-parametric method that fits a monotonic function to map
    model outputs to calibrated probabilities. Most flexible but
    can overfit on small datasets.

    Best used when you have a large calibration set (>1000 samples).
    """

    def __init__(self):
        self.calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds='clip'
        )
        self._fitted = False

    def fit(
        self,
        probs: np.ndarray,
        labels: np.ndarray
    ) -> 'IsotonicCalibration':
        """
        Fit isotonic regression on validation data.

        Args:
            probs: Uncalibrated probabilities (after sigmoid)
            labels: True labels

        Returns:
            self (fitted)
        """
        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        probs = probs.flatten()
        labels = labels.flatten()

        self.calibrator.fit(probs, labels)
        self._fitted = True

        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """
        Get calibrated probabilities.

        Args:
            probs: Uncalibrated probabilities

        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            raise RuntimeError("IsotonicCalibration not fitted. Call fit() first.")

        if isinstance(probs, torch.Tensor):
            probs = probs.detach().cpu().numpy()

        probs = probs.flatten()
        calibrated = self.calibrator.predict(probs)

        return calibrated

    def get_confidence(
        self,
        probs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get calibrated predictions and confidence scores.

        Args:
            probs: Uncalibrated probabilities

        Returns:
            Tuple of (predictions, confidence_scores)
        """
        calibrated_probs = self.predict(probs)
        predictions = (calibrated_probs >= 0.5).astype(int)

        confidence = np.where(
            predictions == 1,
            calibrated_probs,
            1 - calibrated_probs
        )

        return predictions, confidence


def select_best_calibrator(
    logits: np.ndarray,
    labels: np.ndarray,
    val_split: float = 0.5
) -> Tuple[Any, str, CalibrationMetrics]:
    """
    Automatically select the best calibration method.

    Splits data, fits all methods, and selects based on ECE.

    Args:
        logits: Model logits
        labels: True labels
        val_split: Fraction for validation

    Returns:
        Tuple of (best_calibrator, method_name, metrics)
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    logits = logits.flatten()
    labels = labels.flatten()

    # Split data
    n = len(logits)
    split_idx = int(n * val_split)

    train_logits, val_logits = logits[:split_idx], logits[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    results = {}

    # Temperature Scaling
    temp_scaling = TemperatureScaling()
    temp_scaling.fit(
        torch.tensor(train_logits),
        torch.tensor(train_labels)
    )
    temp_probs = temp_scaling(torch.tensor(val_logits)).detach().numpy()
    temp_preds = (temp_probs >= 0.5).astype(int)
    temp_conf = np.where(temp_preds == 1, temp_probs, 1 - temp_probs)
    temp_metrics = CalibrationMetrics.compute(temp_conf, temp_preds, val_labels)
    results['temperature'] = (temp_scaling, temp_metrics)

    # Platt Scaling
    platt = PlattScaling()
    platt.fit(train_logits, train_labels)
    platt_probs = platt.predict_proba(val_logits)
    platt_preds = (platt_probs >= 0.5).astype(int)
    platt_conf = np.where(platt_preds == 1, platt_probs, 1 - platt_probs)
    platt_metrics = CalibrationMetrics.compute(platt_conf, platt_preds, val_labels)
    results['platt'] = (platt, platt_metrics)

    # Isotonic (needs probabilities)
    train_probs = 1 / (1 + np.exp(-train_logits))
    val_probs = 1 / (1 + np.exp(-val_logits))

    isotonic = IsotonicCalibration()
    isotonic.fit(train_probs, train_labels)
    iso_probs = isotonic.predict(val_probs)
    iso_preds = (iso_probs >= 0.5).astype(int)
    iso_conf = np.where(iso_preds == 1, iso_probs, 1 - iso_probs)
    iso_metrics = CalibrationMetrics.compute(iso_conf, iso_preds, val_labels)
    results['isotonic'] = (isotonic, iso_metrics)

    # Select best based on ECE
    best_name = min(results.keys(), key=lambda k: results[k][1].ece)
    best_calibrator, best_metrics = results[best_name]

    return best_calibrator, best_name, best_metrics
