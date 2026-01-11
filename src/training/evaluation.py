"""Model evaluation with trading-specific metrics.

Provides comprehensive evaluation for trading models including:
- Direction accuracy and classification metrics
- Confidence calibration metrics
- Simulated trading performance
- Multi-horizon evaluation
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class CalibrationMetrics:
    """Calibration metrics for probability predictions.

    Attributes:
        ece: Expected Calibration Error.
        mce: Maximum Calibration Error.
        brier_score: Brier score for probability accuracy.
        reliability_diagram: Binned accuracy vs confidence.
    """

    ece: float = 0.0
    mce: float = 0.0
    brier_score: float = 0.0
    reliability_diagram: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class DirectionMetrics:
    """Direction prediction metrics.

    Attributes:
        accuracy: Overall accuracy.
        precision: Precision per class.
        recall: Recall per class.
        f1_score: F1 score per class.
        confusion_matrix: Confusion matrix.
    """

    accuracy: float = 0.0
    precision: Dict[str, float] = field(default_factory=dict)
    recall: Dict[str, float] = field(default_factory=dict)
    f1_score: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None


@dataclass
class TradingMetrics:
    """Simulated trading metrics.

    Attributes:
        sharpe_ratio: Risk-adjusted return.
        sortino_ratio: Downside risk-adjusted return.
        max_drawdown: Maximum drawdown.
        total_return: Total return.
        win_rate: Winning trade percentage.
        profit_factor: Gross profit / gross loss.
        expectancy: Expected profit per trade.
    """

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0


@dataclass
class EvaluationResult:
    """Complete evaluation results.

    Attributes:
        direction_metrics: Direction prediction metrics.
        calibration_metrics: Calibration metrics.
        trading_metrics: Simulated trading metrics.
        horizon_metrics: Metrics per prediction horizon.
        summary: Summary statistics.
    """

    direction_metrics: DirectionMetrics = field(default_factory=DirectionMetrics)
    calibration_metrics: CalibrationMetrics = field(default_factory=CalibrationMetrics)
    trading_metrics: TradingMetrics = field(default_factory=TradingMetrics)
    horizon_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "direction": {
                "accuracy": self.direction_metrics.accuracy,
                "precision": self.direction_metrics.precision,
                "recall": self.direction_metrics.recall,
                "f1_score": self.direction_metrics.f1_score,
            },
            "calibration": {
                "ece": self.calibration_metrics.ece,
                "mce": self.calibration_metrics.mce,
                "brier_score": self.calibration_metrics.brier_score,
            },
            "trading": {
                "sharpe_ratio": self.trading_metrics.sharpe_ratio,
                "sortino_ratio": self.trading_metrics.sortino_ratio,
                "max_drawdown": self.trading_metrics.max_drawdown,
                "total_return": self.trading_metrics.total_return,
                "win_rate": self.trading_metrics.win_rate,
                "profit_factor": self.trading_metrics.profit_factor,
                "expectancy": self.trading_metrics.expectancy,
            },
            "horizon_metrics": self.horizon_metrics,
            "summary": self.summary,
        }


class ModelEvaluator:
    """Comprehensive model evaluation for trading predictions.

    Evaluates:
    - Direction prediction accuracy
    - Probability calibration
    - Simulated trading performance
    - Multi-horizon predictions

    Example:
        ```python
        evaluator = ModelEvaluator()
        result = evaluator.evaluate(
            model=trained_model,
            test_loader=test_loader,
            prices=price_series,
        )
        print(f"Accuracy: {result.direction_metrics.accuracy:.2%}")
        print(f"Sharpe: {result.trading_metrics.sharpe_ratio:.2f}")
        ```
    """

    def __init__(
        self,
        device: str = "cpu",
        confidence_threshold: float = 0.6,
        num_calibration_bins: int = 10,
    ):
        """Initialize evaluator.

        Args:
            device: Device for model inference.
            confidence_threshold: Minimum confidence for trading.
            num_calibration_bins: Bins for calibration metrics.
        """
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.num_calibration_bins = num_calibration_bins

    def evaluate(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        prices: Optional[pd.Series] = None,
        horizons: Optional[List[int]] = None,
    ) -> EvaluationResult:
        """Run complete evaluation.

        Args:
            model: Trained model.
            test_loader: Test data loader.
            prices: Price series for trading simulation.
            horizons: Prediction horizons to evaluate.

        Returns:
            Complete EvaluationResult.
        """
        model.to(self.device)
        model.eval()

        # Collect predictions
        all_predictions = []
        all_targets = []
        all_confidences = []

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1] if len(batch) > 1 else None
                else:
                    inputs = batch.to(self.device)
                    targets = None

                outputs = model(inputs)

                # Extract predictions
                if isinstance(outputs, dict):
                    # Multi-output model
                    pred_prices = outputs.get("price", outputs.get("predictions"))
                    pred_direction = outputs.get("direction_logits")
                    pred_confidence = outputs.get("alpha")  # From Beta distribution

                    if pred_direction is not None:
                        # Convert logits to class predictions
                        if pred_direction.dim() == 3:
                            # Shape: (batch, horizons, classes)
                            # Use softmax and take argmax
                            pred_probs = torch.softmax(pred_direction, dim=-1)
                            pred_classes = pred_direction.argmax(dim=-1)
                            # For binary targets, collapse to 0 (down) or 1 (up)
                            # by checking if prediction leans toward class 0 or classes 1+2
                            all_predictions.append(pred_classes.cpu())
                        elif pred_direction.dim() == 2:
                            # Shape: (batch, classes)
                            pred_classes = pred_direction.argmax(dim=-1)
                            all_predictions.append(pred_classes.cpu())
                        else:
                            all_predictions.append(pred_direction.cpu())

                    if pred_confidence is not None and "beta" in outputs:
                        # Calculate confidence from Beta distribution
                        alpha = outputs["alpha"].cpu()
                        beta = outputs["beta"].cpu()
                        confidence = alpha / (alpha + beta)  # Mean of Beta
                        all_confidences.append(confidence)

                    # Store targets - they are the raw labels from data loader
                    if targets is not None:
                        target_tensor = targets.cpu() if isinstance(targets, torch.Tensor) else torch.tensor(targets)
                        all_targets.append(target_tensor)
                    elif pred_prices is not None:
                        all_targets.append(torch.zeros_like(pred_prices.cpu()))
                else:
                    # Simple output (single tensor)
                    if outputs.dim() >= 2 and outputs.shape[-1] > 1:
                        # Classification logits
                        all_predictions.append(outputs.argmax(dim=-1).cpu())
                    else:
                        # Regression or probability
                        all_predictions.append(outputs.cpu())
                    if targets is not None:
                        all_targets.append(targets.cpu() if isinstance(targets, torch.Tensor) else torch.tensor(targets))

        # Concatenate results
        if all_predictions:
            predictions = torch.cat(all_predictions, dim=0).numpy()
        else:
            predictions = np.array([])

        if all_targets:
            targets = torch.cat(all_targets, dim=0).numpy()
        else:
            targets = np.array([])

        if all_confidences:
            confidences = torch.cat(all_confidences, dim=0).numpy()
        else:
            confidences = np.ones_like(predictions) * 0.5

        # Calculate metrics
        result = EvaluationResult()

        if len(predictions) > 0 and len(targets) > 0:
            # Direction metrics
            result.direction_metrics = self._calculate_direction_metrics(
                predictions, targets
            )

            # Calibration metrics
            if len(confidences) > 0:
                result.calibration_metrics = self._calculate_calibration_metrics(
                    confidences, predictions, targets
                )

            # Trading metrics (if prices provided)
            if prices is not None:
                result.trading_metrics = self._calculate_trading_metrics(
                    predictions, confidences, prices
                )

            # Per-horizon metrics
            if horizons is not None and predictions.ndim > 1:
                result.horizon_metrics = self._calculate_horizon_metrics(
                    predictions, targets, horizons
                )

            # Summary
            result.summary = self._generate_summary(result)

        return result

    def _calculate_direction_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> DirectionMetrics:
        """Calculate direction prediction metrics."""
        # Flatten for overall metrics
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        # Handle targets: convert to direction labels
        if not np.issubdtype(target_flat.dtype, np.integer):
            # Check if binary (0.0 or 1.0) or continuous
            unique_targets = np.unique(target_flat[~np.isnan(target_flat)])
            if len(unique_targets) <= 2 and all(v in [0.0, 1.0] for v in unique_targets):
                # Binary labels: 0=down, 1=up - keep as is
                target_flat = target_flat.astype(int)
            else:
                # Continuous returns: use sign, map -1,0,1 -> 0,1,2
                target_flat = np.sign(target_flat).astype(int) + 1
                target_flat[target_flat == 1] = 1  # -1 -> 0 (down)
                # 0 stays 1 (neutral), 1 -> 2 (up)

        # Handle predictions: convert to direction labels
        if not np.issubdtype(pred_flat.dtype, np.integer):
            # Check if probabilities (0-1 range) or continuous
            pred_min, pred_max = np.nanmin(pred_flat), np.nanmax(pred_flat)
            if pred_min >= 0 and pred_max <= 1:
                # Probabilities: threshold at 0.5 for binary classification
                pred_flat = (pred_flat >= 0.5).astype(int)
            else:
                # Continuous: use sign
                pred_flat = np.sign(pred_flat).astype(int)
                if pred_flat.min() < 0:
                    pred_flat = pred_flat + 1  # Map -1,0,1 -> 0,1,2

        # Check if targets are binary (0, 1) but predictions are multi-class (0, 1, 2)
        target_unique = np.unique(target_flat)
        pred_unique = np.unique(pred_flat)
        if len(target_unique) <= 2 and max(target_unique) <= 1 and max(pred_unique) > 1:
            # Map 3-class predictions to binary: 0=down, 1+=up
            pred_flat = (pred_flat > 0).astype(int)

        # Ensure consistent mapping for -1, 0, 1 labels
        unique_labels = np.unique(np.concatenate([pred_flat, target_flat]))
        if -1 in unique_labels:
            pred_flat = pred_flat + 1
            target_flat = target_flat + 1

        # Calculate accuracy
        accuracy = np.mean(pred_flat == target_flat)

        # Calculate per-class metrics
        classes = np.unique(np.concatenate([pred_flat, target_flat]))
        num_classes = len(classes)
        precision = {}
        recall = {}
        f1 = {}

        for cls in classes:
            cls_name = self._class_name(cls, num_classes)

            # True positives, false positives, false negatives
            tp = np.sum((pred_flat == cls) & (target_flat == cls))
            fp = np.sum((pred_flat == cls) & (target_flat != cls))
            fn = np.sum((pred_flat != cls) & (target_flat == cls))

            # Precision
            precision[cls_name] = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # Recall
            recall[cls_name] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # F1
            if precision[cls_name] + recall[cls_name] > 0:
                f1[cls_name] = 2 * precision[cls_name] * recall[cls_name] / (
                    precision[cls_name] + recall[cls_name]
                )
            else:
                f1[cls_name] = 0.0

        # Confusion matrix
        n_classes = len(classes)
        confusion = np.zeros((n_classes, n_classes), dtype=int)
        for i, true_cls in enumerate(classes):
            for j, pred_cls in enumerate(classes):
                confusion[i, j] = np.sum(
                    (target_flat == true_cls) & (pred_flat == pred_cls)
                )

        return DirectionMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=confusion,
        )

    def _class_name(self, cls: int, num_classes: int = 3) -> str:
        """Get human-readable class name.

        For binary classification (2 classes): 0=down, 1=up
        For ternary classification (3 classes): 0=down, 1=neutral, 2=up
        """
        if num_classes <= 2:
            names = {0: "down", 1: "up"}
        else:
            names = {0: "down", 1: "neutral", 2: "up"}
        return names.get(int(cls), f"class_{cls}")

    def _calculate_calibration_metrics(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> CalibrationMetrics:
        """Calculate probability calibration metrics."""
        # Flatten
        conf_flat = confidences.flatten()
        pred_flat = predictions.flatten()
        target_flat = targets.flatten()

        # Convert targets to binary correctness
        correct = (pred_flat == target_flat).astype(float)

        # Calculate ECE and MCE using binning
        bin_boundaries = np.linspace(0, 1, self.num_calibration_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        ece = 0.0
        mce = 0.0

        for i in range(self.num_calibration_bins):
            in_bin = (conf_flat > bin_boundaries[i]) & (
                conf_flat <= bin_boundaries[i + 1]
            )
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                avg_confidence = np.mean(conf_flat[in_bin])
                avg_accuracy = np.mean(correct[in_bin])

                bin_accuracies.append(avg_accuracy)
                bin_confidences.append(avg_confidence)
                bin_counts.append(np.sum(in_bin))

                # ECE contribution
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

                # MCE update
                mce = max(mce, np.abs(avg_accuracy - avg_confidence))

        # Brier score
        # For multi-class, use one-vs-all
        brier_score = np.mean((conf_flat - correct) ** 2)

        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            brier_score=brier_score,
            reliability_diagram={
                "accuracies": bin_accuracies,
                "confidences": bin_confidences,
                "counts": bin_counts,
            },
        )

    def _calculate_trading_metrics(
        self,
        predictions: np.ndarray,
        confidences: np.ndarray,
        prices: pd.Series,
    ) -> TradingMetrics:
        """Calculate simulated trading metrics."""
        # Flatten predictions
        pred_flat = predictions.flatten()
        conf_flat = confidences.flatten()

        # Align with prices
        n_samples = min(len(pred_flat), len(prices) - 1)
        pred_flat = pred_flat[:n_samples]
        conf_flat = conf_flat[:n_samples]

        # Calculate returns
        price_array = prices.values
        returns = np.diff(price_array) / price_array[:-1]
        returns = returns[:n_samples]

        # Convert predictions to positions (-1, 0, 1)
        positions = np.zeros(n_samples)

        for i in range(n_samples):
            if conf_flat[i] >= self.confidence_threshold:
                if pred_flat[i] == 2 or pred_flat[i] > 0:  # Up
                    positions[i] = 1
                elif pred_flat[i] == 0 or pred_flat[i] < 0:  # Down
                    positions[i] = -1
                # Neutral stays 0

        # Calculate strategy returns
        strategy_returns = positions * returns

        # Calculate metrics
        if len(strategy_returns) > 0 and np.std(strategy_returns) > 0:
            # Sharpe ratio (annualized, assuming daily returns)
            sharpe = (
                np.mean(strategy_returns)
                / np.std(strategy_returns)
                * np.sqrt(252)
            )

            # Sortino ratio
            downside_returns = strategy_returns[strategy_returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                sortino = (
                    np.mean(strategy_returns) / downside_std * np.sqrt(252)
                    if downside_std > 0
                    else 0.0
                )
            else:
                sortino = float("inf") if np.mean(strategy_returns) > 0 else 0.0

            # Cumulative returns for drawdown
            cumulative = np.cumprod(1 + strategy_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = np.abs(np.min(drawdowns))

            # Total return
            total_return = cumulative[-1] - 1 if len(cumulative) > 0 else 0.0

            # Win rate and profit factor
            wins = strategy_returns[strategy_returns > 0]
            losses = strategy_returns[strategy_returns < 0]

            win_rate = len(wins) / len(strategy_returns) if len(strategy_returns) > 0 else 0.0

            gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
            gross_loss = np.abs(np.sum(losses)) if len(losses) > 0 else 0.0
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )

            # Expectancy
            avg_win = np.mean(wins) if len(wins) > 0 else 0.0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
            expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        else:
            sharpe = 0.0
            sortino = 0.0
            max_drawdown = 0.0
            total_return = 0.0
            win_rate = 0.0
            profit_factor = 0.0
            expectancy = 0.0

        return TradingMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            total_return=total_return,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
        )

    def _calculate_horizon_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        horizons: List[int],
    ) -> Dict[int, Dict[str, float]]:
        """Calculate metrics per prediction horizon."""
        horizon_metrics = {}

        # Predictions shape: (samples, horizons) or (samples, horizons, classes)
        if predictions.ndim == 1:
            # Single horizon
            horizons = [horizons[0]] if horizons else [1]
            predictions = predictions.reshape(-1, 1)
            targets = targets.reshape(-1, 1)

        for i, horizon in enumerate(horizons):
            if i >= predictions.shape[1]:
                break

            pred_h = predictions[:, i]
            target_h = targets[:, i] if targets.ndim > 1 else targets

            # Calculate accuracy for this horizon
            if not np.issubdtype(target_h.dtype, np.integer):
                target_h = np.sign(target_h)
                target_h[target_h == 0] = 1

            if not np.issubdtype(pred_h.dtype, np.integer):
                pred_h = np.sign(pred_h)
                pred_h[pred_h == 0] = 1

            accuracy = np.mean(pred_h == target_h)

            horizon_metrics[horizon] = {
                "accuracy": accuracy,
                "samples": len(pred_h),
            }

        return horizon_metrics

    def _generate_summary(self, result: EvaluationResult) -> Dict[str, Any]:
        """Generate evaluation summary."""
        return {
            "overall_accuracy": result.direction_metrics.accuracy,
            "calibration_ece": result.calibration_metrics.ece,
            "trading_sharpe": result.trading_metrics.sharpe_ratio,
            "trading_return": result.trading_metrics.total_return,
            "meets_accuracy_target": result.direction_metrics.accuracy > 0.55,
            "meets_sharpe_target": result.trading_metrics.sharpe_ratio > 1.5,
            "meets_drawdown_target": result.trading_metrics.max_drawdown < 0.15,
        }

    def generate_report(
        self,
        result: EvaluationResult,
        model_name: str = "Model",
    ) -> str:
        """Generate human-readable evaluation report.

        Args:
            result: Evaluation result.
            model_name: Name of the model.

        Returns:
            Formatted report string.
        """
        lines = [
            "=" * 60,
            f"EVALUATION REPORT: {model_name}",
            "=" * 60,
            "",
            "DIRECTION PREDICTION METRICS",
            "-" * 40,
            f"  Accuracy:           {result.direction_metrics.accuracy:.2%}",
        ]

        for cls, prec in result.direction_metrics.precision.items():
            rec = result.direction_metrics.recall.get(cls, 0)
            f1 = result.direction_metrics.f1_score.get(cls, 0)
            lines.append(f"  {cls.capitalize():12} P:{prec:.2f} R:{rec:.2f} F1:{f1:.2f}")

        lines.extend([
            "",
            "CALIBRATION METRICS",
            "-" * 40,
            f"  ECE:                {result.calibration_metrics.ece:.4f}",
            f"  MCE:                {result.calibration_metrics.mce:.4f}",
            f"  Brier Score:        {result.calibration_metrics.brier_score:.4f}",
            "",
            "TRADING METRICS (Simulated)",
            "-" * 40,
            f"  Sharpe Ratio:       {result.trading_metrics.sharpe_ratio:.2f}",
            f"  Sortino Ratio:      {result.trading_metrics.sortino_ratio:.2f}",
            f"  Max Drawdown:       {result.trading_metrics.max_drawdown:.2%}",
            f"  Total Return:       {result.trading_metrics.total_return:.2%}",
            f"  Win Rate:           {result.trading_metrics.win_rate:.2%}",
            f"  Profit Factor:      {result.trading_metrics.profit_factor:.2f}",
            f"  Expectancy:         {result.trading_metrics.expectancy:.4f}",
        ])

        if result.horizon_metrics:
            lines.extend([
                "",
                "PER-HORIZON METRICS",
                "-" * 40,
            ])
            for horizon, metrics in result.horizon_metrics.items():
                lines.append(
                    f"  Horizon {horizon}: Accuracy={metrics['accuracy']:.2%}"
                )

        lines.extend([
            "",
            "TARGET ACHIEVEMENT",
            "-" * 40,
            f"  Accuracy > 55%:     {'PASS' if result.summary.get('meets_accuracy_target') else 'FAIL'}",
            f"  Sharpe > 1.5:       {'PASS' if result.summary.get('meets_sharpe_target') else 'FAIL'}",
            f"  Drawdown < 15%:     {'PASS' if result.summary.get('meets_drawdown_target') else 'FAIL'}",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)


def evaluate_ensemble(
    models: Dict[str, torch.nn.Module],
    test_loaders: Dict[str, DataLoader],
    weights: Optional[Dict[str, float]] = None,
    prices: Optional[pd.Series] = None,
) -> Dict[str, EvaluationResult]:
    """Evaluate multiple models and ensemble.

    Args:
        models: Dictionary of model name to model.
        test_loaders: Dictionary of model name to test loader.
        weights: Optional ensemble weights.
        prices: Price series for trading simulation.

    Returns:
        Dictionary of model name to evaluation result.
    """
    evaluator = ModelEvaluator()
    results = {}

    # Evaluate individual models
    for name, model in models.items():
        if name in test_loaders:
            result = evaluator.evaluate(
                model=model,
                test_loader=test_loaders[name],
                prices=prices,
            )
            results[name] = result
            logger.info(f"{name}: Accuracy={result.direction_metrics.accuracy:.2%}")

    return results
