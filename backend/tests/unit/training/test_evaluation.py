"""Tests for model evaluation module."""

import pytest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock

from src.training.evaluation import (
    CalibrationMetrics,
    DirectionMetrics,
    TradingMetrics,
    EvaluationResult,
    ModelEvaluator,
    evaluate_ensemble,
)


class TestCalibrationMetrics:
    """Tests for CalibrationMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = CalibrationMetrics()

        assert metrics.ece == 0.0
        assert metrics.mce == 0.0
        assert metrics.brier_score == 0.0
        assert metrics.reliability_diagram == {}

    def test_custom_values(self):
        """Test custom metric values."""
        metrics = CalibrationMetrics(
            ece=0.05,
            mce=0.1,
            brier_score=0.2,
            reliability_diagram={"accuracies": [0.5, 0.6]},
        )

        assert metrics.ece == 0.05
        assert metrics.mce == 0.1
        assert metrics.brier_score == 0.2


class TestDirectionMetrics:
    """Tests for DirectionMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = DirectionMetrics()

        assert metrics.accuracy == 0.0
        assert metrics.precision == {}
        assert metrics.recall == {}
        assert metrics.f1_score == {}
        assert metrics.confusion_matrix is None

    def test_custom_values(self):
        """Test custom metric values."""
        cm = np.array([[10, 2], [3, 15]])
        metrics = DirectionMetrics(
            accuracy=0.83,
            precision={"down": 0.77, "up": 0.88},
            recall={"down": 0.83, "up": 0.83},
            f1_score={"down": 0.8, "up": 0.86},
            confusion_matrix=cm,
        )

        assert metrics.accuracy == 0.83
        assert metrics.precision["up"] == 0.88
        assert np.array_equal(metrics.confusion_matrix, cm)


class TestTradingMetrics:
    """Tests for TradingMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = TradingMetrics()

        assert metrics.sharpe_ratio == 0.0
        assert metrics.sortino_ratio == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.total_return == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.profit_factor == 0.0
        assert metrics.expectancy == 0.0


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_default_result(self):
        """Test default result values."""
        result = EvaluationResult()

        assert isinstance(result.direction_metrics, DirectionMetrics)
        assert isinstance(result.calibration_metrics, CalibrationMetrics)
        assert isinstance(result.trading_metrics, TradingMetrics)
        assert result.horizon_metrics == {}
        assert result.summary == {}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = EvaluationResult(
            direction_metrics=DirectionMetrics(accuracy=0.75),
            trading_metrics=TradingMetrics(sharpe_ratio=1.5),
        )

        data = result.to_dict()

        assert data["direction"]["accuracy"] == 0.75
        assert data["trading"]["sharpe_ratio"] == 1.5


class TestModelEvaluator:
    """Tests for ModelEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create an evaluator instance."""
        return ModelEvaluator(
            device="cpu",
            confidence_threshold=0.6,
            num_calibration_bins=10,
        )

    @pytest.fixture
    def mock_model(self):
        """Create a mock model."""
        model = MagicMock()
        model.eval = MagicMock()
        model.to = MagicMock(return_value=model)
        return model

    @pytest.fixture
    def sample_loader(self):
        """Create sample data loader."""
        X = torch.randn(100, 50, 10)
        y = torch.randint(0, 2, (100,)).float()
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=32)

    def test_init(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator(
            device="cpu",
            confidence_threshold=0.7,
            num_calibration_bins=15,
        )

        assert str(evaluator.device) == "cpu"
        assert evaluator.confidence_threshold == 0.7
        assert evaluator.num_calibration_bins == 15

    def test_calculate_direction_metrics_binary(self, evaluator):
        """Test direction metrics with binary labels."""
        predictions = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        targets = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])

        result = evaluator._calculate_direction_metrics(predictions, targets)

        assert 0 <= result.accuracy <= 1
        assert "down" in result.precision
        assert "up" in result.precision

    def test_calculate_direction_metrics_probabilities(self, evaluator):
        """Test direction metrics with probability predictions."""
        predictions = np.array([0.3, 0.7, 0.8, 0.4, 0.9, 0.2, 0.6, 0.55, 0.1, 0.45])
        targets = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])

        result = evaluator._calculate_direction_metrics(predictions, targets)

        # After thresholding at 0.5:
        # preds: [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
        # Expected accuracy: 7/10 = 0.7
        assert abs(result.accuracy - 0.7) < 0.01

    def test_calculate_direction_metrics_multiclass(self, evaluator):
        """Test direction metrics with 3-class predictions and binary targets."""
        # 3-class: 0=down, 1=neutral, 2=up -> maps to binary: 0=down, 1/2=up
        predictions = np.array([0, 1, 2, 0, 2, 0, 1, 1, 0, 0])
        targets = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])

        result = evaluator._calculate_direction_metrics(predictions, targets)

        # After mapping: preds = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
        # Expected accuracy: 7/10 = 0.7
        assert abs(result.accuracy - 0.7) < 0.01

    def test_calculate_calibration_metrics(self, evaluator):
        """Test calibration metrics calculation."""
        confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
        predictions = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        targets = np.array([1, 1, 0, 1, 0, 1, 0, 0, 0, 0])

        result = evaluator._calculate_calibration_metrics(
            confidences, predictions, targets
        )

        assert 0 <= result.ece <= 1
        assert 0 <= result.mce <= 1
        assert 0 <= result.brier_score <= 1

    def test_calculate_trading_metrics(self, evaluator):
        """Test trading metrics calculation."""
        predictions = np.array([1, 1, 0, 1, 1, 0, 0, 1, 1, 0])
        confidences = np.array([0.8, 0.7, 0.9, 0.6, 0.8, 0.7, 0.9, 0.6, 0.7, 0.8])

        # Create price series with some movement
        prices = pd.Series([100 + i + np.sin(i) for i in range(12)])

        result = evaluator._calculate_trading_metrics(predictions, confidences, prices)

        assert isinstance(result.sharpe_ratio, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.win_rate, float)
        assert 0 <= result.win_rate <= 1

    def test_calculate_trading_metrics_no_trades(self, evaluator):
        """Test trading metrics when no trades are made (low confidence)."""
        evaluator.confidence_threshold = 0.99  # Very high threshold

        predictions = np.array([1, 1, 0, 1, 1])
        confidences = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # All below threshold
        prices = pd.Series([100, 101, 102, 103, 104, 105, 106])

        result = evaluator._calculate_trading_metrics(predictions, confidences, prices)

        assert result.sharpe_ratio == 0.0
        assert result.win_rate == 0.0

    def test_class_name_binary(self, evaluator):
        """Test class name generation for binary classification."""
        assert evaluator._class_name(0, num_classes=2) == "down"
        assert evaluator._class_name(1, num_classes=2) == "up"

    def test_class_name_ternary(self, evaluator):
        """Test class name generation for ternary classification."""
        assert evaluator._class_name(0, num_classes=3) == "down"
        assert evaluator._class_name(1, num_classes=3) == "neutral"
        assert evaluator._class_name(2, num_classes=3) == "up"

    def test_generate_report(self, evaluator):
        """Test report generation."""
        result = EvaluationResult(
            direction_metrics=DirectionMetrics(
                accuracy=0.65,
                precision={"down": 0.6, "up": 0.7},
                recall={"down": 0.5, "up": 0.8},
                f1_score={"down": 0.55, "up": 0.75},
            ),
            calibration_metrics=CalibrationMetrics(
                ece=0.05,
                mce=0.1,
                brier_score=0.2,
            ),
            trading_metrics=TradingMetrics(
                sharpe_ratio=1.5,
                sortino_ratio=2.0,
                max_drawdown=0.1,
                total_return=0.15,
                win_rate=0.55,
                profit_factor=1.2,
                expectancy=0.002,
            ),
            summary={
                "meets_accuracy_target": True,
                "meets_sharpe_target": False,
                "meets_drawdown_target": True,
            },
        )

        report = evaluator.generate_report(result, "TestModel")

        assert "TestModel" in report
        assert "65.00%" in report
        assert "1.50" in report  # Sharpe
        assert "PASS" in report  # For accuracy target


class TestModelEvaluatorIntegration:
    """Integration tests for ModelEvaluator."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 3)  # 3 classes
                self.alpha = torch.nn.Linear(10, 1)
                self.beta = torch.nn.Linear(10, 1)

            def forward(self, x):
                # x: (batch, seq, features)
                last = x[:, -1, :]  # Take last timestep
                logits = self.fc(last)
                alpha = torch.nn.functional.softplus(self.alpha(last)) + 1
                beta = torch.nn.functional.softplus(self.beta(last)) + 1
                return {
                    "price": logits[:, :1],
                    "direction_logits": logits.unsqueeze(1),  # (batch, 1, 3)
                    "alpha": alpha,
                    "beta": beta,
                }

        return SimpleModel()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100

        X = torch.randn(n_samples, 20, 10)
        y = torch.randint(0, 2, (n_samples,)).float()
        prices = pd.Series([100 + i * 0.01 * np.random.randn() for i in range(n_samples + 10)])

        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32)

        return loader, prices

    def test_full_evaluation(self, simple_model, sample_data):
        """Test complete evaluation workflow."""
        loader, prices = sample_data
        evaluator = ModelEvaluator(device="cpu")

        result = evaluator.evaluate(
            model=simple_model,
            test_loader=loader,
            prices=prices,
        )

        assert isinstance(result, EvaluationResult)
        assert 0 <= result.direction_metrics.accuracy <= 1
        assert isinstance(result.trading_metrics.sharpe_ratio, float)
        assert "meets_accuracy_target" in result.summary


class TestEvaluateEnsemble:
    """Tests for evaluate_ensemble function."""

    @pytest.fixture
    def simple_models(self):
        """Create simple models for ensemble testing."""
        class SimpleModel(torch.nn.Module):
            def __init__(self, seed):
                super().__init__()
                torch.manual_seed(seed)
                self.fc = torch.nn.Linear(10, 1)

            def forward(self, x):
                return {"price": self.fc(x[:, -1, :])}

        return {
            "model_a": SimpleModel(1),
            "model_b": SimpleModel(2),
        }

    @pytest.fixture
    def sample_loaders(self):
        """Create sample data loaders."""
        X = torch.randn(50, 20, 10)
        y = torch.randint(0, 2, (50,)).float()
        dataset = TensorDataset(X, y)

        return {
            "model_a": DataLoader(dataset, batch_size=16),
            "model_b": DataLoader(dataset, batch_size=16),
        }

    def test_evaluate_ensemble(self, simple_models, sample_loaders):
        """Test ensemble evaluation."""
        prices = pd.Series([100 + i for i in range(60)])

        results = evaluate_ensemble(
            models=simple_models,
            test_loaders=sample_loaders,
            prices=prices,
        )

        assert "model_a" in results
        assert "model_b" in results
        assert isinstance(results["model_a"], EvaluationResult)
        assert isinstance(results["model_b"], EvaluationResult)
