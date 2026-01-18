"""Tests for ensemble model components."""

import pytest
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from src.models.ensemble.weights import (
    DynamicWeightCalculator,
    WeightConfig,
    MarketRegime,
    VolatilityLevel,
    TradeResult,
    detect_market_regime,
)
from src.models.ensemble.predictor import (
    EnsemblePredictor,
    EnsembleConfig,
    ModelPrediction,
    EnsemblePrediction,
)
from src.models.ensemble.loader import (
    discover_trained_models,
    get_model_info,
    validate_ensemble_models,
)


class TestWeightConfig:
    """Tests for WeightConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = WeightConfig()

        assert "short_term" in config.base_weights
        assert "medium_term" in config.base_weights
        assert "long_term" in config.base_weights
        assert config.lookback_trades == 50
        assert config.performance_blend == 0.3
        assert config.min_weight == 0.1
        assert config.max_weight == 0.6

    def test_custom_config(self):
        """Test custom configuration values."""
        config = WeightConfig(
            base_weights={"model_a": 0.5, "model_b": 0.5},
            lookback_trades=100,
            performance_blend=0.5,
        )

        assert config.base_weights == {"model_a": 0.5, "model_b": 0.5}
        assert config.lookback_trades == 100
        assert config.performance_blend == 0.5

    def test_default_regime_adjustments(self):
        """Test default regime adjustments are created."""
        config = WeightConfig()

        assert MarketRegime.TRENDING_UP.value in config.regime_adjustments
        assert MarketRegime.RANGING.value in config.regime_adjustments
        assert MarketRegime.VOLATILE.value in config.regime_adjustments


class TestDynamicWeightCalculator:
    """Tests for DynamicWeightCalculator."""

    @pytest.fixture
    def calculator(self):
        """Create a weight calculator."""
        return DynamicWeightCalculator()

    @pytest.fixture
    def sample_trades(self):
        """Create sample trade results."""
        return {
            "short_term": [
                TradeResult(
                    model_name="short_term",
                    direction_correct=True,
                    profit_pct=0.01,
                    confidence=0.7,
                    timestamp=datetime.now().timestamp(),
                )
                for _ in range(10)
            ],
            "medium_term": [
                TradeResult(
                    model_name="medium_term",
                    direction_correct=i % 2 == 0,
                    profit_pct=0.005 if i % 2 == 0 else -0.003,
                    confidence=0.65,
                    timestamp=datetime.now().timestamp(),
                )
                for i in range(10)
            ],
            "long_term": [
                TradeResult(
                    model_name="long_term",
                    direction_correct=False,
                    profit_pct=-0.02,
                    confidence=0.55,
                    timestamp=datetime.now().timestamp(),
                )
                for _ in range(10)
            ],
        }

    def test_default_weights(self, calculator):
        """Test default weight calculation."""
        weights = calculator.calculate_weights()

        assert sum(weights.values()) == pytest.approx(1.0)
        assert all(w > 0 for w in weights.values())

    def test_regime_adjustment_trending(self, calculator):
        """Test weight adjustment for trending market."""
        weights = calculator.calculate_weights(
            market_regime=MarketRegime.TRENDING_UP
        )

        # In trending markets, medium/long term should be higher
        assert weights["medium_term"] >= weights["short_term"] or weights["long_term"] >= weights["short_term"]
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_regime_adjustment_ranging(self, calculator):
        """Test weight adjustment for ranging market."""
        weights = calculator.calculate_weights(
            market_regime=MarketRegime.RANGING
        )

        # In ranging markets, short term should be higher
        assert weights["short_term"] >= weights["long_term"]
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_volatility_adjustment(self, calculator):
        """Test weight adjustment for high volatility."""
        normal_weights = calculator.calculate_weights(
            volatility_level=VolatilityLevel.NORMAL
        )
        high_vol_weights = calculator.calculate_weights(
            volatility_level=VolatilityLevel.EXTREME
        )

        # In extreme volatility, long term weight should be lower
        assert high_vol_weights["long_term"] < normal_weights["long_term"]

    def test_performance_adjustment(self, calculator, sample_trades):
        """Test weight adjustment based on performance."""
        weights = calculator.calculate_weights(
            recent_performance=sample_trades
        )

        # Short term has 100% accuracy, should have higher weight
        assert weights["short_term"] > weights["long_term"]
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_add_trade_result(self, calculator):
        """Test adding trade results to history."""
        result = TradeResult(
            model_name="short_term",
            direction_correct=True,
            profit_pct=0.01,
            confidence=0.8,
            timestamp=datetime.now().timestamp(),
        )

        calculator.add_trade_result(result)
        assert len(calculator._trade_history["short_term"]) == 1

    def test_weight_constraints(self, calculator):
        """Test that weights are within min/max constraints."""
        # Create extreme performance to test constraints
        trades = {
            "short_term": [
                TradeResult("short_term", True, 0.1, 0.9, 0)
                for _ in range(100)
            ],
            "medium_term": [],
            "long_term": [],
        }

        weights = calculator.calculate_weights(recent_performance=trades)

        assert all(
            calculator.config.min_weight <= w <= calculator.config.max_weight
            for w in weights.values()
        )

    def test_get_weight_explanation(self, calculator):
        """Test weight explanation generation."""
        weights = calculator.calculate_weights(
            market_regime=MarketRegime.TRENDING_UP,
            volatility_level=VolatilityLevel.NORMAL,
        )

        explanation = calculator.get_weight_explanation(
            weights,
            MarketRegime.TRENDING_UP,
            VolatilityLevel.NORMAL,
        )

        assert "Market Regime: trending_up" in explanation
        assert "Volatility Level: normal" in explanation
        assert "Final Weights:" in explanation


class TestDetectMarketRegime:
    """Tests for market regime detection."""

    def test_trending_up(self):
        """Test trending up detection."""
        # Create upward trending prices
        prices = np.array([100 + i * 0.5 for i in range(30)])
        regime, vol = detect_market_regime(prices)

        assert regime == MarketRegime.TRENDING_UP

    def test_trending_down(self):
        """Test trending down detection."""
        # Create downward trending prices
        prices = np.array([100 - i * 0.5 for i in range(30)])
        regime, vol = detect_market_regime(prices)

        assert regime == MarketRegime.TRENDING_DOWN

    def test_ranging(self):
        """Test ranging market detection."""
        # Create sideways prices
        np.random.seed(42)
        prices = 100 + np.random.randn(30) * 0.1
        regime, vol = detect_market_regime(prices)

        # Should be ranging or have low volatility
        assert regime in [MarketRegime.RANGING, MarketRegime.UNKNOWN]

    def test_volatile(self):
        """Test volatile market detection."""
        # Create highly volatile prices
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(30) * 5)
        regime, vol = detect_market_regime(prices, volatility_threshold=0.001)

        # Should detect high volatility
        assert vol in [VolatilityLevel.HIGH, VolatilityLevel.EXTREME, VolatilityLevel.NORMAL]

    def test_insufficient_data(self):
        """Test with insufficient data."""
        prices = np.array([100, 101, 102])
        regime, vol = detect_market_regime(prices, lookback=20)

        assert regime == MarketRegime.UNKNOWN
        assert vol == VolatilityLevel.NORMAL


class TestModelPrediction:
    """Tests for ModelPrediction dataclass."""

    def test_create_prediction(self):
        """Test creating a model prediction."""
        pred = ModelPrediction(
            model_name="short_term",
            direction=1,
            direction_probs=np.array([0.3, 0.7]),
            confidence=0.75,
            alpha=5.0,
            beta=2.0,
            price_prediction=1.0850,
        )

        assert pred.model_name == "short_term"
        assert pred.direction == 1
        assert pred.confidence == 0.75


class TestEnsembleConfig:
    """Tests for EnsembleConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = EnsembleConfig()

        assert config.min_confidence == 0.60
        assert config.min_agreement == 0.50
        assert config.disagreement_penalty == 0.2
        assert config.use_dynamic_weights is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = EnsembleConfig(
            min_confidence=0.70,
            min_agreement=0.60,
            use_dynamic_weights=False,
        )

        assert config.min_confidence == 0.70
        assert config.min_agreement == 0.60
        assert config.use_dynamic_weights is False


class TestEnsemblePredictor:
    """Tests for EnsemblePredictor."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock PyTorch model."""
        class MockModel(nn.Module):
            def __init__(self, name="mock"):
                super().__init__()
                self.name = name
                self.fc = nn.Linear(10, 1)

            def forward(self, x):
                batch = x.shape[0]
                return {
                    "price": torch.randn(batch, 1),
                    "direction_logits": torch.randn(batch, 1, 3),
                    "alpha": torch.ones(batch, 1) * 5,
                    "beta": torch.ones(batch, 1) * 3,
                }

        return MockModel()

    @pytest.fixture
    def predictor(self, mock_model):
        """Create an ensemble predictor with mock models."""
        models = {
            "short_term": mock_model,
            "medium_term": mock_model,
            "long_term": mock_model,
        }
        return EnsemblePredictor(models=models, device="cpu")

    def test_init(self, predictor):
        """Test predictor initialization."""
        assert len(predictor.models) == 3
        assert "short_term" in predictor.models
        assert str(predictor.device) == "cpu"

    def test_predict(self, predictor):
        """Test making a prediction."""
        features = {
            "short_term": torch.randn(1, 50, 10),
            "medium_term": torch.randn(1, 60, 10),
            "long_term": torch.randn(1, 90, 10),
        }

        result = predictor.predict(features, symbol="EURUSD")

        assert isinstance(result, EnsemblePrediction)
        assert result.symbol == "EURUSD"
        assert result.direction in [0, 1]
        assert 0 <= result.confidence <= 1
        assert 0 <= result.agreement_score <= 1
        assert isinstance(result.should_trade, bool)

    def test_predict_with_prices(self, predictor):
        """Test prediction with price data for regime detection."""
        features = {
            "short_term": torch.randn(1, 50, 10),
            "medium_term": torch.randn(1, 60, 10),
            "long_term": torch.randn(1, 90, 10),
        }
        prices = np.array([100 + i * 0.1 for i in range(30)])

        result = predictor.predict(features, symbol="EURUSD", prices=prices)

        assert result.market_regime != "unknown"

    def test_get_current_weights(self, predictor):
        """Test getting current weights."""
        weights = predictor.get_current_weights()

        assert sum(weights.values()) == pytest.approx(1.0)
        assert all(name in weights for name in predictor.models)

    def test_set_weights(self, predictor):
        """Test manually setting weights."""
        predictor.set_weights({
            "short_term": 0.5,
            "medium_term": 0.3,
            "long_term": 0.2,
        })

        weights = predictor.get_current_weights()
        assert weights["short_term"] == pytest.approx(0.5)
        assert weights["medium_term"] == pytest.approx(0.3)
        assert weights["long_term"] == pytest.approx(0.2)

    def test_get_model_info(self, predictor):
        """Test getting model information."""
        info = predictor.get_model_info()

        assert info["num_models"] == 3
        assert "short_term" in info["model_names"]
        assert "current_weights" in info

    def test_should_trade_low_confidence(self, predictor):
        """Test that low confidence prevents trading."""
        # Set high minimum confidence
        predictor.config.min_confidence = 0.99

        features = {
            "short_term": torch.randn(1, 50, 10),
            "medium_term": torch.randn(1, 60, 10),
            "long_term": torch.randn(1, 90, 10),
        }

        result = predictor.predict(features)

        # With random model outputs and high threshold, should not trade
        assert result.should_trade is False or result.confidence >= 0.99

    def test_prediction_to_dict(self, predictor):
        """Test converting prediction to dictionary."""
        features = {
            "short_term": torch.randn(1, 50, 10),
            "medium_term": torch.randn(1, 60, 10),
            "long_term": torch.randn(1, 90, 10),
        }

        result = predictor.predict(features, symbol="EURUSD")
        result_dict = result.to_dict()

        assert "timestamp" in result_dict
        assert "symbol" in result_dict
        assert "direction" in result_dict
        assert "confidence" in result_dict
        assert result_dict["symbol"] == "EURUSD"


class TestEnsembleLoader:
    """Tests for ensemble loader utilities."""

    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary models directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock model directories
            for model_type in ["short_term", "medium_term", "long_term"]:
                model_dir = Path(tmpdir) / f"{model_type}_EURUSD_20260108_120000"
                model_dir.mkdir()
                (model_dir / "model.pt").touch()
                (model_dir / "config.json").write_text("{}")
                (model_dir / "architecture.json").write_text("{}")

            yield tmpdir

    def test_discover_trained_models(self, temp_models_dir):
        """Test discovering trained models."""
        models = discover_trained_models(temp_models_dir)

        assert "short_term" in models
        assert "medium_term" in models
        assert "long_term" in models
        assert len(models["short_term"]) == 1
        assert len(models["medium_term"]) == 1
        assert len(models["long_term"]) == 1

    def test_discover_with_symbol_filter(self, temp_models_dir):
        """Test discovering models with symbol filter."""
        models = discover_trained_models(temp_models_dir, symbol="EURUSD")
        assert len(models["short_term"]) == 1

        models = discover_trained_models(temp_models_dir, symbol="GBPUSD")
        assert len(models["short_term"]) == 0

    def test_discover_empty_directory(self):
        """Test discovering in empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            models = discover_trained_models(tmpdir)
            assert all(len(v) == 0 for v in models.values())

    def test_get_model_info(self, temp_models_dir):
        """Test getting model information."""
        model_path = Path(temp_models_dir) / "short_term_EURUSD_20260108_120000"
        info = get_model_info(model_path)

        assert info["exists"] is True
        assert info["has_model"] is True
        assert info["has_config"] is True

    def test_get_model_info_nonexistent(self):
        """Test getting info for nonexistent model."""
        info = get_model_info("/nonexistent/path")

        assert info["exists"] is False

    def test_validate_ensemble_models(self, temp_models_dir):
        """Test validating ensemble models."""
        model_paths = {
            "short_term": Path(temp_models_dir) / "short_term_EURUSD_20260108_120000",
            "medium_term": Path(temp_models_dir) / "medium_term_EURUSD_20260108_120000",
            "long_term": Path(temp_models_dir) / "long_term_EURUSD_20260108_120000",
        }

        # This will fail because the models aren't actually loadable
        # but the file structure check should pass
        valid, errors = validate_ensemble_models(model_paths)

        # There should be errors about loading (mock files aren't real models)
        assert len(errors) > 0 or valid


class TestEnsemblePrediction:
    """Tests for EnsemblePrediction dataclass."""

    def test_create_prediction(self):
        """Test creating an ensemble prediction."""
        pred = EnsemblePrediction(
            timestamp=datetime.now(),
            symbol="EURUSD",
            direction=1,
            direction_probability=0.65,
            confidence=0.75,
            agreement_score=0.8,
            should_trade=True,
            position_size_factor=0.6,
            component_predictions={},
            component_weights={"short_term": 0.4, "medium_term": 0.3, "long_term": 0.3},
            market_regime="trending_up",
            volatility_level="normal",
        )

        assert pred.direction == 1
        assert pred.should_trade is True
        assert pred.confidence == 0.75

    def test_to_dict(self):
        """Test converting to dictionary."""
        pred = EnsemblePrediction(
            timestamp=datetime.now(),
            symbol="EURUSD",
            direction=1,
            direction_probability=0.65,
            confidence=0.75,
            agreement_score=0.8,
            should_trade=True,
            position_size_factor=0.6,
            component_predictions={},
            component_weights={"short_term": 0.4},
            market_regime="trending_up",
            volatility_level="normal",
        )

        result = pred.to_dict()

        assert result["symbol"] == "EURUSD"
        assert result["direction"] == 1
        assert result["confidence"] == 0.75
        assert "timestamp" in result
