"""Unit tests for model hyperparameter configuration."""

import pytest
from src.config.model_config import XGBoostHyperparameters, ModelHyperparameters


class TestXGBoostHyperparameters:
    """Test XGBoost hyperparameters dataclass."""

    def test_creation(self):
        """Test creating XGBoost hyperparameters."""
        hyp = XGBoostHyperparameters(
            n_estimators=150, max_depth=5, learning_rate=0.03
        )

        assert hyp.n_estimators == 150
        assert hyp.max_depth == 5
        assert hyp.learning_rate == 0.03
        assert hyp.min_child_weight == 3  # Default
        assert hyp.subsample == 0.8  # Default
        assert hyp.colsample_bytree == 0.8  # Default
        assert hyp.reg_alpha == 0.1  # Default
        assert hyp.reg_lambda == 1.0  # Default
        assert hyp.gamma == 0.1  # Default
        assert hyp.eval_metric == "logloss"  # Default
        assert hyp.random_state == 42  # Default

    def test_to_dict(self):
        """Test conversion to dictionary."""
        hyp = XGBoostHyperparameters(
            n_estimators=150, max_depth=5, learning_rate=0.03
        )
        result = hyp.to_dict()

        assert isinstance(result, dict)
        assert result["n_estimators"] == 150
        assert result["max_depth"] == 5
        assert result["learning_rate"] == 0.03


class TestModelHyperparameters:
    """Test model hyperparameters for all timeframes."""

    def test_defaults(self):
        """Test model hyperparameters load with correct defaults."""
        config = ModelHyperparameters()

        # 1H Model defaults
        assert config.model_1h.n_estimators == 150
        assert config.model_1h.max_depth == 5
        assert config.model_1h.learning_rate == 0.03

        # 4H Model defaults
        assert config.model_4h.n_estimators == 120
        assert config.model_4h.max_depth == 4
        assert config.model_4h.learning_rate == 0.03

        # Daily Model defaults
        assert config.model_daily.n_estimators == 80
        assert config.model_daily.max_depth == 3
        assert config.model_daily.learning_rate == 0.03

    def test_override_1h(self):
        """Test 1H model hyperparameters can be overridden."""
        config = ModelHyperparameters()
        config.model_1h.n_estimators = 200
        config.model_1h.max_depth = 6

        assert config.model_1h.n_estimators == 200
        assert config.model_1h.max_depth == 6

    def test_override_4h(self):
        """Test 4H model hyperparameters can be overridden."""
        config = ModelHyperparameters()
        config.model_4h.n_estimators = 150
        config.model_4h.learning_rate = 0.01

        assert config.model_4h.n_estimators == 150
        assert config.model_4h.learning_rate == 0.01

    def test_override_daily(self):
        """Test Daily model hyperparameters can be overridden."""
        config = ModelHyperparameters()
        config.model_daily.max_depth = 4
        config.model_daily.subsample = 0.9

        assert config.model_daily.max_depth == 4
        assert config.model_daily.subsample == 0.9

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ModelHyperparameters()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "model_1h" in result
        assert "model_4h" in result
        assert "model_daily" in result
        assert isinstance(result["model_1h"], dict)
        assert result["model_1h"]["n_estimators"] == 150
