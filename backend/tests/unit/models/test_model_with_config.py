"""Test ImprovedTimeframeModel with centralized config.

This module tests that models correctly load hyperparameters from
TradingConfig and can be created using factory methods.
"""

import pytest
import numpy as np

from src.models.multi_timeframe.improved_model import (
    ImprovedTimeframeModel,
    ImprovedModelConfig,
)
from src.config import TradingConfig


class TestModelWithConfig:
    """Test ImprovedTimeframeModel with centralized configuration."""

    def test_1h_model_with_default_config(self):
        """Test 1H model loads hyperparameters from default TradingConfig."""
        # Create model using factory method
        model = ImprovedTimeframeModel.create_1h_model()

        # Verify trading_config is set
        assert model.trading_config is not None
        assert isinstance(model.trading_config, TradingConfig)

        # Verify hyperparams are loaded from centralized config
        assert model.hyperparams is not None
        assert model.hyperparams == model.trading_config.hyperparameters.model_1h

        # Verify default values match model_config.py
        assert model.hyperparams.n_estimators == 150
        assert model.hyperparams.max_depth == 5
        assert model.hyperparams.learning_rate == 0.03

    def test_1h_model_with_custom_config(self):
        """Test 1H model with custom TradingConfig."""
        # Create custom config
        config = TradingConfig()
        config.hyperparameters.model_1h.n_estimators = 200
        config.hyperparameters.model_1h.max_depth = 6

        # Create model with custom config
        model = ImprovedTimeframeModel.create_1h_model(trading_config=config)

        # Verify custom hyperparameters are used
        assert model.hyperparams.n_estimators == 200
        assert model.hyperparams.max_depth == 6
        assert model.hyperparams.learning_rate == 0.03  # Unchanged

    def test_4h_model_with_default_config(self):
        """Test 4H model loads hyperparameters from default TradingConfig."""
        model = ImprovedTimeframeModel.create_4h_model()

        # Verify hyperparams match centralized config
        assert model.hyperparams is not None
        assert model.hyperparams == model.trading_config.hyperparameters.model_4h

        # Verify default values
        assert model.hyperparams.n_estimators == 120
        assert model.hyperparams.max_depth == 4
        assert model.hyperparams.learning_rate == 0.03

    def test_4h_model_with_custom_config(self):
        """Test 4H model with custom TradingConfig."""
        config = TradingConfig()
        config.hyperparameters.model_4h.n_estimators = 100
        config.hyperparameters.model_4h.learning_rate = 0.05

        model = ImprovedTimeframeModel.create_4h_model(trading_config=config)

        assert model.hyperparams.n_estimators == 100
        assert model.hyperparams.max_depth == 4  # Unchanged
        assert model.hyperparams.learning_rate == 0.05

    def test_daily_model_with_default_config(self):
        """Test Daily model loads hyperparameters from default TradingConfig."""
        model = ImprovedTimeframeModel.create_daily_model()

        # Verify hyperparams match centralized config
        assert model.hyperparams is not None
        assert model.hyperparams == model.trading_config.hyperparameters.model_daily

        # Verify default values
        assert model.hyperparams.n_estimators == 80
        assert model.hyperparams.max_depth == 3
        assert model.hyperparams.learning_rate == 0.03

    def test_daily_model_with_custom_config(self):
        """Test Daily model with custom TradingConfig."""
        config = TradingConfig()
        config.hyperparameters.model_daily.max_depth = 4
        config.hyperparameters.model_daily.subsample = 0.9

        model = ImprovedTimeframeModel.create_daily_model(trading_config=config)

        assert model.hyperparams.n_estimators == 80  # Unchanged
        assert model.hyperparams.max_depth == 4
        assert model.hyperparams.subsample == 0.9

    def test_model_hyperparams_have_all_xgboost_params(self):
        """Verify hyperparams object has all required XGBoost parameters."""
        model = ImprovedTimeframeModel.create_1h_model()

        # Verify all required parameters exist
        required_params = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "gamma",
        ]

        for param in required_params:
            assert hasattr(model.hyperparams, param), f"Missing parameter: {param}"
            value = getattr(model.hyperparams, param)
            assert value is not None, f"Parameter {param} is None"

    def test_factory_method_kwargs_override(self):
        """Test factory methods accept kwargs for ImprovedModelConfig overrides."""
        # Create fresh config to avoid test pollution
        config = TradingConfig()

        # Create model with kwargs override
        model = ImprovedTimeframeModel.create_1h_model(
            trading_config=config,
            include_sentiment_features=True,
            use_calibration=True
        )

        # Verify kwargs were applied to ImprovedModelConfig
        assert model.config.include_sentiment_features is True
        assert model.config.use_calibration is True

        # Verify hyperparams come from TradingConfig (check they're loaded, not specific value)
        assert model.hyperparams is not None
        assert model.hyperparams == config.hyperparameters.model_1h
        assert hasattr(model.hyperparams, 'n_estimators')

    def test_legacy_direct_instantiation(self):
        """Test backward compatibility with direct ImprovedTimeframeModel instantiation."""
        # Create fresh trading config to avoid pollution
        trading_cfg = TradingConfig()

        # Create model the old way (without factory method, but pass trading_config)
        config = ImprovedModelConfig.hourly_model()
        model = ImprovedTimeframeModel(config, trading_config=trading_cfg)

        # Should still work and load centralized hyperparams
        assert model.hyperparams is not None
        assert model.hyperparams == trading_cfg.hyperparameters.model_1h
        assert model.hyperparams.n_estimators > 0  # Verifies it's loaded

    def test_optimized_hyperparams_override(self):
        """Test that config.hyperparams dict overrides centralized config."""
        # Create config with optimized hyperparams override
        config = ImprovedModelConfig.hourly_model()
        config.hyperparams = {
            "n_estimators": 250,
            "max_depth": 7,
            "learning_rate": 0.01,
        }

        model = ImprovedTimeframeModel(config)

        # Should use dict hyperparams (Priority 1)
        assert isinstance(model.hyperparams, dict)
        assert model.hyperparams["n_estimators"] == 250
        assert model.hyperparams["max_depth"] == 7

    def test_model_creation_without_errors(self):
        """Test that models can be created without errors."""
        # Create fresh config to avoid test pollution
        config = TradingConfig()

        # This tests the _create_model() method integration
        model = ImprovedTimeframeModel.create_1h_model(trading_config=config)

        # Create XGBoost model internally
        xgb_model = model._create_model()

        # Verify it's an XGBoost classifier
        from xgboost import XGBClassifier
        assert isinstance(xgb_model, XGBClassifier)

        # Verify hyperparameters were applied from TradingConfig
        assert xgb_model.n_estimators == config.hyperparameters.model_1h.n_estimators
        assert xgb_model.max_depth == config.hyperparameters.model_1h.max_depth
        assert xgb_model.learning_rate == config.hyperparameters.model_1h.learning_rate

    def test_different_timeframes_have_different_hyperparams(self):
        """Verify each timeframe model has unique hyperparameters."""
        # Use fresh config to avoid test pollution
        config = TradingConfig()
        model_1h = ImprovedTimeframeModel.create_1h_model(trading_config=config)
        model_4h = ImprovedTimeframeModel.create_4h_model(trading_config=config)
        model_daily = ImprovedTimeframeModel.create_daily_model(trading_config=config)

        # Verify they're different objects
        assert model_1h.hyperparams is not model_4h.hyperparams
        assert model_4h.hyperparams is not model_daily.hyperparams

        # Verify each model loads its corresponding hyperparams from config
        assert model_1h.hyperparams == config.hyperparameters.model_1h
        assert model_4h.hyperparams == config.hyperparameters.model_4h
        assert model_daily.hyperparams == config.hyperparameters.model_daily

        # Verify n_estimators are different (should generally decrease with timeframe)
        assert model_1h.hyperparams.n_estimators != model_4h.hyperparams.n_estimators
        assert model_1h.hyperparams.n_estimators > 0  # Sanity check

    def test_model_config_still_has_tp_sl_from_centralized_config(self):
        """Verify TP/SL parameters are loaded from centralized config (Week 2)."""
        model = ImprovedTimeframeModel.create_1h_model()

        # TP/SL should come from centralized config (already implemented)
        assert model.config.tp_pips == 25.0  # From TradingConfig.timeframes["1H"]
        assert model.config.sl_pips == 15.0
        assert model.config.max_holding_bars == 12
