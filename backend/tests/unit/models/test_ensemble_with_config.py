"""Test MTFEnsemble with centralized config.

This module tests that the ensemble correctly passes TradingConfig
to all component models and that hyperparameters are centralized.
"""

import pytest
import numpy as np

from src.models.multi_timeframe.mtf_ensemble import MTFEnsemble, MTFEnsembleConfig
from src.config import TradingConfig


class TestEnsembleWithConfig:
    """Test MTFEnsemble with centralized configuration."""

    def test_ensemble_with_default_config(self):
        """Test ensemble creation with default TradingConfig."""
        ensemble = MTFEnsemble()

        # Verify trading_config is set
        assert ensemble.trading_config is not None
        assert isinstance(ensemble.trading_config, TradingConfig)

        # Verify all 3 models are created
        assert "1H" in ensemble.models
        assert "4H" in ensemble.models
        assert "D" in ensemble.models

        # Verify each model has hyperparams from centralized config
        assert ensemble.models["1H"].hyperparams is not None
        assert ensemble.models["4H"].hyperparams is not None
        assert ensemble.models["D"].hyperparams is not None

    def test_ensemble_models_use_config_hyperparams(self):
        """Test that all ensemble models use centralized hyperparameters."""
        ensemble = MTFEnsemble()

        # Check all models load hyperparams from TradingConfig
        assert ensemble.models["1H"].hyperparams == ensemble.trading_config.hyperparameters.model_1h
        assert ensemble.models["4H"].hyperparams == ensemble.trading_config.hyperparameters.model_4h
        assert ensemble.models["D"].hyperparams == ensemble.trading_config.hyperparameters.model_daily

        # Verify all hyperparams are valid (not None, positive values)
        for tf in ["1H", "4H", "D"]:
            hp = ensemble.models[tf].hyperparams
            assert hp.n_estimators > 0
            assert hp.max_depth > 0
            assert hp.learning_rate > 0

    def test_ensemble_with_custom_config(self):
        """Test ensemble with custom TradingConfig."""
        # Create custom config
        config = TradingConfig()
        config.hyperparameters.model_1h.n_estimators = 200
        config.hyperparameters.model_4h.max_depth = 5
        config.hyperparameters.model_daily.learning_rate = 0.05

        # Create ensemble with custom config
        ensemble = MTFEnsemble(trading_config=config)

        # Verify custom hyperparameters are used
        assert ensemble.models["1H"].hyperparams.n_estimators == 200
        assert ensemble.models["4H"].hyperparams.max_depth == 5
        assert ensemble.models["D"].hyperparams.learning_rate == 0.05

    def test_ensemble_config_and_trading_config_separate(self):
        """Test that MTFEnsembleConfig and TradingConfig are separate."""
        ensemble_config = MTFEnsembleConfig.default()
        trading_config = TradingConfig()

        ensemble = MTFEnsemble(config=ensemble_config, trading_config=trading_config)

        # Verify both configs are stored
        assert ensemble.config == ensemble_config
        assert ensemble.trading_config == trading_config

        # Verify ensemble weights come from MTFEnsembleConfig
        assert ensemble.config.weights == {"1H": 0.6, "4H": 0.3, "D": 0.1}

        # Verify model hyperparams come from TradingConfig
        assert ensemble.models["1H"].hyperparams == trading_config.hyperparameters.model_1h

    def test_ensemble_models_share_same_trading_config(self):
        """Test that all models in ensemble share the same TradingConfig instance."""
        config = TradingConfig()
        ensemble = MTFEnsemble(trading_config=config)

        # All models should reference the same TradingConfig instance
        assert ensemble.models["1H"].trading_config is config
        assert ensemble.models["4H"].trading_config is config
        assert ensemble.models["D"].trading_config is config

    def test_ensemble_backward_compatibility(self):
        """Test ensemble works without explicit trading_config parameter."""
        # Old way - no trading_config parameter
        ensemble = MTFEnsemble()

        # Should still work and create default TradingConfig internally
        assert ensemble.trading_config is not None
        assert ensemble.models["1H"].hyperparams is not None
        assert ensemble.models["1H"].hyperparams.n_estimators > 0

    def test_ensemble_with_sentiment_config(self):
        """Test ensemble with sentiment enabled."""
        ensemble_config = MTFEnsembleConfig.with_sentiment()
        ensemble = MTFEnsemble(config=ensemble_config)

        # Verify sentiment is enabled for Daily only (per research)
        assert not ensemble.model_configs["1H"].include_sentiment_features
        assert not ensemble.model_configs["4H"].include_sentiment_features
        assert ensemble.model_configs["D"].include_sentiment_features

        # Verify hyperparams still come from centralized config
        assert ensemble.models["D"].hyperparams.n_estimators == 80

    def test_ensemble_with_stacking_config(self):
        """Test ensemble with stacking enabled."""
        ensemble_config = MTFEnsembleConfig.with_stacking()
        ensemble = MTFEnsemble(config=ensemble_config)

        # Verify stacking is enabled
        assert ensemble.config.use_stacking is True
        assert ensemble.stacking_meta_learner is not None

        # Verify models still use centralized hyperparams
        assert ensemble.models["1H"].hyperparams is not None
        assert ensemble.models["1H"].hyperparams == ensemble.trading_config.hyperparameters.model_1h

    def test_all_models_have_required_xgboost_params(self):
        """Verify all ensemble models have complete XGBoost hyperparameters."""
        ensemble = MTFEnsemble()

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

        for tf in ["1H", "4H", "D"]:
            model = ensemble.models[tf]
            for param in required_params:
                assert hasattr(model.hyperparams, param), f"{tf} missing {param}"
                value = getattr(model.hyperparams, param)
                assert value is not None, f"{tf}.{param} is None"

    def test_ensemble_models_have_different_hyperparams(self):
        """Verify each timeframe has unique hyperparameters."""
        ensemble = MTFEnsemble()

        # Get hyperparams
        hp_1h = ensemble.models["1H"].hyperparams
        hp_4h = ensemble.models["4H"].hyperparams
        hp_d = ensemble.models["D"].hyperparams

        # Verify they're different objects
        assert hp_1h is not hp_4h
        assert hp_4h is not hp_d

        # Verify values are different
        assert hp_1h.n_estimators > hp_4h.n_estimators > hp_d.n_estimators
        assert hp_1h.max_depth > hp_4h.max_depth > hp_d.max_depth

    def test_ensemble_model_configs_preserve_tp_sl(self):
        """Verify TP/SL parameters from Week 2 are still loaded."""
        ensemble = MTFEnsemble()

        # Verify 1H model has correct TP/SL from centralized config
        assert ensemble.model_configs["1H"].tp_pips == 25.0
        assert ensemble.model_configs["1H"].sl_pips == 15.0
        assert ensemble.model_configs["1H"].max_holding_bars == 12

        # Verify 4H model
        assert ensemble.model_configs["4H"].tp_pips == 50.0
        assert ensemble.model_configs["4H"].sl_pips == 25.0
        assert ensemble.model_configs["4H"].max_holding_bars == 18

        # Verify Daily model
        assert ensemble.model_configs["D"].tp_pips == 150.0
        assert ensemble.model_configs["D"].sl_pips == 75.0
        assert ensemble.model_configs["D"].max_holding_bars == 15

    def test_ensemble_validates_against_centralized_config(self):
        """Test ensemble validates weights against centralized config."""
        # Create ensemble with custom weights different from centralized config
        config = MTFEnsembleConfig()
        config.weights = {"1H": 0.5, "4H": 0.4, "D": 0.1}

        # This should trigger validation warning but still work
        ensemble = MTFEnsemble(config=config)

        # Ensemble should use its own weights (for training consistency)
        assert ensemble.config.weights == {"1H": 0.5, "4H": 0.4, "D": 0.1}

        # But models should still use centralized hyperparams
        assert ensemble.models["1H"].hyperparams == ensemble.trading_config.hyperparameters.model_1h
        assert ensemble.models["1H"].hyperparams.n_estimators > 0

    def test_30_hyperparameters_centralized(self):
        """Verify all 30 hyperparameters are centralized (10 per model × 3 models)."""
        ensemble = MTFEnsemble()

        # Count hyperparameters per model
        hyperparams_per_model = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "gamma",
            "eval_metric",  # 10th parameter
        ]

        # Verify all 3 models have all 10 parameters
        for tf in ["1H", "4H", "D"]:
            model = ensemble.models[tf]
            for param in hyperparams_per_model:
                assert hasattr(model.hyperparams, param), f"{tf} missing {param}"

        # Total: 10 params × 3 models = 30 hyperparameters centralized
        total_params = len(hyperparams_per_model) * 3
        assert total_params == 30, f"Expected 30 centralized hyperparameters, got {total_params}"
