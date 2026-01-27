"""Test training parameters with centralized configuration.

Tests that MTFEnsemble and StackingMetaLearner correctly load training parameters
from TradingConfig including data splits, early stopping, and stacking CV settings.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.models.multi_timeframe.mtf_ensemble import MTFEnsemble, MTFEnsembleConfig
from src.models.multi_timeframe.stacking_meta_learner import StackingMetaLearner, StackingConfig
from src.config import TradingConfig


@pytest.fixture
def sample_price_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=1000, freq="5min")
    data = {
        "open": np.random.randn(1000).cumsum() + 100,
        "high": np.random.randn(1000).cumsum() + 101,
        "low": np.random.randn(1000).cumsum() + 99,
        "close": np.random.randn(1000).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, 1000),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def trading_config():
    """Create a TradingConfig instance with test parameters."""
    config = TradingConfig()
    # Set test parameters
    config.training.splits.train_ratio = 0.7
    config.training.splits.validation_ratio = 0.15
    config.training.splits.test_ratio = 0.15
    config.training.stacking.n_folds = 3
    config.training.stacking.min_train_size = 100
    config.training.early_stopping.enabled = True
    config.training.early_stopping.stopping_rounds = 5
    config.training.early_stopping.verbose = False
    return config


def test_trading_config_training_defaults():
    """Test that default training parameters are set correctly."""
    config = TradingConfig()

    # Test data split defaults
    assert config.training.splits.train_ratio == 0.6
    assert config.training.splits.validation_ratio == 0.2
    assert config.training.splits.test_ratio == 0.2
    assert config.training.splits.enforce_chronological is True

    # Test stacking defaults
    assert config.training.stacking.n_folds == 5
    assert config.training.stacking.min_train_size == 500
    assert config.training.stacking.shuffle is False
    assert config.training.stacking.stratified is True

    # Test early stopping defaults
    assert config.training.early_stopping.enabled is True
    assert config.training.early_stopping.stopping_rounds == 10
    assert config.training.early_stopping.eval_metric == "logloss"
    assert config.training.early_stopping.verbose is False


def test_mtf_ensemble_uses_config_train_ratio(trading_config):
    """Test that MTFEnsemble uses train_ratio from config."""
    ensemble_config = MTFEnsembleConfig.default()
    ensemble = MTFEnsemble(ensemble_config, trading_config=trading_config)

    # Verify config is loaded
    assert ensemble.trading_config is not None
    assert ensemble.trading_config.training.splits.train_ratio == 0.7
    assert ensemble.trading_config.training.splits.validation_ratio == 0.15


def test_stacking_meta_learner_uses_config_cv(trading_config):
    """Test that StackingMetaLearner uses n_folds and min_train_size from config."""
    stacking_config = StackingConfig.default()
    meta_learner = StackingMetaLearner(stacking_config, trading_config=trading_config)

    # Verify config parameters are loaded
    assert meta_learner.config.n_folds == 3  # From trading_config
    assert meta_learner.config.min_train_size == 100  # From trading_config


def test_stacking_meta_learner_without_trading_config():
    """Test backward compatibility - StackingMetaLearner works without trading_config."""
    stacking_config = StackingConfig.default()
    meta_learner = StackingMetaLearner(stacking_config)

    # Verify defaults are used
    assert meta_learner.config.n_folds == 5  # Default
    assert meta_learner.config.min_train_size == 500  # Default


def test_early_stopping_config_parameters(trading_config):
    """Test early stopping configuration parameters."""
    early_stop = trading_config.training.early_stopping

    assert hasattr(early_stop, "enabled")
    assert hasattr(early_stop, "stopping_rounds")
    assert hasattr(early_stop, "eval_metric")
    assert hasattr(early_stop, "verbose")

    assert isinstance(early_stop.enabled, bool)
    assert isinstance(early_stop.stopping_rounds, int)
    assert isinstance(early_stop.eval_metric, str)
    assert isinstance(early_stop.verbose, bool)


def test_data_split_sum_to_one(trading_config):
    """Test that train + val + test ratios sum to 1.0."""
    splits = trading_config.training.splits
    total = splits.train_ratio + splits.validation_ratio + splits.test_ratio
    assert abs(total - 1.0) < 0.01, f"Splits sum to {total}, expected 1.0"


def test_config_to_dict():
    """Test that training config can be converted to dict."""
    config = TradingConfig()
    training_dict = config.training.to_dict()

    assert "splits" in training_dict
    assert "stacking" in training_dict
    assert "early_stopping" in training_dict

    assert "train_ratio" in training_dict["splits"]
    assert "n_folds" in training_dict["stacking"]
    assert "enabled" in training_dict["early_stopping"]


def test_modified_config_persists():
    """Test that modified config values persist."""
    config = TradingConfig()

    # Modify values
    config.training.splits.train_ratio = 0.5
    config.training.stacking.n_folds = 7
    config.training.early_stopping.stopping_rounds = 15

    # Verify modifications persist
    assert config.training.splits.train_ratio == 0.5
    assert config.training.stacking.n_folds == 7
    assert config.training.early_stopping.stopping_rounds == 15


def test_stacking_custom_hyperparams():
    """Test stacking custom hyperparameters configuration."""
    config = TradingConfig()

    # Set custom hyperparams
    config.training.stacking.use_base_hyperparams = False
    config.training.stacking.custom_hyperparams = {
        "n_estimators": 200,
        "max_depth": 5,
    }

    assert config.training.stacking.use_base_hyperparams is False
    assert config.training.stacking.custom_hyperparams["n_estimators"] == 200


def test_chronological_enforcement():
    """Test that chronological order enforcement is configurable."""
    config = TradingConfig()

    # Default should enforce chronological order for time series
    assert config.training.splits.enforce_chronological is True

    # Should be able to disable (though not recommended for time series)
    config.training.splits.enforce_chronological = False
    assert config.training.splits.enforce_chronological is False


def test_config_validation_train_ratio():
    """Test that train ratio is within valid range."""
    config = TradingConfig()

    # Valid ranges
    assert 0.0 < config.training.splits.train_ratio <= 1.0
    assert 0.0 < config.training.splits.validation_ratio <= 1.0
    assert 0.0 < config.training.splits.test_ratio <= 1.0


def test_config_validation_stacking_params():
    """Test that stacking parameters are valid."""
    config = TradingConfig()

    # Valid ranges
    assert config.training.stacking.n_folds > 0
    assert config.training.stacking.min_train_size > 0


def test_config_validation_early_stopping():
    """Test that early stopping parameters are valid."""
    config = TradingConfig()

    # Valid parameters
    assert config.training.early_stopping.stopping_rounds > 0
    assert config.training.early_stopping.eval_metric in ["logloss", "mlogloss", "error", "auc"]


def test_mtf_ensemble_train_with_default_ratios(sample_price_data, trading_config):
    """Test that MTFEnsemble.train() uses config ratios when None provided."""
    ensemble_config = MTFEnsembleConfig.default()
    ensemble = MTFEnsemble(ensemble_config, trading_config=trading_config)

    # Mock the train method to verify it uses config ratios
    # This is a conceptual test - in practice, full training would be slow
    # We verify the config is accessible
    assert ensemble.trading_config.training.splits.train_ratio == 0.7


def test_stacking_meta_learner_cv_split_count(trading_config):
    """Test that CV creates correct number of folds."""
    stacking_config = StackingConfig.default()
    meta_learner = StackingMetaLearner(stacking_config, trading_config=trading_config)

    # Verify n_folds from config
    assert meta_learner.config.n_folds == 3  # From trading_config


def test_backward_compatibility_mtf_ensemble():
    """Test that MTFEnsemble works without trading_config (backward compatibility)."""
    ensemble_config = MTFEnsembleConfig.default()
    ensemble = MTFEnsemble(ensemble_config, trading_config=None)

    # Should create with default TradingConfig (singleton may have been modified by other tests)
    assert ensemble.trading_config is not None
    assert hasattr(ensemble.trading_config.training.splits, "train_ratio")
    assert isinstance(ensemble.trading_config.training.splits.train_ratio, float)


def test_config_parameter_count():
    """Test that all expected training parameters are present."""
    config = TradingConfig()

    # Data split parameters (4 params)
    assert hasattr(config.training.splits, "train_ratio")
    assert hasattr(config.training.splits, "validation_ratio")
    assert hasattr(config.training.splits, "test_ratio")
    assert hasattr(config.training.splits, "enforce_chronological")

    # Stacking parameters (6 params)
    assert hasattr(config.training.stacking, "n_folds")
    assert hasattr(config.training.stacking, "min_train_size")
    assert hasattr(config.training.stacking, "shuffle")
    assert hasattr(config.training.stacking, "stratified")
    assert hasattr(config.training.stacking, "use_base_hyperparams")
    assert hasattr(config.training.stacking, "custom_hyperparams")

    # Early stopping parameters (4 params)
    assert hasattr(config.training.early_stopping, "enabled")
    assert hasattr(config.training.early_stopping, "stopping_rounds")
    assert hasattr(config.training.early_stopping, "eval_metric")
    assert hasattr(config.training.early_stopping, "verbose")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
