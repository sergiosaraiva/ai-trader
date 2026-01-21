"""Tests for RFECVConfig dataclass."""

import pytest

from src.models.feature_selection.rfecv_config import RFECVConfig


class TestRFECVConfig:
    """Tests for RFECV configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RFECVConfig()

        # Feature elimination settings
        assert config.step == 0.1
        assert config.min_features_to_select == 20

        # Cross-validation settings
        assert config.cv == 5

        # XGBoost estimator parameters
        assert config.n_estimators == 100
        assert config.max_depth == 4
        assert config.learning_rate == 0.1

        # Evaluation settings
        assert config.scoring == "accuracy"
        assert config.n_jobs == -1
        assert config.verbose == 1

        # Caching settings
        assert config.cache_enabled is True
        assert config.cache_dir == "models/feature_selections"

        # Reproducibility
        assert config.random_state == 42

    def test_custom_config(self):
        """Test custom configuration."""
        config = RFECVConfig(
            step=0.2,
            min_features_to_select=10,
            cv=3,
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            scoring="roc_auc",
            n_jobs=4,
            verbose=0,
            cache_enabled=False,
            cache_dir="custom/cache/dir",
            random_state=123,
        )

        assert config.step == 0.2
        assert config.min_features_to_select == 10
        assert config.cv == 3
        assert config.n_estimators == 50
        assert config.max_depth == 3
        assert config.learning_rate == 0.05
        assert config.scoring == "roc_auc"
        assert config.n_jobs == 4
        assert config.verbose == 0
        assert config.cache_enabled is False
        assert config.cache_dir == "custom/cache/dir"
        assert config.random_state == 123

    def test_step_validation_range(self):
        """Test that step values are in valid range (0.0-1.0)."""
        # Valid step values
        config1 = RFECVConfig(step=0.05)
        assert config1.step == 0.05

        config2 = RFECVConfig(step=1.0)
        assert config2.step == 1.0

        config3 = RFECVConfig(step=0.5)
        assert config3.step == 0.5

        # Note: Dataclass doesn't enforce validation, but we document expected range

    def test_min_features_to_select_positive(self):
        """Test that min_features_to_select is positive."""
        config = RFECVConfig(min_features_to_select=1)
        assert config.min_features_to_select == 1

        config2 = RFECVConfig(min_features_to_select=100)
        assert config2.min_features_to_select == 100

    def test_cv_folds_positive(self):
        """Test that cv folds is positive."""
        config = RFECVConfig(cv=2)
        assert config.cv == 2

        config2 = RFECVConfig(cv=10)
        assert config2.cv == 10

    def test_config_is_dataclass(self):
        """Test that RFECVConfig is a dataclass."""
        config = RFECVConfig()
        assert hasattr(config, "__dataclass_fields__")

    def test_config_immutability(self):
        """Test that config fields can be modified (not frozen)."""
        config = RFECVConfig()
        config.step = 0.15
        assert config.step == 0.15

    def test_cache_enabled_true_by_default(self):
        """Test that caching is enabled by default."""
        config = RFECVConfig()
        assert config.cache_enabled is True

    def test_cache_disabled(self):
        """Test that caching can be disabled."""
        config = RFECVConfig(cache_enabled=False)
        assert config.cache_enabled is False

    def test_reproducibility_seed(self):
        """Test that random_state is set for reproducibility."""
        config = RFECVConfig()
        assert config.random_state == 42

    def test_parallel_processing_enabled(self):
        """Test that parallel processing is enabled by default."""
        config = RFECVConfig()
        assert config.n_jobs == -1  # Use all cores
