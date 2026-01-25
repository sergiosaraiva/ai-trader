"""Tests for gradient boosting framework support (XGBoost, LightGBM, CatBoost)."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from src.models.multi_timeframe.improved_model import (
    ImprovedModelConfig,
    ImprovedTimeframeModel,
    HAS_LIGHTGBM,
    HAS_CATBOOST,
)


class TestGradientBoostingFrameworks:
    """Tests for multiple gradient boosting framework support."""

    def test_config_default_model_type_is_xgboost(self):
        """Default model type should be xgboost."""
        config = ImprovedModelConfig(name="test", base_timeframe="1H")
        assert config.model_type == "xgboost"

    def test_config_accepts_lightgbm_model_type(self):
        """Config should accept lightgbm as model type."""
        config = ImprovedModelConfig(
            name="test",
            base_timeframe="1H",
            model_type="lightgbm"
        )
        assert config.model_type == "lightgbm"

    def test_config_accepts_catboost_model_type(self):
        """Config should accept catboost as model type."""
        config = ImprovedModelConfig(
            name="test",
            base_timeframe="1H",
            model_type="catboost"
        )
        assert config.model_type == "catboost"

    def test_xgboost_model_creation(self):
        """XGBoost model should be created correctly."""
        from xgboost import XGBClassifier

        config = ImprovedModelConfig(
            name="test",
            base_timeframe="1H",
            model_type="xgboost"
        )
        model_wrapper = ImprovedTimeframeModel(config)
        model = model_wrapper._create_model()

        assert isinstance(model, XGBClassifier)

    @pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
    def test_lightgbm_model_creation(self):
        """LightGBM model should be created correctly when installed."""
        from lightgbm import LGBMClassifier

        config = ImprovedModelConfig(
            name="test",
            base_timeframe="1H",
            model_type="lightgbm"
        )
        model_wrapper = ImprovedTimeframeModel(config)
        model = model_wrapper._create_model()

        assert isinstance(model, LGBMClassifier)

    @pytest.mark.skipif(not HAS_CATBOOST, reason="CatBoost not installed")
    def test_catboost_model_creation(self):
        """CatBoost model should be created correctly when installed."""
        from catboost import CatBoostClassifier

        config = ImprovedModelConfig(
            name="test",
            base_timeframe="1H",
            model_type="catboost"
        )
        model_wrapper = ImprovedTimeframeModel(config)
        model = model_wrapper._create_model()

        assert isinstance(model, CatBoostClassifier)

    @pytest.mark.skipif(HAS_LIGHTGBM, reason="Test only when LightGBM is not installed")
    def test_lightgbm_raises_import_error_when_not_installed(self):
        """Should raise ImportError when LightGBM not installed."""
        config = ImprovedModelConfig(
            name="test",
            base_timeframe="1H",
            model_type="lightgbm"
        )
        model_wrapper = ImprovedTimeframeModel(config)

        with pytest.raises(ImportError) as exc_info:
            model_wrapper._create_model()

        assert "LightGBM is not installed" in str(exc_info.value)

    @pytest.mark.skipif(HAS_CATBOOST, reason="Test only when CatBoost is not installed")
    def test_catboost_raises_import_error_when_not_installed(self):
        """Should raise ImportError when CatBoost not installed."""
        config = ImprovedModelConfig(
            name="test",
            base_timeframe="1H",
            model_type="catboost"
        )
        model_wrapper = ImprovedTimeframeModel(config)

        with pytest.raises(ImportError) as exc_info:
            model_wrapper._create_model()

        assert "CatBoost is not installed" in str(exc_info.value)

    def test_invalid_model_type_raises_error(self):
        """Invalid model type should raise ValueError."""
        config = ImprovedModelConfig(
            name="test",
            base_timeframe="1H",
            model_type="invalid_framework"
        )
        model_wrapper = ImprovedTimeframeModel(config)

        with pytest.raises(ValueError) as exc_info:
            model_wrapper._create_model()

        assert "Unknown model type" in str(exc_info.value)


class TestXGBoostBackwardCompatibility:
    """Tests to ensure XGBoost backward compatibility."""

    def test_default_hourly_model_uses_xgboost(self):
        """Default hourly model should use XGBoost."""
        config = ImprovedModelConfig.hourly_model()
        assert config.model_type == "xgboost"

    def test_default_four_hour_model_uses_xgboost(self):
        """Default 4H model should use XGBoost."""
        config = ImprovedModelConfig.four_hour_model()
        assert config.model_type == "xgboost"

    def test_default_daily_model_uses_xgboost(self):
        """Default daily model should use XGBoost."""
        config = ImprovedModelConfig.daily_model()
        assert config.model_type == "xgboost"

    def test_hyperparams_work_with_xgboost(self):
        """Custom hyperparameters should work with XGBoost."""
        from xgboost import XGBClassifier

        custom_params = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
        }

        config = ImprovedModelConfig(
            name="test",
            base_timeframe="1H",
            model_type="xgboost",
            hyperparams=custom_params,
        )
        model_wrapper = ImprovedTimeframeModel(config)
        model = model_wrapper._create_model()

        assert isinstance(model, XGBClassifier)
        assert model.n_estimators == 100
        assert model.max_depth == 3


class TestMTFEnsembleModelType:
    """Tests for model_type propagation in MTFEnsembleConfig."""

    def test_mtf_ensemble_config_default_model_type(self):
        """MTFEnsembleConfig should default to xgboost."""
        from src.models.multi_timeframe.mtf_ensemble import MTFEnsembleConfig

        config = MTFEnsembleConfig()
        assert config.model_type == "xgboost"

    def test_mtf_ensemble_config_accepts_lightgbm(self):
        """MTFEnsembleConfig should accept lightgbm."""
        from src.models.multi_timeframe.mtf_ensemble import MTFEnsembleConfig

        config = MTFEnsembleConfig(model_type="lightgbm")
        assert config.model_type == "lightgbm"

    def test_mtf_ensemble_config_accepts_catboost(self):
        """MTFEnsembleConfig should accept catboost."""
        from src.models.multi_timeframe.mtf_ensemble import MTFEnsembleConfig

        config = MTFEnsembleConfig(model_type="catboost")
        assert config.model_type == "catboost"


class TestLightGBMParameterMapping:
    """Tests for LightGBM parameter translation."""

    @pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
    def test_lightgbm_common_params_translated(self):
        """Common parameters should be translated to LightGBM equivalents."""
        from lightgbm import LGBMClassifier

        config = ImprovedModelConfig(
            name="test",
            base_timeframe="1H",
            model_type="lightgbm",
            hyperparams={
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "gamma": 0.1,
            }
        )
        model_wrapper = ImprovedTimeframeModel(config)
        model = model_wrapper._create_model()

        assert isinstance(model, LGBMClassifier)
        assert model.n_estimators == 100
        assert model.max_depth == 5
        assert model.learning_rate == 0.1
        # Check translated params
        assert model.subsample == 0.8
        assert model.colsample_bytree == 0.8
        assert model.min_child_samples == 1  # min_child_weight -> min_child_samples
        assert model.min_split_gain == 0.1   # gamma -> min_split_gain

    @pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
    def test_lightgbm_default_params_set(self):
        """LightGBM should have sensible defaults."""
        from lightgbm import LGBMClassifier

        config = ImprovedModelConfig(
            name="test",
            base_timeframe="1H",
            model_type="lightgbm"
        )
        model_wrapper = ImprovedTimeframeModel(config)
        model = model_wrapper._create_model()

        assert isinstance(model, LGBMClassifier)
        assert model.verbose == -1  # Suppress output
        assert model.force_col_wise is True  # Better for many features


class TestCatBoostParameterMapping:
    """Tests for CatBoost parameter translation."""

    @pytest.mark.skipif(not HAS_CATBOOST, reason="CatBoost not installed")
    def test_catboost_common_params_translated(self):
        """Common parameters should be translated to CatBoost equivalents."""
        from catboost import CatBoostClassifier

        config = ImprovedModelConfig(
            name="test",
            base_timeframe="1H",
            model_type="catboost",
            hyperparams={
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
            }
        )
        model_wrapper = ImprovedTimeframeModel(config)
        model = model_wrapper._create_model()

        assert isinstance(model, CatBoostClassifier)
        assert model.get_params()["iterations"] == 100  # n_estimators -> iterations
        assert model.get_params()["depth"] == 5  # max_depth -> depth
        assert model.get_params()["learning_rate"] == 0.1
        # Check translated params
        assert model.get_params()["subsample"] == 0.8
        assert model.get_params()["rsm"] == 0.8  # colsample_bytree -> rsm
        assert model.get_params()["l2_leaf_reg"] == 1.0  # reg_lambda -> l2_leaf_reg

    @pytest.mark.skipif(not HAS_CATBOOST, reason="CatBoost not installed")
    def test_catboost_default_params_set(self):
        """CatBoost should have sensible defaults."""
        from catboost import CatBoostClassifier

        config = ImprovedModelConfig(
            name="test",
            base_timeframe="1H",
            model_type="catboost"
        )
        model_wrapper = ImprovedTimeframeModel(config)
        model = model_wrapper._create_model()

        assert isinstance(model, CatBoostClassifier)
        assert model.get_params()["verbose"] == 0  # Suppress output
        assert model.get_params()["allow_writing_files"] is False  # No file creation


class TestFrameworkTrainingCompatibility:
    """Tests to ensure all frameworks can train on the same data."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        n_samples = 100
        n_features = 10

        np.random.seed(42)
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        y = pd.Series(np.random.choice([0, 1, 2], size=n_samples))

        return X, y

    def test_xgboost_can_train(self, sample_data):
        """XGBoost should train successfully."""
        from xgboost import XGBClassifier

        X, y = sample_data
        config = ImprovedModelConfig(
            name="test",
            base_timeframe="1H",
            model_type="xgboost",
            hyperparams={"n_estimators": 10}  # Fast training
        )
        model_wrapper = ImprovedTimeframeModel(config)
        model = model_wrapper._create_model()

        # Should train without errors
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(y)
        assert isinstance(model, XGBClassifier)

    @pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not installed")
    def test_lightgbm_can_train(self, sample_data):
        """LightGBM should train successfully."""
        from lightgbm import LGBMClassifier

        X, y = sample_data
        config = ImprovedModelConfig(
            name="test",
            base_timeframe="1H",
            model_type="lightgbm",
            hyperparams={"n_estimators": 10}  # Fast training
        )
        model_wrapper = ImprovedTimeframeModel(config)
        model = model_wrapper._create_model()

        # Should train without errors
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(y)
        assert isinstance(model, LGBMClassifier)

    @pytest.mark.skipif(not HAS_CATBOOST, reason="CatBoost not installed")
    def test_catboost_can_train(self, sample_data):
        """CatBoost should train successfully."""
        from catboost import CatBoostClassifier

        X, y = sample_data
        config = ImprovedModelConfig(
            name="test",
            base_timeframe="1H",
            model_type="catboost",
            hyperparams={"n_estimators": 10}  # Fast training
        )
        model_wrapper = ImprovedTimeframeModel(config)
        model = model_wrapper._create_model()

        # Should train without errors
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(y)
        assert isinstance(model, CatBoostClassifier)


class TestFrameworkAvailabilityFlags:
    """Tests for framework availability detection."""

    def test_has_lightgbm_is_boolean(self):
        """HAS_LIGHTGBM should be a boolean."""
        assert isinstance(HAS_LIGHTGBM, bool)

    def test_has_catboost_is_boolean(self):
        """HAS_CATBOOST should be a boolean."""
        assert isinstance(HAS_CATBOOST, bool)

    def test_availability_flags_imported(self):
        """Should be able to import availability flags."""
        from src.models.multi_timeframe.improved_model import (
            HAS_LIGHTGBM,
            HAS_CATBOOST,
        )

        # Just check they exist
        assert HAS_LIGHTGBM in [True, False]
        assert HAS_CATBOOST in [True, False]
