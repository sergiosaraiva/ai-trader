"""Tests for RFECVSelector.

CRITICAL: These tests verify that TimeSeriesSplit is used (no shuffling)
to prevent data leakage in time series feature selection.
"""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock, patch
from sklearn.model_selection import TimeSeriesSplit

from src.models.feature_selection.rfecv_config import RFECVConfig
from src.models.feature_selection.rfecv_selector import RFECVSelector


@pytest.fixture
def sample_data():
    """Generate sample synthetic data for testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 50

    # Generate synthetic feature matrix
    X = np.random.randn(n_samples, n_features)

    # Generate binary labels with slight class imbalance
    y = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55])

    # Generate feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]

    return X, y, feature_names


@pytest.fixture
def minimal_data():
    """Generate minimal data for fast tests."""
    np.random.seed(42)
    n_samples = 50
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], size=n_samples)
    feature_names = [f"feat_{i}" for i in range(n_features)]

    return X, y, feature_names


class TestRFECVSelector:
    """Tests for RFECV selector."""

    def test_initialization_default_config(self):
        """Test selector initialization with default config."""
        selector = RFECVSelector()

        assert selector.config is not None
        assert isinstance(selector.config, RFECVConfig)
        assert selector.selector is None
        assert selector.selected_features is None
        assert selector.selected_indices is None
        assert selector.cv_scores is None

    def test_initialization_custom_config(self):
        """Test selector initialization with custom config."""
        config = RFECVConfig(
            step=0.2,
            min_features_to_select=10,
            cv=3,
        )
        selector = RFECVSelector(config)

        assert selector.config.step == 0.2
        assert selector.config.min_features_to_select == 10
        assert selector.config.cv == 3

    @patch("src.models.feature_selection.rfecv_selector.RFECV")
    def test_fit_uses_timeseriessplit(self, mock_rfecv_class, minimal_data):
        """CRITICAL: Test that TimeSeriesSplit is used (prevents data leakage)."""
        X, y, feature_names = minimal_data

        # Create mock RFECV instance
        mock_rfecv_instance = MagicMock()
        mock_rfecv_instance.support_ = np.array([True, False, True, False, True,
                                                   False, True, False, True, False])
        mock_rfecv_instance.n_features_ = 5
        mock_rfecv_instance.cv_results_ = {
            "mean_test_score": np.array([0.6, 0.65, 0.7]),
            "std_test_score": np.array([0.1, 0.08, 0.06]),
        }
        mock_rfecv_instance.estimator_ = MagicMock()
        mock_rfecv_instance.estimator_.feature_importances_ = np.array([0.1, 0.2, 0.15, 0.25, 0.3])

        mock_rfecv_class.return_value = mock_rfecv_instance

        # Run fit
        config = RFECVConfig(verbose=0)
        selector = RFECVSelector(config)
        selector.fit(X, y, feature_names)

        # Verify RFECV was called with TimeSeriesSplit
        assert mock_rfecv_class.called
        call_kwargs = mock_rfecv_class.call_args[1]

        # Check that cv parameter is TimeSeriesSplit
        cv_param = call_kwargs["cv"]
        assert isinstance(cv_param, TimeSeriesSplit)
        assert cv_param.n_splits == config.cv

    @patch("src.models.feature_selection.rfecv_selector.RFECV")
    def test_fit_reduces_features(self, mock_rfecv_class, minimal_data):
        """Test that RFECV reduces the number of features."""
        X, y, feature_names = minimal_data

        # Mock RFECV to select 5 out of 10 features
        mock_rfecv_instance = MagicMock()
        mock_rfecv_instance.support_ = np.array([True, False, True, False, True,
                                                   False, True, False, True, False])
        mock_rfecv_instance.n_features_ = 5
        mock_rfecv_instance.cv_results_ = {
            "mean_test_score": np.array([0.6, 0.65, 0.7]),
            "std_test_score": np.array([0.1, 0.08, 0.06]),
        }
        mock_rfecv_instance.estimator_ = MagicMock()
        mock_rfecv_instance.estimator_.feature_importances_ = np.array([0.1, 0.2, 0.15, 0.25, 0.3])

        mock_rfecv_class.return_value = mock_rfecv_instance

        # Run fit
        config = RFECVConfig(verbose=0)
        selector = RFECVSelector(config)
        selected_features, selected_indices = selector.fit(X, y, feature_names)

        # Verify features were reduced
        assert len(selected_features) == 5
        assert len(selected_indices) == 5
        assert len(selected_features) < len(feature_names)

        # Verify selected features match expected indices
        expected_features = [feature_names[i] for i in [0, 2, 4, 6, 8]]
        assert selected_features == expected_features

    @patch("src.models.feature_selection.rfecv_selector.RFECV")
    def test_fit_respects_min_features(self, mock_rfecv_class, minimal_data):
        """Test that min_features_to_select is respected."""
        X, y, feature_names = minimal_data

        min_features = 5

        # Mock RFECV
        mock_rfecv_instance = MagicMock()
        mock_rfecv_instance.support_ = np.array([True] * 5 + [False] * 5)
        mock_rfecv_instance.n_features_ = min_features
        mock_rfecv_instance.cv_results_ = {
            "mean_test_score": np.array([0.6, 0.65]),
            "std_test_score": np.array([0.1, 0.08]),
        }
        mock_rfecv_instance.estimator_ = MagicMock()
        mock_rfecv_instance.estimator_.feature_importances_ = np.array([0.2] * 5)

        mock_rfecv_class.return_value = mock_rfecv_instance

        # Run fit with min_features constraint
        config = RFECVConfig(min_features_to_select=min_features, verbose=0)
        selector = RFECVSelector(config)
        selected_features, selected_indices = selector.fit(X, y, feature_names)

        # Verify min_features was passed to RFECV
        call_kwargs = mock_rfecv_class.call_args[1]
        assert call_kwargs["min_features_to_select"] == min_features

        # Verify at least min_features were selected
        assert len(selected_features) >= min_features

    @patch("src.models.feature_selection.rfecv_selector.RFECV")
    def test_fit_populates_cv_scores(self, mock_rfecv_class, minimal_data):
        """Test that cv_scores are populated after fit."""
        X, y, feature_names = minimal_data

        # Mock RFECV with CV results
        mock_rfecv_instance = MagicMock()
        mock_rfecv_instance.support_ = np.array([True] * 5 + [False] * 5)
        mock_rfecv_instance.n_features_ = 5
        mock_rfecv_instance.cv_results_ = {
            "mean_test_score": np.array([0.6, 0.65, 0.7]),
            "std_test_score": np.array([0.1, 0.08, 0.06]),
        }
        mock_rfecv_instance.estimator_ = MagicMock()
        mock_rfecv_instance.estimator_.feature_importances_ = np.array([0.2] * 5)

        mock_rfecv_class.return_value = mock_rfecv_instance

        # Run fit
        config = RFECVConfig(verbose=0, step=0.2, min_features_to_select=5)
        selector = RFECVSelector(config)
        selector.fit(X, y, feature_names)

        # Verify cv_scores are populated
        assert selector.cv_scores is not None
        assert "cv_scores_mean" in selector.cv_scores
        assert "cv_scores_std" in selector.cv_scores
        assert "n_features" in selector.cv_scores
        assert "optimal_n_features" in selector.cv_scores

        assert len(selector.cv_scores["cv_scores_mean"]) == 3
        assert len(selector.cv_scores["cv_scores_std"]) == 3
        assert selector.cv_scores["optimal_n_features"] == 5

    def test_fit_invalid_feature_names_length(self, minimal_data):
        """Test that fit raises error if feature_names length doesn't match X."""
        X, y, _ = minimal_data
        wrong_feature_names = [f"feat_{i}" for i in range(5)]  # Wrong length

        config = RFECVConfig(verbose=0)
        selector = RFECVSelector(config)

        with pytest.raises(ValueError, match="Feature names length.*must match"):
            selector.fit(X, y, wrong_feature_names)

    def test_fit_invalid_xy_length(self, minimal_data):
        """Test that fit raises error if X and y lengths don't match."""
        X, y, feature_names = minimal_data
        y_wrong = y[:-10]  # Wrong length

        config = RFECVConfig(verbose=0)
        selector = RFECVSelector(config)

        with pytest.raises(ValueError, match="X length.*must match y length"):
            selector.fit(X, y_wrong, feature_names)

    @patch("src.models.feature_selection.rfecv_selector.RFECV")
    def test_transform_after_fit(self, mock_rfecv_class, minimal_data):
        """Test transform after fit."""
        X, y, feature_names = minimal_data

        # Mock RFECV
        mock_rfecv_instance = MagicMock()
        mock_rfecv_instance.support_ = np.array([True] * 5 + [False] * 5)
        mock_rfecv_instance.n_features_ = 5
        mock_rfecv_instance.cv_results_ = {
            "mean_test_score": np.array([0.7]),
            "std_test_score": np.array([0.05]),
        }
        mock_rfecv_instance.estimator_ = MagicMock()
        mock_rfecv_instance.estimator_.feature_importances_ = np.array([0.2] * 5)

        # Mock transform to return subset of features
        X_transformed = X[:, :5]
        mock_rfecv_instance.transform.return_value = X_transformed

        mock_rfecv_class.return_value = mock_rfecv_instance

        # Fit and transform
        config = RFECVConfig(verbose=0)
        selector = RFECVSelector(config)
        selector.fit(X, y, feature_names)

        result = selector.transform(X)

        # Verify transform was called
        assert mock_rfecv_instance.transform.called
        assert result.shape == (X.shape[0], 5)

    def test_transform_before_fit_raises_error(self, minimal_data):
        """Test that transform raises error if called before fit."""
        X, _, _ = minimal_data

        config = RFECVConfig(verbose=0)
        selector = RFECVSelector(config)

        with pytest.raises(RuntimeError, match="Selector not fitted"):
            selector.transform(X)

    @patch("src.models.feature_selection.rfecv_selector.RFECV")
    def test_get_selected_features(self, mock_rfecv_class, minimal_data):
        """Test get_selected_features after fit."""
        X, y, feature_names = minimal_data

        # Mock RFECV
        mock_rfecv_instance = MagicMock()
        mock_rfecv_instance.support_ = np.array([True, False, True, False, True,
                                                   False, True, False, True, False])
        mock_rfecv_instance.n_features_ = 5
        mock_rfecv_instance.cv_results_ = {
            "mean_test_score": np.array([0.7]),
            "std_test_score": np.array([0.05]),
        }
        mock_rfecv_instance.estimator_ = MagicMock()
        mock_rfecv_instance.estimator_.feature_importances_ = np.array([0.2] * 5)

        mock_rfecv_class.return_value = mock_rfecv_instance

        # Fit
        config = RFECVConfig(verbose=0)
        selector = RFECVSelector(config)
        selector.fit(X, y, feature_names)

        # Get selected features
        selected = selector.get_selected_features()

        assert len(selected) == 5
        assert selected == [feature_names[i] for i in [0, 2, 4, 6, 8]]

    def test_get_selected_features_before_fit_raises_error(self):
        """Test that get_selected_features raises error if called before fit."""
        config = RFECVConfig(verbose=0)
        selector = RFECVSelector(config)

        with pytest.raises(RuntimeError, match="Selector not fitted"):
            selector.get_selected_features()

    @patch("src.models.feature_selection.rfecv_selector.RFECV")
    def test_get_feature_importances(self, mock_rfecv_class, minimal_data):
        """Test get_feature_importances after fit."""
        X, y, feature_names = minimal_data

        # Mock RFECV
        mock_rfecv_instance = MagicMock()
        mock_rfecv_instance.support_ = np.array([True] * 5 + [False] * 5)
        mock_rfecv_instance.n_features_ = 5
        mock_rfecv_instance.cv_results_ = {
            "mean_test_score": np.array([0.7]),
            "std_test_score": np.array([0.05]),
        }
        mock_rfecv_instance.estimator_ = MagicMock()
        mock_rfecv_instance.estimator_.feature_importances_ = np.array([0.1, 0.2, 0.15, 0.25, 0.3])

        mock_rfecv_class.return_value = mock_rfecv_instance

        # Fit
        config = RFECVConfig(verbose=0)
        selector = RFECVSelector(config)
        selector.fit(X, y, feature_names)

        # Get importances
        importances = selector.get_feature_importances()

        assert len(importances) == 5
        assert all(feat in importances for feat in [f"feat_{i}" for i in range(5)])
        assert all(isinstance(imp, float) for imp in importances.values())

    def test_get_feature_importances_before_fit_raises_error(self):
        """Test that get_feature_importances raises error if called before fit."""
        config = RFECVConfig(verbose=0)
        selector = RFECVSelector(config)

        with pytest.raises(RuntimeError, match="Selector not fitted"):
            selector.get_feature_importances()

    @patch("src.models.feature_selection.rfecv_selector.RFECV")
    def test_fit_with_all_features_important(self, mock_rfecv_class, minimal_data):
        """Test edge case where all features are selected as important."""
        X, y, feature_names = minimal_data

        # Mock RFECV to keep all features
        mock_rfecv_instance = MagicMock()
        mock_rfecv_instance.support_ = np.array([True] * len(feature_names))
        mock_rfecv_instance.n_features_ = len(feature_names)
        mock_rfecv_instance.cv_results_ = {
            "mean_test_score": np.array([0.7]),
            "std_test_score": np.array([0.05]),
        }
        mock_rfecv_instance.estimator_ = MagicMock()
        mock_rfecv_instance.estimator_.feature_importances_ = np.array([0.1] * len(feature_names))

        mock_rfecv_class.return_value = mock_rfecv_instance

        # Fit
        config = RFECVConfig(verbose=0)
        selector = RFECVSelector(config)
        selected_features, selected_indices = selector.fit(X, y, feature_names)

        # All features should be selected
        assert len(selected_features) == len(feature_names)
        assert len(selected_indices) == len(feature_names)

    @patch("src.models.feature_selection.rfecv_selector.RFECV")
    def test_fit_with_few_samples(self, mock_rfecv_class):
        """Test edge case with very few samples."""
        # Generate minimal data
        np.random.seed(42)
        X = np.random.randn(20, 10)  # Only 20 samples
        y = np.random.choice([0, 1], size=20)
        feature_names = [f"feat_{i}" for i in range(10)]

        # Mock RFECV
        mock_rfecv_instance = MagicMock()
        mock_rfecv_instance.support_ = np.array([True] * 5 + [False] * 5)
        mock_rfecv_instance.n_features_ = 5
        mock_rfecv_instance.cv_results_ = {
            "mean_test_score": np.array([0.6]),
            "std_test_score": np.array([0.1]),
        }
        mock_rfecv_instance.estimator_ = MagicMock()
        mock_rfecv_instance.estimator_.feature_importances_ = np.array([0.2] * 5)

        mock_rfecv_class.return_value = mock_rfecv_instance

        # Fit with small cv folds
        config = RFECVConfig(cv=2, verbose=0)  # Only 2 folds for small data
        selector = RFECVSelector(config)
        selected_features, selected_indices = selector.fit(X, y, feature_names)

        # Should still work
        assert len(selected_features) == 5
        assert len(selected_indices) == 5

    @patch("src.models.feature_selection.rfecv_selector.XGBClassifier")
    def test_fit_uses_xgboost_estimator(self, mock_xgb_class, minimal_data):
        """Test that XGBoost estimator is used with correct parameters."""
        X, y, feature_names = minimal_data

        # Mock XGBoost
        mock_xgb_instance = MagicMock()
        mock_xgb_class.return_value = mock_xgb_instance

        # We need to mock RFECV as well to avoid actual fitting
        with patch("src.models.feature_selection.rfecv_selector.RFECV") as mock_rfecv_class:
            mock_rfecv_instance = MagicMock()
            mock_rfecv_instance.support_ = np.array([True] * 5 + [False] * 5)
            mock_rfecv_instance.n_features_ = 5
            mock_rfecv_instance.cv_results_ = {
                "mean_test_score": np.array([0.7]),
                "std_test_score": np.array([0.05]),
            }
            mock_rfecv_instance.estimator_ = MagicMock()
            mock_rfecv_instance.estimator_.feature_importances_ = np.array([0.2] * 5)
            mock_rfecv_class.return_value = mock_rfecv_instance

            # Fit
            config = RFECVConfig(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=42,
                verbose=0,
            )
            selector = RFECVSelector(config)
            selector.fit(X, y, feature_names)

            # Verify XGBoost was instantiated with correct parameters
            assert mock_xgb_class.called
            call_kwargs = mock_xgb_class.call_args[1]
            assert call_kwargs["n_estimators"] == 100
            assert call_kwargs["max_depth"] == 4
            assert call_kwargs["learning_rate"] == 0.1
            assert call_kwargs["random_state"] == 42
            assert call_kwargs["eval_metric"] == "logloss"
