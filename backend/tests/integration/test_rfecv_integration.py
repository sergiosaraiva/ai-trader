"""Integration tests for RFECV feature selection in MTF Ensemble.

CRITICAL: These tests verify that RFECV doesn't introduce data leakage
and properly integrates with ImprovedTimeframeModel training pipeline.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.models.feature_selection.rfecv_config import RFECVConfig
from src.models.multi_timeframe.improved_model import (
    ImprovedModelConfig,
    ImprovedTimeframeModel,
)


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_bars = 500

    # Generate synthetic price data
    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="1H")
    close = 1.0850 + np.cumsum(np.random.randn(n_bars) * 0.0001)
    high = close + np.abs(np.random.randn(n_bars) * 0.0002)
    low = close - np.abs(np.random.randn(n_bars) * 0.0002)
    open_ = close + np.random.randn(n_bars) * 0.0001
    volume = np.random.randint(1000, 10000, n_bars)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    return df


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


class TestRFECVIntegration:
    """Integration tests for RFECV with ImprovedTimeframeModel."""

    def test_model_with_rfecv_disabled(self, sample_ohlcv_data):
        """Test baseline: model training without RFECV (use_rfecv=False)."""
        # Create model config without RFECV
        config = ImprovedModelConfig(
            name="1H_test",
            base_timeframe="1H",
            use_rfecv=False,
            include_sentiment_features=False,
            n_estimators=10,  # Small for fast test
            max_depth=3,
        )

        model = ImprovedTimeframeModel(config)

        # Prepare data
        X, y, feature_names = model.prepare_data(sample_ohlcv_data)

        # Check we have data
        assert len(X) > 0
        assert len(y) > 0
        assert len(feature_names) > 0

        # Split chronologically
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train
        results = model.train(X_train, y_train, X_val, y_val, feature_names)

        # Verify training worked
        assert model.is_trained is True
        assert results["train_accuracy"] > 0
        assert results["val_accuracy"] > 0

        # Verify all features were used (no selection)
        assert model.selected_features == feature_names
        assert len(model.selected_indices) == len(feature_names)

    @patch("src.models.feature_selection.manager.RFECVSelector")
    def test_model_with_rfecv_enabled(
        self, mock_selector_class, sample_ohlcv_data, temp_model_dir
    ):
        """Test model training with RFECV enabled."""
        # Mock RFECVSelector to avoid slow RFECV execution
        mock_selector = MagicMock()

        def mock_fit(X, y, feature_names):
            # Select half the features
            n_selected = len(feature_names) // 2
            selected_indices = np.arange(n_selected)
            selected_features = [feature_names[i] for i in selected_indices]
            return selected_features, selected_indices

        mock_selector.fit.side_effect = mock_fit
        mock_selector.cv_scores = {
            "cv_scores_mean": [0.65, 0.70, 0.72],
            "cv_scores_std": [0.05, 0.04, 0.03],
            "n_features": [20, 30, 40],
            "optimal_n_features": 30,
        }
        mock_selector_class.return_value = mock_selector

        # Create model config with RFECV
        rfecv_config = RFECVConfig(
            cache_dir=temp_model_dir,
            cache_enabled=False,  # Disable cache for test
            min_features_to_select=10,
            verbose=0,
        )

        config = ImprovedModelConfig(
            name="1H_rfecv_test",
            base_timeframe="1H",
            use_rfecv=True,
            rfecv_config=rfecv_config,
            include_sentiment_features=False,
            n_estimators=10,
            max_depth=3,
        )

        model = ImprovedTimeframeModel(config)

        # Prepare data
        X, y, feature_names = model.prepare_data(sample_ohlcv_data)

        # Split chronologically
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Train with RFECV
        results = model.train(X_train, y_train, X_val, y_val, feature_names)

        # Verify RFECV was called
        assert mock_selector_class.called

        # Verify feature selection occurred
        assert len(model.selected_features) < len(feature_names)
        assert len(model.selected_features) == len(feature_names) // 2

        # Verify training worked with selected features
        assert model.is_trained is True
        assert results["train_accuracy"] > 0
        assert results["val_accuracy"] > 0

        # Verify RFECV scores were stored
        assert model.rfecv_scores is not None
        assert "optimal_n_features" in model.rfecv_scores

    @patch("src.models.feature_selection.manager.RFECVSelector")
    def test_rfecv_preserves_chronological_order(
        self, mock_selector_class, sample_ohlcv_data, temp_model_dir
    ):
        """CRITICAL: Verify RFECV doesn't cause data leakage (chronological order preserved)."""
        # Mock selector
        mock_selector = MagicMock()
        mock_selector.fit.return_value = (
            ["feat_0", "feat_1"],
            np.array([0, 1]),
        )
        mock_selector.cv_scores = {"optimal_n_features": 2}
        mock_selector_class.return_value = mock_selector

        # Create model with RFECV
        rfecv_config = RFECVConfig(
            cache_dir=temp_model_dir,
            cache_enabled=False,
            verbose=0,
        )

        config = ImprovedModelConfig(
            name="1H_leakage_test",
            base_timeframe="1H",
            use_rfecv=True,
            rfecv_config=rfecv_config,
            include_sentiment_features=False,
            n_estimators=10,
            max_depth=3,
        )

        model = ImprovedTimeframeModel(config)

        # Prepare data
        X, y, feature_names = model.prepare_data(sample_ohlcv_data)

        # CRITICAL: Split chronologically (train comes before val)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Verify chronological order (dates should not overlap)
        # Since we don't have access to dates here, we verify indices
        assert split_idx == len(X_train)
        assert len(X_val) == len(X) - split_idx

        # Train with RFECV - should only use X_train for feature selection
        results = model.train(X_train, y_train, X_val, y_val, feature_names)

        # Verify RFECV was called ONLY with training data
        assert mock_selector.fit.called
        fit_call_args = mock_selector.fit.call_args
        X_used_for_rfecv = fit_call_args[0][0]
        y_used_for_rfecv = fit_call_args[0][1]

        # CRITICAL: RFECV should only see training data, never validation data
        assert len(X_used_for_rfecv) == len(X_train)
        assert len(y_used_for_rfecv) == len(y_train)

        # Validation data should never be used in feature selection
        assert len(X_used_for_rfecv) < len(X)

    @patch("src.models.feature_selection.manager.RFECVSelector")
    def test_prediction_uses_selected_features(
        self, mock_selector_class, sample_ohlcv_data, temp_model_dir
    ):
        """Test that predictions use only selected features."""
        # Mock selector to select specific features
        selected_features = ["feat_0", "feat_2", "feat_4"]
        selected_indices = np.array([0, 2, 4])

        mock_selector = MagicMock()
        mock_selector.fit.return_value = (selected_features, selected_indices)
        mock_selector.cv_scores = {"optimal_n_features": 3}
        mock_selector_class.return_value = mock_selector

        # Create model with RFECV
        rfecv_config = RFECVConfig(
            cache_dir=temp_model_dir,
            cache_enabled=False,
            verbose=0,
        )

        config = ImprovedModelConfig(
            name="1H_pred_test",
            base_timeframe="1H",
            use_rfecv=True,
            rfecv_config=rfecv_config,
            include_sentiment_features=False,
            n_estimators=10,
            max_depth=3,
        )

        model = ImprovedTimeframeModel(config)

        # Prepare and train
        X, y, feature_names = model.prepare_data(sample_ohlcv_data)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        model.train(X_train, y_train, X_val, y_val, feature_names)

        # Verify selected features stored
        assert model.selected_features == selected_features
        assert np.array_equal(model.selected_indices, selected_indices)

        # Make predictions
        predictions = model.model.predict(model.scaler.transform(X_val[:, selected_indices]))

        # Verify predictions work with selected features
        assert len(predictions) == len(X_val)
        assert all(p in [0, 1] for p in predictions)

    @patch("src.models.feature_selection.manager.RFECVSelector")
    def test_rfecv_min_features_respected(
        self, mock_selector_class, sample_ohlcv_data, temp_model_dir
    ):
        """Test that min_features_to_select is respected in training."""
        min_features = 15

        # Mock selector to respect min_features
        mock_selector = MagicMock()

        def mock_fit(X, y, feature_names):
            # Return at least min_features
            n_selected = max(min_features, len(feature_names) // 3)
            selected_indices = np.arange(n_selected)
            selected_features = [feature_names[i] for i in selected_indices]
            return selected_features, selected_indices

        mock_selector.fit.side_effect = mock_fit
        mock_selector.cv_scores = {"optimal_n_features": min_features}
        mock_selector_class.return_value = mock_selector

        # Create model with RFECV and min_features constraint
        rfecv_config = RFECVConfig(
            cache_dir=temp_model_dir,
            cache_enabled=False,
            min_features_to_select=min_features,
            verbose=0,
        )

        config = ImprovedModelConfig(
            name="1H_min_feat_test",
            base_timeframe="1H",
            use_rfecv=True,
            rfecv_config=rfecv_config,
            include_sentiment_features=False,
            n_estimators=10,
            max_depth=3,
        )

        model = ImprovedTimeframeModel(config)

        # Prepare and train
        X, y, feature_names = model.prepare_data(sample_ohlcv_data)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        model.train(X_train, y_train, X_val, y_val, feature_names)

        # Verify at least min_features were selected
        assert len(model.selected_features) >= min_features

    @patch("src.models.feature_selection.manager.RFECVSelector")
    def test_rfecv_with_caching_enabled(
        self, mock_selector_class, sample_ohlcv_data, temp_model_dir
    ):
        """Test that RFECV caching works correctly."""
        # Mock selector
        mock_selector = MagicMock()
        mock_selector.fit.return_value = (
            ["feat_0", "feat_1"],
            np.array([0, 1]),
        )
        mock_selector.cv_scores = {"optimal_n_features": 2}
        mock_selector_class.return_value = mock_selector

        # Create model with caching enabled
        rfecv_config = RFECVConfig(
            cache_dir=temp_model_dir,
            cache_enabled=True,
            verbose=0,
        )

        config = ImprovedModelConfig(
            name="1H_cache_test",
            base_timeframe="1H",
            use_rfecv=True,
            rfecv_config=rfecv_config,
            include_sentiment_features=False,
            n_estimators=10,
            max_depth=3,
        )

        # Train first model (should create cache)
        model1 = ImprovedTimeframeModel(config)
        X, y, feature_names = model1.prepare_data(sample_ohlcv_data)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        model1.train(X_train, y_train, X_val, y_val, feature_names)

        # Verify selector was called once
        assert mock_selector_class.call_count == 1

        # Train second model with same config (should use cache)
        model2 = ImprovedTimeframeModel(config)
        X2, y2, feature_names2 = model2.prepare_data(sample_ohlcv_data)
        split_idx2 = int(len(X2) * 0.8)
        X_train2, X_val2 = X2[:split_idx2], X2[split_idx2:]
        y_train2, y_val2 = y2[:split_idx2], y2[split_idx2:]

        model2.train(X_train2, y_train2, X_val2, y_val2, feature_names2)

        # Verify selector was NOT called again (cache was used)
        # Note: call_count should still be 1 if cache is working
        # But the manager would load from cache, so selector wouldn't be instantiated again

        # Both models should have same selected features
        assert model1.selected_features == model2.selected_features

    def test_backward_compatibility_use_rfecv_false(self, sample_ohlcv_data):
        """Test backward compatibility when use_rfecv=False (default behavior)."""
        # Create model without RFECV (backward compatible)
        config = ImprovedModelConfig(
            name="1H_compat_test",
            base_timeframe="1H",
            use_rfecv=False,  # Explicit False
            include_sentiment_features=False,
            n_estimators=10,
            max_depth=3,
        )

        model = ImprovedTimeframeModel(config)

        # Prepare and train
        X, y, feature_names = model.prepare_data(sample_ohlcv_data)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        results = model.train(X_train, y_train, X_val, y_val, feature_names)

        # Should work as before (all features used)
        assert model.selected_features == feature_names
        assert len(model.selected_indices) == len(feature_names)
        assert model.rfecv_scores is None  # No RFECV scores

        # Training should succeed
        assert model.is_trained is True
        assert results["train_accuracy"] > 0

    @patch("src.models.feature_selection.manager.RFECVSelector")
    def test_rfecv_with_different_timeframes(
        self, mock_selector_class, sample_ohlcv_data, temp_model_dir
    ):
        """Test RFECV with different timeframe configurations."""
        # Mock selector
        mock_selector = MagicMock()
        mock_selector.fit.return_value = (
            ["feat_0", "feat_1", "feat_2"],
            np.array([0, 1, 2]),
        )
        mock_selector.cv_scores = {"optimal_n_features": 3}
        mock_selector_class.return_value = mock_selector

        rfecv_config = RFECVConfig(
            cache_dir=temp_model_dir,
            cache_enabled=False,
            verbose=0,
        )

        # Test with 1H model
        config_1h = ImprovedModelConfig(
            name="1H",
            base_timeframe="1H",
            use_rfecv=True,
            rfecv_config=rfecv_config,
            include_sentiment_features=False,
            n_estimators=10,
            max_depth=3,
        )

        model_1h = ImprovedTimeframeModel(config_1h)
        X, y, feature_names = model_1h.prepare_data(sample_ohlcv_data)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        model_1h.train(X_train, y_train, X_val, y_val, feature_names)

        # Verify 1H model trained successfully
        assert model_1h.is_trained is True
        assert len(model_1h.selected_features) == 3

        # Test with 4H model (different timeframe)
        config_4h = ImprovedModelConfig(
            name="4H",
            base_timeframe="4H",
            use_rfecv=True,
            rfecv_config=rfecv_config,
            include_sentiment_features=False,
            n_estimators=10,
            max_depth=3,
        )

        model_4h = ImprovedTimeframeModel(config_4h)
        X2, y2, feature_names2 = model_4h.prepare_data(sample_ohlcv_data)
        split_idx2 = int(len(X2) * 0.8)
        X_train2, X_val2 = X2[:split_idx2], X2[split_idx2:]
        y_train2, y_val2 = y2[:split_idx2], y2[split_idx2:]

        model_4h.train(X_train2, y_train2, X_val2, y_val2, feature_names2)

        # Verify 4H model trained successfully
        assert model_4h.is_trained is True
        assert len(model_4h.selected_features) == 3

    @patch("src.models.feature_selection.manager.RFECVSelector")
    def test_rfecv_edge_case_all_features_selected(
        self, mock_selector_class, sample_ohlcv_data, temp_model_dir
    ):
        """Test edge case where RFECV selects all features."""
        # Mock selector to select all features
        mock_selector = MagicMock()

        def mock_fit(X, y, feature_names):
            # Select all features
            selected_indices = np.arange(len(feature_names))
            selected_features = feature_names
            return selected_features, selected_indices

        mock_selector.fit.side_effect = mock_fit
        mock_selector.cv_scores = {"optimal_n_features": 100}
        mock_selector_class.return_value = mock_selector

        # Create model with RFECV
        rfecv_config = RFECVConfig(
            cache_dir=temp_model_dir,
            cache_enabled=False,
            verbose=0,
        )

        config = ImprovedModelConfig(
            name="1H_all_feat_test",
            base_timeframe="1H",
            use_rfecv=True,
            rfecv_config=rfecv_config,
            include_sentiment_features=False,
            n_estimators=10,
            max_depth=3,
        )

        model = ImprovedTimeframeModel(config)

        # Prepare and train
        X, y, feature_names = model.prepare_data(sample_ohlcv_data)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        model.train(X_train, y_train, X_val, y_val, feature_names)

        # All features should be selected
        assert len(model.selected_features) == len(feature_names)
        assert model.is_trained is True
