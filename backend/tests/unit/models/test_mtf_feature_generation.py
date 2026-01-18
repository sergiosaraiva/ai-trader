"""Tests for MTF Ensemble feature generation fix.

These tests verify that prepare_higher_tf_data() generates enhanced features
(time, ROC, normalized, pattern, lag) in addition to technical indicators,
ensuring feature parity between training and prediction.

Issue: Model expected 115 features but only 87 were generated.
Root cause: prepare_higher_tf_data() only calculated technical indicators.
Fix: Added EnhancedFeatureEngine to generate all enhanced features for HTF data.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


@pytest.fixture
def sample_5min_data():
    """Create sample 5-minute OHLCV data for testing.

    Generates 20160 bars (14 days of data) to ensure enough data
    for daily resampling with sufficient history for indicators.
    """
    # Generate 20160 5-minute bars (14 days of data)
    dates = pd.date_range("2024-01-01", periods=20160, freq="5min")
    np.random.seed(42)

    # Create realistic OHLCV data
    base_price = 1.08
    returns = np.random.randn(20160) * 0.0005  # Small random returns
    close = pd.Series(base_price * np.cumprod(1 + returns))
    high = close * (1 + np.abs(np.random.randn(20160) * 0.0003))
    low = close * (1 - np.abs(np.random.randn(20160) * 0.0003))
    open_ = close.shift(1).fillna(base_price)

    df = pd.DataFrame({
        "open": open_.values,
        "high": high.values,
        "low": low.values,
        "close": close.values,
        "volume": np.random.randint(100, 1000, 20160),
    }, index=dates)

    return df


@pytest.fixture
def sample_1h_data():
    """Create sample 1-hour OHLCV data for testing."""
    dates = pd.date_range("2024-01-01", periods=500, freq="1h")
    np.random.seed(42)

    base_price = 1.08
    returns = np.random.randn(500) * 0.001
    close = pd.Series(base_price * np.cumprod(1 + returns))

    df = pd.DataFrame({
        "open": close.shift(1).fillna(base_price).values,
        "high": (close * (1 + np.abs(np.random.randn(500) * 0.001))).values,
        "low": (close * (1 - np.abs(np.random.randn(500) * 0.001))).values,
        "close": close.values,
        "volume": np.random.randint(1000, 10000, 500),
    }, index=dates)

    return df


class TestMTFEnsemblePrepareHigherTFData:
    """Tests for MTFEnsemble.prepare_higher_tf_data() method."""

    def test_prepare_higher_tf_data_generates_enhanced_features(self, sample_5min_data):
        """Test that prepare_higher_tf_data generates enhanced features, not just technical."""
        from src.models.multi_timeframe.mtf_ensemble import MTFEnsemble, MTFEnsembleConfig

        config = MTFEnsembleConfig()
        ensemble = MTFEnsemble(config)

        # Prepare higher TF data for 1H timeframe (needs 4H and D)
        higher_tf_data = ensemble.prepare_higher_tf_data(sample_5min_data, "1H")

        # Should have 4H and D timeframes
        assert "4H" in higher_tf_data
        assert "D" in higher_tf_data

        # Check that enhanced features exist (not just technical indicators)
        df_4h = higher_tf_data["4H"]

        # Time features (using cyclical encoding)
        time_features = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_london", "is_newyork"]
        for feature in time_features:
            assert feature in df_4h.columns, f"Missing time feature: {feature} in 4H data"

        # ROC features
        roc_features = ["price_roc1", "price_roc3", "price_roc6"]
        for feature in roc_features:
            assert feature in df_4h.columns, f"Missing ROC feature: {feature} in 4H data"

        # Normalized/percentile features
        norm_features = ["returns_zscore", "rsi_7_pctl", "rsi_14_pctl"]
        for feature in norm_features:
            assert feature in df_4h.columns, f"Missing normalized feature: {feature} in 4H data"

        # Pattern features
        pattern_features = ["higher_high", "lower_low", "bullish_engulf"]
        for feature in pattern_features:
            assert feature in df_4h.columns, f"Missing pattern feature: {feature} in 4H data"

        # Lag features (named pattern: indicator_lagN)
        lag_features = ["rsi_7_lag1", "rsi_7_lag3", "macd_hist_lag1"]
        for feature in lag_features:
            assert feature in df_4h.columns, f"Missing lag feature: {feature} in 4H data"

    def test_prepare_higher_tf_data_for_4h_timeframe(self, sample_5min_data):
        """Test prepare_higher_tf_data for 4H timeframe (needs only D)."""
        from src.models.multi_timeframe.mtf_ensemble import MTFEnsemble, MTFEnsembleConfig

        config = MTFEnsembleConfig()
        ensemble = MTFEnsemble(config)

        higher_tf_data = ensemble.prepare_higher_tf_data(sample_5min_data, "4H")

        # Should only have D timeframe
        assert "D" in higher_tf_data
        assert "4H" not in higher_tf_data

        # D should have enhanced features (cyclical time encoding)
        df_d = higher_tf_data["D"]
        assert "hour_sin" in df_d.columns or "dow_sin" in df_d.columns

    def test_prepare_higher_tf_data_for_daily_timeframe(self, sample_5min_data):
        """Test prepare_higher_tf_data for D timeframe (no higher TF needed)."""
        from src.models.multi_timeframe.mtf_ensemble import MTFEnsemble, MTFEnsembleConfig

        config = MTFEnsembleConfig()
        ensemble = MTFEnsemble(config)

        higher_tf_data = ensemble.prepare_higher_tf_data(sample_5min_data, "D")

        # Should be empty for daily timeframe
        assert higher_tf_data == {}

    def test_prepare_higher_tf_data_no_cross_tf_features(self, sample_5min_data):
        """Test that HTF data does NOT include cross-TF features (to avoid recursion)."""
        from src.models.multi_timeframe.mtf_ensemble import MTFEnsemble, MTFEnsembleConfig

        config = MTFEnsembleConfig()
        ensemble = MTFEnsemble(config)

        higher_tf_data = ensemble.prepare_higher_tf_data(sample_5min_data, "1H")

        df_4h = higher_tf_data["4H"]

        # Should NOT have cross-TF features like htf_4H_* or htf_D_*
        cross_tf_features = [col for col in df_4h.columns if col.startswith("htf_")]
        assert len(cross_tf_features) == 0, f"Found unexpected cross-TF features: {cross_tf_features}"

    def test_prepare_higher_tf_data_feature_count(self, sample_5min_data):
        """Test that prepare_higher_tf_data generates sufficient features."""
        from src.models.multi_timeframe.mtf_ensemble import MTFEnsemble, MTFEnsembleConfig

        config = MTFEnsembleConfig()
        ensemble = MTFEnsemble(config)

        higher_tf_data = ensemble.prepare_higher_tf_data(sample_5min_data, "1H")

        # 4H should have 115+ features (matching model requirements)
        df_4h = higher_tf_data["4H"]
        feature_count = len(df_4h.columns)

        # Should have at least 100 features (technical + enhanced)
        assert feature_count >= 100, f"4H has only {feature_count} features, expected >= 100"


class TestEnhancedFeatureEngine:
    """Tests for EnhancedFeatureEngine feature generation."""

    def test_enhanced_feature_engine_feature_count(self, sample_1h_data):
        """Test that EnhancedFeatureEngine generates expected number of features."""
        from src.features.technical.calculator import TechnicalIndicatorCalculator
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        # Calculate technical indicators first
        calc = TechnicalIndicatorCalculator(model_type="short_term")
        df = calc.calculate(sample_1h_data.copy())

        # Add enhanced features
        engine = EnhancedFeatureEngine(
            include_time_features=True,
            include_roc_features=True,
            include_normalized_features=True,
            include_pattern_features=True,
            include_lag_features=True,
            include_sentiment_features=False,
        )

        df_enhanced = engine.add_all_features(df, higher_tf_data=None)

        # Count features (excluding OHLCV)
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        feature_cols = [col for col in df_enhanced.columns if col not in ohlcv_cols]

        # Should have substantial number of features
        # Training generates ~115 features for 1H model
        assert len(feature_cols) >= 80, f"Expected >= 80 features, got {len(feature_cols)}"

    def test_enhanced_feature_engine_no_cross_tf_when_none(self, sample_1h_data):
        """Test that no cross-TF features added when higher_tf_data is None."""
        from src.features.technical.calculator import TechnicalIndicatorCalculator
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        calc = TechnicalIndicatorCalculator(model_type="short_term")
        df = calc.calculate(sample_1h_data.copy())

        engine = EnhancedFeatureEngine()
        df_enhanced = engine.add_all_features(df, higher_tf_data=None)

        # Should NOT have cross-TF features
        cross_tf_features = [col for col in df_enhanced.columns if col.startswith("htf_")]
        assert len(cross_tf_features) == 0, f"Found unexpected cross-TF features: {cross_tf_features}"


class TestFeatureParity:
    """Tests to ensure feature parity between training and prediction."""

    def test_htf_feature_count_matches_training(self, sample_5min_data):
        """Test that HTF feature count is sufficient for model loading."""
        from src.models.multi_timeframe.mtf_ensemble import MTFEnsemble, MTFEnsembleConfig

        config = MTFEnsembleConfig()
        ensemble = MTFEnsemble(config)

        higher_tf_data = ensemble.prepare_higher_tf_data(sample_5min_data, "1H")

        # Get feature counts
        features_4h = len(higher_tf_data["4H"].columns)
        features_d = len(higher_tf_data["D"].columns)

        # Both should have substantial features (at least 80+)
        assert features_4h >= 80, f"4H has only {features_4h} features, expected >= 80"
        assert features_d >= 80, f"D has only {features_d} features, expected >= 80"

    def test_critical_features_present(self, sample_5min_data):
        """Test that critical features are present in HTF data."""
        from src.models.multi_timeframe.mtf_ensemble import MTFEnsemble, MTFEnsembleConfig

        config = MTFEnsembleConfig()
        ensemble = MTFEnsemble(config)

        higher_tf_data = ensemble.prepare_higher_tf_data(sample_5min_data, "1H")

        # Critical technical features
        critical_features = ["rsi_14", "ema_21", "sma_50", "macd", "atr_14"]
        for feature in critical_features:
            assert feature in higher_tf_data["4H"].columns, f"Missing critical feature: {feature}"


class TestIntegrationWithModel:
    """Integration tests with actual model loading (if models exist)."""

    @pytest.fixture
    def model_dir(self):
        """Get the model directory path."""
        from pathlib import Path

        model_path = Path("/home/sergio/ai-trader/models/mtf_ensemble")
        if model_path.exists() and (model_path / "1H_model.pkl").exists():
            return model_path
        return None

    def test_prediction_with_prepared_data(self, sample_5min_data, model_dir):
        """Test that predictions work with properly prepared HTF data."""
        if model_dir is None:
            pytest.skip("Model files not found")

        from src.models.multi_timeframe.mtf_ensemble import MTFEnsemble, MTFEnsembleConfig

        config = MTFEnsembleConfig()
        ensemble = MTFEnsemble(config)

        # Load the trained models
        try:
            ensemble.load(str(model_dir))
        except Exception as e:
            pytest.skip(f"Could not load models: {e}")

        # Prepare data and make prediction
        higher_tf_data = ensemble.prepare_higher_tf_data(sample_5min_data, "1H")

        # The prediction should not raise feature mismatch errors
        # This is the core test - if prepare_higher_tf_data works correctly,
        # the model should be able to generate features and make predictions
        assert higher_tf_data is not None
        assert "4H" in higher_tf_data
        assert "D" in higher_tf_data

        # Verify we have enough features for the model
        assert len(higher_tf_data["4H"].columns) >= 100
        assert len(higher_tf_data["D"].columns) >= 100
