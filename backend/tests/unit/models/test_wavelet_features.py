"""Comprehensive tests for wavelet decomposition features in EnhancedFeatureEngine.

Tests the _add_wavelet_features method which uses Discrete Wavelet Transform (DWT)
to decompose price data into multi-scale components, separating trend from noise.

Research shows 22% RMSE reduction when combined with ML models.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Try importing pywt to check availability
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


@pytest.fixture
def sample_ohlcv_small():
    """Create small OHLCV data (less than minimum for wavelet)."""
    dates = pd.date_range("2024-01-01", periods=20, freq="5min")
    np.random.seed(42)

    base_price = 1.08
    returns = np.random.randn(20) * 0.0005
    close = pd.Series(base_price * np.cumprod(1 + returns))

    df = pd.DataFrame({
        "open": close.shift(1).fillna(base_price).values,
        "high": (close * (1 + np.abs(np.random.randn(20) * 0.0003))).values,
        "low": (close * (1 - np.abs(np.random.randn(20) * 0.0003))).values,
        "close": close.values,
        "volume": np.random.randint(100, 1000, 20),
    }, index=dates)

    return df


@pytest.fixture
def sample_ohlcv_sufficient():
    """Create OHLCV data with sufficient rows for wavelet (>= 32)."""
    dates = pd.date_range("2024-01-01", periods=100, freq="5min")
    np.random.seed(42)

    base_price = 1.08
    returns = np.random.randn(100) * 0.0005
    close = pd.Series(base_price * np.cumprod(1 + returns))

    df = pd.DataFrame({
        "open": close.shift(1).fillna(base_price).values,
        "high": (close * (1 + np.abs(np.random.randn(100) * 0.0003))).values,
        "low": (close * (1 - np.abs(np.random.randn(100) * 0.0003))).values,
        "close": close.values,
        "volume": np.random.randint(100, 1000, 100),
    }, index=dates)

    return df


@pytest.fixture
def sample_ohlcv_with_nan():
    """Create OHLCV data with NaN values in close column."""
    dates = pd.date_range("2024-01-01", periods=100, freq="5min")
    np.random.seed(42)

    base_price = 1.08
    returns = np.random.randn(100) * 0.0005
    close = pd.Series(base_price * np.cumprod(1 + returns))

    # Introduce NaN values
    close.iloc[10:15] = np.nan
    close.iloc[50] = np.nan

    df = pd.DataFrame({
        "open": close.shift(1).fillna(base_price).values,
        "high": (close * (1 + np.abs(np.random.randn(100) * 0.0003))).values,
        "low": (close * (1 - np.abs(np.random.randn(100) * 0.0003))).values,
        "close": close.values,
        "volume": np.random.randint(100, 1000, 100),
    }, index=dates)

    return df


@pytest.fixture
def sample_ohlcv_no_close():
    """Create OHLCV data without close column."""
    dates = pd.date_range("2024-01-01", periods=100, freq="5min")
    np.random.seed(42)

    base_price = 1.08

    df = pd.DataFrame({
        "open": np.full(100, base_price),
        "high": np.full(100, base_price * 1.001),
        "low": np.full(100, base_price * 0.999),
        "volume": np.random.randint(100, 1000, 100),
    }, index=dates)

    return df


class TestWaveletFeaturesBasicFunctionality:
    """Tests for basic wavelet feature generation."""

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_wavelet_features_generated_correctly(self, sample_ohlcv_sufficient):
        """Test that wavelet features are generated with correct structure."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        engine = EnhancedFeatureEngine(
            include_time_features=False,
            include_roc_features=False,
            include_normalized_features=False,
            include_pattern_features=False,
            include_lag_features=False,
            include_sentiment_features=False,
            include_wavelet_features=True,
        )

        result = engine.add_all_features(sample_ohlcv_sufficient, higher_tf_data=None)

        # Check that result is a DataFrame
        assert isinstance(result, pd.DataFrame)

        # Check that original columns are preserved
        assert "close" in result.columns
        assert "open" in result.columns

        # Check that wavelet features exist
        wavelet_cols = [c for c in result.columns if c.startswith("wavelet_")]
        assert len(wavelet_cols) > 0, "No wavelet features were generated"

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_wavelet_features_have_correct_names(self, sample_ohlcv_sufficient):
        """Test that all expected wavelet feature columns are created."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        engine = EnhancedFeatureEngine(
            include_time_features=False,
            include_roc_features=False,
            include_normalized_features=False,
            include_pattern_features=False,
            include_lag_features=False,
            include_sentiment_features=False,
            include_wavelet_features=True,
        )

        result = engine.add_all_features(sample_ohlcv_sufficient, higher_tf_data=None)

        # Expected wavelet feature names (4 levels of decomposition)
        expected_features = [
            "wavelet_trend",                # Approximation (trend)
            "wavelet_trend_slope",          # Trend slope
            "wavelet_detail_1",             # Detail level 1 (highest freq noise)
            "wavelet_detail_2",             # Detail level 2
            "wavelet_detail_3",             # Detail level 3
            "wavelet_detail_4",             # Detail level 4 (lowest freq noise)
            "wavelet_approx_energy",        # Energy in approximation
            "wavelet_detail_1_energy",      # Energy in detail 1
            "wavelet_detail_2_energy",      # Energy in detail 2
            "wavelet_detail_3_energy",      # Energy in detail 3
            "wavelet_detail_4_energy",      # Energy in detail 4
            "wavelet_volatility",           # Sum of detail energies
            "wavelet_trend_strength",       # Ratio of approx to details
        ]

        for feature in expected_features:
            assert feature in result.columns, f"Missing wavelet feature: {feature}"

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_wavelet_features_no_nan_in_output(self, sample_ohlcv_sufficient):
        """Test that wavelet features don't introduce excessive NaN values."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        engine = EnhancedFeatureEngine(
            include_wavelet_features=True,
        )

        result = engine.add_all_features(sample_ohlcv_sufficient, higher_tf_data=None)

        wavelet_cols = [c for c in result.columns if c.startswith("wavelet_")]

        for col in wavelet_cols:
            nan_count = result[col].isna().sum()
            # Allow some NaN from diff operations but not excessive
            assert nan_count < len(result) * 0.1, f"{col} has too many NaN values: {nan_count}/{len(result)}"


class TestWaveletFeaturesEdgeCases:
    """Tests for edge cases in wavelet feature generation."""

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_small_dataframe_skips_wavelet(self, sample_ohlcv_small):
        """Test that wavelet features are skipped for DataFrames with < 32 rows."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        engine = EnhancedFeatureEngine(
            include_time_features=False,
            include_roc_features=False,
            include_normalized_features=False,
            include_pattern_features=False,
            include_lag_features=False,
            include_sentiment_features=False,
            include_wavelet_features=True,
        )

        result = engine.add_all_features(sample_ohlcv_small, higher_tf_data=None)

        # Wavelet features should NOT be added for small data
        wavelet_cols = [c for c in result.columns if c.startswith("wavelet_")]
        assert len(wavelet_cols) == 0, f"Wavelet features should not be added for < 32 rows, but found: {wavelet_cols}"

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_dataframe_with_nan_values_handles_gracefully(self, sample_ohlcv_with_nan):
        """Test that wavelet features handle NaN values in close column."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        engine = EnhancedFeatureEngine(
            include_time_features=False,
            include_roc_features=False,
            include_normalized_features=False,
            include_pattern_features=False,
            include_lag_features=False,
            include_sentiment_features=False,
            include_wavelet_features=True,
        )

        # Should not raise an error
        result = engine.add_all_features(sample_ohlcv_with_nan, higher_tf_data=None)

        # Wavelet features should be generated (NaN are handled by ffill/bfill)
        wavelet_cols = [c for c in result.columns if c.startswith("wavelet_")]
        assert len(wavelet_cols) > 0, "Wavelet features should be generated despite NaN values"

        # Check that wavelet features are not all NaN
        for col in wavelet_cols:
            assert not result[col].isna().all(), f"{col} is all NaN"

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_dataframe_without_close_column_skips_wavelet(self, sample_ohlcv_no_close):
        """Test that wavelet features are skipped when 'close' column is missing."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        engine = EnhancedFeatureEngine(
            include_time_features=False,
            include_roc_features=False,
            include_normalized_features=False,
            include_pattern_features=False,
            include_lag_features=False,
            include_sentiment_features=False,
            include_wavelet_features=True,
        )

        result = engine.add_all_features(sample_ohlcv_no_close, higher_tf_data=None)

        # Wavelet features should NOT be added without close column
        wavelet_cols = [c for c in result.columns if c.startswith("wavelet_")]
        assert len(wavelet_cols) == 0, f"Wavelet features should not be added without 'close' column, but found: {wavelet_cols}"

    def test_pywt_not_available_skips_wavelet(self, sample_ohlcv_sufficient):
        """Test that wavelet features are skipped gracefully when pywt is not available."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        with patch('src.models.multi_timeframe.enhanced_features.PYWT_AVAILABLE', False):
            engine = EnhancedFeatureEngine(
                include_time_features=False,
                include_roc_features=False,
                include_normalized_features=False,
                include_pattern_features=False,
                include_lag_features=False,
                include_sentiment_features=False,
                include_wavelet_features=True,
            )

            result = engine.add_all_features(sample_ohlcv_sufficient, higher_tf_data=None)

            # Wavelet features should NOT be added when pywt unavailable
            wavelet_cols = [c for c in result.columns if c.startswith("wavelet_")]
            assert len(wavelet_cols) == 0, "Wavelet features should not be added when pywt unavailable"


class TestWaveletFeaturesToggling:
    """Tests for enabling/disabling wavelet features."""

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_wavelet_features_disabled_by_flag(self, sample_ohlcv_sufficient):
        """Test that include_wavelet_features=False disables wavelet features."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        engine = EnhancedFeatureEngine(
            include_time_features=False,
            include_roc_features=False,
            include_normalized_features=False,
            include_pattern_features=False,
            include_lag_features=False,
            include_sentiment_features=False,
            include_wavelet_features=False,  # Disabled
        )

        result = engine.add_all_features(sample_ohlcv_sufficient, higher_tf_data=None)

        # No wavelet features should be present
        wavelet_cols = [c for c in result.columns if c.startswith("wavelet_")]
        assert len(wavelet_cols) == 0, f"Wavelet features should be disabled, but found: {wavelet_cols}"

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_wavelet_features_enabled_by_default(self, sample_ohlcv_sufficient):
        """Test that wavelet features are enabled by default."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        # Create engine with default settings (include_wavelet_features should be True)
        engine = EnhancedFeatureEngine()

        result = engine.add_all_features(sample_ohlcv_sufficient, higher_tf_data=None)

        # Wavelet features should be present
        wavelet_cols = [c for c in result.columns if c.startswith("wavelet_")]
        assert len(wavelet_cols) > 0, "Wavelet features should be enabled by default"


class TestWaveletFeaturesValueRanges:
    """Tests for wavelet feature value ranges and normalization."""

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_normalized_features_within_reasonable_bounds(self, sample_ohlcv_sufficient):
        """Test that normalized wavelet features are within reasonable bounds."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        engine = EnhancedFeatureEngine(
            include_wavelet_features=True,
        )

        result = engine.add_all_features(sample_ohlcv_sufficient, higher_tf_data=None)

        # Normalized features (divided by close price) should be in reasonable range
        normalized_features = [
            "wavelet_trend",
            "wavelet_trend_slope",
            "wavelet_detail_1",
            "wavelet_detail_2",
            "wavelet_detail_3",
            "wavelet_detail_4",
        ]

        for feature in normalized_features:
            if feature in result.columns:
                values = result[feature].dropna()
                if len(values) > 0:
                    # Normalized by close, so should be around 0-2 range (allowing some variance)
                    assert values.abs().max() < 10, f"{feature} has unreasonable values: max={values.abs().max()}"

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_energy_features_are_positive(self, sample_ohlcv_sufficient):
        """Test that energy features are non-negative."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        engine = EnhancedFeatureEngine(
            include_wavelet_features=True,
        )

        result = engine.add_all_features(sample_ohlcv_sufficient, higher_tf_data=None)

        # Energy features should be non-negative
        energy_features = [
            "wavelet_approx_energy",
            "wavelet_detail_1_energy",
            "wavelet_detail_2_energy",
            "wavelet_detail_3_energy",
            "wavelet_detail_4_energy",
            "wavelet_volatility",
        ]

        for feature in energy_features:
            if feature in result.columns:
                values = result[feature].dropna()
                if len(values) > 0:
                    assert (values >= 0).all(), f"{feature} has negative values: min={values.min()}"

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_trend_strength_ratio_is_valid(self, sample_ohlcv_sufficient):
        """Test that trend strength (ratio) is a valid positive number."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        engine = EnhancedFeatureEngine(
            include_wavelet_features=True,
        )

        result = engine.add_all_features(sample_ohlcv_sufficient, higher_tf_data=None)

        if "wavelet_trend_strength" in result.columns:
            values = result["wavelet_trend_strength"].dropna()
            if len(values) > 0:
                # Trend strength is a ratio, should be positive
                assert (values >= 0).all(), "wavelet_trend_strength should be non-negative"
                # Should not have extreme outliers
                assert values.max() < 1e6, f"wavelet_trend_strength has extreme values: max={values.max()}"


class TestWaveletFeaturesIntegration:
    """Integration tests for wavelet features with other feature types."""

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_wavelet_works_alongside_other_features(self, sample_ohlcv_sufficient):
        """Test that wavelet features work alongside time, ROC, normalized, pattern, and lag features."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        engine = EnhancedFeatureEngine(
            include_time_features=True,
            include_roc_features=True,
            include_normalized_features=True,
            include_pattern_features=True,
            include_lag_features=True,
            include_sentiment_features=False,
            include_wavelet_features=True,
        )

        result = engine.add_all_features(sample_ohlcv_sufficient, higher_tf_data=None)

        # Check that all feature types are present
        time_features = [c for c in result.columns if c in ["hour_sin", "hour_cos", "is_newyork"]]
        roc_features = [c for c in result.columns if "_roc" in c]
        pattern_features = [c for c in result.columns if c in ["higher_high", "lower_low", "bullish_engulf"]]
        wavelet_features = [c for c in result.columns if c.startswith("wavelet_")]

        assert len(time_features) > 0, "Time features missing"
        assert len(roc_features) > 0, "ROC features missing"
        assert len(pattern_features) > 0, "Pattern features missing"
        assert len(wavelet_features) > 0, "Wavelet features missing"

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_wavelet_preserves_dataframe_index(self, sample_ohlcv_sufficient):
        """Test that wavelet features preserve the DataFrame index."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        engine = EnhancedFeatureEngine(
            include_wavelet_features=True,
        )

        original_index = sample_ohlcv_sufficient.index.copy()
        result = engine.add_all_features(sample_ohlcv_sufficient, higher_tf_data=None)

        # Index should be preserved
        assert result.index.equals(original_index), "DataFrame index was modified"

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_wavelet_features_count(self, sample_ohlcv_sufficient):
        """Test that expected number of wavelet features are generated."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        engine = EnhancedFeatureEngine(
            include_time_features=False,
            include_roc_features=False,
            include_normalized_features=False,
            include_pattern_features=False,
            include_lag_features=False,
            include_sentiment_features=False,
            include_wavelet_features=True,
        )

        result = engine.add_all_features(sample_ohlcv_sufficient, higher_tf_data=None)

        wavelet_cols = [c for c in result.columns if c.startswith("wavelet_")]

        # Should have 13 wavelet features (based on implementation):
        # 1. wavelet_trend
        # 2. wavelet_trend_slope
        # 3-6. wavelet_detail_1 to wavelet_detail_4
        # 7. wavelet_approx_energy
        # 8-11. wavelet_detail_1_energy to wavelet_detail_4_energy
        # 12. wavelet_volatility
        # 13. wavelet_trend_strength
        assert len(wavelet_cols) == 13, f"Expected 13 wavelet features, got {len(wavelet_cols)}: {wavelet_cols}"


class TestWaveletFeaturesWithTechnicalIndicators:
    """Tests for wavelet features integration with technical indicators."""

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_wavelet_with_technical_indicators(self, sample_ohlcv_sufficient):
        """Test wavelet features work with pre-calculated technical indicators."""
        from src.features.technical.calculator import TechnicalIndicatorCalculator
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        # Calculate technical indicators first
        calc = TechnicalIndicatorCalculator(model_type="short_term")
        df_with_tech = calc.calculate(sample_ohlcv_sufficient.copy())

        # Add enhanced features including wavelet
        engine = EnhancedFeatureEngine(
            include_wavelet_features=True,
        )
        result = engine.add_all_features(df_with_tech, higher_tf_data=None)

        # Should have both technical indicators and wavelet features
        tech_indicators = [c for c in result.columns if any(ind in c for ind in ["rsi", "ema", "macd", "atr"])]
        wavelet_features = [c for c in result.columns if c.startswith("wavelet_")]

        assert len(tech_indicators) > 0, "Technical indicators missing"
        assert len(wavelet_features) > 0, "Wavelet features missing"

        # Total feature count should be substantial
        feature_cols = [c for c in result.columns if c not in ["open", "high", "low", "close", "volume"]]
        assert len(feature_cols) >= 50, f"Expected >= 50 features, got {len(feature_cols)}"

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_wavelet_features_dont_interfere_with_other_features(self, sample_ohlcv_sufficient):
        """Test that wavelet features don't modify or interfere with existing features."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        # Generate features without wavelet
        engine_no_wavelet = EnhancedFeatureEngine(
            include_time_features=True,
            include_roc_features=True,
            include_normalized_features=True,
            include_pattern_features=True,
            include_lag_features=False,  # Skip lag to reduce comparison complexity
            include_sentiment_features=False,
            include_wavelet_features=False,
        )
        result_no_wavelet = engine_no_wavelet.add_all_features(sample_ohlcv_sufficient.copy(), higher_tf_data=None)

        # Generate features with wavelet
        engine_with_wavelet = EnhancedFeatureEngine(
            include_time_features=True,
            include_roc_features=True,
            include_normalized_features=True,
            include_pattern_features=True,
            include_lag_features=False,  # Skip lag to reduce comparison complexity
            include_sentiment_features=False,
            include_wavelet_features=True,
        )
        result_with_wavelet = engine_with_wavelet.add_all_features(sample_ohlcv_sufficient.copy(), higher_tf_data=None)

        # All non-wavelet columns should be identical
        non_wavelet_cols = [c for c in result_no_wavelet.columns if not c.startswith("wavelet_")]

        for col in non_wavelet_cols:
            if col in result_with_wavelet.columns:
                # Compare values (allowing for floating point precision)
                pd.testing.assert_series_equal(
                    result_no_wavelet[col],
                    result_with_wavelet[col],
                    check_names=False,
                    obj=f"Column {col} was modified by wavelet feature generation"
                )


class TestWaveletFeaturesRobustness:
    """Robustness tests for wavelet features."""

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_wavelet_with_all_nan_close(self, sample_ohlcv_sufficient):
        """Test that wavelet handles all-NaN close gracefully."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        df = sample_ohlcv_sufficient.copy()
        df["close"] = np.nan

        engine = EnhancedFeatureEngine(
            include_wavelet_features=True,
        )

        # Should not crash
        result = engine.add_all_features(df, higher_tf_data=None)

        # Wavelet features should not be added (all NaN close)
        wavelet_cols = [c for c in result.columns if c.startswith("wavelet_")]
        # Either no features or all NaN features
        if len(wavelet_cols) > 0:
            for col in wavelet_cols:
                assert result[col].isna().all(), f"{col} should be all NaN when close is all NaN"

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_wavelet_with_constant_price(self, sample_ohlcv_sufficient):
        """Test wavelet features with constant price (no volatility)."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        df = sample_ohlcv_sufficient.copy()
        df["close"] = 1.08  # Constant price

        engine = EnhancedFeatureEngine(
            include_wavelet_features=True,
        )

        result = engine.add_all_features(df, higher_tf_data=None)

        # Wavelet features should be generated
        wavelet_cols = [c for c in result.columns if c.startswith("wavelet_")]
        assert len(wavelet_cols) > 0, "Wavelet features should be generated for constant price"

        # Energy features should be near zero for constant price
        if "wavelet_volatility" in result.columns:
            volatility = result["wavelet_volatility"].dropna()
            if len(volatility) > 0:
                # Volatility should be very low for constant price
                assert volatility.mean() < 0.1, f"Volatility should be low for constant price, got {volatility.mean()}"

    @pytest.mark.skipif(not PYWT_AVAILABLE, reason="pywt not available")
    def test_wavelet_with_exact_32_rows(self):
        """Test wavelet with exactly 32 rows (minimum for 4-level decomposition)."""
        from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine

        dates = pd.date_range("2024-01-01", periods=32, freq="5min")
        np.random.seed(42)

        base_price = 1.08
        returns = np.random.randn(32) * 0.0005
        close = pd.Series(base_price * np.cumprod(1 + returns))

        df = pd.DataFrame({
            "open": close.shift(1).fillna(base_price).values,
            "high": (close * (1 + np.abs(np.random.randn(32) * 0.0003))).values,
            "low": (close * (1 - np.abs(np.random.randn(32) * 0.0003))).values,
            "close": close.values,
            "volume": np.random.randint(100, 1000, 32),
        }, index=dates)

        engine = EnhancedFeatureEngine(
            include_wavelet_features=True,
        )

        result = engine.add_all_features(df, higher_tf_data=None)

        # Wavelet features should be generated with exactly 32 rows
        wavelet_cols = [c for c in result.columns if c.startswith("wavelet_")]
        assert len(wavelet_cols) > 0, "Wavelet features should be generated with 32 rows"
