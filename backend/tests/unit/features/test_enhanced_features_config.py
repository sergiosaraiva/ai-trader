"""Test enhanced features with centralized configuration.

Tests that EnhancedFeatureEngine correctly loads parameters from TradingConfig
for lag features, ROC features, session features, and cyclical features.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine
from src.config import TradingConfig


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start="2023-01-01", periods=200, freq="5min")
    data = {
        "open": np.random.randn(200).cumsum() + 100,
        "high": np.random.randn(200).cumsum() + 101,
        "low": np.random.randn(200).cumsum() + 99,
        "close": np.random.randn(200).cumsum() + 100,
        "volume": np.random.randint(1000, 10000, 200),
    }
    df = pd.DataFrame(data, index=dates)

    # Add indicators for testing
    df["rsi_14"] = 50 + np.random.randn(200) * 10
    df["macd"] = np.random.randn(200)
    df["macd_hist"] = np.random.randn(200)
    df["adx_14"] = 20 + np.random.randn(200) * 5
    df["atr_14"] = 0.01 + np.random.rand(200) * 0.005
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()

    return df


@pytest.fixture
def trading_config():
    """Create a TradingConfig instance."""
    return TradingConfig()


def test_enhanced_feature_engine_with_config(sample_ohlcv_data, trading_config):
    """Test that EnhancedFeatureEngine loads from TradingConfig."""
    engine = EnhancedFeatureEngine(config=trading_config)

    # Verify config is loaded
    assert engine.config is not None
    assert engine.lag_periods == trading_config.feature_engineering.lags.standard_lags


def test_enhanced_feature_engine_without_config(sample_ohlcv_data):
    """Test that EnhancedFeatureEngine works without config (backward compatibility)."""
    engine = EnhancedFeatureEngine()

    # Verify defaults are used
    assert engine.config is None
    assert engine.lag_periods == [1, 2, 3, 6, 12]


def test_lag_features_with_config(sample_ohlcv_data, trading_config):
    """Test that lag features use config parameters."""
    # Modify config to test custom lag periods
    trading_config.feature_engineering.lags.standard_lags = [1, 5, 10]

    engine = EnhancedFeatureEngine(config=trading_config, include_lag_features=True)
    result = engine._add_lag_features(sample_ohlcv_data.copy())

    # Verify lag features are created with config periods
    assert "returns_lag1" in result.columns
    assert "returns_lag5" in result.columns
    assert "returns_lag10" in result.columns

    # Verify old default periods are NOT present (if they weren't in config)
    if 2 not in trading_config.feature_engineering.lags.standard_lags:
        assert "returns_lag2" not in result.columns


def test_roc_features_with_config(sample_ohlcv_data, trading_config):
    """Test that ROC features use config parameters."""
    # Modify config to test custom ROC periods
    trading_config.feature_engineering.lags.rsi_roc_periods = [2, 4]
    trading_config.feature_engineering.lags.macd_roc_periods = [5]
    trading_config.feature_engineering.lags.price_roc_periods = [1, 7, 14]

    engine = EnhancedFeatureEngine(config=trading_config, include_roc_features=True)
    result = engine._add_roc_features(sample_ohlcv_data.copy())

    # Verify RSI ROC features with config periods
    assert "rsi_14_roc2" in result.columns
    assert "rsi_14_roc4" in result.columns

    # Verify MACD ROC features with config periods
    assert "macd_roc5" in result.columns

    # Verify price ROC features with config periods
    assert "price_roc1" in result.columns
    assert "price_roc7" in result.columns
    assert "price_roc14" in result.columns

    # Verify old defaults are NOT present (if they weren't in config)
    if 3 not in trading_config.feature_engineering.lags.rsi_roc_periods:
        assert "rsi_14_roc3" not in result.columns


def test_session_features_with_config(sample_ohlcv_data, trading_config):
    """Test that session features use config parameters."""
    # Modify config to test custom session times
    trading_config.feature_engineering.sessions.asian_session = (0, 7)
    trading_config.feature_engineering.sessions.london_session = (7, 15)
    trading_config.feature_engineering.sessions.ny_session = (12, 20)

    engine = EnhancedFeatureEngine(config=trading_config, include_time_features=True)
    result = engine._add_time_features(sample_ohlcv_data.copy())

    # Verify session features are created
    assert "is_asian" in result.columns
    assert "is_london" in result.columns
    assert "is_newyork" in result.columns
    assert "is_overlap" in result.columns

    # Test session detection at boundaries
    # Create test data with specific hours
    test_dates = pd.date_range(start="2023-01-01 06:00", periods=3, freq="1H")
    test_df = sample_ohlcv_data.iloc[:3].copy()
    test_df.index = test_dates

    result = engine._add_time_features(test_df)

    # Hour 6 should be in Asian session (0-7)
    assert result.loc[test_dates[0], "is_asian"] == 1
    # Hour 7 should be in London session (7-15)
    assert result.loc[test_dates[1], "is_london"] == 1
    # Hour 8 should be in London session (7-15)
    assert result.loc[test_dates[2], "is_london"] == 1


def test_cyclical_features_with_config(sample_ohlcv_data, trading_config):
    """Test that cyclical features use config parameters."""
    # Modify config to test custom cycle parameters
    trading_config.feature_engineering.cyclical.hour_encoding_cycles = 24
    trading_config.feature_engineering.cyclical.day_of_week_cycles = 7
    trading_config.feature_engineering.cyclical.day_of_month_cycles = 31

    engine = EnhancedFeatureEngine(config=trading_config, include_time_features=True)
    result = engine._add_time_features(sample_ohlcv_data.copy())

    # Verify cyclical features are created
    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns
    assert "dow_sin" in result.columns
    assert "dow_cos" in result.columns
    assert "month_sin" in result.columns
    assert "month_cos" in result.columns

    # Verify cyclical encoding is correct (values between -1 and 1)
    assert result["hour_sin"].min() >= -1.0
    assert result["hour_sin"].max() <= 1.0
    assert result["hour_cos"].min() >= -1.0
    assert result["hour_cos"].max() <= 1.0
    assert result["dow_sin"].min() >= -1.0
    assert result["dow_sin"].max() <= 1.0


def test_all_features_with_config(sample_ohlcv_data, trading_config):
    """Test that all features are generated correctly with config."""
    engine = EnhancedFeatureEngine(
        config=trading_config,
        include_time_features=True,
        include_roc_features=True,
        include_lag_features=True,
        include_normalized_features=True,
        include_pattern_features=True,
    )

    result = engine.add_all_features(sample_ohlcv_data.copy())

    # Verify result has more columns than input
    assert len(result.columns) > len(sample_ohlcv_data.columns)

    # Verify no NaN in feature columns (except for lag features at the beginning)
    # Check last 50 rows (after lags stabilize)
    non_null_rows = result.iloc[-50:]
    null_counts = non_null_rows.isnull().sum()

    # Some features may have NaN, but not all
    assert null_counts.max() < len(non_null_rows)


def test_config_parameter_consistency(trading_config):
    """Test that config parameters are consistent with expected types."""
    # Test lag parameters
    assert isinstance(trading_config.feature_engineering.lags.standard_lags, list)
    assert all(isinstance(x, int) for x in trading_config.feature_engineering.lags.standard_lags)

    # Test session parameters
    assert isinstance(trading_config.feature_engineering.sessions.asian_session, tuple)
    assert len(trading_config.feature_engineering.sessions.asian_session) == 2

    # Test cyclical parameters
    assert isinstance(trading_config.feature_engineering.cyclical.hour_encoding_cycles, int)
    assert trading_config.feature_engineering.cyclical.hour_encoding_cycles > 0


def test_custom_lag_periods_override(sample_ohlcv_data, trading_config):
    """Test that custom lag_periods parameter overrides config."""
    custom_lags = [2, 4, 8]
    engine = EnhancedFeatureEngine(config=trading_config, lag_periods=custom_lags)

    # Verify custom lags override config
    assert engine.lag_periods == custom_lags


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
