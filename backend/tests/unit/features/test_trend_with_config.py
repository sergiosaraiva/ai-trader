"""Unit tests for trend indicators with configuration support."""

import pytest
import pandas as pd
import numpy as np
from src.features.technical.trend import TrendIndicators
from src.config.trading_config import TradingConfig
from src.config.indicator_config import TrendIndicators as TrendConfig


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=300, freq="1h")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": 100 + np.cumsum(np.random.randn(300) * 0.1),
        "high": 100 + np.cumsum(np.random.randn(300) * 0.1) + 0.5,
        "low": 100 + np.cumsum(np.random.randn(300) * 0.1) - 0.5,
        "close": 100 + np.cumsum(np.random.randn(300) * 0.1),
        "volume": np.random.randint(1000, 10000, 300),
    })
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


class TestTrendIndicatorsWithConfig:
    """Test trend indicators accept and use config correctly."""

    def test_calculate_all_with_default_config(self, sample_data):
        """Test calculate_all works with default config."""
        indicators = TrendIndicators()
        result = indicators.calculate_all(sample_data)

        # Verify default SMA periods are created
        assert "sma_5" in result.columns
        assert "sma_10" in result.columns
        assert "sma_20" in result.columns
        assert "sma_50" in result.columns
        assert "sma_100" in result.columns
        assert "sma_200" in result.columns

        # Verify default EMA periods are created
        assert "ema_5" in result.columns
        assert "ema_20" in result.columns

        # Verify ADX with default period (14)
        assert "adx_14" in result.columns
        assert "plus_di_14" in result.columns
        assert "minus_di_14" in result.columns

        # Verify Aroon with default period (25)
        assert "aroon_up_25" in result.columns
        assert "aroon_down_25" in result.columns
        assert "aroon_osc_25" in result.columns

    def test_calculate_all_with_custom_config(self, sample_data):
        """Test calculate_all works with custom config."""
        config = TradingConfig()
        config.indicators.trend.sma_periods = [10, 20]
        config.indicators.trend.ema_periods = [12, 26]
        config.indicators.trend.adx_period = 20
        config.indicators.trend.aroon_period = 30

        indicators = TrendIndicators()
        result = indicators.calculate_all(sample_data, config=config)

        # Verify custom SMA periods
        assert "sma_10" in result.columns
        assert "sma_20" in result.columns
        assert "sma_5" not in result.columns  # Should not be created

        # Verify custom EMA periods
        assert "ema_12" in result.columns
        assert "ema_26" in result.columns

        # Verify custom ADX period
        assert "adx_20" in result.columns
        assert "plus_di_20" in result.columns
        assert "minus_di_20" in result.columns
        assert "adx_14" not in result.columns  # Should not be created

        # Verify custom Aroon period
        assert "aroon_up_30" in result.columns
        assert "aroon_down_30" in result.columns
        assert "aroon_osc_30" in result.columns
        assert "aroon_up_25" not in result.columns  # Should not be created

    def test_ma_crossovers_with_default_config(self, sample_data):
        """Test MA crossovers use default config pairs."""
        indicators = TrendIndicators()
        result = indicators.calculate_all(sample_data)

        # Verify default SMA crossover pairs: [(5, 20), (20, 50), (50, 200)]
        assert "sma_5_sma_20_cross" in result.columns
        assert "sma_20_sma_50_cross" in result.columns
        assert "sma_50_sma_200_cross" in result.columns

        # Verify default EMA crossover pairs: [(5, 20), (12, 26)]
        assert "ema_5_ema_20_cross" in result.columns
        # Note: ema_12 and ema_26 must both be created for the crossover to exist
        # Default ema_periods includes 5,10,20,50,100,200 but not 12 or 26
        # So ema_12_ema_26_cross won't be created unless we have those EMAs
        # This is expected behavior - crossover requires both MAs to exist

    def test_ma_crossovers_with_custom_config(self, sample_data):
        """Test MA crossovers use custom config pairs."""
        config = TradingConfig()
        config.indicators.trend.sma_periods = [10, 20, 50]
        config.indicators.trend.ema_periods = [9, 21]
        config.indicators.trend.sma_crossover_pairs = [(10, 50)]
        config.indicators.trend.ema_crossover_pairs = [(9, 21)]

        indicators = TrendIndicators()
        result = indicators.calculate_all(sample_data, config=config)

        # Verify custom SMA crossover pairs
        assert "sma_10_sma_50_cross" in result.columns
        assert "sma_5_sma_20_cross" not in result.columns  # Should not be created

        # Verify custom EMA crossover pairs
        assert "ema_9_ema_21_cross" in result.columns
        assert "ema_12_ema_26_cross" not in result.columns  # Should not be created

    def test_wma_periods_from_config(self, sample_data):
        """Test WMA uses periods from config."""
        config = TradingConfig()
        config.indicators.trend.wma_periods = [15, 30]

        indicators = TrendIndicators()
        result = indicators.calculate_all(sample_data, config=config)

        # Verify custom WMA periods
        assert "wma_15" in result.columns
        assert "wma_30" in result.columns
        assert "wma_10" not in result.columns  # Default should not be created

    def test_backward_compatibility_no_config(self, sample_data):
        """Test backward compatibility when no config is passed."""
        indicators = TrendIndicators()
        result = indicators.calculate_all(sample_data)

        # Should use defaults and work exactly as before
        assert "sma_5" in result.columns
        assert "sma_200" in result.columns
        assert "ema_5" in result.columns
        assert "adx_14" in result.columns
        assert "aroon_up_25" in result.columns

    def test_indicator_values_are_valid(self, sample_data):
        """Test that indicator values are calculated correctly."""
        indicators = TrendIndicators()
        result = indicators.calculate_all(sample_data)

        # SMA should be numeric and not all NaN
        assert result["sma_20"].notna().sum() > 0
        assert np.isfinite(result["sma_20"].dropna()).all()

        # ADX should be between 0 and 100
        adx_values = result["adx_14"].dropna()
        assert (adx_values >= 0).all()
        assert (adx_values <= 100).all()

        # Aroon should be between 0 and 100
        aroon_up_values = result["aroon_up_25"].dropna()
        assert (aroon_up_values >= 0).all()
        assert (aroon_up_values <= 100).all()

    def test_feature_names_tracking(self, sample_data):
        """Test that feature names are correctly tracked."""
        indicators = TrendIndicators()
        result = indicators.calculate_all(sample_data)
        feature_names = indicators.get_feature_names()

        # Verify feature names list is populated
        assert len(feature_names) > 0

        # Verify all tracked features exist in result
        for feature in feature_names:
            assert feature in result.columns

    def test_price_to_ma_calculations(self, sample_data):
        """Test price relative to MA calculations work with config."""
        config = TradingConfig()
        config.indicators.trend.sma_periods = [20, 50, 200]

        indicators = TrendIndicators()
        result = indicators.calculate_all(sample_data, config=config)

        # Verify price to MA features are created
        assert "price_to_sma_20" in result.columns
        assert "price_to_sma_50" in result.columns
        assert "price_to_sma_200" in result.columns

        # Verify values are percentage differences
        price_to_sma_20 = result["price_to_sma_20"].dropna()
        assert (price_to_sma_20 >= -1).all()  # Should be within reasonable range
        assert (price_to_sma_20 <= 1).all()

    def test_empty_crossover_pairs(self, sample_data):
        """Test handling of empty crossover pairs."""
        config = TradingConfig()
        config.indicators.trend.sma_crossover_pairs = []
        config.indicators.trend.ema_crossover_pairs = []

        indicators = TrendIndicators()
        result = indicators.calculate_all(sample_data, config=config)

        # Should not create any crossover features
        crossover_cols = [col for col in result.columns if "_cross" in col]
        assert len(crossover_cols) == 0
