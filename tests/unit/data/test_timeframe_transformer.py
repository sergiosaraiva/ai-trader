"""Unit tests for TimeframeTransformer."""

from datetime import datetime, timedelta

import pandas as pd
import pytest
import numpy as np

from src.data.processors.timeframe_transformer import (
    TimeframeTransformer,
    TimeframeConfig,
    TimeframeTransformError,
    STANDARD_TIMEFRAMES,
    resample_ohlcv,
)


@pytest.fixture
def sample_5m_data():
    """Create sample 5-minute OHLCV data with valid OHLCV relationships."""
    dates = pd.date_range("2024-01-01", periods=1000, freq="5min")
    np.random.seed(42)
    base_price = 1.1
    prices = base_price + np.cumsum(np.random.randn(1000) * 0.0001)

    # Generate open and close
    opens = prices.copy()
    closes = prices + np.random.randn(1000) * 0.0005

    # Generate high and low that respect OHLCV constraints
    highs = np.maximum(opens, closes) + np.random.rand(1000) * 0.001
    lows = np.minimum(opens, closes) - np.random.rand(1000) * 0.001

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.random.randint(100, 1000, 1000).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def transformer():
    """Create TimeframeTransformer instance."""
    return TimeframeTransformer()


class TestTimeframeConfig:
    """Tests for TimeframeConfig."""

    def test_standard_timeframes_exist(self):
        """Test standard timeframes are defined."""
        expected = ["1M", "5M", "15M", "30M", "1H", "4H", "1D", "1W", "1MO"]
        for tf in expected:
            assert tf in STANDARD_TIMEFRAMES

    def test_timeframe_config_attributes(self):
        """Test TimeframeConfig attributes."""
        config = STANDARD_TIMEFRAMES["1H"]
        assert config.name == "1H"
        assert config.minutes == 60
        assert config.pandas_freq == "1h"
        assert config.input_window > 0


class TestTimeframeTransformer:
    """Tests for TimeframeTransformer class."""

    def test_init_default(self, transformer):
        """Test default initialization."""
        assert transformer.timeframes == STANDARD_TIMEFRAMES

    def test_init_custom_timeframes(self):
        """Test custom timeframe configuration."""
        custom = {
            "CUSTOM": TimeframeConfig("CUSTOM", 120, "2h", 100, [1, 2])
        }
        transformer = TimeframeTransformer(timeframes=custom)
        assert "CUSTOM" in transformer.timeframes

    def test_get_timeframe_config(self, transformer):
        """Test getting timeframe config."""
        config = transformer.get_timeframe_config("1H")
        assert config.minutes == 60

    def test_get_timeframe_config_case_insensitive(self, transformer):
        """Test case insensitivity."""
        config = transformer.get_timeframe_config("1h")
        assert config.minutes == 60

    def test_get_timeframe_config_unknown(self, transformer):
        """Test error for unknown timeframe."""
        with pytest.raises(TimeframeTransformError, match="Unknown timeframe"):
            transformer.get_timeframe_config("UNKNOWN")

    def test_resample_5m_to_1h(self, transformer, sample_5m_data):
        """Test resampling from 5M to 1H."""
        result = transformer.resample(sample_5m_data, "5M", "1H")

        # Should have fewer rows
        assert len(result) < len(sample_5m_data)

        # Verify OHLCV columns present
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns

        # Verify index is datetime
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_resample_ohlcv_aggregation(self, transformer, sample_5m_data):
        """Test OHLCV aggregation rules."""
        result = transformer.resample(sample_5m_data, "5M", "15M")

        # Verify aggregation produces valid OHLCV structure
        # Each resampled candle should have: open <= max(open, close), low <= open, close, high
        for idx in range(min(5, len(result))):
            row = result.iloc[idx]
            assert row["low"] <= row["high"], "Low should be <= High"
            assert row["low"] <= row["open"], "Low should be <= Open"
            assert row["low"] <= row["close"], "Low should be <= Close"
            assert row["high"] >= row["open"], "High should be >= Open"
            assert row["high"] >= row["close"], "High should be >= Close"
            assert row["volume"] > 0, "Volume should be positive"

    def test_resample_upsample_error(self, transformer, sample_5m_data):
        """Test error when trying to upsample."""
        with pytest.raises(TimeframeTransformError, match="Cannot upsample"):
            transformer.resample(sample_5m_data, "5M", "1M")

    def test_resample_requires_datetime_index(self, transformer):
        """Test error without DatetimeIndex."""
        df = pd.DataFrame(
            {"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.05], "volume": [100]}
        )
        with pytest.raises(TimeframeTransformError, match="DatetimeIndex"):
            transformer.resample(df, "5M", "1H")

    def test_create_multi_timeframe_features(self, transformer, sample_5m_data):
        """Test multi-timeframe feature creation."""
        result = transformer.create_multi_timeframe_features(
            sample_5m_data,
            source_timeframe="5M",
            target_timeframes=["15M", "1H"],
            include_original=True,
            prefix_columns=True,
        )

        # Should have columns for each timeframe
        assert any("5M_" in col for col in result.columns)
        assert any("15M_" in col for col in result.columns)
        assert any("1H_" in col for col in result.columns)

        # Should have same or fewer rows after forward-fill alignment
        assert len(result) <= len(sample_5m_data)
        # Should have more columns than original
        assert len(result.columns) > len(sample_5m_data.columns)

    def test_create_cross_timeframe_features(self, transformer, sample_5m_data):
        """Test cross-timeframe feature creation."""
        result = transformer.create_cross_timeframe_features(
            sample_5m_data,
            source_timeframe="5M",
            higher_timeframes=["1H"],
        )

        # Should have cross-timeframe features
        assert "rel_close_1H" in result.columns
        assert "htf_direction_1H" in result.columns
        assert "htf_range_1H" in result.columns

    def test_sliding_window_transform(self, transformer, sample_5m_data):
        """Test sliding window transformation."""
        result = transformer.sliding_window_transform(
            sample_5m_data,
            source_timeframe="5M",
            target_timeframe="15M",
        )

        assert "sw_open" in result.columns
        assert "sw_high" in result.columns
        assert "sw_close" in result.columns

    def test_get_required_history(self, transformer):
        """Test required history calculation."""
        history = transformer.get_required_history(
            target_timeframe="1H",
            num_candles=100,
            source_timeframe="5M",
        )

        # 1H = 12 * 5M, so 100 candles needs ~1212 source candles
        assert history > 100 * 12

    def test_validate_timeframe_data(self, transformer, sample_5m_data):
        """Test timeframe validation."""
        is_valid, issues = transformer.validate_timeframe_data(
            sample_5m_data, "5M"
        )

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_timeframe_data_wrong_frequency(self, transformer, sample_5m_data):
        """Test validation detects wrong frequency."""
        # Data is 5M but we claim it's 1H
        is_valid, issues = transformer.validate_timeframe_data(
            sample_5m_data, "1H"
        )

        assert is_valid is False
        assert len(issues) > 0


class TestResampleOHLCVFunction:
    """Tests for convenience function."""

    def test_resample_ohlcv(self, sample_5m_data):
        """Test resample_ohlcv convenience function."""
        result = resample_ohlcv(sample_5m_data, "1H", "5M")

        assert len(result) < len(sample_5m_data)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_resample_ohlcv_auto_detect(self, sample_5m_data):
        """Test auto-detection of source timeframe."""
        result = resample_ohlcv(sample_5m_data, "1H")

        assert len(result) < len(sample_5m_data)
