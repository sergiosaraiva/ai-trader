"""Unit tests for feature engineering configuration."""

import pytest
from src.config.feature_config import (
    LagParameters,
    SessionParameters,
    CyclicalEncoding,
    FeatureParameters,
)


class TestLagParameters:
    """Test lag parameters configuration."""

    def test_defaults(self):
        """Test lag parameters load with correct defaults."""
        config = LagParameters()

        assert config.standard_lags == [1, 2, 3, 6, 12]
        assert config.rsi_roc_periods == [3, 6]
        assert config.macd_roc_periods == [3]
        assert config.adx_roc_periods == [3]
        assert config.atr_roc_periods == [3, 6]
        assert config.price_roc_periods == [1, 3, 6, 12]
        assert config.volume_roc_periods == [3, 6]

    def test_override(self):
        """Test lag parameters can be overridden."""
        config = LagParameters()
        config.standard_lags = [1, 5, 10]
        config.rsi_roc_periods = [5]

        assert config.standard_lags == [1, 5, 10]
        assert config.rsi_roc_periods == [5]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = LagParameters()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "standard_lags" in result
        assert result["standard_lags"] == [1, 2, 3, 6, 12]


class TestSessionParameters:
    """Test trading session parameters."""

    def test_defaults(self):
        """Test session parameters load with correct defaults."""
        config = SessionParameters()

        assert config.asian_session == (0, 8)
        assert config.london_session == (8, 16)
        assert config.ny_session == (13, 22)
        assert config.overlap_session == (13, 16)
        assert config.timezone_offset_hours == 0

    def test_override(self):
        """Test session parameters can be overridden."""
        config = SessionParameters()
        config.asian_session = (1, 9)
        config.timezone_offset_hours = -5

        assert config.asian_session == (1, 9)
        assert config.timezone_offset_hours == -5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = SessionParameters()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "asian_session" in result
        assert result["london_session"] == (8, 16)


class TestCyclicalEncoding:
    """Test cyclical encoding parameters."""

    def test_defaults(self):
        """Test cyclical encoding loads with correct defaults."""
        config = CyclicalEncoding()

        assert config.hour_encoding_cycles == 24
        assert config.day_of_week_cycles == 7
        assert config.day_of_month_cycles == 31

    def test_override(self):
        """Test cyclical encoding can be overridden."""
        config = CyclicalEncoding()
        config.hour_encoding_cycles = 12
        config.day_of_week_cycles = 5  # Weekdays only

        assert config.hour_encoding_cycles == 12
        assert config.day_of_week_cycles == 5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = CyclicalEncoding()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "hour_encoding_cycles" in result
        assert result["day_of_week_cycles"] == 7


class TestFeatureParameters:
    """Test complete feature parameters wrapper."""

    def test_defaults(self):
        """Test feature parameters load with correct structure."""
        config = FeatureParameters()

        assert isinstance(config.lags, LagParameters)
        assert isinstance(config.sessions, SessionParameters)
        assert isinstance(config.cyclical, CyclicalEncoding)
        assert config.percentile_window == 50
        assert config.zscore_window == 50

    def test_nested_access(self):
        """Test nested parameter access works."""
        config = FeatureParameters()

        assert config.lags.standard_lags == [1, 2, 3, 6, 12]
        assert config.sessions.asian_session == (0, 8)
        assert config.cyclical.hour_encoding_cycles == 24

    def test_override_normalization(self):
        """Test normalization parameters can be overridden."""
        config = FeatureParameters()
        config.percentile_window = 100
        config.zscore_window = 75

        assert config.percentile_window == 100
        assert config.zscore_window == 75

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = FeatureParameters()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "lags" in result
        assert "sessions" in result
        assert "cyclical" in result
        assert "percentile_window" in result
        assert isinstance(result["lags"], dict)
