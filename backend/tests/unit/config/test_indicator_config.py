"""Unit tests for indicator configuration."""

import pytest
from src.config.indicator_config import (
    TrendIndicators,
    MomentumIndicators,
    VolatilityIndicators,
    VolumeIndicators,
    IndicatorParameters,
)


class TestTrendIndicators:
    """Test trend indicator configuration."""

    def test_defaults(self):
        """Test trend indicators load with correct defaults."""
        config = TrendIndicators()

        assert config.sma_periods == [5, 10, 20, 50, 100, 200]
        assert config.ema_periods == [5, 10, 20, 50, 100, 200]
        assert config.wma_periods == [10, 20, 50]
        assert config.adx_period == 14
        assert config.aroon_period == 25
        assert config.supertrend_period == 10
        assert config.supertrend_multiplier == 3.0
        assert config.sma_crossover_pairs == [(5, 20), (20, 50), (50, 200)]
        assert config.ema_crossover_pairs == [(5, 20), (12, 26)]

    def test_override(self):
        """Test trend indicators can be overridden."""
        config = TrendIndicators()
        config.sma_periods = [10, 20, 50]
        config.adx_period = 20

        assert config.sma_periods == [10, 20, 50]
        assert config.adx_period == 20

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TrendIndicators()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "sma_periods" in result
        assert "ema_periods" in result
        assert result["adx_period"] == 14


class TestMomentumIndicators:
    """Test momentum indicator configuration."""

    def test_defaults(self):
        """Test momentum indicators load with correct defaults."""
        config = MomentumIndicators()

        assert config.rsi_periods == [7, 14, 21]
        assert config.stochastic_k_period == 14
        assert config.stochastic_d_period == 3
        assert config.macd_fast == 12
        assert config.macd_slow == 26
        assert config.macd_signal == 9
        assert config.cci_periods == [14, 20]
        assert config.cci_constant == 0.015
        assert config.williams_period == 14
        assert config.mfi_period == 14

    def test_override(self):
        """Test momentum indicators can be overridden."""
        config = MomentumIndicators()
        config.rsi_periods = [14, 28, 42]
        config.macd_fast = 10

        assert config.rsi_periods == [14, 28, 42]
        assert config.macd_fast == 10

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = MomentumIndicators()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "rsi_periods" in result
        assert result["macd_fast"] == 12


class TestVolatilityIndicators:
    """Test volatility indicator configuration."""

    def test_defaults(self):
        """Test volatility indicators load with correct defaults."""
        config = VolatilityIndicators()

        assert config.atr_period == 14
        assert config.natr_period == 14
        assert config.bollinger_period == 20
        assert config.bollinger_std == 2.0
        assert config.keltner_period == 20
        assert config.keltner_multiplier == 2.0
        assert config.donchian_period == 20
        assert config.std_periods == [10, 20]

    def test_override(self):
        """Test volatility indicators can be overridden."""
        config = VolatilityIndicators()
        config.bollinger_period = 30
        config.bollinger_std = 2.5

        assert config.bollinger_period == 30
        assert config.bollinger_std == 2.5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = VolatilityIndicators()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "atr_period" in result
        assert result["bollinger_period"] == 20


class TestVolumeIndicators:
    """Test volume indicator configuration."""

    def test_defaults(self):
        """Test volume indicators load with correct defaults."""
        config = VolumeIndicators()

        assert config.cmf_period == 20
        assert config.emv_period == 14
        assert config.emv_scaling_factor == 1e8
        assert config.force_index_period == 13
        assert config.adosc_fast == 3
        assert config.adosc_slow == 10
        assert config.volume_sma_periods == [10, 20]

    def test_override(self):
        """Test volume indicators can be overridden."""
        config = VolumeIndicators()
        config.cmf_period = 25
        config.volume_sma_periods = [5, 10, 20]

        assert config.cmf_period == 25
        assert config.volume_sma_periods == [5, 10, 20]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = VolumeIndicators()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "cmf_period" in result
        assert result["emv_period"] == 14


class TestIndicatorParameters:
    """Test complete indicator parameters wrapper."""

    def test_defaults(self):
        """Test indicator parameters load with correct structure."""
        config = IndicatorParameters()

        assert isinstance(config.trend, TrendIndicators)
        assert isinstance(config.momentum, MomentumIndicators)
        assert isinstance(config.volatility, VolatilityIndicators)
        assert isinstance(config.volume, VolumeIndicators)

    def test_nested_access(self):
        """Test nested parameter access works."""
        config = IndicatorParameters()

        assert config.trend.sma_periods == [5, 10, 20, 50, 100, 200]
        assert config.momentum.rsi_periods == [7, 14, 21]
        assert config.volatility.atr_period == 14
        assert config.volume.cmf_period == 20

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = IndicatorParameters()
        result = config.to_dict()

        assert isinstance(result, dict)
        assert "trend" in result
        assert "momentum" in result
        assert "volatility" in result
        assert "volume" in result
        assert isinstance(result["trend"], dict)
