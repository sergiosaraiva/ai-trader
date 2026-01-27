"""Integration tests for technical calculator with configuration support."""

import pytest
import pandas as pd
import numpy as np
from src.features.technical.indicators import TechnicalIndicators
from src.config.trading_config import TradingConfig


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


class TestTechnicalIndicatorsIntegration:
    """Integration tests for full technical indicator pipeline."""

    def test_calculate_all_with_default_config(self, sample_data):
        """Test full pipeline with default configuration."""
        calculator = TechnicalIndicators()
        result = calculator.calculate_all(sample_data)

        # Verify original columns are preserved
        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns

        # Verify trend indicators exist
        assert "sma_20" in result.columns
        assert "ema_20" in result.columns
        assert "adx_14" in result.columns

        # Verify momentum indicators exist
        assert "rsi_14" in result.columns
        assert "macd" in result.columns
        assert "stoch_k_14" in result.columns

        # Verify volatility indicators exist
        assert "atr_14" in result.columns
        assert "bb_upper_20" in result.columns

        # Verify volume indicators exist
        assert "obv" in result.columns
        assert "cmf_20" in result.columns

    def test_calculate_all_with_custom_config(self, sample_data):
        """Test full pipeline with custom configuration."""
        config = TradingConfig()

        # Customize trend indicators
        config.indicators.trend.sma_periods = [10, 30]
        config.indicators.trend.ema_periods = [12, 26]
        config.indicators.trend.adx_period = 20

        # Customize momentum indicators
        config.indicators.momentum.rsi_periods = [9, 21]
        config.indicators.momentum.macd_fast = 8
        config.indicators.momentum.stochastic_k_period = 20

        # Customize volatility indicators
        config.indicators.volatility.atr_period = 20
        config.indicators.volatility.bollinger_period = 25

        # Customize volume indicators
        config.indicators.volume.cmf_period = 25
        config.indicators.volume.emv_period = 20

        calculator = TechnicalIndicators(config=config)
        result = calculator.calculate_all(sample_data)

        # Verify custom trend indicators
        assert "sma_10" in result.columns
        assert "sma_30" in result.columns
        assert "sma_20" not in result.columns  # Default should not be created
        assert "adx_20" in result.columns
        assert "adx_14" not in result.columns

        # Verify custom momentum indicators
        assert "rsi_9" in result.columns
        assert "rsi_21" in result.columns
        assert "rsi_14" not in result.columns
        assert "stoch_k_20" in result.columns
        assert "stoch_k_14" not in result.columns

        # Verify custom volatility indicators
        assert "atr_20" in result.columns
        assert "atr_14" not in result.columns
        assert "bb_upper_25" in result.columns
        assert "bb_upper_20" not in result.columns

        # Verify custom volume indicators
        assert "cmf_25" in result.columns
        assert "cmf_20" not in result.columns
        assert "emv_20" in result.columns
        assert "emv_14" not in result.columns

    def test_calculate_with_config_parameter_override(self, sample_data):
        """Test that config parameter overrides instance config."""
        instance_config = TradingConfig()
        instance_config.indicators.trend.sma_periods = [10, 20]

        param_config = TradingConfig()
        param_config.indicators.trend.sma_periods = [30, 50]

        calculator = TechnicalIndicators(config=instance_config)
        result = calculator.calculate_all(sample_data, config=param_config)

        # Should use param_config, not instance_config
        assert "sma_30" in result.columns
        assert "sma_50" in result.columns
        assert "sma_10" not in result.columns
        assert "sma_20" not in result.columns

    def test_selective_indicator_groups(self, sample_data):
        """Test calculating only specific indicator groups."""
        config = TradingConfig()
        config.indicators.trend.sma_periods = [20]
        config.indicators.momentum.rsi_periods = [14]

        calculator = TechnicalIndicators(config=config)

        # Calculate only trend indicators
        result_trend = calculator.calculate_all(
            sample_data,
            include_groups=["trend"]
        )
        assert "sma_20" in result_trend.columns
        assert "rsi_14" not in result_trend.columns

        # Calculate only momentum indicators
        result_momentum = calculator.calculate_all(
            sample_data,
            include_groups=["momentum"]
        )
        assert "rsi_14" in result_momentum.columns
        assert "sma_20" not in result_momentum.columns

    def test_feature_names_tracking(self, sample_data):
        """Test that feature names are tracked correctly."""
        calculator = TechnicalIndicators()
        result = calculator.calculate_all(sample_data)

        # Get feature names from all indicator groups
        trend_features = calculator.trend.get_feature_names()
        momentum_features = calculator.momentum.get_feature_names()
        volatility_features = calculator.volatility.get_feature_names()
        volume_features = calculator.volume.get_feature_names()

        # All should be in result
        for feature in trend_features + momentum_features + volatility_features + volume_features:
            assert feature in result.columns

    def test_backward_compatibility(self, sample_data):
        """Test that existing code without config still works."""
        # Old usage pattern (no config)
        calculator = TechnicalIndicators()
        result = calculator.calculate_all(sample_data)

        # Should produce all default indicators
        assert "sma_5" in result.columns
        assert "sma_200" in result.columns
        assert "rsi_7" in result.columns
        assert "rsi_14" in result.columns
        assert "atr_14" in result.columns
        assert "bb_upper_20" in result.columns

    def test_consistency_across_groups(self, sample_data):
        """Test that config is applied consistently across all groups."""
        config = TradingConfig()

        # Set specific values for each group
        config.indicators.trend.sma_periods = [15]
        config.indicators.momentum.rsi_periods = [15]
        config.indicators.volatility.atr_period = 15
        config.indicators.volume.cmf_period = 15

        calculator = TechnicalIndicators(config=config)
        result = calculator.calculate_all(sample_data)

        # All should use period 15
        assert "sma_15" in result.columns
        assert "rsi_15" in result.columns
        assert "atr_15" in result.columns
        assert "cmf_15" in result.columns

    def test_indicator_values_are_valid(self, sample_data):
        """Test that all calculated indicators have valid values."""
        calculator = TechnicalIndicators()
        result = calculator.calculate_all(sample_data)

        # Check for non-infinite values
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            non_nan_values = result[col].dropna()
            if len(non_nan_values) > 0:
                assert np.isfinite(non_nan_values).all(), f"Column {col} has infinite values"

    def test_multiple_calculations_with_different_configs(self, sample_data):
        """Test multiple calculations with different configs produce different results."""
        config1 = TradingConfig()
        config1.indicators.trend.sma_periods = [20]

        config2 = TradingConfig()
        config2.indicators.trend.sma_periods = [50]

        calculator = TechnicalIndicators()

        result1 = calculator.calculate_all(sample_data, config=config1)
        result2 = calculator.calculate_all(sample_data, config=config2)

        # Results should be different
        assert "sma_20" in result1.columns
        assert "sma_20" not in result2.columns
        assert "sma_50" in result2.columns
        assert "sma_50" not in result1.columns

    def test_empty_include_groups(self, sample_data):
        """Test with empty include_groups."""
        calculator = TechnicalIndicators()
        result = calculator.calculate_all(sample_data, include_groups=[])

        # Should only have original OHLCV columns
        indicator_cols = [col for col in result.columns
                         if col not in ["open", "high", "low", "close", "volume", "timestamp"]]
        assert len(indicator_cols) == 0

    def test_config_independence_between_instances(self, sample_data):
        """Test that different calculator instances don't share config."""
        config1 = TradingConfig()
        config1.indicators.trend.sma_periods = [10]

        config2 = TradingConfig()
        config2.indicators.trend.sma_periods = [20]

        calc1 = TechnicalIndicators(config=config1)
        calc2 = TechnicalIndicators(config=config2)

        result1 = calc1.calculate_all(sample_data, include_groups=["trend"])
        result2 = calc2.calculate_all(sample_data, include_groups=["trend"])

        assert "sma_10" in result1.columns
        assert "sma_10" not in result2.columns
        assert "sma_20" in result2.columns
        assert "sma_20" not in result1.columns

    def test_large_period_values(self, sample_data):
        """Test that large period values are handled correctly."""
        config = TradingConfig()
        config.indicators.trend.sma_periods = [100, 200]
        config.indicators.momentum.rsi_periods = [50]
        config.indicators.volatility.atr_period = 50

        calculator = TechnicalIndicators(config=config)
        result = calculator.calculate_all(sample_data)

        # Should create indicators even with large periods
        assert "sma_100" in result.columns
        assert "sma_200" in result.columns
        assert "rsi_50" in result.columns
        assert "atr_50" in result.columns

        # Values should exist after warmup period
        assert result["sma_100"].notna().sum() > 0
        assert result["sma_200"].notna().sum() > 0

    def test_minimal_config(self, sample_data):
        """Test with minimal indicator configuration."""
        config = TradingConfig()
        config.indicators.trend.sma_periods = [20]
        config.indicators.trend.ema_periods = []
        config.indicators.trend.wma_periods = []
        config.indicators.momentum.rsi_periods = [14]
        config.indicators.momentum.cci_periods = []
        config.indicators.volatility.std_periods = []
        config.indicators.volume.volume_sma_periods = []

        calculator = TechnicalIndicators(config=config)
        result = calculator.calculate_all(sample_data)

        # Should only have minimal indicators
        assert "sma_20" in result.columns
        assert "rsi_14" in result.columns
        assert "ema_5" not in result.columns
        assert "wma_10" not in result.columns
        assert "cci_14" not in result.columns
