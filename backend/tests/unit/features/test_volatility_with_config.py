"""Unit tests for volatility indicators with configuration support."""

import pytest
import pandas as pd
import numpy as np
from src.features.technical.volatility import VolatilityIndicators
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


class TestVolatilityIndicatorsWithConfig:
    """Test volatility indicators accept and use config correctly."""

    def test_calculate_all_with_default_config(self, sample_data):
        """Test calculate_all works with default config."""
        indicators = VolatilityIndicators()
        result = indicators.calculate_all(sample_data)

        # Verify default ATR period
        assert "atr_14" in result.columns

        # Verify default NATR period
        assert "natr_14" in result.columns

        # Verify default Bollinger Bands (period=20, std=2)
        assert "bb_upper_20" in result.columns
        assert "bb_middle_20" in result.columns
        assert "bb_lower_20" in result.columns
        assert "bb_width_20" in result.columns
        assert "bb_pct_20" in result.columns

        # Verify default Keltner Channel (period=20, multiplier=2)
        assert "kc_upper_20" in result.columns
        assert "kc_middle_20" in result.columns
        assert "kc_lower_20" in result.columns

        # Verify default Donchian Channel (period=20)
        assert "dc_upper_20" in result.columns
        assert "dc_middle_20" in result.columns
        assert "dc_lower_20" in result.columns

        # Verify default stddev periods
        assert "stddev_10" in result.columns
        assert "stddev_20" in result.columns

        # Verify default historical volatility periods
        assert "hvol_10" in result.columns
        assert "hvol_20" in result.columns
        assert "hvol_30" in result.columns

    def test_calculate_all_with_custom_config(self, sample_data):
        """Test calculate_all works with custom config."""
        config = TradingConfig()
        config.indicators.volatility.atr_period = 20
        config.indicators.volatility.natr_period = 20
        config.indicators.volatility.bollinger_period = 25
        config.indicators.volatility.bollinger_std = 2.5
        config.indicators.volatility.keltner_period = 25
        config.indicators.volatility.keltner_multiplier = 2.5
        config.indicators.volatility.donchian_period = 25
        config.indicators.volatility.std_periods = [15, 30]
        config.indicators.volatility.hvol_periods = [15, 25]

        indicators = VolatilityIndicators()
        result = indicators.calculate_all(sample_data, config=config)

        # Verify custom ATR period
        assert "atr_20" in result.columns
        assert "atr_14" not in result.columns  # Should not be created

        # Verify custom NATR period
        assert "natr_20" in result.columns
        assert "natr_14" not in result.columns  # Should not be created

        # Verify custom Bollinger Bands
        assert "bb_upper_25" in result.columns
        assert "bb_middle_25" in result.columns
        assert "bb_lower_25" in result.columns
        assert "bb_upper_20" not in result.columns  # Should not be created

        # Verify custom Keltner Channel
        assert "kc_upper_25" in result.columns
        assert "kc_middle_25" in result.columns
        assert "kc_lower_25" in result.columns
        assert "kc_upper_20" not in result.columns  # Should not be created

        # Verify custom Donchian Channel
        assert "dc_upper_25" in result.columns
        assert "dc_middle_25" in result.columns
        assert "dc_lower_25" in result.columns
        assert "dc_upper_20" not in result.columns  # Should not be created

        # Verify custom stddev periods
        assert "stddev_15" in result.columns
        assert "stddev_30" in result.columns
        assert "stddev_10" not in result.columns  # Should not be created

        # Verify custom historical volatility periods
        assert "hvol_15" in result.columns
        assert "hvol_25" in result.columns
        assert "hvol_10" not in result.columns  # Should not be created

    def test_bollinger_std_deviation(self, sample_data):
        """Test Bollinger Bands use custom standard deviation."""
        config1 = TradingConfig()
        config1.indicators.volatility.bollinger_period = 20
        config1.indicators.volatility.bollinger_std = 2.0

        config2 = TradingConfig()
        config2.indicators.volatility.bollinger_period = 20
        config2.indicators.volatility.bollinger_std = 3.0

        indicators1 = VolatilityIndicators()
        result1 = indicators1.calculate_all(sample_data, config=config1)

        indicators2 = VolatilityIndicators()
        result2 = indicators2.calculate_all(sample_data, config=config2)

        # Width should be different due to different std deviations
        width1 = result1["bb_width_20"].dropna()
        width2 = result2["bb_width_20"].dropna()
        assert not np.allclose(width1, width2)
        # Width with std=3 should be larger
        assert width2.mean() > width1.mean()

    def test_keltner_multiplier(self, sample_data):
        """Test Keltner Channel uses custom multiplier."""
        config1 = TradingConfig()
        config1.indicators.volatility.keltner_period = 20
        config1.indicators.volatility.keltner_multiplier = 2.0

        config2 = TradingConfig()
        config2.indicators.volatility.keltner_period = 20
        config2.indicators.volatility.keltner_multiplier = 3.0

        indicators1 = VolatilityIndicators()
        result1 = indicators1.calculate_all(sample_data, config=config1)

        indicators2 = VolatilityIndicators()
        result2 = indicators2.calculate_all(sample_data, config=config2)

        # Band width should be different due to different multipliers
        upper1 = result1["kc_upper_20"].dropna()
        lower1 = result1["kc_lower_20"].dropna()
        width1 = (upper1 - lower1).mean()

        upper2 = result2["kc_upper_20"].dropna()
        lower2 = result2["kc_lower_20"].dropna()
        width2 = (upper2 - lower2).mean()

        assert not np.allclose(width1, width2)
        # Width with multiplier=3 should be larger
        assert width2 > width1

    def test_hvol_annualization_factor(self, sample_data):
        """Test historical volatility uses annualization factor."""
        config1 = TradingConfig()
        config1.indicators.volatility.hvol_periods = [20]
        config1.indicators.volatility.hvol_annualization_factor = 252

        config2 = TradingConfig()
        config2.indicators.volatility.hvol_periods = [20]
        config2.indicators.volatility.hvol_annualization_factor = 365

        indicators1 = VolatilityIndicators()
        result1 = indicators1.calculate_all(sample_data, config=config1)

        indicators2 = VolatilityIndicators()
        result2 = indicators2.calculate_all(sample_data, config=config2)

        # Values should be different due to different annualization factors
        hvol1 = result1["hvol_20"].dropna()
        hvol2 = result2["hvol_20"].dropna()
        assert not np.allclose(hvol1, hvol2)

    def test_backward_compatibility_no_config(self, sample_data):
        """Test backward compatibility when no config is passed."""
        indicators = VolatilityIndicators()
        result = indicators.calculate_all(sample_data)

        # Should use defaults and work exactly as before
        assert "atr_14" in result.columns
        assert "natr_14" in result.columns
        assert "bb_upper_20" in result.columns
        assert "kc_upper_20" in result.columns
        assert "dc_upper_20" in result.columns
        assert "stddev_10" in result.columns
        assert "hvol_20" in result.columns

    def test_indicator_values_are_valid(self, sample_data):
        """Test that indicator values are calculated correctly."""
        indicators = VolatilityIndicators()
        result = indicators.calculate_all(sample_data)

        # ATR should be positive
        atr = result["atr_14"].dropna()
        assert (atr > 0).all()

        # NATR should be positive percentage
        natr = result["natr_14"].dropna()
        assert (natr > 0).all()

        # Bollinger bands should be ordered: lower < middle < upper
        bb_lower = result["bb_lower_20"].dropna()
        bb_middle = result["bb_middle_20"].dropna()
        bb_upper = result["bb_upper_20"].dropna()
        assert (bb_lower <= bb_middle).all()
        assert (bb_middle <= bb_upper).all()

        # Bollinger %B should be between 0 and 1 for most values
        bb_pct = result["bb_pct_20"].dropna()
        # Allow some outliers, but most should be in range
        in_range_pct = ((bb_pct >= 0) & (bb_pct <= 1)).sum() / len(bb_pct)
        assert in_range_pct > 0.7  # At least 70% in range

        # Standard deviation should be positive
        stddev = result["stddev_10"].dropna()
        assert (stddev > 0).all()

    def test_feature_names_tracking(self, sample_data):
        """Test that feature names are correctly tracked."""
        indicators = VolatilityIndicators()
        result = indicators.calculate_all(sample_data)
        feature_names = indicators.get_feature_names()

        # Verify feature names list is populated
        assert len(feature_names) > 0

        # Verify all tracked features exist in result
        for feature in feature_names:
            assert feature in result.columns

    def test_donchian_channel_values(self, sample_data):
        """Test Donchian Channel calculations are correct."""
        config = TradingConfig()
        config.indicators.volatility.donchian_period = 20

        indicators = VolatilityIndicators()
        result = indicators.calculate_all(sample_data, config=config)

        # Upper should be max high, lower should be min low
        dc_upper = result["dc_upper_20"].dropna()
        dc_lower = result["dc_lower_20"].dropna()
        dc_middle = result["dc_middle_20"].dropna()

        # Middle should be average of upper and lower
        expected_middle = (dc_upper + dc_lower) / 2
        np.testing.assert_array_almost_equal(
            dc_middle.values, expected_middle.values, decimal=10
        )

        # Upper should be >= lower
        assert (dc_upper >= dc_lower).all()

    def test_empty_periods_list(self, sample_data):
        """Test handling of empty periods lists."""
        config = TradingConfig()
        config.indicators.volatility.std_periods = []
        config.indicators.volatility.hvol_periods = []

        indicators = VolatilityIndicators()
        result = indicators.calculate_all(sample_data, config=config)

        # Should not create any stddev columns
        stddev_cols = [col for col in result.columns if col.startswith("stddev_")]
        assert len(stddev_cols) == 0

        # Should not create any hvol columns
        hvol_cols = [col for col in result.columns if col.startswith("hvol_")]
        assert len(hvol_cols) == 0

    def test_true_range_calculation(self, sample_data):
        """Test True Range calculation is correct."""
        indicators = VolatilityIndicators()
        tr = indicators.true_range(sample_data)

        # True Range should be positive
        assert (tr.dropna() > 0).all()

        # True Range should be at least high - low
        hl_range = sample_data["high"] - sample_data["low"]
        assert (tr.dropna() >= hl_range.loc[tr.dropna().index]).all()
