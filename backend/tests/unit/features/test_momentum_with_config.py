"""Unit tests for momentum indicators with configuration support."""

import pytest
import pandas as pd
import numpy as np
from src.features.technical.momentum import MomentumIndicators
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


class TestMomentumIndicatorsWithConfig:
    """Test momentum indicators accept and use config correctly."""

    def test_calculate_all_with_default_config(self, sample_data):
        """Test calculate_all works with default config."""
        indicators = MomentumIndicators()
        result = indicators.calculate_all(sample_data)

        # Verify default RSI periods
        assert "rsi_7" in result.columns
        assert "rsi_14" in result.columns
        assert "rsi_21" in result.columns

        # Verify default Stochastic periods
        assert "stoch_k_14" in result.columns
        assert "stoch_d_14" in result.columns

        # Verify default MACD
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

        # Verify default CCI periods
        assert "cci_14" in result.columns
        assert "cci_20" in result.columns

        # Verify Williams %R with default period
        assert "willr_14" in result.columns

        # Verify MFI with default period
        assert "mfi_14" in result.columns

        # Verify TSI
        assert "tsi" in result.columns

    def test_calculate_all_with_custom_config(self, sample_data):
        """Test calculate_all works with custom config."""
        config = TradingConfig()
        config.indicators.momentum.rsi_periods = [9, 21]
        config.indicators.momentum.stochastic_k_period = 20
        config.indicators.momentum.stochastic_d_period = 5
        config.indicators.momentum.macd_fast = 8
        config.indicators.momentum.macd_slow = 21
        config.indicators.momentum.macd_signal = 5
        config.indicators.momentum.cci_periods = [20]
        config.indicators.momentum.williams_period = 21
        config.indicators.momentum.mfi_period = 20
        config.indicators.momentum.tsi_long = 20
        config.indicators.momentum.tsi_short = 10

        indicators = MomentumIndicators()
        result = indicators.calculate_all(sample_data, config=config)

        # Verify custom RSI periods
        assert "rsi_9" in result.columns
        assert "rsi_21" in result.columns
        assert "rsi_7" not in result.columns  # Should not be created
        assert "rsi_14" not in result.columns  # Should not be created

        # Verify custom Stochastic periods
        assert "stoch_k_20" in result.columns
        assert "stoch_d_20" in result.columns
        assert "stoch_k_14" not in result.columns  # Should not be created

        # Verify custom CCI periods
        assert "cci_20" in result.columns
        assert "cci_14" not in result.columns  # Should not be created

        # Verify custom Williams %R period
        assert "willr_21" in result.columns
        assert "willr_14" not in result.columns  # Should not be created

        # Verify custom MFI period
        assert "mfi_20" in result.columns
        assert "mfi_14" not in result.columns  # Should not be created

    def test_momentum_and_roc_periods(self, sample_data):
        """Test momentum and ROC use periods from config."""
        config = TradingConfig()
        config.indicators.momentum.momentum_periods = [5, 12]
        config.indicators.momentum.roc_periods = [5, 12]

        indicators = MomentumIndicators()
        result = indicators.calculate_all(sample_data, config=config)

        # Verify custom momentum periods
        assert "mom_5" in result.columns
        assert "mom_12" in result.columns
        assert "mom_10" not in result.columns  # Default should not be created

        # Verify custom ROC periods
        assert "roc_5" in result.columns
        assert "roc_12" in result.columns
        assert "roc_10" not in result.columns  # Default should not be created

    def test_cci_constant_from_config(self, sample_data):
        """Test CCI uses constant from config."""
        config = TradingConfig()
        config.indicators.momentum.cci_periods = [14]
        config.indicators.momentum.cci_constant = 0.020

        indicators = MomentumIndicators()
        result1 = indicators.calculate_all(sample_data, config=config)

        # Compare with default constant
        config2 = TradingConfig()
        config2.indicators.momentum.cci_periods = [14]
        config2.indicators.momentum.cci_constant = 0.015

        indicators2 = MomentumIndicators()
        result2 = indicators2.calculate_all(sample_data, config=config2)

        # Values should be different due to different constants
        cci1 = result1["cci_14"].dropna()
        cci2 = result2["cci_14"].dropna()
        assert not np.allclose(cci1, cci2)

    def test_backward_compatibility_no_config(self, sample_data):
        """Test backward compatibility when no config is passed."""
        indicators = MomentumIndicators()
        result = indicators.calculate_all(sample_data)

        # Should use defaults and work exactly as before
        assert "rsi_7" in result.columns
        assert "rsi_14" in result.columns
        assert "rsi_21" in result.columns
        assert "stoch_k_14" in result.columns
        assert "macd" in result.columns
        assert "cci_14" in result.columns
        assert "willr_14" in result.columns
        assert "mfi_14" in result.columns

    def test_indicator_values_are_valid(self, sample_data):
        """Test that indicator values are calculated correctly."""
        indicators = MomentumIndicators()
        result = indicators.calculate_all(sample_data)

        # RSI should be between 0 and 100
        rsi_14 = result["rsi_14"].dropna()
        assert (rsi_14 >= 0).all()
        assert (rsi_14 <= 100).all()

        # Stochastic should be between 0 and 100
        stoch_k = result["stoch_k_14"].dropna()
        assert (stoch_k >= 0).all()
        assert (stoch_k <= 100).all()

        # Williams %R should be between -100 and 0
        willr = result["willr_14"].dropna()
        assert (willr >= -100).all()
        assert (willr <= 0).all()

        # MFI should be between 0 and 100
        mfi = result["mfi_14"].dropna()
        assert (mfi >= 0).all()
        assert (mfi <= 100).all()

        # MACD should be numeric
        assert result["macd"].notna().sum() > 0
        assert np.isfinite(result["macd"].dropna()).all()

    def test_feature_names_tracking(self, sample_data):
        """Test that feature names are correctly tracked."""
        indicators = MomentumIndicators()
        result = indicators.calculate_all(sample_data)
        feature_names = indicators.get_feature_names()

        # Verify feature names list is populated
        assert len(feature_names) > 0

        # Verify all tracked features exist in result
        for feature in feature_names:
            assert feature in result.columns

    def test_macd_with_custom_parameters(self, sample_data):
        """Test MACD calculation with custom parameters."""
        config = TradingConfig()
        config.indicators.momentum.macd_fast = 8
        config.indicators.momentum.macd_slow = 21
        config.indicators.momentum.macd_signal = 5

        indicators = MomentumIndicators()
        result = indicators.calculate_all(sample_data, config=config)

        # MACD columns should exist
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_hist" in result.columns

        # Verify histogram is difference between MACD and signal
        macd_hist = result["macd_hist"].dropna()
        macd_line = result.loc[macd_hist.index, "macd"]
        signal_line = result.loc[macd_hist.index, "macd_signal"]
        np.testing.assert_array_almost_equal(
            macd_hist.values, (macd_line - signal_line).values, decimal=10
        )

    def test_tsi_with_custom_parameters(self, sample_data):
        """Test TSI calculation with custom parameters."""
        config = TradingConfig()
        config.indicators.momentum.tsi_long = 20
        config.indicators.momentum.tsi_short = 10

        indicators = MomentumIndicators()
        result = indicators.calculate_all(sample_data, config=config)

        # TSI should exist and be numeric
        assert "tsi" in result.columns
        tsi_values = result["tsi"].dropna()
        assert len(tsi_values) > 0
        assert np.isfinite(tsi_values).all()

    def test_empty_periods_list(self, sample_data):
        """Test handling of empty periods lists."""
        config = TradingConfig()
        config.indicators.momentum.rsi_periods = []
        config.indicators.momentum.cci_periods = []
        config.indicators.momentum.momentum_periods = []
        config.indicators.momentum.roc_periods = []

        indicators = MomentumIndicators()
        result = indicators.calculate_all(sample_data, config=config)

        # Should not create any RSI columns
        rsi_cols = [col for col in result.columns if col.startswith("rsi_")]
        assert len(rsi_cols) == 0

        # Should not create any CCI columns
        cci_cols = [col for col in result.columns if col.startswith("cci_")]
        assert len(cci_cols) == 0

        # Should not create any momentum columns
        mom_cols = [col for col in result.columns if col.startswith("mom_")]
        assert len(mom_cols) == 0

        # Should not create any ROC columns
        roc_cols = [col for col in result.columns if col.startswith("roc_")]
        assert len(roc_cols) == 0
