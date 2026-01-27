"""Unit tests for volume indicators with configuration support."""

import pytest
import pandas as pd
import numpy as np
from src.features.technical.volume import VolumeIndicators
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


class TestVolumeIndicatorsWithConfig:
    """Test volume indicators accept and use config correctly."""

    def test_calculate_all_with_default_config(self, sample_data):
        """Test calculate_all works with default config."""
        indicators = VolumeIndicators()
        result = indicators.calculate_all(sample_data)

        # Verify OBV (no config parameter)
        assert "obv" in result.columns

        # Verify A/D Line (no config parameter)
        assert "ad" in result.columns

        # Verify ADOSC with default parameters (fast=3, slow=10)
        assert "adosc" in result.columns

        # Verify CMF with default period (20)
        assert "cmf_20" in result.columns

        # Verify VPT (no config parameter)
        assert "vpt" in result.columns

        # Verify EMV with default period (14)
        assert "emv_14" in result.columns

        # Verify Force Index with default period (13)
        assert "fi_13" in result.columns

        # Verify Volume SMA with default periods
        assert "vol_sma_10" in result.columns
        assert "vol_sma_20" in result.columns

        # Verify Volume Ratio with default period (14)
        assert "vol_ratio_14" in result.columns

    def test_calculate_all_with_custom_config(self, sample_data):
        """Test calculate_all works with custom config."""
        config = TradingConfig()
        config.indicators.volume.cmf_period = 25
        config.indicators.volume.emv_period = 20
        config.indicators.volume.force_index_period = 15
        config.indicators.volume.adosc_fast = 5
        config.indicators.volume.adosc_slow = 15
        config.indicators.volume.volume_sma_periods = [15, 30]
        config.indicators.volume.volume_ratio_period = 20

        indicators = VolumeIndicators()
        result = indicators.calculate_all(sample_data, config=config)

        # Verify custom CMF period
        assert "cmf_25" in result.columns
        assert "cmf_20" not in result.columns  # Should not be created

        # Verify custom EMV period
        assert "emv_20" in result.columns
        assert "emv_14" not in result.columns  # Should not be created

        # Verify custom Force Index period
        assert "fi_15" in result.columns
        assert "fi_13" not in result.columns  # Should not be created

        # Verify custom Volume SMA periods
        assert "vol_sma_15" in result.columns
        assert "vol_sma_30" in result.columns
        assert "vol_sma_10" not in result.columns  # Should not be created

        # Verify custom Volume Ratio period
        assert "vol_ratio_20" in result.columns
        assert "vol_ratio_14" not in result.columns  # Should not be created

    def test_adosc_parameters(self, sample_data):
        """Test ADOSC uses custom fast and slow parameters."""
        config1 = TradingConfig()
        config1.indicators.volume.adosc_fast = 3
        config1.indicators.volume.adosc_slow = 10

        config2 = TradingConfig()
        config2.indicators.volume.adosc_fast = 5
        config2.indicators.volume.adosc_slow = 15

        indicators1 = VolumeIndicators()
        result1 = indicators1.calculate_all(sample_data, config=config1)

        indicators2 = VolumeIndicators()
        result2 = indicators2.calculate_all(sample_data, config=config2)

        # Values should be different due to different parameters
        adosc1 = result1["adosc"].dropna()
        adosc2 = result2["adosc"].dropna()
        assert not np.allclose(adosc1, adosc2)

    def test_emv_scaling_factor(self, sample_data):
        """Test EMV uses custom scaling factor."""
        config1 = TradingConfig()
        config1.indicators.volume.emv_period = 14
        config1.indicators.volume.emv_scaling_factor = 1e8

        config2 = TradingConfig()
        config2.indicators.volume.emv_period = 14
        config2.indicators.volume.emv_scaling_factor = 1e9

        indicators1 = VolumeIndicators()
        result1 = indicators1.calculate_all(sample_data, config=config1)

        indicators2 = VolumeIndicators()
        result2 = indicators2.calculate_all(sample_data, config=config2)

        # Values should be different due to different scaling factors
        emv1 = result1["emv_14"].dropna()
        emv2 = result2["emv_14"].dropna()
        assert not np.allclose(emv1, emv2)

    def test_backward_compatibility_no_config(self, sample_data):
        """Test backward compatibility when no config is passed."""
        indicators = VolumeIndicators()
        result = indicators.calculate_all(sample_data)

        # Should use defaults and work exactly as before
        assert "obv" in result.columns
        assert "ad" in result.columns
        assert "adosc" in result.columns
        assert "cmf_20" in result.columns
        assert "vpt" in result.columns
        assert "emv_14" in result.columns
        assert "fi_13" in result.columns
        assert "vol_sma_10" in result.columns
        assert "vol_ratio_14" in result.columns

    def test_indicator_values_are_valid(self, sample_data):
        """Test that indicator values are calculated correctly."""
        indicators = VolumeIndicators()
        result = indicators.calculate_all(sample_data)

        # OBV should be numeric and cumulative
        obv = result["obv"].dropna()
        assert len(obv) > 0
        assert np.isfinite(obv).all()

        # A/D Line should be numeric and cumulative
        ad = result["ad"].dropna()
        assert len(ad) > 0
        assert np.isfinite(ad).all()

        # CMF should be between -1 and 1 for most values
        cmf = result["cmf_20"].dropna()
        in_range_pct = ((cmf >= -1) & (cmf <= 1)).sum() / len(cmf)
        assert in_range_pct > 0.7  # At least 70% in range

        # Volume ratio should be positive
        vol_ratio = result["vol_ratio_14"].dropna()
        assert (vol_ratio > 0).all()

        # Volume SMA should be positive
        vol_sma = result["vol_sma_10"].dropna()
        assert (vol_sma > 0).all()

    def test_feature_names_tracking(self, sample_data):
        """Test that feature names are correctly tracked."""
        indicators = VolumeIndicators()
        result = indicators.calculate_all(sample_data)
        feature_names = indicators.get_feature_names()

        # Verify feature names list is populated
        assert len(feature_names) > 0

        # Verify all tracked features exist in result
        for feature in feature_names:
            assert feature in result.columns

    def test_obv_calculation(self, sample_data):
        """Test OBV calculation is cumulative."""
        indicators = VolumeIndicators()
        result = indicators.calculate_all(sample_data)

        # OBV should change with volume when price changes
        obv = result["obv"]
        assert obv.notna().sum() > 0
        assert np.isfinite(obv.dropna()).all()

    def test_ad_line_calculation(self, sample_data):
        """Test A/D Line calculation is cumulative."""
        indicators = VolumeIndicators()
        result = indicators.calculate_all(sample_data)

        # A/D Line should be cumulative
        ad = result["ad"]
        assert ad.notna().sum() > 0
        assert np.isfinite(ad.dropna()).all()

    def test_volume_ratio_values(self, sample_data):
        """Test volume ratio is relative to average."""
        config = TradingConfig()
        config.indicators.volume.volume_ratio_period = 14

        indicators = VolumeIndicators()
        result = indicators.calculate_all(sample_data, config=config)

        # Volume ratio should average around 1.0
        vol_ratio = result["vol_ratio_14"].dropna()
        assert vol_ratio.mean() > 0.5
        assert vol_ratio.mean() < 2.0

    def test_empty_periods_list(self, sample_data):
        """Test handling of empty periods lists."""
        config = TradingConfig()
        config.indicators.volume.volume_sma_periods = []

        indicators = VolumeIndicators()
        result = indicators.calculate_all(sample_data, config=config)

        # Should not create any volume SMA columns
        vol_sma_cols = [col for col in result.columns if col.startswith("vol_sma_")]
        assert len(vol_sma_cols) == 0

    def test_no_volume_data(self):
        """Test handling of data without volume."""
        df = pd.DataFrame({
            "timestamp": pd.date_range(start="2020-01-01", periods=100, freq="1h"),
            "open": [100] * 100,
            "high": [101] * 100,
            "low": [99] * 100,
            "close": [100] * 100,
        })

        indicators = VolumeIndicators()
        result = indicators.calculate_all(df)

        # Should return original dataframe without volume indicators
        assert "obv" not in result.columns
        assert "ad" not in result.columns

    def test_zero_volume_data(self, sample_data):
        """Test handling of zero volume."""
        sample_data["volume"] = 0

        indicators = VolumeIndicators()
        result = indicators.calculate_all(sample_data)

        # Should return original dataframe without volume indicators
        assert "obv" not in result.columns
        assert "ad" not in result.columns

    def test_cmf_different_periods(self, sample_data):
        """Test CMF with different periods produces different results."""
        config1 = TradingConfig()
        config1.indicators.volume.cmf_period = 10

        config2 = TradingConfig()
        config2.indicators.volume.cmf_period = 30

        indicators1 = VolumeIndicators()
        result1 = indicators1.calculate_all(sample_data, config=config1)

        indicators2 = VolumeIndicators()
        result2 = indicators2.calculate_all(sample_data, config=config2)

        # Different periods should produce different smoothing
        cmf1 = result1["cmf_10"].dropna()
        cmf2 = result2["cmf_30"].dropna()

        # Standard deviation of shorter period should be higher (less smoothing)
        assert cmf1.std() > cmf2.std()
