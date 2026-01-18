"""Unit tests for TechnicalIndicatorCalculator."""

import pandas as pd
import pytest
import numpy as np

from src.features.technical.calculator import (
    TechnicalIndicatorCalculator,
    CalculatorConfig,
    calculate_indicators,
    get_feature_names,
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data."""
    dates = pd.date_range("2024-01-01", periods=200, freq="1h")
    np.random.seed(42)
    base_price = 1.1
    prices = base_price + np.cumsum(np.random.randn(200) * 0.001)

    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + np.random.rand(200) * 0.005,
            "low": prices - np.random.rand(200) * 0.005,
            "close": prices + np.random.randn(200) * 0.002,
            "volume": np.random.randint(100, 1000, 200).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def calculator():
    """Create calculator with default config."""
    return TechnicalIndicatorCalculator(model_type="medium_term")


class TestCalculatorConfig:
    """Tests for CalculatorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CalculatorConfig()
        assert config.model_type == "medium_term"
        assert config.drop_na is True
        assert config.fill_method == "ffill"
        assert len(config.enabled_categories) == 4

    def test_custom_config(self):
        """Test custom configuration."""
        config = CalculatorConfig(
            model_type="short_term",
            drop_na=False,
            fill_method="interpolate",
            max_priority=1,
        )
        assert config.model_type == "short_term"
        assert config.drop_na is False
        assert config.fill_method == "interpolate"
        assert config.max_priority == 1


class TestTechnicalIndicatorCalculator:
    """Tests for TechnicalIndicatorCalculator."""

    def test_init_default(self):
        """Test default initialization."""
        calculator = TechnicalIndicatorCalculator()
        assert calculator.config.model_type == "medium_term"

    def test_init_with_model_type(self):
        """Test initialization with model type."""
        calculator = TechnicalIndicatorCalculator(model_type="short_term")
        assert calculator.config.model_type == "short_term"

    def test_init_with_config(self):
        """Test initialization with config object."""
        config = CalculatorConfig(model_type="long_term")
        calculator = TechnicalIndicatorCalculator(config=config)
        assert calculator.config.model_type == "long_term"

    def test_init_with_indicator_config(self):
        """Test initialization with direct indicator config."""
        indicator_config = {
            "trend": {"sma": {"enabled": True, "periods": [10, 20]}},
            "momentum": {"rsi": {"enabled": True, "periods": [7]}},
        }
        calculator = TechnicalIndicatorCalculator(indicator_config=indicator_config)
        assert "trend" in calculator._indicator_config

    def test_calculate_returns_dataframe(self, calculator, sample_ohlcv_data):
        """Test calculate returns a DataFrame."""
        result = calculator.calculate(sample_ohlcv_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_calculate_adds_features(self, calculator, sample_ohlcv_data):
        """Test calculate adds technical features."""
        result = calculator.calculate(sample_ohlcv_data)
        # Should have more columns than original OHLCV
        assert len(result.columns) > 5

    def test_calculate_includes_derived_features(self, calculator, sample_ohlcv_data):
        """Test derived price features are added."""
        result = calculator.calculate(sample_ohlcv_data, include_derived=True)
        assert "returns" in result.columns
        assert "range" in result.columns
        assert "body" in result.columns

    def test_calculate_without_derived_features(self, calculator, sample_ohlcv_data):
        """Test calculation without derived features."""
        result = calculator.calculate(sample_ohlcv_data, include_derived=False)
        # Should still have some features from indicators
        assert len(result.columns) >= 5

    def test_calculate_preserves_datetime_index(self, calculator, sample_ohlcv_data):
        """Test datetime index is preserved."""
        result = calculator.calculate(sample_ohlcv_data)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_validate_input_missing_columns(self, calculator):
        """Test validation error for missing columns."""
        df = pd.DataFrame({"open": [1.0], "high": [1.1]}, index=pd.DatetimeIndex(["2024-01-01"]))
        with pytest.raises(ValueError, match="Missing required columns"):
            calculator.calculate(df)

    def test_validate_input_no_datetime_index(self, calculator):
        """Test validation error without DatetimeIndex."""
        df = pd.DataFrame(
            {"open": [1.0], "high": [1.1], "low": [0.9], "close": [1.05], "volume": [100]}
        )
        with pytest.raises(ValueError, match="DatetimeIndex"):
            calculator.calculate(df)

    def test_validate_input_empty_dataframe(self, calculator):
        """Test validation error for empty DataFrame."""
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([]),
        )
        with pytest.raises(ValueError, match="empty"):
            calculator.calculate(df)

    def test_get_feature_names(self, calculator, sample_ohlcv_data):
        """Test getting feature names after calculation."""
        calculator.calculate(sample_ohlcv_data)
        names = calculator.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0
        # Should not include original OHLCV
        assert "open" not in names
        assert "close" not in names

    def test_get_feature_count(self, calculator, sample_ohlcv_data):
        """Test getting feature count."""
        calculator.calculate(sample_ohlcv_data)
        count = calculator.get_feature_count()
        assert count > 0
        assert count == len(calculator.get_feature_names())

    def test_get_feature_groups(self, calculator, sample_ohlcv_data):
        """Test getting features organized by groups."""
        calculator.calculate(sample_ohlcv_data)
        groups = calculator.get_feature_groups()
        assert isinstance(groups, dict)
        # Should have at least some groups with features
        assert len(groups) > 0

    def test_calculate_for_model(self, sample_ohlcv_data):
        """Test calculate_for_model method."""
        calculator = TechnicalIndicatorCalculator()
        result = calculator.calculate_for_model(sample_ohlcv_data, "short_term")
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_get_config(self, calculator):
        """Test getting current configuration."""
        config = calculator.get_config()
        assert isinstance(config, dict)

    def test_set_config(self, calculator):
        """Test setting new configuration."""
        new_config = {"trend": {"sma": {"enabled": True, "periods": [5]}}}
        calculator.set_config(new_config)
        assert calculator._indicator_config == new_config

    def test_nan_handling_ffill(self, sample_ohlcv_data):
        """Test NaN handling with ffill."""
        config = CalculatorConfig(fill_method="ffill", drop_na=True)
        calculator = TechnicalIndicatorCalculator(config=config)
        result = calculator.calculate(sample_ohlcv_data)
        assert not result.isna().any().any()

    def test_nan_handling_interpolate(self, sample_ohlcv_data):
        """Test NaN handling with interpolation."""
        config = CalculatorConfig(fill_method="interpolate", drop_na=True)
        calculator = TechnicalIndicatorCalculator(config=config)
        result = calculator.calculate(sample_ohlcv_data)
        assert not result.isna().any().any()


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_calculate_indicators(self, sample_ohlcv_data):
        """Test calculate_indicators function."""
        result = calculate_indicators(sample_ohlcv_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_calculate_indicators_with_model_type(self, sample_ohlcv_data):
        """Test calculate_indicators with model type."""
        result = calculate_indicators(sample_ohlcv_data, model_type="short_term")
        assert isinstance(result, pd.DataFrame)

    def test_get_feature_names_with_df(self, sample_ohlcv_data):
        """Test get_feature_names with DataFrame."""
        names = get_feature_names(model_type="medium_term", df=sample_ohlcv_data)
        assert isinstance(names, list)
        assert len(names) > 0
