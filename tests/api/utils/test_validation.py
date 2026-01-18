"""Tests for validation utilities.

This module tests all validation functions including:
- Numeric value clamping and conversion
- DataFrame access with bounds checking
- Data structure validation
- Safe division operations
"""

import pytest
import pandas as pd
import numpy as np

from src.api.utils.validation import (
    clamp,
    safe_float,
    safe_iloc,
    validate_dataframe,
    safe_division,
)


class TestClamp:
    """Tests for the clamp function."""

    def test_value_within_range(self):
        """Test that values within range are unchanged."""
        assert clamp(50, 0, 100) == 50
        assert clamp(25.5, 0, 100) == 25.5
        assert clamp(0.5, 0, 1) == 0.5

    def test_value_below_minimum(self):
        """Test that values below minimum are clamped to min."""
        assert clamp(-10, 0, 100) == 0
        assert clamp(-50, 0, 100) == 0
        assert clamp(-0.5, 0, 1) == 0

    def test_value_above_maximum(self):
        """Test that values above maximum are clamped to max."""
        assert clamp(150, 0, 100) == 100
        assert clamp(200, 0, 100) == 100
        assert clamp(1.5, 0, 1) == 1

    def test_boundary_values(self):
        """Test that boundary values are handled correctly."""
        assert clamp(0, 0, 100) == 0
        assert clamp(100, 0, 100) == 100
        assert clamp(50, 50, 50) == 50

    def test_negative_range(self):
        """Test clamping with negative ranges."""
        assert clamp(-5, -10, -1) == -5
        assert clamp(-15, -10, -1) == -10
        assert clamp(0, -10, -1) == -1


class TestSafeFloat:
    """Tests for the safe_float function."""

    def test_valid_numeric_values(self):
        """Test conversion of valid numeric types."""
        assert safe_float(42) == 42.0
        assert safe_float(3.14) == 3.14
        assert safe_float("123.45") == 123.45
        assert safe_float("100") == 100.0

    def test_invalid_values_return_default(self):
        """Test that invalid values return the default."""
        assert safe_float(None) == 0.0
        assert safe_float("abc") == 0.0
        assert safe_float("") == 0.0
        assert safe_float([1, 2, 3]) == 0.0

    def test_custom_default_value(self):
        """Test using custom default values."""
        assert safe_float(None, default=10.0) == 10.0
        assert safe_float("invalid", default=-1.0) == -1.0
        assert safe_float("", default=99.9) == 99.9

    def test_min_bound_enforcement(self):
        """Test that minimum bounds are enforced."""
        assert safe_float(50, min_val=100) == 100
        assert safe_float(-10, min_val=0) == 0
        assert safe_float(100, min_val=100) == 100

    def test_max_bound_enforcement(self):
        """Test that maximum bounds are enforced."""
        assert safe_float(150, max_val=100) == 100
        assert safe_float(50, max_val=100) == 50
        assert safe_float(100, max_val=100) == 100

    def test_both_bounds_enforcement(self):
        """Test that both min and max bounds are enforced."""
        assert safe_float(150, min_val=0, max_val=100) == 100.0
        assert safe_float(-10, min_val=0, max_val=100) == 0.0
        assert safe_float(50, min_val=0, max_val=100) == 50.0

    def test_string_with_whitespace(self):
        """Test conversion of strings with whitespace."""
        assert safe_float("  123.45  ") == 123.45
        assert safe_float("\t100\n") == 100.0

    @pytest.mark.parametrize(
        "value,expected",
        [
            (0, 0.0),
            (1, 1.0),
            (-1, -1.0),
            (1e10, 1e10),
            (1e-10, 1e-10),
        ],
    )
    def test_edge_case_numbers(self, value, expected):
        """Test edge case numeric values."""
        assert safe_float(value) == expected


class TestSafeIloc:
    """Tests for the safe_iloc function."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        return pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

    def test_valid_index_with_column(self, sample_df):
        """Test valid DataFrame access with column name."""
        assert safe_iloc(sample_df, 0, "a") == 1
        assert safe_iloc(sample_df, 2, "b") == 30
        assert safe_iloc(sample_df, 4, "a") == 5

    def test_valid_negative_index(self, sample_df):
        """Test negative indexing (from end of DataFrame)."""
        assert safe_iloc(sample_df, -1, "a") == 5
        assert safe_iloc(sample_df, -2, "b") == 40
        assert safe_iloc(sample_df, -5, "a") == 1

    def test_index_without_column(self, sample_df):
        """Test accessing entire row without column specification."""
        row = safe_iloc(sample_df, 0)
        assert row is not None
        assert row["a"] == 1
        assert row["b"] == 10

    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame()
        assert safe_iloc(empty_df, 0, default="empty") == "empty"
        assert safe_iloc(empty_df, -1, default=None) is None

    def test_none_dataframe(self):
        """Test behavior with None DataFrame."""
        assert safe_iloc(None, 0, default="none") == "none"
        assert safe_iloc(None, 0, "a", default=-1) == -1

    def test_index_out_of_bounds_positive(self, sample_df):
        """Test positive index out of bounds."""
        assert safe_iloc(sample_df, 10, "a", default=0) == 0
        assert safe_iloc(sample_df, 100, "b", default=-1) == -1
        assert safe_iloc(sample_df, 5, "a", default=999) == 999

    def test_index_out_of_bounds_negative(self, sample_df):
        """Test negative index out of bounds."""
        assert safe_iloc(sample_df, -10, "a", default=0) == 0
        assert safe_iloc(sample_df, -100, "b", default=-1) == -1

    def test_missing_column(self, sample_df):
        """Test accessing a column that doesn't exist."""
        assert safe_iloc(sample_df, 0, "missing_col", default="not_found") == "not_found"
        assert safe_iloc(sample_df, 2, "xyz", default=None) is None

    def test_boundary_indices(self, sample_df):
        """Test boundary indices (first and last)."""
        # First row
        assert safe_iloc(sample_df, 0, "a") == 1
        # Last row
        assert safe_iloc(sample_df, 4, "a") == 5
        # Using -1 for last
        assert safe_iloc(sample_df, -1, "a") == 5

    def test_with_numeric_column_index(self, sample_df):
        """Test accessing DataFrame with numeric column index."""
        assert safe_iloc(sample_df, 0, 0) == 1
        assert safe_iloc(sample_df, 2, 1) == 30

    @pytest.mark.parametrize(
        "default_value",
        [0, -1, None, "default", [], {}],
    )
    def test_various_default_types(self, default_value):
        """Test that various default value types are returned correctly."""
        empty_df = pd.DataFrame()
        assert safe_iloc(empty_df, 0, default=default_value) == default_value


class TestValidateDataFrame:
    """Tests for the validate_dataframe function."""

    @pytest.fixture
    def valid_df(self):
        """Create a valid DataFrame for testing."""
        return pd.DataFrame(
            {"open": [1, 2, 3], "high": [4, 5, 6], "low": [0, 1, 2], "close": [2, 3, 4]}
        )

    def test_valid_dataframe(self, valid_df):
        """Test that valid DataFrame passes validation."""
        assert validate_dataframe(valid_df, ["open", "high", "low", "close"], min_rows=1)
        assert validate_dataframe(valid_df, ["open", "close"], min_rows=2)
        assert validate_dataframe(valid_df, ["open"], min_rows=3)

    def test_none_dataframe(self):
        """Test that None DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="is None"):
            validate_dataframe(None, ["open"], min_rows=1)

    def test_empty_dataframe(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="is empty"):
            validate_dataframe(empty_df, ["open"], min_rows=1)

    def test_insufficient_rows(self, valid_df):
        """Test that DataFrame with too few rows raises ValueError."""
        with pytest.raises(ValueError, match="has 3 rows, but 5 required"):
            validate_dataframe(valid_df, ["open"], min_rows=5)

        with pytest.raises(ValueError, match="has 3 rows, but 10 required"):
            validate_dataframe(valid_df, ["open", "close"], min_rows=10)

    def test_missing_single_column(self, valid_df):
        """Test that missing single column raises ValueError."""
        with pytest.raises(ValueError, match="missing required columns.*missing"):
            validate_dataframe(valid_df, ["open", "missing"], min_rows=1)

    def test_missing_multiple_columns(self, valid_df):
        """Test that missing multiple columns raises ValueError."""
        with pytest.raises(ValueError, match="missing required columns.*xyz.*abc"):
            validate_dataframe(valid_df, ["open", "xyz", "abc"], min_rows=1)

    def test_custom_name_in_error(self):
        """Test that custom DataFrame name appears in error messages."""
        df = pd.DataFrame({"a": [1]})

        with pytest.raises(ValueError, match="MyDataFrame is empty"):
            validate_dataframe(pd.DataFrame(), ["a"], min_rows=1, name="MyDataFrame")

        with pytest.raises(ValueError, match="PriceData missing required columns"):
            validate_dataframe(df, ["missing"], min_rows=1, name="PriceData")

    def test_exact_min_rows(self, valid_df):
        """Test validation with exact minimum rows."""
        assert validate_dataframe(valid_df, ["open"], min_rows=3)

    def test_no_required_columns(self, valid_df):
        """Test validation with no required columns."""
        assert validate_dataframe(valid_df, [], min_rows=1)

    def test_all_columns_required(self, valid_df):
        """Test validation requiring all columns."""
        required = ["open", "high", "low", "close"]
        assert validate_dataframe(valid_df, required, min_rows=1)

    def test_partial_columns_required(self, valid_df):
        """Test validation with subset of columns."""
        assert validate_dataframe(valid_df, ["open", "close"], min_rows=1)
        assert validate_dataframe(valid_df, ["high"], min_rows=1)

    def test_single_row_dataframe(self):
        """Test validation with single-row DataFrame."""
        single_row_df = pd.DataFrame({"a": [1], "b": [2]})
        assert validate_dataframe(single_row_df, ["a", "b"], min_rows=1)

        with pytest.raises(ValueError, match="has 1 rows, but 2 required"):
            validate_dataframe(single_row_df, ["a"], min_rows=2)

    def test_error_message_includes_available_columns(self, valid_df):
        """Test that error message includes available columns."""
        with pytest.raises(
            ValueError,
            match="Available columns:.*open.*high.*low.*close",
        ):
            validate_dataframe(valid_df, ["missing"], min_rows=1)


class TestSafeDivision:
    """Tests for the safe_division function."""

    def test_normal_division(self):
        """Test normal division operations."""
        assert safe_division(10, 2) == 5.0
        assert safe_division(100, 4) == 25.0
        assert safe_division(7, 2) == 3.5
        assert safe_division(1, 3) == pytest.approx(0.333333, rel=1e-5)

    def test_division_by_zero_returns_default(self):
        """Test that division by zero returns the default value."""
        assert safe_division(10, 0) == 0.0
        assert safe_division(100, 0) == 0.0
        assert safe_division(-5, 0) == 0.0

    def test_division_by_zero_custom_default(self):
        """Test division by zero with custom default."""
        assert safe_division(10, 0, default=1.0) == 1.0
        assert safe_division(100, 0, default=-1.0) == -1.0
        assert safe_division(5, 0, default=99.9) == 99.9

    def test_zero_numerator(self):
        """Test division with zero numerator."""
        assert safe_division(0, 10) == 0.0
        assert safe_division(0, 5) == 0.0
        assert safe_division(0, 1) == 0.0

    def test_negative_values(self):
        """Test division with negative values."""
        assert safe_division(-10, 2) == -5.0
        assert safe_division(10, -2) == -5.0
        assert safe_division(-10, -2) == 5.0

    def test_floating_point_division(self):
        """Test division with floating point numbers."""
        assert safe_division(10.5, 2.5) == 4.2
        assert safe_division(1.0, 3.0) == pytest.approx(0.333333, rel=1e-5)
        assert safe_division(0.1, 0.2) == 0.5

    def test_large_numbers(self):
        """Test division with large numbers."""
        assert safe_division(1e10, 1e5) == 1e5
        assert safe_division(1e20, 1e10) == 1e10

    def test_small_numbers(self):
        """Test division with very small numbers."""
        assert safe_division(1e-5, 1e-3) == pytest.approx(0.01, rel=1e-5)
        assert safe_division(1e-10, 1e-5) == pytest.approx(1e-5, rel=1e-5)

    @pytest.mark.parametrize(
        "numerator,denominator,expected",
        [
            (10, 2, 5.0),
            (100, 10, 10.0),
            (7, 4, 1.75),
            (1, 1, 1.0),
            (-10, 2, -5.0),
        ],
    )
    def test_division_cases(self, numerator, denominator, expected):
        """Test various division cases with parametrization."""
        assert safe_division(numerator, denominator) == expected

    def test_exact_zero_denominator(self):
        """Test that exactly zero denominator is handled."""
        assert safe_division(5, 0.0, default=0.0) == 0.0
        assert safe_division(5, -0.0, default=0.0) == 0.0
