"""Validation utilities for API services.

This module provides common validation and safety functions for:
- Numeric value clamping and safe conversions
- DataFrame access with bounds checking
- Data structure validation

All functions follow defensive programming principles to prevent
runtime errors from invalid data.
"""

import logging
from typing import Any, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a numeric value between min and max bounds.

    Args:
        value: The value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value between min_val and max_val

    Example:
        >>> clamp(150, 0, 100)
        100
        >>> clamp(-10, 0, 100)
        0
        >>> clamp(50, 0, 100)
        50
    """
    return max(min_val, min(max_val, value))


def safe_float(
    value: Any,
    default: float = 0.0,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> float:
    """Safely convert a value to float with optional bounds checking.

    Args:
        value: Value to convert (can be str, int, float, or None)
        default: Default value if conversion fails
        min_val: Optional minimum value for clamping
        max_val: Optional maximum value for clamping

    Returns:
        Float value, clamped if bounds provided

    Example:
        >>> safe_float("123.45")
        123.45
        >>> safe_float("invalid", default=0.0)
        0.0
        >>> safe_float(150, min_val=0, max_val=100)
        100.0
    """
    try:
        result = float(value)
        if min_val is not None and max_val is not None:
            result = clamp(result, min_val, max_val)
        elif min_val is not None:
            result = max(min_val, result)
        elif max_val is not None:
            result = min(max_val, result)
        return result
    except (TypeError, ValueError, AttributeError):
        logger.debug(
            f"Failed to convert value to float: {value}, using default: {default}"
        )
        return default


def safe_iloc(
    df: pd.DataFrame,
    index: int,
    column: Optional[Union[str, int]] = None,
    default: Any = None,
) -> Any:
    """Safely access DataFrame using iloc with bounds checking.

    Args:
        df: DataFrame to access
        index: Row index to access (can be negative for reverse indexing)
        column: Optional column name or index
        default: Default value if access fails

    Returns:
        Value at the specified location, or default if access fails

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3]})
        >>> safe_iloc(df, -1, 'a')
        3
        >>> safe_iloc(df, 10, 'a', default=0)
        0
    """
    if df is None or df.empty:
        logger.debug("DataFrame is None or empty, returning default")
        return default

    try:
        # Check bounds
        if index < 0:
            # Negative indexing
            if abs(index) > len(df):
                logger.debug(
                    f"Index {index} out of bounds for DataFrame with {len(df)} rows"
                )
                return default
        else:
            # Positive indexing
            if index >= len(df):
                logger.debug(
                    f"Index {index} out of bounds for DataFrame with {len(df)} rows"
                )
                return default

        # Access the value
        if column is not None:
            # Use .loc for column access to avoid FutureWarning
            row = df.iloc[index]
            if isinstance(column, int):
                return row.iloc[column]
            else:
                return row[column]
        else:
            return df.iloc[index]

    except (IndexError, KeyError, AttributeError) as e:
        logger.debug(f"Failed to access DataFrame iloc[{index}]: {e}")
        return default


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    min_rows: int = 1,
    name: str = "DataFrame",
) -> bool:
    """Validate that a DataFrame has the expected structure.

    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        min_rows: Minimum number of rows required
        name: Name of the DataFrame for error messages

    Returns:
        True if validation passes

    Raises:
        ValueError: If validation fails with descriptive error message

    Example:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> validate_dataframe(df, ['a', 'b'], min_rows=2)
        True
        >>> validate_dataframe(df, ['a', 'c'], min_rows=1)
        ValueError: DataFrame validation failed...
    """
    if df is None:
        raise ValueError(f"{name} is None")

    if df.empty:
        raise ValueError(f"{name} is empty")

    if len(df) < min_rows:
        raise ValueError(
            f"{name} has {len(df)} rows, but {min_rows} required"
        )

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"{name} missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    return True


def safe_division(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """Safely perform division with zero-denominator protection.

    Args:
        numerator: The numerator value
        denominator: The denominator value
        default: Default value to return if denominator is zero

    Returns:
        Result of division, or default if denominator is zero

    Example:
        >>> safe_division(10, 2)
        5.0
        >>> safe_division(10, 0, default=0.0)
        0.0
    """
    if denominator == 0:
        logger.debug(
            f"Division by zero avoided: {numerator}/{denominator}, returning {default}"
        )
        return default
    return numerator / denominator
