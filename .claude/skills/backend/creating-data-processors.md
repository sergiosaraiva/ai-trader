---
name: creating-data-processors
description: Creates data processor classes following the validate/clean/transform pipeline pattern. Use when building ETL pipelines, OHLCV data processing, or feature preparation for ML models. pandas/numpy stack.
---

# Creating Data Processors

## Quick Reference

- Create processor class with `validate()`, `clean()`, `transform()` methods
- Always `df.copy()` at start of methods to avoid mutation
- Return modified DataFrame for method chaining
- Store scalers in `self.scalers` dict for inverse transforms
- Raise `ValueError` with specific messages in validation

## When to Use

- Processing OHLCV market data
- Creating ML training sequences from time series
- Normalizing/denormalizing feature data
- Validating incoming data quality
- Resampling data to different timeframes

## When NOT to Use

- Simple column calculations (use indicator pattern)
- One-time data exploration (use notebooks)
- Real-time streaming data (use dedicated handler)

## Implementation Guide with Decision Tree

```
What processing step?
├─ Validation → validate() method, raises ValueError
│   └─ Check: columns exist, dtypes correct, no nulls
├─ Cleaning → clean() method, returns DataFrame
│   └─ Sort, dedupe, fill missing, remove invalid rows
├─ Transformation → add_derived_features() or normalize()
│   └─ Returns DataFrame, stores scalers if needed
└─ Sequence creation → create_sequences()
    └─ Returns (X, y) numpy arrays

Need to reverse transform?
├─ Yes → Store params in self.scalers dict
└─ No → Just transform and return
```

## Examples

**Example 1: Processor Class Structure**

```python
# From: src/data/processors/ohlcv.py:1-14
"""OHLCV data processor."""

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd


class OHLCVProcessor:
    """Process and transform OHLCV data."""

    def __init__(self):
        """Initialize OHLCV processor."""
        self.scalers = {}
```

**Explanation**: Simple init, `scalers` dict stores normalization parameters for later inverse transform.

**Example 2: Validation Method with Clear Errors**

```python
# From: src/data/processors/ohlcv.py:15-47
def validate(self, df: pd.DataFrame) -> bool:
    """
    Validate OHLCV dataframe.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        True if valid, raises ValueError otherwise
    """
    required_columns = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be DatetimeIndex")

    if df.isnull().any().any():
        null_counts = df.isnull().sum()
        raise ValueError(f"DataFrame contains null values: {null_counts}")

    # Validate OHLC relationships
    invalid_high = df["high"] < df[["open", "close"]].max(axis=1)
    invalid_low = df["low"] > df[["open", "close"]].min(axis=1)

    if invalid_high.any():
        raise ValueError(f"Invalid high values at: {df.index[invalid_high].tolist()[:5]}")
    if invalid_low.any():
        raise ValueError(f"Invalid low values at: {df.index[invalid_low].tolist()[:5]}")

    return True
```

**Explanation**: Validate method checks all invariants. Raises ValueError with specific details about what's wrong. Shows first 5 problematic rows.

**Example 3: Clean Method with Chaining**

```python
# From: src/data/processors/ohlcv.py:49-74
def clean(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean OHLCV data.

    Args:
        df: Raw OHLCV dataframe

    Returns:
        Cleaned dataframe
    """
    df = df.copy()

    # Sort by index
    df = df.sort_index()

    # Remove duplicates
    df = df[~df.index.duplicated(keep="last")]

    # Forward fill missing values (common in forex)
    df = df.ffill()

    # Remove rows with zero volume (market closed)
    if "volume" in df.columns:
        df = df[df["volume"] > 0]

    return df
```

**Explanation**: Always `df.copy()` first. Chain operations. Return cleaned DataFrame.

**Example 4: Derived Feature Addition**

```python
# From: src/data/processors/ohlcv.py:76-108
def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived price features.

    Args:
        df: OHLCV dataframe

    Returns:
        DataFrame with additional features
    """
    df = df.copy()

    # Returns
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Price range features
    df["range"] = df["high"] - df["low"]
    df["body"] = df["close"] - df["open"]
    df["body_pct"] = df["body"] / df["open"]

    # Shadows
    df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

    # Relative position
    df["close_position"] = (df["close"] - df["low"]) / (df["high"] - df["low"])

    # Gap
    df["gap"] = df["open"] - df["close"].shift(1)
    df["gap_pct"] = df["gap"] / df["close"].shift(1)

    return df
```

**Explanation**: Add candlestick-derived features. All calculations vectorized with pandas.

**Example 5: Normalization with Scaler Storage**

```python
# From: src/data/processors/ohlcv.py:147-192
def normalize(
    self,
    df: pd.DataFrame,
    method: str = "zscore",
    columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize dataframe columns.

    Args:
        df: Input dataframe
        method: Normalization method ('zscore', 'minmax', 'robust')
        columns: Columns to normalize (default: all numeric)

    Returns:
        Tuple of (normalized dataframe, scaler parameters)
    """
    df = df.copy()
    columns = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    scalers = {}

    for col in columns:
        if col not in df.columns:
            continue

        if method == "zscore":
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / (std + 1e-8)
            scalers[col] = {"method": "zscore", "mean": mean, "std": std}

        elif method == "minmax":
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)
            scalers[col] = {"method": "minmax", "min": min_val, "max": max_val}

        elif method == "robust":
            median = df[col].median()
            q75, q25 = df[col].quantile([0.75, 0.25])
            iqr = q75 - q25
            df[col] = (df[col] - median) / (iqr + 1e-8)
            scalers[col] = {"method": "robust", "median": median, "iqr": iqr}

    self.scalers = scalers
    return df, scalers
```

**Explanation**: Returns tuple of (df, scalers). Add `1e-8` to denominators to prevent division by zero. Store scalers for inverse transform.

**Example 6: Sequence Creation for Time Series**

```python
# From: src/data/processors/ohlcv.py:230-258
def create_sequences(
    self,
    df: pd.DataFrame,
    sequence_length: int,
    target_column: str = "close",
    prediction_horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series modeling.

    Args:
        df: Input dataframe
        sequence_length: Length of input sequences
        target_column: Column to predict
        prediction_horizon: Steps ahead to predict

    Returns:
        Tuple of (X sequences, y targets)
    """
    data = df.values
    target_idx = df.columns.tolist().index(target_column)

    X, y = [], []
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length + prediction_horizon - 1, target_idx])

    return np.array(X), np.array(y)
```

**Explanation**: Creates sliding window sequences. X shape: (samples, sequence_length, features). y shape: (samples,). CRITICAL: Use chronological order, no shuffling before split.

## Quality Checklist

- [ ] `df.copy()` at start of each method that modifies data
- [ ] Validation raises `ValueError` with specific error messages
- [ ] Clean method handles: sorting, deduplication, missing values
- [ ] Normalization stores parameters in `self.scalers`
- [ ] Denormalize method can reverse any normalize operation
- [ ] Sequence creation maintains chronological order
- [ ] All methods have docstrings with Args/Returns

## Common Mistakes

- **Mutating input DataFrame**: Side effects in caller → Always `df.copy()` first
- **Division by zero**: NaN/inf values → Add `1e-8` to denominators
- **Shuffling time series**: Data leakage → Keep chronological order in sequences
- **Missing scaler storage**: Can't inverse transform → Store all params in scalers dict

## Validation

- [ ] Pattern confirmed in `src/data/processors/ohlcv.py:1-258`
- [ ] Validation pattern at lines 15-47
- [ ] Sequence creation at lines 230-258

## OHLCV-Specific Processing

For OHLCV market data, apply these additional patterns:

### OHLCV Validation Rules
```python
# Validate OHLC relationships
invalid_high = df["high"] < df[["open", "close"]].max(axis=1)
invalid_low = df["low"] > df[["open", "close"]].min(axis=1)
if invalid_high.any() or invalid_low.any():
    raise ValueError("Invalid OHLC relationships")
```

### OHLCV Resampling (Timeframe Conversion)
```python
# Aggregate: open=first, high=max, low=min, close=last, volume=sum
resampled = df.resample("4h").agg({
    "open": "first", "high": "max", "low": "min",
    "close": "last", "volume": "sum"
}).dropna()
```

### Derived Candlestick Features
```python
df["returns"] = df["close"].pct_change()
df["range"] = df["high"] - df["low"]
df["body"] = df["close"] - df["open"]
df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]
```

See `src/data/processors/ohlcv.py` for complete implementation.

## Related Skills

- [creating-technical-indicators](../feature-engineering/creating-technical-indicators.md) - For adding indicator features
- [implementing-prediction-models](./implementing-prediction-models.md) - Consumes processed data
- [validating-time-series-data](../quality-testing/validating-time-series-data.md) - For data quality checks
