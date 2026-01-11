---
name: validating-time-series-data
description: Validates time series data for ML models including chronological integrity, OHLCV relationships, and data leakage prevention. Use when preparing trading data for training or backtesting. pandas/numpy validation patterns.
---

# Validating Time Series Data

## Quick Reference

- Index must be `DatetimeIndex` (sorted ascending)
- OHLCV validation: `high >= max(open, close)`, `low <= min(open, close)`
- No future data leakage: train/val/test must be chronological
- Drop NaN after indicator calculation, not before
- Never shuffle time series before train/test split

## When to Use

- Preparing data for model training
- Loading external data sources
- Before running backtests
- After feature engineering
- Data quality checks in pipelines

## When NOT to Use

- Real-time tick validation (different pattern)
- Non-OHLCV data (use generic validation)
- Simple existence checks (use asserts)

## Implementation Guide with Decision Tree

```
Validation flow:
├─ Structure validation
│   ├─ Required columns exist? → ['open', 'high', 'low', 'close', 'volume']
│   ├─ DatetimeIndex? → isinstance(df.index, pd.DatetimeIndex)
│   └─ Sorted ascending? → df.index.is_monotonic_increasing
├─ Value validation
│   ├─ OHLC relationships? → high >= open/close, low <= open/close
│   ├─ No null values? → df.isnull().any().any()
│   └─ Positive values? → (df[['open','high','low','close']] > 0).all().all()
├─ Time series validation
│   ├─ No duplicate timestamps? → ~df.index.duplicated()
│   ├─ Regular frequency? → pd.infer_freq(df.index)
│   └─ No gaps > expected? → df.index.diff().max()
└─ ML-specific validation
    ├─ Chronological split? → train_end < val_start < test_start
    ├─ No future leakage? → Features don't use future data
    └─ Enough samples? → len(df) > sequence_length + horizon
```

## Examples

**Example 1: Complete OHLCV Validation**

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
    # High must be >= the highest of open and close
    invalid_high = df["high"] < df[["open", "close"]].max(axis=1)
    # Low must be <= the lowest of open and close
    invalid_low = df["low"] > df[["open", "close"]].min(axis=1)

    if invalid_high.any():
        raise ValueError(f"Invalid high values at: {df.index[invalid_high].tolist()[:5]}")
    if invalid_low.any():
        raise ValueError(f"Invalid low values at: {df.index[invalid_low].tolist()[:5]}")

    return True
```

**Explanation**: Comprehensive validation: columns, index type, nulls, OHLC relationships. Show first 5 problematic rows in errors.

**Example 2: Chronological Split Validation**

```python
# Validate chronological train/val/test split
def validate_chronological_split(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    timestamps_train: pd.DatetimeIndex,
    timestamps_val: pd.DatetimeIndex,
    timestamps_test: pd.DatetimeIndex,
) -> bool:
    """
    Validate that splits are chronological (no data leakage).

    CRITICAL: Time series must be split chronologically, never shuffled.
    Train < Val < Test in time order.
    """
    # Check timestamps are monotonically increasing within each set
    if not timestamps_train.is_monotonic_increasing:
        raise ValueError("Train timestamps not monotonically increasing")
    if not timestamps_val.is_monotonic_increasing:
        raise ValueError("Val timestamps not monotonically increasing")
    if not timestamps_test.is_monotonic_increasing:
        raise ValueError("Test timestamps not monotonically increasing")

    # Check splits don't overlap
    if timestamps_train.max() >= timestamps_val.min():
        raise ValueError(
            f"Train/Val overlap: train max {timestamps_train.max()} >= "
            f"val min {timestamps_val.min()}"
        )
    if timestamps_val.max() >= timestamps_test.min():
        raise ValueError(
            f"Val/Test overlap: val max {timestamps_val.max()} >= "
            f"test min {timestamps_test.min()}"
        )

    print(f"Train: {timestamps_train.min()} to {timestamps_train.max()}")
    print(f"Val:   {timestamps_val.min()} to {timestamps_val.max()}")
    print(f"Test:  {timestamps_test.min()} to {timestamps_test.max()}")

    return True
```

**Explanation**: CRITICAL check for time series ML. Train must end before val starts. Val must end before test starts. No overlap allowed.

**Example 3: Data Leakage Detection**

```python
# Check for future data leakage in features
def check_feature_leakage(
    df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = "close",
) -> List[str]:
    """
    Detect potential future data leakage in features.

    Leakage occurs when features use information from the future.
    Common causes:
    - Using future returns (shift with positive number)
    - Target encoding without proper cross-validation
    - Rolling calculations including current row
    """
    suspicious_features = []

    for col in feature_columns:
        # Check if feature is correlated with future target
        future_returns = df[target_column].pct_change().shift(-1)  # Next period return

        # High correlation with future might indicate leakage
        correlation = df[col].corr(future_returns)
        if abs(correlation) > 0.5:  # Threshold
            suspicious_features.append({
                "column": col,
                "correlation_with_future": correlation,
                "warning": "High correlation with future returns"
            })

    if suspicious_features:
        print("WARNING: Potential data leakage detected:")
        for s in suspicious_features:
            print(f"  {s['column']}: {s['warning']} (r={s['correlation_with_future']:.3f})")

    return suspicious_features
```

**Explanation**: Detect features that might use future information. High correlation with future returns is suspicious. Investigate before training.

**Example 4: Sequence Creation Validation**

```python
# From: src/data/processors/ohlcv.py:230-258
# Validate sequence creation maintains temporal integrity
def validate_sequences(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: pd.DatetimeIndex,
    sequence_length: int,
    prediction_horizon: int,
) -> bool:
    """
    Validate created sequences maintain temporal integrity.

    Args:
        X: Feature sequences (n_samples, sequence_length, n_features)
        y: Target values (n_samples,)
        timestamps: Original timestamps
        sequence_length: Lookback window
        prediction_horizon: Steps ahead to predict

    Returns:
        True if valid
    """
    n_samples = len(X)
    expected_samples = len(timestamps) - sequence_length - prediction_horizon + 1

    if n_samples != expected_samples:
        raise ValueError(
            f"Sample count mismatch: got {n_samples}, expected {expected_samples}"
        )

    # Check sequence shapes
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y sample count mismatch: {X.shape[0]} vs {y.shape[0]}")

    if X.shape[1] != sequence_length:
        raise ValueError(f"Sequence length mismatch: got {X.shape[1]}, expected {sequence_length}")

    # Verify no future data in X
    # X[i] should only contain data from timestamps[i:i+sequence_length]
    # y[i] should be from timestamps[i+sequence_length+prediction_horizon-1]

    print(f"Sequences validated:")
    print(f"  Samples: {n_samples}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  First X uses: timestamps[0:{sequence_length}]")
    print(f"  First y from: timestamps[{sequence_length + prediction_horizon - 1}]")

    return True
```

**Explanation**: Verify sequence creation logic. X contains lookback, y contains future target. Sample count should match expected.

**Example 5: Indicator NaN Handling**

```python
# Validate NaN handling after indicator calculation
def validate_indicator_nans(
    df: pd.DataFrame,
    indicator_columns: List[str],
    warmup_periods: int = 200,
) -> pd.DataFrame:
    """
    Validate and handle NaN values from indicator calculations.

    Indicators need warmup period (e.g., 200-period SMA needs 200 bars).
    NaN rows should be dropped AFTER calculation, not before.

    Args:
        df: DataFrame with indicators
        indicator_columns: List of indicator column names
        warmup_periods: Expected warmup period

    Returns:
        DataFrame with NaN rows dropped
    """
    # Check NaN pattern
    nan_counts = df[indicator_columns].isnull().sum()
    has_nans = nan_counts[nan_counts > 0]

    if len(has_nans) > 0:
        print(f"NaN counts by indicator (expected from warmup):")
        for col, count in has_nans.items():
            print(f"  {col}: {count} NaN values")

    # Drop rows with NaN in any indicator
    df_clean = df.dropna(subset=indicator_columns)
    dropped = len(df) - len(df_clean)

    if dropped > warmup_periods * 1.5:
        print(f"WARNING: Dropped {dropped} rows (expected ~{warmup_periods})")
        print("Check for data quality issues beyond warmup period")

    print(f"Rows after dropping NaN: {len(df_clean)} (dropped {dropped})")

    return df_clean
```

**Explanation**: Indicators create NaN for warmup period. Drop AFTER calculation. Warn if too many dropped (indicates data issues).

**Example 6: Complete Validation Pipeline**

```python
# Complete time series validation pipeline
import pandas as pd
import numpy as np
from src.data.processors.ohlcv import OHLCVProcessor
from src.features.technical import TechnicalIndicators

def validate_training_data(
    filepath: str,
    sequence_length: int = 168,
    prediction_horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict:
    """
    Complete validation pipeline for training data.

    Returns dict with validated X_train, X_val, X_test, y_train, y_val, y_test.
    """
    # 1. Load data
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    df.columns = [c.lower() for c in df.columns]

    # 2. Validate OHLCV structure
    processor = OHLCVProcessor()
    processor.validate(df)
    print("OHLCV structure: VALID")

    # 3. Clean data
    df = processor.clean(df)

    # 4. Verify monotonic index
    if not df.index.is_monotonic_increasing:
        raise ValueError("Index not monotonically increasing after cleaning")
    print("Index monotonic: VALID")

    # 5. Add indicators
    indicators = TechnicalIndicators()
    df = indicators.calculate_all(df)
    feature_names = indicators.trend.get_feature_names() + indicators.momentum.get_feature_names()

    # 6. Handle indicator NaN
    df = validate_indicator_nans(df, feature_names)

    # 7. Normalize
    df_norm, scalers = processor.normalize(df, method="zscore")

    # 8. Create sequences
    X, y = processor.create_sequences(
        df_norm,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
    )

    # 9. Chronological split (NO SHUFFLE)
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # 10. Validate splits
    timestamps = df.index[sequence_length + prediction_horizon - 1:]
    timestamps_train = timestamps[:train_end]
    timestamps_val = timestamps[train_end:val_end]
    timestamps_test = timestamps[val_end:]

    validate_chronological_split(
        X_train, X_val, X_test,
        timestamps_train, timestamps_val, timestamps_test
    )

    print("\nFinal dataset shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scalers": scalers,
        "feature_names": feature_names,
    }
```

**Explanation**: Complete pipeline: load, validate structure, clean, add features, handle NaN, normalize, sequence, split, validate split. Returns ready-to-train data.

## Quality Checklist

- [ ] Index is DatetimeIndex
- [ ] Index is sorted (monotonic increasing)
- [ ] No duplicate timestamps
- [ ] OHLCV relationships valid (high >= open/close, low <= open/close)
- [ ] No null values in critical columns
- [ ] Train/val/test split is chronological (not shuffled)
- [ ] NaN dropped AFTER indicator calculation
- [ ] Enough samples after dropping NaN
- [ ] No future data leakage in features

## Common Mistakes

- **Shuffling before split**: Data leakage → NEVER shuffle time series
- **Dropping NaN before indicators**: Loses data → Drop AFTER calculation
- **Using future in features**: Leakage → shift(-n) looks into future
- **Random split**: Leakage → Use chronological split only
- **Ignoring OHLC validation**: Bad data → Always check high/low relationships

## Validation

- [ ] OHLCV validation in `src/data/processors/ohlcv.py:15-47`
- [ ] Sequence creation in `src/data/processors/ohlcv.py:230-258`

## Related Skills

- [processing-ohlcv-data](../data-layer/processing-ohlcv-data.md) - For data processing
- [creating-technical-indicators](../feature-engineering/creating-technical-indicators.md) - For indicator NaN handling
- [implementing-prediction-models](../backend/implementing-prediction-models.md) - Consumes validated data
