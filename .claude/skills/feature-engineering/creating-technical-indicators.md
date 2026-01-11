---
name: creating-technical-indicators
description: Creates technical indicator calculator classes following the project's feature tracking pattern. Use when adding new trend, momentum, volatility, or volume indicators for trading models. pandas/numpy stack with pandas-ta support.
---

# Creating Technical Indicators

## Quick Reference

- Create calculator class with `__init__`, `get_feature_names()`, `calculate_all()` methods
- Track generated columns in `self._feature_names: List[str]`
- Always `df.copy()` at start, return modified DataFrame for chaining
- Name columns as `indicator_period` (e.g., `sma_20`, `rsi_14`)
- Reset `_feature_names = []` at start of `calculate_all()`

## When to Use

- Adding new trend indicators (SMA, EMA, ADX, Aroon)
- Creating momentum oscillators (RSI, MACD, Stochastic)
- Implementing volatility measures (ATR, Bollinger Bands)
- Building volume indicators (OBV, VWAP, CMF)
- Creating derived features (price-to-MA, crossovers)

## When NOT to Use

- Raw OHLCV processing (use data processor)
- Model-specific feature engineering (do in model)
- Non-technical features (fundamental, sentiment)

## Implementation Guide with Decision Tree

```
What indicator category?
├─ Trend → src/features/technical/trend.py
│   └─ Uses: close, high, low
├─ Momentum → src/features/technical/momentum.py
│   └─ Uses: close, high, low (some need volume)
├─ Volatility → src/features/technical/volatility.py
│   └─ Uses: high, low, close
└─ Volume → src/features/technical/volume.py
    └─ Uses: close, volume (requires volume data)

Multiple periods needed?
├─ Yes → Loop over periods list, append each to _feature_names
└─ No → Single calculation, single name append
```

## Examples

**Example 1: Indicator Calculator Class Structure**

```python
# From: src/features/technical/trend.py:1-18
"""Trend indicators for technical analysis."""

from typing import List
import numpy as np
import pandas as pd


class TrendIndicators:
    """Calculate trend-based technical indicators."""

    def __init__(self):
        """Initialize trend indicators."""
        self._feature_names: List[str] = []

    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self._feature_names.copy()
```

**Explanation**: Simple structure. `_feature_names` tracks all columns added. Return copy in getter to prevent external mutation.

**Example 2: calculate_all() Entry Point**

```python
# From: src/features/technical/trend.py:19-39
def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all trend indicators."""
    df = df.copy()
    self._feature_names = []

    # Moving averages
    df = self.sma(df, periods=[5, 10, 20, 50, 100, 200])
    df = self.ema(df, periods=[5, 10, 20, 50, 100, 200])
    df = self.wma(df, periods=[10, 20, 50])

    # Trend direction
    df = self.adx(df, period=14)
    df = self.aroon(df, period=25)

    # Price relative to MAs
    df = self.price_to_ma(df)

    # MA crossovers
    df = self.ma_crossovers(df)

    return df
```

**Explanation**: Reset `_feature_names` to empty list. Chain all indicator methods. Each method appends to `_feature_names`. Return df for chaining.

**Example 3: Multi-Period Indicator (SMA, EMA)**

```python
# From: src/features/technical/trend.py:41-55
def sma(self, df: pd.DataFrame, periods: List[int], column: str = "close") -> pd.DataFrame:
    """Simple Moving Average."""
    for period in periods:
        col_name = f"sma_{period}"
        df[col_name] = df[column].rolling(window=period).mean()
        self._feature_names.append(col_name)
    return df

def ema(self, df: pd.DataFrame, periods: List[int], column: str = "close") -> pd.DataFrame:
    """Exponential Moving Average."""
    for period in periods:
        col_name = f"ema_{period}"
        df[col_name] = df[column].ewm(span=period, adjust=False).mean()
        self._feature_names.append(col_name)
    return df
```

**Explanation**: Loop over periods. Column naming convention: `indicator_period`. Append each column name to tracking list.

**Example 4: Multi-Output Indicator (ADX with +DI/-DI)**

```python
# From: src/features/technical/trend.py:91-124
def adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed values
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()

    df[f"adx_{period}"] = adx.values
    df[f"plus_di_{period}"] = plus_di.values
    df[f"minus_di_{period}"] = minus_di.values

    self._feature_names.extend([f"adx_{period}", f"plus_di_{period}", f"minus_di_{period}"])
    return df
```

**Explanation**: Multi-output indicators add multiple columns. Use `extend()` for multiple names. Add `1e-10` to prevent division by zero.

**Example 5: RSI (Momentum Category)**

```python
# From: src/features/technical/momentum.py:37-54
def rsi(self, df: pd.DataFrame, periods: List[int], column: str = "close") -> pd.DataFrame:
    """Relative Strength Index."""
    for period in periods:
        delta = df[column].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        col_name = f"rsi_{period}"
        df[col_name] = rsi
        self._feature_names.append(col_name)

    return df
```

**Explanation**: Standard RSI calculation. Use Wilder's smoothing (ewm). Protect division with `1e-10`.

**Example 6: MACD (Multiple Related Outputs)**

```python
# From: src/features/technical/momentum.py:72-93
def macd(
    self,
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close",
) -> pd.DataFrame:
    """Moving Average Convergence Divergence."""
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram

    self._feature_names.extend(["macd", "macd_signal", "macd_hist"])
    return df
```

**Explanation**: MACD has fixed output names (not period-suffixed). Three related columns. Use `extend()` for multiple names.

**Example 7: Composite Indicator Class**

```python
# From: src/features/technical/indicators.py:1-58
"""Main technical indicators class combining all indicator types."""

from typing import List, Optional
import pandas as pd

from .trend import TrendIndicators
from .momentum import MomentumIndicators
from .volatility import VolatilityIndicators
from .volume import VolumeIndicators


class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator.

    Combines trend, momentum, volatility, and volume indicators.
    """

    def __init__(self):
        """Initialize technical indicators calculator."""
        self.trend = TrendIndicators()
        self.momentum = MomentumIndicators()
        self.volatility = VolatilityIndicators()
        self.volume = VolumeIndicators()

    def calculate_all(
        self,
        df: pd.DataFrame,
        include_groups: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Calculate all technical indicators.

        Args:
            df: OHLCV dataframe with columns [open, high, low, close, volume]
            include_groups: List of indicator groups to include
                           ('trend', 'momentum', 'volatility', 'volume')
                           If None, includes all groups.

        Returns:
            DataFrame with all calculated indicators
        """
        result = df.copy()
        include_groups = include_groups or ["trend", "momentum", "volatility", "volume"]

        if "trend" in include_groups:
            result = self.trend.calculate_all(result)

        if "momentum" in include_groups:
            result = self.momentum.calculate_all(result)

        if "volatility" in include_groups:
            result = self.volatility.calculate_all(result)

        if "volume" in include_groups:
            result = self.volume.calculate_all(result)

        return result
```

**Explanation**: Composite pattern combines all indicator types. Selective calculation with `include_groups`. Each sub-calculator maintains its own `_feature_names`.

## Quality Checklist

- [ ] Class has `__init__` with `self._feature_names: List[str] = []`
- [ ] `get_feature_names()` returns `self._feature_names.copy()`
- [ ] `calculate_all()` starts with `df = df.copy()` and `self._feature_names = []`
- [ ] Column names follow `indicator_period` pattern (e.g., `sma_20`)
- [ ] Each method appends column names to `_feature_names`
- [ ] Division operations protected with `+ 1e-10`
- [ ] All methods return `df` for chaining
- [ ] Docstrings describe what indicator measures

## Common Mistakes

- **Not tracking feature names**: Unknown features in model → Append to `_feature_names` for every column
- **Mutating input df**: Side effects in caller → Always `df.copy()` at start
- **Division by zero**: NaN values → Add `1e-10` to denominators
- **Inconsistent naming**: Hard to identify features → Use `indicator_period` convention
- **Forgetting to reset**: Duplicate features → Reset `_feature_names = []` in `calculate_all()`

## Validation

- [ ] Pattern confirmed in `src/features/technical/trend.py:1-224`
- [ ] Pattern confirmed in `src/features/technical/momentum.py:1-200`
- [ ] Composite pattern in `src/features/technical/indicators.py:1-89`

## Related Skills

- [configuring-indicator-yaml](./configuring-indicator-yaml.md) - For enabling/disabling indicators
- [creating-data-processors](../backend/creating-data-processors.md) - For preprocessing before indicators
- [implementing-prediction-models](../backend/implementing-prediction-models.md) - Consumes indicator features
