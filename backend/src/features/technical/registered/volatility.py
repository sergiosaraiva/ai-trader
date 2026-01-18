"""
Volatility Indicators - Auto-registered volatility-based indicators.

All indicators use the @indicator decorator for automatic registration.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from ..registry import indicator


@indicator(
    name="atr",
    category="volatility",
    description="Average True Range - measures market volatility",
    params={
        "period": {"type": int, "default": 14, "min": 2, "max": 100, "description": "Lookback period"},
        "normalized": {"type": bool, "default": False, "description": "Normalize by close price (NATR)"},
    },
    priority=0,  # P0 - Critical
)
def calculate_atr(
    df: pd.DataFrame, period: int = 14, normalized: bool = False, **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Average True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(span=period, adjust=False).mean()

    columns = []

    col_name = f"atr_{period}"
    df[col_name] = atr
    columns.append(col_name)

    if normalized:
        natr_col = f"natr_{period}"
        df[natr_col] = (atr / close) * 100
        columns.append(natr_col)

    return df, columns


@indicator(
    name="bollinger",
    category="volatility",
    description="Bollinger Bands - volatility bands around a moving average",
    params={
        "period": {"type": int, "default": 20, "min": 5, "max": 100, "description": "MA period"},
        "std_dev": {"type": float, "default": 2.0, "min": 0.5, "max": 4.0, "description": "Standard deviation multiplier"},
        "include_width": {"type": bool, "default": True, "description": "Include bandwidth"},
        "include_percent_b": {"type": bool, "default": True, "description": "Include %B"},
    },
    priority=0,  # P0 - Critical
)
def calculate_bollinger(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    include_width: bool = True,
    include_percent_b: bool = True,
    **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Bollinger Bands."""
    close = df["close"]

    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()

    upper = sma + std_dev * std
    lower = sma - std_dev * std

    columns = [f"bb_upper_{period}", f"bb_middle_{period}", f"bb_lower_{period}"]
    df[f"bb_upper_{period}"] = upper
    df[f"bb_middle_{period}"] = sma
    df[f"bb_lower_{period}"] = lower

    if include_width:
        width = (upper - lower) / (sma + 1e-10)
        df[f"bb_width_{period}"] = width
        columns.append(f"bb_width_{period}")

    if include_percent_b:
        percent_b = (close - lower) / (upper - lower + 1e-10)
        df[f"bb_pctb_{period}"] = percent_b
        columns.append(f"bb_pctb_{period}")

    return df, columns


@indicator(
    name="keltner",
    category="volatility",
    description="Keltner Channel - ATR-based volatility bands",
    params={
        "period": {"type": int, "default": 20, "min": 5, "max": 100, "description": "EMA period"},
        "atr_period": {"type": int, "default": 10, "min": 5, "max": 50, "description": "ATR period"},
        "multiplier": {"type": float, "default": 2.0, "min": 0.5, "max": 4.0, "description": "ATR multiplier"},
    },
    priority=1,  # P1
)
def calculate_keltner(
    df: pd.DataFrame,
    period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
    **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Keltner Channel."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # EMA of close
    ema = close.ewm(span=period, adjust=False).mean()

    # ATR
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False).mean()

    upper = ema + multiplier * atr
    lower = ema - multiplier * atr

    df[f"kc_upper_{period}"] = upper
    df[f"kc_middle_{period}"] = ema
    df[f"kc_lower_{period}"] = lower

    return df, [f"kc_upper_{period}", f"kc_middle_{period}", f"kc_lower_{period}"]


@indicator(
    name="donchian",
    category="volatility",
    description="Donchian Channel - highest high and lowest low over a period",
    params={
        "period": {"type": int, "default": 20, "min": 5, "max": 100, "description": "Lookback period"},
    },
    priority=1,  # P1
)
def calculate_donchian(df: pd.DataFrame, period: int = 20, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Donchian Channel."""
    upper = df["high"].rolling(window=period).max()
    lower = df["low"].rolling(window=period).min()
    middle = (upper + lower) / 2

    df[f"dc_upper_{period}"] = upper
    df[f"dc_middle_{period}"] = middle
    df[f"dc_lower_{period}"] = lower

    return df, [f"dc_upper_{period}", f"dc_middle_{period}", f"dc_lower_{period}"]


@indicator(
    name="historical_volatility",
    category="volatility",
    description="Historical Volatility - standard deviation of log returns, annualized",
    params={
        "period": {"type": int, "default": 20, "min": 5, "max": 100, "description": "Lookback period"},
        "annualization_factor": {"type": int, "default": 252, "description": "Trading days per year"},
    },
    priority=1,  # P1
)
def calculate_historical_volatility(
    df: pd.DataFrame,
    period: int = 20,
    annualization_factor: int = 252,
    **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Historical Volatility (annualized)."""
    log_returns = np.log(df["close"] / df["close"].shift(1))
    hv = log_returns.rolling(window=period).std() * np.sqrt(annualization_factor) * 100

    col_name = f"hv_{period}"
    df[col_name] = hv
    return df, [col_name]


@indicator(
    name="true_range",
    category="volatility",
    description="True Range - single period volatility measure",
    params={},
    priority=2,  # P2
)
def calculate_true_range(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate True Range."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df["true_range"] = tr
    return df, ["true_range"]


@indicator(
    name="chaikin_volatility",
    category="volatility",
    description="Chaikin Volatility - rate of change in trading range",
    params={
        "ema_period": {"type": int, "default": 10, "min": 5, "max": 50, "description": "EMA period"},
        "roc_period": {"type": int, "default": 10, "min": 5, "max": 50, "description": "ROC period"},
    },
    priority=2,  # P2
)
def calculate_chaikin_volatility(
    df: pd.DataFrame,
    ema_period: int = 10,
    roc_period: int = 10,
    **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Chaikin Volatility."""
    hl_range = df["high"] - df["low"]
    ema_range = hl_range.ewm(span=ema_period, adjust=False).mean()
    cv = ((ema_range - ema_range.shift(roc_period)) / (ema_range.shift(roc_period) + 1e-10)) * 100

    df["chaikin_vol"] = cv
    return df, ["chaikin_vol"]


@indicator(
    name="ulcer_index",
    category="volatility",
    description="Ulcer Index - measures downside volatility/risk",
    params={
        "period": {"type": int, "default": 14, "min": 5, "max": 50, "description": "Lookback period"},
    },
    priority=2,  # P2
)
def calculate_ulcer_index(df: pd.DataFrame, period: int = 14, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Ulcer Index."""
    close = df["close"]
    rolling_max = close.rolling(window=period).max()
    drawdown_pct = ((close - rolling_max) / rolling_max) * 100

    ulcer = np.sqrt((drawdown_pct ** 2).rolling(window=period).mean())

    col_name = f"ulcer_{period}"
    df[col_name] = ulcer
    return df, [col_name]


@indicator(
    name="volatility_ratio",
    category="volatility",
    description="Volatility Ratio - current volatility vs historical average",
    params={
        "short_period": {"type": int, "default": 5, "min": 2, "max": 20, "description": "Short period"},
        "long_period": {"type": int, "default": 20, "min": 10, "max": 100, "description": "Long period"},
    },
    priority=2,  # P2
)
def calculate_volatility_ratio(
    df: pd.DataFrame,
    short_period: int = 5,
    long_period: int = 20,
    **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Volatility Ratio (short ATR / long ATR)."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)

    short_atr = tr.rolling(window=short_period).mean()
    long_atr = tr.rolling(window=long_period).mean()

    vol_ratio = short_atr / (long_atr + 1e-10)

    df["vol_ratio"] = vol_ratio
    return df, ["vol_ratio"]
