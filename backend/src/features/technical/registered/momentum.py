"""
Momentum Indicators - Auto-registered momentum-based indicators.

All indicators use the @indicator decorator for automatic registration.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from ..registry import indicator


@indicator(
    name="rsi",
    category="momentum",
    description="Relative Strength Index - momentum oscillator measuring speed and magnitude of price changes",
    params={
        "period": {"type": int, "default": 14, "min": 2, "max": 100, "description": "Lookback period"},
        "column": {"type": str, "default": "close", "choices": ["open", "high", "low", "close"]},
    },
    priority=0,  # P0 - Critical indicator
)
def calculate_rsi(df: pd.DataFrame, period: int = 14, column: str = "close", **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Relative Strength Index."""
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    col_name = f"rsi_{period}"
    df[col_name] = rsi
    return df, [col_name]


@indicator(
    name="macd",
    category="momentum",
    description="Moving Average Convergence Divergence - trend-following momentum indicator",
    params={
        "fast": {"type": int, "default": 12, "min": 2, "max": 50, "description": "Fast EMA period"},
        "slow": {"type": int, "default": 26, "min": 5, "max": 100, "description": "Slow EMA period"},
        "signal": {"type": int, "default": 9, "min": 2, "max": 50, "description": "Signal line period"},
        "column": {"type": str, "default": "close"},
    },
    priority=0,  # P0 - Critical indicator
)
def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = "close",
    **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate MACD, Signal, and Histogram."""
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = histogram

    return df, ["macd", "macd_signal", "macd_hist"]


@indicator(
    name="stochastic",
    category="momentum",
    description="Stochastic Oscillator - compares closing price to price range over a period",
    params={
        "k_period": {"type": int, "default": 14, "min": 3, "max": 50, "description": "%K period"},
        "d_period": {"type": int, "default": 3, "min": 1, "max": 20, "description": "%D smoothing period"},
        "smooth_k": {"type": int, "default": 3, "min": 1, "max": 10, "description": "%K smoothing"},
    },
    priority=1,  # P1 - Important
)
def calculate_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
    **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Stochastic Oscillator %K and %D."""
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()

    stoch_k = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-10)

    # Apply smoothing to %K
    if smooth_k > 1:
        stoch_k = stoch_k.rolling(window=smooth_k).mean()

    stoch_d = stoch_k.rolling(window=d_period).mean()

    df[f"stoch_k_{k_period}"] = stoch_k
    df[f"stoch_d_{k_period}"] = stoch_d

    return df, [f"stoch_k_{k_period}", f"stoch_d_{k_period}"]


@indicator(
    name="cci",
    category="momentum",
    description="Commodity Channel Index - measures current price level relative to average price",
    params={
        "period": {"type": int, "default": 20, "min": 5, "max": 100, "description": "Lookback period"},
    },
    priority=1,  # P1
)
def calculate_cci(df: pd.DataFrame, period: int = 20, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Commodity Channel Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3

    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cci = (tp - sma) / (0.015 * mad + 1e-10)

    col_name = f"cci_{period}"
    df[col_name] = cci
    return df, [col_name]


@indicator(
    name="momentum",
    category="momentum",
    description="Momentum - rate of change in price",
    params={
        "period": {"type": int, "default": 10, "min": 1, "max": 100, "description": "Lookback period"},
        "column": {"type": str, "default": "close"},
    },
    priority=1,  # P1
)
def calculate_momentum(
    df: pd.DataFrame, period: int = 10, column: str = "close", **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Momentum indicator."""
    col_name = f"mom_{period}"
    df[col_name] = df[column] - df[column].shift(period)
    return df, [col_name]


@indicator(
    name="roc",
    category="momentum",
    description="Rate of Change - percentage change in price over a period",
    params={
        "period": {"type": int, "default": 10, "min": 1, "max": 100, "description": "Lookback period"},
        "column": {"type": str, "default": "close"},
    },
    priority=1,  # P1
)
def calculate_roc(
    df: pd.DataFrame, period: int = 10, column: str = "close", **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Rate of Change."""
    col_name = f"roc_{period}"
    df[col_name] = (df[column] - df[column].shift(period)) / (df[column].shift(period) + 1e-10) * 100
    return df, [col_name]


@indicator(
    name="williams_r",
    category="momentum",
    description="Williams %R - momentum indicator showing overbought/oversold levels",
    params={
        "period": {"type": int, "default": 14, "min": 5, "max": 50, "description": "Lookback period"},
    },
    priority=1,  # P1
)
def calculate_williams_r(df: pd.DataFrame, period: int = 14, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Williams %R."""
    high_max = df["high"].rolling(window=period).max()
    low_min = df["low"].rolling(window=period).min()

    willr = -100 * (high_max - df["close"]) / (high_max - low_min + 1e-10)

    col_name = f"willr_{period}"
    df[col_name] = willr
    return df, [col_name]


@indicator(
    name="mfi",
    category="momentum",
    description="Money Flow Index - volume-weighted RSI",
    params={
        "period": {"type": int, "default": 14, "min": 5, "max": 50, "description": "Lookback period"},
    },
    priority=1,  # P1
)
def calculate_mfi(df: pd.DataFrame, period: int = 14, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Money Flow Index."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mf = tp * df["volume"]

    pos_mf = mf.where(tp > tp.shift(1), 0)
    neg_mf = mf.where(tp < tp.shift(1), 0)

    pos_mf_sum = pos_mf.rolling(window=period).sum()
    neg_mf_sum = neg_mf.rolling(window=period).sum()

    mfi = 100 - (100 / (1 + pos_mf_sum / (neg_mf_sum + 1e-10)))

    col_name = f"mfi_{period}"
    df[col_name] = mfi
    return df, [col_name]


@indicator(
    name="tsi",
    category="momentum",
    description="True Strength Index - double-smoothed momentum indicator",
    params={
        "long_period": {"type": int, "default": 25, "min": 10, "max": 50, "description": "Long smoothing period"},
        "short_period": {"type": int, "default": 13, "min": 5, "max": 25, "description": "Short smoothing period"},
        "column": {"type": str, "default": "close"},
    },
    priority=2,  # P2
)
def calculate_tsi(
    df: pd.DataFrame,
    long_period: int = 25,
    short_period: int = 13,
    column: str = "close",
    **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate True Strength Index."""
    diff = df[column].diff()

    double_smoothed = diff.ewm(span=long_period, adjust=False).mean().ewm(span=short_period, adjust=False).mean()
    double_smoothed_abs = (
        diff.abs().ewm(span=long_period, adjust=False).mean().ewm(span=short_period, adjust=False).mean()
    )

    tsi = 100 * double_smoothed / (double_smoothed_abs + 1e-10)

    df["tsi"] = tsi
    return df, ["tsi"]


@indicator(
    name="ultimate_oscillator",
    category="momentum",
    description="Ultimate Oscillator - multi-timeframe momentum oscillator",
    params={
        "period1": {"type": int, "default": 7, "min": 3, "max": 20, "description": "Short period"},
        "period2": {"type": int, "default": 14, "min": 7, "max": 30, "description": "Medium period"},
        "period3": {"type": int, "default": 28, "min": 14, "max": 50, "description": "Long period"},
    },
    priority=2,  # P2
)
def calculate_ultimate_oscillator(
    df: pd.DataFrame,
    period1: int = 7,
    period2: int = 14,
    period3: int = 28,
    **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Ultimate Oscillator."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = pd.concat([high, prev_close], axis=1).max(axis=1) - pd.concat([low, prev_close], axis=1).min(axis=1)

    avg1 = bp.rolling(period1).sum() / (tr.rolling(period1).sum() + 1e-10)
    avg2 = bp.rolling(period2).sum() / (tr.rolling(period2).sum() + 1e-10)
    avg3 = bp.rolling(period3).sum() / (tr.rolling(period3).sum() + 1e-10)

    uo = 100 * (4 * avg1 + 2 * avg2 + avg3) / 7

    df["uo"] = uo
    return df, ["uo"]


@indicator(
    name="awesome_oscillator",
    category="momentum",
    description="Awesome Oscillator - difference between fast and slow SMAs of midpoint",
    params={
        "fast": {"type": int, "default": 5, "min": 2, "max": 20, "description": "Fast SMA period"},
        "slow": {"type": int, "default": 34, "min": 10, "max": 50, "description": "Slow SMA period"},
    },
    priority=2,  # P2
)
def calculate_awesome_oscillator(
    df: pd.DataFrame, fast: int = 5, slow: int = 34, **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Awesome Oscillator."""
    midpoint = (df["high"] + df["low"]) / 2
    ao = midpoint.rolling(window=fast).mean() - midpoint.rolling(window=slow).mean()

    df["ao"] = ao
    return df, ["ao"]
