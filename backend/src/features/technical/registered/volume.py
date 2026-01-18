"""
Volume Indicators - Auto-registered volume-based indicators.

All indicators use the @indicator decorator for automatic registration.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from ..registry import indicator


@indicator(
    name="obv",
    category="volume",
    description="On Balance Volume - cumulative volume based on price direction",
    params={},
    priority=0,  # P0 - Critical
)
def calculate_obv(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate On Balance Volume."""
    close = df["close"]
    volume = df["volume"]

    obv = pd.Series(index=df.index, dtype=float)
    obv.iloc[0] = 0

    for i in range(1, len(df)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]

    df["obv"] = obv
    return df, ["obv"]


@indicator(
    name="vwap",
    category="volume",
    description="Volume Weighted Average Price - average price weighted by volume",
    params={
        "period": {"type": int, "default": 0, "description": "Rolling period (0 = cumulative for the day)"},
    },
    priority=1,  # P1
)
def calculate_vwap(df: pd.DataFrame, period: int = 0, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate VWAP."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_volume = typical_price * df["volume"]

    if period > 0:
        vwap = tp_volume.rolling(period).sum() / (df["volume"].rolling(period).sum() + 1e-10)
        col_name = f"vwap_{period}"
    else:
        vwap = tp_volume.cumsum() / (df["volume"].cumsum() + 1e-10)
        col_name = "vwap"

    df[col_name] = vwap
    return df, [col_name]


@indicator(
    name="cmf",
    category="volume",
    description="Chaikin Money Flow - measures buying/selling pressure",
    params={
        "period": {"type": int, "default": 20, "min": 5, "max": 50, "description": "Lookback period"},
    },
    priority=1,  # P1
)
def calculate_cmf(df: pd.DataFrame, period: int = 20, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Chaikin Money Flow."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low + 1e-10)

    # Money Flow Volume
    mfv = mfm * volume

    # CMF
    cmf = mfv.rolling(window=period).sum() / (volume.rolling(window=period).sum() + 1e-10)

    col_name = f"cmf_{period}"
    df[col_name] = cmf
    return df, [col_name]


@indicator(
    name="ad_line",
    category="volume",
    description="Accumulation/Distribution Line - cumulative indicator based on close location",
    params={},
    priority=1,  # P1
)
def calculate_ad_line(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Accumulation/Distribution Line."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]

    # Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low + 1e-10)

    # A/D Line
    ad = (mfm * volume).cumsum()

    df["ad_line"] = ad
    return df, ["ad_line"]


@indicator(
    name="force_index",
    category="volume",
    description="Force Index - combines price and volume to measure buying/selling force",
    params={
        "period": {"type": int, "default": 13, "min": 1, "max": 50, "description": "EMA period (1 for raw)"},
    },
    priority=1,  # P1
)
def calculate_force_index(df: pd.DataFrame, period: int = 13, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Force Index."""
    close = df["close"]
    volume = df["volume"]

    # Raw force
    force = close.diff() * volume

    # Smoothed
    if period > 1:
        force = force.ewm(span=period, adjust=False).mean()

    col_name = f"force_{period}"
    df[col_name] = force
    return df, [col_name]


@indicator(
    name="mfi",
    category="volume",
    description="Money Flow Index - volume-weighted RSI",
    params={
        "period": {"type": int, "default": 14, "min": 5, "max": 50, "description": "Lookback period"},
    },
    priority=1,  # P1
)
def calculate_mfi_volume(df: pd.DataFrame, period: int = 14, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Money Flow Index (volume category version)."""
    # Note: This is also in momentum.py - having it in volume for organizational purposes
    # The registry will use the last registered version
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
    name="vpt",
    category="volume",
    description="Volume Price Trend - cumulative volume adjusted by price change",
    params={},
    priority=2,  # P2
)
def calculate_vpt(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Volume Price Trend."""
    close = df["close"]
    volume = df["volume"]

    pct_change = close.pct_change()
    vpt = (pct_change * volume).cumsum()

    df["vpt"] = vpt
    return df, ["vpt"]


@indicator(
    name="eom",
    category="volume",
    description="Ease of Movement - relates price change to volume",
    params={
        "period": {"type": int, "default": 14, "min": 5, "max": 50, "description": "EMA smoothing period"},
    },
    priority=2,  # P2
)
def calculate_eom(df: pd.DataFrame, period: int = 14, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Ease of Movement."""
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Distance moved
    dm = ((high + low) / 2) - ((high.shift(1) + low.shift(1)) / 2)

    # Box ratio
    box_ratio = (volume / 1e6) / (high - low + 1e-10)

    # EMV
    emv = dm / (box_ratio + 1e-10)

    # Smoothed
    emv_smoothed = emv.ewm(span=period, adjust=False).mean()

    col_name = f"eom_{period}"
    df[col_name] = emv_smoothed
    return df, [col_name]


@indicator(
    name="nvi",
    category="volume",
    description="Negative Volume Index - tracks price on down-volume days",
    params={},
    priority=2,  # P2
)
def calculate_nvi(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Negative Volume Index."""
    close = df["close"]
    volume = df["volume"]

    nvi = pd.Series(index=df.index, dtype=float)
    nvi.iloc[0] = 1000  # Starting value

    for i in range(1, len(df)):
        if volume.iloc[i] < volume.iloc[i - 1]:
            pct_change = (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1]
            nvi.iloc[i] = nvi.iloc[i - 1] * (1 + pct_change)
        else:
            nvi.iloc[i] = nvi.iloc[i - 1]

    df["nvi"] = nvi
    return df, ["nvi"]


@indicator(
    name="pvi",
    category="volume",
    description="Positive Volume Index - tracks price on up-volume days",
    params={},
    priority=2,  # P2
)
def calculate_pvi(df: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Positive Volume Index."""
    close = df["close"]
    volume = df["volume"]

    pvi = pd.Series(index=df.index, dtype=float)
    pvi.iloc[0] = 1000  # Starting value

    for i in range(1, len(df)):
        if volume.iloc[i] > volume.iloc[i - 1]:
            pct_change = (close.iloc[i] - close.iloc[i - 1]) / close.iloc[i - 1]
            pvi.iloc[i] = pvi.iloc[i - 1] * (1 + pct_change)
        else:
            pvi.iloc[i] = pvi.iloc[i - 1]

    df["pvi"] = pvi
    return df, ["pvi"]


@indicator(
    name="volume_oscillator",
    category="volume",
    description="Volume Oscillator - difference between two volume EMAs",
    params={
        "fast": {"type": int, "default": 5, "min": 2, "max": 20, "description": "Fast EMA period"},
        "slow": {"type": int, "default": 20, "min": 10, "max": 50, "description": "Slow EMA period"},
    },
    priority=2,  # P2
)
def calculate_volume_oscillator(
    df: pd.DataFrame,
    fast: int = 5,
    slow: int = 20,
    **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Volume Oscillator."""
    volume = df["volume"]

    fast_ema = volume.ewm(span=fast, adjust=False).mean()
    slow_ema = volume.ewm(span=slow, adjust=False).mean()

    vol_osc = ((fast_ema - slow_ema) / slow_ema) * 100

    df["vol_osc"] = vol_osc
    return df, ["vol_osc"]


@indicator(
    name="relative_volume",
    category="volume",
    description="Relative Volume - current volume vs average volume",
    params={
        "period": {"type": int, "default": 20, "min": 5, "max": 100, "description": "Average period"},
    },
    priority=1,  # P1
)
def calculate_relative_volume(df: pd.DataFrame, period: int = 20, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Relative Volume (RVOL)."""
    volume = df["volume"]
    avg_volume = volume.rolling(window=period).mean()

    rvol = volume / (avg_volume + 1e-10)

    col_name = f"rvol_{period}"
    df[col_name] = rvol
    return df, [col_name]
