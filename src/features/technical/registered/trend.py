"""
Trend Indicators - Auto-registered trend-following indicators.

All indicators use the @indicator decorator for automatic registration.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd

from ..registry import indicator


@indicator(
    name="sma",
    category="trend",
    description="Simple Moving Average - arithmetic mean of price over a period",
    params={
        "period": {"type": int, "default": 20, "min": 2, "max": 500, "description": "Lookback period"},
        "column": {"type": str, "default": "close", "choices": ["open", "high", "low", "close"]},
    },
    priority=1,  # P1
)
def calculate_sma(
    df: pd.DataFrame, period: int = 20, column: str = "close", **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Simple Moving Average."""
    col_name = f"sma_{period}"
    df[col_name] = df[column].rolling(window=period).mean()
    return df, [col_name]


@indicator(
    name="ema",
    category="trend",
    description="Exponential Moving Average - weighted average giving more weight to recent prices",
    params={
        "period": {"type": int, "default": 20, "min": 2, "max": 500, "description": "Lookback period"},
        "column": {"type": str, "default": "close", "choices": ["open", "high", "low", "close"]},
    },
    priority=0,  # P0 - Critical
)
def calculate_ema(
    df: pd.DataFrame, period: int = 20, column: str = "close", **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Exponential Moving Average."""
    col_name = f"ema_{period}"
    df[col_name] = df[column].ewm(span=period, adjust=False).mean()
    return df, [col_name]


@indicator(
    name="wma",
    category="trend",
    description="Weighted Moving Average - linearly weighted average",
    params={
        "period": {"type": int, "default": 20, "min": 2, "max": 200, "description": "Lookback period"},
        "column": {"type": str, "default": "close"},
    },
    priority=2,  # P2
)
def calculate_wma(
    df: pd.DataFrame, period: int = 20, column: str = "close", **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Weighted Moving Average."""
    weights = np.arange(1, period + 1)

    col_name = f"wma_{period}"
    df[col_name] = df[column].rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )
    return df, [col_name]


@indicator(
    name="dema",
    category="trend",
    description="Double Exponential Moving Average - reduces lag of standard EMA",
    params={
        "period": {"type": int, "default": 20, "min": 2, "max": 200, "description": "Lookback period"},
        "column": {"type": str, "default": "close"},
    },
    priority=2,  # P2
)
def calculate_dema(
    df: pd.DataFrame, period: int = 20, column: str = "close", **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Double Exponential Moving Average."""
    ema1 = df[column].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    dema = 2 * ema1 - ema2

    col_name = f"dema_{period}"
    df[col_name] = dema
    return df, [col_name]


@indicator(
    name="tema",
    category="trend",
    description="Triple Exponential Moving Average - further reduces lag",
    params={
        "period": {"type": int, "default": 20, "min": 2, "max": 200, "description": "Lookback period"},
        "column": {"type": str, "default": "close"},
    },
    priority=2,  # P2
)
def calculate_tema(
    df: pd.DataFrame, period: int = 20, column: str = "close", **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Triple Exponential Moving Average."""
    ema1 = df[column].ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    tema = 3 * ema1 - 3 * ema2 + ema3

    col_name = f"tema_{period}"
    df[col_name] = tema
    return df, [col_name]


@indicator(
    name="adx",
    category="trend",
    description="Average Directional Index - measures trend strength",
    params={
        "period": {"type": int, "default": 14, "min": 5, "max": 50, "description": "Lookback period"},
        "include_di": {"type": bool, "default": True, "description": "Include +DI and -DI"},
    },
    priority=0,  # P0 - Critical
)
def calculate_adx(
    df: pd.DataFrame, period: int = 14, include_di: bool = True, **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate ADX and optionally +DI/-DI."""
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

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)

    # Smoothed values
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10)

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()

    columns = [f"adx_{period}"]
    df[f"adx_{period}"] = adx

    if include_di:
        df[f"plus_di_{period}"] = plus_di
        df[f"minus_di_{period}"] = minus_di
        columns.extend([f"plus_di_{period}", f"minus_di_{period}"])

    return df, columns


@indicator(
    name="aroon",
    category="trend",
    description="Aroon Indicator - identifies trend and trend strength",
    params={
        "period": {"type": int, "default": 25, "min": 5, "max": 50, "description": "Lookback period"},
    },
    priority=1,  # P1
)
def calculate_aroon(df: pd.DataFrame, period: int = 25, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Aroon Up, Down, and Oscillator."""
    aroon_up = 100 * df["high"].rolling(window=period + 1).apply(
        lambda x: (period - x.argmax()) / period, raw=True
    )
    aroon_down = 100 * df["low"].rolling(window=period + 1).apply(
        lambda x: (period - x.argmin()) / period, raw=True
    )
    aroon_osc = aroon_up - aroon_down

    df[f"aroon_up_{period}"] = aroon_up
    df[f"aroon_down_{period}"] = aroon_down
    df[f"aroon_osc_{period}"] = aroon_osc

    return df, [f"aroon_up_{period}", f"aroon_down_{period}", f"aroon_osc_{period}"]


@indicator(
    name="supertrend",
    category="trend",
    description="Supertrend - trend-following indicator based on ATR",
    params={
        "period": {"type": int, "default": 10, "min": 5, "max": 50, "description": "ATR period"},
        "multiplier": {"type": float, "default": 3.0, "min": 1.0, "max": 10.0, "description": "ATR multiplier"},
    },
    priority=1,  # P1
)
def calculate_supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0, **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Supertrend indicator."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Calculate ATR
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    # Calculate basic bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    # Calculate Supertrend
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = 1

    for i in range(1, len(df)):
        if close.iloc[i] > supertrend.iloc[i - 1]:
            supertrend.iloc[i] = lower_band.iloc[i]
            direction.iloc[i] = 1
        else:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = -1

        # Adjust based on previous supertrend
        if direction.iloc[i] == 1 and supertrend.iloc[i] < supertrend.iloc[i - 1]:
            supertrend.iloc[i] = supertrend.iloc[i - 1]
        if direction.iloc[i] == -1 and supertrend.iloc[i] > supertrend.iloc[i - 1]:
            supertrend.iloc[i] = supertrend.iloc[i - 1]

    df[f"supertrend_{period}"] = supertrend
    df[f"supertrend_dir_{period}"] = direction

    return df, [f"supertrend_{period}", f"supertrend_dir_{period}"]


@indicator(
    name="ichimoku",
    category="trend",
    description="Ichimoku Cloud - comprehensive indicator showing support/resistance and trend",
    params={
        "tenkan": {"type": int, "default": 9, "min": 5, "max": 20, "description": "Tenkan-sen period"},
        "kijun": {"type": int, "default": 26, "min": 10, "max": 50, "description": "Kijun-sen period"},
        "senkou_b": {"type": int, "default": 52, "min": 20, "max": 100, "description": "Senkou Span B period"},
    },
    priority=1,  # P1
)
def calculate_ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Ichimoku Cloud components."""
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Tenkan-sen (Conversion Line)
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2

    # Kijun-sen (Base Line)
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2

    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)

    # Senkou Span B (Leading Span B)
    senkou_span_b = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)

    # Chikou Span (Lagging Span)
    chikou_span = close.shift(-kijun)

    df["ichimoku_tenkan"] = tenkan_sen
    df["ichimoku_kijun"] = kijun_sen
    df["ichimoku_senkou_a"] = senkou_span_a
    df["ichimoku_senkou_b"] = senkou_span_b
    df["ichimoku_chikou"] = chikou_span

    return df, [
        "ichimoku_tenkan", "ichimoku_kijun",
        "ichimoku_senkou_a", "ichimoku_senkou_b", "ichimoku_chikou"
    ]


@indicator(
    name="vwma",
    category="trend",
    description="Volume Weighted Moving Average",
    params={
        "period": {"type": int, "default": 20, "min": 2, "max": 200, "description": "Lookback period"},
    },
    priority=2,  # P2
)
def calculate_vwma(df: pd.DataFrame, period: int = 20, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Volume Weighted Moving Average."""
    vwma = (df["close"] * df["volume"]).rolling(period).sum() / (df["volume"].rolling(period).sum() + 1e-10)

    col_name = f"vwma_{period}"
    df[col_name] = vwma
    return df, [col_name]


@indicator(
    name="parabolic_sar",
    category="trend",
    description="Parabolic SAR - trailing stop and reverse indicator",
    params={
        "af_start": {"type": float, "default": 0.02, "min": 0.01, "max": 0.1, "description": "Initial acceleration factor"},
        "af_increment": {"type": float, "default": 0.02, "min": 0.01, "max": 0.1, "description": "AF increment"},
        "af_max": {"type": float, "default": 0.2, "min": 0.1, "max": 0.5, "description": "Maximum AF"},
    },
    priority=1,  # P1
)
def calculate_parabolic_sar(
    df: pd.DataFrame,
    af_start: float = 0.02,
    af_increment: float = 0.02,
    af_max: float = 0.2,
    **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Parabolic SAR."""
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)

    psar = np.zeros(n)
    af = np.zeros(n)
    trend = np.zeros(n)  # 1 = uptrend, -1 = downtrend
    ep = np.zeros(n)  # Extreme point

    # Initialize
    trend[0] = 1 if close[0] > close[min(1, n-1)] else -1
    psar[0] = low[0] if trend[0] == 1 else high[0]
    ep[0] = high[0] if trend[0] == 1 else low[0]
    af[0] = af_start

    for i in range(1, n):
        # Calculate SAR
        psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])

        # Adjust SAR based on trend
        if trend[i-1] == 1:  # Uptrend
            psar[i] = min(psar[i], low[i-1], low[max(0, i-2)])
            if low[i] < psar[i]:  # Reversal
                trend[i] = -1
                psar[i] = ep[i-1]
                ep[i] = low[i]
                af[i] = af_start
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # Downtrend
            psar[i] = max(psar[i], high[i-1], high[max(0, i-2)])
            if high[i] > psar[i]:  # Reversal
                trend[i] = 1
                psar[i] = ep[i-1]
                ep[i] = high[i]
                af[i] = af_start
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_increment, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]

    df["psar"] = psar
    df["psar_trend"] = trend

    return df, ["psar", "psar_trend"]


@indicator(
    name="ma_crossover",
    category="trend",
    description="Moving Average Crossover signals",
    params={
        "fast": {"type": int, "default": 10, "min": 2, "max": 100, "description": "Fast MA period"},
        "slow": {"type": int, "default": 30, "min": 5, "max": 200, "description": "Slow MA period"},
        "ma_type": {"type": str, "default": "ema", "choices": ["sma", "ema"], "description": "MA type"},
    },
    priority=1,  # P1
)
def calculate_ma_crossover(
    df: pd.DataFrame,
    fast: int = 10,
    slow: int = 30,
    ma_type: str = "ema",
    **kwargs
) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate Moving Average Crossover."""
    if ma_type == "ema":
        fast_ma = df["close"].ewm(span=fast, adjust=False).mean()
        slow_ma = df["close"].ewm(span=slow, adjust=False).mean()
    else:
        fast_ma = df["close"].rolling(fast).mean()
        slow_ma = df["close"].rolling(slow).mean()

    # Crossover signal: 1 = bullish cross, -1 = bearish cross, 0 = no cross
    ma_diff = fast_ma - slow_ma
    ma_diff_prev = ma_diff.shift(1)

    crossover = pd.Series(0, index=df.index)
    crossover[(ma_diff > 0) & (ma_diff_prev <= 0)] = 1  # Bullish
    crossover[(ma_diff < 0) & (ma_diff_prev >= 0)] = -1  # Bearish

    col_name = f"ma_cross_{fast}_{slow}"
    df[col_name] = crossover
    return df, [col_name]
