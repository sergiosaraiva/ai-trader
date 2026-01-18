"""Market regime filter for trading decisions.

Based on backtesting results:
- 1H: Best in trending_down (54.55%), avoid ranging (46.29%)
- 4H: Best in ranging (56.00%) and trending_up (55.00%)

This filter should be applied BEFORE making trading decisions to ensure
we only trade in favorable market conditions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"


@dataclass
class RegimeAnalysis:
    """Results of regime analysis."""
    regime: MarketRegime
    adx: float
    trend_strength: float
    volatility_ratio: float
    should_trade: bool
    confidence_modifier: float  # Multiply position size by this
    reason: str


class RegimeFilter:
    """Filter trading signals based on market regime.

    Optimal regimes by timeframe (from backtesting):
    - 1H: trending_down (54.55% accuracy)
    - 4H: ranging (56.00%), trending_up (55.00%)
    """

    # Optimal regimes per timeframe based on backtesting
    # Updated based on actual trading backtest results
    OPTIMAL_REGIMES = {
        "15m": [MarketRegime.TRENDING_DOWN],
        "1H": [MarketRegime.TRENDING_DOWN],  # 44.44% win rate (best for 1H)
        "4H": [MarketRegime.RANGING, MarketRegime.TRENDING_UP],  # 52.38%, 52.94%
        "1D": [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN],
    }

    # Regimes to avoid per timeframe
    AVOID_REGIMES = {
        "1H": [MarketRegime.TRENDING_UP],  # 22.50% - very poor
        "4H": [MarketRegime.TRENDING_DOWN, MarketRegime.VOLATILE],  # 33.33% - poor
    }

    # ADX thresholds
    ADX_TRENDING_THRESHOLD = 25.0
    ADX_RANGING_THRESHOLD = 20.0

    # Volatility thresholds
    VOLATILITY_HIGH_THRESHOLD = 2.0  # ATR ratio

    def __init__(
        self,
        timeframe: str = "1H",
        ma_period: int = 20,
        adx_period: int = 14,
        atr_period: int = 14,
    ):
        """Initialize regime filter.

        Args:
            timeframe: Trading timeframe (15m, 1H, 4H, 1D)
            ma_period: Period for moving average
            adx_period: Period for ADX calculation
            atr_period: Period for ATR calculation
        """
        self.timeframe = timeframe
        self.ma_period = ma_period
        self.adx_period = adx_period
        self.atr_period = atr_period

        self.optimal_regimes = self.OPTIMAL_REGIMES.get(
            timeframe, [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]
        )
        self.avoid_regimes = self.AVOID_REGIMES.get(timeframe, [])

    def calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average Directional Index."""
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        n = len(df)
        period = self.adx_period

        # True Range
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )

        # Directional Movement
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

        # Smoothed averages using Wilder's smoothing
        atr = np.zeros(n)
        plus_dm_smooth = np.zeros(n)
        minus_dm_smooth = np.zeros(n)

        # Initial values (simple average)
        atr[period-1] = np.mean(tr[:period])
        plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
        minus_dm_smooth[period-1] = np.mean(minus_dm[:period])

        # Wilder's smoothing
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period

        # Directional Indicators
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        dx = np.zeros(n)

        for i in range(period-1, n):
            if atr[i] > 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
                di_sum = plus_di[i] + minus_di[i]
                if di_sum > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

        # ADX (smoothed DX)
        adx = np.zeros(n)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period

        return pd.Series(adx, index=df.index)

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.rolling(self.atr_period).mean()

    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime from recent price data.

        Args:
            df: DataFrame with OHLCV data (at least 50 bars)

        Returns:
            Current market regime
        """
        if len(df) < 50:
            return MarketRegime.UNKNOWN

        # Calculate indicators
        close = df["close"]
        ma = close.rolling(self.ma_period).mean()
        adx = self.calculate_adx(df)
        atr = self.calculate_atr(df)

        # Get latest values
        current_close = close.iloc[-1]
        current_ma = ma.iloc[-1]
        current_adx = adx.iloc[-1]
        current_atr = atr.iloc[-1]
        avg_atr = atr.iloc[-20:].mean()

        # Calculate momentum
        returns_5 = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6]

        # Volatility ratio
        volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0

        # Determine regime
        if volatility_ratio > self.VOLATILITY_HIGH_THRESHOLD:
            return MarketRegime.VOLATILE

        if current_adx > self.ADX_TRENDING_THRESHOLD:
            # Trending market
            if current_close > current_ma and returns_5 > 0:
                return MarketRegime.TRENDING_UP
            elif current_close < current_ma and returns_5 < 0:
                return MarketRegime.TRENDING_DOWN
            else:
                # Mixed signals, use price vs MA
                return MarketRegime.TRENDING_UP if current_close > current_ma else MarketRegime.TRENDING_DOWN

        if current_adx < self.ADX_RANGING_THRESHOLD:
            return MarketRegime.RANGING

        # ADX between 20-25, transitional
        return MarketRegime.RANGING

    def analyze(self, df: pd.DataFrame) -> RegimeAnalysis:
        """Analyze market regime and determine if we should trade.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            RegimeAnalysis with trading recommendation
        """
        regime = self.detect_regime(df)

        # Calculate metrics for reporting
        adx = self.calculate_adx(df).iloc[-1] if len(df) >= 50 else 0.0
        atr = self.calculate_atr(df)
        volatility_ratio = atr.iloc[-1] / atr.iloc[-20:].mean() if len(df) >= 50 else 1.0

        # Trend strength (normalized)
        close = df["close"]
        ma = close.rolling(self.ma_period).mean()
        trend_strength = (close.iloc[-1] - ma.iloc[-1]) / ma.iloc[-1] if len(df) >= self.ma_period else 0.0

        # Determine if we should trade
        should_trade = True
        confidence_modifier = 1.0
        reason = ""

        if regime == MarketRegime.UNKNOWN:
            should_trade = False
            confidence_modifier = 0.0
            reason = "Insufficient data to determine regime"
        elif regime in self.avoid_regimes:
            should_trade = False
            confidence_modifier = 0.0
            reason = f"Regime {regime.value} has poor historical accuracy for {self.timeframe}"
        elif regime in self.optimal_regimes:
            should_trade = True
            confidence_modifier = 1.0
            reason = f"Optimal regime {regime.value} for {self.timeframe}"
        else:
            # Neutral regime - trade with reduced size
            should_trade = True
            confidence_modifier = 0.5
            reason = f"Neutral regime {regime.value} - reduced position size"

        return RegimeAnalysis(
            regime=regime,
            adx=float(adx) if not np.isnan(adx) else 0.0,
            trend_strength=float(trend_strength) if not np.isnan(trend_strength) else 0.0,
            volatility_ratio=float(volatility_ratio) if not np.isnan(volatility_ratio) else 1.0,
            should_trade=should_trade,
            confidence_modifier=confidence_modifier,
            reason=reason,
        )

    def filter_signal(
        self,
        df: pd.DataFrame,
        signal_direction: int,  # 1 for buy, -1 for sell
        signal_confidence: float,
    ) -> tuple[bool, float, str]:
        """Filter a trading signal based on regime.

        Args:
            df: Recent OHLCV data
            signal_direction: 1 for buy, -1 for sell
            signal_confidence: Model's confidence in the signal

        Returns:
            Tuple of (should_trade, adjusted_confidence, reason)
        """
        analysis = self.analyze(df)

        if not analysis.should_trade:
            return False, 0.0, analysis.reason

        # Adjust confidence based on regime
        adjusted_confidence = signal_confidence * analysis.confidence_modifier

        # Additional check: in trending regime, only trade in trend direction
        if analysis.regime == MarketRegime.TRENDING_UP and signal_direction < 0:
            # Selling in uptrend - reduce confidence
            adjusted_confidence *= 0.7
            reason = f"{analysis.reason} (counter-trend signal, reduced confidence)"
        elif analysis.regime == MarketRegime.TRENDING_DOWN and signal_direction > 0:
            # Buying in downtrend - reduce confidence
            adjusted_confidence *= 0.7
            reason = f"{analysis.reason} (counter-trend signal, reduced confidence)"
        else:
            reason = analysis.reason

        return True, adjusted_confidence, reason


def create_regime_filter(timeframe: str = "1H") -> RegimeFilter:
    """Factory function to create a regime filter.

    Args:
        timeframe: Trading timeframe

    Returns:
        Configured RegimeFilter instance
    """
    return RegimeFilter(timeframe=timeframe)
