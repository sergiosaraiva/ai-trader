"""Market regime detection for adaptive trading.

This module classifies market conditions into distinct regimes:
- Trend regimes: STRONG_TREND, WEAK_TREND, RANGING
- Volatility regimes: HIGH, NORMAL, LOW
- Combined market regimes for trading decisions

Usage:
    detector = RegimeDetector()
    regimes = detector.detect_regime(df)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrendRegime(Enum):
    """Trend regime classification."""
    STRONG_UPTREND = "strong_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    WEAK_UPTREND = "weak_uptrend"
    WEAK_DOWNTREND = "weak_downtrend"
    RANGING = "ranging"


class VolatilityRegime(Enum):
    """Volatility regime classification."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class MarketRegime(Enum):
    """Combined market regime for trading decisions."""
    TRENDING_HIGH_VOL = "trending_high_vol"      # Strong trends with high volatility
    TRENDING_NORMAL = "trending_normal"          # Clear trends with normal volatility
    TRENDING_LOW_VOL = "trending_low_vol"        # Trends but low volatility (squeeze)
    RANGING_HIGH_VOL = "ranging_high_vol"        # Choppy, dangerous
    RANGING_NORMAL = "ranging_normal"            # Standard ranging market
    RANGING_LOW_VOL = "ranging_low_vol"          # Quiet, low opportunity


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    # ADX thresholds for trend detection
    adx_strong_trend: float = 25.0
    adx_weak_trend: float = 15.0

    # ATR percentile thresholds for volatility
    atr_high_pctl: float = 75.0
    atr_low_pctl: float = 25.0
    atr_lookback: int = 100  # Bars for percentile calculation

    # VIX thresholds (if available)
    vix_high: float = 25.0
    vix_low: float = 15.0

    # Trend direction thresholds
    ema_slope_threshold: float = 0.0001  # Min slope for trend direction
    price_vs_ma_threshold: float = 0.002  # 0.2% above/below MA for trend confirmation

    # Smoothing
    regime_smoothing: int = 3  # Bars to confirm regime change


@dataclass
class RegimeState:
    """Current regime state with metadata."""
    trend_regime: TrendRegime
    volatility_regime: VolatilityRegime
    market_regime: MarketRegime

    # Underlying metrics
    adx: float = 0.0
    atr_percentile: float = 50.0
    ema_slope: float = 0.0
    price_vs_ema: float = 0.0
    vix: Optional[float] = None

    # Confidence in regime classification
    trend_confidence: float = 0.5
    volatility_confidence: float = 0.5

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "trend_regime": self.trend_regime.value,
            "volatility_regime": self.volatility_regime.value,
            "market_regime": self.market_regime.value,
            "adx": self.adx,
            "atr_percentile": self.atr_percentile,
            "ema_slope": self.ema_slope,
            "price_vs_ema": self.price_vs_ema,
            "vix": self.vix,
            "trend_confidence": self.trend_confidence,
            "volatility_confidence": self.volatility_confidence,
        }


class RegimeDetector:
    """Detects market regimes from price data."""

    def __init__(self, config: Optional[RegimeConfig] = None):
        """Initialize regime detector.

        Args:
            config: Regime detection configuration
        """
        self.config = config or RegimeConfig()
        self._atr_history = []

    def detect_regime(
        self,
        df: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Detect regime for each bar in the dataframe.

        Args:
            df: OHLCV dataframe with technical indicators (needs ADX, ATR, EMA)
            vix_series: Optional VIX series aligned with df index

        Returns:
            DataFrame with regime columns added
        """
        result = df.copy()

        # Ensure required indicators exist
        result = self._ensure_indicators(result)

        # Detect trend regime
        result = self._detect_trend_regime(result)

        # Detect volatility regime
        result = self._detect_volatility_regime(result, vix_series)

        # Combine into market regime
        result = self._combine_regimes(result)

        # Add regime features for model input
        result = self._add_regime_features(result)

        return result

    def get_current_regime(self, df: pd.DataFrame, vix: Optional[float] = None) -> RegimeState:
        """Get the current regime state from the latest bar.

        Args:
            df: OHLCV dataframe with technical indicators
            vix: Current VIX value (optional)

        Returns:
            RegimeState with current classification
        """
        # Detect regimes for all data
        df_with_regime = self.detect_regime(df, pd.Series([vix], index=[df.index[-1]]) if vix else None)

        # Get last row
        last = df_with_regime.iloc[-1]

        return RegimeState(
            trend_regime=TrendRegime(last["trend_regime"]),
            volatility_regime=VolatilityRegime(last["volatility_regime"]),
            market_regime=MarketRegime(last["market_regime"]),
            adx=last.get("adx_14", 0),
            atr_percentile=last.get("atr_percentile", 50),
            ema_slope=last.get("ema_slope", 0),
            price_vs_ema=last.get("price_vs_ema", 0),
            vix=vix,
            trend_confidence=last.get("trend_confidence", 0.5),
            volatility_confidence=last.get("volatility_confidence", 0.5),
        )

    def _ensure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure required indicators exist, calculate if missing."""
        result = df.copy()

        # ADX - Average Directional Index
        if "adx_14" not in result.columns and "adx" not in result.columns:
            result = self._calculate_adx(result)
        elif "adx" in result.columns and "adx_14" not in result.columns:
            result["adx_14"] = result["adx"]

        # ATR - Average True Range
        if "atr_14" not in result.columns and "atr" not in result.columns:
            result = self._calculate_atr(result)
        elif "atr" in result.columns and "atr_14" not in result.columns:
            result["atr_14"] = result["atr"]

        # EMA for trend direction
        if "ema_21" not in result.columns:
            result["ema_21"] = result["close"].ewm(span=21, adjust=False).mean()

        if "ema_55" not in result.columns:
            result["ema_55"] = result["close"].ewm(span=55, adjust=False).mean()

        return result

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX indicator."""
        result = df.copy()

        # True Range
        high = result["high"]
        low = result["low"]
        close = result["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()

        result["adx_14"] = adx
        result["plus_di"] = plus_di
        result["minus_di"] = minus_di

        return result

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ATR indicator."""
        result = df.copy()

        high = result["high"]
        low = result["low"]
        close = result["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        result["atr_14"] = tr.ewm(span=period, adjust=False).mean()

        return result

    def _detect_trend_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect trend regime from ADX and price action."""
        result = df.copy()

        adx = result["adx_14"]
        ema_21 = result["ema_21"]
        ema_55 = result["ema_55"]
        close = result["close"]

        # Calculate EMA slope (rate of change)
        ema_slope = ema_21.diff(5) / ema_21.shift(5)
        result["ema_slope"] = ema_slope

        # Price relative to EMA
        price_vs_ema = (close - ema_21) / ema_21
        result["price_vs_ema"] = price_vs_ema

        # Trend alignment (EMA 21 vs EMA 55)
        trend_alignment = (ema_21 - ema_55) / ema_55
        result["trend_alignment"] = trend_alignment

        # Classify trend regime
        trend_regime = []
        trend_confidence = []

        for i in range(len(result)):
            adx_val = adx.iloc[i]
            slope = ema_slope.iloc[i] if pd.notna(ema_slope.iloc[i]) else 0
            price_rel = price_vs_ema.iloc[i] if pd.notna(price_vs_ema.iloc[i]) else 0
            align = trend_alignment.iloc[i] if pd.notna(trend_alignment.iloc[i]) else 0

            # Determine trend strength
            if adx_val >= self.config.adx_strong_trend:
                strength = "strong"
                conf = min(1.0, adx_val / 40)  # Confidence scales with ADX
            elif adx_val >= self.config.adx_weak_trend:
                strength = "weak"
                conf = 0.5 + (adx_val - self.config.adx_weak_trend) / 20
            else:
                strength = "ranging"
                conf = 1.0 - adx_val / self.config.adx_weak_trend

            # Determine trend direction
            if strength == "ranging":
                regime = TrendRegime.RANGING
            else:
                # Use multiple confirmations for direction
                bullish_signals = 0
                if slope > self.config.ema_slope_threshold:
                    bullish_signals += 1
                if price_rel > self.config.price_vs_ma_threshold:
                    bullish_signals += 1
                if align > 0:
                    bullish_signals += 1

                if bullish_signals >= 2:
                    regime = TrendRegime.STRONG_UPTREND if strength == "strong" else TrendRegime.WEAK_UPTREND
                elif bullish_signals <= 1:
                    bearish_signals = 0
                    if slope < -self.config.ema_slope_threshold:
                        bearish_signals += 1
                    if price_rel < -self.config.price_vs_ma_threshold:
                        bearish_signals += 1
                    if align < 0:
                        bearish_signals += 1

                    if bearish_signals >= 2:
                        regime = TrendRegime.STRONG_DOWNTREND if strength == "strong" else TrendRegime.WEAK_DOWNTREND
                    else:
                        regime = TrendRegime.RANGING
                        conf = 0.5  # Unclear direction
                else:
                    regime = TrendRegime.RANGING
                    conf = 0.5

            trend_regime.append(regime.value)
            trend_confidence.append(conf)

        result["trend_regime"] = trend_regime
        result["trend_confidence"] = trend_confidence

        # Smooth regime changes
        result["trend_regime"] = self._smooth_regime(result["trend_regime"])

        return result

    def _detect_volatility_regime(
        self,
        df: pd.DataFrame,
        vix_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Detect volatility regime from ATR and optionally VIX."""
        result = df.copy()

        atr = result["atr_14"]

        # Calculate ATR percentile (rolling)
        lookback = self.config.atr_lookback
        atr_pctl = atr.rolling(lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
            raw=False
        )
        result["atr_percentile"] = atr_pctl.fillna(50)

        # Classify volatility regime
        vol_regime = []
        vol_confidence = []

        for i in range(len(result)):
            atr_pct = result["atr_percentile"].iloc[i]

            # Get VIX if available
            vix = None
            if vix_series is not None and result.index[i] in vix_series.index:
                vix = vix_series.loc[result.index[i]]

            # Determine volatility level
            if atr_pct >= self.config.atr_high_pctl:
                regime = VolatilityRegime.HIGH
                conf = min(1.0, (atr_pct - self.config.atr_high_pctl) / 25 + 0.5)
            elif atr_pct <= self.config.atr_low_pctl:
                regime = VolatilityRegime.LOW
                conf = min(1.0, (self.config.atr_low_pctl - atr_pct) / 25 + 0.5)
            else:
                regime = VolatilityRegime.NORMAL
                # Distance from extremes determines confidence
                dist_from_mid = abs(atr_pct - 50)
                conf = 1.0 - dist_from_mid / 25

            # Adjust with VIX if available
            if vix is not None and not np.isnan(vix):
                if vix >= self.config.vix_high and regime != VolatilityRegime.HIGH:
                    regime = VolatilityRegime.HIGH
                    conf = 0.7
                elif vix <= self.config.vix_low and regime != VolatilityRegime.LOW:
                    regime = VolatilityRegime.LOW
                    conf = 0.7

            vol_regime.append(regime.value)
            vol_confidence.append(conf)

        result["volatility_regime"] = vol_regime
        result["volatility_confidence"] = vol_confidence

        # Smooth regime changes
        result["volatility_regime"] = self._smooth_regime(result["volatility_regime"])

        return result

    def _combine_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine trend and volatility regimes into market regime."""
        result = df.copy()

        market_regime = []

        for i in range(len(result)):
            trend = result["trend_regime"].iloc[i]
            vol = result["volatility_regime"].iloc[i]

            # Is it trending?
            is_trending = trend in [
                TrendRegime.STRONG_UPTREND.value,
                TrendRegime.STRONG_DOWNTREND.value,
                TrendRegime.WEAK_UPTREND.value,
                TrendRegime.WEAK_DOWNTREND.value,
            ]

            # Combine into market regime
            if is_trending:
                if vol == VolatilityRegime.HIGH.value:
                    regime = MarketRegime.TRENDING_HIGH_VOL
                elif vol == VolatilityRegime.LOW.value:
                    regime = MarketRegime.TRENDING_LOW_VOL
                else:
                    regime = MarketRegime.TRENDING_NORMAL
            else:  # Ranging
                if vol == VolatilityRegime.HIGH.value:
                    regime = MarketRegime.RANGING_HIGH_VOL
                elif vol == VolatilityRegime.LOW.value:
                    regime = MarketRegime.RANGING_LOW_VOL
                else:
                    regime = MarketRegime.RANGING_NORMAL

            market_regime.append(regime.value)

        result["market_regime"] = market_regime

        return result

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add one-hot encoded regime features for model input."""
        result = df.copy()

        # Trend regime one-hot
        for regime in TrendRegime:
            result[f"regime_trend_{regime.value}"] = (
                result["trend_regime"] == regime.value
            ).astype(int)

        # Volatility regime one-hot
        for regime in VolatilityRegime:
            result[f"regime_vol_{regime.value}"] = (
                result["volatility_regime"] == regime.value
            ).astype(int)

        # Market regime one-hot
        for regime in MarketRegime:
            result[f"regime_{regime.value}"] = (
                result["market_regime"] == regime.value
            ).astype(int)

        # Simplified binary features
        result["is_trending"] = result["trend_regime"].isin([
            TrendRegime.STRONG_UPTREND.value,
            TrendRegime.STRONG_DOWNTREND.value,
            TrendRegime.WEAK_UPTREND.value,
            TrendRegime.WEAK_DOWNTREND.value,
        ]).astype(int)

        result["is_strong_trend"] = result["trend_regime"].isin([
            TrendRegime.STRONG_UPTREND.value,
            TrendRegime.STRONG_DOWNTREND.value,
        ]).astype(int)

        result["is_uptrend"] = result["trend_regime"].isin([
            TrendRegime.STRONG_UPTREND.value,
            TrendRegime.WEAK_UPTREND.value,
        ]).astype(int)

        result["is_high_vol"] = (
            result["volatility_regime"] == VolatilityRegime.HIGH.value
        ).astype(int)

        result["is_low_vol"] = (
            result["volatility_regime"] == VolatilityRegime.LOW.value
        ).astype(int)

        return result

    def _smooth_regime(self, regime_series: pd.Series, min_bars: int = None) -> pd.Series:
        """Smooth regime changes to avoid whipsaws.

        Only changes regime if new regime persists for min_bars.
        """
        if min_bars is None:
            min_bars = self.config.regime_smoothing

        if min_bars <= 1:
            return regime_series

        result = regime_series.copy()
        current_regime = regime_series.iloc[0]
        counter = 0

        for i in range(len(regime_series)):
            if regime_series.iloc[i] == current_regime:
                counter = 0
                result.iloc[i] = current_regime
            else:
                counter += 1
                if counter >= min_bars:
                    current_regime = regime_series.iloc[i]
                    counter = 0
                result.iloc[i] = current_regime

        return result


def get_regime_stats(df: pd.DataFrame, trades: pd.DataFrame) -> Dict:
    """Calculate trading statistics by regime.

    Args:
        df: DataFrame with regime columns
        trades: DataFrame with trade results (entry_time, pnl_pips, etc.)

    Returns:
        Dict with stats per regime
    """
    stats = {}

    for regime in MarketRegime:
        # Filter bars for this regime
        regime_mask = df["market_regime"] == regime.value
        regime_bars = df[regime_mask]

        if len(regime_bars) == 0:
            stats[regime.value] = {
                "bar_count": 0,
                "bar_pct": 0,
                "trades": 0,
                "win_rate": 0,
                "avg_pips": 0,
                "total_pips": 0,
            }
            continue

        # Find trades that occurred in this regime
        regime_trades = []
        for _, trade in trades.iterrows():
            entry_time = trade["entry_time"]
            if entry_time in df.index:
                if df.loc[entry_time, "market_regime"] == regime.value:
                    regime_trades.append(trade)

        regime_trades_df = pd.DataFrame(regime_trades) if regime_trades else pd.DataFrame()

        # Calculate stats
        if len(regime_trades_df) > 0:
            wins = (regime_trades_df["pnl_pips"] > 0).sum()
            win_rate = wins / len(regime_trades_df) * 100
            avg_pips = regime_trades_df["pnl_pips"].mean()
            total_pips = regime_trades_df["pnl_pips"].sum()
        else:
            win_rate = 0
            avg_pips = 0
            total_pips = 0

        stats[regime.value] = {
            "bar_count": len(regime_bars),
            "bar_pct": len(regime_bars) / len(df) * 100,
            "trades": len(regime_trades_df),
            "win_rate": win_rate,
            "avg_pips": avg_pips,
            "total_pips": total_pips,
        }

    return stats
