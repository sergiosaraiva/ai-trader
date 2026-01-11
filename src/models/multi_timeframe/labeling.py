"""Advanced labeling methods for trading models.

This module implements sophisticated labeling approaches that produce
better training signals than simple next-bar direction prediction.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LabelMethod(Enum):
    """Available labeling methods."""
    NEXT_BAR = "next_bar"  # Original simple method
    MULTI_BAR = "multi_bar"  # Forward N bars return
    TRIPLE_BARRIER = "triple_barrier"  # TP/SL/Time barriers
    VOLATILITY_ADJUSTED = "volatility_adjusted"  # Dynamic threshold


@dataclass
class LabelingConfig:
    """Configuration for labeling."""
    method: LabelMethod = LabelMethod.TRIPLE_BARRIER

    # Multi-bar settings
    forward_bars: int = 12  # How many bars to look ahead
    threshold_pips: float = 10.0  # Minimum move to label

    # Triple barrier settings
    tp_pips: float = 20.0  # Take profit in pips
    sl_pips: float = 10.0  # Stop loss in pips
    max_holding_bars: int = 24  # Maximum bars before timeout

    # Volatility adjusted settings
    atr_multiplier: float = 2.0  # Threshold as multiple of ATR

    # Common
    pip_value: float = 0.0001  # Pip size for EUR/USD
    min_samples_per_class: float = 0.3  # Minimum class balance


class AdvancedLabeler:
    """Creates training labels using advanced methods."""

    def __init__(self, config: Optional[LabelingConfig] = None):
        self.config = config or LabelingConfig()

    def create_labels(
        self,
        df: pd.DataFrame,
        method: Optional[LabelMethod] = None,
    ) -> Tuple[pd.Series, pd.Series]:
        """Create labels using specified method.

        Args:
            df: OHLCV DataFrame
            method: Labeling method (uses config default if None)

        Returns:
            Tuple of (labels, valid_mask)
        """
        method = method or self.config.method

        if method == LabelMethod.NEXT_BAR:
            return self._label_next_bar(df)
        elif method == LabelMethod.MULTI_BAR:
            return self._label_multi_bar(df)
        elif method == LabelMethod.TRIPLE_BARRIER:
            return self._label_triple_barrier(df)
        elif method == LabelMethod.VOLATILITY_ADJUSTED:
            return self._label_volatility_adjusted(df)
        else:
            raise ValueError(f"Unknown labeling method: {method}")

    def _label_next_bar(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Original simple next-bar labeling (for comparison)."""
        returns = df["close"].pct_change(1).shift(-1)
        threshold = self.config.threshold_pips * self.config.pip_value

        labels = pd.Series(index=df.index, dtype=float)
        labels[returns > threshold] = 1
        labels[returns < -threshold] = 0

        valid_mask = ~labels.isna()
        return labels, valid_mask

    def _label_multi_bar(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Label based on forward N-bar returns.

        This looks at the price change over the next N bars,
        providing more meaningful signals than single-bar prediction.
        """
        n = self.config.forward_bars
        threshold = self.config.threshold_pips * self.config.pip_value

        # Forward returns over N bars
        forward_returns = (df["close"].shift(-n) - df["close"]) / df["close"]

        labels = pd.Series(index=df.index, dtype=float)
        labels[forward_returns > threshold] = 1  # Bullish
        labels[forward_returns < -threshold] = 0  # Bearish
        # Returns between -threshold and +threshold remain NaN (excluded)

        valid_mask = ~labels.isna()

        # Check class balance
        if valid_mask.sum() > 0:
            balance = labels[valid_mask].mean()
            logger.info(
                f"Multi-bar labeling: {valid_mask.sum()} valid samples, "
                f"class balance: {balance:.1%} bullish"
            )

        return labels, valid_mask

    def _label_triple_barrier(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Triple barrier labeling method.

        Labels based on which barrier is hit first:
        - Take profit barrier hit → 1 (bullish)
        - Stop loss barrier hit → 0 (bearish)
        - Time barrier hit → direction at expiry

        This method produces labels that align with actual trading outcomes.
        """
        tp = self.config.tp_pips * self.config.pip_value
        sl = self.config.sl_pips * self.config.pip_value
        max_bars = self.config.max_holding_bars

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        n = len(df)

        labels = np.full(n, np.nan)

        for i in range(n - max_bars):
            entry_price = closes[i]
            tp_price = entry_price + tp
            sl_price = entry_price - sl

            label = None

            # Check each future bar
            for j in range(i + 1, min(i + max_bars + 1, n)):
                # Check if TP hit (high touches TP)
                if highs[j] >= tp_price:
                    label = 1
                    break
                # Check if SL hit (low touches SL)
                if lows[j] <= sl_price:
                    label = 0
                    break

            # Time barrier - use final direction
            if label is None:
                final_idx = min(i + max_bars, n - 1)
                final_return = closes[final_idx] - entry_price
                label = 1 if final_return > 0 else 0

            labels[i] = label

        labels_series = pd.Series(labels, index=df.index)
        valid_mask = ~labels_series.isna()

        # Log statistics
        if valid_mask.sum() > 0:
            balance = labels_series[valid_mask].mean()
            logger.info(
                f"Triple barrier labeling: {valid_mask.sum()} valid samples, "
                f"class balance: {balance:.1%} bullish"
            )

        return labels_series, valid_mask

    def _label_volatility_adjusted(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Volatility-adjusted labeling.

        Uses ATR to set dynamic thresholds, accounting for
        changing market volatility.
        """
        # Calculate ATR if not present
        if "atr_14" not in df.columns:
            high = df["high"]
            low = df["low"]
            close = df["close"]

            tr = pd.concat([
                high - low,
                abs(high - close.shift(1)),
                abs(low - close.shift(1))
            ], axis=1).max(axis=1)
            atr = tr.ewm(span=14, adjust=False).mean()
        else:
            atr = df["atr_14"]

        # Dynamic threshold based on ATR
        threshold = atr * self.config.atr_multiplier

        n = self.config.forward_bars
        max_bars = self.config.max_holding_bars

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        thresholds = threshold.values
        n_rows = len(df)

        labels = np.full(n_rows, np.nan)

        for i in range(n_rows - max_bars):
            entry_price = closes[i]
            thresh = thresholds[i]

            if np.isnan(thresh) or thresh <= 0:
                continue

            # Look for significant move within forward window
            max_up = 0
            max_down = 0

            for j in range(i + 1, min(i + max_bars + 1, n_rows)):
                up_move = highs[j] - entry_price
                down_move = entry_price - lows[j]
                max_up = max(max_up, up_move)
                max_down = max(max_down, down_move)

                # Early exit if threshold reached
                if max_up >= thresh or max_down >= thresh:
                    break

            # Label based on which direction had larger move
            if max_up >= thresh and max_up > max_down:
                labels[i] = 1
            elif max_down >= thresh and max_down > max_up:
                labels[i] = 0
            # Else remains NaN (no significant move)

        labels_series = pd.Series(labels, index=df.index)
        valid_mask = ~labels_series.isna()

        if valid_mask.sum() > 0:
            balance = labels_series[valid_mask].mean()
            logger.info(
                f"Volatility-adjusted labeling: {valid_mask.sum()} valid samples, "
                f"class balance: {balance:.1%} bullish"
            )

        return labels_series, valid_mask


def create_triple_barrier_labels(
    df: pd.DataFrame,
    tp_pips: float = 20.0,
    sl_pips: float = 10.0,
    max_holding_bars: int = 24,
) -> Tuple[pd.Series, pd.Series]:
    """Convenience function for triple barrier labeling.

    Args:
        df: OHLCV DataFrame
        tp_pips: Take profit in pips
        sl_pips: Stop loss in pips
        max_holding_bars: Maximum holding period

    Returns:
        Tuple of (labels, valid_mask)
    """
    config = LabelingConfig(
        method=LabelMethod.TRIPLE_BARRIER,
        tp_pips=tp_pips,
        sl_pips=sl_pips,
        max_holding_bars=max_holding_bars,
    )
    labeler = AdvancedLabeler(config)
    return labeler.create_labels(df)


def create_multi_bar_labels(
    df: pd.DataFrame,
    forward_bars: int = 12,
    threshold_pips: float = 10.0,
) -> Tuple[pd.Series, pd.Series]:
    """Convenience function for multi-bar labeling.

    Args:
        df: OHLCV DataFrame
        forward_bars: Number of bars to look ahead
        threshold_pips: Minimum move threshold in pips

    Returns:
        Tuple of (labels, valid_mask)
    """
    config = LabelingConfig(
        method=LabelMethod.MULTI_BAR,
        forward_bars=forward_bars,
        threshold_pips=threshold_pips,
    )
    labeler = AdvancedLabeler(config)
    return labeler.create_labels(df)
