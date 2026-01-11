"""Multi-timeframe predictor with signal aggregation.

Combines predictions from multiple timeframes to generate
high-quality trading signals.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .mtf_model import MultiTimeframeModel, TimeframePrediction, TimeframeConfig

logger = logging.getLogger(__name__)


class SignalAlignment(Enum):
    """Signal alignment status across timeframes."""

    STRONG_BULLISH = "strong_bullish"  # All timeframes agree UP
    WEAK_BULLISH = "weak_bullish"  # Primary bullish, others neutral/mixed
    NEUTRAL = "neutral"  # No clear signal
    WEAK_BEARISH = "weak_bearish"  # Primary bearish, others neutral/mixed
    STRONG_BEARISH = "strong_bearish"  # All timeframes agree DOWN
    CONFLICTING = "conflicting"  # Timeframes disagree


@dataclass
class AggregatedPrediction:
    """Aggregated prediction from all timeframes."""

    # Individual predictions
    predictions: Dict[str, TimeframePrediction]

    # Aggregated results
    direction: int  # 1 = UP, 0 = DOWN, -1 = NO TRADE
    confidence: float  # Combined confidence
    alignment: SignalAlignment

    # Primary signal details
    primary_timeframe: str
    primary_confidence: float
    primary_direction: int

    # Confirmation status
    confirmation_agrees: bool
    trend_agrees: bool

    # Regime info
    primary_regime: str

    # Timing
    timestamp: datetime

    # Trade decision
    should_trade: bool
    trade_reason: str

    @property
    def is_bullish(self) -> bool:
        return self.direction == 1 and self.should_trade

    @property
    def is_bearish(self) -> bool:
        return self.direction == 0 and self.should_trade

    @property
    def quality_score(self) -> float:
        """Overall quality score 0-1."""
        score = self.confidence

        # Bonus for alignment
        if self.alignment in [SignalAlignment.STRONG_BULLISH, SignalAlignment.STRONG_BEARISH]:
            score *= 1.2
        elif self.alignment == SignalAlignment.CONFLICTING:
            score *= 0.7

        # Bonus for confirmation
        if self.confirmation_agrees:
            score *= 1.1
        if self.trend_agrees:
            score *= 1.1

        return min(1.0, score)


class MTFPredictor:
    """Aggregates predictions from multiple timeframes.

    Trading logic:
    1. Primary (5min) must have confidence >= threshold
    2. Confirmation (15min) should agree or be neutral
    3. Trend (30min) should be aligned or neutral
    4. Regime should be favorable

    Only trade when conditions align.
    """

    # Confidence thresholds
    PRIMARY_MIN_CONFIDENCE = 0.65
    CONFIRMATION_MIN_CONFIDENCE = 0.52
    TREND_MIN_CONFIDENCE = 0.52

    # Regime settings
    FAVORABLE_REGIMES = ["trending_down", "trending_up"]
    AVOID_REGIMES = ["volatile"]

    def __init__(
        self,
        mtf_model: MultiTimeframeModel,
        primary_timeframe: str = "5min",
        confirmation_timeframe: str = "15min",
        trend_timeframe: str = "30min",
        primary_min_confidence: float = 0.65,
        require_confirmation: bool = True,
        require_trend_alignment: bool = True,
    ):
        """Initialize MTF predictor.

        Args:
            mtf_model: Multi-timeframe model manager
            primary_timeframe: Timeframe for primary signals
            confirmation_timeframe: Timeframe for confirmation
            trend_timeframe: Timeframe for trend filter
            primary_min_confidence: Minimum confidence for primary signal
            require_confirmation: Whether confirmation must agree
            require_trend_alignment: Whether trend must be aligned
        """
        self.mtf_model = mtf_model
        self.primary_timeframe = primary_timeframe
        self.confirmation_timeframe = confirmation_timeframe
        self.trend_timeframe = trend_timeframe

        self.primary_min_confidence = primary_min_confidence
        self.require_confirmation = require_confirmation
        self.require_trend_alignment = require_trend_alignment

        # Statistics
        self.total_predictions = 0
        self.trade_signals = 0

    def predict(self, df_5min: pd.DataFrame) -> AggregatedPrediction:
        """Generate aggregated prediction from 5-minute data.

        Args:
            df_5min: Recent 5-minute OHLCV data

        Returns:
            AggregatedPrediction with trade decision
        """
        self.total_predictions += 1

        # Get predictions from all timeframes
        predictions = self.mtf_model.predict_all(df_5min)

        # Extract individual predictions
        primary = predictions.get(self.primary_timeframe)
        confirmation = predictions.get(self.confirmation_timeframe)
        trend = predictions.get(self.trend_timeframe)

        # Default result (no trade)
        timestamp = df_5min.index[-1] if len(df_5min) > 0 else datetime.now()

        if primary is None or not primary.is_valid:
            return self._create_no_trade(
                predictions, "Primary timeframe prediction invalid", timestamp
            )

        # Check primary confidence
        if primary.confidence < self.primary_min_confidence:
            return self._create_no_trade(
                predictions,
                f"Primary confidence {primary.confidence:.1%} below {self.primary_min_confidence:.1%}",
                timestamp,
            )

        # Determine signal alignment
        alignment = self._calculate_alignment(primary, confirmation, trend)

        # Check confirmation agreement
        confirmation_agrees = self._check_confirmation(primary, confirmation)
        if self.require_confirmation and not confirmation_agrees:
            if confirmation and confirmation.is_valid and confirmation.confidence >= self.CONFIRMATION_MIN_CONFIDENCE:
                return self._create_no_trade(
                    predictions,
                    f"Confirmation disagrees: primary={primary.direction}, conf={confirmation.direction}",
                    timestamp,
                    primary=primary,
                    confirmation_agrees=False,
                )

        # Check trend alignment
        trend_agrees = self._check_trend(primary, trend)
        if self.require_trend_alignment and not trend_agrees:
            if trend and trend.is_valid and trend.confidence >= self.TREND_MIN_CONFIDENCE:
                return self._create_no_trade(
                    predictions,
                    f"Trend disagrees: primary={primary.direction}, trend={trend.direction}",
                    timestamp,
                    primary=primary,
                    confirmation_agrees=confirmation_agrees,
                )

        # Check regime
        if primary.regime in self.AVOID_REGIMES:
            return self._create_no_trade(
                predictions,
                f"Unfavorable regime: {primary.regime}",
                timestamp,
                primary=primary,
                confirmation_agrees=confirmation_agrees,
            )

        # Calculate combined confidence
        combined_confidence = self._calculate_combined_confidence(
            primary, confirmation, trend
        )

        # Generate trade signal
        self.trade_signals += 1

        direction_str = "LONG" if primary.direction == 1 else "SHORT"
        reason = (
            f"{direction_str}: primary={primary.confidence:.1%}, "
            f"alignment={alignment.value}, regime={primary.regime}"
        )

        return AggregatedPrediction(
            predictions=predictions,
            direction=primary.direction,
            confidence=combined_confidence,
            alignment=alignment,
            primary_timeframe=self.primary_timeframe,
            primary_confidence=primary.confidence,
            primary_direction=primary.direction,
            confirmation_agrees=confirmation_agrees,
            trend_agrees=trend_agrees,
            primary_regime=primary.regime,
            timestamp=timestamp,
            should_trade=True,
            trade_reason=reason,
        )

    def _create_no_trade(
        self,
        predictions: Dict[str, TimeframePrediction],
        reason: str,
        timestamp: datetime,
        primary: Optional[TimeframePrediction] = None,
        confirmation_agrees: bool = False,
    ) -> AggregatedPrediction:
        """Create a no-trade prediction."""
        return AggregatedPrediction(
            predictions=predictions,
            direction=-1,
            confidence=0.0,
            alignment=SignalAlignment.NEUTRAL,
            primary_timeframe=self.primary_timeframe,
            primary_confidence=primary.confidence if primary else 0.0,
            primary_direction=primary.direction if primary else -1,
            confirmation_agrees=confirmation_agrees,
            trend_agrees=False,
            primary_regime=primary.regime if primary else "unknown",
            timestamp=timestamp,
            should_trade=False,
            trade_reason=reason,
        )

    def _calculate_alignment(
        self,
        primary: TimeframePrediction,
        confirmation: Optional[TimeframePrediction],
        trend: Optional[TimeframePrediction],
    ) -> SignalAlignment:
        """Calculate signal alignment across timeframes."""
        directions = [primary.direction]

        if confirmation and confirmation.is_valid and confirmation.confidence >= 0.52:
            directions.append(confirmation.direction)

        if trend and trend.is_valid and trend.confidence >= 0.52:
            directions.append(trend.direction)

        # Check agreement
        all_up = all(d == 1 for d in directions)
        all_down = all(d == 0 for d in directions)

        if len(directions) >= 3:
            if all_up:
                return SignalAlignment.STRONG_BULLISH
            elif all_down:
                return SignalAlignment.STRONG_BEARISH
            elif directions.count(1) >= 2:
                return SignalAlignment.WEAK_BULLISH
            elif directions.count(0) >= 2:
                return SignalAlignment.WEAK_BEARISH
            else:
                return SignalAlignment.CONFLICTING
        elif len(directions) == 2:
            if all_up:
                return SignalAlignment.WEAK_BULLISH
            elif all_down:
                return SignalAlignment.WEAK_BEARISH
            else:
                return SignalAlignment.CONFLICTING
        else:
            if primary.direction == 1:
                return SignalAlignment.WEAK_BULLISH
            else:
                return SignalAlignment.WEAK_BEARISH

    def _check_confirmation(
        self,
        primary: TimeframePrediction,
        confirmation: Optional[TimeframePrediction],
    ) -> bool:
        """Check if confirmation agrees with primary."""
        if confirmation is None or not confirmation.is_valid:
            return True  # No confirmation available, allow trade

        if confirmation.confidence < self.CONFIRMATION_MIN_CONFIDENCE:
            return True  # Low confidence confirmation, treat as neutral

        return confirmation.direction == primary.direction

    def _check_trend(
        self,
        primary: TimeframePrediction,
        trend: Optional[TimeframePrediction],
    ) -> bool:
        """Check if trend agrees with primary."""
        if trend is None or not trend.is_valid:
            return True  # No trend available, allow trade

        if trend.confidence < self.TREND_MIN_CONFIDENCE:
            return True  # Low confidence trend, treat as neutral

        return trend.direction == primary.direction

    def _calculate_combined_confidence(
        self,
        primary: TimeframePrediction,
        confirmation: Optional[TimeframePrediction],
        trend: Optional[TimeframePrediction],
    ) -> float:
        """Calculate combined confidence from all timeframes."""
        # Weighted average with primary having most weight
        weights = {"primary": 0.6, "confirmation": 0.25, "trend": 0.15}
        total_weight = weights["primary"]
        weighted_sum = primary.confidence * weights["primary"]

        if confirmation and confirmation.is_valid:
            # Only add if agrees with primary
            if confirmation.direction == primary.direction:
                weighted_sum += confirmation.confidence * weights["confirmation"]
                total_weight += weights["confirmation"]

        if trend and trend.is_valid:
            # Only add if agrees with primary
            if trend.direction == primary.direction:
                weighted_sum += trend.confidence * weights["trend"]
                total_weight += weights["trend"]

        return weighted_sum / total_weight if total_weight > 0 else primary.confidence

    def get_statistics(self) -> Dict[str, any]:
        """Get predictor statistics."""
        trade_rate = (
            self.trade_signals / self.total_predictions
            if self.total_predictions > 0
            else 0
        )
        return {
            "total_predictions": self.total_predictions,
            "trade_signals": self.trade_signals,
            "trade_rate": trade_rate,
        }

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self.total_predictions = 0
        self.trade_signals = 0
