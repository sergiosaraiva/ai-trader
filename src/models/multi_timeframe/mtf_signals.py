"""Multi-timeframe signal generator for scalping.

Converts MTF predictions into actionable trading signals
with position sizing, stop-loss, and take-profit levels.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from .mtf_predictor import AggregatedPrediction, SignalAlignment

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Signal strength levels."""

    VERY_STRONG = "very_strong"  # Conf >= 75%, all TF aligned
    STRONG = "strong"  # Conf >= 70%, most TF aligned
    MODERATE = "moderate"  # Conf >= 65%, primary strong
    WEAK = "weak"  # Conf >= 60%, minimal alignment
    NONE = "none"  # No signal


class TradeAction(Enum):
    """Trading actions."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


@dataclass
class ScalperSignal:
    """Trading signal for scalper system."""

    # Action
    action: TradeAction
    strength: SignalStrength

    # Direction and confidence
    direction: int  # 1 = UP, 0 = DOWN, -1 = NONE
    confidence: float
    alignment: SignalAlignment

    # Position sizing
    position_size_pct: float  # % of equity
    risk_amount_pct: float  # % of equity at risk

    # Price levels
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    stop_loss_pips: float
    take_profit_pips: float
    risk_reward_ratio: float

    # Metadata
    symbol: str
    timeframe: str
    timestamp: datetime
    reason: str

    # Quality
    quality_score: float

    @property
    def is_trade_signal(self) -> bool:
        return self.action in [TradeAction.BUY, TradeAction.SELL]

    @property
    def is_long(self) -> bool:
        return self.action == TradeAction.BUY

    @property
    def is_short(self) -> bool:
        return self.action == TradeAction.SELL


@dataclass
class ScalperConfig:
    """Configuration for scalper signal generator."""

    # Position sizing
    base_position_pct: float = 0.02  # 2% base position
    max_position_pct: float = 0.05  # 5% max position
    risk_per_trade_pct: float = 0.01  # 1% risk per trade

    # Stop-loss settings (in pips for forex)
    base_stop_loss_pips: float = 10.0  # Base SL
    min_stop_loss_pips: float = 5.0  # Minimum SL
    max_stop_loss_pips: float = 20.0  # Maximum SL

    # Take-profit settings
    base_risk_reward: float = 2.0  # Base R:R ratio
    min_risk_reward: float = 1.5  # Minimum R:R
    max_risk_reward: float = 3.0  # Maximum R:R

    # Confidence thresholds
    very_strong_confidence: float = 0.75
    strong_confidence: float = 0.70
    moderate_confidence: float = 0.65
    weak_confidence: float = 0.60

    # Position size multipliers by strength
    very_strong_multiplier: float = 1.5
    strong_multiplier: float = 1.2
    moderate_multiplier: float = 1.0
    weak_multiplier: float = 0.7

    # Pip value (for EURUSD, 1 pip = 0.0001)
    pip_value: float = 0.0001


class MTFSignalGenerator:
    """Generates trading signals from MTF predictions.

    Handles:
    - Signal strength classification
    - Position sizing based on confidence
    - Stop-loss and take-profit calculation
    - Risk management
    """

    def __init__(
        self,
        config: Optional[ScalperConfig] = None,
        symbol: str = "EURUSD",
    ):
        """Initialize signal generator.

        Args:
            config: Scalper configuration
            symbol: Trading symbol
        """
        self.config = config or ScalperConfig()
        self.symbol = symbol

        # Statistics
        self.signals_generated = 0
        self.buy_signals = 0
        self.sell_signals = 0
        self.hold_signals = 0

    def generate_signal(
        self,
        prediction: AggregatedPrediction,
        current_price: float,
        atr_pips: Optional[float] = None,
    ) -> ScalperSignal:
        """Generate trading signal from MTF prediction.

        Args:
            prediction: Aggregated prediction from MTF predictor
            current_price: Current market price
            atr_pips: Optional ATR in pips for dynamic SL/TP

        Returns:
            ScalperSignal with trade details
        """
        self.signals_generated += 1

        # Check if we should trade
        if not prediction.should_trade:
            self.hold_signals += 1
            return self._create_hold_signal(prediction, current_price)

        # Determine signal strength
        strength = self._calculate_strength(prediction)

        # Calculate position size
        position_size = self._calculate_position_size(prediction, strength)

        # Calculate stop-loss and take-profit
        sl_pips, tp_pips = self._calculate_sl_tp(prediction, strength, atr_pips)

        # Convert to prices
        sl_price, tp_price = self._calculate_price_levels(
            current_price, prediction.direction, sl_pips, tp_pips
        )

        # Calculate risk-reward ratio
        rr_ratio = tp_pips / sl_pips if sl_pips > 0 else 0

        # Calculate risk amount
        risk_amount = position_size * (sl_pips * self.config.pip_value / current_price)

        # Determine action
        if prediction.direction == 1:
            action = TradeAction.BUY
            self.buy_signals += 1
        else:
            action = TradeAction.SELL
            self.sell_signals += 1

        return ScalperSignal(
            action=action,
            strength=strength,
            direction=prediction.direction,
            confidence=prediction.confidence,
            alignment=prediction.alignment,
            position_size_pct=position_size,
            risk_amount_pct=risk_amount,
            entry_price=current_price,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            stop_loss_pips=sl_pips,
            take_profit_pips=tp_pips,
            risk_reward_ratio=rr_ratio,
            symbol=self.symbol,
            timeframe=prediction.primary_timeframe,
            timestamp=prediction.timestamp,
            reason=prediction.trade_reason,
            quality_score=prediction.quality_score,
        )

    def _create_hold_signal(
        self,
        prediction: AggregatedPrediction,
        current_price: float,
    ) -> ScalperSignal:
        """Create a hold signal."""
        return ScalperSignal(
            action=TradeAction.HOLD,
            strength=SignalStrength.NONE,
            direction=-1,
            confidence=prediction.confidence,
            alignment=prediction.alignment,
            position_size_pct=0.0,
            risk_amount_pct=0.0,
            entry_price=current_price,
            stop_loss_price=0.0,
            take_profit_price=0.0,
            stop_loss_pips=0.0,
            take_profit_pips=0.0,
            risk_reward_ratio=0.0,
            symbol=self.symbol,
            timeframe=prediction.primary_timeframe,
            timestamp=prediction.timestamp,
            reason=prediction.trade_reason,
            quality_score=0.0,
        )

    def _calculate_strength(self, prediction: AggregatedPrediction) -> SignalStrength:
        """Calculate signal strength from prediction."""
        conf = prediction.confidence
        alignment = prediction.alignment

        # Strong alignment bonus
        alignment_bonus = 0.0
        if alignment in [SignalAlignment.STRONG_BULLISH, SignalAlignment.STRONG_BEARISH]:
            alignment_bonus = 0.05
        elif alignment in [SignalAlignment.WEAK_BULLISH, SignalAlignment.WEAK_BEARISH]:
            alignment_bonus = 0.02

        effective_conf = conf + alignment_bonus

        if effective_conf >= self.config.very_strong_confidence:
            return SignalStrength.VERY_STRONG
        elif effective_conf >= self.config.strong_confidence:
            return SignalStrength.STRONG
        elif effective_conf >= self.config.moderate_confidence:
            return SignalStrength.MODERATE
        elif effective_conf >= self.config.weak_confidence:
            return SignalStrength.WEAK
        else:
            return SignalStrength.NONE

    def _calculate_position_size(
        self,
        prediction: AggregatedPrediction,
        strength: SignalStrength,
    ) -> float:
        """Calculate position size based on confidence and strength."""
        base = self.config.base_position_pct

        # Apply multiplier based on strength
        if strength == SignalStrength.VERY_STRONG:
            multiplier = self.config.very_strong_multiplier
        elif strength == SignalStrength.STRONG:
            multiplier = self.config.strong_multiplier
        elif strength == SignalStrength.MODERATE:
            multiplier = self.config.moderate_multiplier
        elif strength == SignalStrength.WEAK:
            multiplier = self.config.weak_multiplier
        else:
            return 0.0

        # Scale by confidence above threshold
        conf_scale = (prediction.confidence - 0.5) * 2  # 0 to 1
        multiplier *= (0.8 + 0.4 * conf_scale)  # 0.8 to 1.2

        # Apply alignment bonus
        if prediction.alignment in [
            SignalAlignment.STRONG_BULLISH,
            SignalAlignment.STRONG_BEARISH,
        ]:
            multiplier *= 1.1

        position_size = base * multiplier

        # Cap at maximum
        return min(position_size, self.config.max_position_pct)

    def _calculate_sl_tp(
        self,
        prediction: AggregatedPrediction,
        strength: SignalStrength,
        atr_pips: Optional[float] = None,
    ) -> tuple[float, float]:
        """Calculate stop-loss and take-profit in pips."""
        # Base SL from ATR or config
        if atr_pips is not None and atr_pips > 0:
            base_sl = atr_pips * 1.5  # 1.5x ATR
        else:
            base_sl = self.config.base_stop_loss_pips

        # Adjust SL based on strength (tighter for stronger signals)
        if strength == SignalStrength.VERY_STRONG:
            sl_multiplier = 0.8
            rr_ratio = self.config.max_risk_reward
        elif strength == SignalStrength.STRONG:
            sl_multiplier = 0.9
            rr_ratio = (self.config.base_risk_reward + self.config.max_risk_reward) / 2
        elif strength == SignalStrength.MODERATE:
            sl_multiplier = 1.0
            rr_ratio = self.config.base_risk_reward
        else:
            sl_multiplier = 1.2
            rr_ratio = self.config.min_risk_reward

        sl_pips = base_sl * sl_multiplier

        # Clamp SL
        sl_pips = max(self.config.min_stop_loss_pips, sl_pips)
        sl_pips = min(self.config.max_stop_loss_pips, sl_pips)

        # Calculate TP based on R:R ratio
        tp_pips = sl_pips * rr_ratio

        return sl_pips, tp_pips

    def _calculate_price_levels(
        self,
        current_price: float,
        direction: int,
        sl_pips: float,
        tp_pips: float,
    ) -> tuple[float, float]:
        """Convert pip distances to price levels."""
        pip_value = self.config.pip_value

        if direction == 1:  # LONG
            sl_price = current_price - (sl_pips * pip_value)
            tp_price = current_price + (tp_pips * pip_value)
        else:  # SHORT
            sl_price = current_price + (sl_pips * pip_value)
            tp_price = current_price - (tp_pips * pip_value)

        return sl_price, tp_price

    def get_statistics(self) -> dict:
        """Get signal generation statistics."""
        total = self.signals_generated
        return {
            "total_signals": total,
            "buy_signals": self.buy_signals,
            "sell_signals": self.sell_signals,
            "hold_signals": self.hold_signals,
            "buy_rate": self.buy_signals / total if total > 0 else 0,
            "sell_rate": self.sell_signals / total if total > 0 else 0,
            "trade_rate": (self.buy_signals + self.sell_signals) / total if total > 0 else 0,
        }

    def reset_statistics(self) -> None:
        """Reset statistics."""
        self.signals_generated = 0
        self.buy_signals = 0
        self.sell_signals = 0
        self.hold_signals = 0
