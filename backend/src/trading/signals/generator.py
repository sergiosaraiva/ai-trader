"""
Signal Generator.

Converts ML model predictions to actionable trading signals
with confidence-based position sizing.

Enhanced in Phase 6 to integrate with EnsemblePredictor for
real-time feature computation and stop-loss/take-profit calculation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Any, Tuple, Union
import math
import logging

import pandas as pd

from .actions import Action, TradingSignal, SignalStrength, get_signal_strength
from ..risk.profiles import RiskProfile
from ..circuit_breakers.base import CircuitBreakerState, TradingState
from ..filters.regime_filter import RegimeFilter, RegimeAnalysis, MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class EnsemblePrediction:
    """Prediction from ensemble of models.

    Can be created directly or from EnsemblePredictor output.
    """

    # Direction prediction (from Beta distribution)
    direction_probability: float  # Mean = α/(α+β)
    confidence: float             # Computed from concentration

    # Beta parameters
    alpha: float = 1.0
    beta: float = 1.0

    # Individual model signals (-1 to 1)
    short_term_signal: float = 0.0
    medium_term_signal: float = 0.0
    long_term_signal: float = 0.0

    # Agreement metrics
    ensemble_agreement: float = 1.0     # How much models agree (0-1)

    # Market regime information
    market_regime: str = "unknown"
    volatility_level: str = "normal"

    @property
    def concentration(self) -> float:
        """Total concentration (α + β)."""
        return self.alpha + self.beta

    @property
    def is_bullish(self) -> bool:
        """Check if prediction is bullish."""
        return self.direction_probability > 0.5

    @property
    def is_bearish(self) -> bool:
        """Check if prediction is bearish."""
        return self.direction_probability < 0.5

    @classmethod
    def from_predictor_output(cls, pred: Any) -> "EnsemblePrediction":
        """Create from EnsemblePredictor output.

        Args:
            pred: EnsemblePrediction from models.ensemble.predictor

        Returns:
            EnsemblePrediction for signal generation
        """
        # Extract individual model signals from component predictions
        short_signal = 0.0
        medium_signal = 0.0
        long_signal = 0.0

        if hasattr(pred, 'component_predictions'):
            for name, comp in pred.component_predictions.items():
                # Convert direction (0/1) to signal (-1 to 1)
                signal_val = comp.direction * 2 - 1  # 0->-1, 1->1
                signal_val *= comp.confidence  # Scale by confidence

                if 'short' in name.lower():
                    short_signal = signal_val
                elif 'medium' in name.lower():
                    medium_signal = signal_val
                elif 'long' in name.lower():
                    long_signal = signal_val

        # Calculate alpha/beta from direction probability and confidence
        # Higher confidence = higher concentration
        concentration = 2 + pred.confidence * 18  # Map [0,1] to [2,20]

        if pred.direction == 1:  # Bullish
            alpha = concentration * pred.direction_probability
            beta = concentration * (1 - pred.direction_probability)
        else:  # Bearish
            alpha = concentration * pred.direction_probability
            beta = concentration * (1 - pred.direction_probability)

        return cls(
            direction_probability=pred.direction_probability,
            confidence=pred.confidence,
            alpha=alpha,
            beta=beta,
            short_term_signal=short_signal,
            medium_term_signal=medium_signal,
            long_term_signal=long_signal,
            ensemble_agreement=pred.agreement_score,
            market_regime=getattr(pred, 'market_regime', 'unknown'),
            volatility_level=getattr(pred, 'volatility_level', 'normal'),
        )


@dataclass
class Position:
    """Current position information."""
    symbol: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float


class SignalGenerator:
    """
    Converts model predictions to trading signals.

    DECISION LOGIC:
    1. Get ensemble prediction (Beta distribution output)
    2. Check circuit breaker state
    3. Apply confidence threshold (from risk profile)
    4. Determine action (BUY/SELL/HOLD)
    5. Calculate position size based on confidence
    6. Set stop-loss and take-profit levels
    """

    def __init__(
        self,
        risk_profile: RiskProfile,
        regime_filter: Optional[RegimeFilter] = None,
        timeframe: str = "1H",
    ):
        """
        Initialize signal generator.

        Args:
            risk_profile: Risk profile with thresholds and limits
            regime_filter: Optional regime filter for market condition filtering
            timeframe: Trading timeframe (used if regime_filter not provided)
        """
        self.risk_profile = risk_profile
        self.regime_filter = regime_filter or RegimeFilter(timeframe=timeframe)
        self.last_regime_analysis: Optional[RegimeAnalysis] = None

    def generate_signal(
        self,
        prediction: EnsemblePrediction,
        symbol: str,
        current_price: float,
        breaker_state: CircuitBreakerState,
        current_position: Optional[Position] = None,
        atr: float = 0.0,  # For stop-loss calculation
        market_data: Optional[pd.DataFrame] = None,  # For regime filtering
    ) -> TradingSignal:
        """
        Generate trading signal from model prediction.

        Args:
            prediction: Ensemble prediction with confidence
            symbol: Trading symbol
            current_price: Current market price
            breaker_state: Circuit breaker state
            current_position: Current position if any
            atr: Average True Range for stop-loss
            market_data: Recent OHLCV data for regime filtering (optional but recommended)

        Returns:
            TradingSignal with action and position sizing
        """
        # Check if trading is halted
        if breaker_state.is_halted:
            return TradingSignal.create_hold(
                symbol=symbol,
                reason=f"Circuit breaker: {', '.join(breaker_state.reasons)}"
            )

        # Apply regime filter if market data is provided
        regime_modifier = 1.0
        if market_data is not None and len(market_data) >= 50:
            self.last_regime_analysis = self.regime_filter.analyze(market_data)

            if not self.last_regime_analysis.should_trade:
                return TradingSignal.create_hold(
                    symbol=symbol,
                    reason=f"Regime filter: {self.last_regime_analysis.reason}"
                )

            # Apply regime confidence modifier
            regime_modifier = self.last_regime_analysis.confidence_modifier
            logger.debug(
                f"Regime: {self.last_regime_analysis.regime.value}, "
                f"ADX: {self.last_regime_analysis.adx:.1f}, "
                f"modifier: {regime_modifier:.2f}"
            )

        # Get effective thresholds (may be overridden by breaker)
        min_confidence = max(
            self.risk_profile.min_confidence_to_trade,
            breaker_state.min_confidence_override or 0,
        )

        # Adjust confidence by regime modifier
        effective_confidence = prediction.confidence * regime_modifier

        # Check confidence threshold
        if effective_confidence < min_confidence:
            return TradingSignal.create_hold(
                symbol=symbol,
                reason=f"Confidence {prediction.confidence:.1%} below threshold {min_confidence:.1%}"
            )

        # Determine action
        action, reason = self._determine_action(prediction, current_position)

        if action == Action.HOLD:
            return TradingSignal(
                action=Action.HOLD,
                symbol=symbol,
                timestamp=datetime.now(),
                confidence=prediction.confidence,
                direction_probability=prediction.direction_probability,
                alpha=prediction.alpha,
                beta=prediction.beta,
                concentration=prediction.concentration,
                short_term_signal=prediction.short_term_signal,
                medium_term_signal=prediction.medium_term_signal,
                long_term_signal=prediction.long_term_signal,
                ensemble_agreement=prediction.ensemble_agreement,
                reason=reason,
            )

        # Calculate position size (with regime modifier)
        position_size_pct = self._calculate_position_size(
            prediction=prediction,
            breaker_state=breaker_state,
            regime_modifier=regime_modifier,
        )

        # Calculate stop-loss and take-profit
        stop_loss_pct, take_profit_pct = self._calculate_exits(
            prediction=prediction,
            action=action,
            atr=atr,
            current_price=current_price,
        )

        # Build signal
        signal = TradingSignal(
            action=action,
            symbol=symbol,
            timestamp=datetime.now(),
            confidence=prediction.confidence,
            direction_probability=prediction.direction_probability,
            alpha=prediction.alpha,
            beta=prediction.beta,
            concentration=prediction.concentration,
            position_size_pct=position_size_pct,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            risk_reward_ratio=take_profit_pct / stop_loss_pct if stop_loss_pct > 0 else 0,
            short_term_signal=prediction.short_term_signal,
            medium_term_signal=prediction.medium_term_signal,
            long_term_signal=prediction.long_term_signal,
            ensemble_agreement=prediction.ensemble_agreement,
            signal_strength=get_signal_strength(prediction.confidence),
            reason=reason,
        )

        # Set price levels
        if atr > 0:
            if action == Action.BUY:
                signal.stop_loss_price = current_price * (1 - stop_loss_pct)
                signal.take_profit_price = current_price * (1 + take_profit_pct)
            else:
                signal.stop_loss_price = current_price * (1 + stop_loss_pct)
                signal.take_profit_price = current_price * (1 - take_profit_pct)

        return signal

    def _determine_action(
        self,
        prediction: EnsemblePrediction,
        current_position: Optional[Position],
    ) -> Tuple[Action, str]:
        """Determine trading action based on prediction."""

        # Check for neutral zone
        if 0.48 < prediction.direction_probability < 0.52:
            return Action.HOLD, "Direction probability in neutral zone"

        # Determine base direction
        if prediction.is_bullish:
            base_action = Action.BUY
            direction = "LONG"
        else:
            base_action = Action.SELL
            direction = "SHORT"

        # Check model agreement
        signals = [
            prediction.short_term_signal,
            prediction.medium_term_signal,
            prediction.long_term_signal,
        ]

        # If not all models agree on direction
        bullish_count = sum(1 for s in signals if s > 0)
        if bullish_count == 1 or bullish_count == 2:
            if prediction.confidence < 0.70:
                return Action.HOLD, f"Mixed model signals with confidence {prediction.confidence:.1%}"

        # Handle existing position
        if current_position:
            if current_position.side == 'long' and base_action == Action.SELL:
                return Action.CLOSE_LONG, f"Reversing from LONG to SHORT"
            elif current_position.side == 'short' and base_action == Action.BUY:
                return Action.CLOSE_SHORT, f"Reversing from SHORT to LONG"
            elif current_position.side == 'long' and base_action == Action.BUY:
                return Action.HOLD, "Already in LONG position"
            elif current_position.side == 'short' and base_action == Action.SELL:
                return Action.HOLD, "Already in SHORT position"

        reason = f"{direction}: prob={prediction.direction_probability:.1%}, conf={prediction.confidence:.1%}"
        return base_action, reason

    def _calculate_position_size(
        self,
        prediction: EnsemblePrediction,
        breaker_state: CircuitBreakerState,
        regime_modifier: float = 1.0,
    ) -> float:
        """
        Calculate position size based on confidence and Kelly criterion.

        FORMULA:
        size = base_size * confidence_factor * kelly_factor * agreement_factor * breaker_multiplier * regime_modifier

        Where:
        - base_size: From risk profile
        - confidence_factor: Linear scaling with confidence
        - kelly_factor: Based on edge and Kelly fraction
        - agreement_factor: Ensemble agreement bonus
        - breaker_multiplier: From circuit breaker state
        - regime_modifier: From regime filter (1.0 for optimal, 0.5 for neutral)
        """

        # Base size from risk profile
        base_size = self.risk_profile.base_position_pct

        # Confidence factor: scale linearly from min to full confidence
        min_conf = self.risk_profile.min_confidence_to_trade
        full_conf = self.risk_profile.full_position_confidence
        conf_range = full_conf - min_conf

        if conf_range > 0:
            conf_above_min = prediction.confidence - min_conf
            confidence_factor = min(1.0, max(0.0, conf_above_min / conf_range))
        else:
            confidence_factor = 1.0

        # Kelly factor (simplified)
        # Estimated edge based on confidence
        estimated_edge = (prediction.confidence - 0.5) * 2  # Scale to [0, 1]
        kelly_size = estimated_edge * self.risk_profile.kelly_fraction
        kelly_factor = max(0.1, min(1.0, kelly_size))

        # Ensemble agreement factor (sqrt to dampen effect)
        agreement_factor = math.sqrt(prediction.ensemble_agreement)

        # Circuit breaker multiplier
        breaker_multiplier = breaker_state.size_multiplier

        # Calculate final size
        position_size = (
            base_size *
            confidence_factor *
            kelly_factor *
            agreement_factor *
            breaker_multiplier *
            regime_modifier
        )

        # Cap at maximum
        position_size = min(position_size, self.risk_profile.max_position_pct)

        return max(0.0, position_size)

    def _calculate_exits(
        self,
        prediction: EnsemblePrediction,
        action: Action,
        atr: float,
        current_price: float,
    ) -> Tuple[float, float]:
        """
        Calculate stop-loss and take-profit levels.

        Uses ATR-based stops with confidence adjustment:
        - Higher confidence = tighter stops (more confident in direction)
        - Lower confidence = wider stops (give trade room)

        Target risk-reward ratio: 1.5 to 2.0
        """

        # Base stop-loss (2x ATR as percentage)
        if atr > 0 and current_price > 0:
            base_stop_pct = (atr * 2) / current_price
        else:
            base_stop_pct = 0.02  # Default 2%

        # Adjust stop by confidence (inverted - lower confidence = wider stop)
        confidence_adj = 1.0 + (1.0 - prediction.confidence) * 0.5  # 1.0 to 1.25
        stop_loss_pct = base_stop_pct * confidence_adj

        # Ensure minimum stop
        stop_loss_pct = max(stop_loss_pct, 0.005)  # At least 0.5%

        # Take profit based on risk-reward ratio (1.5 to 2.0)
        # Higher confidence = higher R:R target
        if prediction.confidence >= 0.80:
            rr_ratio = 2.5
        elif prediction.confidence >= 0.70:
            rr_ratio = 2.0
        else:
            rr_ratio = 1.5

        take_profit_pct = stop_loss_pct * rr_ratio

        return stop_loss_pct, take_profit_pct

    def generate_signal_from_predictor(
        self,
        predictor_output: Any,
        symbol: str,
        current_price: float,
        breaker_state: CircuitBreakerState,
        current_position: Optional[Position] = None,
        atr: float = 0.0,
        market_data: Optional[pd.DataFrame] = None,
    ) -> TradingSignal:
        """
        Generate trading signal directly from EnsemblePredictor output.

        This is a convenience method that converts EnsemblePredictor output
        to our internal EnsemblePrediction format and generates a signal.

        Args:
            predictor_output: Output from EnsemblePredictor.predict()
            symbol: Trading symbol
            current_price: Current market price
            breaker_state: Circuit breaker state
            current_position: Current position if any
            atr: Average True Range for stop-loss
            market_data: Recent OHLCV data for regime filtering

        Returns:
            TradingSignal with action and position sizing
        """
        # Convert predictor output to internal format
        prediction = EnsemblePrediction.from_predictor_output(predictor_output)

        # Generate signal using standard method
        return self.generate_signal(
            prediction=prediction,
            symbol=symbol,
            current_price=current_price,
            breaker_state=breaker_state,
            current_position=current_position,
            atr=atr,
            market_data=market_data,
        )

    def calculate_stop_loss_price(
        self,
        entry_price: float,
        action: Action,
        atr: float,
        confidence: float = 0.65,
    ) -> float:
        """
        Calculate stop-loss price based on ATR.

        Args:
            entry_price: Entry price
            action: BUY or SELL action
            atr: Average True Range
            confidence: Prediction confidence

        Returns:
            Stop-loss price
        """
        # Base stop distance (2x ATR)
        stop_distance = atr * 2

        # Adjust by confidence (lower confidence = wider stop)
        confidence_adj = 1.0 + (1.0 - confidence) * 0.5
        stop_distance *= confidence_adj

        # Calculate price
        if action == Action.BUY:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def calculate_take_profit_price(
        self,
        entry_price: float,
        action: Action,
        stop_loss_price: float,
        confidence: float = 0.65,
    ) -> float:
        """
        Calculate take-profit price based on risk-reward ratio.

        Args:
            entry_price: Entry price
            action: BUY or SELL action
            stop_loss_price: Stop-loss price
            confidence: Prediction confidence

        Returns:
            Take-profit price
        """
        # Risk-reward ratio based on confidence
        if confidence >= 0.80:
            rr_ratio = 2.5
        elif confidence >= 0.70:
            rr_ratio = 2.0
        else:
            rr_ratio = 1.5

        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss_price)

        # Calculate take profit
        take_profit_distance = stop_distance * rr_ratio

        if action == Action.BUY:
            return entry_price + take_profit_distance
        else:
            return entry_price - take_profit_distance

    def calculate_position_units(
        self,
        position_size_pct: float,
        account_equity: float,
        current_price: float,
        lot_size: float = 100000,  # Standard forex lot
    ) -> float:
        """
        Calculate position size in units/lots.

        Args:
            position_size_pct: Position size as percentage of equity
            account_equity: Current account equity
            current_price: Current price
            lot_size: Size of one lot (100,000 for forex)

        Returns:
            Position size in lots
        """
        # Calculate notional value
        notional = account_equity * position_size_pct

        # Calculate lots
        lots = notional / (current_price * lot_size)

        # Round to standard lot increments (0.01 mini lots)
        return round(lots, 2)
