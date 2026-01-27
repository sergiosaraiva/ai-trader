"""Circuit breaker system for risk management.

This module provides the TradingCircuitBreaker class that enforces:
- Daily loss limits
- Consecutive loss limits
- Monthly drawdown limits (future enhancement)
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional, TYPE_CHECKING

from sqlalchemy import func
from sqlalchemy.orm import Session

# Deferred import to avoid circular dependency
if TYPE_CHECKING:
    from ...api.database.models import Trade, CircuitBreakerEvent

logger = logging.getLogger(__name__)


class TradingCircuitBreaker:
    """Circuit breaker system to prevent excessive losses.

    Checks:
    1. Daily loss limit (% of balance)
    2. Consecutive loss limit (# of trades)
    3. Monthly drawdown limit (future)

    Returns tuple: (can_trade: bool, reason: Optional[str])
    """

    def __init__(self, config):
        """Initialize circuit breaker.

        Args:
            config: ConservativeHybridParameters configuration object
        """
        self.config = config

    def can_trade(self, db: Session, balance: float) -> Tuple[bool, Optional[str], float]:
        """Check if trading is allowed based on circuit breaker rules.

        Args:
            db: Database session
            balance: Current account balance

        Returns:
            Tuple of (can_trade: bool, reason: Optional[str], risk_reduction_factor: float)
            - If can_trade is False, reason contains the trigger description
            - risk_reduction_factor: 0.2-1.0 based on consecutive losses (1.0 = normal)
        """
        # Check 1: Daily loss limit (HARD STOP - unchanged)
        daily_pnl = self.get_daily_pnl(db)
        daily_loss_limit_amount = balance * (abs(self.config.daily_loss_limit_percent) / 100.0)

        if daily_pnl < 0 and abs(daily_pnl) >= daily_loss_limit_amount:
            # Calculate loss percentage
            loss_pct = (daily_pnl / balance) * 100.0

            reason = (
                f"Daily loss limit breached: ${daily_pnl:.2f} "
                f"(limit: {self.config.daily_loss_limit_percent:.1f}% = ${-daily_loss_limit_amount:.2f})"
            )
            logger.warning(f"Circuit breaker triggered: {reason}")

            # Persist circuit breaker event
            self._persist_breaker_event(
                db=db,
                breaker_type="daily_loss_limit",
                value=loss_pct,
                metadata={
                    "daily_pnl": daily_pnl,
                    "balance": balance,
                    "limit_amount": -daily_loss_limit_amount,
                    "limit_percent": self.config.daily_loss_limit_percent
                }
            )

            return False, reason, 0.0

        # Check 2: Progressive risk reduction based on consecutive losses
        risk_reduction_factor = self._calculate_risk_reduction(db)

        # All checks passed - trading allowed with potential risk reduction
        return True, None, risk_reduction_factor

    def get_daily_pnl(self, db: Session) -> float:
        """Calculate total P&L for today.

        Checks for persisted circuit breaker events to maintain state across restarts.

        Args:
            db: Database session

        Returns:
            Total P&L in USD for today (can be negative)
        """
        from ...api.database.models import Trade, CircuitBreakerEvent

        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        # Check if breaker was already triggered today
        breaker_event = db.query(CircuitBreakerEvent).filter(
            CircuitBreakerEvent.breaker_type == "daily_loss_limit",
            CircuitBreakerEvent.action == "triggered",
            CircuitBreakerEvent.triggered_at >= today_start,
            CircuitBreakerEvent.recovered_at.is_(None)
        ).first()

        if breaker_event:
            logger.warning(f"Daily loss limit already breached (persisted): {breaker_event.value:.2f}%")
            return breaker_event.value

        # Query closed trades for today
        result = db.query(func.sum(Trade.pnl_usd)).filter(
            Trade.status == "closed",
            Trade.exit_time >= today_start
        ).scalar()

        return result if result is not None else 0.0

    def get_consecutive_losses(self, db: Session) -> int:
        """Count consecutive losing trades from most recent.

        Args:
            db: Database session

        Returns:
            Number of consecutive losing trades
        """
        from ...api.database.models import Trade

        # Get recent closed trades ordered by exit time descending
        recent_trades = db.query(Trade).filter(
            Trade.status == "closed"
        ).order_by(Trade.exit_time.desc()).limit(50).all()

        consecutive_losses = 0
        for trade in recent_trades:
            if trade.pnl_usd is None:
                continue
            if trade.pnl_usd <= 0:
                consecutive_losses += 1
            else:
                # First winning trade breaks the streak
                break

        return consecutive_losses

    def _persist_breaker_event(
        self,
        db: Session,
        breaker_type: str,
        value: float,
        metadata: Optional[dict] = None
    ) -> None:
        """Persist a circuit breaker event to the database.

        Args:
            db: Database session
            breaker_type: Type of breaker (e.g., "daily_loss_limit")
            value: The value that triggered the breaker
            metadata: Additional context
        """
        from ...api.database.models import CircuitBreakerEvent

        try:
            # Check if event already exists for today
            today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            existing_event = db.query(CircuitBreakerEvent).filter(
                CircuitBreakerEvent.breaker_type == breaker_type,
                CircuitBreakerEvent.action == "triggered",
                CircuitBreakerEvent.triggered_at >= today_start,
                CircuitBreakerEvent.recovered_at.is_(None)
            ).first()

            if existing_event:
                logger.debug(f"Circuit breaker event already persisted: {breaker_type}")
                return

            # Create new event
            event = CircuitBreakerEvent(
                breaker_type=breaker_type,
                action="triggered",
                triggered_at=datetime.now(timezone.utc),
                value=value,
                event_metadata=metadata  # Renamed from 'metadata' to match model
            )
            db.add(event)
            db.commit()
            logger.info(f"Persisted circuit breaker event: {breaker_type} (value: {value:.2f})")
        except Exception as e:
            logger.error(f"Failed to persist circuit breaker event: {e}")
            db.rollback()

    def _calculate_risk_reduction(self, db: Session) -> float:
        """Calculate risk reduction factor based on consecutive losses.

        Uses progressive reduction if enabled, otherwise falls back to hard stop.

        Args:
            db: Database session

        Returns:
            Risk reduction factor (0.2 to 1.0)
            - 1.0: Normal risk (< 5 consecutive losses)
            - 0.8: 20% reduction (5 losses)
            - 0.6: 40% reduction (6 losses)
            - 0.4: 60% reduction (7 losses)
            - 0.2: 80% reduction (8+ losses, minimum floor)
        """
        from ...api.database.models import RiskReductionState

        if not self.config.enable_progressive_reduction:
            # Legacy behavior: hard stop at consecutive loss limit
            consecutive_losses = self.get_consecutive_losses(db)
            if consecutive_losses >= self.config.consecutive_loss_limit:
                return 0.0  # Complete stop
            return 1.0  # Normal risk

        try:
            # Get or create risk reduction state
            state = db.query(RiskReductionState).first()
            if state is None:
                # Initialize state on first run
                state = RiskReductionState(
                    consecutive_losses=0,
                    risk_reduction_factor=1.0
                )
                db.add(state)
                db.commit()
                logger.info("Initialized risk reduction state")
                return 1.0

            # Calculate risk reduction factor from consecutive losses
            risk_factor = self._calculate_risk_reduction_from_losses(state.consecutive_losses)

            # Log risk adjustment if reduced
            if risk_factor < 1.0:
                logger.warning(
                    f"Progressive risk reduction active: {state.consecutive_losses} consecutive losses, "
                    f"risk reduced to {risk_factor * 100:.0f}% of normal"
                )

            return risk_factor

        except Exception as e:
            logger.error(f"Failed to calculate risk reduction: {e}", exc_info=True)
            # Fail-safe: return normal risk on error
            return 1.0

    def _calculate_risk_reduction_from_losses(self, consecutive_losses: int) -> float:
        """Calculate risk reduction factor from consecutive loss count.

        Args:
            consecutive_losses: Number of consecutive losing trades

        Returns:
            Risk reduction factor (0.2 to 1.0)
        """
        if consecutive_losses < self.config.consecutive_loss_limit:
            # No reduction until threshold reached
            return 1.0

        # Calculate number of losses beyond threshold
        excess_losses = consecutive_losses - self.config.consecutive_loss_limit + 1

        # Apply 20% reduction per excess loss
        reduction_factor = 1.0 - (excess_losses * self.config.risk_reduction_per_loss)

        # Floor at minimum risk factor (never zero)
        reduction_factor = max(reduction_factor, self.config.min_risk_factor)

        return reduction_factor

    def record_trade_outcome(self, db: Session, trade_id: int, is_winner: bool) -> None:
        """Record trade outcome and update risk reduction state.

        Args:
            db: Database session
            trade_id: Trade ID
            is_winner: True if trade was profitable
        """
        from ...api.database.models import RiskReductionState

        if not self.config.enable_progressive_reduction:
            return  # Skip if progressive reduction disabled

        try:
            # Get or create state
            state = db.query(RiskReductionState).first()
            if state is None:
                state = RiskReductionState(
                    consecutive_losses=0,
                    risk_reduction_factor=1.0
                )
                db.add(state)

            # Update consecutive loss count
            if is_winner:
                # Winning trade: reduce consecutive losses (recovery)
                if state.consecutive_losses > 0:
                    state.consecutive_losses = max(0, state.consecutive_losses - 1)
                    logger.info(
                        f"Winning trade recorded. Consecutive losses reduced to {state.consecutive_losses}"
                    )
            else:
                # Losing trade: increment consecutive losses
                state.consecutive_losses += 1
                logger.warning(
                    f"Losing trade recorded. Consecutive losses increased to {state.consecutive_losses}"
                )

            # Recalculate risk reduction factor
            state.risk_reduction_factor = self._calculate_risk_reduction_from_losses(
                state.consecutive_losses
            )
            state.last_trade_id = trade_id
            state.updated_at = datetime.now(timezone.utc)

            db.commit()

            # Log significant state changes
            if state.risk_reduction_factor < 1.0:
                logger.warning(
                    f"Risk reduction updated: factor={state.risk_reduction_factor:.2f}, "
                    f"consecutive_losses={state.consecutive_losses}"
                )

        except Exception as e:
            logger.error(f"Failed to record trade outcome: {e}", exc_info=True)
            db.rollback()

    def get_monthly_drawdown(self, db: Session, initial_balance: float) -> float:
        """Calculate current month's drawdown (future enhancement).

        Args:
            db: Database session
            initial_balance: Starting balance for the month

        Returns:
            Drawdown percentage for current month
        """
        # Get start of current month
        now = datetime.now(timezone.utc)
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Calculate P&L for current month
        result = db.query(func.sum(Trade.pnl_usd)).filter(
            Trade.status == "closed",
            Trade.exit_time >= month_start
        ).scalar()

        monthly_pnl = result if result is not None else 0.0
        current_balance = initial_balance + monthly_pnl

        # Calculate peak balance for the month (simplified - assumes peak = initial)
        # TODO: Track peak balance properly
        peak_balance = max(initial_balance, current_balance)

        drawdown_pct = ((peak_balance - current_balance) / peak_balance * 100.0) if peak_balance > 0 else 0.0

        return drawdown_pct
