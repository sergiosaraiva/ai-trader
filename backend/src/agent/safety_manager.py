"""Safety Manager - coordinates all safety mechanisms for the trading agent.

Integrates circuit breakers, kill switch, and risk management to protect
capital from catastrophic losses.
"""

import asyncio
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass
from sqlalchemy.orm import Session

from .safety_config import SafetyConfig
from ..trading.circuit_breakers.manager import CircuitBreakerManager
from ..trading.circuit_breakers.base import TradeResult, TradingState, CircuitBreakerState
from ..trading.safety.kill_switch import KillSwitch, KillSwitchConfig, TriggerType
from ..trading.risk.profiles import RiskProfile, RiskLevel
from ..api.database.models import CircuitBreakerEvent

logger = logging.getLogger(__name__)


@dataclass
class SafetyStatus:
    """Current safety status from all mechanisms."""

    # Overall safety state
    is_safe_to_trade: bool
    circuit_breaker_triggered: bool
    kill_switch_active: bool

    # Circuit breaker state
    circuit_breaker_state: str  # "active", "reduced", "halted", "recovering"
    active_breakers: list
    breaker_reasons: list
    size_multiplier: float
    min_confidence_override: Optional[float]

    # Kill switch state
    kill_switch_reason: Optional[str]
    kill_switch_trigger_time: Optional[datetime]

    # Daily counters
    daily_trades: int
    daily_loss_pct: float
    daily_loss_amount: float

    # Account metrics
    current_equity: float
    peak_equity: float
    current_drawdown_pct: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "is_safe_to_trade": self.is_safe_to_trade,
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
            "kill_switch_active": self.kill_switch_active,
            "circuit_breaker_state": self.circuit_breaker_state,
            "active_breakers": self.active_breakers,
            "breaker_reasons": self.breaker_reasons,
            "size_multiplier": self.size_multiplier,
            "min_confidence_override": self.min_confidence_override,
            "kill_switch_reason": self.kill_switch_reason,
            "kill_switch_trigger_time": (
                self.kill_switch_trigger_time.isoformat()
                if self.kill_switch_trigger_time
                else None
            ),
            "daily_trades": self.daily_trades,
            "daily_loss_pct": self.daily_loss_pct,
            "daily_loss_amount": self.daily_loss_amount,
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "current_drawdown_pct": self.current_drawdown_pct,
        }


class SafetyManager:
    """Coordinates all safety mechanisms for the trading agent.

    Integrates:
    - Circuit breakers (consecutive loss, drawdown, model degradation)
    - Kill switch (emergency stop)
    - Risk limits (daily loss, trade count)
    - Database audit trail

    Usage:
        safety_manager = SafetyManager(config, initial_equity=100000, db_session_factory)

        # Check before each trade
        status = safety_manager.check_safety(
            current_equity=current_equity,
            is_broker_connected=broker.is_connected()
        )

        if not status.is_safe_to_trade:
            logger.warning(f"Trading not safe: {status.breaker_reasons}")
            return

        # Record trade results
        safety_manager.record_trade_result(trade_result)

        # Emergency stop
        safety_manager.trigger_kill_switch("Manual stop requested")
    """

    def __init__(
        self,
        config: SafetyConfig,
        initial_equity: float,
        db_session_factory: Callable[[], Session],
    ):
        """Initialize safety manager.

        Args:
            config: Safety configuration
            initial_equity: Starting account equity
            db_session_factory: Factory function to create database sessions
        """
        self.config = config
        self.initial_equity = initial_equity
        self.db_session_factory = db_session_factory

        # Create moderate risk profile for circuit breaker manager
        risk_profile = RiskProfile(
            name="Agent Safety",
            description="Safety profile for agent circuit breakers",
            level=RiskLevel.MODERATE,
            min_confidence_to_trade=0.65,
            full_position_confidence=0.85,
            max_position_pct=0.05,
            base_position_pct=0.02,
            kelly_fraction=0.50,
            max_daily_loss_pct=config.max_daily_loss_percent,
            max_weekly_loss_pct=config.max_daily_loss_percent * 2,
            max_drawdown_pct=config.max_drawdown_percent,
            consecutive_loss_halt=config.max_consecutive_losses,
            cooldown_hours=12,
            max_portfolio_heat=0.10,
            max_correlation_exposure=0.06,
            max_positions=5,
            max_trades_per_day=config.max_daily_trades,
            min_trade_interval_seconds=300,
        )

        # Initialize circuit breaker manager
        self.circuit_breaker_manager = CircuitBreakerManager(
            risk_profile=risk_profile,
            initial_equity=initial_equity,
        )

        # Initialize kill switch
        kill_switch_config = KillSwitchConfig(
            max_daily_loss_pct=config.max_daily_loss_percent,
            max_daily_loss_amount=config.max_daily_loss_amount,
            max_daily_trades=config.max_daily_trades,
            max_trades_per_hour=config.max_trades_per_hour,
            max_disconnection_seconds=config.max_disconnection_seconds,
            auto_reset_next_day=config.auto_reset_next_day,
            require_authorization_code=config.require_token_for_reset,
        )

        self.kill_switch = KillSwitch(config=kill_switch_config)

        # Thread-safe lock for daily counters and equity tracking
        self._lock = threading.Lock()

        # Daily counters (protected by _lock)
        self._daily_trades = 0
        self._daily_start_equity = initial_equity
        self._current_equity = initial_equity

        logger.info(
            f"SafetyManager initialized: "
            f"initial_equity=${initial_equity:,.2f}, "
            f"config={config}"
        )

    def check_safety(
        self,
        current_equity: Optional[float] = None,
        is_broker_connected: bool = True,
        confidence: Optional[float] = None,
        ensemble_agreement: Optional[float] = None,
    ) -> SafetyStatus:
        """Check all safety mechanisms before trading.

        This should be called BEFORE every trade decision. If is_safe_to_trade
        is False, do not execute the trade.

        Args:
            current_equity: Current account equity
            is_broker_connected: Whether broker is connected
            confidence: Current prediction confidence
            ensemble_agreement: Model ensemble agreement

        Returns:
            SafetyStatus with detailed safety information
        """
        with self._lock:
            # Update equity tracking
            if current_equity is not None:
                self._current_equity = current_equity

            # Calculate daily loss metrics
            daily_loss_amount = self._daily_start_equity - self._current_equity
            daily_loss_pct = (
                (daily_loss_amount / self._daily_start_equity * 100)
                if self._daily_start_equity > 0
                else 0.0
            )
            daily_trades = self._daily_trades
            current_equity_val = self._current_equity

        # 1. Check kill switch first (highest priority)
        kill_switch_triggered = self.kill_switch.check_all(
            daily_loss_pct=daily_loss_pct,
            daily_loss_amount=daily_loss_amount,
            trade_count=daily_trades,
            is_connected=is_broker_connected,
        )

        if kill_switch_triggered or self.kill_switch.is_active:
            return SafetyStatus(
                is_safe_to_trade=False,
                circuit_breaker_triggered=False,
                kill_switch_active=True,
                circuit_breaker_state="active",
                active_breakers=["kill_switch"],
                breaker_reasons=[
                    self.kill_switch.state.trigger.reason
                    if self.kill_switch.state.trigger
                    else "Kill switch active"
                ],
                size_multiplier=0.0,
                min_confidence_override=None,
                kill_switch_reason=(
                    self.kill_switch.state.trigger.reason
                    if self.kill_switch.state.trigger
                    else None
                ),
                kill_switch_trigger_time=(
                    self.kill_switch.state.activated_at
                    if self.kill_switch.state.activated_at
                    else None
                ),
                daily_trades=daily_trades,
                daily_loss_pct=daily_loss_pct,
                daily_loss_amount=daily_loss_amount,
                current_equity=current_equity_val,
                peak_equity=self.circuit_breaker_manager.breakers[
                    "drawdown"
                ].peak_equity,
                current_drawdown_pct=self.circuit_breaker_manager.breakers[
                    "drawdown"
                ].current_drawdown_pct
                * 100,
            )

        # 2. Check circuit breakers
        cb_state = self.circuit_breaker_manager.check_all(
            current_equity=current_equity_val,
            ensemble_agreement=ensemble_agreement,
            confidence=confidence,
        )

        # Determine if safe to trade
        is_safe = cb_state.can_trade

        # Get drawdown info
        drawdown_breaker = self.circuit_breaker_manager.breakers["drawdown"]
        current_drawdown_pct = drawdown_breaker.current_drawdown_pct * 100

        return SafetyStatus(
            is_safe_to_trade=is_safe,
            circuit_breaker_triggered=cb_state.is_halted,
            kill_switch_active=False,
            circuit_breaker_state=cb_state.overall_state.value,
            active_breakers=cb_state.active_breakers,
            breaker_reasons=cb_state.reasons,
            size_multiplier=cb_state.size_multiplier,
            min_confidence_override=cb_state.min_confidence_override,
            kill_switch_reason=None,
            kill_switch_trigger_time=None,
            daily_trades=daily_trades,
            daily_loss_pct=daily_loss_pct,
            daily_loss_amount=daily_loss_amount,
            current_equity=current_equity_val,
            peak_equity=drawdown_breaker.peak_equity,
            current_drawdown_pct=current_drawdown_pct,
        )

    def record_trade_result(self, trade_result: TradeResult) -> None:
        """Update safety mechanisms with trade outcome.

        This should be called AFTER each trade is closed with the final result.

        Args:
            trade_result: Completed trade result
        """
        with self._lock:
            # Update daily counter
            self._daily_trades += 1

            # Update current equity
            self._current_equity += trade_result.pnl

            # Capture values for logging outside lock
            daily_trades = self._daily_trades
            current_equity = self._current_equity

        # Record on circuit breakers (outside lock - has its own thread safety)
        self.circuit_breaker_manager.record_trade(trade_result)

        # Log to database if trade triggered any breakers
        cb_state = self.circuit_breaker_manager.current_state
        if cb_state.active_breakers:
            self._log_event(
                event_type="trade_result",
                breaker_type=",".join(cb_state.active_breakers),
                severity="warning" if cb_state.can_trade else "critical",
                action="triggered" if not cb_state.can_trade else "warning",
                reason="; ".join(cb_state.reasons),
                value=trade_result.pnl,
            )

        logger.debug(
            f"Trade recorded: pnl={trade_result.pnl:.2f}, "
            f"daily_trades={daily_trades}, "
            f"equity={current_equity:.2f}"
        )

    def trigger_kill_switch(self, reason: str) -> None:
        """Trigger emergency stop.

        Args:
            reason: Reason for kill switch activation
        """
        self.kill_switch.trigger(reason=reason, trigger_type=TriggerType.MANUAL)

        # Log to database
        self._log_event(
            event_type="kill_switch",
            breaker_type="kill_switch",
            severity="critical",
            action="triggered",
            reason=reason,
        )

        logger.critical(f"Kill switch triggered: {reason}")

    def reset_kill_switch(self, authorization: str = "", force: bool = False) -> bool:
        """Reset kill switch after review.

        Args:
            authorization: Authorization code from get_reset_code()
            force: Force reset without authorization (dangerous!)

        Returns:
            True if reset successful, False otherwise
        """
        success = self.kill_switch.reset(authorization=authorization, force=force)

        if success:
            self._log_event(
                event_type="kill_switch",
                breaker_type="kill_switch",
                severity="warning",
                action="reset",
                reason="Kill switch reset by operator",
            )
            logger.info("Kill switch reset successfully")
        else:
            logger.warning("Kill switch reset failed (invalid authorization)")

        return success

    def get_reset_code(self) -> str:
        """Generate authorization code for kill switch reset.

        Returns:
            Authorization code (valid for 5 minutes)
        """
        return self.kill_switch.get_reset_code()

    def reset_circuit_breaker(self, breaker_name: str) -> bool:
        """Reset a specific circuit breaker.

        Args:
            breaker_name: Name of breaker to reset

        Returns:
            True if reset successful, False otherwise
        """
        if breaker_name not in self.circuit_breaker_manager.breakers:
            logger.warning(f"Unknown circuit breaker: {breaker_name}")
            return False

        breaker = self.circuit_breaker_manager.breakers[breaker_name]
        breaker.reset()

        self._log_event(
            event_type="circuit_breaker",
            breaker_type=breaker_name,
            severity="warning",
            action="reset",
            reason=f"Circuit breaker '{breaker_name}' reset by operator",
        )

        logger.info(f"Circuit breaker '{breaker_name}' reset")
        return True

    def reset_daily_counters(self) -> None:
        """Reset daily counters at start of new trading day.

        Thread-safe: uses lock to prevent race conditions.
        """
        with self._lock:
            self._daily_trades = 0
            self._daily_start_equity = self._current_equity
            current_equity = self._current_equity

        self.kill_switch.reset_daily_counters()

        logger.info(
            f"Daily counters reset: start_equity=${current_equity:,.2f}"
        )

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive safety status for API/frontend.

        Thread-safe: uses lock to capture consistent snapshot of counters.

        Returns:
            Dictionary with all safety information
        """
        cb_status = self.circuit_breaker_manager.get_status()
        ks_stats = self.kill_switch.get_stats()

        with self._lock:
            daily_loss_amount = self._daily_start_equity - self._current_equity
            daily_loss_pct = (
                (daily_loss_amount / self._daily_start_equity * 100)
                if self._daily_start_equity > 0
                else 0.0
            )
            daily_trades = self._daily_trades
            daily_start_equity = self._daily_start_equity
            current_equity = self._current_equity

        return {
            "is_safe_to_trade": (
                cb_status["can_trade"] and not ks_stats["is_active"]
            ),
            "circuit_breakers": cb_status,
            "kill_switch": ks_stats,
            "daily_metrics": {
                "trades": daily_trades,
                "loss_pct": daily_loss_pct,
                "loss_amount": daily_loss_amount,
                "start_equity": daily_start_equity,
            },
            "account_metrics": {
                "current_equity": current_equity,
                "peak_equity": self.circuit_breaker_manager.breakers[
                    "drawdown"
                ].peak_equity,
                "drawdown_pct": self.circuit_breaker_manager.breakers[
                    "drawdown"
                ].current_drawdown_pct
                * 100,
            },
        }

    async def _log_event_async(
        self,
        event_type: str,
        breaker_type: str,
        severity: str,
        action: str,
        reason: str,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> None:
        """Async version of _log_event that doesn't block event loop.

        Args:
            event_type: Type of event (trade_result, kill_switch, circuit_breaker)
            breaker_type: Which breaker triggered (consecutive_loss, drawdown, etc.)
            severity: Event severity (warning, critical)
            action: Action taken (triggered, recovered, reset, warning)
            reason: Human-readable reason
            value: Current value that triggered
            threshold: Threshold that was exceeded
        """
        await asyncio.to_thread(
            self._log_event,
            event_type=event_type,
            breaker_type=breaker_type,
            severity=severity,
            action=action,
            reason=reason,
            value=value,
            threshold=threshold,
        )

    def _log_event(
        self,
        event_type: str,
        breaker_type: str,
        severity: str,
        action: str,
        reason: str,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> None:
        """Log safety event to database (synchronous).

        This is the synchronous implementation. Use _log_event_async() from
        async contexts to avoid blocking the event loop.

        Args:
            event_type: Type of event (trade_result, kill_switch, circuit_breaker)
            breaker_type: Which breaker triggered (consecutive_loss, drawdown, etc.)
            severity: Event severity (warning, critical)
            action: Action taken (triggered, recovered, reset, warning)
            reason: Human-readable reason
            value: Current value that triggered
            threshold: Threshold that was exceeded
        """
        session = None
        try:
            session = self.db_session_factory()

            event = CircuitBreakerEvent(
                breaker_type=breaker_type,
                severity=severity,
                action=action,
                reason=reason,
                value=value,
                threshold=threshold,
                triggered_at=datetime.utcnow(),
            )

            session.add(event)
            session.commit()

            logger.debug(
                f"Safety event logged: {breaker_type}/{severity}/{action} - {reason}"
            )

        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Failed to log safety event: {e}")
        finally:
            if session:
                session.close()
