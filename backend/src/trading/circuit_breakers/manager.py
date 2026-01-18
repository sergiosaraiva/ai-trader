"""
Circuit Breaker Manager.

Coordinates all circuit breakers and provides unified state management.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

from .base import (
    CircuitBreaker,
    CircuitBreakerAction,
    CircuitBreakerState,
    TradeResult,
    TradingState,
    Severity,
    RecoveryRequirement,
)
from .consecutive_loss import ConsecutiveLossBreaker
from .drawdown import DrawdownBreaker
from .model_degradation import ModelDegradationBreaker
from ..risk.profiles import RiskProfile


class RecoveryPhase(Enum):
    """Recovery phases after circuit breaker trigger."""
    COOLDOWN = "cooldown"      # No trading, waiting
    TESTING = "testing"        # Minimal size, validating
    GRADUATED = "graduated"    # Increasing size with wins
    RESTORED = "restored"      # Full trading restored


@dataclass
class RecoveryState:
    """Current recovery state."""
    phase: RecoveryPhase
    can_trade: bool
    size_multiplier: float = 1.0
    time_remaining: Optional[timedelta] = None
    progress: str = ""
    message: str = ""


@dataclass
class RecoveryProtocol:
    """
    Manages recovery after circuit breaker activation.

    RECOVERY PHASES:
    1. COOLDOWN: No trading, monitor only
    2. TESTING: Reduced position sizes, validate model
    3. GRADUATED: Increasing position sizes with wins
    4. RESTORED: Full trading restored
    """

    # Configuration
    cooldown_end: datetime = None
    testing_required_trades: int = 5
    graduated_wins_needed: int = 3

    # Current phase
    current_phase: RecoveryPhase = RecoveryPhase.RESTORED

    # Testing phase tracking
    testing_trades: int = 0
    testing_wins: int = 0

    # Graduated phase tracking
    graduated_consecutive_wins: int = 0
    current_size_multiplier: float = 1.0

    def start_recovery(self, requirement: RecoveryRequirement) -> None:
        """Start recovery protocol."""
        self.current_phase = RecoveryPhase.COOLDOWN
        self.cooldown_end = datetime.now() + timedelta(hours=requirement.cooldown_hours)
        self.testing_required_trades = 5
        self.graduated_wins_needed = requirement.wins_to_restore
        self.current_size_multiplier = requirement.reduced_size_on_resume

        # Reset tracking
        self.testing_trades = 0
        self.testing_wins = 0
        self.graduated_consecutive_wins = 0

    def update(self, trade_result: Optional[TradeResult] = None) -> RecoveryState:
        """Update recovery state after trade or time passage."""

        # COOLDOWN phase
        if self.current_phase == RecoveryPhase.COOLDOWN:
            if datetime.now() >= self.cooldown_end:
                self.current_phase = RecoveryPhase.TESTING
                return RecoveryState(
                    phase=RecoveryPhase.TESTING,
                    can_trade=True,
                    size_multiplier=0.10,
                    message="Entering testing phase with minimal position sizes"
                )
            return RecoveryState(
                phase=RecoveryPhase.COOLDOWN,
                can_trade=False,
                time_remaining=self.cooldown_end - datetime.now(),
                message=f"In cooldown until {self.cooldown_end.strftime('%Y-%m-%d %H:%M')}"
            )

        # TESTING phase
        if self.current_phase == RecoveryPhase.TESTING:
            if trade_result:
                self.testing_trades += 1
                if trade_result.is_win:
                    self.testing_wins += 1

            if self.testing_trades >= self.testing_required_trades:
                win_rate = self.testing_wins / self.testing_trades if self.testing_trades > 0 else 0

                if win_rate >= 0.5:  # At least 50% win rate
                    self.current_phase = RecoveryPhase.GRADUATED
                    self.current_size_multiplier = 0.25
                    return RecoveryState(
                        phase=RecoveryPhase.GRADUATED,
                        can_trade=True,
                        size_multiplier=0.25,
                        message=f"Testing passed ({win_rate:.0%}), entering graduated recovery"
                    )
                else:
                    # Back to cooldown
                    self.cooldown_end = datetime.now() + timedelta(hours=24)
                    self.current_phase = RecoveryPhase.COOLDOWN
                    self.testing_trades = 0
                    self.testing_wins = 0
                    return RecoveryState(
                        phase=RecoveryPhase.COOLDOWN,
                        can_trade=False,
                        message=f"Testing failed ({win_rate:.0%}), returning to 24h cooldown"
                    )

            return RecoveryState(
                phase=RecoveryPhase.TESTING,
                can_trade=True,
                size_multiplier=0.10,
                progress=f"{self.testing_trades}/{self.testing_required_trades} test trades"
            )

        # GRADUATED phase
        if self.current_phase == RecoveryPhase.GRADUATED:
            if trade_result:
                if trade_result.is_win:
                    self.graduated_consecutive_wins += 1
                    # Increase size multiplier
                    self.current_size_multiplier = min(1.0, self.current_size_multiplier + 0.25)
                else:
                    self.graduated_consecutive_wins = 0
                    # Decrease size multiplier
                    self.current_size_multiplier = max(0.25, self.current_size_multiplier - 0.25)

            if self.graduated_consecutive_wins >= self.graduated_wins_needed:
                self.current_phase = RecoveryPhase.RESTORED
                self.current_size_multiplier = 1.0
                return RecoveryState(
                    phase=RecoveryPhase.RESTORED,
                    can_trade=True,
                    size_multiplier=1.0,
                    message="Full trading restored!"
                )

            return RecoveryState(
                phase=RecoveryPhase.GRADUATED,
                can_trade=True,
                size_multiplier=self.current_size_multiplier,
                progress=f"{self.graduated_consecutive_wins}/{self.graduated_wins_needed} consecutive wins"
            )

        # RESTORED
        return RecoveryState(
            phase=RecoveryPhase.RESTORED,
            can_trade=True,
            size_multiplier=1.0
        )

    @property
    def is_recovering(self) -> bool:
        """Check if in recovery mode."""
        return self.current_phase != RecoveryPhase.RESTORED


class CircuitBreakerManager:
    """
    Coordinates all circuit breakers.

    Aggregates state from multiple breakers and provides
    unified interface for the trading robot.
    """

    def __init__(
        self,
        risk_profile: RiskProfile,
        initial_equity: float = 100000,
    ):
        """
        Initialize circuit breaker manager.

        Args:
            risk_profile: Risk profile with breaker configuration
            initial_equity: Starting account equity
        """
        self.risk_profile = risk_profile
        self.initial_equity = initial_equity

        # Initialize breakers
        self.breakers: Dict[str, CircuitBreaker] = {
            'consecutive_loss': ConsecutiveLossBreaker(
                max_consecutive_losses=risk_profile.consecutive_loss_halt,
                base_cooldown_hours=risk_profile.cooldown_hours,
            ),
            'drawdown': DrawdownBreaker(
                max_drawdown_pct=risk_profile.max_drawdown_pct,
            ),
            'model_degradation': ModelDegradationBreaker(
                min_rolling_accuracy=0.45,
                rolling_window=20,
            ),
        }

        # Initialize drawdown breaker with equity
        self.breakers['drawdown'].initialize(initial_equity)

        # Recovery protocol
        self.recovery = RecoveryProtocol()

        # State tracking
        self.current_state = CircuitBreakerState(
            overall_state=TradingState.ACTIVE
        )
        self.trigger_history: List[Dict[str, Any]] = []

        # Manual halt flag
        self._manual_halt = False
        self._manual_halt_reason = ""

    def check_all(
        self,
        current_equity: float = None,
        ensemble_agreement: float = None,
        confidence: float = None,
        **kwargs
    ) -> CircuitBreakerState:
        """
        Check all circuit breakers and return aggregate state.

        Args:
            current_equity: Current account equity
            ensemble_agreement: Model ensemble agreement
            confidence: Current prediction confidence

        Returns:
            CircuitBreakerState with aggregate information
        """
        # Check for manual halt first
        if self._manual_halt:
            self.current_state = CircuitBreakerState(
                overall_state=TradingState.HALTED,
                active_breakers=['manual'],
                reasons=[self._manual_halt_reason],
                size_multiplier=0.0,
            )
            return self.current_state

        # Then check if in recovery
        if self.recovery.is_recovering:
            recovery_state = self.recovery.update()
            if not recovery_state.can_trade:
                self.current_state = CircuitBreakerState(
                    overall_state=TradingState.HALTED,
                    active_breakers=['recovery'],
                    reasons=[recovery_state.message],
                    size_multiplier=0.0,
                )
                return self.current_state

            # In recovery but can trade - apply recovery multiplier
            base_multiplier = recovery_state.size_multiplier
        else:
            base_multiplier = 1.0

        # Update drawdown breaker if equity provided
        if current_equity is not None:
            self.breakers['drawdown'].update_equity(current_equity)

        # Check all breakers
        active_breakers = []
        reasons = []
        most_severe_action = CircuitBreakerAction(action=TradingState.ACTIVE)
        min_size_multiplier = base_multiplier
        highest_confidence_override = None

        for name, breaker in self.breakers.items():
            action = breaker.check(
                confidence=confidence,
                ensemble_agreement=ensemble_agreement,
            )

            if action.action != TradingState.ACTIVE:
                active_breakers.append(name)
                reasons.append(action.reason)

                # Track most severe action
                if self._is_more_severe(action, most_severe_action):
                    most_severe_action = action

                # Track minimum size multiplier
                if action.size_multiplier is not None:
                    min_size_multiplier = min(min_size_multiplier, action.size_multiplier)

                # Track highest confidence override
                if action.min_confidence_override is not None:
                    if highest_confidence_override is None:
                        highest_confidence_override = action.min_confidence_override
                    else:
                        highest_confidence_override = max(
                            highest_confidence_override,
                            action.min_confidence_override
                        )

        # Build aggregate state
        self.current_state = CircuitBreakerState(
            overall_state=most_severe_action.action,
            active_breakers=active_breakers,
            reasons=reasons,
            size_multiplier=min_size_multiplier,
            min_confidence_override=highest_confidence_override,
        )

        # Start recovery if halted
        if most_severe_action.action == TradingState.HALTED:
            if most_severe_action.recovery_requirement and not self.recovery.is_recovering:
                self.recovery.start_recovery(most_severe_action.recovery_requirement)
                self._log_trigger(most_severe_action)

        return self.current_state

    def record_trade(self, trade: TradeResult) -> CircuitBreakerState:
        """
        Record trade result and update all breakers.

        Args:
            trade: Completed trade result

        Returns:
            Updated CircuitBreakerState
        """
        # Record on all breakers
        most_severe_action = CircuitBreakerAction(action=TradingState.ACTIVE)

        for name, breaker in self.breakers.items():
            action = breaker.record_trade(trade)
            if self._is_more_severe(action, most_severe_action):
                most_severe_action = action

        # Update recovery if in recovery mode
        if self.recovery.is_recovering:
            self.recovery.update(trade)

        # Return current state after checks
        return self.check_all()

    def _is_more_severe(
        self,
        action1: CircuitBreakerAction,
        action2: CircuitBreakerAction
    ) -> bool:
        """Compare severity of two actions."""
        severity_order = {
            TradingState.ACTIVE: 0,
            TradingState.REDUCED: 1,
            TradingState.RECOVERING: 2,
            TradingState.HALTED: 3,
        }
        return severity_order.get(action1.action, 0) > severity_order.get(action2.action, 0)

    def _log_trigger(self, action: CircuitBreakerAction) -> None:
        """Log circuit breaker trigger."""
        self.trigger_history.append({
            'timestamp': datetime.now(),
            'action': action.action.value,
            'reason': action.reason,
            'severity': action.severity.value,
        })

    def force_halt(self, reason: str) -> None:
        """Force trading halt (manual kill switch)."""
        self._manual_halt = True
        self._manual_halt_reason = reason
        self.current_state = CircuitBreakerState(
            overall_state=TradingState.HALTED,
            active_breakers=['manual'],
            reasons=[reason],
            size_multiplier=0.0,
        )
        self._log_trigger(CircuitBreakerAction(
            action=TradingState.HALTED,
            reason=f"Manual halt: {reason}",
            severity=Severity.CRITICAL,
        ))

    def force_resume(self) -> None:
        """Force resume trading (use with caution)."""
        self._manual_halt = False
        self._manual_halt_reason = ""
        self.recovery = RecoveryProtocol()
        for breaker in self.breakers.values():
            breaker.reset()
        self.current_state = CircuitBreakerState(overall_state=TradingState.ACTIVE)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        return {
            'overall_state': self.current_state.overall_state.value,
            'can_trade': self.current_state.can_trade,
            'size_multiplier': self.current_state.size_multiplier,
            'min_confidence_override': self.current_state.min_confidence_override,
            'active_breakers': self.current_state.active_breakers,
            'reasons': self.current_state.reasons,
            'recovery_phase': self.recovery.current_phase.value if self.recovery.is_recovering else None,
            'breaker_stats': {
                name: breaker.get_stats() for name, breaker in self.breakers.items()
            },
            'trigger_history_count': len(self.trigger_history),
            'recent_triggers': self.trigger_history[-5:] if self.trigger_history else [],
        }
