"""
Consecutive Loss Circuit Breaker.

Halts trading after N consecutive losses to prevent
continued losses when the model is consistently wrong.
"""

from dataclasses import dataclass, field
from typing import List
from datetime import datetime

from .base import (
    CircuitBreaker,
    CircuitBreakerAction,
    TradeResult,
    TradingState,
    Severity,
    RecoveryRequirement,
)


@dataclass
class ConsecutiveLossBreaker(CircuitBreaker):
    """
    Halts trading after N consecutive losses.

    RATIONALE:
    If the model is correct ~55% of the time, the probability of
    N consecutive losses is (0.45)^N:

    | N Losses | Probability | Should Happen Every |
    |----------|-------------|---------------------|
    | 3        | 9.1%        | 11 trades           |
    | 4        | 4.1%        | 25 trades           |
    | 5        | 1.8%        | 55 trades           |
    | 6        | 0.8%        | 120 trades          |
    | 7        | 0.4%        | 270 trades          |

    For a CONSERVATIVE profile (3 loss halt), we expect false triggers
    about 1 in 11 trades - acceptable for capital protection.

    For MODERATE profile (5 loss halt), false triggers 1 in 55 trades.
    """

    max_consecutive_losses: int = 5
    base_cooldown_hours: int = 12
    consecutive_losses: int = 0
    trade_history: List[TradeResult] = field(default_factory=list)
    trigger_time: datetime = None

    @property
    def name(self) -> str:
        return "ConsecutiveLossBreaker"

    def check(self, **kwargs) -> CircuitBreakerAction:
        """Check current state without recording new trade."""
        if self.consecutive_losses >= self.max_consecutive_losses:
            return CircuitBreakerAction(
                action=TradingState.HALTED,
                reason=f"Consecutive loss limit reached: {self.consecutive_losses}",
                severity=Severity.HIGH,
                recovery_requirement=RecoveryRequirement(
                    cooldown_hours=self._get_cooldown_hours(),
                    reduced_size_on_resume=0.5,
                    wins_to_restore=3,
                )
            )
        return CircuitBreakerAction(action=TradingState.ACTIVE)

    def record_trade(self, trade: TradeResult) -> CircuitBreakerAction:
        """
        Record trade result and check for trigger.

        Args:
            trade: Completed trade result

        Returns:
            CircuitBreakerAction indicating if trading should halt
        """
        self.trade_history.append(trade)

        if trade.is_loss:
            self.consecutive_losses += 1

            if self.consecutive_losses >= self.max_consecutive_losses:
                self.trigger_time = datetime.now()
                return CircuitBreakerAction(
                    action=TradingState.HALTED,
                    reason=f"Consecutive loss limit reached: {self.consecutive_losses}",
                    severity=Severity.HIGH,
                    recovery_requirement=RecoveryRequirement(
                        cooldown_hours=self._get_cooldown_hours(),
                        reduced_size_on_resume=0.5,
                        wins_to_restore=3,
                    )
                )

            # Warning when approaching limit
            if self.consecutive_losses >= self.max_consecutive_losses - 1:
                return CircuitBreakerAction(
                    action=TradingState.REDUCED,
                    reason=f"Approaching consecutive loss limit: {self.consecutive_losses}/{self.max_consecutive_losses}",
                    severity=Severity.MEDIUM,
                    size_multiplier=0.5,
                    min_confidence_override=0.80,
                )
        else:
            # Win resets the counter
            self.consecutive_losses = 0

        return CircuitBreakerAction(action=TradingState.ACTIVE)

    def _get_cooldown_hours(self) -> int:
        """
        Calculate cooldown hours based on severity.

        Longer streaks get longer cooldowns.
        """
        extra_losses = max(0, self.consecutive_losses - self.max_consecutive_losses)
        return self.base_cooldown_hours * (1 + extra_losses)

    def reset(self) -> None:
        """Reset breaker state."""
        self.consecutive_losses = 0
        self.trade_history.clear()
        self.trigger_time = None

    def get_stats(self) -> dict:
        """Get current breaker statistics."""
        return {
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_losses': self.max_consecutive_losses,
            'total_trades': len(self.trade_history),
            'trigger_time': self.trigger_time,
            'is_triggered': self.consecutive_losses >= self.max_consecutive_losses,
        }
