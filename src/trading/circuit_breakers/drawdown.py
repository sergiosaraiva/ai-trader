"""
Drawdown Protection Circuit Breaker.

Provides progressive protection as drawdown increases,
reducing position sizes and eventually halting trading.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .base import (
    CircuitBreaker,
    CircuitBreakerAction,
    TradingState,
    Severity,
    RecoveryRequirement,
)


@dataclass
class DrawdownBreaker(CircuitBreaker):
    """
    Progressive protection as drawdown increases.

    STRATEGY:
    - 50% of limit: Reduce position sizes to 50%
    - 75% of limit: Only very high confidence trades, 25% size
    - 100% of limit: Full halt

    This provides graduated protection rather than sudden stops,
    giving the model a chance to recover while limiting damage.
    """

    max_drawdown_pct: float = 0.15  # 15% default max drawdown
    peak_equity: float = 0.0
    initial_equity: float = 0.0
    current_equity: float = 0.0
    last_update: datetime = None

    @property
    def name(self) -> str:
        return "DrawdownBreaker"

    def initialize(self, initial_equity: float) -> None:
        """Initialize with starting equity."""
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        self.peak_equity = initial_equity
        self.last_update = datetime.now()

    def update_equity(self, current_equity: float) -> CircuitBreakerAction:
        """
        Update equity and check drawdown.

        Args:
            current_equity: Current account equity

        Returns:
            CircuitBreakerAction based on drawdown level
        """
        self.current_equity = current_equity
        self.last_update = datetime.now()

        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        return self.check()

    def check(self, **kwargs) -> CircuitBreakerAction:
        """Check current drawdown and return appropriate action."""
        if self.peak_equity <= 0:
            return CircuitBreakerAction(action=TradingState.ACTIVE)

        # Calculate current drawdown
        drawdown_pct = (self.peak_equity - self.current_equity) / self.peak_equity
        drawdown_ratio = drawdown_pct / self.max_drawdown_pct

        # 100% of limit - HALT
        if drawdown_ratio >= 1.0:
            return CircuitBreakerAction(
                action=TradingState.HALTED,
                reason=f"Maximum drawdown reached: {drawdown_pct:.2%} (limit: {self.max_drawdown_pct:.2%})",
                severity=Severity.CRITICAL,
                recovery_requirement=RecoveryRequirement(
                    cooldown_hours=72,
                    reduced_size_on_resume=0.25,
                    wins_to_restore=5,
                )
            )

        # 75% of limit - Very reduced trading
        if drawdown_ratio >= 0.75:
            return CircuitBreakerAction(
                action=TradingState.REDUCED,
                reason=f"Drawdown at {drawdown_pct:.2%} ({drawdown_ratio:.0%} of limit)",
                severity=Severity.HIGH,
                size_multiplier=0.25,
                min_confidence_override=0.85,  # Only very confident trades
            )

        # 50% of limit - Reduced trading
        if drawdown_ratio >= 0.50:
            return CircuitBreakerAction(
                action=TradingState.REDUCED,
                reason=f"Drawdown at {drawdown_pct:.2%} ({drawdown_ratio:.0%} of limit)",
                severity=Severity.MEDIUM,
                size_multiplier=0.50,
            )

        # 25% of limit - Warning
        if drawdown_ratio >= 0.25:
            return CircuitBreakerAction(
                action=TradingState.REDUCED,
                reason=f"Drawdown warning: {drawdown_pct:.2%}",
                severity=Severity.LOW,
                size_multiplier=0.75,
            )

        return CircuitBreakerAction(action=TradingState.ACTIVE)

    def reset(self) -> None:
        """Reset breaker - typically called when starting fresh."""
        self.peak_equity = self.current_equity
        self.last_update = datetime.now()

    @property
    def current_drawdown_pct(self) -> float:
        """Get current drawdown percentage."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    @property
    def drawdown_ratio(self) -> float:
        """Get drawdown as ratio of limit."""
        return self.current_drawdown_pct / self.max_drawdown_pct if self.max_drawdown_pct > 0 else 0

    def get_stats(self) -> dict:
        """Get current breaker statistics."""
        return {
            'peak_equity': self.peak_equity,
            'current_equity': self.current_equity,
            'initial_equity': self.initial_equity,
            'current_drawdown_pct': self.current_drawdown_pct,
            'max_drawdown_pct': self.max_drawdown_pct,
            'drawdown_ratio': self.drawdown_ratio,
            'last_update': self.last_update,
        }
