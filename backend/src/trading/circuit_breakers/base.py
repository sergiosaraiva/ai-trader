"""
Circuit Breaker Base Classes and Types.

Circuit breakers automatically halt trading under adverse conditions
to protect capital from catastrophic losses.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Any


class TradingState(Enum):
    """Trading state enumeration."""
    ACTIVE = "active"           # Normal trading
    REDUCED = "reduced"         # Reduced position sizes
    HALTED = "halted"           # No trading allowed
    RECOVERING = "recovering"   # In recovery protocol


class Severity(Enum):
    """Circuit breaker trigger severity."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RecoveryRequirement:
    """Requirements for resuming trading after halt."""
    cooldown_hours: int = 12
    reduced_size_on_resume: float = 0.5  # Fraction of normal size
    wins_to_restore: int = 3  # Consecutive wins to restore full trading
    requires_recalibration: bool = False  # Model needs recalibration


@dataclass
class CircuitBreakerAction:
    """Action returned by circuit breaker check."""
    action: TradingState
    reason: str = ""
    severity: Severity = Severity.LOW
    recovery_requirement: Optional[RecoveryRequirement] = None
    size_multiplier: Optional[float] = None  # For REDUCED state
    min_confidence_override: Optional[float] = None  # Higher confidence required
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_halted(self) -> bool:
        """Check if trading is halted."""
        return self.action == TradingState.HALTED

    @property
    def is_reduced(self) -> bool:
        """Check if trading is reduced."""
        return self.action == TradingState.REDUCED

    @property
    def is_active(self) -> bool:
        """Check if trading is active."""
        return self.action == TradingState.ACTIVE


@dataclass
class TradeResult:
    """Result of a completed trade."""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    entry_time: datetime
    exit_time: datetime
    confidence_at_entry: float = 0.0
    ensemble_agreement: float = 0.0

    @property
    def is_win(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0

    @property
    def is_loss(self) -> bool:
        """Check if trade was a loss."""
        return self.pnl < 0


class CircuitBreaker(ABC):
    """
    Abstract base class for circuit breakers.

    Each circuit breaker monitors a specific condition and returns
    an action when the condition is triggered.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Breaker name for logging."""
        pass

    @abstractmethod
    def check(self, **kwargs) -> CircuitBreakerAction:
        """
        Check if circuit breaker should trigger.

        Returns:
            CircuitBreakerAction with appropriate state
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset breaker state."""
        pass

    def record_trade(self, trade: TradeResult) -> CircuitBreakerAction:
        """
        Record trade and check for trigger.

        Default implementation just calls check().
        Override for breakers that track trade history.
        """
        return self.check()


@dataclass
class CircuitBreakerState:
    """Aggregate state from all circuit breakers."""
    overall_state: TradingState
    active_breakers: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    size_multiplier: float = 1.0
    min_confidence_override: Optional[float] = None
    recovery_end_time: Optional[datetime] = None

    @property
    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        return self.overall_state in [TradingState.ACTIVE, TradingState.REDUCED]

    @property
    def is_halted(self) -> bool:
        """Check if trading is halted."""
        return self.overall_state == TradingState.HALTED

    def get_effective_size_multiplier(self) -> float:
        """Get effective position size multiplier."""
        return self.size_multiplier if self.can_trade else 0.0
