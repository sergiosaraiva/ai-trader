"""
Circuit Breakers Module.

Provides automatic trading protection through various circuit breakers:
- ConsecutiveLossBreaker: Halts after N consecutive losses
- DrawdownBreaker: Progressive protection as drawdown increases
- ModelDegradationBreaker: Detects model performance degradation
- CircuitBreakerManager: Coordinates all breakers
"""

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
from .manager import (
    CircuitBreakerManager,
    RecoveryProtocol,
    RecoveryPhase,
    RecoveryState,
)

__all__ = [
    # Base types
    'CircuitBreaker',
    'CircuitBreakerAction',
    'CircuitBreakerState',
    'TradeResult',
    'TradingState',
    'Severity',
    'RecoveryRequirement',
    # Breaker implementations
    'ConsecutiveLossBreaker',
    'DrawdownBreaker',
    'ModelDegradationBreaker',
    # Manager
    'CircuitBreakerManager',
    'RecoveryProtocol',
    'RecoveryPhase',
    'RecoveryState',
]
