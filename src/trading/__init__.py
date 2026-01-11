"""Trading engine module.

Phase 6: Trading Robot Core components.
"""

# Phase 6 Components - these have clean imports
from .orders.manager import (
    OrderManager,
    Order as NewOrder,
    OrderType as NewOrderType,
    OrderSide as NewOrderSide,
    OrderStatus,
    OrderResult,
    BracketOrder,
    ExecutionMode,
)
from .positions.manager import (
    PositionManager as NewPositionManager,
    Position as NewPosition,
    PositionSide,
    PositionStatus,
)
from .account.manager import AccountManager, AccountState, DailyStats
from .signals.generator import SignalGenerator, EnsemblePrediction
from .signals.actions import TradingSignal, Action, SignalStrength
from .risk.profiles import RiskProfile, RiskLevel, load_risk_profile
from .circuit_breakers.manager import CircuitBreakerManager
from .circuit_breakers.base import CircuitBreakerState, TradingState, TradeResult
from .robot.core import TradingRobot, RobotStatus, RobotState, run_robot
from .robot.config import RobotConfig

# Legacy imports - wrapped to handle import errors gracefully
try:
    from .engine import TradingEngine
except ImportError:
    TradingEngine = None

try:
    from .execution import OrderExecutor, Order, OrderType, OrderSide
except ImportError:
    OrderExecutor = None
    Order = None
    OrderType = None
    OrderSide = None

try:
    from .position import PositionManager as LegacyPositionManager, Position as LegacyPosition
except ImportError:
    LegacyPositionManager = None
    LegacyPosition = None

__all__ = [
    # Phase 6: Orders
    "OrderManager",
    "NewOrder",
    "NewOrderType",
    "NewOrderSide",
    "OrderStatus",
    "OrderResult",
    "BracketOrder",
    "ExecutionMode",
    # Phase 6: Positions
    "NewPositionManager",
    "NewPosition",
    "PositionSide",
    "PositionStatus",
    # Phase 6: Account
    "AccountManager",
    "AccountState",
    "DailyStats",
    # Phase 6: Signals
    "SignalGenerator",
    "TradingSignal",
    "Action",
    "SignalStrength",
    "EnsemblePrediction",
    # Phase 6: Risk
    "RiskProfile",
    "RiskLevel",
    "load_risk_profile",
    # Phase 6: Circuit Breakers
    "CircuitBreakerManager",
    "CircuitBreakerState",
    "TradingState",
    "TradeResult",
    # Phase 6: Robot
    "TradingRobot",
    "RobotStatus",
    "RobotState",
    "RobotConfig",
    "run_robot",
    # Legacy (keep for backward compatibility)
    "TradingEngine",
    "OrderExecutor",
    "Order",
    "OrderType",
    "OrderSide",
    "LegacyPositionManager",
    "LegacyPosition",
]
