"""
Simulation and Backtesting Module.

Provides comprehensive backtesting and simulation capabilities:
- Market simulator for historical data replay
- Enhanced backtester with trading robot integration
- Realistic execution simulation
- Performance metrics calculation
"""

# Legacy backtester (for backward compatibility)
from .backtester import Backtester, BacktestResult

# Paper trading
from .paper_trading import PaperTrader

# Metrics
from .metrics import PerformanceMetrics

# Phase 7: Enhanced simulation components
from .market_simulator import (
    MarketSimulator,
    MarketBar,
    MarketSnapshot,
    MarketSession,
    MarketStatus,
    FOREX_SESSION,
    US_STOCK_SESSION,
)

from .backtester_v2 import (
    EnhancedBacktester,
    BacktestConfig,
    BacktestStatus,
    BacktestResult as EnhancedBacktestResult,
    Trade,
)

__all__ = [
    # Legacy
    "Backtester",
    "BacktestResult",
    "PaperTrader",
    "PerformanceMetrics",
    # Market Simulator
    "MarketSimulator",
    "MarketBar",
    "MarketSnapshot",
    "MarketSession",
    "MarketStatus",
    "FOREX_SESSION",
    "US_STOCK_SESSION",
    # Enhanced Backtester
    "EnhancedBacktester",
    "BacktestConfig",
    "BacktestStatus",
    "EnhancedBacktestResult",
    "Trade",
]
