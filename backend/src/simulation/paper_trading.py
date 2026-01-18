"""Paper trading module for live simulation."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
import threading
import time

from ..models.base import BaseModel
from ..trading.positions.manager import PositionManager

# Try to import legacy modules, fall back gracefully
try:
    from ..trading.engine import TradingEngine, TradingMode
except ImportError:
    from enum import Enum
    class TradingMode(Enum):
        PAPER = "paper"
        LIVE = "live"

    class TradingEngine:
        def __init__(self, mode=None):
            self.mode = mode
            self.ensemble = None
        def initialize(self, **kwargs):
            pass
        def start(self):
            pass
        def stop(self):
            pass

try:
    from ..trading.execution import OrderExecutor
except ImportError:
    class OrderExecutor:
        def __init__(self, mode="simulation"):
            self.mode = mode

try:
    from ..trading.risk import RiskManager, RiskLimits
except ImportError:
    from dataclasses import dataclass as dc

    @dc
    class RiskLimits:
        max_position_pct: float = 0.1
        max_drawdown_pct: float = 0.2
        max_daily_trades: int = 10

    class RiskManager:
        def __init__(self, initial_balance, risk_limits):
            self.account_balance = initial_balance
            self.risk_limits = risk_limits
            self.daily_pnl = 0
            self._is_halted = False

        def get_risk_metrics(self):
            return {
                "account_balance": self.account_balance,
                "daily_pnl": self.daily_pnl,
                "is_halted": self._is_halted,
            }


class PaperTrader:
    """
    Paper trading simulator using real-time or near-real-time data.

    Features:
    - Simulated order execution
    - Real-time P&L tracking
    - Integration with live data feeds
    - Performance monitoring
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        risk_limits: Optional[RiskLimits] = None,
        data_source: Optional[Any] = None,
    ):
        """
        Initialize paper trader.

        Args:
            initial_balance: Starting account balance
            risk_limits: Risk management limits
            data_source: Data source for market data
        """
        self.initial_balance = initial_balance
        self.risk_limits = risk_limits or RiskLimits()
        self.data_source = data_source

        # Components
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(initial_balance, self.risk_limits)
        self.executor = OrderExecutor(mode="simulation")

        self.trading_engine = TradingEngine(mode=TradingMode.PAPER)
        self.trading_engine.initialize(
            ensemble=None,  # Set later
            executor=self.executor,
            risk_manager=self.risk_manager,
            position_manager=self.position_manager,
        )

        # State
        self.is_running = False
        self._trading_thread: Optional[threading.Thread] = None
        self.symbols: List[str] = []
        self.models: Dict[str, BaseModel] = {}

        # Tracking
        self.equity_history: List[Dict] = []
        self.trade_history: List[Dict] = []

    def set_model(self, model: Any) -> None:
        """Set the prediction model."""
        self.trading_engine.ensemble = model

    def add_symbol(self, symbol: str) -> None:
        """Add a symbol to trade."""
        if symbol not in self.symbols:
            self.symbols.append(symbol)

    def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from trading."""
        if symbol in self.symbols:
            self.symbols.remove(symbol)

    def start(self) -> None:
        """Start paper trading."""
        if self.is_running:
            return

        self.is_running = True
        self.trading_engine.start()

        # Start trading loop in background thread
        self._trading_thread = threading.Thread(target=self._trading_loop)
        self._trading_thread.daemon = True
        self._trading_thread.start()

        print(f"Paper trading started for symbols: {self.symbols}")

    def stop(self) -> None:
        """Stop paper trading."""
        self.is_running = False
        self.trading_engine.stop()

        if self._trading_thread:
            self._trading_thread.join(timeout=5.0)

        print("Paper trading stopped")

    def _trading_loop(self) -> None:
        """Main trading loop."""
        while self.is_running:
            try:
                for symbol in self.symbols:
                    self._process_symbol(symbol)

                self._record_equity()

                # Sleep between iterations
                time.sleep(1.0)

            except Exception as e:
                print(f"Error in trading loop: {e}")
                time.sleep(5.0)

    def _process_symbol(self, symbol: str) -> None:
        """Process updates for a symbol."""
        if self.data_source is None:
            return

        try:
            # Get current market data
            current_price = self.data_source.get_current_price(symbol)

            # Update positions
            self.position_manager.update_price(symbol, current_price.get("last", 0))

            # Check stops
            for position in self.position_manager.check_stop_loss_take_profit():
                self._close_position(position)

            # Get features and generate signal
            # In real implementation, would prepare features here
            # signal = self.trading_engine.on_market_data(symbol, current_price, features)

        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    def _close_position(self, position) -> None:
        """Close a position."""
        closed = self.position_manager.close_position(
            position.symbol, position.current_price
        )
        if closed:
            self.trade_history.append({
                "timestamp": datetime.now(),
                "type": "close",
                "symbol": position.symbol,
                "pnl": position.unrealized_pnl,
            })

    def _record_equity(self) -> None:
        """Record current equity."""
        portfolio = self.position_manager.get_portfolio_metrics()
        equity = self.risk_manager.account_balance + portfolio.get("total_unrealized_pnl", 0)

        self.equity_history.append({
            "timestamp": datetime.now(),
            "equity": equity,
            "balance": self.risk_manager.account_balance,
            "unrealized_pnl": portfolio.get("total_unrealized_pnl", 0),
        })

    def get_status(self) -> Dict[str, Any]:
        """Get current paper trading status."""
        portfolio = self.position_manager.get_portfolio_metrics()
        risk_metrics = self.risk_manager.get_risk_metrics()

        return {
            "is_running": self.is_running,
            "symbols": self.symbols,
            "balance": risk_metrics["account_balance"],
            "equity": risk_metrics["account_balance"] + portfolio.get("total_unrealized_pnl", 0),
            "unrealized_pnl": portfolio.get("total_unrealized_pnl", 0),
            "positions": portfolio.get("total_positions", 0),
            "daily_pnl": risk_metrics["daily_pnl"],
            "is_halted": risk_metrics["is_halted"],
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.equity_history:
            return {}

        equities = [e["equity"] for e in self.equity_history]
        initial = equities[0] if equities else self.initial_balance
        final = equities[-1] if equities else self.initial_balance

        return {
            "initial_balance": self.initial_balance,
            "current_equity": final,
            "total_return": (final - initial) / initial if initial > 0 else 0,
            "total_trades": len(self.trade_history),
            "winning_trades": sum(1 for t in self.trade_history if t.get("pnl", 0) > 0),
            "losing_trades": sum(1 for t in self.trade_history if t.get("pnl", 0) < 0),
        }

    def reset(self) -> None:
        """Reset paper trading state."""
        if self.is_running:
            self.stop()

        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(self.initial_balance, self.risk_limits)
        self.equity_history = []
        self.trade_history = []
