"""Main trading engine orchestrating predictions and execution."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from ..models.base import Prediction
from ..models.ensemble import TechnicalEnsemble
from .execution import OrderExecutor, Order, OrderType, OrderSide
from .risk import RiskManager
from .position import PositionManager


class TradingMode(Enum):
    """Trading operation mode."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


@dataclass
class TradingSignal:
    """Trading signal from model predictions."""

    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL, HOLD
    strength: float  # 0.0 to 1.0
    confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    prediction: Optional[Prediction] = None


class TradingEngine:
    """
    Main trading engine that orchestrates the trading pipeline.

    Responsibilities:
    - Receive market data updates
    - Generate predictions using ensemble model
    - Convert predictions to trading signals
    - Execute orders through broker
    - Manage positions and risk
    """

    def __init__(
        self,
        mode: TradingMode = TradingMode.PAPER,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize trading engine.

        Args:
            mode: Trading mode (backtest, paper, live)
            config: Engine configuration
        """
        self.mode = mode
        self.config = config or {}

        self.ensemble: Optional[TechnicalEnsemble] = None
        self.executor: Optional[OrderExecutor] = None
        self.risk_manager: Optional[RiskManager] = None
        self.position_manager: Optional[PositionManager] = None

        self.is_running = False
        self.trade_history: List[Dict] = []
        self.signal_history: List[TradingSignal] = []

    def initialize(
        self,
        ensemble: TechnicalEnsemble,
        executor: OrderExecutor,
        risk_manager: RiskManager,
        position_manager: PositionManager,
    ) -> None:
        """Initialize engine components."""
        self.ensemble = ensemble
        self.executor = executor
        self.risk_manager = risk_manager
        self.position_manager = position_manager

    def start(self) -> None:
        """Start the trading engine."""
        if self.ensemble is None:
            raise ValueError("Ensemble model not initialized")
        if self.executor is None:
            raise ValueError("Order executor not initialized")

        self.is_running = True
        print(f"Trading engine started in {self.mode.value} mode")

    def stop(self) -> None:
        """Stop the trading engine."""
        self.is_running = False
        print("Trading engine stopped")

    def on_market_data(
        self,
        symbol: str,
        data: Dict[str, Any],
        features: Optional[Any] = None,
    ) -> Optional[TradingSignal]:
        """
        Process new market data.

        Args:
            symbol: Trading symbol
            data: Market data (OHLCV)
            features: Pre-computed features for prediction

        Returns:
            Trading signal if generated
        """
        if not self.is_running:
            return None

        # Generate prediction
        if features is not None and self.ensemble is not None:
            prediction = self.ensemble.predict(features)
            signal_info = self.ensemble.get_signal(prediction)

            signal = TradingSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                action=signal_info["action"],
                strength=signal_info["strength"],
                confidence=signal_info["confidence"],
                reason=signal_info["reason"],
                prediction=prediction,
            )

            # Add stop loss and take profit
            signal = self._add_risk_levels(signal, data)

            self.signal_history.append(signal)
            return signal

        return None

    def process_signal(self, signal: TradingSignal) -> Optional[Order]:
        """
        Process trading signal and potentially execute order.

        Args:
            signal: Trading signal

        Returns:
            Executed order or None
        """
        if not self.is_running:
            return None

        # Check risk limits
        if self.risk_manager and not self.risk_manager.check_signal(signal):
            print(f"Signal rejected by risk manager: {signal.symbol}")
            return None

        # Check for existing position
        if self.position_manager:
            position = self.position_manager.get_position(signal.symbol)

            # Don't open conflicting positions
            if position and position.side != signal.action:
                # Close existing position first
                close_order = self._create_close_order(position)
                if close_order and self.executor:
                    self.executor.submit_order(close_order)

        # Create and submit order
        if signal.action in ["BUY", "SELL"]:
            order = self._create_order_from_signal(signal)
            if order and self.executor:
                executed = self.executor.submit_order(order)
                if executed:
                    self.trade_history.append({
                        "timestamp": datetime.now(),
                        "symbol": signal.symbol,
                        "action": signal.action,
                        "signal_strength": signal.strength,
                        "confidence": signal.confidence,
                        "order_id": order.id,
                    })
                return executed

        return None

    def _add_risk_levels(
        self, signal: TradingSignal, data: Dict[str, Any]
    ) -> TradingSignal:
        """Add stop loss and take profit levels to signal."""
        if self.risk_manager:
            current_price = data.get("close", 0)
            atr = data.get("atr", current_price * 0.01)  # Default 1% ATR

            if signal.action == "BUY":
                signal.stop_loss = current_price - (atr * 2)
                signal.take_profit = current_price + (atr * 3)
            elif signal.action == "SELL":
                signal.stop_loss = current_price + (atr * 2)
                signal.take_profit = current_price - (atr * 3)

        return signal

    def _create_order_from_signal(self, signal: TradingSignal) -> Optional[Order]:
        """Create order from trading signal."""
        if self.risk_manager is None:
            return None

        position_size = self.risk_manager.calculate_position_size(
            symbol=signal.symbol,
            signal_strength=signal.strength,
            stop_loss_distance=signal.stop_loss or 0,
        )

        if position_size <= 0:
            return None

        return Order(
            symbol=signal.symbol,
            side=OrderSide.BUY if signal.action == "BUY" else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=position_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

    def _create_close_order(self, position) -> Optional[Order]:
        """Create order to close existing position."""
        return Order(
            symbol=position.symbol,
            side=OrderSide.SELL if position.side == "BUY" else OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "mode": self.mode.value,
            "is_running": self.is_running,
            "signals_generated": len(self.signal_history),
            "trades_executed": len(self.trade_history),
            "positions": (
                self.position_manager.get_all_positions()
                if self.position_manager
                else []
            ),
        }

    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate trading performance metrics."""
        if not self.trade_history:
            return {}

        # Basic metrics from trade history
        # Full implementation would track P&L per trade
        return {
            "total_trades": len(self.trade_history),
            "signals_generated": len(self.signal_history),
        }
