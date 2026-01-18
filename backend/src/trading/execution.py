"""Order execution and management."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import uuid


class OrderType(Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Trading order."""

    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    broker_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "filled_price": self.filled_price,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "broker_id": self.broker_id,
        }


class OrderExecutor:
    """
    Handles order submission and execution.

    Supports multiple execution modes:
    - Simulation: Instant fills at current price
    - Paper: Submit to paper trading API
    - Live: Submit to live broker API
    """

    def __init__(self, mode: str = "simulation", config: Optional[Dict[str, Any]] = None):
        """
        Initialize order executor.

        Args:
            mode: Execution mode (simulation, paper, live)
            config: Execution configuration
        """
        self.mode = mode
        self.config = config or {}
        self.orders: Dict[str, Order] = {}
        self.broker_client = None

    def connect(self) -> bool:
        """Connect to broker."""
        if self.mode == "simulation":
            return True

        # Initialize broker client based on config
        broker = self.config.get("broker", "alpaca")

        if broker == "alpaca":
            return self._connect_alpaca()
        elif broker == "mt5":
            return self._connect_mt5()

        return False

    def _connect_alpaca(self) -> bool:
        """Connect to Alpaca broker."""
        try:
            from alpaca.trading.client import TradingClient

            api_key = self.config.get("api_key")
            secret_key = self.config.get("secret_key")
            paper = self.mode != "live"

            self.broker_client = TradingClient(api_key, secret_key, paper=paper)
            return True

        except Exception as e:
            print(f"Failed to connect to Alpaca: {e}")
            return False

    def _connect_mt5(self) -> bool:
        """Connect to MetaTrader 5."""
        try:
            import MetaTrader5 as mt5

            if not mt5.initialize():
                return False

            login = self.config.get("login")
            password = self.config.get("password")
            server = self.config.get("server")

            if login and password and server:
                authorized = mt5.login(login, password=password, server=server)
                return authorized

            return True

        except Exception as e:
            print(f"Failed to connect to MT5: {e}")
            return False

    def submit_order(self, order: Order) -> Optional[Order]:
        """
        Submit order for execution.

        Args:
            order: Order to submit

        Returns:
            Executed order or None if failed
        """
        order.status = OrderStatus.SUBMITTED
        self.orders[order.id] = order

        if self.mode == "simulation":
            return self._simulate_fill(order)

        elif self.mode in ["paper", "live"]:
            return self._broker_submit(order)

        return None

    def _simulate_fill(self, order: Order) -> Order:
        """Simulate order fill."""
        # In simulation, orders fill instantly at the specified or current price
        fill_price = order.price if order.price else self._get_simulated_price(order.symbol)

        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.filled_at = datetime.now()

        return order

    def _get_simulated_price(self, symbol: str) -> float:
        """Get simulated current price."""
        # In real implementation, would get from data feed
        # For now, return a placeholder
        return 1.0

    def _broker_submit(self, order: Order) -> Optional[Order]:
        """Submit order to broker."""
        broker = self.config.get("broker", "alpaca")

        if broker == "alpaca":
            return self._alpaca_submit(order)
        elif broker == "mt5":
            return self._mt5_submit(order)

        return None

    def _alpaca_submit(self, order: Order) -> Optional[Order]:
        """Submit order to Alpaca."""
        try:
            from alpaca.trading.requests import (
                MarketOrderRequest,
                LimitOrderRequest,
                StopOrderRequest,
            )
            from alpaca.trading.enums import OrderSide as AlpacaSide, TimeInForce

            side = AlpacaSide.BUY if order.side == OrderSide.BUY else AlpacaSide.SELL

            if order.order_type == OrderType.MARKET:
                request = MarketOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            elif order.order_type == OrderType.LIMIT:
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    limit_price=order.price,
                    time_in_force=TimeInForce.DAY,
                )
            else:
                request = StopOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=side,
                    stop_price=order.stop_price,
                    time_in_force=TimeInForce.DAY,
                )

            result = self.broker_client.submit_order(request)
            order.broker_id = result.id
            order.status = OrderStatus.SUBMITTED

            return order

        except Exception as e:
            print(f"Alpaca order failed: {e}")
            order.status = OrderStatus.REJECTED
            return None

    def _mt5_submit(self, order: Order) -> Optional[Order]:
        """Submit order to MetaTrader 5."""
        try:
            import MetaTrader5 as mt5

            order_type = (
                mt5.ORDER_TYPE_BUY if order.side == OrderSide.BUY else mt5.ORDER_TYPE_SELL
            )

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": order.symbol,
                "volume": order.quantity,
                "type": order_type,
                "deviation": 20,
                "magic": 123456,
                "comment": "AI Trader",
                "type_time": mt5.ORDER_TIME_GTC,
            }

            if order.stop_loss:
                request["sl"] = order.stop_loss
            if order.take_profit:
                request["tp"] = order.take_profit

            result = mt5.order_send(request)

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                order.broker_id = str(result.order)
                order.status = OrderStatus.FILLED
                order.filled_price = result.price
                order.filled_quantity = order.quantity
                order.filled_at = datetime.now()
                return order
            else:
                order.status = OrderStatus.REJECTED
                return None

        except Exception as e:
            print(f"MT5 order failed: {e}")
            order.status = OrderStatus.REJECTED
            return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        order = self.orders.get(order_id)
        if not order:
            return False

        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False

        order.status = OrderStatus.CANCELLED
        return True

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        open_statuses = [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]

        orders = [o for o in self.orders.values() if o.status in open_statuses]

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders
