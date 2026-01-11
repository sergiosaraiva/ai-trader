"""
Order Management System.

Handles order creation, submission, tracking, and bracket orders.
Supports both simulation and production execution modes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import uuid
import logging

from ..signals.actions import TradingSignal, Action

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"          # Created but not submitted
    SUBMITTED = "submitted"      # Sent to broker
    ACCEPTED = "accepted"        # Accepted by broker
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ExecutionMode(Enum):
    """Execution mode enumeration."""
    SIMULATION = "simulation"
    PAPER = "paper"
    PRODUCTION = "production"


@dataclass
class Order:
    """
    Trading order representation.

    Represents a single order to be submitted to a broker.
    """
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    status: OrderStatus = OrderStatus.PENDING

    # Price fields
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None

    # Execution details
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None

    # Metadata
    signal_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        """Check if order is still open."""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
        ]

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == OrderStatus.FILLED

    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        ]

    @property
    def remaining_quantity(self) -> float:
        """Get remaining quantity to fill."""
        return self.quantity - self.filled_quantity

    @property
    def fill_percentage(self) -> float:
        """Get fill percentage."""
        return (self.filled_quantity / self.quantity * 100) if self.quantity > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "status": self.status.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "filled_quantity": self.filled_quantity,
            "average_fill_price": self.average_fill_price,
            "commission": self.commission,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
        }


@dataclass
class OrderResult:
    """Result of order submission."""
    success: bool
    order_id: str
    status: OrderStatus
    message: str = ""
    fill_price: Optional[float] = None
    filled_quantity: float = 0.0
    commission: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "order_id": self.order_id,
            "status": self.status.value,
            "message": self.message,
            "fill_price": self.fill_price,
            "filled_quantity": self.filled_quantity,
            "commission": self.commission,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BracketOrder:
    """
    Bracket order with entry, stop-loss, and take-profit.

    Used for risk-managed entries with predefined exit levels.
    """
    bracket_id: str
    entry_order: Order
    stop_loss_order: Optional[Order] = None
    take_profit_order: Optional[Order] = None

    # Status
    is_active: bool = False
    is_filled: bool = False
    is_closed: bool = False

    # Tracking
    entry_fill_price: Optional[float] = None
    exit_fill_price: Optional[float] = None
    exit_reason: str = ""

    # PnL
    realized_pnl: float = 0.0

    @property
    def status(self) -> str:
        """Get bracket order status."""
        if self.is_closed:
            return "closed"
        if self.is_filled:
            return "active"
        if self.is_active:
            return "entry_pending"
        return "created"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bracket_id": self.bracket_id,
            "status": self.status,
            "entry_order": self.entry_order.to_dict(),
            "stop_loss_order": self.stop_loss_order.to_dict() if self.stop_loss_order else None,
            "take_profit_order": self.take_profit_order.to_dict() if self.take_profit_order else None,
            "entry_fill_price": self.entry_fill_price,
            "exit_fill_price": self.exit_fill_price,
            "exit_reason": self.exit_reason,
            "realized_pnl": self.realized_pnl,
        }


class OrderExecutor(ABC):
    """Abstract base class for order execution."""

    @abstractmethod
    def submit_order(self, order: Order) -> OrderResult:
        """Submit order for execution."""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current order status."""
        pass


class SimulationExecutor(OrderExecutor):
    """Simulated order executor for backtesting and paper trading."""

    def __init__(
        self,
        base_spread_pct: float = 0.0002,
        slippage_pct: float = 0.0001,
        commission_per_lot: float = 0.0,
        get_price_callback: Optional[Callable[[str], float]] = None,
    ):
        """
        Initialize simulation executor.

        Args:
            base_spread_pct: Base spread as percentage
            slippage_pct: Slippage as percentage
            commission_per_lot: Commission per lot traded
            get_price_callback: Callback to get current price
        """
        self.base_spread_pct = base_spread_pct
        self.slippage_pct = slippage_pct
        self.commission_per_lot = commission_per_lot
        self.get_price = get_price_callback or (lambda s: 1.0)
        self.orders: Dict[str, Order] = {}

    def submit_order(self, order: Order) -> OrderResult:
        """Submit order for simulated execution."""
        # Store order
        self.orders[order.order_id] = order

        # Get current price
        price = self.get_price(order.symbol)

        # Apply spread and slippage
        if order.side == OrderSide.BUY:
            fill_price = price * (1 + self.base_spread_pct / 2 + self.slippage_pct)
        else:
            fill_price = price * (1 - self.base_spread_pct / 2 - self.slippage_pct)

        # Calculate commission
        commission = self.commission_per_lot * order.quantity

        # For market orders, fill immediately
        if order.order_type == OrderType.MARKET:
            order.status = OrderStatus.FILLED
            order.submitted_at = datetime.now()
            order.filled_at = datetime.now()
            order.filled_quantity = order.quantity
            order.average_fill_price = fill_price
            order.commission = commission

            return OrderResult(
                success=True,
                order_id=order.order_id,
                status=OrderStatus.FILLED,
                message="Order filled in simulation",
                fill_price=fill_price,
                filled_quantity=order.quantity,
                commission=commission,
            )

        # For limit orders, accept but don't fill immediately
        order.status = OrderStatus.ACCEPTED
        order.submitted_at = datetime.now()

        return OrderResult(
            success=True,
            order_id=order.order_id,
            status=OrderStatus.ACCEPTED,
            message="Order accepted, waiting for fill",
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.is_terminal:
            return False

        order.status = OrderStatus.CANCELLED
        order.cancelled_at = datetime.now()
        return True

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def check_limit_orders(self, symbol: str, current_price: float) -> List[OrderResult]:
        """
        Check and fill limit orders that have hit their price.

        Args:
            symbol: Symbol to check
            current_price: Current market price

        Returns:
            List of order results for filled orders
        """
        results = []

        for order in list(self.orders.values()):
            if order.symbol != symbol or order.is_terminal:
                continue

            should_fill = False
            fill_price = current_price

            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and order.limit_price >= current_price:
                    should_fill = True
                    fill_price = order.limit_price
                elif order.side == OrderSide.SELL and order.limit_price <= current_price:
                    should_fill = True
                    fill_price = order.limit_price

            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and current_price >= order.stop_price:
                    should_fill = True
                elif order.side == OrderSide.SELL and current_price <= order.stop_price:
                    should_fill = True

            if should_fill:
                commission = self.commission_per_lot * order.quantity
                order.status = OrderStatus.FILLED
                order.filled_at = datetime.now()
                order.filled_quantity = order.quantity
                order.average_fill_price = fill_price
                order.commission = commission

                results.append(OrderResult(
                    success=True,
                    order_id=order.order_id,
                    status=OrderStatus.FILLED,
                    message="Limit/Stop order filled",
                    fill_price=fill_price,
                    filled_quantity=order.quantity,
                    commission=commission,
                ))

        return results


class OrderManager:
    """
    Order management system.

    Coordinates order creation, submission, tracking, and bracket orders.
    """

    def __init__(
        self,
        execution_mode: ExecutionMode = ExecutionMode.SIMULATION,
        executor: Optional[OrderExecutor] = None,
        get_price_callback: Optional[Callable[[str], float]] = None,
    ):
        """
        Initialize order manager.

        Args:
            execution_mode: Mode of execution
            executor: Order executor (uses simulation if not provided)
            get_price_callback: Callback to get current price
        """
        self.execution_mode = execution_mode
        self.get_price = get_price_callback or (lambda s: 1.0)

        # Initialize executor
        if executor:
            self.executor = executor
        elif execution_mode == ExecutionMode.SIMULATION:
            self.executor = SimulationExecutor(get_price_callback=self.get_price)
        else:
            raise ValueError(f"No executor provided for mode {execution_mode}")

        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.bracket_orders: Dict[str, BracketOrder] = {}

        # Callbacks
        self._on_fill_callbacks: List[Callable[[Order], None]] = []
        self._on_cancel_callbacks: List[Callable[[Order], None]] = []

        logger.info(f"OrderManager initialized in {execution_mode.value} mode")

    def create_order(
        self,
        signal: TradingSignal,
        account_equity: float,
        lot_size: float = 100000,
    ) -> Order:
        """
        Create order from trading signal.

        Args:
            signal: Trading signal
            account_equity: Current account equity
            lot_size: Lot size for position calculation

        Returns:
            Created Order (not yet submitted)
        """
        # Determine side
        if signal.action in [Action.BUY, Action.CLOSE_SHORT]:
            side = OrderSide.BUY
        elif signal.action in [Action.SELL, Action.CLOSE_LONG]:
            side = OrderSide.SELL
        else:
            raise ValueError(f"Cannot create order for action {signal.action}")

        # Calculate quantity
        notional = account_equity * signal.position_size_pct
        price = self.get_price(signal.symbol)
        quantity = notional / (price * lot_size) if price > 0 else 0

        # Round to standard increments
        quantity = round(quantity, 2)

        # Create order
        order = Order(
            order_id=str(uuid.uuid4()),
            symbol=signal.symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            signal_id=str(uuid.uuid4()),
            metadata={
                "confidence": signal.confidence,
                "direction_probability": signal.direction_probability,
            },
        )

        logger.info(
            f"Created {side.value} order for {signal.symbol}: "
            f"{quantity:.2f} lots at market"
        )

        return order

    def create_bracket_order(
        self,
        signal: TradingSignal,
        account_equity: float,
        lot_size: float = 100000,
    ) -> BracketOrder:
        """
        Create bracket order with entry, stop-loss, and take-profit.

        Args:
            signal: Trading signal with SL/TP prices
            account_equity: Current account equity
            lot_size: Lot size for position calculation

        Returns:
            BracketOrder with entry and exit orders
        """
        # Create entry order
        entry_order = self.create_order(signal, account_equity, lot_size)

        # Determine exit side (opposite of entry)
        exit_side = OrderSide.SELL if entry_order.side == OrderSide.BUY else OrderSide.BUY

        # Create stop-loss order
        stop_loss_order = None
        if signal.stop_loss_price:
            stop_loss_order = Order(
                order_id=str(uuid.uuid4()),
                symbol=signal.symbol,
                side=exit_side,
                order_type=OrderType.STOP,
                quantity=entry_order.quantity,
                stop_price=signal.stop_loss_price,
                parent_order_id=entry_order.order_id,
            )

        # Create take-profit order
        take_profit_order = None
        if signal.take_profit_price:
            take_profit_order = Order(
                order_id=str(uuid.uuid4()),
                symbol=signal.symbol,
                side=exit_side,
                order_type=OrderType.LIMIT,
                quantity=entry_order.quantity,
                limit_price=signal.take_profit_price,
                parent_order_id=entry_order.order_id,
            )

        # Create bracket
        bracket = BracketOrder(
            bracket_id=str(uuid.uuid4()),
            entry_order=entry_order,
            stop_loss_order=stop_loss_order,
            take_profit_order=take_profit_order,
        )

        logger.info(
            f"Created bracket order {bracket.bracket_id}: "
            f"Entry={entry_order.side.value}, "
            f"SL={signal.stop_loss_price}, TP={signal.take_profit_price}"
        )

        return bracket

    def submit_order(self, order: Order) -> OrderResult:
        """
        Submit order for execution.

        Args:
            order: Order to submit

        Returns:
            OrderResult with execution status
        """
        # Store order
        self.orders[order.order_id] = order

        # Submit to executor
        result = self.executor.submit_order(order)

        # Call fill callback if filled
        if result.status == OrderStatus.FILLED:
            for callback in self._on_fill_callbacks:
                try:
                    callback(order)
                except Exception as e:
                    logger.error(f"Error in fill callback: {e}")

        return result

    def submit_bracket_order(self, bracket: BracketOrder) -> OrderResult:
        """
        Submit bracket order for execution.

        Args:
            bracket: Bracket order to submit

        Returns:
            OrderResult for entry order
        """
        # Store bracket
        self.bracket_orders[bracket.bracket_id] = bracket
        bracket.is_active = True

        # Submit entry order
        result = self.submit_order(bracket.entry_order)

        if result.status == OrderStatus.FILLED:
            bracket.is_filled = True
            bracket.entry_fill_price = result.fill_price

            # Submit stop-loss and take-profit orders
            if bracket.stop_loss_order:
                self.submit_order(bracket.stop_loss_order)
            if bracket.take_profit_order:
                self.submit_order(bracket.take_profit_order)

        return result

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False

        success = self.executor.cancel_order(order_id)

        if success:
            order = self.orders[order_id]
            for callback in self._on_cancel_callbacks:
                try:
                    callback(order)
                except Exception as e:
                    logger.error(f"Error in cancel callback: {e}")

        return success

    def cancel_bracket_order(self, bracket_id: str) -> bool:
        """
        Cancel all orders in a bracket.

        Args:
            bracket_id: Bracket ID to cancel

        Returns:
            True if all orders cancelled
        """
        if bracket_id not in self.bracket_orders:
            return False

        bracket = self.bracket_orders[bracket_id]
        cancelled_all = True

        # Cancel entry if not filled
        if bracket.entry_order.is_open:
            if not self.cancel_order(bracket.entry_order.order_id):
                cancelled_all = False

        # Cancel stop-loss
        if bracket.stop_loss_order and bracket.stop_loss_order.is_open:
            if not self.cancel_order(bracket.stop_loss_order.order_id):
                cancelled_all = False

        # Cancel take-profit
        if bracket.take_profit_order and bracket.take_profit_order.is_open:
            if not self.cancel_order(bracket.take_profit_order.order_id):
                cancelled_all = False

        bracket.is_active = False
        bracket.is_closed = True
        bracket.exit_reason = "cancelled"

        return cancelled_all

    def update_bracket_status(
        self,
        symbol: str,
        current_price: float,
    ) -> List[BracketOrder]:
        """
        Update bracket orders based on current price.

        Args:
            symbol: Symbol to check
            current_price: Current market price

        Returns:
            List of brackets that were closed
        """
        closed_brackets = []

        # Check limit orders in simulation
        if isinstance(self.executor, SimulationExecutor):
            fill_results = self.executor.check_limit_orders(symbol, current_price)

            for result in fill_results:
                order = self.orders.get(result.order_id)
                if not order:
                    continue

                # Call fill callback
                for callback in self._on_fill_callbacks:
                    try:
                        callback(order)
                    except Exception as e:
                        logger.error(f"Error in fill callback: {e}")

        # Update bracket statuses
        for bracket in self.bracket_orders.values():
            if bracket.is_closed or bracket.entry_order.symbol != symbol:
                continue

            # Check if SL/TP hit
            sl_filled = bracket.stop_loss_order and bracket.stop_loss_order.is_filled
            tp_filled = bracket.take_profit_order and bracket.take_profit_order.is_filled

            if sl_filled or tp_filled:
                bracket.is_closed = True
                bracket.is_active = False

                if sl_filled:
                    bracket.exit_fill_price = bracket.stop_loss_order.average_fill_price
                    bracket.exit_reason = "stop_loss"
                elif tp_filled:
                    bracket.exit_fill_price = bracket.take_profit_order.average_fill_price
                    bracket.exit_reason = "take_profit"

                # Calculate PnL
                if bracket.entry_fill_price and bracket.exit_fill_price:
                    if bracket.entry_order.side == OrderSide.BUY:
                        bracket.realized_pnl = (
                            (bracket.exit_fill_price - bracket.entry_fill_price) *
                            bracket.entry_order.quantity
                        )
                    else:
                        bracket.realized_pnl = (
                            (bracket.entry_fill_price - bracket.exit_fill_price) *
                            bracket.entry_order.quantity
                        )

                # Cancel the other exit order
                if sl_filled and bracket.take_profit_order:
                    self.cancel_order(bracket.take_profit_order.order_id)
                elif tp_filled and bracket.stop_loss_order:
                    self.cancel_order(bracket.stop_loss_order.order_id)

                closed_brackets.append(bracket)
                logger.info(
                    f"Bracket {bracket.bracket_id} closed via {bracket.exit_reason}: "
                    f"PnL={bracket.realized_pnl:.4f}"
                )

        return closed_brackets

    def get_open_orders(self) -> List[Order]:
        """Get all open orders."""
        return [o for o in self.orders.values() if o.is_open]

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_bracket(self, bracket_id: str) -> Optional[BracketOrder]:
        """Get bracket order by ID."""
        return self.bracket_orders.get(bracket_id)

    def get_active_brackets(self) -> List[BracketOrder]:
        """Get all active bracket orders."""
        return [b for b in self.bracket_orders.values() if b.is_active]

    def on_fill(self, callback: Callable[[Order], None]) -> None:
        """Register callback for order fill events."""
        self._on_fill_callbacks.append(callback)

    def on_cancel(self, callback: Callable[[Order], None]) -> None:
        """Register callback for order cancel events."""
        self._on_cancel_callbacks.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get order manager statistics."""
        open_orders = self.get_open_orders()
        filled_orders = [o for o in self.orders.values() if o.is_filled]
        active_brackets = self.get_active_brackets()
        closed_brackets = [b for b in self.bracket_orders.values() if b.is_closed]

        total_pnl = sum(b.realized_pnl for b in closed_brackets)

        return {
            "execution_mode": self.execution_mode.value,
            "total_orders": len(self.orders),
            "open_orders": len(open_orders),
            "filled_orders": len(filled_orders),
            "total_brackets": len(self.bracket_orders),
            "active_brackets": len(active_brackets),
            "closed_brackets": len(closed_brackets),
            "total_realized_pnl": total_pnl,
        }

    def reset(self) -> None:
        """Reset order manager state."""
        self.orders.clear()
        self.bracket_orders.clear()
        if isinstance(self.executor, SimulationExecutor):
            self.executor.orders.clear()
        logger.info("OrderManager state reset")
