"""
Production Execution Engine.

Provides production-grade order execution with broker integration,
validation, safety controls, and position reconciliation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import asyncio
import logging

from ..orders.manager import Order, OrderType, OrderSide, OrderStatus
from ..brokers.base import (
    BrokerAdapter,
    BrokerConfig,
    BrokerOrder,
    BrokerPosition,
    Quote,
    AccountInfo,
    ConnectionStatus,
    BrokerError,
    OrderRejectedError,
    InsufficientFundsError,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Order validation failed."""
    def __init__(self, message: str, violations: List[str] = None):
        super().__init__(message)
        self.violations = violations or []


class ReconciliationError(Exception):
    """Position reconciliation failed."""
    pass


@dataclass
class OrderValidationConfig:
    """Configuration for order validation."""
    # Size limits
    max_order_size: float = 1000000.0
    min_order_size: float = 1.0
    max_position_size: float = 10000000.0

    # Value limits
    max_order_value: float = 1000000.0
    max_daily_volume: float = 10000000.0

    # Price limits
    max_price_deviation_pct: float = 5.0  # Max deviation from market price

    # Symbol restrictions
    allowed_symbols: List[str] = field(default_factory=list)
    blocked_symbols: List[str] = field(default_factory=list)

    # Order type restrictions
    allow_market_orders: bool = True
    allow_limit_orders: bool = True
    allow_stop_orders: bool = True

    # Time restrictions
    allow_extended_hours: bool = False

    # Safety
    require_kill_switch_check: bool = True
    max_orders_per_minute: int = 60
    max_orders_per_day: int = 1000


@dataclass
class ProductionConfig:
    """Configuration for production execution engine."""
    # Broker settings
    broker_config: BrokerConfig = None

    # Validation
    validation_config: OrderValidationConfig = field(default_factory=OrderValidationConfig)

    # Position reconciliation
    reconciliation_interval_seconds: float = 60.0
    auto_close_unknown_positions: bool = False

    # Connection
    auto_reconnect: bool = True
    reconnect_delay_seconds: float = 5.0
    max_reconnect_attempts: int = 10

    # Order timeout
    order_timeout_seconds: float = 30.0
    cancel_timeout_seconds: float = 10.0

    # Logging
    log_all_orders: bool = True
    log_all_fills: bool = True


@dataclass
class OrderValidationResult:
    """Result of order validation."""
    is_valid: bool
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid


class OrderValidator:
    """
    Validates orders before submission.

    Checks size limits, price reasonableness, symbol permissions, etc.
    """

    def __init__(self, config: OrderValidationConfig):
        """
        Initialize order validator.

        Args:
            config: Validation configuration
        """
        self.config = config
        self._orders_this_minute: List[datetime] = []
        self._orders_today: List[datetime] = []
        self._daily_volume: float = 0.0
        self._last_day_reset: datetime = datetime.now().replace(hour=0, minute=0, second=0)

    def validate(
        self,
        order: Order,
        current_price: float,
        account_info: Optional[AccountInfo] = None,
        current_position_value: float = 0.0,
        kill_switch_active: bool = False,
    ) -> OrderValidationResult:
        """
        Validate an order.

        Args:
            order: Order to validate
            current_price: Current market price for the symbol
            account_info: Current account information
            current_position_value: Current position value in this symbol
            kill_switch_active: Whether kill switch is triggered

        Returns:
            Validation result
        """
        violations = []
        warnings = []

        # Check kill switch
        if self.config.require_kill_switch_check and kill_switch_active:
            violations.append("Kill switch is active - trading halted")

        # Check symbol permissions
        if self.config.allowed_symbols and order.symbol not in self.config.allowed_symbols:
            violations.append(f"Symbol {order.symbol} not in allowed list")

        if order.symbol in self.config.blocked_symbols:
            violations.append(f"Symbol {order.symbol} is blocked")

        # Check order type permissions
        if order.order_type == OrderType.MARKET and not self.config.allow_market_orders:
            violations.append("Market orders not allowed")
        if order.order_type == OrderType.LIMIT and not self.config.allow_limit_orders:
            violations.append("Limit orders not allowed")
        if order.order_type == OrderType.STOP and not self.config.allow_stop_orders:
            violations.append("Stop orders not allowed")

        # Check size limits
        if order.quantity < self.config.min_order_size:
            violations.append(f"Order size {order.quantity} below minimum {self.config.min_order_size}")
        if order.quantity > self.config.max_order_size:
            violations.append(f"Order size {order.quantity} exceeds maximum {self.config.max_order_size}")

        # Check order value
        order_value = order.quantity * current_price
        if order_value > self.config.max_order_value:
            violations.append(f"Order value {order_value:.2f} exceeds maximum {self.config.max_order_value:.2f}")

        # Check position size
        new_position_value = current_position_value + order_value
        if new_position_value > self.config.max_position_size:
            violations.append(f"Resulting position {new_position_value:.2f} exceeds maximum {self.config.max_position_size:.2f}")

        # Check price deviation for limit orders
        if order.order_type == OrderType.LIMIT and order.limit_price:
            deviation_pct = abs(order.limit_price - current_price) / current_price * 100
            if deviation_pct > self.config.max_price_deviation_pct:
                warnings.append(f"Limit price deviates {deviation_pct:.2f}% from market")

        # Check rate limits
        self._clean_rate_limit_windows()

        if len(self._orders_this_minute) >= self.config.max_orders_per_minute:
            violations.append(f"Rate limit exceeded: {self.config.max_orders_per_minute} orders/minute")

        if len(self._orders_today) >= self.config.max_orders_per_day:
            violations.append(f"Daily order limit exceeded: {self.config.max_orders_per_day} orders/day")

        # Check daily volume
        if self._daily_volume + order_value > self.config.max_daily_volume:
            violations.append(f"Daily volume limit exceeded")

        # Check buying power
        if account_info:
            if order.side == OrderSide.BUY and order_value > account_info.buying_power:
                violations.append(f"Insufficient buying power: need {order_value:.2f}, have {account_info.buying_power:.2f}")

        return OrderValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
        )

    def record_order(self, order: Order, price: float) -> None:
        """Record an order for rate limiting."""
        now = datetime.now()
        self._orders_this_minute.append(now)
        self._orders_today.append(now)
        self._daily_volume += order.quantity * price

    def _clean_rate_limit_windows(self) -> None:
        """Clean expired rate limit windows."""
        now = datetime.now()

        # Clean minute window
        cutoff_minute = now - timedelta(minutes=1)
        self._orders_this_minute = [t for t in self._orders_this_minute if t > cutoff_minute]

        # Reset daily counters at midnight
        if now.date() > self._last_day_reset.date():
            self._orders_today = []
            self._daily_volume = 0.0
            self._last_day_reset = now

    def reset_daily_counters(self) -> None:
        """Manually reset daily counters."""
        self._orders_today = []
        self._daily_volume = 0.0
        self._last_day_reset = datetime.now()


class ProductionExecutionEngine:
    """
    Production execution engine with full broker integration.

    Features:
    - Real broker order execution
    - Order validation before submission
    - Kill switch integration
    - Automatic position reconciliation
    - Connection monitoring and auto-reconnect
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        config: Optional[ProductionConfig] = None,
        kill_switch_callback: Optional[Callable[[], bool]] = None,
    ):
        """
        Initialize production execution engine.

        Args:
            broker: Broker adapter for order execution
            config: Production configuration
            kill_switch_callback: Callback to check if kill switch is active
        """
        self.broker = broker
        self.config = config or ProductionConfig()
        self._kill_switch_callback = kill_switch_callback or (lambda: False)

        # Order validation
        self.validator = OrderValidator(self.config.validation_config)

        # Order tracking
        self._pending_orders: Dict[str, Order] = {}
        self._submitted_orders: Dict[str, BrokerOrder] = {}
        self._order_mapping: Dict[str, str] = {}  # client_id -> broker_id

        # Position tracking
        self._known_positions: Dict[str, BrokerPosition] = {}

        # Callbacks
        self._on_fill: List[Callable[[Order, BrokerOrder], None]] = []
        self._on_reject: List[Callable[[Order, str], None]] = []
        self._on_error: List[Callable[[str, Exception], None]] = []

        # Background tasks
        self._reconciliation_task: Optional[asyncio.Task] = None
        self._running = False

        # Statistics
        self._stats = {
            "orders_submitted": 0,
            "orders_filled": 0,
            "orders_rejected": 0,
            "orders_cancelled": 0,
            "validation_failures": 0,
            "connection_errors": 0,
            "reconciliation_mismatches": 0,
        }

    async def start(self) -> None:
        """Start the execution engine."""
        if self._running:
            return

        self._running = True

        # Start position reconciliation
        if self.config.reconciliation_interval_seconds > 0:
            self._reconciliation_task = asyncio.create_task(
                self._reconciliation_loop()
            )

        logger.info("Production execution engine started")

    async def stop(self) -> None:
        """Stop the execution engine."""
        self._running = False

        # Stop reconciliation
        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass
            self._reconciliation_task = None

        logger.info("Production execution engine stopped")

    async def submit_order(self, order: Order) -> BrokerOrder:
        """
        Submit an order to the broker.

        Args:
            order: Order to submit

        Returns:
            BrokerOrder with submission result

        Raises:
            ValidationError: If order fails validation
            BrokerError: If broker rejects order
        """
        # Get current price for validation
        try:
            quote = await self.broker.get_quote(order.symbol)
            current_price = quote.mid
        except BrokerError:
            current_price = order.limit_price or 0.0

        # Get account info
        try:
            account_info = await self.broker.get_account()
        except BrokerError:
            account_info = None

        # Get current position value
        try:
            position = await self.broker.get_position(order.symbol)
            current_position_value = position.market_value if position else 0.0
        except BrokerError:
            current_position_value = 0.0

        # Validate order
        validation = self.validator.validate(
            order=order,
            current_price=current_price,
            account_info=account_info,
            current_position_value=current_position_value,
            kill_switch_active=self._kill_switch_callback(),
        )

        if not validation.is_valid:
            self._stats["validation_failures"] += 1
            error_msg = "; ".join(validation.violations)
            logger.warning(f"Order validation failed: {error_msg}")

            for callback in self._on_reject:
                try:
                    callback(order, error_msg)
                except Exception as e:
                    logger.error(f"Error in reject callback: {e}")

            raise ValidationError(f"Order validation failed: {error_msg}", validation.violations)

        # Log warnings
        for warning in validation.warnings:
            logger.warning(f"Order warning: {warning}")

        # Track pending order
        self._pending_orders[order.order_id] = order

        try:
            # Submit to broker
            broker_order = await self.broker.submit_order(
                symbol=order.symbol,
                side="buy" if order.side == OrderSide.BUY else "sell",
                quantity=order.quantity,
                order_type=self._map_order_type(order.order_type),
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                time_in_force="day",
                client_order_id=order.order_id,
            )

            # Record for rate limiting
            self.validator.record_order(order, current_price)

            # Track submitted order
            self._submitted_orders[broker_order.order_id] = broker_order
            self._order_mapping[order.order_id] = broker_order.order_id

            # Update order status
            order.status = OrderStatus.SUBMITTED
            if broker_order.is_filled:
                order.status = OrderStatus.FILLED
                order.filled_quantity = broker_order.filled_quantity
                order.average_fill_price = broker_order.average_fill_price

                # Notify fill
                for callback in self._on_fill:
                    try:
                        callback(order, broker_order)
                    except Exception as e:
                        logger.error(f"Error in fill callback: {e}")

                self._stats["orders_filled"] += 1

            self._stats["orders_submitted"] += 1

            if self.config.log_all_orders:
                logger.info(
                    f"Order submitted: {order.order_id} -> {broker_order.order_id} "
                    f"{order.side.value} {order.quantity} {order.symbol} @ {order.order_type.value}"
                )

            return broker_order

        except OrderRejectedError as e:
            self._stats["orders_rejected"] += 1
            order.status = OrderStatus.REJECTED

            for callback in self._on_reject:
                try:
                    callback(order, str(e))
                except Exception as ex:
                    logger.error(f"Error in reject callback: {ex}")

            raise

        except InsufficientFundsError as e:
            self._stats["orders_rejected"] += 1
            order.status = OrderStatus.REJECTED

            for callback in self._on_reject:
                try:
                    callback(order, str(e))
                except Exception as ex:
                    logger.error(f"Error in reject callback: {ex}")

            raise

        except BrokerError as e:
            self._stats["connection_errors"] += 1
            del self._pending_orders[order.order_id]

            for callback in self._on_error:
                try:
                    callback("submit_order", e)
                except Exception as ex:
                    logger.error(f"Error in error callback: {ex}")

            raise

        finally:
            # Clean up pending
            if order.order_id in self._pending_orders:
                del self._pending_orders[order.order_id]

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Client order ID to cancel

        Returns:
            True if cancellation accepted
        """
        # Get broker order ID
        broker_order_id = self._order_mapping.get(order_id, order_id)

        try:
            result = await asyncio.wait_for(
                self.broker.cancel_order(broker_order_id),
                timeout=self.config.cancel_timeout_seconds,
            )

            if result:
                self._stats["orders_cancelled"] += 1
                logger.info(f"Order cancelled: {order_id}")

            return result

        except asyncio.TimeoutError:
            logger.error(f"Cancel order timeout: {order_id}")
            return False
        except BrokerError as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            Number of orders cancelled
        """
        cancelled = await self.broker.cancel_all_orders(symbol)
        self._stats["orders_cancelled"] += cancelled
        logger.info(f"Cancelled {cancelled} orders")
        return cancelled

    async def get_order_status(self, order_id: str) -> Optional[BrokerOrder]:
        """
        Get current status of an order.

        Args:
            order_id: Client order ID

        Returns:
            BrokerOrder if found
        """
        broker_order_id = self._order_mapping.get(order_id, order_id)
        return await self.broker.get_order(broker_order_id)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[BrokerOrder]:
        """Get all open orders."""
        return await self.broker.get_open_orders(symbol)

    async def get_positions(self) -> List[BrokerPosition]:
        """Get all positions from broker."""
        return await self.broker.get_positions()

    async def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """Get position for a symbol."""
        return await self.broker.get_position(symbol)

    async def close_position(
        self,
        symbol: str,
        quantity: Optional[float] = None,
    ) -> Optional[BrokerOrder]:
        """
        Close a position.

        Args:
            symbol: Symbol to close
            quantity: Optional quantity (None = close all)

        Returns:
            Closing order if successful
        """
        return await self.broker.close_position(symbol, quantity)

    async def close_all_positions(self) -> List[BrokerOrder]:
        """Close all open positions."""
        orders = await self.broker.close_all_positions()
        logger.info(f"Closed {len(orders)} positions")
        return orders

    async def reconcile_positions(self) -> Dict[str, Any]:
        """
        Reconcile positions with broker.

        Returns:
            Reconciliation report
        """
        try:
            broker_positions = await self.broker.get_positions()

            # Build position map
            broker_map = {p.symbol: p for p in broker_positions}

            mismatches = []
            unknown_positions = []

            # Check for unknown positions
            for symbol, position in broker_map.items():
                if symbol not in self._known_positions:
                    unknown_positions.append(symbol)
                    self._known_positions[symbol] = position

            if unknown_positions:
                self._stats["reconciliation_mismatches"] += len(unknown_positions)
                logger.warning(f"Unknown positions found: {unknown_positions}")

                if self.config.auto_close_unknown_positions:
                    for symbol in unknown_positions:
                        try:
                            await self.broker.close_position(symbol)
                            logger.info(f"Auto-closed unknown position: {symbol}")
                        except BrokerError as e:
                            logger.error(f"Failed to close unknown position {symbol}: {e}")

            return {
                "broker_positions": len(broker_positions),
                "known_positions": len(self._known_positions),
                "unknown_positions": unknown_positions,
                "mismatches": mismatches,
            }

        except BrokerError as e:
            logger.error(f"Position reconciliation failed: {e}")
            raise ReconciliationError(f"Position reconciliation failed: {e}")

    async def _reconciliation_loop(self) -> None:
        """Background loop for position reconciliation."""
        while self._running:
            try:
                await asyncio.sleep(self.config.reconciliation_interval_seconds)

                if self.broker.is_connected:
                    await self.reconcile_positions()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Reconciliation error: {e}")

    def _map_order_type(self, order_type: OrderType) -> str:
        """Map internal order type to broker order type."""
        mapping = {
            OrderType.MARKET: "market",
            OrderType.LIMIT: "limit",
            OrderType.STOP: "stop",
            OrderType.STOP_LIMIT: "stop_limit",
        }
        return mapping.get(order_type, "market")

    def on_fill(self, callback: Callable[[Order, BrokerOrder], None]) -> None:
        """Register callback for order fills."""
        self._on_fill.append(callback)

    def on_reject(self, callback: Callable[[Order, str], None]) -> None:
        """Register callback for order rejections."""
        self._on_reject.append(callback)

    def on_error(self, callback: Callable[[str, Exception], None]) -> None:
        """Register callback for errors."""
        self._on_error.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get execution engine statistics."""
        return {
            **self._stats,
            "pending_orders": len(self._pending_orders),
            "submitted_orders": len(self._submitted_orders),
            "known_positions": len(self._known_positions),
            "broker_connected": self.broker.is_connected,
            "running": self._running,
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        for key in self._stats:
            self._stats[key] = 0


class BrokerExecutor:
    """
    Adapter to integrate ProductionExecutionEngine with OrderManager.

    Provides the same interface as SimulationExecutor for use with OrderManager.
    """

    def __init__(self, production_engine: ProductionExecutionEngine):
        """
        Initialize broker executor.

        Args:
            production_engine: Production execution engine
        """
        self.engine = production_engine
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
            return self._loop

    def submit_order(self, order: Order) -> Dict[str, Any]:
        """
        Submit order synchronously.

        Args:
            order: Order to submit

        Returns:
            Order result dictionary
        """
        loop = self._get_loop()

        try:
            broker_order = loop.run_until_complete(
                self.engine.submit_order(order)
            )

            return {
                "success": True,
                "order_id": broker_order.order_id,
                "status": broker_order.status,
                "filled_quantity": broker_order.filled_quantity,
                "average_fill_price": broker_order.average_fill_price,
            }

        except (ValidationError, BrokerError) as e:
            return {
                "success": False,
                "error": str(e),
            }

    def cancel_order(self, order_id: str) -> bool:
        """Cancel order synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(self.engine.cancel_order(order_id))

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status synchronously."""
        loop = self._get_loop()
        broker_order = loop.run_until_complete(
            self.engine.get_order_status(order_id)
        )

        if broker_order:
            return broker_order.to_dict()
        return None
