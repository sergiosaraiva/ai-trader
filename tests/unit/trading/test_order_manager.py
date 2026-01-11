"""Tests for Order Manager."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from src.trading.orders.manager import (
    OrderManager,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    OrderResult,
    BracketOrder,
    ExecutionMode,
    SimulationExecutor,
)
from src.trading.signals.actions import TradingSignal, Action


class TestOrder:
    """Tests for Order dataclass."""

    def test_order_creation(self):
        """Test basic order creation."""
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )

        assert order.order_id == "test-001"
        assert order.symbol == "EURUSD"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.MARKET
        assert order.quantity == 1.0
        assert order.status == OrderStatus.PENDING

    def test_order_is_open(self):
        """Test is_open property."""
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            status=OrderStatus.PENDING,
        )
        assert order.is_open is True

        order.status = OrderStatus.FILLED
        assert order.is_open is False

        order.status = OrderStatus.CANCELLED
        assert order.is_open is False

    def test_order_is_filled(self):
        """Test is_filled property."""
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )

        assert order.is_filled is False
        order.status = OrderStatus.FILLED
        assert order.is_filled is True

    def test_order_remaining_quantity(self):
        """Test remaining quantity calculation."""
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            filled_quantity=0.3,
        )

        assert order.remaining_quantity == pytest.approx(0.7)

    def test_order_fill_percentage(self):
        """Test fill percentage calculation."""
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
            filled_quantity=0.5,
        )

        assert order.fill_percentage == pytest.approx(50.0)

    def test_order_to_dict(self):
        """Test order serialization."""
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )

        d = order.to_dict()
        assert d["order_id"] == "test-001"
        assert d["symbol"] == "EURUSD"
        assert d["side"] == "buy"
        assert d["order_type"] == "market"


class TestSimulationExecutor:
    """Tests for SimulationExecutor."""

    def test_submit_market_order(self):
        """Test market order fills immediately."""
        executor = SimulationExecutor(
            base_spread_pct=0.0002,
            slippage_pct=0.0001,
        )

        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )

        result = executor.submit_order(order)

        assert result.success is True
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 1.0
        assert result.fill_price is not None

    def test_submit_limit_order(self):
        """Test limit order accepted but not immediately filled."""
        executor = SimulationExecutor()

        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=1.0950,
        )

        result = executor.submit_order(order)

        assert result.success is True
        assert result.status == OrderStatus.ACCEPTED
        assert result.filled_quantity == 0

    def test_cancel_order(self):
        """Test order cancellation."""
        executor = SimulationExecutor()

        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=1.0950,
        )

        executor.submit_order(order)
        success = executor.cancel_order("test-001")

        assert success is True
        assert order.status == OrderStatus.CANCELLED

    def test_check_limit_orders_buy(self):
        """Test limit order fill when price hits."""
        executor = SimulationExecutor(get_price_callback=lambda s: 1.0)

        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=1.0100,
        )

        executor.submit_order(order)

        # Price above limit - should not fill
        results = executor.check_limit_orders("EURUSD", 1.0200)
        assert len(results) == 0

        # Price at or below limit - should fill
        results = executor.check_limit_orders("EURUSD", 1.0050)
        assert len(results) == 1
        assert results[0].status == OrderStatus.FILLED

    def test_check_stop_orders_sell(self):
        """Test stop order fill when price hits."""
        executor = SimulationExecutor(get_price_callback=lambda s: 1.0)

        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.SELL,
            order_type=OrderType.STOP,
            quantity=1.0,
            stop_price=1.0000,
        )

        executor.submit_order(order)

        # Price above stop - should not fill
        results = executor.check_limit_orders("EURUSD", 1.0100)
        assert len(results) == 0

        # Price at or below stop - should fill
        results = executor.check_limit_orders("EURUSD", 0.9900)
        assert len(results) == 1
        assert results[0].status == OrderStatus.FILLED


class TestOrderManager:
    """Tests for OrderManager."""

    def test_initialization(self):
        """Test order manager initialization."""
        manager = OrderManager(execution_mode=ExecutionMode.SIMULATION)

        assert manager.execution_mode == ExecutionMode.SIMULATION
        assert len(manager.orders) == 0
        assert len(manager.bracket_orders) == 0

    def test_create_order_from_signal_buy(self):
        """Test order creation from BUY signal."""
        manager = OrderManager(
            execution_mode=ExecutionMode.SIMULATION,
            get_price_callback=lambda s: 1.1000,
        )

        signal = TradingSignal(
            action=Action.BUY,
            symbol="EURUSD",
            timestamp=datetime.now(),
            confidence=0.75,
            direction_probability=0.8,
            position_size_pct=0.02,
        )

        order = manager.create_order(
            signal=signal,
            account_equity=100000,
            lot_size=100000,
        )

        assert order.side == OrderSide.BUY
        assert order.symbol == "EURUSD"
        assert order.order_type == OrderType.MARKET
        assert order.quantity > 0

    def test_create_order_from_signal_sell(self):
        """Test order creation from SELL signal."""
        manager = OrderManager(
            execution_mode=ExecutionMode.SIMULATION,
            get_price_callback=lambda s: 1.1000,
        )

        signal = TradingSignal(
            action=Action.SELL,
            symbol="EURUSD",
            timestamp=datetime.now(),
            confidence=0.75,
            direction_probability=0.2,
            position_size_pct=0.02,
        )

        order = manager.create_order(
            signal=signal,
            account_equity=100000,
            lot_size=100000,
        )

        assert order.side == OrderSide.SELL

    def test_create_bracket_order(self):
        """Test bracket order creation."""
        manager = OrderManager(
            execution_mode=ExecutionMode.SIMULATION,
            get_price_callback=lambda s: 1.1000,
        )

        signal = TradingSignal(
            action=Action.BUY,
            symbol="EURUSD",
            timestamp=datetime.now(),
            confidence=0.75,
            direction_probability=0.8,
            position_size_pct=0.02,
            stop_loss_price=1.0900,
            take_profit_price=1.1200,
        )

        bracket = manager.create_bracket_order(
            signal=signal,
            account_equity=100000,
        )

        assert bracket.entry_order is not None
        assert bracket.entry_order.side == OrderSide.BUY
        assert bracket.stop_loss_order is not None
        assert bracket.stop_loss_order.side == OrderSide.SELL
        assert bracket.stop_loss_order.stop_price == 1.0900
        assert bracket.take_profit_order is not None
        assert bracket.take_profit_order.side == OrderSide.SELL
        assert bracket.take_profit_order.limit_price == 1.1200

    def test_submit_order(self):
        """Test order submission."""
        manager = OrderManager(execution_mode=ExecutionMode.SIMULATION)

        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )

        result = manager.submit_order(order)

        assert result.success is True
        assert "test-001" in manager.orders

    def test_submit_bracket_order(self):
        """Test bracket order submission."""
        manager = OrderManager(
            execution_mode=ExecutionMode.SIMULATION,
            get_price_callback=lambda s: 1.1000,
        )

        signal = TradingSignal(
            action=Action.BUY,
            symbol="EURUSD",
            timestamp=datetime.now(),
            confidence=0.75,
            direction_probability=0.8,
            position_size_pct=0.02,
            stop_loss_price=1.0900,
            take_profit_price=1.1200,
        )

        bracket = manager.create_bracket_order(
            signal=signal,
            account_equity=100000,
        )

        result = manager.submit_bracket_order(bracket)

        assert result.success is True
        assert bracket.is_active is True
        assert bracket.is_filled is True  # Market order fills immediately

    def test_cancel_order(self):
        """Test order cancellation."""
        manager = OrderManager(execution_mode=ExecutionMode.SIMULATION)

        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=1.0950,
        )

        manager.submit_order(order)
        success = manager.cancel_order("test-001")

        assert success is True
        assert manager.orders["test-001"].status == OrderStatus.CANCELLED

    def test_get_open_orders(self):
        """Test getting open orders."""
        manager = OrderManager(execution_mode=ExecutionMode.SIMULATION)

        # Submit a limit order (won't fill immediately)
        order1 = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            limit_price=1.0950,
        )
        manager.submit_order(order1)

        # Submit a market order (fills immediately)
        order2 = Order(
            order_id="test-002",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        manager.submit_order(order2)

        open_orders = manager.get_open_orders()

        # Only limit order should be open
        assert len(open_orders) == 1
        assert open_orders[0].order_id == "test-001"

    def test_on_fill_callback(self):
        """Test fill callback is called."""
        manager = OrderManager(execution_mode=ExecutionMode.SIMULATION)

        filled_orders = []
        manager.on_fill(lambda o: filled_orders.append(o))

        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        manager.submit_order(order)

        assert len(filled_orders) == 1
        assert filled_orders[0].order_id == "test-001"

    def test_get_stats(self):
        """Test getting statistics."""
        manager = OrderManager(execution_mode=ExecutionMode.SIMULATION)

        # Submit some orders
        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        manager.submit_order(order)

        stats = manager.get_stats()

        assert stats["execution_mode"] == "simulation"
        assert stats["total_orders"] == 1
        assert stats["filled_orders"] == 1

    def test_reset(self):
        """Test manager reset."""
        manager = OrderManager(execution_mode=ExecutionMode.SIMULATION)

        order = Order(
            order_id="test-001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        manager.submit_order(order)

        manager.reset()

        assert len(manager.orders) == 0
        assert len(manager.bracket_orders) == 0
