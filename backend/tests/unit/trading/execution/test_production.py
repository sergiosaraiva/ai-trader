"""Tests for production execution engine."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.trading.execution.production import (
    ProductionExecutionEngine,
    ProductionConfig,
    OrderValidator,
    OrderValidationConfig,
    OrderValidationResult,
    ValidationError,
    ReconciliationError,
    BrokerExecutor,
)
from src.trading.orders.manager import Order, OrderType, OrderSide, OrderStatus
from src.trading.brokers.base import (
    BrokerAdapter,
    BrokerOrder,
    BrokerPosition,
    Quote,
    AccountInfo,
    BrokerError,
    OrderRejectedError,
)


class TestOrderValidationConfig:
    """Tests for OrderValidationConfig."""

    def test_default_config(self):
        """Test default validation configuration."""
        config = OrderValidationConfig()

        assert config.max_order_size == 1000000.0
        assert config.min_order_size == 1.0
        assert config.max_orders_per_minute == 60
        assert config.require_kill_switch_check is True

    def test_custom_config(self):
        """Test custom validation configuration."""
        config = OrderValidationConfig(
            max_order_size=10000.0,
            min_order_size=10.0,
            allowed_symbols=["EURUSD", "GBPUSD"],
        )

        assert config.max_order_size == 10000.0
        assert config.min_order_size == 10.0
        assert "EURUSD" in config.allowed_symbols


class TestOrderValidationResult:
    """Tests for OrderValidationResult."""

    def test_valid_result(self):
        """Test valid result."""
        result = OrderValidationResult(is_valid=True)

        assert result.is_valid is True
        assert bool(result) is True
        assert len(result.violations) == 0

    def test_invalid_result(self):
        """Test invalid result with violations."""
        result = OrderValidationResult(
            is_valid=False,
            violations=["Order size too large", "Symbol not allowed"],
        )

        assert result.is_valid is False
        assert bool(result) is False
        assert len(result.violations) == 2


class TestOrderValidator:
    """Tests for OrderValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator for testing."""
        config = OrderValidationConfig(
            max_order_size=10000.0,
            min_order_size=10.0,
            max_order_value=100000.0,
            max_position_size=500000.0,
            allowed_symbols=["EURUSD", "GBPUSD"],
            blocked_symbols=["BLOCKED"],
            max_price_deviation_pct=5.0,
            max_orders_per_minute=10,
            max_orders_per_day=100,
        )
        return OrderValidator(config)

    @pytest.fixture
    def valid_order(self):
        """Create valid test order."""
        return Order(
            order_id="TEST001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

    def test_valid_order(self, validator, valid_order):
        """Test validation of valid order."""
        result = validator.validate(
            order=valid_order,
            current_price=1.1000,
        )

        assert result.is_valid is True

    def test_order_too_small(self, validator, valid_order):
        """Test validation fails for too small order."""
        valid_order.quantity = 1.0  # Below min of 10

        result = validator.validate(
            order=valid_order,
            current_price=1.1000,
        )

        assert result.is_valid is False
        assert any("below minimum" in v for v in result.violations)

    def test_order_too_large(self, validator, valid_order):
        """Test validation fails for too large order."""
        valid_order.quantity = 50000.0  # Above max of 10000

        result = validator.validate(
            order=valid_order,
            current_price=1.1000,
        )

        assert result.is_valid is False
        assert any("exceeds maximum" in v for v in result.violations)

    def test_symbol_not_allowed(self, validator, valid_order):
        """Test validation fails for non-allowed symbol."""
        valid_order.symbol = "USDJPY"  # Not in allowed list

        result = validator.validate(
            order=valid_order,
            current_price=1.1000,
        )

        assert result.is_valid is False
        assert any("not in allowed list" in v for v in result.violations)

    def test_symbol_blocked(self, validator, valid_order):
        """Test validation fails for blocked symbol."""
        valid_order.symbol = "BLOCKED"

        # Add to allowed list first
        validator.config.allowed_symbols.append("BLOCKED")

        result = validator.validate(
            order=valid_order,
            current_price=1.1000,
        )

        assert result.is_valid is False
        assert any("blocked" in v for v in result.violations)

    def test_kill_switch_active(self, validator, valid_order):
        """Test validation fails when kill switch is active."""
        result = validator.validate(
            order=valid_order,
            current_price=1.1000,
            kill_switch_active=True,
        )

        assert result.is_valid is False
        assert any("Kill switch" in v for v in result.violations)

    def test_order_value_exceeded(self, validator, valid_order):
        """Test validation fails for excessive order value."""
        valid_order.quantity = 5000.0

        result = validator.validate(
            order=valid_order,
            current_price=50.0,  # 5000 * 50 = 250000 > 100000
        )

        assert result.is_valid is False
        assert any("Order value" in v for v in result.violations)

    def test_rate_limit_exceeded(self, validator, valid_order):
        """Test validation fails when rate limit exceeded."""
        # Record 10 orders to hit the limit
        for _ in range(10):
            validator.record_order(valid_order, 1.1000)

        result = validator.validate(
            order=valid_order,
            current_price=1.1000,
        )

        assert result.is_valid is False
        assert any("Rate limit" in v for v in result.violations)

    def test_insufficient_buying_power(self, validator, valid_order):
        """Test validation fails with insufficient buying power."""
        account_info = AccountInfo(
            account_id="ACC123",
            currency="USD",
            balance=1000.0,
            equity=1000.0,
            margin_used=0.0,
            margin_available=1000.0,
            buying_power=100.0,  # Very low
            cash=1000.0,
        )

        result = validator.validate(
            order=valid_order,
            current_price=1.1000,  # Order value = 100 * 1.1 = 110
            account_info=account_info,
        )

        assert result.is_valid is False
        assert any("buying power" in v for v in result.violations)

    def test_price_deviation_warning(self, validator):
        """Test price deviation generates warning."""
        order = Order(
            order_id="TEST001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100.0,
            limit_price=1.2000,  # 9% above current
        )

        result = validator.validate(
            order=order,
            current_price=1.1000,
        )

        assert len(result.warnings) > 0
        assert any("deviates" in w for w in result.warnings)

    def test_record_order(self, validator, valid_order):
        """Test order recording for rate limiting."""
        validator.record_order(valid_order, 1.1000)

        assert len(validator._orders_this_minute) == 1
        assert len(validator._orders_today) == 1
        assert validator._daily_volume == 100.0 * 1.1000

    def test_reset_daily_counters(self, validator, valid_order):
        """Test daily counter reset."""
        validator.record_order(valid_order, 1.1000)
        validator.reset_daily_counters()

        assert len(validator._orders_today) == 0
        assert validator._daily_volume == 0.0


class TestProductionConfig:
    """Tests for ProductionConfig."""

    def test_default_config(self):
        """Test default production configuration."""
        config = ProductionConfig()

        assert config.reconciliation_interval_seconds == 60.0
        assert config.auto_reconnect is True
        assert config.order_timeout_seconds == 30.0


class TestProductionExecutionEngine:
    """Tests for ProductionExecutionEngine."""

    @pytest.fixture
    def mock_broker(self):
        """Create mock broker."""
        broker = Mock(spec=BrokerAdapter)
        broker.is_connected = True

        # Mock async methods
        broker.get_quote = AsyncMock(return_value=Quote(
            symbol="EURUSD",
            bid=1.0999,
            ask=1.1001,
            bid_size=100000.0,
            ask_size=100000.0,
            last=1.1000,
            last_size=10000.0,
            timestamp=datetime.now(),
        ))

        broker.get_account = AsyncMock(return_value=AccountInfo(
            account_id="ACC123",
            currency="USD",
            balance=100000.0,
            equity=100000.0,
            margin_used=0.0,
            margin_available=100000.0,
            buying_power=400000.0,
            cash=100000.0,
        ))

        broker.get_position = AsyncMock(return_value=None)
        broker.get_positions = AsyncMock(return_value=[])

        broker.submit_order = AsyncMock(return_value=BrokerOrder(
            order_id="BROKER001",
            client_order_id="TEST001",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=100.0,
            filled_quantity=100.0,
            remaining_quantity=0.0,
            average_fill_price=1.1000,
            status="filled",
        ))

        broker.cancel_order = AsyncMock(return_value=True)
        broker.get_order = AsyncMock(return_value=None)
        broker.get_open_orders = AsyncMock(return_value=[])

        return broker

    @pytest.fixture
    def engine(self, mock_broker):
        """Create production execution engine."""
        config = ProductionConfig(
            validation_config=OrderValidationConfig(
                allowed_symbols=[],  # Allow all
                require_kill_switch_check=True,
            ),
        )
        return ProductionExecutionEngine(
            broker=mock_broker,
            config=config,
            kill_switch_callback=lambda: False,
        )

    @pytest.fixture
    def valid_order(self):
        """Create valid test order."""
        return Order(
            order_id="TEST001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

    @pytest.mark.asyncio
    async def test_submit_order_success(self, engine, valid_order, mock_broker):
        """Test successful order submission."""
        result = await engine.submit_order(valid_order)

        assert result.order_id == "BROKER001"
        assert result.is_filled is True
        mock_broker.submit_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_submit_order_validation_failure(self, engine, mock_broker):
        """Test order submission with validation failure."""
        engine._kill_switch_callback = lambda: True  # Kill switch active

        order = Order(
            order_id="TEST001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        with pytest.raises(ValidationError) as exc_info:
            await engine.submit_order(order)

        assert "Kill switch" in str(exc_info.value)
        mock_broker.submit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_submit_order_broker_rejection(self, engine, valid_order, mock_broker):
        """Test order submission with broker rejection."""
        mock_broker.submit_order = AsyncMock(
            side_effect=OrderRejectedError("Order rejected", reason="Test rejection")
        )

        with pytest.raises(OrderRejectedError):
            await engine.submit_order(valid_order)

        assert engine._stats["orders_rejected"] == 1

    @pytest.mark.asyncio
    async def test_cancel_order(self, engine, mock_broker):
        """Test order cancellation."""
        # First submit an order
        order = Order(
            order_id="TEST001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )
        await engine.submit_order(order)

        # Then cancel it
        result = await engine.cancel_order("TEST001")

        assert result is True
        mock_broker.cancel_order.assert_called()

    @pytest.mark.asyncio
    async def test_get_positions(self, engine, mock_broker):
        """Test getting positions."""
        mock_broker.get_positions = AsyncMock(return_value=[
            BrokerPosition(
                symbol="EURUSD",
                quantity=100.0,
                side="long",
                average_entry_price=1.1000,
                current_price=1.1050,
                market_value=11050.0,
                cost_basis=11000.0,
                unrealized_pnl=50.0,
                unrealized_pnl_pct=0.45,
            ),
        ])

        positions = await engine.get_positions()

        assert len(positions) == 1
        assert positions[0].symbol == "EURUSD"

    @pytest.mark.asyncio
    async def test_fill_callback(self, engine, valid_order, mock_broker):
        """Test fill callback is called."""
        callback = Mock()
        engine.on_fill(callback)

        await engine.submit_order(valid_order)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_reject_callback(self, engine, mock_broker):
        """Test reject callback is called."""
        callback = Mock()
        engine.on_reject(callback)
        engine._kill_switch_callback = lambda: True

        order = Order(
            order_id="TEST001",
            symbol="EURUSD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
        )

        with pytest.raises(ValidationError):
            await engine.submit_order(order)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconcile_positions(self, engine, mock_broker):
        """Test position reconciliation."""
        mock_broker.get_positions = AsyncMock(return_value=[
            BrokerPosition(
                symbol="EURUSD",
                quantity=100.0,
                side="long",
                average_entry_price=1.1000,
                current_price=1.1050,
                market_value=11050.0,
                cost_basis=11000.0,
                unrealized_pnl=50.0,
                unrealized_pnl_pct=0.45,
            ),
        ])

        result = await engine.reconcile_positions()

        assert result["broker_positions"] == 1
        assert "EURUSD" in result["unknown_positions"]

    def test_get_stats(self, engine):
        """Test getting engine statistics."""
        stats = engine.get_stats()

        assert "orders_submitted" in stats
        assert "orders_filled" in stats
        assert "validation_failures" in stats
        assert stats["broker_connected"] is True

    @pytest.mark.asyncio
    async def test_start_stop(self, engine):
        """Test engine start and stop."""
        await engine.start()
        assert engine._running is True

        await engine.stop()
        assert engine._running is False


class TestBrokerExecutor:
    """Tests for BrokerExecutor adapter."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock production engine."""
        engine = Mock(spec=ProductionExecutionEngine)
        return engine

    def test_initialization(self, mock_engine):
        """Test broker executor initialization."""
        executor = BrokerExecutor(mock_engine)
        assert executor.engine == mock_engine
