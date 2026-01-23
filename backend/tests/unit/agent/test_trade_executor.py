"""Unit tests for TradeExecutor.

Tests trade execution, position sizing, and database safety mechanisms.
CRITICAL: Tests for dangerous fallbacks and orphaned trade handling.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import from conftest
from .conftest import AgentConfig, Base

from trading.brokers.base import (
    BrokerError,
    OrderRejectedError,
    InsufficientFundsError,
    BrokerOrder,
    BrokerPosition,
)
from trading.signals.actions import TradingSignal, Action
from api.database.models import Trade

from agent.trade_executor import TradeExecutor
from agent.broker_manager import BrokerManager
from agent.models import TradeResult, PositionStatus


@pytest.fixture
def in_memory_db():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal


@pytest.fixture
def agent_config():
    """Create agent config for testing."""
    return AgentConfig(
        mode="paper",
        max_position_size=0.1,
        use_kelly_sizing=False,
    )


@pytest.fixture
def mock_broker_manager():
    """Create mock broker manager."""
    manager = Mock(spec=BrokerManager)
    manager.is_connected = Mock(return_value=True)
    manager.get_account_info = AsyncMock(return_value={
        "account_id": "12345678",
        "equity": 100000.0,
        "balance": 100000.0,
        "margin_available": 50000.0,
    })
    manager.get_open_positions = AsyncMock(return_value=[])

    # Mock broker
    mock_broker = AsyncMock()
    mock_broker.submit_order = AsyncMock()
    mock_broker.close_position = AsyncMock()
    manager.broker = mock_broker

    return manager


@pytest.fixture
def sample_signal():
    """Create sample trading signal."""
    return TradingSignal(
        action=Action.BUY,
        symbol="EURUSD",
        timestamp=datetime.now(),
        confidence=0.75,
        direction_probability=0.75,
        position_size_pct=0.05,
        stop_loss_price=1.08000,
        take_profit_price=1.09000,
        risk_reward_ratio=2.0,
    )


class TestTradeExecutorPositionSizing:
    """Test position sizing calculations - CRITICAL safety tests."""

    @pytest.mark.asyncio
    async def test_position_size_with_kelly_criterion(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test Kelly Criterion position sizing."""
        agent_config.use_kelly_sizing = True

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        # Calculate position size
        equity = 100000.0
        position_size = executor._calculate_position_size(sample_signal, equity)

        # Kelly = (odds * p - q) / odds
        # odds = 2.0, p = 0.75, q = 0.25
        # kelly = (2.0 * 0.75 - 0.25) / 2.0 = 0.625
        # capped at 0.25 = 25%
        expected_kelly = 0.25
        expected_notional = equity * expected_kelly
        expected_lots = expected_notional / (sample_signal.stop_loss_price * 100000)

        assert position_size == pytest.approx(expected_lots, rel=0.01)

    @pytest.mark.asyncio
    async def test_position_size_with_fixed_percentage(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test fixed percentage position sizing."""
        agent_config.use_kelly_sizing = False

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        equity = 100000.0
        position_size = executor._calculate_position_size(sample_signal, equity)

        # Fixed: signal's position_size_pct = 0.05 = 5%
        expected_notional = equity * 0.05
        expected_lots = expected_notional / (sample_signal.stop_loss_price * 100000)

        assert position_size == pytest.approx(expected_lots, rel=0.01)

    @pytest.mark.asyncio
    async def test_position_size_capped_at_maximum(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test position size is capped at max_position_size."""
        agent_config.use_kelly_sizing = False
        agent_config.max_position_size = 0.02  # 2% max

        # Signal wants 5%
        sample_signal.position_size_pct = 0.05

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        equity = 100000.0
        position_size = executor._calculate_position_size(sample_signal, equity)

        # Should be capped at 2%
        expected_notional = equity * 0.02
        expected_lots = expected_notional / (sample_signal.stop_loss_price * 100000)

        assert position_size == pytest.approx(expected_lots, rel=0.01)

    @pytest.mark.asyncio
    async def test_position_size_fails_without_stop_loss(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """CRITICAL: Position sizing uses stop_loss_price - verify fallback to 1.0."""
        sample_signal.stop_loss_price = None

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        equity = 100000.0
        position_size = executor._calculate_position_size(sample_signal, equity)

        # Should use fallback of 1.0
        expected_notional = equity * sample_signal.position_size_pct
        expected_lots = expected_notional / (1.0 * 100000)

        assert position_size == pytest.approx(expected_lots, rel=0.01)


class TestTradeExecutorExecution:
    """Test trade execution flow."""

    @pytest.mark.asyncio
    async def test_execute_signal_success_paper_mode(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test successful trade execution in paper mode."""
        agent_config.mode = "paper"

        # Mock successful order
        mock_broker_manager.broker.submit_order = AsyncMock(return_value=BrokerOrder(
            order_id="123456",
            client_order_id="client_123",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.05,
            filled_quantity=0.05,
            remaining_quantity=0.0,
            status="filled",
            average_fill_price=1.08500,
            filled_at=datetime.now(),
        ))

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        result = await executor.execute_signal(sample_signal)

        assert result.success is True
        assert result.trade_id is not None
        assert result.mt5_ticket == 123456
        assert result.entry_price == 1.08500

    @pytest.mark.asyncio
    async def test_execute_signal_success_live_mode(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test successful trade execution in live mode."""
        agent_config.mode = "live"

        # Mock successful order
        mock_broker_manager.broker.submit_order = AsyncMock(return_value=BrokerOrder(
            order_id="789",
            client_order_id="client_789",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.05,
            filled_quantity=0.05,
            remaining_quantity=0.0,
            status="filled",
            average_fill_price=1.08500,
        ))

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        result = await executor.execute_signal(sample_signal)

        assert result.success is True
        assert result.trade_id is not None

    @pytest.mark.asyncio
    async def test_execute_signal_invalid_action(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test rejection of invalid signal action."""
        sample_signal.action = Action.HOLD

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        result = await executor.execute_signal(sample_signal)

        assert result.success is False
        assert "Invalid signal action" in result.error

    @pytest.mark.asyncio
    async def test_execute_signal_broker_not_connected(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test execution fails when broker not connected."""
        mock_broker_manager.is_connected = Mock(return_value=False)

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        result = await executor.execute_signal(sample_signal)

        assert result.success is False
        assert "not connected" in result.error

    @pytest.mark.asyncio
    async def test_execute_signal_fails_without_account_info(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """CRITICAL: Execution MUST fail if account info unavailable (no dangerous fallback)."""
        mock_broker_manager.get_account_info = AsyncMock(return_value=None)

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        result = await executor.execute_signal(sample_signal)

        assert result.success is False
        assert "Failed to get account information" in result.error
        # Must NOT submit order
        mock_broker_manager.broker.submit_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_signal_zero_position_size(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test execution fails when position size calculates to zero."""
        # Force zero position size
        sample_signal.position_size_pct = 0.0

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        result = await executor.execute_signal(sample_signal)

        assert result.success is False
        assert "position size is zero" in result.error

    @pytest.mark.asyncio
    async def test_execute_signal_order_rejected(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test handling of order rejection."""
        mock_broker_manager.broker.submit_order = AsyncMock(
            side_effect=OrderRejectedError("Order rejected", reason="INVALID_PRICE")
        )

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        result = await executor.execute_signal(sample_signal)

        assert result.success is False
        assert "Order rejected" in result.error
        assert "INVALID_PRICE" in result.error

    @pytest.mark.asyncio
    async def test_execute_signal_insufficient_funds(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """CRITICAL: Test handling of insufficient margin."""
        mock_broker_manager.broker.submit_order = AsyncMock(
            side_effect=InsufficientFundsError(
                "Insufficient margin", required=5000.0, available=1000.0
            )
        )

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        result = await executor.execute_signal(sample_signal)

        assert result.success is False
        assert "Insufficient funds" in result.error
        assert "required=5000" in result.error
        assert "available=1000" in result.error

    @pytest.mark.asyncio
    async def test_execute_signal_broker_error(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test handling of generic broker errors."""
        mock_broker_manager.broker.submit_order = AsyncMock(
            side_effect=BrokerError("API error")
        )

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        result = await executor.execute_signal(sample_signal)

        assert result.success is False
        assert "Broker error" in result.error

    @pytest.mark.asyncio
    async def test_execute_signal_order_not_filled(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test handling when order is not filled."""
        mock_broker_manager.broker.submit_order = AsyncMock(return_value=BrokerOrder(
            order_id="123",
            client_order_id="client_123",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.05,
            filled_quantity=0.0,
            remaining_quantity=0.05,
            status="pending",
        ))

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        result = await executor.execute_signal(sample_signal)

        assert result.success is False
        assert "Order not filled" in result.error


class TestTradeExecutorDatabaseSafety:
    """Test database failure recovery - CRITICAL for orphaned trade handling."""

    @pytest.mark.asyncio
    async def test_database_failure_after_execution(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal, caplog
    ):
        """CRITICAL: Handle trade execution + database failure (orphaned trade)."""
        # Mock successful order execution
        mock_broker_manager.broker.submit_order = AsyncMock(return_value=BrokerOrder(
            order_id="999",
            client_order_id="client_999",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.05,
            filled_quantity=0.05,
            remaining_quantity=0.0,
            status="filled",
            average_fill_price=1.08500,
        ))

        # Mock database failure
        def failing_session_factory():
            session = in_memory_db()
            original_add = session.add
            def failing_add(*args, **kwargs):
                raise Exception("Database connection lost")
            session.add = failing_add
            return session

        executor = TradeExecutor(mock_broker_manager, agent_config, failing_session_factory)

        caplog.clear()

        result = await executor.execute_signal(sample_signal)

        # Trade was executed successfully
        assert result.success is True
        assert result.mt5_ticket == 999
        assert result.entry_price == 1.08500

        # But database failed
        assert result.trade_id is None
        assert result.error is not None
        assert "Database error" in result.error

        # CRITICAL: Must log orphaned trade details
        critical_logs = [r for r in caplog.records if r.levelname == "CRITICAL"]
        assert len(critical_logs) > 0

        critical_msg = critical_logs[0].message
        assert "TRADE EXECUTED BUT NOT RECORDED" in critical_msg
        assert "999" in critical_msg  # MT5 ticket
        assert "EURUSD" in critical_msg
        assert "0.05" in critical_msg  # quantity
        assert "1.08500" in critical_msg  # price

    @pytest.mark.asyncio
    async def test_database_record_created_on_success(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test database record is properly created."""
        mock_broker_manager.broker.submit_order = AsyncMock(return_value=BrokerOrder(
            order_id="777",
            client_order_id="client_777",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.05,
            filled_quantity=0.05,
            remaining_quantity=0.0,
            status="filled",
            average_fill_price=1.08500,
        ))

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        result = await executor.execute_signal(sample_signal)

        assert result.success is True
        assert result.trade_id is not None

        # Verify database record
        session = in_memory_db()
        trade = session.query(Trade).filter(Trade.id == result.trade_id).first()
        assert trade is not None
        assert trade.symbol == "EURUSD"
        assert trade.direction == "long"
        assert trade.entry_price == 1.08500
        assert trade.lot_size == 0.05
        assert trade.mt5_ticket == 777
        assert trade.status == "open"
        assert trade.execution_mode == agent_config.mode
        assert trade.broker == "mt5"
        session.close()


class TestTradeExecutorPositionManagement:
    """Test position checking and closing."""

    @pytest.mark.asyncio
    async def test_check_open_positions_returns_positions_to_close(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test checking positions identifies ones that should be closed."""
        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        # Add an open trade
        executor._open_trades[1] = {
            "trade_id": 1,
            "mt5_ticket": 123,
            "symbol": "EURUSD",
            "direction": "long",
            "entry_price": 1.08000,
            "quantity": 0.05,
            "stop_loss_price": 1.07500,
            "take_profit_price": 1.09000,
            "entry_time": datetime.now(),
            "max_bars": 24,
        }

        # Mock broker position (hit take profit)
        mock_broker_manager.get_open_positions = AsyncMock(return_value=[{
            "symbol": "EURUSD",
            "current_price": 1.09100,  # Above TP
            "unrealized_pnl": 110.0,
        }])

        positions_to_close = await executor.check_open_positions()

        assert len(positions_to_close) == 1
        assert positions_to_close[0].trade_id == 1
        assert positions_to_close[0].should_close is True
        assert positions_to_close[0].close_reason == "take_profit"

    @pytest.mark.asyncio
    async def test_check_open_positions_not_connected(
        self, agent_config, mock_broker_manager, in_memory_db
    ):
        """Test check positions when broker not connected."""
        mock_broker_manager.is_connected = Mock(return_value=False)

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        positions_to_close = await executor.check_open_positions()

        assert positions_to_close == []

    @pytest.mark.asyncio
    async def test_check_open_positions_removes_closed_trades(
        self, agent_config, mock_broker_manager, in_memory_db
    ):
        """Test that closed trades are removed from tracking."""
        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        # Add open trade
        executor._open_trades[1] = {
            "trade_id": 1,
            "mt5_ticket": 123,
            "symbol": "EURUSD",
            "direction": "long",
            "entry_price": 1.08000,
        }

        # No positions in broker (already closed)
        mock_broker_manager.get_open_positions = AsyncMock(return_value=[])

        await executor.check_open_positions()

        # Should be removed from tracking
        assert 1 not in executor._open_trades

    @pytest.mark.asyncio
    async def test_close_position_success(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test closing a position successfully."""
        # First execute a trade
        mock_broker_manager.broker.submit_order = AsyncMock(return_value=BrokerOrder(
            order_id="555",
            client_order_id="client_555",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.05,
            filled_quantity=0.05,
            remaining_quantity=0.0,
            status="filled",
            average_fill_price=1.08500,
        ))

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)
        result = await executor.execute_signal(sample_signal)

        trade_id = result.trade_id

        # Mock close order
        mock_broker_manager.broker.close_position = AsyncMock(return_value=BrokerOrder(
            order_id="556",
            client_order_id="close_556",
            symbol="EURUSD",
            side="sell",
            order_type="market",
            quantity=0.05,
            filled_quantity=0.05,
            remaining_quantity=0.0,
            status="filled",
            average_fill_price=1.09000,
        ))

        # Close the position
        success = await executor.close_position(trade_id, "take_profit")

        assert success is True
        assert trade_id not in executor._open_trades

    @pytest.mark.asyncio
    async def test_close_position_not_found(
        self, agent_config, mock_broker_manager, in_memory_db
    ):
        """Test closing position that doesn't exist."""
        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        success = await executor.close_position(999, "take_profit")

        assert success is False

    @pytest.mark.asyncio
    async def test_close_all_positions(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test closing all positions."""
        # Execute two trades
        mock_broker_manager.broker.submit_order = AsyncMock(return_value=BrokerOrder(
            order_id="100",
            client_order_id="client_100",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=0.05,
            filled_quantity=0.05,
            remaining_quantity=0.0,
            status="filled",
            average_fill_price=1.08500,
        ))

        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        await executor.execute_signal(sample_signal)
        await executor.execute_signal(sample_signal)

        assert executor.get_open_trade_count() == 2

        # Mock close
        mock_broker_manager.broker.close_position = AsyncMock(return_value=BrokerOrder(
            order_id="200",
            client_order_id="close_200",
            symbol="EURUSD",
            side="sell",
            order_type="market",
            quantity=0.05,
            filled_quantity=0.05,
            remaining_quantity=0.0,
            status="filled",
            average_fill_price=1.09000,
        ))

        closed_count = await executor.close_all_positions("shutdown")

        assert closed_count == 2
        assert executor.get_open_trade_count() == 0

    def test_get_open_trade_count(
        self, agent_config, mock_broker_manager, in_memory_db
    ):
        """Test getting open trade count."""
        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        assert executor.get_open_trade_count() == 0

        executor._open_trades[1] = {"trade_id": 1}
        executor._open_trades[2] = {"trade_id": 2}

        assert executor.get_open_trade_count() == 2

    def test_get_open_trades(
        self, agent_config, mock_broker_manager, in_memory_db
    ):
        """Test getting list of open trades."""
        executor = TradeExecutor(mock_broker_manager, agent_config, in_memory_db)

        trade1 = {"trade_id": 1, "symbol": "EURUSD"}
        trade2 = {"trade_id": 2, "symbol": "GBPUSD"}

        executor._open_trades[1] = trade1
        executor._open_trades[2] = trade2

        trades = executor.get_open_trades()

        assert len(trades) == 2
        assert trade1 in trades
        assert trade2 in trades


class TestTradeExecutorDatabaseTimeouts:
    """Test database timeout protection - HIGH PRIORITY FIX."""

    @pytest.mark.asyncio
    async def test_store_trade_timeout_creates_orphaned_trade(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test that timeout on _store_trade creates orphaned trade entry."""
        executor = TradeExecutor(
            broker_manager=mock_broker_manager,
            config=agent_config,
            db_session_factory=in_memory_db,
        )

        # Mock successful order submission
        filled_order = BrokerOrder(
            order_id="MT5_12345",
            symbol="EURUSD",
            order_type="market",
            side="buy",
            quantity=0.1,
            status="filled",
            filled_quantity=0.1,
            average_fill_price=1.08500,
        )
        mock_broker_manager.broker.submit_order.return_value = filled_order

        # Mock get_current_price to return valid price
        mock_broker_manager.get_current_price = AsyncMock(return_value={
            "bid": 1.08500,
            "ask": 1.08520,
        })

        # Mock _store_trade to timeout
        original_store = executor._store_trade

        def slow_store(*args, **kwargs):
            import time
            time.sleep(15)  # Exceeds 10s timeout
            return original_store(*args, **kwargs)

        with patch.object(executor, '_store_trade', side_effect=slow_store):
            result = await executor.execute_signal(sample_signal)

        # Trade was executed in MT5 but DB store timed out
        assert result.success is True
        assert result.trade_id is None  # No DB record
        assert result.mt5_ticket == 12345  # But MT5 ticket exists
        assert "Database timeout" in result.error

        # Verify orphaned trade was tracked
        assert executor.get_orphaned_trade_count() == 1
        orphaned = executor.get_orphaned_trades()
        assert len(orphaned) == 1
        assert orphaned[0]["mt5_ticket"] == 12345
        assert orphaned[0]["symbol"] == "EURUSD"

    @pytest.mark.asyncio
    async def test_store_trade_completes_within_timeout(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test that normal DB operation completes within timeout."""
        executor = TradeExecutor(
            broker_manager=mock_broker_manager,
            config=agent_config,
            db_session_factory=in_memory_db,
        )

        # Mock successful order submission
        filled_order = BrokerOrder(
            order_id="MT5_12345",
            symbol="EURUSD",
            order_type="market",
            side="buy",
            quantity=0.1,
            status="filled",
            filled_quantity=0.1,
            average_fill_price=1.08500,
        )
        mock_broker_manager.broker.submit_order.return_value = filled_order

        # Mock get_current_price
        mock_broker_manager.get_current_price = AsyncMock(return_value={
            "bid": 1.08500,
            "ask": 1.08520,
        })

        # Execute signal - should complete normally
        result = await executor.execute_signal(sample_signal)

        # Trade should be successful
        assert result.success is True
        assert result.trade_id is not None  # DB record created
        assert result.mt5_ticket == 12345
        assert result.error is None

        # No orphaned trades
        assert executor.get_orphaned_trade_count() == 0

    @pytest.mark.asyncio
    async def test_update_trade_exit_timeout_still_closes_position(
        self, agent_config, mock_broker_manager, in_memory_db
    ):
        """Test that timeout on exit update still closes position in MT5."""
        executor = TradeExecutor(
            broker_manager=mock_broker_manager,
            config=agent_config,
            db_session_factory=in_memory_db,
        )

        # Add an open trade
        executor._open_trades[1] = {
            "trade_id": 1,
            "mt5_ticket": 12345,
            "symbol": "EURUSD",
            "direction": "long",
            "entry_price": 1.08500,
            "quantity": 0.1,
            "stop_loss_price": 1.08000,
            "take_profit_price": 1.09000,
            "entry_time": datetime.now(),
            "max_bars": 24,
        }

        # Mock successful position close in MT5
        close_order = BrokerOrder(
            order_id="MT5_12346",
            symbol="EURUSD",
            order_type="market",
            side="sell",
            quantity=0.1,
            status="filled",
            filled_quantity=0.1,
            average_fill_price=1.08700,
        )
        mock_broker_manager.broker.close_position.return_value = close_order

        # Mock _update_trade_exit to timeout
        original_update = executor._update_trade_exit

        def slow_update(*args, **kwargs):
            import time
            time.sleep(15)  # Exceeds 10s timeout
            return original_update(*args, **kwargs)

        with patch.object(executor, '_update_trade_exit', side_effect=slow_update):
            result = await executor.close_position(1, "manual_close")

        # Position was closed in MT5 despite DB timeout
        assert result is True
        assert 1 not in executor._open_trades  # Position removed from tracking

    @pytest.mark.asyncio
    async def test_store_trade_timeout_error_handling(
        self, agent_config, mock_broker_manager, in_memory_db, sample_signal
    ):
        """Test error handling for database timeout exceptions."""
        executor = TradeExecutor(
            broker_manager=mock_broker_manager,
            config=agent_config,
            db_session_factory=in_memory_db,
        )

        # Mock successful order submission
        filled_order = BrokerOrder(
            order_id="MT5_12345",
            symbol="EURUSD",
            order_type="market",
            side="buy",
            quantity=0.1,
            status="filled",
            filled_quantity=0.1,
            average_fill_price=1.08500,
        )
        mock_broker_manager.broker.submit_order.return_value = filled_order

        # Mock get_current_price
        mock_broker_manager.get_current_price = AsyncMock(return_value={
            "bid": 1.08500,
            "ask": 1.08520,
        })

        # Mock _store_trade to raise TimeoutError
        import asyncio
        with patch.object(executor, '_store_trade', side_effect=asyncio.TimeoutError):
            # Wrap in asyncio.wait_for simulation
            with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError):
                result = await executor.execute_signal(sample_signal)

        # Verify error is handled gracefully
        assert result.success is True  # Trade executed in MT5
        assert "timeout" in result.error.lower() or "Database" in result.error

    @pytest.mark.asyncio
    async def test_database_timeout_constant_value(self):
        """Test that DB_TIMEOUT_SECONDS constant is properly defined."""
        from agent.trade_executor import DB_TIMEOUT_SECONDS

        # Verify timeout is set to 10 seconds as per fix
        assert DB_TIMEOUT_SECONDS == 10.0
        assert isinstance(DB_TIMEOUT_SECONDS, float)

    @pytest.mark.asyncio
    async def test_orphaned_trade_retry_success(
        self, agent_config, mock_broker_manager, in_memory_db
    ):
        """Test that orphaned trade retry can succeed."""
        executor = TradeExecutor(
            broker_manager=mock_broker_manager,
            config=agent_config,
            db_session_factory=in_memory_db,
        )

        # Create an orphaned trade entry
        orphaned_trade = {
            "mt5_ticket": 12345,
            "symbol": "EURUSD",
            "side": "buy",
            "quantity": 0.1,
            "entry_price": 1.08500,
            "entry_time": datetime.now(),
            "confidence": 0.75,
            "stop_loss": 1.08000,
            "take_profit": 1.09000,
            "db_error": "Timeout",
            "retry_count": 0,
        }
        executor._orphaned_trades.append(orphaned_trade)

        # Attempt immediate retry
        trade_id = await executor._retry_store_orphaned_trade(orphaned_trade)

        # Verify retry succeeded
        assert trade_id is not None
        assert isinstance(trade_id, int)

        # Verify orphaned trade was removed from queue
        assert orphaned_trade not in executor._orphaned_trades

    @pytest.mark.asyncio
    async def test_orphaned_trade_retry_failure(
        self, agent_config, mock_broker_manager, in_memory_db
    ):
        """Test that orphaned trade retry handles failures gracefully."""
        executor = TradeExecutor(
            broker_manager=mock_broker_manager,
            config=agent_config,
            db_session_factory=in_memory_db,
        )

        # Create an orphaned trade entry that will fail to store
        orphaned_trade = {
            "mt5_ticket": 12345,
            "symbol": "EURUSD",
            "side": "buy",
            "quantity": 0.1,
            "entry_price": 1.08500,
            "entry_time": datetime.now(),
            "confidence": 0.75,
            "stop_loss": 1.08000,
            "take_profit": 1.09000,
            "db_error": "Timeout",
            "retry_count": 0,
        }

        # Mock DB session factory to raise exception
        def failing_session():
            raise Exception("Database connection failed")

        executor_failing = TradeExecutor(
            broker_manager=mock_broker_manager,
            config=agent_config,
            db_session_factory=failing_session,
        )
        executor_failing._orphaned_trades.append(orphaned_trade.copy())

        # Attempt retry - should fail gracefully
        trade_id = await executor_failing._retry_store_orphaned_trade(
            executor_failing._orphaned_trades[0]
        )

        # Verify retry failed
        assert trade_id is None

        # Verify retry_count was incremented
        assert executor_failing._orphaned_trades[0]["retry_count"] == 1
