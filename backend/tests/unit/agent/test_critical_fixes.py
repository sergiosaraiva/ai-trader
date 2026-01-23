"""Unit tests for critical bug fixes in the AI Trading Agent.

Tests cover:
1. Position size calculation safety checks
2. Orphaned trade handling
3. Thread safety in safety manager
4. Trade result recording in trading cycle
5. Broker manager price fetching
"""

import asyncio
import threading
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# We need to mock trading modules before importing from conftest
# conftest will handle src.agent setup, we just need src.trading
import importlib.util

# Mock trading modules before any agent imports - create proper hierarchy
# Get or create src module
if "src" not in sys.modules:
    mock_src = type(sys)("src")
    sys.modules["src"] = mock_src
else:
    mock_src = sys.modules["src"]

mock_trading = type(sys)("trading")
mock_brokers = type(sys)("brokers")
mock_brokers_base = type(sys)("base")
mock_signals = type(sys)("signals")
mock_signals_actions = type(sys)("actions")
mock_brokers_mt5 = type(sys)("mt5")

# Add mock classes to broker.base
mock_brokers_base.BrokerConfig = Mock
mock_brokers_base.BrokerType = Mock
mock_brokers_base.ConnectionStatus = Mock
mock_brokers_base.BrokerError = Exception
mock_brokers_base.AuthenticationError = Exception
mock_brokers_base.ConnectionError = Exception
mock_brokers_base.OrderRejectedError = Exception
mock_brokers_base.InsufficientFundsError = Exception
mock_brokers_base.BrokerOrder = Mock
mock_brokers_base.BrokerPosition = Mock

# Add mock classes to signals.actions
class MockActionEnum:
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

mock_signals_actions.Action = MockActionEnum
mock_signals_actions.TradingSignal = Mock

# Add mock MT5Broker
mock_brokers_mt5.MT5Broker = Mock

# Assemble hierarchy
mock_brokers.base = mock_brokers_base
mock_brokers.mt5 = mock_brokers_mt5
mock_signals.actions = mock_signals_actions
mock_trading.brokers = mock_brokers
mock_trading.signals = mock_signals
mock_src.trading = mock_trading

# Register in sys.modules
sys.modules["src.trading"] = mock_trading
sys.modules["src.trading.brokers"] = mock_brokers
sys.modules["src.trading.brokers.base"] = mock_brokers_base
sys.modules["src.trading.brokers.mt5"] = mock_brokers_mt5
sys.modules["src.trading.signals"] = mock_signals
sys.modules["src.trading.signals.actions"] = mock_signals_actions

# Import from conftest (provides SafetyManager and config classes, sets up src.agent)
from .conftest import AgentConfig, SafetyConfig, SafetyManager, SafetyStatus

# Add RiskLevel.MODERATE attribute that safety_manager needs
if "src.trading.risk.profiles" in sys.modules:
    profiles_module = sys.modules["src.trading.risk.profiles"]
    if hasattr(profiles_module, "RiskLevel"):
        # RiskLevel is a Mock, add MODERATE attribute
        profiles_module.RiskLevel.MODERATE = "MODERATE"

# Now load agent modules using importlib to avoid import errors
# Load config first
config_path = src_path / "agent" / "config.py"
spec = importlib.util.spec_from_file_location("src.agent.config", config_path)
config_module = importlib.util.module_from_spec(spec)
sys.modules["src.agent.config"] = config_module
spec.loader.exec_module(config_module)

# Load broker_manager
broker_manager_path = src_path / "agent" / "broker_manager.py"
spec = importlib.util.spec_from_file_location("src.agent.broker_manager", broker_manager_path)
broker_manager_module = importlib.util.module_from_spec(spec)
sys.modules["src.agent.broker_manager"] = broker_manager_module
spec.loader.exec_module(broker_manager_module)

# Load models
models_path = src_path / "agent" / "models.py"
spec = importlib.util.spec_from_file_location("src.agent.models", models_path)
models_module = importlib.util.module_from_spec(spec)
sys.modules["src.agent.models"] = models_module
spec.loader.exec_module(models_module)

# Load trade_executor
trade_executor_path = src_path / "agent" / "trade_executor.py"
spec = importlib.util.spec_from_file_location("src.agent.trade_executor", trade_executor_path)
trade_executor_module = importlib.util.module_from_spec(spec)
sys.modules["src.agent.trade_executor"] = trade_executor_module
spec.loader.exec_module(trade_executor_module)

# Load trading_cycle
trading_cycle_path = src_path / "agent" / "trading_cycle.py"
spec = importlib.util.spec_from_file_location("src.agent.trading_cycle", trading_cycle_path)
trading_cycle_module = importlib.util.module_from_spec(spec)
sys.modules["src.agent.trading_cycle"] = trading_cycle_module
spec.loader.exec_module(trading_cycle_module)

# Now we can import from the loaded modules
TradeExecutor = trade_executor_module.TradeExecutor
BrokerManager = broker_manager_module.BrokerManager
TradingCycle = trading_cycle_module.TradingCycle
PositionStatus = models_module.PositionStatus


# Create mock classes for testing
class MockTradingSignal:
    def __init__(self, **kwargs):
        self.action = kwargs.get("action")
        self.symbol = kwargs.get("symbol", "EURUSD")
        self.timestamp = kwargs.get("timestamp", datetime.now())
        self.confidence = kwargs.get("confidence", 0.75)
        self.direction_probability = kwargs.get("direction_probability", 0.75)
        self.position_size_pct = kwargs.get("position_size_pct", 0.02)
        self.stop_loss_pct = kwargs.get("stop_loss_pct", 0.02)
        self.take_profit_pct = kwargs.get("take_profit_pct", 0.04)
        self.stop_loss_price = kwargs.get("stop_loss_price")
        self.take_profit_price = kwargs.get("take_profit_price")
        self.risk_reward_ratio = kwargs.get("risk_reward_ratio", 2.0)


class MockAction:
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class MockTradeResult:
    """Mock CircuitBreaker TradeResult."""
    def __init__(self, pnl, is_winner, timestamp):
        self.pnl = pnl
        self.is_winner = is_winner
        self.timestamp = timestamp


# ============================================================================
# 1. Position Size Calculation Tests (trade_executor.py)
# ============================================================================


class TestPositionSizeCalculation:
    """Test position size calculation safety checks."""

    @pytest.fixture
    def mock_broker_manager(self):
        """Create mock broker manager."""
        broker = Mock()
        broker.is_connected.return_value = True
        broker_manager = Mock()
        broker_manager.broker = broker
        broker_manager.is_connected.return_value = True
        return broker_manager

    @pytest.fixture
    def mock_db_session_factory(self):
        """Create mock database session factory."""
        def factory():
            session = Mock()
            session.query.return_value.filter.return_value.first.return_value = None
            return session
        return factory

    @pytest.fixture
    def config(self):
        """Create agent config."""
        return AgentConfig(
            mode="paper",
            confidence_threshold=0.65,
            max_position_size=0.05,
            symbol="EURUSD",
            use_kelly_sizing=False,
        )

    @pytest.fixture
    def trade_executor(self, mock_broker_manager, config, mock_db_session_factory):
        """Create TradeExecutor instance."""
        return TradeExecutor(
            broker_manager=mock_broker_manager,
            config=config,
            db_session_factory=mock_db_session_factory,
        )

    @pytest.fixture
    def sample_signal(self):
        """Create sample trading signal."""
        return MockTradingSignal(
            action=MockAction.BUY,
            symbol="EURUSD",
            confidence=0.75,
            direction_probability=0.75,
            position_size_pct=0.02,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
        )

    @pytest.mark.asyncio
    async def test_raises_error_when_no_valid_price_available(
        self, trade_executor, sample_signal, mock_broker_manager
    ):
        """Test that ValueError is raised when no valid price is available."""
        # Mock broker to return None for price
        mock_broker_manager.get_current_price = AsyncMock(return_value=None)

        # Set stop_loss_price to invalid value
        sample_signal.stop_loss_price = None

        # Should raise ValueError
        with pytest.raises(ValueError, match="no valid price available"):
            await trade_executor._calculate_position_size(
                signal=sample_signal,
                equity=100000.0,
            )

    @pytest.mark.asyncio
    async def test_raises_error_when_price_outside_forex_range(
        self, trade_executor, sample_signal, mock_broker_manager
    ):
        """Test that ValueError is raised when price is outside realistic forex range."""
        # Mock broker to return None for price
        mock_broker_manager.get_current_price = AsyncMock(return_value=None)

        # Set stop_loss_price to invalid value (outside 0.5-2.0 range)
        sample_signal.stop_loss_price = 5.0  # Invalid

        # Should raise ValueError
        with pytest.raises(ValueError, match="no valid price available"):
            await trade_executor._calculate_position_size(
                signal=sample_signal,
                equity=100000.0,
            )

    @pytest.mark.asyncio
    async def test_accepts_price_within_forex_range(
        self, trade_executor, sample_signal, mock_broker_manager
    ):
        """Test that price within 0.5-2.0 range is accepted as fallback."""
        # Mock broker to return None for price
        mock_broker_manager.get_current_price = AsyncMock(return_value=None)

        # Set stop_loss_price to valid forex price
        sample_signal.stop_loss_price = 1.0850  # Valid EURUSD price

        # Should not raise error
        lots = await trade_executor._calculate_position_size(
            signal=sample_signal,
            equity=100000.0,
        )

        assert lots > 0

    @pytest.mark.asyncio
    async def test_lot_size_bounded_between_min_and_max(
        self, trade_executor, sample_signal, mock_broker_manager
    ):
        """Test that lot size is bounded between 0.01 and 10.0."""
        # Mock broker to return valid price
        mock_broker_manager.get_current_price = AsyncMock(
            return_value={"bid": 1.0850}
        )

        # Test minimum bound (very small equity)
        lots = await trade_executor._calculate_position_size(
            signal=sample_signal,
            equity=100.0,  # Very small equity
        )
        assert lots >= 0.01, "Lot size should be at least 0.01"

        # Test maximum bound (very large equity)
        lots = await trade_executor._calculate_position_size(
            signal=sample_signal,
            equity=100000000.0,  # Very large equity
        )
        assert lots <= 10.0, "Lot size should not exceed 10.0"

    @pytest.mark.asyncio
    async def test_broker_price_fetched_when_available(
        self, trade_executor, sample_signal, mock_broker_manager
    ):
        """Test that broker price is fetched when broker is connected."""
        # Mock broker to return valid price
        mock_broker_manager.get_current_price = AsyncMock(
            return_value={"bid": 1.0900}
        )

        lots = await trade_executor._calculate_position_size(
            signal=sample_signal,
            equity=100000.0,
        )

        # Verify broker was called
        mock_broker_manager.get_current_price.assert_called_once_with("EURUSD")
        assert lots > 0

    @pytest.mark.asyncio
    async def test_fallback_to_signal_stop_loss_when_broker_unavailable(
        self, trade_executor, sample_signal, mock_broker_manager
    ):
        """Test fallback to signal.stop_loss_price when broker unavailable."""
        # Mock broker to return None
        mock_broker_manager.get_current_price = AsyncMock(return_value=None)

        # Set valid stop_loss_price
        sample_signal.stop_loss_price = 1.0850

        lots = await trade_executor._calculate_position_size(
            signal=sample_signal,
            equity=100000.0,
        )

        assert lots > 0


# ============================================================================
# 2. Orphaned Trade Handling Tests (trade_executor.py)
# ============================================================================


class TestOrphanedTradeHandling:
    """Test orphaned trade handling functionality."""

    @pytest.fixture
    def mock_broker_manager(self):
        """Create mock broker manager."""
        broker = Mock()
        broker.is_connected.return_value = True
        broker_manager = Mock()
        broker_manager.broker = broker
        broker_manager.is_connected.return_value = True
        return broker_manager

    @pytest.fixture
    def failing_db_session_factory(self):
        """Create mock database session factory that fails."""
        def factory():
            session = Mock()
            session.add.side_effect = Exception("Database error")
            return session
        return factory

    @pytest.fixture
    def working_db_session_factory(self):
        """Create mock database session factory that works."""
        def factory():
            session = Mock()
            trade_mock = Mock()
            trade_mock.id = 123
            session.add.return_value = None
            session.commit.return_value = None
            # Simulate the trade object getting an ID after commit
            def add_side_effect(obj):
                obj.id = 123
            session.add.side_effect = add_side_effect
            return session
        return factory

    @pytest.fixture
    def config(self):
        """Create agent config."""
        return AgentConfig(
            mode="paper",
            confidence_threshold=0.65,
            max_position_size=0.05,
            symbol="EURUSD",
        )

    @pytest.mark.asyncio
    async def test_orphaned_trades_added_to_list_when_db_fails(
        self, mock_broker_manager, config, failing_db_session_factory
    ):
        """Test that orphaned trades are added to list when database fails."""
        executor = TradeExecutor(
            broker_manager=mock_broker_manager,
            config=config,
            db_session_factory=failing_db_session_factory,
        )

        # Mock broker order
        mock_order = Mock()
        mock_order.is_filled = True
        mock_order.average_fill_price = 1.0850
        mock_order.order_id = "12345"
        mock_broker_manager.broker.submit_order = AsyncMock(return_value=mock_order)
        mock_broker_manager.get_account_info = AsyncMock(
            return_value={"equity": 100000.0}
        )
        mock_broker_manager.get_current_price = AsyncMock(
            return_value={"bid": 1.0850}
        )

        # Create signal
        signal = MockTradingSignal(
            action=MockAction.BUY,
            symbol="EURUSD",
            confidence=0.75,
            direction_probability=0.75,
            position_size_pct=0.02,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
        )

        # Execute signal (should fail to store in DB)
        result = await executor.execute_signal(signal)

        # Verify orphaned trade was added
        assert executor.get_orphaned_trade_count() >= 1
        orphaned_trades = executor.get_orphaned_trades()
        assert len(orphaned_trades) >= 1
        assert orphaned_trades[0]["mt5_ticket"] == 12345

    def test_get_orphaned_trades_is_thread_safe(
        self, mock_broker_manager, config, failing_db_session_factory
    ):
        """Test that get_orphaned_trades is thread-safe."""
        executor = TradeExecutor(
            broker_manager=mock_broker_manager,
            config=config,
            db_session_factory=failing_db_session_factory,
        )

        # Add some orphaned trades manually
        with executor._orphaned_lock:
            executor._orphaned_trades.append({"mt5_ticket": 1})
            executor._orphaned_trades.append({"mt5_ticket": 2})

        # Access from multiple threads
        results = []

        def access_orphaned():
            trades = executor.get_orphaned_trades()
            results.append(len(trades))

        threads = [threading.Thread(target=access_orphaned) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should see same count
        assert all(count == 2 for count in results)

    @pytest.mark.asyncio
    async def test_retry_store_orphaned_trade_with_exponential_backoff(
        self, mock_broker_manager, config, working_db_session_factory
    ):
        """Test retry with exponential backoff."""
        executor = TradeExecutor(
            broker_manager=mock_broker_manager,
            config=config,
            db_session_factory=working_db_session_factory,
        )

        orphaned_trade = {
            "mt5_ticket": 12345,
            "symbol": "EURUSD",
            "side": "buy",
            "quantity": 0.1,
            "entry_price": 1.0850,
            "entry_time": datetime.now(),
            "confidence": 0.75,
            "stop_loss": 1.0800,
            "take_profit": 1.0900,
            "db_error": "Test error",
            "retry_count": 0,
        }

        start_time = asyncio.get_event_loop().time()
        trade_id = await executor._retry_store_orphaned_trade(orphaned_trade)
        elapsed = asyncio.get_event_loop().time() - start_time

        # Should succeed
        assert trade_id is not None
        # Should have waited (exponential backoff: 2^1 = 2 seconds)
        assert elapsed >= 2.0, "Should wait at least 2 seconds on first retry"

    @pytest.mark.asyncio
    async def test_reconcile_orphaned_trades_with_timeout(
        self, mock_broker_manager, config, working_db_session_factory
    ):
        """Test reconcile_orphaned_trades respects timeout."""
        executor = TradeExecutor(
            broker_manager=mock_broker_manager,
            config=config,
            db_session_factory=working_db_session_factory,
        )

        # Add multiple orphaned trades
        for i in range(5):
            with executor._orphaned_lock:
                executor._orphaned_trades.append({
                    "mt5_ticket": 1000 + i,
                    "symbol": "EURUSD",
                    "side": "buy",
                    "quantity": 0.1,
                    "entry_price": 1.0850,
                    "entry_time": datetime.now(),
                    "confidence": 0.75,
                    "stop_loss": 1.0800,
                    "take_profit": 1.0900,
                    "db_error": "Test error",
                    "retry_count": 0,
                })

        # Reconcile with short timeout
        start_time = asyncio.get_event_loop().time()
        reconciled = await executor.reconcile_orphaned_trades(max_duration_seconds=5.0)
        elapsed = asyncio.get_event_loop().time() - start_time

        # Should timeout before processing all trades
        assert elapsed < 10.0, "Should respect timeout"

    def test_get_orphaned_trade_count_is_accurate(
        self, mock_broker_manager, config, failing_db_session_factory
    ):
        """Test get_orphaned_trade_count returns accurate count."""
        executor = TradeExecutor(
            broker_manager=mock_broker_manager,
            config=config,
            db_session_factory=failing_db_session_factory,
        )

        assert executor.get_orphaned_trade_count() == 0

        # Add orphaned trades
        with executor._orphaned_lock:
            executor._orphaned_trades.append({"mt5_ticket": 1})
            executor._orphaned_trades.append({"mt5_ticket": 2})
            executor._orphaned_trades.append({"mt5_ticket": 3})

        assert executor.get_orphaned_trade_count() == 3

    def test_get_orphaned_trades_returns_copy(
        self, mock_broker_manager, config, failing_db_session_factory
    ):
        """Test that get_orphaned_trades returns a copy of the list."""
        executor = TradeExecutor(
            broker_manager=mock_broker_manager,
            config=config,
            db_session_factory=failing_db_session_factory,
        )

        # Add orphaned trade
        with executor._orphaned_lock:
            executor._orphaned_trades.append({"mt5_ticket": 1})

        # Get copy
        trades_copy = executor.get_orphaned_trades()

        # Modify copy
        trades_copy.append({"mt5_ticket": 2})

        # Original should be unchanged
        assert executor.get_orphaned_trade_count() == 1


# ============================================================================
# 3. Thread Safety Tests (safety_manager.py)
# ============================================================================


class TestThreadSafety:
    """Test thread safety in SafetyManager."""

    @pytest.fixture
    def mock_db_session_factory(self):
        """Create mock database session factory."""
        def factory():
            session = Mock()
            session.query.return_value.filter.return_value.first.return_value = None
            return session
        return factory

    @pytest.fixture
    def safety_config(self):
        """Create safety config."""
        return SafetyConfig(
            max_consecutive_losses=3,
            max_drawdown_percent=10.0,
            max_daily_loss_percent=5.0,
            max_daily_loss_amount=5000.0,
            max_daily_trades=20,
            max_trades_per_hour=5,
            max_disconnection_seconds=300,
            auto_reset_next_day=True,
            require_token_for_reset=False,
        )

    @pytest.fixture
    def safety_manager(self, safety_config, mock_db_session_factory):
        """Create SafetyManager instance."""
        return SafetyManager(
            config=safety_config,
            initial_equity=100000.0,
            db_session_factory=mock_db_session_factory,
        )

    def test_check_safety_is_thread_safe(self, safety_manager):
        """Test that check_safety is thread-safe with concurrent access."""
        results = []

        def check_safety_thread():
            status = safety_manager.check_safety(
                current_equity=95000.0,
                is_broker_connected=True,
            )
            results.append(status.is_safe_to_trade)

        # Run from multiple threads
        threads = [threading.Thread(target=check_safety_thread) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should complete without errors
        assert len(results) == 20

    def test_record_trade_result_is_thread_safe(self, safety_manager):
        """Test that record_trade_result is thread-safe with concurrent access."""
        results = []

        def record_trade():
            trade_result = MockTradeResult(
                pnl=-100.0,
                is_winner=False,
                timestamp=datetime.now(),
            )
            safety_manager.record_trade_result(trade_result)
            results.append(True)

        # Run from multiple threads
        threads = [threading.Thread(target=record_trade) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should complete
        assert len(results) == 20
        # Daily trades counter should be correct
        assert safety_manager._daily_trades == 20

    def test_reset_daily_counters_is_thread_safe(self, safety_manager):
        """Test that reset_daily_counters is thread-safe."""
        # Add some trades
        for _ in range(10):
            trade_result = MockTradeResult(
                pnl=-50.0,
                is_winner=False,
                timestamp=datetime.now(),
            )
            safety_manager.record_trade_result(trade_result)

        results = []

        def reset_counters():
            safety_manager.reset_daily_counters()
            results.append(True)

        # Run from multiple threads
        threads = [threading.Thread(target=reset_counters) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should complete
        assert len(results) == 5
        # Daily trades should be reset
        assert safety_manager._daily_trades == 0

    def test_get_status_returns_consistent_snapshot(self, safety_manager):
        """Test that get_status returns consistent snapshot."""
        # Add some trades
        for _ in range(5):
            trade_result = MockTradeResult(
                pnl=-100.0,
                is_winner=False,
                timestamp=datetime.now(),
            )
            safety_manager.record_trade_result(trade_result)

        # Get status from multiple threads
        results = []

        def get_status_thread():
            status = safety_manager.get_status()
            results.append(status["daily_metrics"]["trades"])

        threads = [threading.Thread(target=get_status_thread) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should see consistent count
        assert all(count == 5 for count in results)


# ============================================================================
# 4. Trade Result Recording Tests (trading_cycle.py)
# ============================================================================


class TestTradeResultRecording:
    """Test that trade results are recorded in safety manager."""

    @pytest.fixture
    def mock_model_service(self):
        """Create mock model service."""
        service = Mock()
        service.is_loaded = True
        service.predict_from_pipeline.return_value = {
            "direction": "long",
            "confidence": 0.75,
            "prob_up": 0.75,
            "prob_down": 0.25,
            "should_trade": True,
            "symbol": "EURUSD",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "1H": {"direction": "long", "confidence": 0.73},
                "4H": {"direction": "long", "confidence": 0.76},
                "D": {"direction": "long", "confidence": 0.77},
            },
            "agreement_count": 3,
            "agreement_score": 1.0,
            "market_regime": "trending_normal",
        }
        return service

    @pytest.fixture
    def mock_db_session_factory(self):
        """Create mock database session factory."""
        def factory():
            session = Mock()
            pred_mock = Mock()
            pred_mock.id = 456
            def add_side_effect(obj):
                obj.id = 456
            session.add.side_effect = add_side_effect
            session.commit.return_value = None
            return session
        return factory

    @pytest.fixture
    def mock_broker_manager(self):
        """Create mock broker manager."""
        broker = Mock()
        broker.is_connected = True
        broker_manager = Mock()
        broker_manager.broker = broker
        broker_manager.is_connected.return_value = True
        broker_manager.get_account_info = AsyncMock(
            return_value={"equity": 98000.0, "balance": 98500.0}
        )
        return broker_manager

    @pytest.fixture
    def mock_trade_executor(self):
        """Create mock trade executor."""
        executor = Mock()

        # Mock check_open_positions to return a position to close
        position = PositionStatus(
            trade_id=123,
            mt5_ticket=12345,
            current_price=1.0900,
            unrealized_pnl=-150.0,
            should_close=True,
            close_reason="stop_loss",
        )
        executor.check_open_positions = AsyncMock(return_value=[position])
        executor.close_position = AsyncMock(return_value=True)

        return executor

    @pytest.fixture
    def mock_safety_manager(self):
        """Create mock safety manager."""
        manager = Mock()

        # Mock check_safety to return safe status
        status = SafetyStatus(
            is_safe_to_trade=True,
            circuit_breaker_triggered=False,
            kill_switch_active=False,
            circuit_breaker_state="active",
            active_breakers=[],
            breaker_reasons=[],
            size_multiplier=1.0,
            min_confidence_override=None,
            kill_switch_reason=None,
            kill_switch_trigger_time=None,
            daily_trades=2,
            daily_loss_pct=2.0,
            daily_loss_amount=2000.0,
            current_equity=98000.0,
            peak_equity=100000.0,
            current_drawdown_pct=2.0,
        )
        manager.check_safety.return_value = status
        manager.record_trade_result = Mock()

        return manager

    @pytest.fixture
    def config(self):
        """Create agent config."""
        return AgentConfig(
            mode="paper",
            confidence_threshold=0.65,
            max_position_size=0.05,
            symbol="EURUSD",
        )

    @pytest.mark.asyncio
    async def test_safety_manager_record_trade_result_called_on_position_close(
        self,
        config,
        mock_model_service,
        mock_db_session_factory,
        mock_broker_manager,
        mock_trade_executor,
        mock_safety_manager,
    ):
        """Test that safety_manager.record_trade_result is called when position closes."""
        cycle = TradingCycle(
            config=config,
            model_service=mock_model_service,
            db_session_factory=mock_db_session_factory,
            broker_manager=mock_broker_manager,
            trade_executor=mock_trade_executor,
            safety_manager=mock_safety_manager,
        )

        result = await cycle.execute(cycle_number=1)

        # Verify position was closed
        mock_trade_executor.close_position.assert_called_once()

        # Verify trade result was recorded
        mock_safety_manager.record_trade_result.assert_called_once()

        # Verify the TradeResult had correct PnL
        call_args = mock_safety_manager.record_trade_result.call_args
        trade_result = call_args[0][0]
        assert trade_result.pnl == -150.0
        assert trade_result.is_winner is False

    @pytest.mark.asyncio
    async def test_broker_equity_passed_to_check_safety(
        self,
        config,
        mock_model_service,
        mock_db_session_factory,
        mock_broker_manager,
        mock_safety_manager,
    ):
        """Test that broker equity is passed to check_safety when available."""
        cycle = TradingCycle(
            config=config,
            model_service=mock_model_service,
            db_session_factory=mock_db_session_factory,
            broker_manager=mock_broker_manager,
            safety_manager=mock_safety_manager,
        )

        await cycle.execute(cycle_number=1)

        # Verify check_safety was called with actual broker equity
        mock_safety_manager.check_safety.assert_called()
        call_kwargs = mock_safety_manager.check_safety.call_args[1]
        assert call_kwargs["current_equity"] == 98000.0

    @pytest.mark.asyncio
    async def test_is_broker_connected_passed_correctly(
        self,
        config,
        mock_model_service,
        mock_db_session_factory,
        mock_broker_manager,
        mock_safety_manager,
    ):
        """Test that is_broker_connected is passed correctly to check_safety."""
        cycle = TradingCycle(
            config=config,
            model_service=mock_model_service,
            db_session_factory=mock_db_session_factory,
            broker_manager=mock_broker_manager,
            safety_manager=mock_safety_manager,
        )

        await cycle.execute(cycle_number=1)

        # Verify check_safety was called with correct connection status
        mock_safety_manager.check_safety.assert_called()
        call_kwargs = mock_safety_manager.check_safety.call_args[1]
        assert call_kwargs["is_broker_connected"] is True


# ============================================================================
# 5. Broker Manager Tests (broker_manager.py)
# ============================================================================


class TestBrokerManager:
    """Test BrokerManager functionality."""

    @pytest.fixture
    def config(self):
        """Create agent config."""
        return AgentConfig(
            mode="paper",
            mt5_login=12345,
            mt5_password="password",
            mt5_server="Demo-Server",
            symbol="EURUSD",
        )

    @pytest.fixture
    def broker_manager(self, config):
        """Create BrokerManager instance."""
        return BrokerManager(config=config)

    @pytest.mark.asyncio
    async def test_get_current_price_returns_proper_format(self, broker_manager):
        """Test that get_current_price returns proper format."""
        # Mock broker
        mock_broker = Mock()
        mock_tick = Mock()
        mock_tick.bid = 1.0850
        mock_tick.ask = 1.0852
        mock_tick.last = 1.0851
        mock_tick.time = datetime.now()
        mock_broker.get_tick = AsyncMock(return_value=mock_tick)
        mock_broker.is_connected = True

        broker_manager.broker = mock_broker
        broker_manager._connected = True

        result = await broker_manager.get_current_price("EURUSD")

        assert result is not None
        assert "symbol" in result
        assert "bid" in result
        assert "ask" in result
        assert "last" in result
        assert result["symbol"] == "EURUSD"
        assert result["bid"] == 1.0850
        assert result["ask"] == 1.0852

    @pytest.mark.asyncio
    async def test_get_current_price_returns_none_when_not_connected(
        self, broker_manager
    ):
        """Test that get_current_price returns None when not connected."""
        # No broker set up
        broker_manager.broker = None
        broker_manager._connected = False

        result = await broker_manager.get_current_price("EURUSD")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_current_price_error_handling(self, broker_manager):
        """Test error handling in get_current_price."""
        # Create a mock BrokerError
        class MockBrokerError(Exception):
            pass

        # Mock broker that raises error
        mock_broker = Mock()
        mock_broker.get_tick = AsyncMock(side_effect=MockBrokerError("Test error"))
        mock_broker.is_connected = True

        broker_manager.broker = mock_broker
        broker_manager._connected = True

        result = await broker_manager.get_current_price("EURUSD")

        # Should return None on error
        assert result is None
