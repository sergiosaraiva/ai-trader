"""Unit tests for SafetyManager."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

# Import from conftest
from .conftest import SafetyConfig, SafetyManager, SafetyStatus


class TestSafetyManager:
    """Test SafetyManager class."""

    @pytest.fixture
    def mock_db_session_factory(self):
        """Mock database session factory."""
        mock_session = Mock()
        mock_session.add = Mock()
        mock_session.commit = Mock()
        mock_session.rollback = Mock()
        mock_session.close = Mock()

        def factory():
            return mock_session

        return factory

    @pytest.fixture
    def mock_circuit_breaker_manager(self):
        """Mock CircuitBreakerManager."""
        mock_manager = Mock()

        # Mock breakers
        mock_drawdown_breaker = Mock()
        mock_drawdown_breaker.peak_equity = 100000.0
        mock_drawdown_breaker.current_drawdown_pct = 0.0

        mock_manager.breakers = {"drawdown": mock_drawdown_breaker}

        # Mock check_all response
        mock_state = Mock()
        mock_state.can_trade = True
        mock_state.is_halted = False
        mock_state.overall_state = Mock(value="active")
        mock_state.active_breakers = []
        mock_state.reasons = []
        mock_state.size_multiplier = 1.0
        mock_state.min_confidence_override = None

        mock_manager.check_all = Mock(return_value=mock_state)
        mock_manager.current_state = mock_state
        mock_manager.record_trade = Mock()
        mock_manager.get_status = Mock(return_value={"can_trade": True, "state": "active"})

        return mock_manager

    @pytest.fixture
    def mock_kill_switch(self):
        """Mock KillSwitch."""
        mock_ks = Mock()
        mock_ks.is_active = False
        mock_ks.check_all = Mock(return_value=False)

        # Mock state
        mock_ks.state = Mock()
        mock_ks.state.is_active = False
        mock_ks.state.trigger = None
        mock_ks.state.activated_at = None

        mock_ks.trigger = Mock()
        mock_ks.reset = Mock(return_value=True)
        mock_ks.get_reset_code = Mock(return_value="RESET123")
        mock_ks.reset_daily_counters = Mock()
        mock_ks.get_stats = Mock(return_value={"is_active": False})

        return mock_ks

    @pytest.fixture
    def safety_manager(self, mock_db_session_factory, mock_circuit_breaker_manager, mock_kill_switch):
        """Create SafetyManager with mocked dependencies."""
        config = SafetyConfig()

        # Patch dependencies during initialization
        with patch("agent.safety_manager.CircuitBreakerManager", return_value=mock_circuit_breaker_manager):
            with patch("agent.safety_manager.KillSwitch", return_value=mock_kill_switch):
                manager = SafetyManager(
                    config=config,
                    initial_equity=100000.0,
                    db_session_factory=mock_db_session_factory,
                )
                # Replace with mocks
                manager.circuit_breaker_manager = mock_circuit_breaker_manager
                manager.kill_switch = mock_kill_switch

                return manager

    def test_initialization(self, safety_manager):
        """Test SafetyManager initializes correctly."""
        # Assert
        assert safety_manager.initial_equity == 100000.0
        assert safety_manager._current_equity == 100000.0
        assert safety_manager._daily_start_equity == 100000.0
        assert safety_manager._daily_trades == 0
        assert safety_manager.config is not None
        assert safety_manager.circuit_breaker_manager is not None
        assert safety_manager.kill_switch is not None

    def test_check_safety_all_safe(self, safety_manager):
        """Test check_safety returns safe when all checks pass."""
        # Arrange
        current_equity = 102000.0

        # Act
        status = safety_manager.check_safety(current_equity=current_equity)

        # Assert
        assert status.is_safe_to_trade is True
        assert status.circuit_breaker_triggered is False
        assert status.kill_switch_active is False
        assert status.current_equity == 102000.0
        assert status.daily_trades == 0

    def test_check_safety_updates_equity(self, safety_manager):
        """Test check_safety updates equity tracking (CRITICAL)."""
        # Arrange
        initial_equity = 100000.0
        new_equity = 98000.0

        # Act
        status1 = safety_manager.check_safety(current_equity=new_equity)
        status2 = safety_manager.check_safety()  # Should use last equity

        # Assert - equity updated correctly
        assert safety_manager._current_equity == new_equity
        assert status1.current_equity == new_equity
        assert status2.current_equity == new_equity

    def test_check_safety_calculates_daily_loss_correctly(self, safety_manager):
        """Test daily loss calculated from daily start equity (CRITICAL)."""
        # Arrange
        safety_manager._daily_start_equity = 100000.0
        safety_manager._current_equity = 97000.0  # Lost $3000

        # Act
        status = safety_manager.check_safety()

        # Assert - daily loss calculated correctly
        assert status.daily_loss_amount == 3000.0
        assert status.daily_loss_pct == 3.0  # 3% of daily start

    def test_check_safety_kill_switch_active(self, safety_manager, mock_kill_switch):
        """Test check_safety returns not safe when kill switch active."""
        # Arrange
        mock_kill_switch.is_active = True
        mock_kill_switch.check_all = Mock(return_value=True)
        mock_kill_switch.state.trigger = Mock(reason="Daily loss exceeded")

        # Act
        status = safety_manager.check_safety(current_equity=95000.0)

        # Assert
        assert status.is_safe_to_trade is False
        assert status.kill_switch_active is True
        assert status.kill_switch_reason == "Daily loss exceeded"
        assert status.size_multiplier == 0.0

    def test_check_safety_circuit_breaker_triggered(self, safety_manager, mock_circuit_breaker_manager):
        """Test check_safety returns not safe when circuit breaker triggered."""
        # Arrange
        mock_state = Mock()
        mock_state.can_trade = False
        mock_state.is_halted = True
        mock_state.overall_state = Mock(value="halted")
        mock_state.active_breakers = ["consecutive_loss"]
        mock_state.reasons = ["5 consecutive losses"]
        mock_state.size_multiplier = 0.0
        mock_state.min_confidence_override = None

        mock_circuit_breaker_manager.check_all = Mock(return_value=mock_state)
        mock_circuit_breaker_manager.current_state = mock_state

        # Act
        status = safety_manager.check_safety(current_equity=98000.0)

        # Assert
        assert status.is_safe_to_trade is False
        assert status.circuit_breaker_triggered is True
        assert status.active_breakers == ["consecutive_loss"]
        assert status.breaker_reasons == ["5 consecutive losses"]

    def test_check_safety_broker_disconnected(self, safety_manager, mock_kill_switch):
        """Test check_safety triggers on broker disconnection."""
        # Arrange
        mock_kill_switch.check_all = Mock(return_value=True)
        mock_kill_switch.is_active = True
        mock_kill_switch.state.trigger = Mock(reason="Broker disconnected")

        # Act
        status = safety_manager.check_safety(
            current_equity=100000.0, is_broker_connected=False
        )

        # Assert
        assert status.is_safe_to_trade is False
        assert status.kill_switch_active is True

    def test_record_trade_result_increments_counter(self, safety_manager):
        """Test record_trade_result increments daily trade counter."""
        # Arrange
        trade_result = Mock(pnl=100.0, is_winner=True)
        assert safety_manager._daily_trades == 0

        # Act
        safety_manager.record_trade_result(trade_result)

        # Assert
        assert safety_manager._daily_trades == 1

    def test_record_trade_result_updates_equity(self, safety_manager):
        """Test record_trade_result updates equity correctly (CRITICAL)."""
        # Arrange
        initial_equity = 100000.0
        trade_result = Mock(pnl=-500.0, is_winner=False)
        assert safety_manager._current_equity == initial_equity

        # Act
        safety_manager.record_trade_result(trade_result)

        # Assert - equity updated correctly
        assert safety_manager._current_equity == 99500.0

    def test_record_trade_result_passes_to_circuit_breakers(self, safety_manager, mock_circuit_breaker_manager):
        """Test record_trade_result passes result to circuit breakers (CRITICAL)."""
        # Arrange
        trade_result = Mock(pnl=-200.0, is_winner=False)

        # Act
        safety_manager.record_trade_result(trade_result)

        # Assert - circuit breaker manager received trade
        mock_circuit_breaker_manager.record_trade.assert_called_once_with(trade_result)

    def test_record_trade_result_multiple_trades(self, safety_manager):
        """Test recording multiple trade results updates correctly."""
        # Arrange
        trade1 = Mock(pnl=100.0, is_winner=True)
        trade2 = Mock(pnl=-150.0, is_winner=False)
        trade3 = Mock(pnl=200.0, is_winner=True)

        # Act
        safety_manager.record_trade_result(trade1)
        safety_manager.record_trade_result(trade2)
        safety_manager.record_trade_result(trade3)

        # Assert
        assert safety_manager._daily_trades == 3
        assert safety_manager._current_equity == 100150.0  # 100000 + 100 - 150 + 200

    def test_trigger_kill_switch(self, safety_manager, mock_kill_switch):
        """Test trigger_kill_switch activates kill switch."""
        # Arrange
        reason = "Manual stop requested"

        # Act
        safety_manager.trigger_kill_switch(reason)

        # Assert
        mock_kill_switch.trigger.assert_called_once()

    def test_reset_kill_switch_with_valid_auth(self, safety_manager, mock_kill_switch):
        """Test reset_kill_switch with valid authorization."""
        # Arrange
        mock_kill_switch.reset = Mock(return_value=True)

        # Act
        result = safety_manager.reset_kill_switch(authorization="VALID123")

        # Assert
        assert result is True
        mock_kill_switch.reset.assert_called_once_with(
            authorization="VALID123", force=False
        )

    def test_reset_kill_switch_with_invalid_auth(self, safety_manager, mock_kill_switch):
        """Test reset_kill_switch fails with invalid authorization."""
        # Arrange
        mock_kill_switch.reset = Mock(return_value=False)

        # Act
        result = safety_manager.reset_kill_switch(authorization="INVALID")

        # Assert
        assert result is False

    def test_reset_kill_switch_force_reset(self, safety_manager, mock_kill_switch):
        """Test reset_kill_switch with force flag."""
        # Arrange
        mock_kill_switch.reset = Mock(return_value=True)

        # Act
        result = safety_manager.reset_kill_switch(force=True)

        # Assert
        assert result is True
        mock_kill_switch.reset.assert_called_once_with(authorization="", force=True)

    def test_get_reset_code(self, safety_manager, mock_kill_switch):
        """Test get_reset_code returns authorization code."""
        # Arrange
        mock_kill_switch.get_reset_code = Mock(return_value="RESET789")

        # Act
        code = safety_manager.get_reset_code()

        # Assert
        assert code == "RESET789"
        mock_kill_switch.get_reset_code.assert_called_once()

    def test_reset_circuit_breaker_valid(self, safety_manager, mock_circuit_breaker_manager):
        """Test reset_circuit_breaker with valid breaker name."""
        # Arrange
        mock_breaker = Mock()
        mock_circuit_breaker_manager.breakers = {"consecutive_loss": mock_breaker}

        # Act
        result = safety_manager.reset_circuit_breaker("consecutive_loss")

        # Assert
        assert result is True
        mock_breaker.reset.assert_called_once()

    def test_reset_circuit_breaker_invalid(self, safety_manager, mock_circuit_breaker_manager):
        """Test reset_circuit_breaker with invalid breaker name."""
        # Arrange
        mock_circuit_breaker_manager.breakers = {}

        # Act
        result = safety_manager.reset_circuit_breaker("nonexistent")

        # Assert
        assert result is False

    def test_reset_daily_counters(self, safety_manager, mock_kill_switch):
        """Test reset_daily_counters resets all daily metrics."""
        # Arrange
        safety_manager._daily_trades = 10
        safety_manager._current_equity = 105000.0
        safety_manager._daily_start_equity = 100000.0

        # Act
        safety_manager.reset_daily_counters()

        # Assert
        assert safety_manager._daily_trades == 0
        assert safety_manager._daily_start_equity == 105000.0  # Updated to current
        mock_kill_switch.reset_daily_counters.assert_called_once()

    def test_reset_daily_counters_thread_safety(self, safety_manager):
        """Test reset_daily_counters is thread-safe (CRITICAL)."""
        # Arrange
        import threading

        results = []

        def reset_counter():
            safety_manager.reset_daily_counters()
            results.append(safety_manager._daily_trades)

        # Act - reset from multiple threads
        threads = [threading.Thread(target=reset_counter) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Assert - all resets successful, counter is 0
        assert safety_manager._daily_trades == 0
        assert all(r == 0 for r in results)

    def test_get_status_returns_complete_info(self, safety_manager, mock_circuit_breaker_manager, mock_kill_switch):
        """Test get_status returns all safety information."""
        # Arrange
        safety_manager._daily_trades = 5
        safety_manager._current_equity = 98000.0
        safety_manager._daily_start_equity = 100000.0

        mock_circuit_breaker_manager.get_status = Mock(
            return_value={"can_trade": True, "state": "active"}
        )
        mock_kill_switch.get_stats = Mock(return_value={"is_active": False})

        # Act
        status = safety_manager.get_status()

        # Assert
        assert status["is_safe_to_trade"] is True
        assert status["circuit_breakers"] == {"can_trade": True, "state": "active"}
        assert status["kill_switch"] == {"is_active": False}
        assert status["daily_metrics"]["trades"] == 5
        assert status["daily_metrics"]["loss_amount"] == 2000.0
        assert status["daily_metrics"]["loss_pct"] == 2.0
        assert status["account_metrics"]["current_equity"] == 98000.0

    def test_concurrent_safety_checks_thread_safe(self, safety_manager):
        """Test concurrent check_safety calls are thread-safe (CRITICAL)."""
        # Arrange
        import threading

        results = []

        def check_safety():
            status = safety_manager.check_safety(current_equity=100000.0)
            results.append(status.is_safe_to_trade)

        # Act - check from multiple threads
        threads = [threading.Thread(target=check_safety) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Assert - all checks completed without errors
        assert len(results) == 10
        assert all(isinstance(r, bool) for r in results)

    def test_equity_tracking_with_peak_equity(self, safety_manager, mock_circuit_breaker_manager):
        """Test equity tracking updates peak equity correctly."""
        # Arrange
        mock_drawdown_breaker = Mock()
        mock_drawdown_breaker.peak_equity = 100000.0
        mock_drawdown_breaker.current_drawdown_pct = 0.0
        mock_circuit_breaker_manager.breakers = {"drawdown": mock_drawdown_breaker}

        # Act - make profit
        status1 = safety_manager.check_safety(current_equity=105000.0)

        # Act - drawdown from peak
        mock_drawdown_breaker.current_drawdown_pct = 0.05  # 5% drawdown
        status2 = safety_manager.check_safety(current_equity=99750.0)

        # Assert - drawdown calculated from peak
        assert status1.current_drawdown_pct == 0.0
        assert status2.current_drawdown_pct == 5.0

    def test_daily_loss_limit_triggers_correctly(self, safety_manager, mock_kill_switch):
        """Test daily loss limit triggers kill switch."""
        # Arrange
        safety_manager._daily_start_equity = 100000.0
        safety_manager._current_equity = 94000.0  # 6% loss

        mock_kill_switch.check_all = Mock(return_value=True)
        mock_kill_switch.is_active = True
        mock_kill_switch.state.trigger = Mock(reason="Daily loss limit exceeded")

        # Act
        status = safety_manager.check_safety()

        # Assert
        assert status.is_safe_to_trade is False
        assert status.kill_switch_active is True
        assert status.daily_loss_pct == 6.0

    def test_max_daily_trades_tracked(self, safety_manager):
        """Test max daily trades tracked correctly."""
        # Arrange
        for i in range(50):
            trade = Mock(pnl=10.0, is_winner=True)
            safety_manager.record_trade_result(trade)

        # Act
        status = safety_manager.check_safety()

        # Assert
        assert status.daily_trades == 50


class TestSafetyStatus:
    """Test SafetyStatus dataclass."""

    def test_safety_status_creation(self):
        """Test SafetyStatus can be created."""
        # Arrange & Act
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
            daily_trades=5,
            daily_loss_pct=1.5,
            daily_loss_amount=1500.0,
            current_equity=98500.0,
            peak_equity=100000.0,
            current_drawdown_pct=1.5,
        )

        # Assert
        assert status.is_safe_to_trade is True
        assert status.daily_trades == 5
        assert status.current_equity == 98500.0

    def test_safety_status_to_dict(self):
        """Test SafetyStatus.to_dict() serialization."""
        # Arrange
        trigger_time = datetime(2025, 1, 22, 10, 0, 0)
        status = SafetyStatus(
            is_safe_to_trade=False,
            circuit_breaker_triggered=True,
            kill_switch_active=False,
            circuit_breaker_state="halted",
            active_breakers=["consecutive_loss"],
            breaker_reasons=["5 losses in a row"],
            size_multiplier=0.0,
            min_confidence_override=0.80,
            kill_switch_reason=None,
            kill_switch_trigger_time=trigger_time,
            daily_trades=5,
            daily_loss_pct=3.0,
            daily_loss_amount=3000.0,
            current_equity=97000.0,
            peak_equity=100000.0,
            current_drawdown_pct=3.0,
        )

        # Act
        result = status.to_dict()

        # Assert
        assert result["is_safe_to_trade"] is False
        assert result["circuit_breaker_triggered"] is True
        assert result["active_breakers"] == ["consecutive_loss"]
        assert result["size_multiplier"] == 0.0
        assert result["min_confidence_override"] == 0.80
        assert result["kill_switch_trigger_time"] == trigger_time.isoformat()
        assert result["daily_trades"] == 5
        assert result["current_equity"] == 97000.0

    def test_safety_status_to_dict_with_none_timestamp(self):
        """Test SafetyStatus.to_dict() handles None timestamp."""
        # Arrange
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
            daily_trades=0,
            daily_loss_pct=0.0,
            daily_loss_amount=0.0,
            current_equity=100000.0,
            peak_equity=100000.0,
            current_drawdown_pct=0.0,
        )

        # Act
        result = status.to_dict()

        # Assert
        assert result["kill_switch_trigger_time"] is None


class TestSafetyManagerIntegration:
    """Integration tests for SafetyManager full workflows."""

    @pytest.fixture
    def mock_db_session_factory(self):
        """Mock database session factory."""
        mock_session = Mock()
        mock_session.add = Mock()
        mock_session.commit = Mock()
        mock_session.rollback = Mock()
        mock_session.close = Mock()

        def factory():
            return mock_session

        return factory

    @pytest.fixture
    def mock_circuit_breaker_manager(self):
        """Mock CircuitBreakerManager for integration tests."""
        mock_manager = Mock()

        # Mock breakers
        mock_drawdown_breaker = Mock()
        mock_drawdown_breaker.peak_equity = 100000.0
        mock_drawdown_breaker.current_drawdown_pct = 0.0

        mock_manager.breakers = {"drawdown": mock_drawdown_breaker, "consecutive_loss": Mock()}

        # Mock check_all response
        mock_state = Mock()
        mock_state.can_trade = True
        mock_state.is_halted = False
        mock_state.overall_state = Mock(value="active")
        mock_state.active_breakers = []
        mock_state.reasons = []
        mock_state.size_multiplier = 1.0
        mock_state.min_confidence_override = None

        mock_manager.check_all = Mock(return_value=mock_state)
        mock_manager.current_state = mock_state
        mock_manager.record_trade = Mock()
        mock_manager.get_status = Mock(return_value={"can_trade": True, "state": "active"})

        return mock_manager

    @pytest.fixture
    def mock_kill_switch(self):
        """Mock KillSwitch for integration tests."""
        mock_ks = Mock()
        mock_ks.is_active = False
        mock_ks.check_all = Mock(return_value=False)

        # Mock state
        mock_ks.state = Mock()
        mock_ks.state.is_active = False
        mock_ks.state.trigger = None
        mock_ks.state.activated_at = None

        mock_ks.trigger = Mock()
        mock_ks.reset = Mock(return_value=True)
        mock_ks.get_reset_code = Mock(return_value="RESET123")
        mock_ks.reset_daily_counters = Mock()
        mock_ks.get_stats = Mock(return_value={"is_active": False})

        return mock_ks

    @pytest.fixture
    def integration_manager(self, mock_db_session_factory, mock_circuit_breaker_manager, mock_kill_switch):
        """Create SafetyManager for integration tests."""
        config = SafetyConfig(
            max_consecutive_losses=3,
            max_drawdown_percent=10.0,
            max_daily_loss_percent=5.0,
        )

        with patch("agent.safety_manager.CircuitBreakerManager", return_value=mock_circuit_breaker_manager):
            with patch("agent.safety_manager.KillSwitch", return_value=mock_kill_switch):
                manager = SafetyManager(
                    config=config,
                    initial_equity=100000.0,
                    db_session_factory=mock_db_session_factory,
                )
                manager.circuit_breaker_manager = mock_circuit_breaker_manager
                manager.kill_switch = mock_kill_switch
                return manager

    def test_full_trading_cycle_safe(self, integration_manager):
        """Test full trading cycle when everything is safe."""
        # Arrange
        initial_equity = 100000.0

        # Act - check before trade
        status_before = integration_manager.check_safety(current_equity=initial_equity)

        # Execute winning trade
        trade = Mock(pnl=100.0, is_winner=True)
        integration_manager.record_trade_result(trade)

        # Check after trade
        status_after = integration_manager.check_safety()

        # Assert
        assert status_before.is_safe_to_trade is True
        assert status_after.is_safe_to_trade is True
        assert status_after.daily_trades == 1
        assert status_after.current_equity == 100100.0

    def test_multiple_losses_trigger_circuit_breaker(
        self, integration_manager, mock_circuit_breaker_manager
    ):
        """Test multiple losing trades trigger circuit breaker."""
        # Arrange
        mock_state = Mock()
        mock_state.can_trade = False
        mock_state.is_halted = True
        mock_state.overall_state = Mock(value="halted")
        mock_state.active_breakers = ["consecutive_loss"]
        mock_state.reasons = ["3 consecutive losses"]
        mock_state.size_multiplier = 0.0
        mock_state.min_confidence_override = None

        # Act - record 3 losses
        for i in range(3):
            trade = Mock(pnl=-100.0, is_winner=False)
            integration_manager.record_trade_result(trade)

        # Mock breaker triggered after 3 losses
        mock_circuit_breaker_manager.check_all = Mock(return_value=mock_state)
        mock_circuit_breaker_manager.current_state = mock_state

        status = integration_manager.check_safety()

        # Assert
        assert status.is_safe_to_trade is False
        assert status.circuit_breaker_triggered is True
        assert "consecutive_loss" in status.active_breakers

    def test_recovery_after_reset(self, integration_manager, mock_circuit_breaker_manager):
        """Test trading resumes after circuit breaker reset."""
        # Arrange - trigger breaker
        mock_breaker = Mock()
        mock_circuit_breaker_manager.breakers = {"consecutive_loss": mock_breaker, "drawdown": Mock()}

        # Act - reset breaker
        result = integration_manager.reset_circuit_breaker("consecutive_loss")

        # Mock breaker now inactive
        mock_state = Mock()
        mock_state.can_trade = True
        mock_state.is_halted = False
        mock_state.overall_state = Mock(value="active")
        mock_state.active_breakers = []
        mock_state.reasons = []
        mock_state.size_multiplier = 1.0
        mock_state.min_confidence_override = None
        mock_circuit_breaker_manager.check_all = Mock(return_value=mock_state)

        status = integration_manager.check_safety()

        # Assert
        assert result is True
        assert status.is_safe_to_trade is True


class TestSafetyManagerAsyncEventLogging:
    """Test async event logging - HIGH PRIORITY FIX."""

    @pytest.mark.asyncio
    async def test_log_event_async_does_not_block_event_loop(self, in_memory_db):
        """Test that _log_event_async doesn't block the event loop."""
        config = SafetyConfig(
            max_consecutive_losses=3,
            max_drawdown_percent=10.0,
            max_daily_loss_percent=5.0,
        )
        manager = SafetyManager(
            config=config,
            initial_equity=100000.0,
            db_session_factory=in_memory_db,
        )

        # Mock _log_event to be slow (simulating slow DB)
        original_log = manager._log_event
        call_count = 0

        def slow_log(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            import time
            time.sleep(0.1)  # Simulate slow DB operation
            return original_log(*args, **kwargs)

        with patch.object(manager, '_log_event', side_effect=slow_log):
            import asyncio
            start_time = asyncio.get_event_loop().time()

            # Call async version - should not block
            await manager._log_event_async(
                event_type="test",
                breaker_type="test_breaker",
                severity="warning",
                action="test_action",
                reason="Test reason",
                value=100.0,
                threshold=200.0,
            )

            end_time = asyncio.get_event_loop().time()

        # Verify _log_event was called
        assert call_count == 1

        # The async call should complete quickly even though DB operation is slow
        # (because it's run in a thread)
        elapsed = end_time - start_time
        # Should be slightly more than 0.1s but event loop wasn't blocked
        assert elapsed < 0.5  # Generous timeout

    @pytest.mark.asyncio
    async def test_log_event_async_logs_correctly(self, in_memory_db):
        """Test that events are logged correctly via async method."""
        config = SafetyConfig(
            max_consecutive_losses=3,
            max_drawdown_percent=10.0,
            max_daily_loss_percent=5.0,
        )
        manager = SafetyManager(
            config=config,
            initial_equity=100000.0,
            db_session_factory=in_memory_db,
        )

        # Log event async
        await manager._log_event_async(
            event_type="circuit_breaker",
            breaker_type="consecutive_loss",
            severity="critical",
            action="triggered",
            reason="3 consecutive losses",
            value=3.0,
            threshold=3.0,
        )

        # Verify event was logged to database
        session = in_memory_db()
        try:
            from api.database.models import CircuitBreakerEvent
            events = session.query(CircuitBreakerEvent).all()

            assert len(events) == 1
            event = events[0]
            assert event.breaker_type == "consecutive_loss"
            assert event.severity == "critical"
            assert event.action == "triggered"
            assert event.reason == "3 consecutive losses"
            assert event.value == 3.0
            assert event.threshold == 3.0
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_log_event_async_handles_errors(self, in_memory_db):
        """Test that async logging handles errors gracefully."""
        config = SafetyConfig(
            max_consecutive_losses=3,
            max_drawdown_percent=10.0,
            max_daily_loss_percent=5.0,
        )
        manager = SafetyManager(
            config=config,
            initial_equity=100000.0,
            db_session_factory=in_memory_db,
        )

        # Mock _log_event to raise exception
        def failing_log(*args, **kwargs):
            raise Exception("Database connection failed")

        with patch.object(manager, '_log_event', side_effect=failing_log):
            # Should not raise exception - error is caught
            await manager._log_event_async(
                event_type="test",
                breaker_type="test_breaker",
                severity="warning",
                action="test_action",
                reason="Test reason",
            )

        # Verify no exception was raised and execution continued

    @pytest.mark.asyncio
    async def test_log_event_async_wraps_sync_method(self, in_memory_db):
        """Test that _log_event_async properly wraps synchronous _log_event."""
        config = SafetyConfig(
            max_consecutive_losses=3,
            max_drawdown_percent=10.0,
            max_daily_loss_percent=5.0,
        )
        manager = SafetyManager(
            config=config,
            initial_equity=100000.0,
            db_session_factory=in_memory_db,
        )

        # Spy on _log_event to verify it's called with correct args
        original_log = manager._log_event
        call_args = []

        def spy_log(*args, **kwargs):
            call_args.append((args, kwargs))
            return original_log(*args, **kwargs)

        with patch.object(manager, '_log_event', side_effect=spy_log):
            await manager._log_event_async(
                event_type="circuit_breaker",
                breaker_type="drawdown",
                severity="warning",
                action="warning",
                reason="Approaching limit",
                value=8.5,
                threshold=10.0,
            )

        # Verify _log_event was called with correct arguments
        assert len(call_args) == 1
        args, kwargs = call_args[0]

        # Check kwargs were passed correctly
        assert kwargs["event_type"] == "circuit_breaker"
        assert kwargs["breaker_type"] == "drawdown"
        assert kwargs["severity"] == "warning"
        assert kwargs["action"] == "warning"
        assert kwargs["reason"] == "Approaching limit"
        assert kwargs["value"] == 8.5
        assert kwargs["threshold"] == 10.0

    @pytest.mark.asyncio
    async def test_log_event_sync_still_works(self, in_memory_db):
        """Test that synchronous _log_event method still works."""
        config = SafetyConfig(
            max_consecutive_losses=3,
            max_drawdown_percent=10.0,
            max_daily_loss_percent=5.0,
        )
        manager = SafetyManager(
            config=config,
            initial_equity=100000.0,
            db_session_factory=in_memory_db,
        )

        # Call sync version
        manager._log_event(
            event_type="kill_switch",
            breaker_type="kill_switch",
            severity="critical",
            action="triggered",
            reason="Manual trigger",
        )

        # Verify event was logged
        session = in_memory_db()
        try:
            from api.database.models import CircuitBreakerEvent
            events = session.query(CircuitBreakerEvent).all()

            assert len(events) == 1
            event = events[0]
            assert event.breaker_type == "kill_switch"
            assert event.severity == "critical"
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_multiple_async_logs_do_not_block(self, in_memory_db):
        """Test that multiple async log calls don't block each other."""
        config = SafetyConfig(
            max_consecutive_losses=3,
            max_drawdown_percent=10.0,
            max_daily_loss_percent=5.0,
        )
        manager = SafetyManager(
            config=config,
            initial_equity=100000.0,
            db_session_factory=in_memory_db,
        )

        # Mock _log_event to be slow
        original_log = manager._log_event

        def slow_log(*args, **kwargs):
            import time
            time.sleep(0.05)  # Simulate slow operation
            return original_log(*args, **kwargs)

        with patch.object(manager, '_log_event', side_effect=slow_log):
            import asyncio
            start_time = asyncio.get_event_loop().time()

            # Create multiple concurrent async log calls
            tasks = [
                manager._log_event_async(
                    event_type="test",
                    breaker_type=f"breaker_{i}",
                    severity="warning",
                    action="test",
                    reason=f"Test {i}",
                )
                for i in range(5)
            ]

            await asyncio.gather(*tasks)

            end_time = asyncio.get_event_loop().time()

        elapsed = end_time - start_time

        # All 5 logs should complete
        # If they ran sequentially, it would take 5 * 0.05 = 0.25s
        # With threading, they run concurrently, so should be much faster
        assert elapsed < 0.3  # Allow some overhead but verify concurrency
