"""Unit tests for AgentRunner."""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import from conftest
from .conftest import AgentConfig, AgentRunner, AgentStatus, Base, _mock_model_service


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal


@pytest.fixture
def config():
    """Create a test configuration."""
    return AgentConfig(
        mode="simulation",
        confidence_threshold=0.70,
        cycle_interval_seconds=1,  # Fast cycles for testing
        initial_capital=100000.0,
    )


@pytest.fixture
def agent_runner(config, db_session):
    """Create an AgentRunner instance with test dependencies."""
    # Reset _mock_model_service for each test
    _mock_model_service.is_initialized = False
    _mock_model_service.is_loaded = False
    _mock_model_service.initialize = Mock(return_value=True)

    # Create runner (model_service is already mocked in conftest)
    runner = AgentRunner(config)
    runner._command_handler = runner._command_handler.__class__(db_session)
    runner._state_manager = runner._state_manager.__class__(db_session)
    yield runner


class TestAgentRunnerInitialization:
    """Test AgentRunner initialization."""

    def test_init_sets_initial_status(self, config, db_session):
        """Test __init__ sets initial status to stopped."""
        # Arrange & Act
        with patch("src.agent.runner.get_session", db_session):
            runner = AgentRunner(config)

        # Assert
        assert runner.status == AgentStatus.STOPPED
        assert runner._running is False
        assert runner._cycle_count == 0

    def test_init_stores_config(self, config, db_session):
        """Test __init__ stores configuration."""
        # Arrange & Act
        with patch("src.agent.runner.get_session", db_session):
            runner = AgentRunner(config)

        # Assert
        assert runner.config == config


class TestAgentRunnerStart:
    """Test AgentRunner start method."""

    @pytest.mark.asyncio
    async def test_start_initializes_state_manager(self, agent_runner, mock_model_service):
        """Test start initializes state manager."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True

        # Act
        result = await agent_runner.start()

        # Assert
        assert result is True
        assert agent_runner._state_manager._state_id is not None

    @pytest.mark.asyncio
    async def test_start_initializes_model_service(self, agent_runner, mock_model_service):
        """Test start initializes model service if not initialized."""
        # Arrange
        mock_model_service.is_initialized = False

        # Act
        result = await agent_runner.start()

        # Assert
        assert result is True
        mock_model_service.initialize.assert_called_once_with(warm_up=True)

    @pytest.mark.asyncio
    async def test_start_skips_model_init_if_already_initialized(self, agent_runner, mock_model_service):
        """Test start skips model initialization if already done."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True

        # Act
        result = await agent_runner.start()

        # Assert
        assert result is True
        mock_model_service.initialize.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_updates_status_to_running(self, agent_runner, mock_model_service):
        """Test start updates status to running."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True

        # Act
        result = await agent_runner.start()

        # Assert
        assert result is True
        assert agent_runner.status == AgentStatus.RUNNING
        assert agent_runner._running is True

    @pytest.mark.asyncio
    async def test_start_starts_command_polling(self, agent_runner, mock_model_service):
        """Test start starts command polling."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True

        # Act
        result = await agent_runner.start()

        # Assert
        assert result is True
        assert agent_runner._command_handler._running is True

    @pytest.mark.asyncio
    async def test_start_creates_main_task(self, agent_runner, mock_model_service):
        """Test start creates main loop task."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True

        # Act
        result = await agent_runner.start()

        # Assert
        assert result is True
        assert agent_runner._main_task is not None
        assert isinstance(agent_runner._main_task, asyncio.Task)

        # Cleanup
        agent_runner._running = False
        await asyncio.sleep(0.01)

    @pytest.mark.asyncio
    async def test_start_returns_false_if_already_running(self, agent_runner, mock_model_service):
        """Test start returns False if already running."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True
        await agent_runner.start()

        # Act
        result = await agent_runner.start()

        # Assert
        assert result is False

        # Cleanup
        await agent_runner.stop()

    @pytest.mark.asyncio
    async def test_start_returns_false_on_model_init_failure(self, agent_runner, mock_model_service):
        """Test start returns False if model initialization fails."""
        # Arrange
        mock_model_service.is_initialized = False
        mock_model_service.initialize.return_value = False

        # Act
        result = await agent_runner.start()

        # Assert
        assert result is False
        assert agent_runner.status == AgentStatus.ERROR

    @pytest.mark.asyncio
    async def test_start_recovers_from_previous_state(self, agent_runner, mock_model_service, db_session):
        """Test start recovers cycle count from previous state."""
        # Arrange - Create existing state
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True

        # Initialize with some cycle count
        agent_runner._state_manager.initialize(agent_runner.config)
        agent_runner._state_manager.update_cycle(cycle_count=42)

        # Act
        result = await agent_runner.start()

        # Assert
        assert result is True
        assert agent_runner._cycle_count == 42

        # Cleanup
        await agent_runner.stop()


class TestAgentRunnerStop:
    """Test AgentRunner stop method."""

    @pytest.mark.asyncio
    async def test_stop_updates_status_to_stopped(self, agent_runner, mock_model_service):
        """Test stop updates status to stopped."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True
        await agent_runner.start()

        # Act
        result = await agent_runner.stop()

        # Assert
        assert result is True
        assert agent_runner.status == AgentStatus.STOPPED
        assert agent_runner._running is False

    @pytest.mark.asyncio
    async def test_stop_cancels_main_task(self, agent_runner, mock_model_service):
        """Test stop cancels main loop task."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True
        await agent_runner.start()
        main_task = agent_runner._main_task

        # Act
        result = await agent_runner.stop()

        # Assert
        assert result is True
        assert main_task.cancelled() or main_task.done()

    @pytest.mark.asyncio
    async def test_stop_stops_command_polling(self, agent_runner, mock_model_service):
        """Test stop stops command polling."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True
        await agent_runner.start()

        # Act
        result = await agent_runner.stop()

        # Assert
        assert result is True
        assert agent_runner._command_handler._running is False

    @pytest.mark.asyncio
    async def test_stop_returns_false_if_not_running(self, agent_runner):
        """Test stop returns False if not running."""
        # Arrange - don't start

        # Act
        result = await agent_runner.stop()

        # Assert
        assert result is False


class TestAgentRunnerPauseResume:
    """Test AgentRunner pause and resume methods."""

    @pytest.mark.asyncio
    async def test_pause_updates_status_to_paused(self, agent_runner, mock_model_service):
        """Test pause updates status to paused."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True
        await agent_runner.start()

        # Act
        result = agent_runner.pause()

        # Assert
        assert result is True
        assert agent_runner.status == AgentStatus.PAUSED

        # Cleanup
        await agent_runner.stop()

    def test_pause_returns_false_if_not_running(self, agent_runner):
        """Test pause returns False if not running."""
        # Arrange - don't start

        # Act
        result = agent_runner.pause()

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_resume_updates_status_to_running(self, agent_runner, mock_model_service):
        """Test resume updates status to running."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True
        await agent_runner.start()
        agent_runner.pause()

        # Act
        result = agent_runner.resume()

        # Assert
        assert result is True
        assert agent_runner.status == AgentStatus.RUNNING

        # Cleanup
        await agent_runner.stop()

    def test_resume_returns_false_if_not_paused(self, agent_runner):
        """Test resume returns False if not paused."""
        # Arrange - don't pause

        # Act
        result = agent_runner.resume()

        # Assert
        assert result is False


class TestAgentRunnerGetStatus:
    """Test AgentRunner get_status method."""

    def test_get_status_returns_status_dict(self, agent_runner, mock_model_service):
        """Test get_status returns status dictionary."""
        # Arrange
        mock_model_service.is_loaded = True

        # Act
        status = agent_runner.get_status()

        # Assert
        assert isinstance(status, dict)
        assert "status" in status
        assert "mode" in status
        assert "cycle_count" in status
        assert "running" in status
        assert "confidence_threshold" in status
        assert "model_loaded" in status

    def test_get_status_includes_cycle_count(self, agent_runner):
        """Test get_status includes current cycle count."""
        # Arrange
        agent_runner._cycle_count = 42

        # Act
        status = agent_runner.get_status()

        # Assert
        assert status["cycle_count"] == 42

    def test_get_status_includes_last_error(self, agent_runner):
        """Test get_status includes last error if present."""
        # Arrange
        agent_runner._last_error = "Test error"

        # Act
        status = agent_runner.get_status()

        # Assert
        assert status["last_error"] == "Test error"


class TestAgentRunnerCommandExecution:
    """Test AgentRunner command execution."""

    @pytest.mark.asyncio
    async def test_execute_command_start(self, agent_runner, mock_model_service):
        """Test executing start command."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True

        # Act
        result = await agent_runner._execute_command("start", {})

        # Assert
        assert result["success"] is True
        assert result["status"] == AgentStatus.RUNNING.value

        # Cleanup
        await agent_runner.stop()

    @pytest.mark.asyncio
    async def test_execute_command_stop(self, agent_runner, mock_model_service):
        """Test executing stop command."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True
        await agent_runner.start()

        # Act
        result = await agent_runner._execute_command("stop", {})

        # Assert
        assert result["success"] is True
        assert result["status"] == AgentStatus.STOPPED.value

    @pytest.mark.asyncio
    async def test_execute_command_pause(self, agent_runner, mock_model_service):
        """Test executing pause command."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True
        await agent_runner.start()

        # Act
        result = await agent_runner._execute_command("pause", {})

        # Assert
        assert result["success"] is True
        assert result["status"] == AgentStatus.PAUSED.value

        # Cleanup
        await agent_runner.stop()

    @pytest.mark.asyncio
    async def test_execute_command_resume(self, agent_runner, mock_model_service):
        """Test executing resume command."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True
        await agent_runner.start()
        agent_runner.pause()

        # Act
        result = await agent_runner._execute_command("resume", {})

        # Assert
        assert result["success"] is True
        assert result["status"] == AgentStatus.RUNNING.value

        # Cleanup
        await agent_runner.stop()

    @pytest.mark.asyncio
    async def test_execute_command_kill(self, agent_runner, mock_model_service):
        """Test executing kill command."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True
        await agent_runner.start()

        # Act
        result = await agent_runner._execute_command("kill", {})

        # Assert
        assert result["status"] == "killed"

        # Verify kill switch activated in state
        state = agent_runner._state_manager.get_state()
        assert state["kill_switch_active"] is True
        assert state["circuit_breaker_state"] == "kill_switch"

    @pytest.mark.asyncio
    async def test_execute_command_update_config(self, agent_runner, config):
        """Test executing update_config command."""
        # Arrange
        updates = {"confidence_threshold": 0.75}

        # Act
        result = await agent_runner._execute_command("update_config", updates)

        # Assert
        assert result["success"] is True
        assert agent_runner.config.confidence_threshold == 0.75
        assert result["config"]["confidence_threshold"] == 0.75

    @pytest.mark.asyncio
    async def test_execute_command_unknown_raises_error(self, agent_runner):
        """Test executing unknown command raises ValueError."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="Unknown command"):
            await agent_runner._execute_command("unknown_command", {})


class TestAgentRunnerCycleExecution:
    """Test AgentRunner cycle execution."""

    @pytest.mark.asyncio
    async def test_execute_cycle_increments_count(self, agent_runner, mock_model_service):
        """Test _execute_cycle increments cycle count."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True
        agent_runner._state_manager.initialize(agent_runner.config)

        # Act
        await agent_runner._execute_cycle()

        # Assert
        assert agent_runner._cycle_count == 1

    @pytest.mark.asyncio
    async def test_execute_cycle_updates_state(self, agent_runner, mock_model_service):
        """Test _execute_cycle updates state manager."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True
        agent_runner._state_manager.initialize(agent_runner.config)

        # Act
        await agent_runner._execute_cycle()

        # Assert
        state = agent_runner._state_manager.get_state()
        assert state["cycle_count"] == 1
        assert state["account_equity"] == agent_runner.config.initial_capital


class TestAgentRunnerMainLoop:
    """Test AgentRunner main loop."""

    @pytest.mark.asyncio
    async def test_main_loop_processes_commands(self, agent_runner, mock_model_service, db_session):
        """Test main loop processes pending commands."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True

        from src.api.database.models import AgentCommand

        session = db_session()
        try:
            cmd = AgentCommand(command="pause", status="pending")
            session.add(cmd)
            session.commit()
        finally:
            session.close()

        # Act
        await agent_runner.start()
        await asyncio.sleep(0.1)  # Let loop process command

        # Assert
        assert agent_runner.status == AgentStatus.PAUSED

        # Verify command was marked completed
        session = db_session()
        try:
            cmd = session.query(AgentCommand).first()
            assert cmd.status == "completed"
        finally:
            session.close()

        # Cleanup
        await agent_runner.stop()

    @pytest.mark.asyncio
    async def test_main_loop_executes_cycles(self, agent_runner, mock_model_service):
        """Test main loop executes trading cycles."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True

        # Act
        await agent_runner.start()
        await asyncio.sleep(1.2)  # Wait for at least one cycle

        # Assert
        assert agent_runner._cycle_count >= 1

        # Cleanup
        await agent_runner.stop()

    @pytest.mark.asyncio
    async def test_main_loop_skips_cycles_when_paused(self, agent_runner, mock_model_service):
        """Test main loop skips cycles when paused."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True

        # Act
        await agent_runner.start()
        agent_runner.pause()
        initial_count = agent_runner._cycle_count
        await asyncio.sleep(1.2)  # Wait

        # Assert - cycle count should not increase
        assert agent_runner._cycle_count == initial_count

        # Cleanup
        await agent_runner.stop()

    @pytest.mark.asyncio
    async def test_main_loop_handles_command_errors(self, agent_runner, mock_model_service, db_session):
        """Test main loop handles command execution errors gracefully."""
        # Arrange
        mock_model_service.is_initialized = True
        mock_model_service.is_loaded = True

        from src.api.database.models import AgentCommand

        session = db_session()
        try:
            # Create command that will fail (invalid command)
            cmd = AgentCommand(command="invalid_command", status="pending")
            session.add(cmd)
            session.commit()
        finally:
            session.close()

        # Act
        await agent_runner.start()
        await asyncio.sleep(0.1)  # Let loop process command

        # Assert - runner should still be running
        assert agent_runner.status == AgentStatus.RUNNING

        # Verify command was marked failed
        session = db_session()
        try:
            cmd = session.query(AgentCommand).first()
            assert cmd.status == "failed"
            assert cmd.error_message is not None
        finally:
            session.close()

        # Cleanup
        await agent_runner.stop()


class TestAgentRunnerDynamicEquityFetching:
    """Test dynamic equity fetching - HIGH PRIORITY FIX."""

    @pytest.mark.asyncio
    async def test_get_current_equity_from_connected_broker(
        self,
        agent_config,
        mock_model_service,
        db_session,
        mock_broker_manager,
    ):
        """Test that _get_current_equity fetches from broker when connected."""
        # Configure for paper mode with broker
        agent_config.mode = "paper"
        agent_config.initial_capital = 100000.0

        agent_runner = AgentRunner(agent_config)

        # Override broker manager with mock
        agent_runner._broker_manager = mock_broker_manager

        # Mock broker is connected and has equity
        mock_broker_manager.is_connected = Mock(return_value=True)
        mock_broker_manager.get_account_info = AsyncMock(return_value={
            "equity": 102500.0,
            "balance": 102000.0,
            "margin_available": 50000.0,
        })

        # Get current equity
        equity = await agent_runner._get_current_equity()

        # Should return broker equity, not initial_capital
        assert equity == 102500.0
        assert equity != agent_config.initial_capital

        # Verify broker was queried
        mock_broker_manager.is_connected.assert_called_once()
        mock_broker_manager.get_account_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_current_equity_broker_disconnected(
        self,
        agent_config,
        mock_model_service,
        db_session,
    ):
        """Test that _get_current_equity falls back to initial_capital when broker disconnected."""
        agent_config.mode = "paper"
        agent_config.initial_capital = 100000.0

        agent_runner = AgentRunner(agent_config)

        # Mock broker manager as disconnected
        mock_broker_manager = Mock(spec=BrokerManager)
        mock_broker_manager.is_connected = Mock(return_value=False)
        agent_runner._broker_manager = mock_broker_manager

        # Get current equity
        equity = await agent_runner._get_current_equity()

        # Should fall back to initial_capital
        assert equity == 100000.0
        assert equity == agent_config.initial_capital

        # Verify broker connection was checked
        mock_broker_manager.is_connected.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_current_equity_broker_error_fallback(
        self,
        agent_config,
        mock_model_service,
        db_session,
    ):
        """Test that _get_current_equity falls back to initial_capital on broker error."""
        agent_config.mode = "paper"
        agent_config.initial_capital = 100000.0

        agent_runner = AgentRunner(agent_config)

        # Mock broker manager that throws error
        mock_broker_manager = Mock(spec=BrokerManager)
        mock_broker_manager.is_connected = Mock(return_value=True)
        mock_broker_manager.get_account_info = AsyncMock(side_effect=Exception("Broker error"))
        agent_runner._broker_manager = mock_broker_manager

        # Get current equity - should handle error gracefully
        equity = await agent_runner._get_current_equity()

        # Should fall back to initial_capital
        assert equity == 100000.0
        assert equity == agent_config.initial_capital

    @pytest.mark.asyncio
    async def test_get_current_equity_no_broker_manager(
        self,
        agent_config,
        mock_model_service,
        db_session,
    ):
        """Test that _get_current_equity handles missing broker manager."""
        agent_config.mode = "simulation"
        agent_config.initial_capital = 100000.0

        agent_runner = AgentRunner(agent_config)

        # No broker manager in simulation mode
        assert agent_runner._broker_manager is None

        # Get current equity
        equity = await agent_runner._get_current_equity()

        # Should return initial_capital
        assert equity == 100000.0
        assert equity == agent_config.initial_capital

    @pytest.mark.asyncio
    async def test_get_current_equity_returns_float(
        self,
        agent_config,
        mock_model_service,
        db_session,
        mock_broker_manager,
    ):
        """Test that _get_current_equity always returns a float."""
        agent_config.mode = "paper"
        agent_config.initial_capital = 100000.0

        agent_runner = AgentRunner(agent_config)
        agent_runner._broker_manager = mock_broker_manager

        # Mock broker with various return types
        mock_broker_manager.is_connected = Mock(return_value=True)
        mock_broker_manager.get_account_info = AsyncMock(return_value={
            "equity": 102500,  # int instead of float
            "balance": 102000.0,
        })

        equity = await agent_runner._get_current_equity()

        # Should return float
        assert isinstance(equity, float)
        assert equity == 102500.0

    @pytest.mark.asyncio
    async def test_get_current_equity_missing_equity_field(
        self,
        agent_config,
        mock_model_service,
        db_session,
        mock_broker_manager,
    ):
        """Test that _get_current_equity handles missing equity field."""
        agent_config.mode = "paper"
        agent_config.initial_capital = 100000.0

        agent_runner = AgentRunner(agent_config)
        agent_runner._broker_manager = mock_broker_manager

        # Mock broker with missing equity field
        mock_broker_manager.is_connected = Mock(return_value=True)
        mock_broker_manager.get_account_info = AsyncMock(return_value={
            "balance": 102000.0,
            # equity field missing
        })

        # Get current equity - should fall back
        equity = await agent_runner._get_current_equity()

        # Should fall back to initial_capital
        assert equity == 100000.0
        assert equity == agent_config.initial_capital

    @pytest.mark.asyncio
    async def test_get_current_equity_zero_equity(
        self,
        agent_config,
        mock_model_service,
        db_session,
        mock_broker_manager,
    ):
        """Test that _get_current_equity handles zero equity."""
        agent_config.mode = "paper"
        agent_config.initial_capital = 100000.0

        agent_runner = AgentRunner(agent_config)
        agent_runner._broker_manager = mock_broker_manager

        # Mock broker with zero equity (margin call scenario)
        mock_broker_manager.is_connected = Mock(return_value=True)
        mock_broker_manager.get_account_info = AsyncMock(return_value={
            "equity": 0.0,
            "balance": 0.0,
        })

        equity = await agent_runner._get_current_equity()

        # Should fall back to initial_capital (zero equity is falsy)
        assert equity == 100000.0
        assert equity == agent_config.initial_capital
