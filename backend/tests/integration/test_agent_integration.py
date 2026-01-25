"""Integration tests for Agent module.

Tests the full flow of command → runner → state persistence.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import importlib.util

# Add src to path and load modules directly
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Load database models
database_path = src_path / "api" / "database"
models_path = database_path / "models.py"
spec = importlib.util.spec_from_file_location("api.database.models", models_path)
models_module = importlib.util.module_from_spec(spec)
sys.modules["api.database.models"] = models_module
spec.loader.exec_module(models_module)

Base = models_module.Base
AgentState = models_module.AgentState
AgentCommand = models_module.AgentCommand

# Load agent modules
agent_path = src_path / "agent"

config_path = agent_path / "config.py"
spec = importlib.util.spec_from_file_location("agent.config", config_path)
config_module = importlib.util.module_from_spec(spec)
sys.modules["agent.config"] = config_module
spec.loader.exec_module(config_module)
AgentConfig = config_module.AgentConfig

# Mock model_service before loading runner
mock_model_service_module = type(sys)("api.services.model_service")
mock_model_service = Mock()
mock_model_service.is_initialized = False
mock_model_service.is_loaded = False
mock_model_service.initialize = Mock(return_value=True)
mock_model_service_module.model_service = mock_model_service
sys.modules["api.services.model_service"] = mock_model_service_module

# Mock get_session
mock_session_module = type(sys)("api.database.session")
mock_session_module.get_session = lambda: None
sys.modules["api.database.session"] = mock_session_module

# Now load runner
runner_path = agent_path / "runner.py"
spec = importlib.util.spec_from_file_location("agent.runner", runner_path)
runner_module = importlib.util.module_from_spec(spec)
sys.modules["agent.runner"] = runner_module
spec.loader.exec_module(runner_module)

AgentRunner = runner_module.AgentRunner
AgentStatus = runner_module.AgentStatus
CommandHandler = runner_module.CommandHandler
StateManager = runner_module.StateManager


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
        cycle_interval_seconds=1,
        initial_capital=100000.0,
    )


@pytest.fixture
def agent_runner(config, db_session):
    """Create an AgentRunner with real dependencies."""
    # Reset mock_model_service
    mock_model_service.is_initialized = True
    mock_model_service.is_loaded = True
    mock_model_service.initialize = Mock(return_value=True)

    # Create runner and inject db_session
    runner = AgentRunner(config)
    runner._command_handler = CommandHandler(db_session)
    runner._state_manager = StateManager(db_session)
    yield runner


class TestAgentCommandFlow:
    """Test full command processing flow."""

    @pytest.mark.asyncio
    async def test_start_command_starts_agent(self, agent_runner, db_session, mock_model_service):
        """Test start command successfully starts agent."""
        # Arrange - Create start command
        session = db_session()
        try:
            cmd = AgentCommand(command="start", status="pending")
            session.add(cmd)
            session.commit()
            command_id = cmd.id
        finally:
            session.close()

        # Act - Start agent (will process command)
        await agent_runner.start()
        await asyncio.sleep(0.1)

        # Assert - Agent should be running
        assert agent_runner.status == AgentStatus.RUNNING

        # Verify command was processed
        session = db_session()
        try:
            cmd = session.query(AgentCommand).filter_by(id=command_id).first()
            assert cmd.status == "completed"
            assert cmd.result is not None
        finally:
            session.close()

        # Cleanup
        await agent_runner.stop()

    @pytest.mark.asyncio
    async def test_stop_command_stops_agent(self, agent_runner, db_session, mock_model_service):
        """Test stop command successfully stops agent."""
        # Arrange - Start agent first
        await agent_runner.start()
        await asyncio.sleep(0.1)

        # Create stop command
        session = db_session()
        try:
            cmd = AgentCommand(command="stop", status="pending")
            session.add(cmd)
            session.commit()
            command_id = cmd.id
        finally:
            session.close()

        # Act - Wait for command to be processed
        await asyncio.sleep(0.2)

        # Assert - Agent should be stopped
        assert agent_runner.status == AgentStatus.STOPPED

        # Verify command was processed
        session = db_session()
        try:
            cmd = session.query(AgentCommand).filter_by(id=command_id).first()
            assert cmd.status == "completed"
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_multiple_commands_processed_in_order(self, agent_runner, db_session, mock_model_service):
        """Test multiple commands are processed in correct order."""
        # Arrange - Create multiple commands
        session = db_session()
        try:
            cmd1 = AgentCommand(
                command="start",
                status="pending",
                created_at=datetime.utcnow(),
            )
            cmd2 = AgentCommand(
                command="pause",
                status="pending",
                created_at=datetime.utcnow(),
            )
            session.add_all([cmd1, cmd2])
            session.commit()
        finally:
            session.close()

        # Act - Start agent and wait for commands to process
        await agent_runner.start()
        await asyncio.sleep(0.2)

        # Assert - Agent should be paused (second command)
        assert agent_runner.status == AgentStatus.PAUSED

        # Verify both commands were processed
        session = db_session()
        try:
            commands = session.query(AgentCommand).order_by(AgentCommand.id).all()
            assert len(commands) == 2
            assert commands[0].status == "completed"
            assert commands[1].status == "completed"
        finally:
            session.close()

        # Cleanup
        await agent_runner.stop()

    @pytest.mark.asyncio
    async def test_failed_command_marked_as_failed(self, agent_runner, db_session, mock_model_service):
        """Test failed commands are marked as failed in database."""
        # Arrange - Create invalid command
        session = db_session()
        try:
            cmd = AgentCommand(command="invalid_command", status="pending")
            session.add(cmd)
            session.commit()
            command_id = cmd.id
        finally:
            session.close()

        # Act
        await agent_runner.start()
        await asyncio.sleep(0.1)

        # Assert - Command should be marked failed
        session = db_session()
        try:
            cmd = session.query(AgentCommand).filter_by(id=command_id).first()
            assert cmd.status == "failed"
            assert "Unknown command" in cmd.error_message
        finally:
            session.close()

        # Cleanup
        await agent_runner.stop()


class TestAgentStatePersistence:
    """Test state persistence across operations."""

    @pytest.mark.asyncio
    async def test_state_persists_during_operation(self, agent_runner, db_session, mock_model_service):
        """Test state is persisted during agent operation."""
        # Arrange & Act
        await agent_runner.start()
        await asyncio.sleep(1.2)  # Wait for cycles to execute

        # Assert - State should be persisted
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state is not None
            assert state.status == "running"
            assert state.mode == "simulation"
            assert state.cycle_count >= 1
        finally:
            session.close()

        # Cleanup
        await agent_runner.stop()

    @pytest.mark.asyncio
    async def test_state_updated_on_status_change(self, agent_runner, db_session, mock_model_service):
        """Test state is updated when status changes."""
        # Arrange
        await agent_runner.start()
        await asyncio.sleep(0.1)

        # Act - Pause agent
        agent_runner.pause()
        await asyncio.sleep(0.1)

        # Assert - State should reflect pause
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state.status == "paused"
        finally:
            session.close()

        # Cleanup
        await agent_runner.stop()

    @pytest.mark.asyncio
    async def test_state_updated_on_cycle_execution(self, agent_runner, db_session, mock_model_service):
        """Test state is updated after each cycle."""
        # Arrange & Act
        await agent_runner.start()
        await asyncio.sleep(1.2)  # Wait for cycles

        # Assert - Cycle count and timestamp should be updated
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state.cycle_count >= 1
            assert state.last_cycle_at is not None
            assert state.account_equity == 100000.0
        finally:
            session.close()

        # Cleanup
        await agent_runner.stop()


class TestAgentCrashRecovery:
    """Test crash recovery scenarios."""

    @pytest.mark.asyncio
    async def test_recovers_cycle_count_after_restart(self, config, db_session, mock_model_service):
        """Test agent recovers cycle count after restart."""
        # Arrange - Start agent, run cycles, then stop
        with patch("src.agent.runner.get_session", db_session):
            with patch("src.agent.runner.model_service", mock_model_service):
                runner1 = AgentRunner(config)
                await runner1.start()
                await asyncio.sleep(1.2)  # Let some cycles run
                cycle_count_before_stop = runner1._cycle_count
                await runner1.stop()

        # Act - Create new runner (simulating restart)
        with patch("src.agent.runner.get_session", db_session):
            with patch("src.agent.runner.model_service", mock_model_service):
                runner2 = AgentRunner(config)
                await runner2.start()
                await asyncio.sleep(0.1)

                # Assert - Cycle count should be recovered
                assert runner2._cycle_count == cycle_count_before_stop

                # Cleanup
                await runner2.stop()

    @pytest.mark.asyncio
    async def test_recovers_from_error_state(self, agent_runner, db_session, mock_model_service):
        """Test agent can recover from error state."""
        # Arrange - Put agent in error state
        agent_runner._state_manager.initialize(agent_runner.config)
        agent_runner._state_manager.update_status("error", error_message="Test error")

        # Act - Start agent (should recover)
        result = await agent_runner.start()

        # Assert
        assert result is True
        assert agent_runner.status == AgentStatus.RUNNING

        # Verify error cleared
        state = agent_runner._state_manager.get_state()
        assert state["error_message"] is None

        # Cleanup
        await agent_runner.stop()


class TestAgentCircuitBreaker:
    """Test circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_kill_switch_stops_agent(self, agent_runner, db_session, mock_model_service):
        """Test kill switch command stops agent and activates kill switch."""
        # Arrange
        await agent_runner.start()
        await asyncio.sleep(0.1)

        # Create kill command
        session = db_session()
        try:
            cmd = AgentCommand(command="kill", status="pending")
            session.add(cmd)
            session.commit()
        finally:
            session.close()

        # Act - Wait for command to be processed
        await asyncio.sleep(0.2)

        # Assert - Agent should be stopped
        assert agent_runner.status == AgentStatus.STOPPED

        # Verify kill switch activated
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state.kill_switch_active is True
            assert state.circuit_breaker_state == "kill_switch"
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_persisted(self, agent_runner, db_session, mock_model_service):
        """Test circuit breaker state is persisted to database."""
        # Arrange
        agent_runner._state_manager.initialize(agent_runner.config)

        # Act
        agent_runner._state_manager.update_circuit_breaker(
            circuit_breaker_state="triggered",
            kill_switch_active=False,
        )

        # Assert
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state.circuit_breaker_state == "triggered"
            assert state.kill_switch_active is False
        finally:
            session.close()


class TestAgentConfigurationUpdates:
    """Test configuration update integration."""

    @pytest.mark.asyncio
    async def test_config_update_command_updates_config(self, agent_runner, db_session, mock_model_service):
        """Test update_config command updates configuration."""
        # Arrange
        await agent_runner.start()
        await asyncio.sleep(0.1)

        # Create update_config command
        session = db_session()
        try:
            cmd = AgentCommand(
                command="update_config",
                status="pending",
                payload={"confidence_threshold": 0.75, "max_position_size": 0.15},
            )
            session.add(cmd)
            session.commit()
        finally:
            session.close()

        # Act - Wait for command to be processed
        await asyncio.sleep(0.2)

        # Assert - Config should be updated
        assert agent_runner.config.confidence_threshold == 0.75
        assert agent_runner.config.max_position_size == 0.15

        # Verify config persisted to state
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state.config["confidence_threshold"] == 0.75
        finally:
            session.close()

        # Cleanup
        await agent_runner.stop()


class TestConcurrentOperations:
    """Test concurrent command handling."""

    @pytest.mark.asyncio
    async def test_handles_rapid_command_submission(self, agent_runner, db_session, mock_model_service):
        """Test agent handles rapid command submission."""
        # Arrange
        await agent_runner.start()
        await asyncio.sleep(0.1)

        # Act - Submit multiple commands rapidly
        session = db_session()
        try:
            commands = [
                AgentCommand(command="pause", status="pending"),
                AgentCommand(command="resume", status="pending"),
                AgentCommand(command="pause", status="pending"),
            ]
            for cmd in commands:
                session.add(cmd)
            session.commit()
        finally:
            session.close()

        # Wait for processing
        await asyncio.sleep(0.5)

        # Assert - All commands should be processed
        session = db_session()
        try:
            processed = (
                session.query(AgentCommand)
                .filter(AgentCommand.status.in_(["completed", "failed"]))
                .all()
            )
            assert len(processed) == 3
        finally:
            session.close()

        # Cleanup
        await agent_runner.stop()
