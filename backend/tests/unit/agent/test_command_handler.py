"""Unit tests for CommandHandler."""

import pytest
import asyncio
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import from conftest
from .conftest import CommandHandler, Base, AgentCommand


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal


@pytest.fixture
def command_handler(db_session):
    """Create a CommandHandler instance with test database."""
    return CommandHandler(db_session)


class TestCommandHandlerPolling:
    """Test CommandHandler polling methods."""

    @pytest.mark.asyncio
    async def test_start_polling_sets_running_flag(self, command_handler):
        """Test start_polling sets running flag."""
        # Arrange & Act
        await command_handler.start_polling()

        # Assert
        assert command_handler._running is True

    @pytest.mark.asyncio
    async def test_start_polling_is_idempotent(self, command_handler):
        """Test start_polling can be called multiple times safely."""
        # Arrange & Act
        await command_handler.start_polling()
        await command_handler.start_polling()

        # Assert
        assert command_handler._running is True

    @pytest.mark.asyncio
    async def test_stop_polling_clears_running_flag(self, command_handler):
        """Test stop_polling clears running flag."""
        # Arrange
        await command_handler.start_polling()

        # Act
        await command_handler.stop_polling()

        # Assert
        assert command_handler._running is False

    @pytest.mark.asyncio
    async def test_poll_commands_returns_empty_when_not_running(self, command_handler, db_session):
        """Test poll_commands returns empty list when not running."""
        # Arrange - add commands but don't start polling
        session = db_session()
        try:
            cmd = AgentCommand(command="start", status="pending")
            session.add(cmd)
            session.commit()
        finally:
            session.close()

        # Act
        commands = await command_handler.poll_commands()

        # Assert
        assert commands == []

    @pytest.mark.asyncio
    async def test_poll_commands_returns_pending_commands(self, command_handler, db_session):
        """Test poll_commands returns pending commands."""
        # Arrange
        session = db_session()
        try:
            cmd1 = AgentCommand(command="start", status="pending")
            cmd2 = AgentCommand(command="pause", status="pending")
            session.add_all([cmd1, cmd2])
            session.commit()
        finally:
            session.close()

        await command_handler.start_polling()

        # Act
        commands = await command_handler.poll_commands()

        # Assert
        assert len(commands) == 2
        assert commands[0]["command"] == "start"
        assert commands[1]["command"] == "pause"

    @pytest.mark.asyncio
    async def test_poll_commands_excludes_non_pending_commands(self, command_handler, db_session):
        """Test poll_commands excludes completed and failed commands."""
        # Arrange
        session = db_session()
        try:
            cmd1 = AgentCommand(command="start", status="pending")
            cmd2 = AgentCommand(command="pause", status="completed")
            cmd3 = AgentCommand(command="stop", status="failed")
            session.add_all([cmd1, cmd2, cmd3])
            session.commit()
        finally:
            session.close()

        await command_handler.start_polling()

        # Act
        commands = await command_handler.poll_commands()

        # Assert
        assert len(commands) == 1
        assert commands[0]["command"] == "start"

    @pytest.mark.asyncio
    async def test_poll_commands_orders_by_created_at(self, command_handler, db_session):
        """Test poll_commands returns commands ordered by creation time."""
        # Arrange
        base_time = datetime.utcnow()
        session = db_session()
        try:
            cmd1 = AgentCommand(
                command="third",
                status="pending",
                created_at=base_time + timedelta(seconds=2),
            )
            cmd2 = AgentCommand(
                command="first",
                status="pending",
                created_at=base_time,
            )
            cmd3 = AgentCommand(
                command="second",
                status="pending",
                created_at=base_time + timedelta(seconds=1),
            )
            session.add_all([cmd1, cmd2, cmd3])
            session.commit()
        finally:
            session.close()

        await command_handler.start_polling()

        # Act
        commands = await command_handler.poll_commands()

        # Assert
        assert len(commands) == 3
        assert commands[0]["command"] == "first"
        assert commands[1]["command"] == "second"
        assert commands[2]["command"] == "third"

    @pytest.mark.asyncio
    async def test_poll_commands_includes_payload(self, command_handler, db_session):
        """Test poll_commands includes command payload."""
        # Arrange
        session = db_session()
        try:
            cmd = AgentCommand(
                command="update_config",
                status="pending",
                payload={"mode": "paper", "confidence": 0.75},
            )
            session.add(cmd)
            session.commit()
        finally:
            session.close()

        await command_handler.start_polling()

        # Act
        commands = await command_handler.poll_commands()

        # Assert
        assert len(commands) == 1
        assert commands[0]["payload"] == {"mode": "paper", "confidence": 0.75}


class TestCommandHandlerMarkProcessing:
    """Test CommandHandler mark_processing method."""

    def test_mark_processing_updates_status(self, command_handler, db_session):
        """Test mark_processing updates command status to processing."""
        # Arrange
        session = db_session()
        try:
            cmd = AgentCommand(command="start", status="pending")
            session.add(cmd)
            session.commit()
            command_id = cmd.id
        finally:
            session.close()

        # Act
        result = command_handler.mark_processing(command_id)

        # Assert
        assert result is True
        session = db_session()
        try:
            cmd = session.query(AgentCommand).filter_by(id=command_id).first()
            assert cmd.status == "processing"
        finally:
            session.close()

    def test_mark_processing_sets_processed_at_timestamp(self, command_handler, db_session):
        """Test mark_processing sets processed_at timestamp."""
        # Arrange
        session = db_session()
        try:
            cmd = AgentCommand(command="start", status="pending")
            session.add(cmd)
            session.commit()
            command_id = cmd.id
        finally:
            session.close()

        # Act
        before_mark = datetime.utcnow()
        result = command_handler.mark_processing(command_id)

        # Assert
        assert result is True
        session = db_session()
        try:
            cmd = session.query(AgentCommand).filter_by(id=command_id).first()
            assert cmd.processed_at is not None
            assert cmd.processed_at >= before_mark
        finally:
            session.close()

    def test_mark_processing_returns_false_for_nonexistent_command(self, command_handler):
        """Test mark_processing returns False for nonexistent command."""
        # Arrange
        nonexistent_id = 999

        # Act
        result = command_handler.mark_processing(nonexistent_id)

        # Assert
        assert result is False


class TestCommandHandlerMarkCompleted:
    """Test CommandHandler mark_completed method."""

    def test_mark_completed_updates_status(self, command_handler, db_session):
        """Test mark_completed updates command status to completed."""
        # Arrange
        session = db_session()
        try:
            cmd = AgentCommand(command="start", status="processing")
            session.add(cmd)
            session.commit()
            command_id = cmd.id
        finally:
            session.close()

        # Act
        result = command_handler.mark_completed(command_id)

        # Assert
        assert result is True
        session = db_session()
        try:
            cmd = session.query(AgentCommand).filter_by(id=command_id).first()
            assert cmd.status == "completed"
        finally:
            session.close()

    def test_mark_completed_stores_result(self, command_handler, db_session):
        """Test mark_completed stores result data."""
        # Arrange
        session = db_session()
        try:
            cmd = AgentCommand(command="start", status="processing")
            session.add(cmd)
            session.commit()
            command_id = cmd.id
        finally:
            session.close()

        result_data = {"success": True, "message": "Agent started"}

        # Act
        result = command_handler.mark_completed(command_id, result_data)

        # Assert
        assert result is True
        session = db_session()
        try:
            cmd = session.query(AgentCommand).filter_by(id=command_id).first()
            assert cmd.result == result_data
        finally:
            session.close()

    def test_mark_completed_clears_error_message(self, command_handler, db_session):
        """Test mark_completed clears error message."""
        # Arrange
        session = db_session()
        try:
            cmd = AgentCommand(
                command="start",
                status="processing",
                error_message="Previous error",
            )
            session.add(cmd)
            session.commit()
            command_id = cmd.id
        finally:
            session.close()

        # Act
        result = command_handler.mark_completed(command_id)

        # Assert
        assert result is True
        session = db_session()
        try:
            cmd = session.query(AgentCommand).filter_by(id=command_id).first()
            assert cmd.error_message is None
        finally:
            session.close()

    def test_mark_completed_returns_false_for_nonexistent_command(self, command_handler):
        """Test mark_completed returns False for nonexistent command."""
        # Arrange
        nonexistent_id = 999

        # Act
        result = command_handler.mark_completed(nonexistent_id)

        # Assert
        assert result is False


class TestCommandHandlerMarkFailed:
    """Test CommandHandler mark_failed method."""

    def test_mark_failed_updates_status(self, command_handler, db_session):
        """Test mark_failed updates command status to failed."""
        # Arrange
        session = db_session()
        try:
            cmd = AgentCommand(command="start", status="processing")
            session.add(cmd)
            session.commit()
            command_id = cmd.id
        finally:
            session.close()

        # Act
        result = command_handler.mark_failed(command_id, "Connection timeout")

        # Assert
        assert result is True
        session = db_session()
        try:
            cmd = session.query(AgentCommand).filter_by(id=command_id).first()
            assert cmd.status == "failed"
        finally:
            session.close()

    def test_mark_failed_stores_error_message(self, command_handler, db_session):
        """Test mark_failed stores error message."""
        # Arrange
        session = db_session()
        try:
            cmd = AgentCommand(command="start", status="processing")
            session.add(cmd)
            session.commit()
            command_id = cmd.id
        finally:
            session.close()

        # Act
        result = command_handler.mark_failed(command_id, "Connection timeout")

        # Assert
        assert result is True
        session = db_session()
        try:
            cmd = session.query(AgentCommand).filter_by(id=command_id).first()
            assert cmd.error_message == "Connection timeout"
        finally:
            session.close()

    def test_mark_failed_returns_false_for_nonexistent_command(self, command_handler):
        """Test mark_failed returns False for nonexistent command."""
        # Arrange
        nonexistent_id = 999

        # Act
        result = command_handler.mark_failed(nonexistent_id, "Error message")

        # Assert
        assert result is False


class TestCommandHandlerWaitForCommand:
    """Test CommandHandler wait_for_command method."""

    @pytest.mark.asyncio
    async def test_wait_for_command_returns_first_pending_command(self, command_handler, db_session):
        """Test wait_for_command returns first pending command."""
        # Arrange
        session = db_session()
        try:
            cmd = AgentCommand(command="start", status="pending")
            session.add(cmd)
            session.commit()
        finally:
            session.close()

        await command_handler.start_polling()

        # Act
        command = await command_handler.wait_for_command(timeout=1.0)

        # Assert
        assert command is not None
        assert command["command"] == "start"

    @pytest.mark.asyncio
    async def test_wait_for_command_returns_none_on_timeout(self, command_handler):
        """Test wait_for_command returns None after timeout."""
        # Arrange - no commands
        await command_handler.start_polling()

        # Act
        command = await command_handler.wait_for_command(timeout=0.1)

        # Assert
        assert command is None

    @pytest.mark.asyncio
    async def test_wait_for_command_returns_none_when_stopped(self, command_handler, db_session):
        """Test wait_for_command returns None when polling stopped."""
        # Arrange
        session = db_session()
        try:
            cmd = AgentCommand(command="start", status="pending")
            session.add(cmd)
            session.commit()
        finally:
            session.close()

        # Don't start polling

        # Act
        command = await command_handler.wait_for_command(timeout=0.1)

        # Assert
        assert command is None

    @pytest.mark.asyncio
    async def test_wait_for_command_waits_for_command_to_arrive(self, command_handler, db_session):
        """Test wait_for_command waits for command to arrive."""
        # Arrange
        await command_handler.start_polling()

        # Add command after a short delay
        async def add_command_later():
            await asyncio.sleep(0.1)
            session = db_session()
            try:
                cmd = AgentCommand(command="start", status="pending")
                session.add(cmd)
                session.commit()
            finally:
                session.close()

        asyncio.create_task(add_command_later())

        # Act
        command = await command_handler.wait_for_command(timeout=1.0)

        # Assert
        assert command is not None
        assert command["command"] == "start"


class TestCommandHandlerGetCommandStatus:
    """Test CommandHandler get_command_status method."""

    def test_get_command_status_returns_status(self, command_handler, db_session):
        """Test get_command_status returns command status."""
        # Arrange
        session = db_session()
        try:
            cmd = AgentCommand(command="start", status="pending")
            session.add(cmd)
            session.commit()
            command_id = cmd.id
        finally:
            session.close()

        # Act
        status = command_handler.get_command_status(command_id)

        # Assert
        assert status == "pending"

    def test_get_command_status_returns_none_for_nonexistent_command(self, command_handler):
        """Test get_command_status returns None for nonexistent command."""
        # Arrange
        nonexistent_id = 999

        # Act
        status = command_handler.get_command_status(nonexistent_id)

        # Assert
        assert status is None


class TestCommandHandlerCleanup:
    """Test CommandHandler cleanup_old_commands method."""

    def test_cleanup_old_commands_deletes_old_completed(self, command_handler, db_session):
        """Test cleanup_old_commands deletes old completed commands."""
        # Arrange
        old_date = datetime.utcnow() - timedelta(days=10)
        session = db_session()
        try:
            cmd1 = AgentCommand(command="start", status="completed", created_at=old_date)
            cmd2 = AgentCommand(command="pause", status="pending")
            session.add_all([cmd1, cmd2])
            session.commit()
        finally:
            session.close()

        # Act
        deleted_count = command_handler.cleanup_old_commands(days=7)

        # Assert
        assert deleted_count == 1
        session = db_session()
        try:
            remaining = session.query(AgentCommand).all()
            assert len(remaining) == 1
            assert remaining[0].command == "pause"
        finally:
            session.close()

    def test_cleanup_old_commands_deletes_old_failed(self, command_handler, db_session):
        """Test cleanup_old_commands deletes old failed commands."""
        # Arrange
        old_date = datetime.utcnow() - timedelta(days=10)
        session = db_session()
        try:
            cmd1 = AgentCommand(command="start", status="failed", created_at=old_date)
            cmd2 = AgentCommand(command="pause", status="pending")
            session.add_all([cmd1, cmd2])
            session.commit()
        finally:
            session.close()

        # Act
        deleted_count = command_handler.cleanup_old_commands(days=7)

        # Assert
        assert deleted_count == 1

    def test_cleanup_old_commands_preserves_recent_commands(self, command_handler, db_session):
        """Test cleanup_old_commands preserves recent commands."""
        # Arrange
        recent_date = datetime.utcnow() - timedelta(days=3)
        session = db_session()
        try:
            cmd1 = AgentCommand(command="start", status="completed", created_at=recent_date)
            cmd2 = AgentCommand(command="pause", status="failed", created_at=recent_date)
            session.add_all([cmd1, cmd2])
            session.commit()
        finally:
            session.close()

        # Act
        deleted_count = command_handler.cleanup_old_commands(days=7)

        # Assert
        assert deleted_count == 0
        session = db_session()
        try:
            remaining = session.query(AgentCommand).all()
            assert len(remaining) == 2
        finally:
            session.close()

    def test_cleanup_old_commands_preserves_pending_and_processing(self, command_handler, db_session):
        """Test cleanup_old_commands preserves pending and processing commands."""
        # Arrange
        old_date = datetime.utcnow() - timedelta(days=10)
        session = db_session()
        try:
            cmd1 = AgentCommand(command="start", status="pending", created_at=old_date)
            cmd2 = AgentCommand(command="pause", status="processing", created_at=old_date)
            session.add_all([cmd1, cmd2])
            session.commit()
        finally:
            session.close()

        # Act
        deleted_count = command_handler.cleanup_old_commands(days=7)

        # Assert
        assert deleted_count == 0
        session = db_session()
        try:
            remaining = session.query(AgentCommand).all()
            assert len(remaining) == 2
        finally:
            session.close()

    def test_cleanup_old_commands_returns_zero_when_no_old_commands(self, command_handler):
        """Test cleanup_old_commands returns 0 when no old commands exist."""
        # Arrange - empty database

        # Act
        deleted_count = command_handler.cleanup_old_commands(days=7)

        # Assert
        assert deleted_count == 0
