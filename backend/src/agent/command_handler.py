"""Agent command polling and processing.

Polls the agent_commands table for new commands and processes them.
Updates command status as processing progresses.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from ..api.database.models import AgentCommand

logger = logging.getLogger(__name__)


class CommandHandler:
    """Handles command polling and processing from database.

    Polls agent_commands table for pending commands and returns them
    to the AgentRunner for execution. Updates command status throughout
    the processing lifecycle.
    """

    # Poll interval in seconds
    POLL_INTERVAL = 1.0

    def __init__(self, session_factory):
        """Initialize command handler.

        Args:
            session_factory: Callable that returns a new database session
        """
        self._session_factory = session_factory
        self._running = False

    async def start_polling(self) -> None:
        """Start polling for commands."""
        if self._running:
            logger.warning("Command polling already running")
            return

        self._running = True
        logger.info("Started command polling")

    async def stop_polling(self) -> None:
        """Stop polling for commands."""
        self._running = False
        logger.info("Stopped command polling")

    async def poll_commands(self) -> List[Dict[str, Any]]:
        """Poll for pending commands.

        Returns:
            List of pending command dictionaries
        """
        if not self._running:
            return []

        try:
            # Run database query in executor to avoid blocking
            loop = asyncio.get_event_loop()
            commands = await loop.run_in_executor(None, self._get_pending_commands)
            return commands

        except Exception as e:
            logger.error(f"Error polling commands: {e}")
            return []

    def _get_pending_commands(self) -> List[Dict[str, Any]]:
        """Get pending commands from database (sync method).

        Returns:
            List of pending command dictionaries
        """
        try:
            session = self._session_factory()
            try:
                # Get all pending commands, ordered by creation time
                commands = (
                    session.query(AgentCommand)
                    .filter_by(status="pending")
                    .order_by(AgentCommand.created_at.asc())
                    .all()
                )

                return [
                    {
                        "id": cmd.id,
                        "command": cmd.command,
                        "payload": cmd.payload or {},
                        "created_at": cmd.created_at,
                    }
                    for cmd in commands
                ]

            finally:
                session.close()

        except SQLAlchemyError as e:
            logger.error(f"Database error getting commands: {e}")
            return []

    def mark_processing(self, command_id: int) -> bool:
        """Mark a command as processing.

        Args:
            command_id: ID of the command

        Returns:
            True if successful, False otherwise
        """
        try:
            session = self._session_factory()
            try:
                command = session.query(AgentCommand).filter_by(id=command_id).first()

                if not command:
                    logger.error(f"Command {command_id} not found")
                    return False

                command.status = "processing"
                command.processed_at = datetime.utcnow()
                session.commit()

                logger.info(f"Marked command {command_id} ({command.command}) as processing")
                return True

            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()

        except SQLAlchemyError as e:
            logger.error(f"Failed to mark command {command_id} as processing: {e}")
            return False

    def mark_completed(
        self,
        command_id: int,
        result: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Mark a command as completed.

        Args:
            command_id: ID of the command
            result: Optional result data

        Returns:
            True if successful, False otherwise
        """
        try:
            session = self._session_factory()
            try:
                command = session.query(AgentCommand).filter_by(id=command_id).first()

                if not command:
                    logger.error(f"Command {command_id} not found")
                    return False

                command.status = "completed"
                command.result = result
                command.error_message = None
                session.commit()

                logger.info(f"Marked command {command_id} ({command.command}) as completed")
                return True

            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()

        except SQLAlchemyError as e:
            logger.error(f"Failed to mark command {command_id} as completed: {e}")
            return False

    def mark_failed(
        self,
        command_id: int,
        error_message: str,
    ) -> bool:
        """Mark a command as failed.

        Args:
            command_id: ID of the command
            error_message: Error message describing the failure

        Returns:
            True if successful, False otherwise
        """
        try:
            session = self._session_factory()
            try:
                command = session.query(AgentCommand).filter_by(id=command_id).first()

                if not command:
                    logger.error(f"Command {command_id} not found")
                    return False

                command.status = "failed"
                command.error_message = error_message
                session.commit()

                logger.error(
                    f"Marked command {command_id} ({command.command}) as failed: {error_message}"
                )
                return True

            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()

        except SQLAlchemyError as e:
            logger.error(f"Failed to mark command {command_id} as failed: {e}")
            return False

    async def wait_for_command(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Wait for the next command with optional timeout.

        Args:
            timeout: Maximum time to wait in seconds (None for no timeout)

        Returns:
            Next command dictionary, or None if timeout or stopped
        """
        start_time = asyncio.get_event_loop().time()

        while self._running:
            commands = await self.poll_commands()

            if commands:
                return commands[0]  # Return first pending command

            # Check timeout
            if timeout is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    return None

            # Wait before polling again
            await asyncio.sleep(self.POLL_INTERVAL)

        return None

    def get_command_status(self, command_id: int) -> Optional[str]:
        """Get the status of a command.

        Args:
            command_id: ID of the command

        Returns:
            Command status string, or None if not found
        """
        try:
            session = self._session_factory()
            try:
                command = session.query(AgentCommand).filter_by(id=command_id).first()

                if not command:
                    return None

                return command.status

            finally:
                session.close()

        except SQLAlchemyError as e:
            logger.error(f"Failed to get command status: {e}")
            return None

    def cleanup_old_commands(self, days: int = 7) -> int:
        """Clean up old completed/failed commands.

        Args:
            days: Delete commands older than this many days

        Returns:
            Number of commands deleted
        """
        try:
            session = self._session_factory()
            try:
                cutoff_date = datetime.utcnow() - timedelta(days=days)

                deleted = (
                    session.query(AgentCommand)
                    .filter(
                        AgentCommand.status.in_(["completed", "failed"]),
                        AgentCommand.created_at < cutoff_date,
                    )
                    .delete(synchronize_session=False)
                )

                session.commit()

                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} old commands")

                return deleted

            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()

        except SQLAlchemyError as e:
            logger.error(f"Failed to cleanup old commands: {e}")
            return 0

