"""Agent state persistence to database.

Manages agent state in the agent_state table for crash recovery and monitoring.
Uses single-row pattern with upsert.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from ..api.database.models import AgentState
from .config import AgentConfig

logger = logging.getLogger(__name__)


class StateManager:
    """Manages agent state persistence to database.

    Uses single-row pattern - there is only one active agent state record.
    Updates are performed via upsert (update if exists, insert if not).
    """

    def __init__(self, session_factory):
        """Initialize state manager.

        Args:
            session_factory: Callable that returns a new database session
        """
        self._session_factory = session_factory
        self._state_id: Optional[int] = None

    def initialize(self, config: AgentConfig) -> bool:
        """Initialize state record or load existing state.

        Args:
            config: Agent configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            session = self._session_factory()
            try:
                # Check if state already exists
                existing_state = session.query(AgentState).first()

                if existing_state:
                    self._state_id = existing_state.id
                    logger.info(
                        f"Loaded existing agent state (id={self._state_id}, "
                        f"status={existing_state.status}, "
                        f"cycle_count={existing_state.cycle_count})"
                    )
                else:
                    # Create initial state
                    new_state = AgentState(
                        status="stopped",
                        mode=config.mode,
                        cycle_count=0,
                        open_positions=0,
                        kill_switch_active=False,
                        config=config.to_dict(),
                        updated_at=datetime.utcnow(),
                    )
                    session.add(new_state)
                    session.commit()
                    self._state_id = new_state.id
                    logger.info(f"Created new agent state (id={self._state_id})")

                return True

            finally:
                session.close()

        except SQLAlchemyError as e:
            logger.error(f"Failed to initialize state: {e}")
            return False

    def update_status(
        self,
        status: str,
        error_message: Optional[str] = None,
    ) -> bool:
        """Update agent status.

        Args:
            status: New status (stopped, starting, running, paused, stopping, error)
            error_message: Optional error message if status is 'error'

        Returns:
            True if successful, False otherwise
        """
        return self._update_state(
            status=status,
            error_message=error_message,
        )

    def update_cycle(
        self,
        cycle_count: int,
        last_prediction: Optional[Dict] = None,
        last_signal: Optional[Dict] = None,
        account_equity: Optional[float] = None,
        open_positions: Optional[int] = None,
    ) -> bool:
        """Update state after a trading cycle.

        Args:
            cycle_count: Current cycle number
            last_prediction: Latest prediction data
            last_signal: Latest trading signal
            account_equity: Current account equity
            open_positions: Number of open positions

        Returns:
            True if successful, False otherwise
        """
        return self._update_state(
            cycle_count=cycle_count,
            last_cycle_at=datetime.utcnow(),
            last_prediction=last_prediction,
            last_signal=last_signal,
            account_equity=account_equity,
            open_positions=open_positions,
        )

    def update_circuit_breaker(
        self,
        circuit_breaker_state: Optional[str],
        kill_switch_active: bool = False,
    ) -> bool:
        """Update circuit breaker state.

        Args:
            circuit_breaker_state: Current circuit breaker status
            kill_switch_active: Whether kill switch is activated

        Returns:
            True if successful, False otherwise
        """
        return self._update_state(
            circuit_breaker_state=circuit_breaker_state,
            kill_switch_active=kill_switch_active,
        )

    def update_config(self, config: AgentConfig) -> bool:
        """Update stored configuration.

        Args:
            config: New agent configuration

        Returns:
            True if successful, False otherwise
        """
        return self._update_state(
            mode=config.mode,
            config=config.to_dict(),
        )

    def set_started(self) -> bool:
        """Mark agent as started.

        Returns:
            True if successful, False otherwise
        """
        return self._update_state(
            status="running",
            started_at=datetime.utcnow(),
            error_message=None,
        )

    def set_stopped(self) -> bool:
        """Mark agent as stopped.

        Returns:
            True if successful, False otherwise
        """
        return self._update_state(
            status="stopped",
            started_at=None,
            error_message=None,
        )

    def get_state(self) -> Optional[Dict[str, Any]]:
        """Get current agent state.

        Returns:
            Dictionary with current state, or None if error
        """
        try:
            session = self._session_factory()
            try:
                if self._state_id:
                    state = session.query(AgentState).filter_by(id=self._state_id).first()
                else:
                    state = session.query(AgentState).first()

                if not state:
                    return None

                return {
                    "status": state.status,
                    "mode": state.mode,
                    "cycle_count": state.cycle_count,
                    "last_cycle_at": state.last_cycle_at.isoformat() if state.last_cycle_at else None,
                    "last_prediction": state.last_prediction,
                    "last_signal": state.last_signal,
                    "account_equity": state.account_equity,
                    "open_positions": state.open_positions,
                    "circuit_breaker_state": state.circuit_breaker_state,
                    "kill_switch_active": state.kill_switch_active,
                    "error_message": state.error_message,
                    "config": state.config,
                    "started_at": state.started_at.isoformat() if state.started_at else None,
                    "updated_at": state.updated_at.isoformat() if state.updated_at else None,
                }

            finally:
                session.close()

        except SQLAlchemyError as e:
            logger.error(f"Failed to get state: {e}")
            return None

    def _update_state(self, **kwargs) -> bool:
        """Internal method to update state fields.

        Args:
            **kwargs: Field names and values to update

        Returns:
            True if successful, False otherwise
        """
        if not self._state_id:
            logger.error("State not initialized, cannot update")
            return False

        try:
            session = self._session_factory()
            try:
                state = session.query(AgentState).filter_by(id=self._state_id).first()

                if not state:
                    logger.error(f"State record {self._state_id} not found")
                    return False

                # Update provided fields
                for key, value in kwargs.items():
                    if hasattr(state, key):
                        setattr(state, key, value)
                    else:
                        logger.warning(f"Unknown state field: {key}")

                # Always update timestamp
                state.updated_at = datetime.utcnow()

                session.commit()
                return True

            except Exception as e:
                session.rollback()
                raise e
            finally:
                session.close()

        except SQLAlchemyError as e:
            logger.error(f"Failed to update state: {e}")
            return False
