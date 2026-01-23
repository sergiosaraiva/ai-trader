"""Unit tests for StateManager."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import from conftest
from .conftest import AgentConfig, StateManager, Base, AgentState


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
        initial_capital=100000.0,
    )


@pytest.fixture
def state_manager(db_session):
    """Create a StateManager instance with test database."""
    return StateManager(db_session)


class TestStateManagerInitialization:
    """Test StateManager initialization."""

    def test_initialize_creates_new_state(self, state_manager, config, db_session):
        """Test initialize creates new state when none exists."""
        # Arrange & Act
        result = state_manager.initialize(config)

        # Assert
        assert result is True
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state is not None
            assert state.status == "stopped"
            assert state.mode == "simulation"
            assert state.cycle_count == 0
            assert state.open_positions == 0
            assert state.kill_switch_active is False
        finally:
            session.close()

    def test_initialize_loads_existing_state(self, state_manager, config, db_session):
        """Test initialize loads existing state instead of creating new."""
        # Arrange - create existing state
        session = db_session()
        try:
            existing_state = AgentState(
                status="running",
                mode="paper",
                cycle_count=42,
                open_positions=2,
                kill_switch_active=False,
                config=config.to_dict(),
                updated_at=datetime.utcnow(),
            )
            session.add(existing_state)
            session.commit()
            state_id = existing_state.id
        finally:
            session.close()

        # Act
        result = state_manager.initialize(config)

        # Assert
        assert result is True
        assert state_manager._state_id == state_id

        # Verify no new state was created
        session = db_session()
        try:
            count = session.query(AgentState).count()
            assert count == 1

            state = session.query(AgentState).first()
            assert state.cycle_count == 42
            assert state.status == "running"
        finally:
            session.close()

    def test_initialize_stores_config(self, state_manager, config, db_session):
        """Test initialize stores configuration in state."""
        # Arrange & Act
        state_manager.initialize(config)

        # Assert
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state.config is not None
            assert state.config["mode"] == "simulation"
            assert state.config["confidence_threshold"] == 0.70
        finally:
            session.close()


class TestStateManagerStatusUpdates:
    """Test StateManager status update methods."""

    def test_update_status_changes_status(self, state_manager, config, db_session):
        """Test update_status changes agent status."""
        # Arrange
        state_manager.initialize(config)

        # Act
        result = state_manager.update_status("running")

        # Assert
        assert result is True
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state.status == "running"
        finally:
            session.close()

    def test_update_status_with_error_message(self, state_manager, config, db_session):
        """Test update_status stores error message."""
        # Arrange
        state_manager.initialize(config)

        # Act
        result = state_manager.update_status("error", error_message="Connection failed")

        # Assert
        assert result is True
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state.status == "error"
            assert state.error_message == "Connection failed"
        finally:
            session.close()

    def test_update_status_returns_false_when_not_initialized(self, state_manager):
        """Test update_status returns False when state not initialized."""
        # Arrange - don't initialize

        # Act
        result = state_manager.update_status("running")

        # Assert
        assert result is False

    def test_set_started_updates_status_and_timestamp(self, state_manager, config, db_session):
        """Test set_started updates status to running and sets started_at."""
        # Arrange
        state_manager.initialize(config)

        # Act
        before_start = datetime.utcnow()
        result = state_manager.set_started()

        # Assert
        assert result is True
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state.status == "running"
            assert state.started_at is not None
            assert state.started_at >= before_start
            assert state.error_message is None
        finally:
            session.close()

    def test_set_stopped_updates_status_and_clears_timestamp(self, state_manager, config, db_session):
        """Test set_stopped updates status to stopped and clears started_at."""
        # Arrange
        state_manager.initialize(config)
        state_manager.set_started()

        # Act
        result = state_manager.set_stopped()

        # Assert
        assert result is True
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state.status == "stopped"
            assert state.started_at is None
            assert state.error_message is None
        finally:
            session.close()


class TestStateManagerCycleUpdates:
    """Test StateManager cycle update methods."""

    def test_update_cycle_increments_count(self, state_manager, config, db_session):
        """Test update_cycle updates cycle count."""
        # Arrange
        state_manager.initialize(config)

        # Act
        result = state_manager.update_cycle(
            cycle_count=1,
            last_prediction=None,
            last_signal=None,
            account_equity=100000.0,
            open_positions=0,
        )

        # Assert
        assert result is True
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state.cycle_count == 1
            assert state.account_equity == 100000.0
            assert state.open_positions == 0
        finally:
            session.close()

    def test_update_cycle_stores_prediction_and_signal(self, state_manager, config, db_session):
        """Test update_cycle stores last prediction and signal."""
        # Arrange
        state_manager.initialize(config)
        prediction = {"signal": "BUY", "confidence": 0.75}
        signal = {"action": "open_long", "size": 0.1}

        # Act
        result = state_manager.update_cycle(
            cycle_count=1,
            last_prediction=prediction,
            last_signal=signal,
        )

        # Assert
        assert result is True
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state.last_prediction == prediction
            assert state.last_signal == signal
        finally:
            session.close()

    def test_update_cycle_sets_last_cycle_timestamp(self, state_manager, config, db_session):
        """Test update_cycle sets last_cycle_at timestamp."""
        # Arrange
        state_manager.initialize(config)

        # Act
        before_update = datetime.utcnow()
        result = state_manager.update_cycle(cycle_count=1)

        # Assert
        assert result is True
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state.last_cycle_at is not None
            assert state.last_cycle_at >= before_update
        finally:
            session.close()


class TestStateManagerCircuitBreaker:
    """Test StateManager circuit breaker methods."""

    def test_update_circuit_breaker_state(self, state_manager, config, db_session):
        """Test update_circuit_breaker updates circuit breaker state."""
        # Arrange
        state_manager.initialize(config)

        # Act
        result = state_manager.update_circuit_breaker(
            circuit_breaker_state="triggered",
            kill_switch_active=False,
        )

        # Assert
        assert result is True
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state.circuit_breaker_state == "triggered"
            assert state.kill_switch_active is False
        finally:
            session.close()

    def test_update_circuit_breaker_activates_kill_switch(self, state_manager, config, db_session):
        """Test update_circuit_breaker can activate kill switch."""
        # Arrange
        state_manager.initialize(config)

        # Act
        result = state_manager.update_circuit_breaker(
            circuit_breaker_state="kill_switch",
            kill_switch_active=True,
        )

        # Assert
        assert result is True
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state.kill_switch_active is True
            assert state.circuit_breaker_state == "kill_switch"
        finally:
            session.close()


class TestStateManagerConfigUpdate:
    """Test StateManager config update methods."""

    def test_update_config_stores_new_config(self, state_manager, config, db_session):
        """Test update_config stores new configuration."""
        # Arrange
        state_manager.initialize(config)
        new_config = AgentConfig(
            mode="paper",
            confidence_threshold=0.75,
        )

        # Act
        result = state_manager.update_config(new_config)

        # Assert
        assert result is True
        session = db_session()
        try:
            state = session.query(AgentState).first()
            assert state.mode == "paper"
            assert state.config["confidence_threshold"] == 0.75
        finally:
            session.close()


class TestStateManagerGetState:
    """Test StateManager get_state method."""

    def test_get_state_returns_current_state(self, state_manager, config, db_session):
        """Test get_state returns current state dictionary."""
        # Arrange
        state_manager.initialize(config)
        state_manager.set_started()
        state_manager.update_cycle(
            cycle_count=5,
            account_equity=102000.0,
            open_positions=2,
        )

        # Act
        state = state_manager.get_state()

        # Assert
        assert state is not None
        assert state["status"] == "running"
        assert state["mode"] == "simulation"
        assert state["cycle_count"] == 5
        assert state["account_equity"] == 102000.0
        assert state["open_positions"] == 2
        assert state["kill_switch_active"] is False

    def test_get_state_returns_none_when_not_initialized(self, db_session):
        """Test get_state returns None when no state exists."""
        # Arrange
        state_manager = StateManager(db_session)

        # Act
        state = state_manager.get_state()

        # Assert
        assert state is None

    def test_get_state_includes_timestamps(self, state_manager, config, db_session):
        """Test get_state includes timestamp fields in ISO format."""
        # Arrange
        state_manager.initialize(config)
        state_manager.set_started()

        # Act
        state = state_manager.get_state()

        # Assert
        assert state is not None
        assert "started_at" in state
        assert "updated_at" in state
        assert state["started_at"] is not None
        assert state["updated_at"] is not None


class TestStateManagerUpdatedAt:
    """Test StateManager automatic updated_at tracking."""

    def test_update_always_refreshes_updated_at(self, state_manager, config, db_session):
        """Test that any update refreshes updated_at timestamp."""
        # Arrange
        state_manager.initialize(config)
        session = db_session()
        try:
            initial_updated_at = session.query(AgentState).first().updated_at
        finally:
            session.close()

        import time
        time.sleep(0.01)

        # Act
        state_manager.update_status("running")

        # Assert
        session = db_session()
        try:
            new_updated_at = session.query(AgentState).first().updated_at
            assert new_updated_at > initial_updated_at
        finally:
            session.close()


class TestStateManagerErrorHandling:
    """Test StateManager error handling."""

    def test_initialize_returns_false_on_database_error(self, config):
        """Test initialize returns False on database error."""
        # Arrange
        def raise_error():
            raise Exception("Database connection failed")

        state_manager = StateManager(raise_error)

        # Act
        result = state_manager.initialize(config)

        # Assert
        assert result is False

    def test_update_status_returns_false_on_database_error(self, config):
        """Test update_status returns False on database error."""
        # Arrange
        def raise_error():
            raise Exception("Database error")

        state_manager = StateManager(raise_error)
        state_manager._state_id = 1  # Fake initialization

        # Act
        result = state_manager.update_status("running")

        # Assert
        assert result is False
