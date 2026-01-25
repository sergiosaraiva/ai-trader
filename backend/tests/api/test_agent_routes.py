"""Tests for agent control endpoints using FastAPI TestClient."""

import pytest
from datetime import datetime
from unittest.mock import Mock
from fastapi.testclient import TestClient
from fastapi import FastAPI


class TestAgentStartEndpoint:
    """Test POST /api/v1/agent/start endpoint."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        self.mock_db = Mock()

    def test_start_agent_successfully_no_existing_state(self):
        """Test starting agent when no existing state exists."""
        from src.api.routes import agent
        from src.api.database.session import get_db

        # Mock no existing agent state
        def mock_get_db():
            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = None  # No existing state

            self.mock_db.query.return_value = mock_query
            self.mock_db.add = Mock()
            self.mock_db.commit = Mock()

            # Mock the created command
            mock_cmd = Mock()
            mock_cmd.id = 1

            def mock_refresh(obj):
                obj.id = 1

            self.mock_db.refresh = mock_refresh

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.post("/api/v1/agent/start", json={
                "mode": "simulation",
                "confidence_threshold": 0.70,
                "cycle_interval_seconds": 60,
                "max_position_size": 0.1,
                "use_kelly_sizing": True,
            })

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "queued"
            assert data["command_id"] == 1
            assert "simulation" in data["message"]
        finally:
            app.dependency_overrides.clear()

    def test_start_agent_when_already_running(self):
        """Test starting agent when already running returns 400."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentState

        def mock_get_db():
            mock_state = Mock(spec=AgentState)
            mock_state.status = "running"

            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = mock_state

            self.mock_db.query.return_value = mock_query

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.post("/api/v1/agent/start", json={})

            assert response.status_code == 400
            assert "already running" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_start_agent_with_custom_configuration(self):
        """Test starting agent with custom configuration."""
        from src.api.routes import agent
        from src.api.database.session import get_db

        def mock_get_db():
            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = None

            self.mock_db.query.return_value = mock_query
            self.mock_db.add = Mock()
            self.mock_db.commit = Mock()

            mock_cmd = Mock()
            mock_cmd.id = 2

            def mock_refresh(obj):
                obj.id = 2

            self.mock_db.refresh = mock_refresh

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.post("/api/v1/agent/start", json={
                "mode": "paper",
                "confidence_threshold": 0.75,
                "cycle_interval_seconds": 120,
                "max_position_size": 0.15,
                "use_kelly_sizing": False,
            })

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "queued"
            assert "paper" in data["message"]
        finally:
            app.dependency_overrides.clear()

    def test_start_agent_with_invalid_configuration(self):
        """Test starting agent with invalid configuration returns 422."""
        from src.api.routes import agent
        from src.api.database.session import get_db

        def mock_get_db():
            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.post("/api/v1/agent/start", json={
                "confidence_threshold": 0.49,  # Invalid: too low
            })

            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()


class TestAgentStopEndpoint:
    """Test POST /api/v1/agent/stop endpoint."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        self.mock_db = Mock()

    def test_stop_running_agent(self):
        """Test stopping a running agent."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentState

        def mock_get_db():
            mock_state = Mock(spec=AgentState)
            mock_state.status = "running"

            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = mock_state

            self.mock_db.query.return_value = mock_query
            self.mock_db.add = Mock()
            self.mock_db.commit = Mock()

            mock_cmd = Mock()
            mock_cmd.id = 10

            def mock_refresh(obj):
                obj.id = 10

            self.mock_db.refresh = mock_refresh

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.post("/api/v1/agent/stop", json={})

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "queued"
            assert data["command_id"] == 10
        finally:
            app.dependency_overrides.clear()

    def test_stop_already_stopped_agent(self):
        """Test stopping an already stopped agent returns 400."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentState

        def mock_get_db():
            mock_state = Mock(spec=AgentState)
            mock_state.status = "stopped"

            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = mock_state

            self.mock_db.query.return_value = mock_query

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.post("/api/v1/agent/stop", json={})

            assert response.status_code == 400
            assert "already stopped" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_stop_with_force_and_close_positions(self):
        """Test stopping agent with force and close_positions flags."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentState

        def mock_get_db():
            mock_state = Mock(spec=AgentState)
            mock_state.status = "running"

            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = mock_state

            self.mock_db.query.return_value = mock_query
            self.mock_db.add = Mock()
            self.mock_db.commit = Mock()

            mock_cmd = Mock()
            mock_cmd.id = 11

            def mock_refresh(obj):
                obj.id = 11

            self.mock_db.refresh = mock_refresh

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.post("/api/v1/agent/stop", json={
                "force": True,
                "close_positions": True,
            })

            assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()


class TestAgentPauseEndpoint:
    """Test POST /api/v1/agent/pause endpoint."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        self.mock_db = Mock()

    def test_pause_running_agent(self):
        """Test pausing a running agent."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentState

        def mock_get_db():
            mock_state = Mock(spec=AgentState)
            mock_state.status = "running"

            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = mock_state

            self.mock_db.query.return_value = mock_query
            self.mock_db.add = Mock()
            self.mock_db.commit = Mock()

            mock_cmd = Mock()
            mock_cmd.id = 20

            def mock_refresh(obj):
                obj.id = 20

            self.mock_db.refresh = mock_refresh

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.post("/api/v1/agent/pause")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "queued"
        finally:
            app.dependency_overrides.clear()

    def test_pause_already_paused_agent(self):
        """Test pausing an already paused agent returns 400."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentState

        def mock_get_db():
            mock_state = Mock(spec=AgentState)
            mock_state.status = "paused"

            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = mock_state

            self.mock_db.query.return_value = mock_query

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.post("/api/v1/agent/pause")

            assert response.status_code == 400
            assert "Cannot pause" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_pause_stopped_agent(self):
        """Test pausing a stopped agent returns 400."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentState

        def mock_get_db():
            mock_state = Mock(spec=AgentState)
            mock_state.status = "stopped"

            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = mock_state

            self.mock_db.query.return_value = mock_query

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.post("/api/v1/agent/pause")

            assert response.status_code == 400
        finally:
            app.dependency_overrides.clear()


class TestAgentResumeEndpoint:
    """Test POST /api/v1/agent/resume endpoint."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        self.mock_db = Mock()

    def test_resume_paused_agent(self):
        """Test resuming a paused agent."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentState

        def mock_get_db():
            mock_state = Mock(spec=AgentState)
            mock_state.status = "paused"

            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = mock_state

            self.mock_db.query.return_value = mock_query
            self.mock_db.add = Mock()
            self.mock_db.commit = Mock()

            mock_cmd = Mock()
            mock_cmd.id = 30

            def mock_refresh(obj):
                obj.id = 30

            self.mock_db.refresh = mock_refresh

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.post("/api/v1/agent/resume")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "queued"
        finally:
            app.dependency_overrides.clear()

    def test_resume_running_agent(self):
        """Test resuming a running agent returns 400."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentState

        def mock_get_db():
            mock_state = Mock(spec=AgentState)
            mock_state.status = "running"

            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = mock_state

            self.mock_db.query.return_value = mock_query

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.post("/api/v1/agent/resume")

            assert response.status_code == 400
            assert "Cannot resume" in response.json()["detail"]
        finally:
            app.dependency_overrides.clear()

    def test_resume_stopped_agent(self):
        """Test resuming a stopped agent returns 400."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentState

        def mock_get_db():
            mock_state = Mock(spec=AgentState)
            mock_state.status = "stopped"

            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = mock_state

            self.mock_db.query.return_value = mock_query

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.post("/api/v1/agent/resume")

            assert response.status_code == 400
        finally:
            app.dependency_overrides.clear()


class TestAgentStatusEndpoint:
    """Test GET /api/v1/agent/status endpoint."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        self.mock_db = Mock()

    def test_get_status_when_agent_running(self):
        """Test getting status when agent is running."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentState

        def mock_get_db():
            mock_state = Mock(spec=AgentState)
            mock_state.status = "running"
            mock_state.mode = "simulation"
            mock_state.cycle_count = 142
            mock_state.last_cycle_at = datetime(2024, 1, 15, 14, 30, 0)
            mock_state.account_equity = 103450.00
            mock_state.open_positions = 1
            mock_state.circuit_breaker_state = None
            mock_state.kill_switch_active = False
            mock_state.error_message = None
            mock_state.started_at = datetime(2024, 1, 15, 12, 0, 0)
            mock_state.last_prediction = {"direction": "long", "confidence": 0.72}
            mock_state.config = {"confidence_threshold": 0.70}

            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = mock_state

            self.mock_db.query.return_value = mock_query

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.get("/api/v1/agent/status")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "running"
            assert data["mode"] == "simulation"
            assert data["cycle_count"] == 142
            assert data["open_positions"] == 1
            assert data["kill_switch_active"] is False
        finally:
            app.dependency_overrides.clear()

    def test_get_status_when_agent_stopped(self):
        """Test getting status when agent is stopped."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentState

        def mock_get_db():
            mock_state = Mock(spec=AgentState)
            mock_state.status = "stopped"
            mock_state.mode = "simulation"
            mock_state.cycle_count = 0
            mock_state.last_cycle_at = None
            mock_state.account_equity = None
            mock_state.open_positions = 0
            mock_state.circuit_breaker_state = None
            mock_state.kill_switch_active = False
            mock_state.error_message = None
            mock_state.started_at = None
            mock_state.last_prediction = None
            mock_state.config = {}

            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = mock_state

            self.mock_db.query.return_value = mock_query

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.get("/api/v1/agent/status")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "stopped"
            assert data["last_cycle_at"] is None
            assert data["uptime_seconds"] is None
        finally:
            app.dependency_overrides.clear()

    def test_get_status_when_no_state_exists(self):
        """Test getting status when no state exists returns 404."""
        from src.api.routes import agent
        from src.api.database.session import get_db

        def mock_get_db():
            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = None

            self.mock_db.query.return_value = mock_query

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.get("/api/v1/agent/status")

            assert response.status_code == 404
        finally:
            app.dependency_overrides.clear()


class TestAgentConfigUpdateEndpoint:
    """Test PUT /api/v1/agent/config endpoint."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        self.mock_db = Mock()

    def test_update_single_field(self):
        """Test updating a single configuration field."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentState

        def mock_get_db():
            mock_state = Mock(spec=AgentState)
            mock_state.status = "running"

            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = mock_state

            self.mock_db.query.return_value = mock_query
            self.mock_db.add = Mock()
            self.mock_db.commit = Mock()

            mock_cmd = Mock()
            mock_cmd.id = 40

            def mock_refresh(obj):
                obj.id = 40

            self.mock_db.refresh = mock_refresh

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.put("/api/v1/agent/config", json={
                "confidence_threshold": 0.75,
            })

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "queued"
            assert "1 fields" in data["message"]
        finally:
            app.dependency_overrides.clear()

    def test_update_multiple_fields(self):
        """Test updating multiple configuration fields."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentState

        def mock_get_db():
            mock_state = Mock(spec=AgentState)
            mock_state.status = "running"

            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = mock_state

            self.mock_db.query.return_value = mock_query
            self.mock_db.add = Mock()
            self.mock_db.commit = Mock()

            mock_cmd = Mock()
            mock_cmd.id = 41

            def mock_refresh(obj):
                obj.id = 41

            self.mock_db.refresh = mock_refresh

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.put("/api/v1/agent/config", json={
                "confidence_threshold": 0.75,
                "cycle_interval_seconds": 120,
            })

            assert response.status_code == 200
            data = response.json()
            assert "2 fields" in data["message"]
        finally:
            app.dependency_overrides.clear()

    def test_invalid_field_values(self):
        """Test updating with invalid field values returns 422."""
        from src.api.routes import agent
        from src.api.database.session import get_db

        def mock_get_db():
            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.put("/api/v1/agent/config", json={
                "confidence_threshold": 0.49,  # Invalid
            })

            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()


class TestKillSwitchEndpoint:
    """Test POST /api/v1/agent/kill-switch endpoint."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        self.mock_db = Mock()

    def test_trigger_kill_switch(self):
        """Test triggering kill switch."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentState

        def mock_get_db():
            mock_state = Mock(spec=AgentState)
            mock_state.kill_switch_active = False

            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = mock_state

            self.mock_db.query.return_value = mock_query
            self.mock_db.add = Mock()
            self.mock_db.commit = Mock()

            mock_cmd = Mock()
            mock_cmd.id = 50

            def mock_refresh(obj):
                obj.id = 50

            self.mock_db.refresh = mock_refresh

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.post("/api/v1/agent/kill-switch", json={
                "action": "trigger",
            })

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "queued"
            assert "halt" in data["message"]
        finally:
            app.dependency_overrides.clear()

    def test_reset_kill_switch(self):
        """Test resetting kill switch."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentState

        def mock_get_db():
            mock_state = Mock(spec=AgentState)
            mock_state.kill_switch_active = True

            mock_query = Mock()
            mock_query.order_by.return_value = mock_query
            mock_query.first.return_value = mock_state

            self.mock_db.query.return_value = mock_query
            self.mock_db.add = Mock()
            self.mock_db.commit = Mock()

            mock_cmd = Mock()
            mock_cmd.id = 51

            def mock_refresh(obj):
                obj.id = 51

            self.mock_db.refresh = mock_refresh

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.post("/api/v1/agent/kill-switch", json={
                "action": "reset",
            })

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "queued"
        finally:
            app.dependency_overrides.clear()

    def test_invalid_action(self):
        """Test invalid action returns 422."""
        from src.api.routes import agent
        from src.api.database.session import get_db

        def mock_get_db():
            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.post("/api/v1/agent/kill-switch", json={
                "action": "invalid",
            })

            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()


class TestAgentMetricsEndpoint:
    """Test GET /api/v1/agent/metrics endpoint."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        self.mock_db = Mock()

    def test_get_metrics_with_no_trades(self):
        """Test getting metrics when no trades exist."""
        from src.api.routes import agent
        from src.api.database.session import get_db

        def mock_get_db():
            # Mock Trade query
            mock_trade_query = Mock()
            mock_trade_query.filter.return_value = mock_trade_query
            mock_trade_query.all.return_value = []

            # Mock CircuitBreakerEvent query
            mock_breaker_query = Mock()
            mock_breaker_query.filter.return_value = mock_breaker_query
            mock_breaker_query.count.return_value = 0

            def mock_query(model):
                from src.api.database.models import Trade, CircuitBreakerEvent
                if model == Trade:
                    return mock_trade_query
                elif model == CircuitBreakerEvent:
                    return mock_breaker_query
                return Mock()

            self.mock_db.query = mock_query

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.get("/api/v1/agent/metrics")

            assert response.status_code == 200
            data = response.json()
            assert data["total_trades"] == 0
            assert data["winning_trades"] == 0
            assert data["losing_trades"] == 0
            assert data["total_pips"] == 0.0
            assert data["circuit_breaker_triggers"] == 0
        finally:
            app.dependency_overrides.clear()

    def test_get_metrics_with_trades(self):
        """Test getting metrics with existing trades."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import Trade

        def mock_get_db():
            # Create mock trades
            trade1 = Mock(spec=Trade)
            trade1.is_winner = True
            trade1.pips = 25.0

            trade2 = Mock(spec=Trade)
            trade2.is_winner = True
            trade2.pips = 30.0

            trade3 = Mock(spec=Trade)
            trade3.is_winner = False
            trade3.pips = -15.0

            # Mock Trade query
            mock_trade_query = Mock()
            mock_trade_query.filter.return_value = mock_trade_query
            mock_trade_query.all.return_value = [trade1, trade2, trade3]

            # Mock CircuitBreakerEvent query
            mock_breaker_query = Mock()
            mock_breaker_query.filter.return_value = mock_breaker_query
            mock_breaker_query.count.return_value = 1

            def mock_query(model):
                from src.api.database.models import Trade, CircuitBreakerEvent
                if model == Trade:
                    return mock_trade_query
                elif model == CircuitBreakerEvent:
                    return mock_breaker_query
                return Mock()

            self.mock_db.query = mock_query

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.get("/api/v1/agent/metrics")

            assert response.status_code == 200
            data = response.json()
            assert data["total_trades"] == 3
            assert data["winning_trades"] == 2
            assert data["losing_trades"] == 1
            assert data["total_pips"] == 40.0
            assert data["circuit_breaker_triggers"] == 1
        finally:
            app.dependency_overrides.clear()

    def test_get_metrics_with_period_filter(self):
        """Test getting metrics with period filter."""
        from src.api.routes import agent
        from src.api.database.session import get_db

        def mock_get_db():
            # Mock Trade query
            mock_trade_query = Mock()
            mock_trade_query.filter.return_value = mock_trade_query
            mock_trade_query.all.return_value = []

            # Mock CircuitBreakerEvent query
            mock_breaker_query = Mock()
            mock_breaker_query.filter.return_value = mock_breaker_query
            mock_breaker_query.count.return_value = 0

            def mock_query(model):
                from src.api.database.models import Trade, CircuitBreakerEvent
                if model == Trade:
                    return mock_trade_query
                elif model == CircuitBreakerEvent:
                    return mock_breaker_query
                return Mock()

            self.mock_db.query = mock_query

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.get("/api/v1/agent/metrics?period=24h")

            assert response.status_code == 200
            data = response.json()
            assert data["period"] == "24h"
        finally:
            app.dependency_overrides.clear()

    def test_invalid_period_format(self):
        """Test invalid period format returns 422."""
        from src.api.routes import agent
        from src.api.database.session import get_db

        def mock_get_db():
            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.get("/api/v1/agent/metrics?period=invalid")

            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()


class TestCommandStatusEndpoint:
    """Test GET /api/v1/agent/commands/{command_id} endpoint."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        self.mock_db = Mock()

    def test_get_existing_command(self):
        """Test getting an existing command."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentCommand

        def mock_get_db():
            mock_cmd = Mock(spec=AgentCommand)
            mock_cmd.id = 42
            mock_cmd.command = "start"
            mock_cmd.status = "completed"
            mock_cmd.created_at = datetime(2024, 1, 15, 14, 0, 0)
            mock_cmd.processed_at = datetime(2024, 1, 15, 14, 0, 2)
            mock_cmd.result = {"agent_status": "running"}
            mock_cmd.error_message = None

            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.first.return_value = mock_cmd

            self.mock_db.query.return_value = mock_query

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.get("/api/v1/agent/commands/42")

            assert response.status_code == 200
            data = response.json()
            assert data["command_id"] == 42
            assert data["command"] == "start"
            assert data["status"] == "completed"
        finally:
            app.dependency_overrides.clear()

    def test_get_non_existent_command(self):
        """Test getting a non-existent command returns 404."""
        from src.api.routes import agent
        from src.api.database.session import get_db

        def mock_get_db():
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.first.return_value = None

            self.mock_db.query.return_value = mock_query

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.get("/api/v1/agent/commands/999")

            assert response.status_code == 404
        finally:
            app.dependency_overrides.clear()


class TestCommandListEndpoint:
    """Test GET /api/v1/agent/commands endpoint."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        self.mock_db = Mock()

    def test_list_commands_with_pagination(self):
        """Test listing commands with pagination."""
        from src.api.routes import agent
        from src.api.database.session import get_db
        from src.api.database.models import AgentCommand

        def mock_get_db():
            mock_cmd1 = Mock(spec=AgentCommand)
            mock_cmd1.id = 42
            mock_cmd1.command = "start"
            mock_cmd1.status = "completed"
            mock_cmd1.created_at = datetime(2024, 1, 15, 14, 0, 0)
            mock_cmd1.processed_at = datetime(2024, 1, 15, 14, 0, 2)

            mock_cmd2 = Mock(spec=AgentCommand)
            mock_cmd2.id = 43
            mock_cmd2.command = "stop"
            mock_cmd2.status = "pending"
            mock_cmd2.created_at = datetime(2024, 1, 15, 14, 5, 0)
            mock_cmd2.processed_at = None

            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.count.return_value = 15
            mock_query.order_by.return_value = mock_query
            mock_query.offset.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.all.return_value = [mock_cmd1, mock_cmd2]

            self.mock_db.query.return_value = mock_query

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.get("/api/v1/agent/commands")

            assert response.status_code == 200
            data = response.json()
            assert data["count"] == 2
            assert data["total"] == 15
            assert len(data["commands"]) == 2
        finally:
            app.dependency_overrides.clear()

    def test_list_commands_with_status_filter(self):
        """Test listing commands with status filter."""
        from src.api.routes import agent
        from src.api.database.session import get_db

        def mock_get_db():
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.count.return_value = 0
            mock_query.order_by.return_value = mock_query
            mock_query.offset.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.all.return_value = []

            self.mock_db.query.return_value = mock_query

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.get("/api/v1/agent/commands?status=pending")

            assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()

    def test_empty_command_list(self):
        """Test listing commands when list is empty."""
        from src.api.routes import agent
        from src.api.database.session import get_db

        def mock_get_db():
            mock_query = Mock()
            mock_query.filter.return_value = mock_query
            mock_query.count.return_value = 0
            mock_query.order_by.return_value = mock_query
            mock_query.offset.return_value = mock_query
            mock_query.limit.return_value = mock_query
            mock_query.all.return_value = []

            self.mock_db.query.return_value = mock_query

            yield self.mock_db

        app = FastAPI()
        app.include_router(agent.router)
        app.dependency_overrides[get_db] = mock_get_db
        client = TestClient(app)

        try:
            response = client.get("/api/v1/agent/commands")

            assert response.status_code == 200
            data = response.json()
            assert data["count"] == 0
            assert data["total"] == 0
            assert len(data["commands"]) == 0
        finally:
            app.dependency_overrides.clear()
