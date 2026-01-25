"""Tests for agent control Pydantic schemas validation."""

import pytest
from pydantic import ValidationError
from src.api.schemas.agent import (
    AgentStartRequest,
    AgentStopRequest,
    AgentConfigUpdateRequest,
    KillSwitchRequest,
    CommandResponse,
    AgentStatusResponse,
    AgentMetricsResponse,
    CommandStatusResponse,
    CommandListItem,
    CommandListResponse,
)


class TestAgentStartRequest:
    """Test AgentStartRequest schema validation."""

    # Valid Schema Tests

    def test_valid_request_with_defaults(self):
        """Test creating valid request with default values."""
        request = AgentStartRequest()

        assert request.mode == "simulation"
        assert request.confidence_threshold == 0.70
        assert request.cycle_interval_seconds == 60
        assert request.max_position_size == 0.1
        assert request.use_kelly_sizing is True

    def test_valid_request_with_all_fields(self):
        """Test creating valid request with all fields specified."""
        request = AgentStartRequest(
            mode="paper",
            confidence_threshold=0.75,
            cycle_interval_seconds=120,
            max_position_size=0.15,
            use_kelly_sizing=False,
        )

        assert request.mode == "paper"
        assert request.confidence_threshold == 0.75
        assert request.cycle_interval_seconds == 120
        assert request.max_position_size == 0.15
        assert request.use_kelly_sizing is False

    def test_valid_live_mode(self):
        """Test creating valid request with live mode."""
        request = AgentStartRequest(mode="live")
        assert request.mode == "live"

    # Invalid Mode Tests

    def test_invalid_mode_raises_error(self):
        """Test invalid mode raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AgentStartRequest(mode="invalid")

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("mode",) for e in errors)

    def test_empty_mode_raises_error(self):
        """Test empty string mode raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AgentStartRequest(mode="")

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("mode",) for e in errors)

    # Confidence Threshold Validation

    def test_confidence_threshold_too_low_raises_error(self):
        """Test confidence_threshold < 0.5 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AgentStartRequest(confidence_threshold=0.49)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("confidence_threshold",) for e in errors)

    def test_confidence_threshold_too_high_raises_error(self):
        """Test confidence_threshold > 0.95 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AgentStartRequest(confidence_threshold=0.96)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("confidence_threshold",) for e in errors)

    def test_confidence_threshold_boundary_values(self):
        """Test confidence_threshold boundary values (0.5 and 0.95) are valid."""
        # Test 0.5
        request_min = AgentStartRequest(confidence_threshold=0.5)
        assert request_min.confidence_threshold == 0.5

        # Test 0.95
        request_max = AgentStartRequest(confidence_threshold=0.95)
        assert request_max.confidence_threshold == 0.95

    # Cycle Interval Validation

    def test_cycle_interval_too_low_raises_error(self):
        """Test cycle_interval_seconds < 10 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AgentStartRequest(cycle_interval_seconds=9)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("cycle_interval_seconds",) for e in errors)

    def test_cycle_interval_too_high_raises_error(self):
        """Test cycle_interval_seconds > 3600 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AgentStartRequest(cycle_interval_seconds=3601)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("cycle_interval_seconds",) for e in errors)

    def test_cycle_interval_boundary_values(self):
        """Test cycle_interval_seconds boundary values (10 and 3600) are valid."""
        # Test 10
        request_min = AgentStartRequest(cycle_interval_seconds=10)
        assert request_min.cycle_interval_seconds == 10

        # Test 3600
        request_max = AgentStartRequest(cycle_interval_seconds=3600)
        assert request_max.cycle_interval_seconds == 3600

    # Max Position Size Validation

    def test_max_position_size_zero_raises_error(self):
        """Test max_position_size = 0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AgentStartRequest(max_position_size=0.0)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("max_position_size",) for e in errors)

    def test_max_position_size_negative_raises_error(self):
        """Test negative max_position_size raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AgentStartRequest(max_position_size=-0.1)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("max_position_size",) for e in errors)

    def test_max_position_size_too_high_raises_error(self):
        """Test max_position_size > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AgentStartRequest(max_position_size=1.1)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("max_position_size",) for e in errors)

    def test_max_position_size_boundary_values(self):
        """Test max_position_size boundary values are valid."""
        # Test very small positive value
        request_min = AgentStartRequest(max_position_size=0.01)
        assert request_min.max_position_size == 0.01

        # Test 1.0
        request_max = AgentStartRequest(max_position_size=1.0)
        assert request_max.max_position_size == 1.0


class TestAgentStopRequest:
    """Test AgentStopRequest schema validation."""

    def test_valid_with_defaults(self):
        """Test creating valid request with default values."""
        request = AgentStopRequest()

        assert request.force is False
        assert request.close_positions is False

    def test_valid_with_force_true(self):
        """Test creating valid request with force=True."""
        request = AgentStopRequest(force=True)

        assert request.force is True
        assert request.close_positions is False

    def test_valid_with_close_positions_true(self):
        """Test creating valid request with close_positions=True."""
        request = AgentStopRequest(close_positions=True)

        assert request.force is False
        assert request.close_positions is True

    def test_valid_with_both_true(self):
        """Test creating valid request with both flags True."""
        request = AgentStopRequest(force=True, close_positions=True)

        assert request.force is True
        assert request.close_positions is True


class TestAgentConfigUpdateRequest:
    """Test AgentConfigUpdateRequest schema validation."""

    def test_valid_with_partial_fields(self):
        """Test creating valid request with partial fields."""
        request = AgentConfigUpdateRequest(confidence_threshold=0.75)

        assert request.confidence_threshold == 0.75
        assert request.cycle_interval_seconds is None
        assert request.max_position_size is None
        assert request.use_kelly_sizing is None

    def test_valid_with_multiple_fields(self):
        """Test creating valid request with multiple fields."""
        request = AgentConfigUpdateRequest(
            confidence_threshold=0.75,
            cycle_interval_seconds=120,
        )

        assert request.confidence_threshold == 0.75
        assert request.cycle_interval_seconds == 120

    def test_valid_with_all_fields(self):
        """Test creating valid request with all fields."""
        request = AgentConfigUpdateRequest(
            confidence_threshold=0.75,
            cycle_interval_seconds=120,
            max_position_size=0.15,
            use_kelly_sizing=False,
        )

        assert request.confidence_threshold == 0.75
        assert request.cycle_interval_seconds == 120
        assert request.max_position_size == 0.15
        assert request.use_kelly_sizing is False

    def test_empty_update_all_none(self):
        """Test creating request with all None (empty update)."""
        request = AgentConfigUpdateRequest()

        assert request.confidence_threshold is None
        assert request.cycle_interval_seconds is None
        assert request.max_position_size is None
        assert request.use_kelly_sizing is None

    def test_invalid_confidence_threshold(self):
        """Test invalid confidence_threshold raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AgentConfigUpdateRequest(confidence_threshold=0.49)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("confidence_threshold",) for e in errors)


class TestKillSwitchRequest:
    """Test KillSwitchRequest schema validation."""

    def test_valid_trigger_action(self):
        """Test creating valid trigger request."""
        request = KillSwitchRequest(action="trigger")

        assert request.action == "trigger"
        assert request.reason is None

    def test_valid_reset_action(self):
        """Test creating valid reset request."""
        request = KillSwitchRequest(action="reset")

        assert request.action == "reset"
        assert request.reason is None

    def test_trigger_with_reason(self):
        """Test trigger action with reason."""
        request = KillSwitchRequest(
            action="trigger",
            reason="Unexpected market volatility"
        )

        assert request.action == "trigger"
        assert request.reason == "Unexpected market volatility"

    def test_invalid_action_raises_error(self):
        """Test invalid action raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            KillSwitchRequest(action="invalid")

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("action",) for e in errors)

    def test_missing_action_raises_error(self):
        """Test missing action raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            KillSwitchRequest()

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("action",) for e in errors)

    def test_reason_too_long_raises_error(self):
        """Test reason > 500 characters raises ValidationError."""
        long_reason = "A" * 501

        with pytest.raises(ValidationError) as exc_info:
            KillSwitchRequest(action="trigger", reason=long_reason)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("reason",) for e in errors)

    def test_reason_boundary_length(self):
        """Test reason at exactly 500 characters is valid."""
        boundary_reason = "A" * 500

        request = KillSwitchRequest(action="trigger", reason=boundary_reason)
        assert len(request.reason) == 500


class TestCommandResponse:
    """Test CommandResponse schema validation."""

    def test_valid_queued_response(self):
        """Test creating valid queued response."""
        response = CommandResponse(
            status="queued",
            command_id=42,
            message="Start command queued successfully",
        )

        assert response.status == "queued"
        assert response.command_id == 42
        assert response.message == "Start command queued successfully"

    def test_valid_error_response(self):
        """Test creating valid error response."""
        response = CommandResponse(
            status="error",
            command_id=43,
            message="Failed to queue command",
        )

        assert response.status == "error"
        assert response.command_id == 43

    def test_missing_required_fields_raise_error(self):
        """Test missing required fields raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CommandResponse(status="queued")

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("command_id",) for e in errors)
        assert any(e["loc"] == ("message",) for e in errors)


class TestAgentStatusResponse:
    """Test AgentStatusResponse schema validation."""

    def test_valid_status_response(self):
        """Test creating valid status response with all fields."""
        response = AgentStatusResponse(
            status="running",
            mode="simulation",
            cycle_count=142,
            last_cycle_at="2024-01-15T14:30:00Z",
            account_equity=103450.00,
            open_positions=1,
            circuit_breaker_state=None,
            kill_switch_active=False,
            error_message=None,
            uptime_seconds=8520.0,
            last_prediction={"direction": "long", "confidence": 0.72},
            config={"confidence_threshold": 0.70},
        )

        assert response.status == "running"
        assert response.mode == "simulation"
        assert response.cycle_count == 142
        assert response.account_equity == 103450.00
        assert response.open_positions == 1
        assert response.kill_switch_active is False

    def test_valid_stopped_status(self):
        """Test creating valid response with stopped status."""
        response = AgentStatusResponse(
            status="stopped",
            mode="simulation",
            cycle_count=0,
            last_cycle_at=None,
            account_equity=None,
            open_positions=0,
            circuit_breaker_state=None,
            kill_switch_active=False,
            error_message=None,
            uptime_seconds=None,
            last_prediction=None,
            config={},
        )

        assert response.status == "stopped"
        assert response.last_cycle_at is None
        assert response.uptime_seconds is None

    def test_negative_cycle_count_raises_error(self):
        """Test negative cycle_count raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AgentStatusResponse(
                status="running",
                mode="simulation",
                cycle_count=-1,
                open_positions=0,
                kill_switch_active=False,
                config={},
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("cycle_count",) for e in errors)


class TestAgentMetricsResponse:
    """Test AgentMetricsResponse schema validation."""

    def test_valid_metrics_response(self):
        """Test creating valid metrics response."""
        response = AgentMetricsResponse(
            total_trades=47,
            winning_trades=29,
            losing_trades=18,
            win_rate=0.617,
            total_pips=892.5,
            profit_factor=2.45,
            sharpe_ratio=3.2,
            max_drawdown=-125.0,
            circuit_breaker_triggers=2,
            period="all",
        )

        assert response.total_trades == 47
        assert response.winning_trades == 29
        assert response.losing_trades == 18
        assert response.win_rate == 0.617
        assert response.total_pips == 892.5
        assert response.profit_factor == 2.45
        assert response.circuit_breaker_triggers == 2

    def test_valid_metrics_with_no_trades(self):
        """Test creating valid metrics response with no trades."""
        response = AgentMetricsResponse(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=None,
            total_pips=0.0,
            profit_factor=None,
            sharpe_ratio=None,
            max_drawdown=None,
            circuit_breaker_triggers=0,
            period="all",
        )

        assert response.total_trades == 0
        assert response.win_rate is None
        assert response.profit_factor is None

    def test_win_rate_too_high_raises_error(self):
        """Test win_rate > 1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AgentMetricsResponse(
                total_trades=10,
                winning_trades=10,
                losing_trades=0,
                win_rate=1.1,
                total_pips=100.0,
                circuit_breaker_triggers=0,
                period="all",
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("win_rate",) for e in errors)

    def test_negative_trades_raises_error(self):
        """Test negative total_trades raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            AgentMetricsResponse(
                total_trades=-1,
                winning_trades=0,
                losing_trades=0,
                total_pips=0.0,
                circuit_breaker_triggers=0,
                period="all",
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("total_trades",) for e in errors)


class TestCommandStatusResponse:
    """Test CommandStatusResponse schema validation."""

    def test_valid_completed_command(self):
        """Test creating valid completed command status."""
        response = CommandStatusResponse(
            command_id=42,
            command="start",
            status="completed",
            created_at="2024-01-15T14:00:00Z",
            processed_at="2024-01-15T14:00:02Z",
            result={"agent_status": "running", "pid": 12345},
            error_message=None,
        )

        assert response.command_id == 42
        assert response.command == "start"
        assert response.status == "completed"
        assert response.result == {"agent_status": "running", "pid": 12345}

    def test_valid_failed_command(self):
        """Test creating valid failed command status."""
        response = CommandStatusResponse(
            command_id=43,
            command="stop",
            status="failed",
            created_at="2024-01-15T14:00:00Z",
            processed_at="2024-01-15T14:00:02Z",
            result=None,
            error_message="Agent not running",
        )

        assert response.status == "failed"
        assert response.error_message == "Agent not running"


class TestCommandListResponse:
    """Test CommandListResponse schema validation."""

    def test_valid_command_list(self):
        """Test creating valid command list response."""
        items = [
            CommandListItem(
                command_id=42,
                command="start",
                status="completed",
                created_at="2024-01-15T14:00:00Z",
                processed_at="2024-01-15T14:00:02Z",
            ),
            CommandListItem(
                command_id=43,
                command="stop",
                status="pending",
                created_at="2024-01-15T14:05:00Z",
                processed_at=None,
            ),
        ]

        response = CommandListResponse(
            commands=items,
            count=2,
            total=15,
        )

        assert len(response.commands) == 2
        assert response.count == 2
        assert response.total == 15

    def test_empty_command_list(self):
        """Test creating valid empty command list."""
        response = CommandListResponse(
            commands=[],
            count=0,
            total=0,
        )

        assert len(response.commands) == 0
        assert response.count == 0
        assert response.total == 0

    def test_negative_count_raises_error(self):
        """Test negative count raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CommandListResponse(
                commands=[],
                count=-1,
                total=0,
            )

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("count",) for e in errors)
