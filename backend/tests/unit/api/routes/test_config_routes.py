"""Unit tests for configuration API routes.

Tests API endpoints in isolation with mocked dependencies.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
from src.api.main import app
from src.api.routes.config import ConfigUpdateRequest


client = TestClient(app)


# ============================================================================
# REQUEST VALIDATION TESTS
# ============================================================================


def test_update_request_validation_valid():
    """Test ConfigUpdateRequest validation with valid data."""
    request_data = {
        "category": "trading",
        "updates": {"confidence_threshold": 0.75},
        "updated_by": "admin",
        "reason": "Test update",
    }

    request = ConfigUpdateRequest(**request_data)

    assert request.category == "trading"
    assert request.updates == {"confidence_threshold": 0.75}
    assert request.updated_by == "admin"
    assert request.reason == "Test update"


def test_update_request_validation_invalid_category():
    """Test that invalid category is rejected by Pydantic."""
    with pytest.raises(ValueError, match="Invalid category"):
        ConfigUpdateRequest(
            category="invalid_category",
            updates={"some_key": "value"},
        )


def test_update_request_validation_missing_updates():
    """Test that missing updates field is rejected."""
    with pytest.raises(ValueError):
        ConfigUpdateRequest(category="trading")


def test_update_request_validation_optional_fields():
    """Test that optional fields can be omitted."""
    request = ConfigUpdateRequest(
        category="trading",
        updates={"confidence_threshold": 0.75},
    )

    assert request.updated_by is None
    assert request.reason is None


def test_update_request_all_valid_categories():
    """Test all valid category values."""
    valid_categories = ["trading", "model", "risk", "system"]

    for category in valid_categories:
        request = ConfigUpdateRequest(
            category=category,
            updates={"test_key": "test_value"},
        )
        assert request.category == category


# ============================================================================
# GET ALL CONFIG ENDPOINT
# ============================================================================


@patch("src.api.routes.config.trading_config")
def test_get_all_config_success(mock_config):
    """Test GET /api/v1/config endpoint success."""
    mock_config.get_all.return_value = {
        "trading": {"confidence_threshold": 0.66},
        "model": {"weight_1h": 0.6},
        "risk": {"max_drawdown_percent": 15.0},
        "system": {"cache_ttl_seconds": 60},
        "metadata": {"initialized": True},
    }

    response = client.get("/api/v1/config")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "data" in data


@patch("src.api.routes.config.trading_config")
def test_get_all_config_exception(mock_config):
    """Test GET /api/v1/config handles exceptions."""
    mock_config.get_all.side_effect = RuntimeError("Config error")

    response = client.get("/api/v1/config")

    assert response.status_code == 500
    assert "Failed to retrieve configuration" in response.json()["detail"]


# ============================================================================
# GET CATEGORY ENDPOINT
# ============================================================================


@patch("src.api.routes.config.trading_config")
def test_get_category_success(mock_config):
    """Test GET /api/v1/config/category/{category} success."""
    mock_config.get_category.return_value = {
        "confidence_threshold": 0.66,
        "default_lot_size": 0.1,
    }

    response = client.get("/api/v1/config/category/trading")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "trading" in data["data"]


@patch("src.api.routes.config.trading_config")
def test_get_category_invalid(mock_config):
    """Test GET category with invalid category returns 400."""
    mock_config.get_category.side_effect = ValueError("Invalid category")

    response = client.get("/api/v1/config/category/invalid")

    assert response.status_code == 400


@patch("src.api.routes.config.trading_config")
def test_get_category_exception(mock_config):
    """Test GET category handles exceptions."""
    mock_config.get_category.side_effect = RuntimeError("Unexpected error")

    response = client.get("/api/v1/config/category/trading")

    assert response.status_code == 500


# ============================================================================
# UPDATE CONFIG ENDPOINT
# ============================================================================


@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_update_config_success(mock_config, mock_get_session):
    """Test PUT /api/v1/config success."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    mock_config.update.return_value = {
        "status": "success",
        "category": "trading",
        "updated": ["confidence_threshold"],
        "timestamp": "2025-01-26T10:00:00",
    }

    request_data = {
        "category": "trading",
        "updates": {"confidence_threshold": 0.75},
        "updated_by": "admin",
    }

    response = client.put("/api/v1/config", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "timestamp" in data

    # Verify update was called with correct parameters
    mock_config.update.assert_called_once_with(
        category="trading",
        updates={"confidence_threshold": 0.75},
        updated_by="admin",
        reason=None,
        db_session=mock_db,
    )


@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_update_config_validation_error(mock_config, mock_get_session):
    """Test PUT /api/v1/config with validation error."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    mock_config.update.side_effect = ValueError("Validation failed: Invalid value")

    request_data = {
        "category": "trading",
        "updates": {"confidence_threshold": 1.5},
        "updated_by": "admin",
    }

    response = client.put("/api/v1/config", json=request_data)

    assert response.status_code == 400
    assert "Validation failed" in response.json()["detail"]


@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_update_config_server_error(mock_config, mock_get_session):
    """Test PUT /api/v1/config with server error."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    mock_config.update.side_effect = RuntimeError("Database connection lost")

    request_data = {
        "category": "trading",
        "updates": {"confidence_threshold": 0.75},
        "updated_by": "admin",
    }

    response = client.put("/api/v1/config", json=request_data)

    assert response.status_code == 500
    assert "Configuration update failed" in response.json()["detail"]


def test_update_config_invalid_request():
    """Test PUT /api/v1/config with invalid request body."""
    # Missing required fields
    response = client.put("/api/v1/config", json={"category": "trading"})

    assert response.status_code == 422  # Pydantic validation error


# ============================================================================
# RELOAD CONFIG ENDPOINT
# ============================================================================


@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_reload_config_success(mock_config, mock_get_session):
    """Test POST /api/v1/config/reload success."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    mock_config.reload.return_value = {
        "status": "success",
        "timestamp": "2025-01-26T10:00:00",
        "message": "Configuration reloaded",
    }

    response = client.post("/api/v1/config/reload")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "timestamp" in data

    mock_config.reload.assert_called_once_with(db_session=mock_db)


@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_reload_config_error(mock_config, mock_get_session):
    """Test POST /api/v1/config/reload with error."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    mock_config.reload.return_value = {
        "status": "error",
        "message": "Reload failed",
    }

    response = client.post("/api/v1/config/reload")

    assert response.status_code == 500
    assert "Reload failed" in response.json()["detail"]


@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_reload_config_exception(mock_config, mock_get_session):
    """Test POST /api/v1/config/reload handles exceptions."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    mock_config.reload.side_effect = RuntimeError("Unexpected error")

    response = client.post("/api/v1/config/reload")

    assert response.status_code == 500
    assert "Configuration reload failed" in response.json()["detail"]


# ============================================================================
# LIST SETTINGS ENDPOINT
# ============================================================================


@patch("src.api.routes.config.get_session")
def test_list_settings_success(mock_get_session):
    """Test GET /api/v1/config/settings success."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    # Mock settings
    mock_setting1 = Mock()
    mock_setting1.id = 1
    mock_setting1.category = "trading"
    mock_setting1.key = "confidence_threshold"
    mock_setting1.value = 0.75
    mock_setting1.value_type = "float"
    mock_setting1.description = None
    mock_setting1.version = 1
    mock_setting1.updated_by = "admin"
    mock_setting1.updated_at.isoformat.return_value = "2025-01-26T10:00:00"
    mock_setting1.created_at.isoformat.return_value = "2025-01-26T09:00:00"

    mock_db.query.return_value.order_by.return_value.all.return_value = [mock_setting1]

    response = client.get("/api/v1/config/settings")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["count"] == 1
    assert len(data["settings"]) == 1


@patch("src.api.routes.config.get_session")
def test_list_settings_empty(mock_get_session):
    """Test GET /api/v1/config/settings with no settings."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    mock_db.query.return_value.order_by.return_value.all.return_value = []

    response = client.get("/api/v1/config/settings")

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["settings"] == []


@patch("src.api.routes.config.get_session")
def test_list_settings_exception(mock_get_session):
    """Test GET /api/v1/config/settings handles exceptions."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    mock_db.query.side_effect = RuntimeError("Database error")

    response = client.get("/api/v1/config/settings")

    assert response.status_code == 500
    assert "Failed to list settings" in response.json()["detail"]


# ============================================================================
# HISTORY ENDPOINT
# ============================================================================


@patch("src.api.routes.config.get_session")
def test_get_history_success(mock_get_session):
    """Test GET /api/v1/config/history success."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    # Mock history record
    mock_history = Mock()
    mock_history.id = 1
    mock_history.category = "trading"
    mock_history.key = "confidence_threshold"
    mock_history.old_value = 0.66
    mock_history.new_value = 0.75
    mock_history.version = 2
    mock_history.changed_by = "admin"
    mock_history.changed_at.isoformat.return_value = "2025-01-26T10:00:00"
    mock_history.reason = "Performance improvement"

    query_mock = mock_db.query.return_value
    query_mock.order_by.return_value.limit.return_value.all.return_value = [mock_history]

    response = client.get("/api/v1/config/history")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["count"] == 1
    assert len(data["history"]) == 1


@patch("src.api.routes.config.get_session")
def test_get_history_with_filters(mock_get_session):
    """Test GET /api/v1/config/history with filters."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    query_mock = mock_db.query.return_value
    query_mock.order_by.return_value.filter.return_value.limit.return_value.all.return_value = []

    response = client.get("/api/v1/config/history?category=trading&limit=10")

    assert response.status_code == 200


@patch("src.api.routes.config.get_session")
def test_get_history_key_without_category(mock_get_session):
    """Test GET /api/v1/config/history with key but no category returns 400."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    response = client.get("/api/v1/config/history?key=confidence_threshold")

    assert response.status_code == 400
    assert "category required" in response.json()["detail"]


@patch("src.api.routes.config.get_session")
def test_get_history_exception(mock_get_session):
    """Test GET /api/v1/config/history handles exceptions."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    mock_db.query.side_effect = RuntimeError("Database error")

    response = client.get("/api/v1/config/history")

    assert response.status_code == 500
    assert "Failed to retrieve history" in response.json()["detail"]


# ============================================================================
# RESET KEY ENDPOINT
# ============================================================================


@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_reset_key_success(mock_config, mock_get_session):
    """Test POST /api/v1/config/reset/{category}/{key} success."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    mock_config.reset_to_defaults.return_value = {
        "status": "success",
        "message": "Reset trading.confidence_threshold to default",
        "value": 0.66,
    }

    response = client.post("/api/v1/config/reset/trading/confidence_threshold")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    mock_config.reset_to_defaults.assert_called_once_with(
        category="trading",
        key="confidence_threshold",
        db_session=mock_db,
    )


@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_reset_key_invalid(mock_config, mock_get_session):
    """Test POST reset with invalid key returns 400."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    mock_config.reset_to_defaults.side_effect = ValueError("Unknown parameter")

    response = client.post("/api/v1/config/reset/trading/invalid_key")

    assert response.status_code == 400


@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_reset_key_exception(mock_config, mock_get_session):
    """Test POST reset handles exceptions."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    mock_config.reset_to_defaults.side_effect = RuntimeError("Database error")

    response = client.post("/api/v1/config/reset/trading/confidence_threshold")

    assert response.status_code == 500
    assert "Failed to reset configuration" in response.json()["detail"]


# ============================================================================
# RESET CATEGORY ENDPOINT
# ============================================================================


@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_reset_category_success(mock_config, mock_get_session):
    """Test POST /api/v1/config/reset/{category} success."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    mock_config.reset_to_defaults.return_value = {
        "status": "success",
        "message": "Reset trading to defaults",
    }

    response = client.post("/api/v1/config/reset/trading")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"

    mock_config.reset_to_defaults.assert_called_once_with(
        category="trading",
        db_session=mock_db,
    )


@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_reset_category_invalid(mock_config, mock_get_session):
    """Test POST reset category with invalid category returns 400."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    mock_config.reset_to_defaults.side_effect = ValueError("Invalid category")

    response = client.post("/api/v1/config/reset/invalid_category")

    assert response.status_code == 400


# ============================================================================
# VALIDATE ENDPOINT
# ============================================================================


@patch("src.api.routes.config.trading_config")
def test_validate_config_valid(mock_config):
    """Test GET /api/v1/config/validate with valid config."""
    mock_config.validate.return_value = []

    response = client.get("/api/v1/config/validate")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "valid"
    assert data["valid"] is True


@patch("src.api.routes.config.trading_config")
def test_validate_config_invalid(mock_config):
    """Test GET /api/v1/config/validate with invalid config."""
    mock_config.validate.return_value = [
        "confidence_threshold must be between 0.0 and 1.0",
        "default_lot_size must be positive",
    ]

    response = client.get("/api/v1/config/validate")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "invalid"
    assert data["valid"] is False
    assert len(data["errors"]) == 2


@patch("src.api.routes.config.trading_config")
def test_validate_config_exception(mock_config):
    """Test GET /api/v1/config/validate handles exceptions."""
    mock_config.validate.side_effect = RuntimeError("Validation error")

    response = client.get("/api/v1/config/validate")

    assert response.status_code == 500
    assert "Validation failed" in response.json()["detail"]


# ============================================================================
# AUTHENTICATION TESTS
# ============================================================================


@patch.dict(os.environ, {"ADMIN_API_KEY": "test-secret-key"})
@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_history_endpoint_rejects_missing_auth(mock_config, mock_get_session):
    """Test that /history endpoint rejects request without auth header."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    # Need to reload dependencies module to pick up new env var
    from importlib import reload
    import src.api.dependencies
    reload(src.api.dependencies)

    response = client.get("/api/v1/config/history")

    assert response.status_code == 401
    assert "Invalid or missing X-Admin-Key" in response.json()["detail"]


@patch.dict(os.environ, {"ADMIN_API_KEY": "test-secret-key"})
@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_history_endpoint_accepts_valid_auth(mock_config, mock_get_session):
    """Test that /history endpoint accepts request with valid auth header."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    # Mock history data
    query_mock = mock_db.query.return_value
    query_mock.order_by.return_value.limit.return_value.all.return_value = []

    # Need to reload dependencies module to pick up new env var
    from importlib import reload
    import src.api.dependencies
    reload(src.api.dependencies)

    response = client.get(
        "/api/v1/config/history",
        headers={"X-Admin-Key": "test-secret-key"}
    )

    assert response.status_code == 200


@patch.dict(os.environ, {"ADMIN_API_KEY": "test-secret-key"})
@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_history_endpoint_rejects_invalid_auth(mock_config, mock_get_session):
    """Test that /history endpoint rejects request with invalid auth header."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    # Need to reload dependencies module to pick up new env var
    from importlib import reload
    import src.api.dependencies
    reload(src.api.dependencies)

    response = client.get(
        "/api/v1/config/history",
        headers={"X-Admin-Key": "wrong-key"}
    )

    assert response.status_code == 401
    assert "Invalid or missing X-Admin-Key" in response.json()["detail"]


@patch.dict(os.environ, {}, clear=True)
@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_history_endpoint_works_without_auth_in_dev_mode(mock_config, mock_get_session):
    """Test that /history endpoint works when ADMIN_API_KEY not set (development mode)."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    # Mock history data
    query_mock = mock_db.query.return_value
    query_mock.order_by.return_value.limit.return_value.all.return_value = []

    # Need to reload dependencies module to clear ADMIN_API_KEY
    from importlib import reload
    import src.api.dependencies
    reload(src.api.dependencies)

    response = client.get("/api/v1/config/history")

    # Should work without auth when ADMIN_API_KEY not set
    assert response.status_code == 200


@patch.dict(os.environ, {"ADMIN_API_KEY": "test-secret-key"})
@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_update_endpoint_requires_auth(mock_config, mock_get_session):
    """Test that /config PUT endpoint requires authentication."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    # Need to reload dependencies module to pick up new env var
    from importlib import reload
    import src.api.dependencies
    reload(src.api.dependencies)

    request_data = {
        "category": "trading",
        "updates": {"confidence_threshold": 0.75},
        "updated_by": "admin",
    }

    # Without auth header
    response = client.put("/api/v1/config", json=request_data)
    assert response.status_code == 401

    # With valid auth header
    mock_config.update.return_value = {
        "status": "success",
        "category": "trading",
        "updated": ["confidence_threshold"],
        "timestamp": "2025-01-26T10:00:00",
    }

    response = client.put(
        "/api/v1/config",
        json=request_data,
        headers={"X-Admin-Key": "test-secret-key"}
    )
    assert response.status_code == 200


@patch.dict(os.environ, {"ADMIN_API_KEY": "test-secret-key"})
@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_reload_endpoint_requires_auth(mock_config, mock_get_session):
    """Test that /config/reload endpoint requires authentication."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    # Need to reload dependencies module to pick up new env var
    from importlib import reload
    import src.api.dependencies
    reload(src.api.dependencies)

    # Without auth header
    response = client.post("/api/v1/config/reload")
    assert response.status_code == 401

    # With valid auth header
    mock_config.reload.return_value = {
        "status": "success",
        "timestamp": "2025-01-26T10:00:00",
        "message": "Configuration reloaded",
    }

    response = client.post(
        "/api/v1/config/reload",
        headers={"X-Admin-Key": "test-secret-key"}
    )
    assert response.status_code == 200


@patch.dict(os.environ, {"ADMIN_API_KEY": "test-secret-key"})
@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_reset_endpoints_require_auth(mock_config, mock_get_session):
    """Test that reset endpoints require authentication."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    # Need to reload dependencies module to pick up new env var
    from importlib import reload
    import src.api.dependencies
    reload(src.api.dependencies)

    # Test reset key without auth
    response = client.post("/api/v1/config/reset/trading/confidence_threshold")
    assert response.status_code == 401

    # Test reset category without auth
    response = client.post("/api/v1/config/reset/trading")
    assert response.status_code == 401

    # With valid auth
    mock_config.reset_to_defaults.return_value = {
        "status": "success",
        "message": "Reset successful",
    }

    response = client.post(
        "/api/v1/config/reset/trading/confidence_threshold",
        headers={"X-Admin-Key": "test-secret-key"}
    )
    assert response.status_code == 200


def test_auth_status_endpoint():
    """Test /config/auth-status endpoint returns auth status."""
    # Need to reload dependencies module
    from importlib import reload
    import src.api.dependencies
    reload(src.api.dependencies)

    response = client.get("/api/v1/config/auth-status")

    assert response.status_code == 200
    data = response.json()
    assert "admin_auth_enabled" in data
    assert isinstance(data["admin_auth_enabled"], bool)


# ============================================================================
# ERROR HANDLING EDGE CASES
# ============================================================================

import os


def test_malformed_json_request():
    """Test that malformed JSON returns proper error."""
    response = client.put(
        "/api/v1/config",
        data="not valid json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422


def test_missing_content_type():
    """Test request without content-type."""
    response = client.put("/api/v1/config", data='{"category": "trading"}')

    # FastAPI should handle this gracefully
    assert response.status_code in [422, 415]


@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_update_with_empty_updates(mock_config, mock_get_session):
    """Test update with empty updates dict."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    mock_config.update.return_value = {
        "status": "success",
        "category": "trading",
        "updated": [],
        "timestamp": "2025-01-26T10:00:00",
    }

    request_data = {
        "category": "trading",
        "updates": {},  # Empty updates
        "updated_by": "admin",
    }

    response = client.put("/api/v1/config", json=request_data)

    # Should still succeed (even if no-op)
    assert response.status_code == 200


# ============================================================================
# RESPONSE SCHEMA VALIDATION
# ============================================================================


@patch("src.api.routes.config.trading_config")
def test_get_all_response_schema(mock_config):
    """Test that response matches expected schema."""
    mock_config.get_all.return_value = {
        "trading": {"confidence_threshold": 0.66},
        "model": {},
        "risk": {},
        "system": {},
        "metadata": {},
    }

    response = client.get("/api/v1/config")

    assert response.status_code == 200
    data = response.json()

    # Verify required fields
    assert "status" in data
    assert "data" in data
    assert isinstance(data["data"], dict)


@patch("src.api.routes.config.get_session")
@patch("src.api.routes.config.trading_config")
def test_update_response_schema(mock_config, mock_get_session):
    """Test update response schema."""
    mock_db = Mock()
    mock_get_session.return_value = mock_db

    mock_config.update.return_value = {
        "status": "success",
        "category": "trading",
        "updated": ["confidence_threshold"],
        "timestamp": "2025-01-26T10:00:00",
    }

    response = client.put("/api/v1/config", json={
        "category": "trading",
        "updates": {"confidence_threshold": 0.75},
    })

    assert response.status_code == 200
    data = response.json()

    # Verify required fields
    assert "status" in data
    assert "data" in data
    assert "timestamp" in data
