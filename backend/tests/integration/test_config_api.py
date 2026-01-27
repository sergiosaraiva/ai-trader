"""Integration tests for configuration API endpoints.

Tests full API workflow including database persistence, hot reload, and error handling.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.api.main import app
from src.api.database.models import Base, ConfigurationSetting, ConfigurationHistory
from src.config.trading_config import TradingConfig


# Test database setup
TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture
def test_db():
    """Create test database."""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def reset_config():
    """Reset config singleton to defaults before each test."""
    config = TradingConfig()
    # Reset to defaults
    config.trading.confidence_threshold = 0.66
    config.trading.default_lot_size = 0.1
    config.model.weight_1h = 0.6
    config.model.weight_4h = 0.3
    config.model.weight_daily = 0.1
    config.risk.max_drawdown_percent = 15.0
    config.system.cache_ttl_seconds = 60
    yield config


# ============================================================================
# GET ENDPOINTS
# ============================================================================


def test_get_all_config(client, reset_config):
    """Test GET /api/v1/config endpoint."""
    response = client.get("/api/v1/config")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "data" in data

    config_data = data["data"]
    assert "trading" in config_data
    assert "model" in config_data
    assert "risk" in config_data
    assert "system" in config_data
    assert "metadata" in config_data

    # Verify default values
    assert config_data["trading"]["confidence_threshold"] == 0.66
    assert config_data["model"]["weight_1h"] == 0.6
    assert config_data["risk"]["max_drawdown_percent"] == 15.0


def test_get_category_config(client, reset_config):
    """Test GET /api/v1/config/category/{category} endpoint."""
    response = client.get("/api/v1/config/category/trading")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "trading" in data["data"]

    trading = data["data"]["trading"]
    assert "confidence_threshold" in trading
    assert "default_lot_size" in trading
    assert "pip_value" in trading


def test_get_invalid_category(client, reset_config):
    """Test GET with invalid category returns 400."""
    response = client.get("/api/v1/config/category/invalid")

    assert response.status_code == 400


def test_get_all_categories(client, reset_config):
    """Test getting all valid categories."""
    categories = ["trading", "model", "risk", "system"]

    for category in categories:
        response = client.get(f"/api/v1/config/category/{category}")
        assert response.status_code == 200
        data = response.json()
        assert category in data["data"]


# ============================================================================
# UPDATE ENDPOINT
# ============================================================================


@patch("src.api.routes.config.get_session")
def test_update_config(mock_get_session, client, test_db, reset_config):
    """Test PUT /api/v1/config endpoint."""
    mock_get_session.return_value = test_db

    update_request = {
        "category": "trading",
        "updates": {
            "confidence_threshold": 0.75,
            "default_lot_size": 0.2,
        },
        "updated_by": "test_user",
        "reason": "Testing update",
    }

    response = client.put("/api/v1/config", json=update_request)

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "timestamp" in data

    # Verify config was updated
    config = TradingConfig()
    assert config.trading.confidence_threshold == 0.75
    assert config.trading.default_lot_size == 0.2


@patch("src.api.routes.config.get_session")
def test_update_invalid_value(mock_get_session, client, test_db, reset_config):
    """Test update with invalid value returns 400."""
    mock_get_session.return_value = test_db

    update_request = {
        "category": "trading",
        "updates": {
            "confidence_threshold": 1.5,  # Invalid
        },
        "updated_by": "test_user",
    }

    response = client.put("/api/v1/config", json=update_request)

    assert response.status_code == 400

    # Config should remain unchanged
    config = TradingConfig()
    assert config.trading.confidence_threshold == 0.66


@patch("src.api.routes.config.get_session")
def test_update_invalid_category(mock_get_session, client, test_db, reset_config):
    """Test update with invalid category returns 400."""
    mock_get_session.return_value = test_db

    update_request = {
        "category": "invalid_category",
        "updates": {"some_key": "some_value"},
        "updated_by": "test_user",
    }

    response = client.put("/api/v1/config", json=update_request)

    assert response.status_code == 422  # Pydantic validation error


@patch("src.api.routes.config.get_session")
def test_update_multiple_parameters(mock_get_session, client, test_db, reset_config):
    """Test updating multiple parameters at once."""
    mock_get_session.return_value = test_db

    update_request = {
        "category": "trading",
        "updates": {
            "confidence_threshold": 0.70,
            "default_lot_size": 0.15,
            "default_tp_pips": 30.0,
            "default_sl_pips": 18.0,
        },
        "updated_by": "test_user",
    }

    response = client.put("/api/v1/config", json=update_request)

    assert response.status_code == 200

    # Verify all parameters were updated
    config = TradingConfig()
    assert config.trading.confidence_threshold == 0.70
    assert config.trading.default_lot_size == 0.15
    assert config.trading.default_tp_pips == 30.0
    assert config.trading.default_sl_pips == 18.0


# ============================================================================
# RELOAD ENDPOINT
# ============================================================================


@patch("src.api.routes.config.get_session")
def test_reload_config(mock_get_session, client, test_db, reset_config):
    """Test POST /api/v1/config/reload endpoint."""
    # First, add some settings to database
    setting = ConfigurationSetting(
        category="trading",
        key="confidence_threshold",
        value=0.80,
        value_type="float",
        version=1,
        updated_by="test_setup",
    )
    test_db.add(setting)
    test_db.commit()

    mock_get_session.return_value = test_db

    # Reload config
    response = client.post("/api/v1/config/reload")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "timestamp" in data

    # Verify config was reloaded from DB
    config = TradingConfig()
    assert config.trading.confidence_threshold == 0.80


@patch("src.api.routes.config.get_session")
def test_reload_with_multiple_settings(mock_get_session, client, test_db, reset_config):
    """Test reload with multiple database settings."""
    # Add multiple settings
    settings = [
        ConfigurationSetting(
            category="trading",
            key="confidence_threshold",
            value=0.75,
            value_type="float",
            version=1,
        ),
        ConfigurationSetting(
            category="model",
            key="agreement_bonus",
            value=0.08,
            value_type="float",
            version=1,
        ),
        ConfigurationSetting(
            category="risk",
            key="max_consecutive_losses",
            value=7,
            value_type="int",
            version=1,
        ),
    ]

    for setting in settings:
        test_db.add(setting)
    test_db.commit()

    mock_get_session.return_value = test_db

    # Reload
    response = client.post("/api/v1/config/reload")

    assert response.status_code == 200

    # Verify all settings were loaded
    config = TradingConfig()
    assert config.trading.confidence_threshold == 0.75
    assert config.model.agreement_bonus == 0.08
    assert config.risk.max_consecutive_losses == 7


# ============================================================================
# SETTINGS ENDPOINT
# ============================================================================


@patch("src.api.routes.config.get_session")
def test_list_all_settings(mock_get_session, client, test_db):
    """Test GET /api/v1/config/settings endpoint."""
    # Add some settings to database
    settings = [
        ConfigurationSetting(
            category="trading",
            key="confidence_threshold",
            value=0.70,
            value_type="float",
            version=1,
            updated_by="test",
        ),
        ConfigurationSetting(
            category="model",
            key="weight_1h",
            value=0.65,
            value_type="float",
            version=2,
            updated_by="admin",
        ),
    ]

    for setting in settings:
        test_db.add(setting)
    test_db.commit()

    mock_get_session.return_value = test_db

    response = client.get("/api/v1/config/settings")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["count"] == 2
    assert "settings" in data

    settings_list = data["settings"]
    assert len(settings_list) == 2

    # Verify setting details
    assert any(s["key"] == "confidence_threshold" for s in settings_list)
    assert any(s["key"] == "weight_1h" for s in settings_list)


@patch("src.api.routes.config.get_session")
def test_list_settings_empty_database(mock_get_session, client, test_db):
    """Test listing settings with empty database."""
    mock_get_session.return_value = test_db

    response = client.get("/api/v1/config/settings")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["count"] == 0
    assert data["settings"] == []


# ============================================================================
# HISTORY ENDPOINT
# ============================================================================


@patch("src.api.routes.config.get_session")
def test_get_config_history(mock_get_session, client, test_db):
    """Test GET /api/v1/config/history endpoint."""
    # Add setting and history
    setting = ConfigurationSetting(
        category="trading",
        key="confidence_threshold",
        value=0.75,
        value_type="float",
        version=2,
    )
    test_db.add(setting)
    test_db.flush()

    history1 = ConfigurationHistory(
        setting_id=setting.id,
        category="trading",
        key="confidence_threshold",
        old_value=None,
        new_value=0.66,
        version=1,
        changed_by="system",
    )
    history2 = ConfigurationHistory(
        setting_id=setting.id,
        category="trading",
        key="confidence_threshold",
        old_value=0.66,
        new_value=0.75,
        version=2,
        changed_by="admin",
        reason="Performance improvement",
    )

    test_db.add(history1)
    test_db.add(history2)
    test_db.commit()

    mock_get_session.return_value = test_db

    response = client.get("/api/v1/config/history")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["count"] == 2
    assert "history" in data

    history = data["history"]
    assert len(history) == 2


@patch("src.api.routes.config.get_session")
def test_get_config_history_filtered_by_category(mock_get_session, client, test_db):
    """Test history endpoint with category filter."""
    # Add multiple history records
    setting1 = ConfigurationSetting(
        category="trading",
        key="confidence_threshold",
        value=0.75,
        value_type="float",
        version=1,
    )
    setting2 = ConfigurationSetting(
        category="model",
        key="agreement_bonus",
        value=0.08,
        value_type="float",
        version=1,
    )
    test_db.add(setting1)
    test_db.add(setting2)
    test_db.flush()

    history1 = ConfigurationHistory(
        setting_id=setting1.id,
        category="trading",
        key="confidence_threshold",
        old_value=None,
        new_value=0.75,
        version=1,
    )
    history2 = ConfigurationHistory(
        setting_id=setting2.id,
        category="model",
        key="agreement_bonus",
        old_value=None,
        new_value=0.08,
        version=1,
    )

    test_db.add(history1)
    test_db.add(history2)
    test_db.commit()

    mock_get_session.return_value = test_db

    # Filter by trading category
    response = client.get("/api/v1/config/history?category=trading")

    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 1
    assert data["history"][0]["category"] == "trading"


@patch("src.api.routes.config.get_session")
def test_get_config_history_with_limit(mock_get_session, client, test_db):
    """Test history endpoint with limit parameter."""
    # Add many history records
    setting = ConfigurationSetting(
        category="trading",
        key="confidence_threshold",
        value=0.80,
        value_type="float",
        version=5,
    )
    test_db.add(setting)
    test_db.flush()

    for i in range(10):
        history = ConfigurationHistory(
            setting_id=setting.id,
            category="trading",
            key="confidence_threshold",
            old_value=0.66 + i * 0.01,
            new_value=0.67 + i * 0.01,
            version=i + 1,
        )
        test_db.add(history)

    test_db.commit()

    mock_get_session.return_value = test_db

    # Request with limit
    response = client.get("/api/v1/config/history?limit=5")

    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 5
    assert len(data["history"]) == 5


# ============================================================================
# RESET ENDPOINTS
# ============================================================================


@patch("src.api.routes.config.get_session")
def test_reset_config_key(mock_get_session, client, test_db, reset_config):
    """Test POST /api/v1/config/reset/{category}/{key} endpoint."""
    # First update config
    config = TradingConfig()
    config.trading.confidence_threshold = 0.80

    mock_get_session.return_value = test_db

    # Reset to default
    response = client.post("/api/v1/config/reset/trading/confidence_threshold")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"

    # Verify reset to default
    assert config.trading.confidence_threshold == 0.66


@patch("src.api.routes.config.get_session")
def test_reset_invalid_key(mock_get_session, client, test_db):
    """Test reset with invalid key returns 400."""
    mock_get_session.return_value = test_db

    response = client.post("/api/v1/config/reset/trading/invalid_key")

    assert response.status_code == 400


@patch("src.api.routes.config.get_session")
def test_reset_category(mock_get_session, client, test_db, reset_config):
    """Test POST /api/v1/config/reset/{category} endpoint."""
    # Update multiple values
    config = TradingConfig()
    config.trading.confidence_threshold = 0.80
    config.trading.default_lot_size = 0.5
    config.trading.default_tp_pips = 50.0

    mock_get_session.return_value = test_db

    # Reset entire category
    response = client.post("/api/v1/config/reset/trading")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"

    # Verify all trading parameters reset
    assert config.trading.confidence_threshold == 0.66
    assert config.trading.default_lot_size == 0.1
    assert config.trading.default_tp_pips == 25.0


# ============================================================================
# VALIDATE ENDPOINT
# ============================================================================


def test_validate_config_valid(client, reset_config):
    """Test GET /api/v1/config/validate with valid config."""
    response = client.get("/api/v1/config/validate")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "valid"
    assert data["valid"] is True


def test_validate_config_invalid(client, reset_config):
    """Test validate endpoint with invalid config."""
    # Make config invalid
    config = TradingConfig()
    config.trading.confidence_threshold = 1.5

    response = client.get("/api/v1/config/validate")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "invalid"
    assert data["valid"] is False
    assert "errors" in data
    assert len(data["errors"]) > 0


# ============================================================================
# DATABASE PERSISTENCE TESTS
# ============================================================================


@patch("src.api.routes.config.get_session")
def test_update_persists_to_database(mock_get_session, client, test_db, reset_config):
    """Test that updates are persisted to database."""
    mock_get_session.return_value = test_db

    update_request = {
        "category": "trading",
        "updates": {"confidence_threshold": 0.75},
        "updated_by": "test_user",
        "reason": "Test persistence",
    }

    response = client.put("/api/v1/config", json=update_request)
    assert response.status_code == 200

    # Verify setting in database
    setting = test_db.query(ConfigurationSetting).filter_by(
        category="trading",
        key="confidence_threshold"
    ).first()

    assert setting is not None
    assert setting.value == 0.75
    assert setting.updated_by == "test_user"

    # Verify history in database
    history = test_db.query(ConfigurationHistory).filter_by(
        category="trading",
        key="confidence_threshold"
    ).first()

    assert history is not None
    assert history.new_value == 0.75
    assert history.changed_by == "test_user"
    assert history.reason == "Test persistence"


@patch("src.api.routes.config.get_session")
def test_multiple_updates_create_history(mock_get_session, client, test_db, reset_config):
    """Test that multiple updates create proper history trail."""
    mock_get_session.return_value = test_db

    # Update 1
    client.put("/api/v1/config", json={
        "category": "trading",
        "updates": {"confidence_threshold": 0.70},
        "updated_by": "user1",
    })

    # Update 2
    client.put("/api/v1/config", json={
        "category": "trading",
        "updates": {"confidence_threshold": 0.75},
        "updated_by": "user2",
    })

    # Update 3
    client.put("/api/v1/config", json={
        "category": "trading",
        "updates": {"confidence_threshold": 0.80},
        "updated_by": "user3",
    })

    # Verify history
    history_records = test_db.query(ConfigurationHistory).filter_by(
        category="trading",
        key="confidence_threshold"
    ).order_by(ConfigurationHistory.version).all()

    assert len(history_records) == 3
    assert history_records[0].new_value == 0.70
    assert history_records[1].new_value == 0.75
    assert history_records[2].new_value == 0.80

    # Verify version increments
    assert history_records[0].version == 1
    assert history_records[1].version == 2
    assert history_records[2].version == 3
