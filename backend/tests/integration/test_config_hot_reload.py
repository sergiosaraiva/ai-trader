"""Integration tests for configuration hot-reload functionality.

Tests hot-reload without service restart, including callback triggering,
cache invalidation, and service coordination.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import time

# Add backend src to path
backend_path = Path(__file__).parent.parent.parent
src_path = backend_path / "src"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import directly from module to avoid __init__.py dependencies
import importlib.util

# Load models module directly
models_spec = importlib.util.spec_from_file_location(
    "models",
    src_path / "api" / "database" / "models.py"
)
models_module = importlib.util.module_from_spec(models_spec)
models_spec.loader.exec_module(models_module)

Base = models_module.Base
ConfigurationSetting = models_module.ConfigurationSetting

# Load trading_config module directly
config_spec = importlib.util.spec_from_file_location(
    "trading_config",
    src_path / "config" / "trading_config.py"
)
trading_config_module = importlib.util.module_from_spec(config_spec)
config_spec.loader.exec_module(trading_config_module)

TradingConfig = trading_config_module.TradingConfig

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
    config._config_version = 0
    config._callbacks = {
        "trading": [],
        "model": [],
        "risk": [],
        "system": [],
        "timeframes": [],
        "agent": [],
        "cache": [],
        "scheduler": [],
        "features": [],
    }
    yield config


# ============================================================================
# HOT RELOAD BASIC TESTS
# ============================================================================


def test_hot_reload_updates_memory(test_db, reset_config):
    """Test that hot reload updates in-memory configuration."""
    config = reset_config

    # Add settings to database
    setting = ConfigurationSetting(
        category="trading",
        key="confidence_threshold",
        value=0.80,
        value_type="float",
        version=1,
    )
    test_db.add(setting)
    test_db.commit()

    # Reload
    result = config.reload(db_session=test_db)

    assert result["status"] == "success"
    assert config.trading.confidence_threshold == 0.80


def test_hot_reload_multiple_settings(test_db, reset_config):
    """Test hot reload with multiple database settings."""
    config = reset_config

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
            category="trading",
            key="default_lot_size",
            value=0.2,
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

    # Reload
    result = config.reload(db_session=test_db)

    assert result["status"] == "success"
    assert config.trading.confidence_threshold == 0.75
    assert config.trading.default_lot_size == 0.2
    assert config.model.agreement_bonus == 0.08
    assert config.risk.max_consecutive_losses == 7


def test_hot_reload_without_db_session(reset_config):
    """Test hot reload without database session returns error."""
    config = reset_config

    result = config.reload(db_session=None)

    assert result["status"] == "error"
    assert "Database session required" in result["message"]


def test_hot_reload_with_invalid_data(test_db, reset_config):
    """Test hot reload with invalid data handles gracefully."""
    config = reset_config

    # Add invalid setting (confidence > 1.0)
    setting = ConfigurationSetting(
        category="trading",
        key="confidence_threshold",
        value=1.5,  # Invalid
        value_type="float",
        version=1,
    )
    test_db.add(setting)
    test_db.commit()

    # Reload should fail validation
    result = config.reload(db_session=test_db)

    assert result["status"] == "error"
    assert "Validation failed" in result["message"]

    # Config should remain unchanged
    assert config.trading.confidence_threshold == 0.66


# ============================================================================
# CONFIG VERSION TESTS
# ============================================================================


def test_hot_reload_increments_config_version(test_db, reset_config):
    """Test that hot reload increments config version."""
    config = reset_config
    initial_version = config.get_config_version()

    # Add setting to database
    setting = ConfigurationSetting(
        category="trading",
        key="confidence_threshold",
        value=0.75,
        value_type="float",
        version=1,
    )
    test_db.add(setting)
    test_db.commit()

    # Reload
    config.reload(db_session=test_db)

    # Version should increment
    new_version = config.get_config_version()
    assert new_version == initial_version + 1


def test_config_version_increments_on_update(reset_config):
    """Test that config version increments on update."""
    config = reset_config
    initial_version = config.get_config_version()

    # Update config
    config.update(
        category="trading",
        updates={"confidence_threshold": 0.75},
        updated_by="test",
        db_session=None,
    )

    # Version should increment
    new_version = config.get_config_version()
    assert new_version == initial_version + 1


def test_multiple_updates_increment_version(reset_config):
    """Test that multiple updates increment version each time."""
    config = reset_config
    initial_version = config.get_config_version()

    # Multiple updates
    for i in range(5):
        config.update(
            category="trading",
            updates={"confidence_threshold": 0.60 + i * 0.01},
            updated_by="test",
            db_session=None,
        )

    # Version should increment by 5
    final_version = config.get_config_version()
    assert final_version == initial_version + 5


# ============================================================================
# CALLBACK TRIGGERING TESTS
# ============================================================================


def test_hot_reload_triggers_callbacks(test_db, reset_config):
    """Test that hot reload triggers all callbacks."""
    config = reset_config

    callback_counts = {
        "trading": 0,
        "model": 0,
        "risk": 0,
        "system": 0,
    }

    def create_callback(category):
        def callback(params):
            callback_counts[category] += 1
        return callback

    # Register callbacks
    for category in callback_counts.keys():
        config.register_callback(category, create_callback(category))

    # Add settings to database
    settings = [
        ConfigurationSetting(category="trading", key="confidence_threshold", value=0.75, value_type="float", version=1),
        ConfigurationSetting(category="model", key="agreement_bonus", value=0.08, value_type="float", version=1),
        ConfigurationSetting(category="risk", key="max_consecutive_losses", value=7, value_type="int", version=1),
    ]

    for setting in settings:
        test_db.add(setting)
    test_db.commit()

    # Reload (should trigger all callbacks, even for unchanged categories)
    config.reload(db_session=test_db)

    # All callbacks should have been triggered
    assert callback_counts["trading"] == 1
    assert callback_counts["model"] == 1
    assert callback_counts["risk"] == 1
    assert callback_counts["system"] == 1  # Triggered even though no DB setting


def test_hot_reload_callback_receives_updated_params(test_db, reset_config):
    """Test that callbacks receive updated parameter objects."""
    config = reset_config

    received_params = []

    def callback(params):
        received_params.append(params.confidence_threshold)

    config.register_callback("trading", callback)

    # Add setting to database
    setting = ConfigurationSetting(
        category="trading",
        key="confidence_threshold",
        value=0.85,
        value_type="float",
        version=1,
    )
    test_db.add(setting)
    test_db.commit()

    # Reload
    config.reload(db_session=test_db)

    # Callback should receive updated value
    assert len(received_params) == 1
    assert received_params[0] == 0.85


def test_hot_reload_callback_failure_doesnt_break_reload(test_db, reset_config):
    """Test that callback failure doesn't break hot reload."""
    config = reset_config

    def failing_callback(params):
        raise RuntimeError("Callback failed!")

    config.register_callback("trading", failing_callback)

    # Add setting to database
    setting = ConfigurationSetting(
        category="trading",
        key="confidence_threshold",
        value=0.80,
        value_type="float",
        version=1,
    )
    test_db.add(setting)
    test_db.commit()

    # Reload should succeed despite callback failure
    result = config.reload(db_session=test_db)

    assert result["status"] == "success"
    assert config.trading.confidence_threshold == 0.80


# ============================================================================
# CACHE INVALIDATION TESTS
# ============================================================================


def test_cache_invalidation_on_hot_reload(test_db, reset_config):
    """Test that cache is invalidated on hot reload."""
    config = reset_config

    # Track cache invalidation via config version
    version_before = config.get_config_version()

    # Add setting
    setting = ConfigurationSetting(
        category="trading",
        key="confidence_threshold",
        value=0.75,
        value_type="float",
        version=1,
    )
    test_db.add(setting)
    test_db.commit()

    # Reload
    config.reload(db_session=test_db)

    # Version should change (indicating caches should invalidate)
    version_after = config.get_config_version()
    assert version_after > version_before


def test_cache_invalidation_callback_integration(test_db, reset_config):
    """Test that services can react to cache invalidation."""
    config = reset_config

    cache_cleared = [False]
    cache_version = [config.get_config_version()]

    def cache_invalidation_callback(params):
        """Simulates a service clearing its cache on config change."""
        new_version = config.get_config_version()
        if new_version != cache_version[0]:
            cache_cleared[0] = True
            cache_version[0] = new_version

    config.register_callback("trading", cache_invalidation_callback)

    # Add setting
    setting = ConfigurationSetting(
        category="trading",
        key="confidence_threshold",
        value=0.80,
        value_type="float",
        version=1,
    )
    test_db.add(setting)
    test_db.commit()

    # Reload
    config.reload(db_session=test_db)

    # Cache should have been cleared
    assert cache_cleared[0] is True


# ============================================================================
# SERVICE COORDINATION TESTS
# ============================================================================


def test_multiple_services_react_to_hot_reload(test_db, reset_config):
    """Test that multiple services can coordinate via callbacks."""
    config = reset_config

    service_states = {
        "model_service": {"reloaded": False, "version": 0},
        "trading_service": {"reloaded": False, "version": 0},
        "risk_service": {"reloaded": False, "version": 0},
    }

    def create_service_callback(service_name, category):
        def callback(params):
            service_states[service_name]["reloaded"] = True
            service_states[service_name]["version"] = config.get_config_version()
        return callback

    # Register service callbacks
    config.register_callback("model", create_service_callback("model_service", "model"))
    config.register_callback("trading", create_service_callback("trading_service", "trading"))
    config.register_callback("risk", create_service_callback("risk_service", "risk"))

    # Add settings
    settings = [
        ConfigurationSetting(category="trading", key="confidence_threshold", value=0.75, value_type="float", version=1),
        ConfigurationSetting(category="model", key="agreement_bonus", value=0.08, value_type="float", version=1),
        ConfigurationSetting(category="risk", key="max_consecutive_losses", value=7, value_type="int", version=1),
    ]

    for setting in settings:
        test_db.add(setting)
    test_db.commit()

    # Reload
    config.reload(db_session=test_db)

    # All services should have reloaded
    assert service_states["model_service"]["reloaded"] is True
    assert service_states["trading_service"]["reloaded"] is True
    assert service_states["risk_service"]["reloaded"] is True

    # All should have same config version
    config_version = config.get_config_version()
    assert service_states["model_service"]["version"] == config_version
    assert service_states["trading_service"]["version"] == config_version
    assert service_states["risk_service"]["version"] == config_version


def test_hot_reload_timestamp_tracking(test_db, reset_config):
    """Test that hot reload tracks last reload timestamp."""
    config = reset_config

    # Add setting
    setting = ConfigurationSetting(
        category="trading",
        key="confidence_threshold",
        value=0.75,
        value_type="float",
        version=1,
    )
    test_db.add(setting)
    test_db.commit()

    # Reload
    result = config.reload(db_session=test_db)

    # Should include timestamp
    assert "timestamp" in result
    assert result["timestamp"] is not None


# ============================================================================
# RELOAD VALIDATION TESTS
# ============================================================================


def test_hot_reload_rejects_invalid_config(test_db, reset_config):
    """Test that hot reload rejects invalid configuration."""
    config = reset_config

    # Add multiple settings that together are invalid (weights don't sum to 1)
    settings = [
        ConfigurationSetting(category="model", key="weight_1h", value=0.8, value_type="float", version=1),
        ConfigurationSetting(category="model", key="weight_4h", value=0.8, value_type="float", version=1),
        ConfigurationSetting(category="model", key="weight_daily", value=0.8, value_type="float", version=1),
    ]

    for setting in settings:
        test_db.add(setting)
    test_db.commit()

    # Reload should fail validation
    result = config.reload(db_session=test_db)

    assert result["status"] == "error"


def test_hot_reload_partial_update_validation(test_db, reset_config):
    """Test hot reload with partial updates validates entire config."""
    config = reset_config

    # Set up config with one invalid setting in DB
    # (valid alone, but will make total weights invalid)
    setting = ConfigurationSetting(
        category="model",
        key="weight_1h",
        value=2.0,  # Too high, will make sum > 1
        value_type="float",
        version=1,
    )
    test_db.add(setting)
    test_db.commit()

    # Reload should fail validation
    result = config.reload(db_session=test_db)

    assert result["status"] == "error"
    assert "Validation failed" in result["message"]


# ============================================================================
# RELOAD METADATA TESTS
# ============================================================================


def test_hot_reload_updates_metadata(test_db, reset_config):
    """Test that hot reload updates config metadata."""
    config = reset_config

    # Add setting
    setting = ConfigurationSetting(
        category="trading",
        key="confidence_threshold",
        value=0.75,
        value_type="float",
        version=1,
    )
    test_db.add(setting)
    test_db.commit()

    # Reload
    config.reload(db_session=test_db)

    # Get metadata
    all_config = config.get_all()
    metadata = all_config["metadata"]

    assert metadata["initialized"] is True
    assert metadata["db_loaded"] is True
    assert metadata["last_reload"] is not None
    assert metadata["config_version"] > 0


def test_get_all_includes_config_version(reset_config):
    """Test that get_all() includes config version in metadata."""
    config = reset_config

    all_config = config.get_all()

    assert "metadata" in all_config
    assert "config_version" in all_config["metadata"]
    assert isinstance(all_config["metadata"]["config_version"], int)


# ============================================================================
# BACKWARD COMPATIBILITY TESTS
# ============================================================================


def test_hot_reload_preserves_unset_defaults(test_db, reset_config):
    """Test that hot reload preserves defaults for unset parameters."""
    config = reset_config

    # Only set one parameter in DB
    setting = ConfigurationSetting(
        category="trading",
        key="confidence_threshold",
        value=0.80,
        value_type="float",
        version=1,
    )
    test_db.add(setting)
    test_db.commit()

    # Reload
    config.reload(db_session=test_db)

    # Updated parameter should change
    assert config.trading.confidence_threshold == 0.80

    # Other parameters should keep defaults
    assert config.trading.default_lot_size == 0.1
    assert config.trading.pip_value == 10.0
    assert config.trading.default_tp_pips == 25.0


def test_hot_reload_with_empty_database(test_db, reset_config):
    """Test hot reload with empty database uses defaults."""
    config = reset_config

    # Reload with no settings in DB
    result = config.reload(db_session=test_db)

    assert result["status"] == "success"

    # Should still have default values
    assert config.trading.confidence_threshold == 0.66
    assert config.model.weight_1h == 0.6
    assert config.risk.max_drawdown_percent == 15.0
