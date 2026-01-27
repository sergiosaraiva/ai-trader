"""Integration tests for configuration system with services.

Tests how configuration changes integrate with model_service and trading_service,
including callback propagation and hot reload impact.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Add backend src to path
backend_path = Path(__file__).parent.parent.parent
src_path = backend_path / "src"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import directly from module to avoid __init__.py dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "trading_config",
    src_path / "config" / "trading_config.py"
)
trading_config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trading_config_module)

TradingConfig = trading_config_module.TradingConfig


# ============================================================================
# CALLBACK PROPAGATION TESTS
# ============================================================================


def test_model_service_receives_config_updates():
    """Test that model_service receives configuration updates via callback."""
    config = TradingConfig()

    # Mock model service
    model_service_called = {"called": False, "new_config": None}

    def model_service_config_callback(params):
        """Simulated model_service callback."""
        model_service_called["called"] = True
        model_service_called["new_config"] = params

    # Register callback
    config.register_callback("model", model_service_config_callback)

    # Update model config
    config.update(
        category="model",
        updates={"weight_1h": 0.7, "weight_4h": 0.2, "weight_daily": 0.1},
        updated_by="admin",
        db_session=None,
    )

    # Verify callback was called
    assert model_service_called["called"]
    assert model_service_called["new_config"] is not None
    assert model_service_called["new_config"].weight_1h == 0.7


def test_trading_service_receives_config_updates():
    """Test that trading_service receives configuration updates via callback."""
    config = TradingConfig()

    # Mock trading service
    trading_service_state = {"confidence": None, "lot_size": None}

    def trading_service_config_callback(params):
        """Simulated trading_service callback."""
        trading_service_state["confidence"] = params.confidence_threshold
        trading_service_state["lot_size"] = params.default_lot_size

    # Register callback
    config.register_callback("trading", trading_service_config_callback)

    # Update trading config
    config.update(
        category="trading",
        updates={
            "confidence_threshold": 0.75,
            "default_lot_size": 0.2,
        },
        updated_by="admin",
        db_session=None,
    )

    # Verify trading service received updates
    assert trading_service_state["confidence"] == 0.75
    assert trading_service_state["lot_size"] == 0.2


def test_risk_manager_receives_config_updates():
    """Test that risk_manager receives configuration updates via callback."""
    config = TradingConfig()

    # Mock risk manager
    risk_manager_state = {"max_drawdown": None, "max_losses": None}

    def risk_manager_config_callback(params):
        """Simulated risk_manager callback."""
        risk_manager_state["max_drawdown"] = params.max_drawdown_percent
        risk_manager_state["max_losses"] = params.max_consecutive_losses

    # Register callback
    config.register_callback("risk", risk_manager_config_callback)

    # Update risk config
    config.update(
        category="risk",
        updates={
            "max_drawdown_percent": 20.0,
            "max_consecutive_losses": 7,
        },
        updated_by="admin",
        db_session=None,
    )

    # Verify risk manager received updates
    assert risk_manager_state["max_drawdown"] == 20.0
    assert risk_manager_state["max_losses"] == 7


def test_multiple_services_receive_updates():
    """Test that multiple services can register and receive callbacks."""
    config = TradingConfig()

    # Mock multiple services
    service_calls = {
        "model_service": False,
        "trading_service": False,
        "prediction_service": False,
    }

    def model_callback(params):
        service_calls["model_service"] = True

    def trading_callback(params):
        service_calls["trading_service"] = True

    def prediction_callback(params):
        service_calls["prediction_service"] = True

    # Register all callbacks for model category
    config.register_callback("model", model_callback)
    config.register_callback("model", trading_callback)
    config.register_callback("model", prediction_callback)

    # Update model config
    config.update(
        category="model",
        updates={"agreement_bonus": 0.08},
        updated_by="admin",
        db_session=None,
    )

    # All services should have received the update
    assert service_calls["model_service"]
    assert service_calls["trading_service"]
    assert service_calls["prediction_service"]


# ============================================================================
# HOT RELOAD IMPACT TESTS
# ============================================================================


def test_hot_reload_updates_all_services():
    """Test that hot reload triggers callbacks for all services."""
    config = TradingConfig()

    # Mock services
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

    # Register callbacks for all categories
    for category in ["trading", "model", "risk", "system"]:
        config.register_callback(category, create_callback(category))

    # Mock database with settings
    mock_db = Mock()
    mock_settings = [
        Mock(category="trading", key="confidence_threshold", value=0.75),
        Mock(category="model", key="weight_1h", value=0.7),
        Mock(category="risk", key="max_drawdown_percent", value=20.0),
        Mock(category="system", key="cache_ttl_seconds", value=120),
    ]
    mock_db.query.return_value.all.return_value = mock_settings

    # Reload config
    result = config.reload(db_session=mock_db)

    assert result["status"] == "success"

    # All services should have been notified
    assert callback_counts["trading"] == 1
    assert callback_counts["model"] == 1
    assert callback_counts["risk"] == 1
    assert callback_counts["system"] == 1


def test_hot_reload_without_breaking_services():
    """Test that services continue working during hot reload."""
    config = TradingConfig()

    # Mock service that maintains state
    service_state = {
        "operational": True,
        "confidence": config.trading.confidence_threshold,
        "update_count": 0,
    }

    def service_callback(params):
        """Service callback that maintains operational state."""
        service_state["confidence"] = params.confidence_threshold
        service_state["update_count"] += 1
        # Service remains operational during update
        assert service_state["operational"]

    config.register_callback("trading", service_callback)

    # Mock database
    mock_db = Mock()
    mock_setting = Mock(category="trading", key="confidence_threshold", value=0.80)
    mock_db.query.return_value.all.return_value = [mock_setting]

    # Reload while service is "operational"
    result = config.reload(db_session=mock_db)

    assert result["status"] == "success"
    assert service_state["operational"]
    assert service_state["confidence"] == 0.80
    assert service_state["update_count"] == 1


# ============================================================================
# CONFIGURATION HIERARCHY TESTS (DB > ENV > DEFAULTS)
# ============================================================================


def test_database_overrides_environment():
    """Test that database values take precedence over environment variables."""
    config = TradingConfig()

    # Set environment variable
    with patch.dict("os.environ", {"CONFIDENCE_THRESHOLD": "0.70"}):
        # Reload from environment
        config._load_from_env()
        assert config.trading.confidence_threshold == 0.70

        # Mock database with different value
        mock_db = Mock()
        mock_setting = Mock(
            category="trading",
            key="confidence_threshold",
            value=0.80  # DB value should win
        )
        mock_db.query.return_value.all.return_value = [mock_setting]

        # Load from database
        config._load_from_db(mock_db)

        # Database value should override environment
        assert config.trading.confidence_threshold == 0.80


def test_environment_overrides_defaults():
    """Test that environment variables override default values."""
    config = TradingConfig()

    # Verify default
    assert config.trading.confidence_threshold == 0.66

    # Set environment variable and reload
    with patch.dict("os.environ", {"CONFIDENCE_THRESHOLD": "0.75"}):
        config._load_from_env()

        # Environment should override default
        assert config.trading.confidence_threshold == 0.75


def test_defaults_used_when_no_overrides():
    """Test that defaults are used when no DB or env overrides exist."""
    config = TradingConfig()

    # Reset to defaults
    config.trading.confidence_threshold = 0.66

    # Mock empty database
    mock_db = Mock()
    mock_db.query.return_value.all.return_value = []

    # Load from DB (empty)
    config._load_from_db(mock_db)

    # Should still have default value
    assert config.trading.confidence_threshold == 0.66


def test_partial_database_overrides():
    """Test that some parameters from DB and others from defaults."""
    config = TradingConfig()

    # Mock database with only one setting
    mock_db = Mock()
    mock_setting = Mock(
        category="trading",
        key="confidence_threshold",
        value=0.80
    )
    mock_db.query.return_value.all.return_value = [mock_setting]

    # Load from database
    config._load_from_db(mock_db)

    # One parameter from DB
    assert config.trading.confidence_threshold == 0.80

    # Other parameters should have defaults
    assert config.trading.default_lot_size == 0.1
    assert config.trading.pip_value == 10.0


# ============================================================================
# SERVICE COORDINATION TESTS
# ============================================================================


def test_config_change_coordinates_services():
    """Test that config changes properly coordinate multiple services."""
    config = TradingConfig()

    # Simulate services that need to coordinate
    service_states = {
        "model_service": {"ready": False, "weights": None},
        "prediction_service": {"ready": False, "model_weights": None},
        "trading_service": {"ready": False, "using_weights": None},
    }

    def model_service_callback(params):
        """Model service updates weights."""
        service_states["model_service"]["weights"] = params.get_weights()
        service_states["model_service"]["ready"] = True

    def prediction_service_callback(params):
        """Prediction service gets weights from model service."""
        # Wait for model service
        if service_states["model_service"]["ready"]:
            service_states["prediction_service"]["model_weights"] = \
                service_states["model_service"]["weights"]
            service_states["prediction_service"]["ready"] = True

    def trading_service_callback(params):
        """Trading service uses weights."""
        # Uses weights after prediction service is ready
        if service_states["prediction_service"]["ready"]:
            service_states["trading_service"]["using_weights"] = \
                service_states["prediction_service"]["model_weights"]
            service_states["trading_service"]["ready"] = True

    # Register callbacks in order
    config.register_callback("model", model_service_callback)
    config.register_callback("model", prediction_service_callback)
    config.register_callback("model", trading_service_callback)

    # Update model config
    config.update(
        category="model",
        updates={
            "weight_1h": 0.7,
            "weight_4h": 0.2,
            "weight_daily": 0.1,
        },
        updated_by="admin",
        db_session=None,
    )

    # All services should coordinate properly
    assert service_states["model_service"]["ready"]
    assert service_states["prediction_service"]["ready"]
    assert service_states["trading_service"]["ready"]

    # Weights should propagate through services
    assert service_states["model_service"]["weights"]["1H"] == 0.7
    assert service_states["prediction_service"]["model_weights"]["1H"] == 0.7
    assert service_states["trading_service"]["using_weights"]["1H"] == 0.7


def test_service_callback_isolation():
    """Test that callback failures in one service don't affect others."""
    config = TradingConfig()

    callback_results = {
        "service1": False,
        "service2": False,  # This will fail
        "service3": False,
    }

    def service1_callback(params):
        callback_results["service1"] = True

    def service2_callback(params):
        callback_results["service2"] = True
        raise RuntimeError("Service 2 failed!")

    def service3_callback(params):
        callback_results["service3"] = True

    # Register all callbacks
    config.register_callback("trading", service1_callback)
    config.register_callback("trading", service2_callback)
    config.register_callback("trading", service3_callback)

    # Update config
    config.update(
        category="trading",
        updates={"confidence_threshold": 0.75},
        updated_by="admin",
        db_session=None,
    )

    # Service 1 and 3 should succeed despite service 2 failure
    assert callback_results["service1"]
    assert callback_results["service2"]  # Was called (but failed)
    assert callback_results["service3"]


# ============================================================================
# REAL-TIME UPDATE TESTS
# ============================================================================


def test_service_uses_updated_config_immediately():
    """Test that services use updated config immediately after update."""
    config = TradingConfig()

    # Simulate service that checks config
    service_confidence_used = []

    def make_prediction():
        """Service function that uses config."""
        return config.trading.confidence_threshold

    # Initial prediction uses default
    service_confidence_used.append(make_prediction())

    # Update config
    config.update(
        category="trading",
        updates={"confidence_threshold": 0.80},
        updated_by="admin",
        db_session=None,
    )

    # Next prediction should use new value immediately
    service_confidence_used.append(make_prediction())

    assert service_confidence_used[0] == 0.66  # Default
    assert service_confidence_used[1] == 0.80  # Updated


def test_concurrent_service_operations_during_config_update():
    """Test that services can operate during config updates."""
    config = TradingConfig()

    # Track service operations
    operation_results = []

    def service_operation():
        """Simulated service operation."""
        # Read config
        conf = config.trading.confidence_threshold
        # Do some work
        result = conf * 100
        operation_results.append(result)

    import threading

    # Start service operations
    threads = []
    for _ in range(5):
        t = threading.Thread(target=service_operation)
        threads.append(t)
        t.start()

    # Update config concurrently
    config.update(
        category="trading",
        updates={"confidence_threshold": 0.75},
        updated_by="admin",
        db_session=None,
    )

    # More service operations
    for _ in range(5):
        t = threading.Thread(target=service_operation)
        threads.append(t)
        t.start()

    # Wait for all
    for t in threads:
        t.join()

    # All operations should complete successfully
    assert len(operation_results) == 10
    # Results should be either 66 or 75 (before/after update)
    for result in operation_results:
        assert result == 66.0 or result == 75.0


# ============================================================================
# VALIDATION WITH SERVICE CONSTRAINTS
# ============================================================================


def test_config_update_validates_service_constraints():
    """Test that config validation considers service constraints."""
    config = TradingConfig()

    # Service that enforces minimum confidence
    validation_failed = {"failed": False}

    def service_validator(params):
        """Service-level validation."""
        if params.confidence_threshold < 0.60:
            validation_failed["failed"] = True

    config.register_callback("trading", service_validator)

    # Try to set confidence below service minimum (but above config minimum)
    config.update(
        category="trading",
        updates={"confidence_threshold": 0.55},
        updated_by="admin",
        db_session=None,
    )

    # Service detected violation
    assert validation_failed["failed"]


def test_service_can_reject_invalid_config():
    """Test that service can effectively reject invalid configuration."""
    config = TradingConfig()

    # Service that tracks if config is valid for its use
    service_state = {"config_valid": True}

    def service_callback(params):
        """Service validates config for its needs."""
        # Service requires weights to sum to exactly 1.0
        weights = params.get_weights()
        total = sum(weights.values())
        if not (0.99 <= total <= 1.01):
            service_state["config_valid"] = False
        else:
            service_state["config_valid"] = True

    config.register_callback("model", service_callback)

    # Update with valid weights
    config.update(
        category="model",
        updates={"weight_1h": 0.6, "weight_4h": 0.3, "weight_daily": 0.1},
        updated_by="admin",
        db_session=None,
    )

    assert service_state["config_valid"]

    # Try to update with invalid weights (validation should catch this)
    try:
        config.update(
            category="model",
            updates={"weight_1h": 0.8, "weight_4h": 0.8, "weight_daily": 0.8},
            updated_by="admin",
            db_session=None,
        )
    except ValueError:
        pass  # Expected validation failure

    # Service state should still be valid (update was rejected)
    assert service_state["config_valid"]
