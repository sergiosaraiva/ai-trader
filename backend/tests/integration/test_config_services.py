"""Integration tests for services using centralized configuration.

Tests how model_service, trading_service, and other services interact
with the centralized configuration system.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

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
    config.agent.mode = "simulation"
    config.cache.prediction_cache_ttl_seconds = 60
    config.features.use_sentiment_daily = True
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
# MODEL SERVICE INTEGRATION TESTS
# ============================================================================


def test_model_service_uses_config_weights(reset_config):
    """Test that model service uses weights from config."""
    config = reset_config

    # Simulate model service reading config
    weights = config.model.get_weights()

    assert weights["1H"] == 0.6
    assert weights["4H"] == 0.3
    assert weights["D"] == 0.1


def test_model_service_reacts_to_weight_change(test_db, reset_config):
    """Test that model service can react to weight changes."""
    config = reset_config

    # Simulate model service state
    model_weights = {"current": None, "reloaded": False}

    def model_callback(params):
        """Simulates model service reloading weights."""
        model_weights["current"] = params.get_weights()
        model_weights["reloaded"] = True

    config.register_callback("model", model_callback)

    # Update weights via database
    settings = [
        ConfigurationSetting(category="model", key="weight_1h", value=0.5, value_type="float", version=1),
        ConfigurationSetting(category="model", key="weight_4h", value=0.35, value_type="float", version=1),
        ConfigurationSetting(category="model", key="weight_daily", value=0.15, value_type="float", version=1),
    ]

    for setting in settings:
        test_db.add(setting)
    test_db.commit()

    # Reload config
    config.reload(db_session=test_db)

    # Model service should have reloaded
    assert model_weights["reloaded"] is True
    assert model_weights["current"]["1H"] == 0.5
    assert model_weights["current"]["4H"] == 0.35
    assert model_weights["current"]["D"] == 0.15


def test_model_service_agreement_bonus(reset_config):
    """Test that model service uses agreement bonus from config."""
    config = reset_config

    # Simulate reading agreement bonus
    bonus = config.model.agreement_bonus

    assert bonus == 0.05

    # Update agreement bonus
    config.update(
        category="model",
        updates={"agreement_bonus": 0.08},
        updated_by="test",
        db_session=None,
    )

    # Should reflect new value
    assert config.model.agreement_bonus == 0.08


def test_model_service_sentiment_configuration(reset_config):
    """Test that model service respects sentiment configuration."""
    config = reset_config

    # Check sentiment settings (CRITICAL: only Daily should have sentiment)
    assert config.features.use_sentiment_1h is False
    assert config.features.use_sentiment_4h is False
    assert config.features.use_sentiment_daily is True


# ============================================================================
# TRADING SERVICE INTEGRATION TESTS
# ============================================================================


def test_trading_service_uses_confidence_threshold(reset_config):
    """Test that trading service uses confidence threshold from config."""
    config = reset_config

    # Simulate trading decision
    prediction_confidence = 0.70
    should_trade = prediction_confidence >= config.trading.confidence_threshold

    assert should_trade is True


def test_trading_service_respects_threshold_update(test_db, reset_config):
    """Test that trading service respects threshold updates."""
    config = reset_config

    # Simulate trading service state
    trading_decisions = []

    def make_trading_decision(confidence):
        """Simulates trading decision based on current config."""
        return confidence >= config.trading.confidence_threshold

    # With default threshold (0.66)
    assert make_trading_decision(0.70) is True
    assert make_trading_decision(0.65) is False

    # Update threshold
    config.update(
        category="trading",
        updates={"confidence_threshold": 0.75},
        updated_by="test",
        db_session=None,
    )

    # Now 0.70 should NOT trade
    assert make_trading_decision(0.70) is False
    assert make_trading_decision(0.80) is True


def test_trading_service_lot_size_configuration(reset_config):
    """Test that trading service uses lot size from config."""
    config = reset_config

    # Simulate position sizing
    lot_size = config.trading.default_lot_size

    assert lot_size == 0.1

    # Update lot size
    config.update(
        category="trading",
        updates={"default_lot_size": 0.2},
        updated_by="test",
        db_session=None,
    )

    assert config.trading.default_lot_size == 0.2


def test_trading_service_tp_sl_configuration(reset_config):
    """Test that trading service uses TP/SL from config."""
    config = reset_config

    # Simulate setting TP/SL levels
    tp_pips = config.trading.default_tp_pips
    sl_pips = config.trading.default_sl_pips

    assert tp_pips == 25.0
    assert sl_pips == 15.0

    # Calculate prices (example: EUR/USD @ 1.0800)
    entry_price = 1.0800
    pip_size = 0.0001

    # Long position
    take_profit = entry_price + (tp_pips * pip_size)
    stop_loss = entry_price - (sl_pips * pip_size)

    assert take_profit == pytest.approx(1.0825)
    assert stop_loss == pytest.approx(1.0785)


# ============================================================================
# RISK SERVICE INTEGRATION TESTS
# ============================================================================


def test_risk_service_max_drawdown_check(reset_config):
    """Test that risk service checks max drawdown from config."""
    config = reset_config

    # Simulate risk check
    current_drawdown = 12.0  # 12% drawdown
    max_drawdown = config.risk.max_drawdown_percent

    assert max_drawdown == 15.0
    assert current_drawdown < max_drawdown  # Should allow trading


def test_risk_service_circuit_breaker_threshold(reset_config):
    """Test that risk service uses 15% circuit breaker threshold."""
    config = reset_config

    # CRITICAL: Max drawdown must be 15% not 10%
    assert config.risk.max_drawdown_percent == 15.0

    # Simulate circuit breaker logic
    current_drawdown = 15.1  # Exceeded threshold
    should_trigger = current_drawdown >= config.risk.max_drawdown_percent

    assert should_trigger is True


def test_risk_service_consecutive_losses(reset_config):
    """Test that risk service tracks consecutive losses."""
    config = reset_config

    # Simulate tracking consecutive losses
    consecutive_losses = 5
    max_consecutive_losses = config.risk.max_consecutive_losses

    assert max_consecutive_losses == 5
    assert consecutive_losses >= max_consecutive_losses  # Should trigger


def test_risk_service_daily_loss_limits(reset_config):
    """Test that risk service checks daily loss limits."""
    config = reset_config

    # Check both percentage and absolute limits
    max_loss_percent = config.risk.max_daily_loss_percent
    max_loss_amount = config.risk.max_daily_loss_amount

    assert max_loss_percent == 5.0
    assert max_loss_amount == 5000.0

    # Simulate daily loss check
    account_balance = 100000.0
    daily_loss = 4500.0

    loss_percent = (daily_loss / account_balance) * 100
    assert loss_percent < max_loss_percent
    assert daily_loss < max_loss_amount


def test_risk_service_trade_limits(reset_config):
    """Test that risk service enforces trade limits."""
    config = reset_config

    # Check trade limits
    max_trades_per_day = config.risk.max_trades_per_day
    max_trades_per_hour = config.risk.max_trades_per_hour

    assert max_trades_per_day == 50
    assert max_trades_per_hour == 20

    # Simulate trade count check
    trades_today = 45
    trades_this_hour = 18

    can_trade_daily = trades_today < max_trades_per_day
    can_trade_hourly = trades_this_hour < max_trades_per_hour

    assert can_trade_daily is True
    assert can_trade_hourly is True


# ============================================================================
# CACHE SERVICE INTEGRATION TESTS
# ============================================================================


def test_cache_service_uses_ttl_from_config(reset_config):
    """Test that cache service uses TTL from config."""
    config = reset_config

    # Simulate cache TTL
    cache_ttl = config.system.cache_ttl_seconds

    assert cache_ttl == 60


def test_cache_service_prediction_cache_config(reset_config):
    """Test that cache service uses prediction cache config."""
    config = reset_config

    # Check prediction cache settings
    prediction_ttl = config.cache.prediction_cache_ttl_seconds
    prediction_max_size = config.cache.prediction_cache_max_size

    assert prediction_ttl == 60
    assert prediction_max_size == 100


def test_cache_service_invalidation_on_config_change(reset_config):
    """Test that cache service can invalidate on config change."""
    config = reset_config

    # Simulate cache state
    cache_state = {
        "version": config.get_config_version(),
        "cleared": False,
    }

    def cache_callback(params):
        """Simulates cache invalidation on config change."""
        new_version = config.get_config_version()
        if new_version != cache_state["version"]:
            cache_state["cleared"] = True
            cache_state["version"] = new_version

    config.register_callback("system", cache_callback)

    # Update config
    config.update(
        category="system",
        updates={"cache_ttl_seconds": 120},
        updated_by="test",
        db_session=None,
    )

    # Cache should be cleared
    assert cache_state["cleared"] is True


# ============================================================================
# AGENT SERVICE INTEGRATION TESTS
# ============================================================================


def test_agent_service_mode_configuration(reset_config):
    """Test that agent service uses mode from config."""
    config = reset_config

    # Check agent mode
    mode = config.agent.mode

    assert mode == "simulation"


def test_agent_service_position_sizing(reset_config):
    """Test that agent service uses position sizing from config."""
    config = reset_config

    # Check position sizing settings
    max_position_size = config.agent.max_position_size
    use_kelly = config.agent.use_kelly_sizing

    assert max_position_size == 0.1
    assert use_kelly is True


def test_agent_service_cycle_interval(reset_config):
    """Test that agent service uses cycle interval from config."""
    config = reset_config

    # Check cycle settings
    cycle_interval = config.agent.cycle_interval_seconds

    assert cycle_interval == 60


def test_agent_service_shutdown_behavior(reset_config):
    """Test that agent service respects shutdown configuration."""
    config = reset_config

    # Check shutdown settings
    close_positions = config.agent.close_positions_on_shutdown
    shutdown_timeout = config.agent.shutdown_timeout_seconds

    assert close_positions is True
    assert shutdown_timeout == 30.0


# ============================================================================
# SCHEDULER SERVICE INTEGRATION TESTS
# ============================================================================


def test_scheduler_service_cron_configuration(reset_config):
    """Test that scheduler service uses cron config."""
    config = reset_config

    # Check cron settings
    pipeline_minute = config.scheduler.pipeline_cron_minute
    prediction_minute = config.scheduler.prediction_cron_minute

    assert pipeline_minute == 55
    assert prediction_minute == 1


def test_scheduler_service_interval_configuration(reset_config):
    """Test that scheduler service uses interval config."""
    config = reset_config

    # Check interval settings
    market_data_interval = config.scheduler.market_data_interval_minutes
    position_check_interval = config.scheduler.position_check_interval_minutes

    assert market_data_interval == 5
    assert position_check_interval == 5


def test_scheduler_service_misfire_configuration(reset_config):
    """Test that scheduler service uses misfire grace time."""
    config = reset_config

    # Check misfire settings
    misfire_grace_time = config.scheduler.misfire_grace_time_seconds

    assert misfire_grace_time == 300  # 5 minutes


# ============================================================================
# CROSS-SERVICE COORDINATION TESTS
# ============================================================================


def test_multiple_services_coordinate_on_config_change(test_db, reset_config):
    """Test that multiple services coordinate on config change."""
    config = reset_config

    # Simulate multiple services
    service_states = {
        "model": {"updated": False},
        "trading": {"updated": False},
        "risk": {"updated": False},
    }

    def create_service_callback(service_name):
        def callback(params):
            service_states[service_name]["updated"] = True
        return callback

    # Register service callbacks
    config.register_callback("model", create_service_callback("model"))
    config.register_callback("trading", create_service_callback("trading"))
    config.register_callback("risk", create_service_callback("risk"))

    # Update config via database
    settings = [
        ConfigurationSetting(category="model", key="weight_1h", value=0.5, value_type="float", version=1),
        ConfigurationSetting(category="trading", key="confidence_threshold", value=0.75, value_type="float", version=1),
        ConfigurationSetting(category="risk", key="max_consecutive_losses", value=7, value_type="int", version=1),
    ]

    for setting in settings:
        test_db.add(setting)
    test_db.commit()

    # Reload config
    config.reload(db_session=test_db)

    # All services should have been notified
    assert service_states["model"]["updated"] is True
    assert service_states["trading"]["updated"] is True
    assert service_states["risk"]["updated"] is True


def test_service_config_version_tracking(reset_config):
    """Test that services can track config version for consistency."""
    config = reset_config

    # Simulate service tracking config version
    service_cache = {
        "config_version": config.get_config_version(),
        "cached_data": {"some": "data"},
    }

    # Update config
    config.update(
        category="trading",
        updates={"confidence_threshold": 0.75},
        updated_by="test",
        db_session=None,
    )

    # Service should detect stale cache
    current_version = config.get_config_version()
    cache_is_stale = service_cache["config_version"] != current_version

    assert cache_is_stale is True


# ============================================================================
# TIMEFRAME CONFIGURATION TESTS
# ============================================================================


def test_service_uses_timeframe_config(reset_config):
    """Test that services use timeframe-specific configuration."""
    config = reset_config

    # Check 1H timeframe config
    tf_1h = config.timeframes["1H"]
    assert tf_1h.tp_pips == 25.0
    assert tf_1h.sl_pips == 15.0
    assert tf_1h.max_holding_bars == 12
    assert tf_1h.weight == 0.6

    # Check 4H timeframe config
    tf_4h = config.timeframes["4H"]
    assert tf_4h.tp_pips == 50.0
    assert tf_4h.sl_pips == 25.0
    assert tf_4h.max_holding_bars == 18
    assert tf_4h.weight == 0.3

    # Check Daily timeframe config
    tf_d = config.timeframes["D"]
    assert tf_d.tp_pips == 150.0
    assert tf_d.sl_pips == 75.0
    assert tf_d.max_holding_bars == 15
    assert tf_d.weight == 0.1


def test_service_updates_timeframe_config(test_db, reset_config):
    """Test that services can update timeframe-specific config."""
    config = reset_config

    # Update 1H timeframe config
    result = config.update_timeframe(
        timeframe="1H",
        updates={"tp_pips": 30.0, "sl_pips": 18.0},
        updated_by="test",
        db_session=None,
    )

    assert result["status"] == "success"
    assert config.timeframes["1H"].tp_pips == 30.0
    assert config.timeframes["1H"].sl_pips == 18.0


# ============================================================================
# BACKWARD COMPATIBILITY TESTS
# ============================================================================


def test_services_work_without_db_persistence(reset_config):
    """Test that services work when DB persistence is not available."""
    config = reset_config

    # Update without DB session
    result = config.update(
        category="trading",
        updates={"confidence_threshold": 0.75},
        updated_by="test",
        db_session=None,
    )

    assert result["status"] == "success"
    assert config.trading.confidence_threshold == 0.75


def test_service_legacy_attribute_access(reset_config):
    """Test that services can access config via legacy patterns."""
    config = reset_config

    # Direct attribute access (legacy pattern)
    confidence = config.trading.confidence_threshold
    weights = config.model.get_weights()
    max_dd = config.risk.max_drawdown_percent

    assert confidence == 0.66
    assert weights["1H"] == 0.6
    assert max_dd == 15.0
