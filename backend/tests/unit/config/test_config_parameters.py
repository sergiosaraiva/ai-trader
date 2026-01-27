"""Tests for configuration parameter dataclasses.

Tests all parameter dataclasses including validation, conversion, and edge cases.
"""

import pytest
import sys
from pathlib import Path

# Add backend src to path
backend_path = Path(__file__).parent.parent.parent.parent
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

TradingParameters = trading_config_module.TradingParameters
ModelParameters = trading_config_module.ModelParameters
RiskParameters = trading_config_module.RiskParameters
SystemParameters = trading_config_module.SystemParameters
TimeframeParameters = trading_config_module.TimeframeParameters
AgentParameters = trading_config_module.AgentParameters
CacheParameters = trading_config_module.CacheParameters
SchedulerParameters = trading_config_module.SchedulerParameters
FeatureParameters = trading_config_module.FeatureParameters


# ============================================================================
# TRADING PARAMETERS TESTS
# ============================================================================


def test_trading_parameters_defaults():
    """Test TradingParameters default values."""
    params = TradingParameters()

    assert params.confidence_threshold == 0.66
    assert params.default_lot_size == 0.1
    assert params.pip_value == 10.0
    assert params.default_tp_pips == 25.0
    assert params.default_sl_pips == 15.0
    assert params.max_holding_hours == 12
    assert params.initial_balance == 100000.0


def test_trading_parameters_to_dict():
    """Test TradingParameters to_dict conversion."""
    params = TradingParameters()
    data = params.to_dict()

    assert isinstance(data, dict)
    assert "confidence_threshold" in data
    assert "default_lot_size" in data
    assert "pip_value" in data
    assert data["confidence_threshold"] == 0.66
    assert data["initial_balance"] == 100000.0


def test_trading_parameters_custom_values():
    """Test TradingParameters with custom values."""
    params = TradingParameters(
        confidence_threshold=0.75,
        default_lot_size=0.2,
        pip_value=20.0,
        default_tp_pips=30.0,
        default_sl_pips=20.0,
        max_holding_hours=24,
        initial_balance=50000.0,
    )

    assert params.confidence_threshold == 0.75
    assert params.default_lot_size == 0.2
    assert params.pip_value == 20.0
    assert params.default_tp_pips == 30.0
    assert params.default_sl_pips == 20.0
    assert params.max_holding_hours == 24
    assert params.initial_balance == 50000.0


# ============================================================================
# MODEL PARAMETERS TESTS
# ============================================================================


def test_model_parameters_defaults():
    """Test ModelParameters default values."""
    params = ModelParameters()

    assert params.weight_1h == 0.6
    assert params.weight_4h == 0.3
    assert params.weight_daily == 0.1
    assert params.agreement_bonus == 0.05
    assert params.use_regime_adjustment is True


def test_model_parameters_get_weights():
    """Test ModelParameters get_weights method."""
    params = ModelParameters()
    weights = params.get_weights()

    assert isinstance(weights, dict)
    assert weights["1H"] == 0.6
    assert weights["4H"] == 0.3
    assert weights["D"] == 0.1
    assert sum(weights.values()) == pytest.approx(1.0)


def test_model_parameters_to_dict():
    """Test ModelParameters to_dict conversion."""
    params = ModelParameters()
    data = params.to_dict()

    assert isinstance(data, dict)
    assert "weight_1h" in data
    assert "weight_4h" in data
    assert "weight_daily" in data
    assert "agreement_bonus" in data
    assert "use_regime_adjustment" in data


def test_model_parameters_custom_weights():
    """Test ModelParameters with custom weights."""
    params = ModelParameters(
        weight_1h=0.5,
        weight_4h=0.35,
        weight_daily=0.15,
    )

    weights = params.get_weights()
    assert weights["1H"] == 0.5
    assert weights["4H"] == 0.35
    assert weights["D"] == 0.15
    assert sum(weights.values()) == pytest.approx(1.0)


# ============================================================================
# RISK PARAMETERS TESTS
# ============================================================================


def test_risk_parameters_defaults():
    """Test RiskParameters default values."""
    params = RiskParameters()

    assert params.max_consecutive_losses == 5
    assert params.max_drawdown_percent == 15.0
    assert params.max_daily_loss_percent == 5.0
    assert params.max_daily_loss_amount == 5000.0
    assert params.enable_model_degradation is False
    assert params.min_win_rate == 0.45
    assert params.degradation_window == 20
    assert params.max_trades_per_day == 50
    assert params.max_trades_per_hour == 20


def test_risk_parameters_max_drawdown_15_percent():
    """Test that max_drawdown_percent defaults to 15% (CRITICAL requirement)."""
    params = RiskParameters()

    assert params.max_drawdown_percent == 15.0, "Max drawdown MUST be 15% not 10%"


def test_risk_parameters_to_dict():
    """Test RiskParameters to_dict conversion."""
    params = RiskParameters()
    data = params.to_dict()

    assert isinstance(data, dict)
    assert "max_consecutive_losses" in data
    assert "max_drawdown_percent" in data
    assert "enable_model_degradation" in data
    assert data["max_drawdown_percent"] == 15.0


def test_risk_parameters_custom_values():
    """Test RiskParameters with custom values."""
    params = RiskParameters(
        max_consecutive_losses=7,
        max_drawdown_percent=12.0,
        max_daily_loss_percent=3.0,
        enable_model_degradation=True,
        min_win_rate=0.50,
    )

    assert params.max_consecutive_losses == 7
    assert params.max_drawdown_percent == 12.0
    assert params.max_daily_loss_percent == 3.0
    assert params.enable_model_degradation is True
    assert params.min_win_rate == 0.50


# ============================================================================
# SYSTEM PARAMETERS TESTS
# ============================================================================


def test_system_parameters_defaults():
    """Test SystemParameters default values."""
    params = SystemParameters()

    assert params.cache_ttl_seconds == 60
    assert params.scheduler_enabled is True
    assert params.db_timeout_seconds == 10.0
    assert params.broker_timeout_seconds == 30.0


def test_system_parameters_to_dict():
    """Test SystemParameters to_dict conversion."""
    params = SystemParameters()
    data = params.to_dict()

    assert isinstance(data, dict)
    assert "cache_ttl_seconds" in data
    assert "scheduler_enabled" in data
    assert "db_timeout_seconds" in data
    assert "broker_timeout_seconds" in data


def test_system_parameters_custom_values():
    """Test SystemParameters with custom values."""
    params = SystemParameters(
        cache_ttl_seconds=120,
        scheduler_enabled=False,
        db_timeout_seconds=5.0,
        broker_timeout_seconds=60.0,
    )

    assert params.cache_ttl_seconds == 120
    assert params.scheduler_enabled is False
    assert params.db_timeout_seconds == 5.0
    assert params.broker_timeout_seconds == 60.0


# ============================================================================
# TIMEFRAME PARAMETERS TESTS
# ============================================================================


def test_timeframe_parameters_1h_defaults():
    """Test TimeframeParameters for 1H timeframe."""
    params = TimeframeParameters(
        timeframe="1H",
        tp_pips=25.0,
        sl_pips=15.0,
        max_holding_bars=12,
        weight=0.6,
    )

    assert params.timeframe == "1H"
    assert params.tp_pips == 25.0
    assert params.sl_pips == 15.0
    assert params.max_holding_bars == 12
    assert params.weight == 0.6


def test_timeframe_parameters_4h_defaults():
    """Test TimeframeParameters for 4H timeframe."""
    params = TimeframeParameters(
        timeframe="4H",
        tp_pips=50.0,
        sl_pips=25.0,
        max_holding_bars=18,
        weight=0.3,
    )

    assert params.timeframe == "4H"
    assert params.tp_pips == 50.0
    assert params.sl_pips == 25.0
    assert params.max_holding_bars == 18
    assert params.weight == 0.3


def test_timeframe_parameters_daily_defaults():
    """Test TimeframeParameters for Daily timeframe."""
    params = TimeframeParameters(
        timeframe="D",
        tp_pips=150.0,
        sl_pips=75.0,
        max_holding_bars=15,
        weight=0.1,
    )

    assert params.timeframe == "D"
    assert params.tp_pips == 150.0
    assert params.sl_pips == 75.0
    assert params.max_holding_bars == 15
    assert params.weight == 0.1


def test_timeframe_parameters_to_dict():
    """Test TimeframeParameters to_dict conversion."""
    params = TimeframeParameters(
        timeframe="1H",
        tp_pips=25.0,
        sl_pips=15.0,
        max_holding_bars=12,
        weight=0.6,
    )

    data = params.to_dict()
    assert isinstance(data, dict)
    assert data["timeframe"] == "1H"
    assert data["tp_pips"] == 25.0
    assert data["weight"] == 0.6


# ============================================================================
# AGENT PARAMETERS TESTS
# ============================================================================


def test_agent_parameters_defaults():
    """Test AgentParameters default values."""
    params = AgentParameters()

    assert params.mode == "simulation"
    assert params.symbol == "EURUSD"
    assert params.max_position_size == 0.1
    assert params.use_kelly_sizing is True
    assert params.cycle_interval_seconds == 60
    assert params.health_port == 8002
    assert params.max_reconnect_attempts == 5
    assert params.max_reconnect_delay == 60.0
    assert params.shutdown_timeout_seconds == 30.0
    assert params.close_positions_on_shutdown is True


def test_agent_parameters_to_dict():
    """Test AgentParameters to_dict conversion."""
    params = AgentParameters()
    data = params.to_dict()

    assert isinstance(data, dict)
    assert "mode" in data
    assert "symbol" in data
    assert "max_position_size" in data
    assert data["mode"] == "simulation"
    assert data["symbol"] == "EURUSD"


def test_agent_parameters_custom_values():
    """Test AgentParameters with custom values."""
    params = AgentParameters(
        mode="live",
        symbol="GBPUSD",
        max_position_size=0.5,
        use_kelly_sizing=False,
        cycle_interval_seconds=120,
    )

    assert params.mode == "live"
    assert params.symbol == "GBPUSD"
    assert params.max_position_size == 0.5
    assert params.use_kelly_sizing is False
    assert params.cycle_interval_seconds == 120


# ============================================================================
# CACHE PARAMETERS TESTS
# ============================================================================


def test_cache_parameters_defaults():
    """Test CacheParameters default values."""
    params = CacheParameters()

    assert params.prediction_cache_ttl_seconds == 60
    assert params.prediction_cache_max_size == 100
    assert params.price_cache_max_size == 50
    assert params.ohlcv_cache_max_size == 20
    assert params.asset_cache_max_size == 100


def test_cache_parameters_to_dict():
    """Test CacheParameters to_dict conversion."""
    params = CacheParameters()
    data = params.to_dict()

    assert isinstance(data, dict)
    assert "prediction_cache_ttl_seconds" in data
    assert "prediction_cache_max_size" in data
    assert data["prediction_cache_ttl_seconds"] == 60


def test_cache_parameters_custom_values():
    """Test CacheParameters with custom values."""
    params = CacheParameters(
        prediction_cache_ttl_seconds=120,
        prediction_cache_max_size=200,
        price_cache_max_size=100,
    )

    assert params.prediction_cache_ttl_seconds == 120
    assert params.prediction_cache_max_size == 200
    assert params.price_cache_max_size == 100


# ============================================================================
# SCHEDULER PARAMETERS TESTS
# ============================================================================


def test_scheduler_parameters_defaults():
    """Test SchedulerParameters default values."""
    params = SchedulerParameters()

    assert params.pipeline_cron_minute == 55
    assert params.prediction_cron_minute == 1
    assert params.market_data_interval_minutes == 5
    assert params.position_check_interval_minutes == 5
    assert params.misfire_grace_time_seconds == 300


def test_scheduler_parameters_to_dict():
    """Test SchedulerParameters to_dict conversion."""
    params = SchedulerParameters()
    data = params.to_dict()

    assert isinstance(data, dict)
    assert "pipeline_cron_minute" in data
    assert "prediction_cron_minute" in data
    assert data["pipeline_cron_minute"] == 55


def test_scheduler_parameters_custom_values():
    """Test SchedulerParameters with custom values."""
    params = SchedulerParameters(
        pipeline_cron_minute=30,
        prediction_cron_minute=5,
        market_data_interval_minutes=10,
    )

    assert params.pipeline_cron_minute == 30
    assert params.prediction_cron_minute == 5
    assert params.market_data_interval_minutes == 10


# ============================================================================
# FEATURE PARAMETERS TESTS
# ============================================================================


def test_feature_parameters_defaults():
    """Test FeatureParameters default values."""
    params = FeatureParameters()

    assert params.use_regime_detection is True
    assert params.regime_lookback_periods == 50
    assert params.use_sentiment_1h is False
    assert params.use_sentiment_4h is False
    assert params.use_sentiment_daily is True
    assert params.sentiment_cache_ttl_seconds == 3600


def test_feature_parameters_sentiment_alignment():
    """Test sentiment alignment with timeframes (CRITICAL requirement)."""
    params = FeatureParameters()

    # EPU/VIX should only be on Daily timeframe
    assert params.use_sentiment_1h is False, "Sentiment should NOT be on 1H"
    assert params.use_sentiment_4h is False, "Sentiment should NOT be on 4H"
    assert params.use_sentiment_daily is True, "Sentiment MUST be on Daily"


def test_feature_parameters_to_dict():
    """Test FeatureParameters to_dict conversion."""
    params = FeatureParameters()
    data = params.to_dict()

    assert isinstance(data, dict)
    assert "use_regime_detection" in data
    assert "use_sentiment_daily" in data
    assert data["use_sentiment_daily"] is True


def test_feature_parameters_custom_values():
    """Test FeatureParameters with custom values."""
    params = FeatureParameters(
        use_regime_detection=False,
        regime_lookback_periods=100,
        sentiment_cache_ttl_seconds=7200,
    )

    assert params.use_regime_detection is False
    assert params.regime_lookback_periods == 100
    assert params.sentiment_cache_ttl_seconds == 7200


# ============================================================================
# PARAMETER IMMUTABILITY TESTS
# ============================================================================


def test_parameters_are_mutable_dataclasses():
    """Test that parameters can be modified (they are not frozen dataclasses)."""
    params = TradingParameters()

    # Should be able to modify
    params.confidence_threshold = 0.75
    assert params.confidence_threshold == 0.75

    params.default_lot_size = 0.2
    assert params.default_lot_size == 0.2


# ============================================================================
# PARAMETER TYPE CHECKING
# ============================================================================


def test_trading_parameters_type_checking():
    """Test TradingParameters accepts correct types."""
    # Should accept correct types
    params = TradingParameters(
        confidence_threshold=0.75,  # float
        default_lot_size=0.1,  # float
        pip_value=10.0,  # float
        max_holding_hours=12,  # int
        initial_balance=100000.0,  # float
    )

    assert isinstance(params.confidence_threshold, float)
    assert isinstance(params.max_holding_hours, int)


def test_agent_parameters_mode_values():
    """Test AgentParameters mode accepts valid strings."""
    # Valid modes
    params1 = AgentParameters(mode="simulation")
    assert params1.mode == "simulation"

    params2 = AgentParameters(mode="live")
    assert params2.mode == "live"


# ============================================================================
# PARAMETER DICT ROUNDTRIP
# ============================================================================


def test_parameters_dict_roundtrip():
    """Test that parameters can be converted to dict and back."""
    original = TradingParameters(
        confidence_threshold=0.75,
        default_lot_size=0.2,
    )

    # Convert to dict
    data = original.to_dict()

    # Create new instance from dict
    restored = TradingParameters(**data)

    # Should match original
    assert restored.confidence_threshold == original.confidence_threshold
    assert restored.default_lot_size == original.default_lot_size
