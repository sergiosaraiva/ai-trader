"""Shared fixtures and imports for agent unit tests.

This module handles direct imports of agent modules without triggering
heavy API dependencies like pandas.
"""

import sys
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest
from unittest.mock import Mock
import importlib.util

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Mock heavy dependencies before any imports
sys.modules["pandas"] = Mock()
sys.modules["pandas_ta"] = Mock()
sys.modules["yfinance"] = Mock()
sys.modules["xgboost"] = Mock()
sys.modules["MetaTrader5"] = Mock()

# Mock apscheduler as a module hierarchy
mock_apscheduler = type(sys)("apscheduler")
mock_apscheduler.schedulers = type(sys)("schedulers")
mock_apscheduler.schedulers.background = type(sys)("background")
mock_apscheduler.schedulers.background.BackgroundScheduler = Mock
mock_apscheduler.triggers = type(sys)("triggers")
mock_apscheduler.triggers.interval = type(sys)("interval")
mock_apscheduler.triggers.interval.IntervalTrigger = Mock
mock_apscheduler.triggers.cron = type(sys)("cron")
mock_apscheduler.triggers.cron.CronTrigger = Mock
sys.modules["apscheduler"] = mock_apscheduler
sys.modules["apscheduler.schedulers"] = mock_apscheduler.schedulers
sys.modules["apscheduler.schedulers.background"] = mock_apscheduler.schedulers.background
sys.modules["apscheduler.triggers"] = mock_apscheduler.triggers
sys.modules["apscheduler.triggers.interval"] = mock_apscheduler.triggers.interval
sys.modules["apscheduler.triggers.cron"] = mock_apscheduler.triggers.cron

# Load database models module directly without importing api package
models_path = src_path / "api" / "database" / "models.py"
spec = importlib.util.spec_from_file_location("api.database.models", models_path)
models_module = importlib.util.module_from_spec(spec)
sys.modules["api.database.models"] = models_module
spec.loader.exec_module(models_module)

Base = models_module.Base
AgentState = models_module.AgentState
AgentCommand = models_module.AgentCommand

# Import agent.config (no dependencies)
from agent.config import AgentConfig

# Mock model_service before any agent module tries to import it
mock_model_service = Mock()
mock_model_service.is_initialized = False
mock_model_service.is_loaded = False
mock_model_service.initialize = Mock(return_value=True)

# Mock api.services.model_service for runner
mock_api_services = type(sys)("services")
mock_api_services.model_service = mock_model_service
sys.modules["src.api.services"] = mock_api_services
sys.modules["src.api.services.model_service"] = type(sys)("model_service")
sys.modules["src.api.services.model_service"].model_service = mock_model_service

# Import agent safety_config (no dependencies)
from agent.safety_config import SafetyConfig

# Mock trading dependencies for safety_manager before loading the module
# Create proper mock modules with attributes - mimic package structure in sys.modules
mock_trading = type(sys)("trading")
mock_circuit_breakers = type(sys)("circuit_breakers")
mock_cb_base = type(sys)("base")
mock_cb_manager = type(sys)("manager")
mock_safety = type(sys)("safety")
mock_kill_switch_mod = type(sys)("kill_switch")
mock_risk = type(sys)("risk")
mock_profiles = type(sys)("profiles")

# Create enum-like mock for RiskLevel
class MockRiskLevel:
    ULTRA_CONSERVATIVE = "ultra_conservative"
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ULTRA_AGGRESSIVE = "ultra_aggressive"

# Set up mock classes that safety_manager will import
mock_cb_manager.CircuitBreakerManager = Mock
mock_cb_base.TradeResult = Mock
mock_cb_base.TradingState = Mock
mock_cb_base.CircuitBreakerState = Mock
mock_kill_switch_mod.KillSwitch = Mock
mock_kill_switch_mod.KillSwitchConfig = Mock
mock_kill_switch_mod.TriggerType = Mock
mock_profiles.RiskProfile = Mock
mock_profiles.RiskLevel = MockRiskLevel

# Assemble the module hierarchy
mock_circuit_breakers.base = mock_cb_base
mock_circuit_breakers.manager = mock_cb_manager
mock_trading.circuit_breakers = mock_circuit_breakers
mock_trading.safety = mock_safety
mock_safety.kill_switch = mock_kill_switch_mod
mock_trading.risk = mock_risk
mock_risk.profiles = mock_profiles

# Create src mock with trading as submodule
mock_src = type(sys)("src")
mock_src.trading = mock_trading

# Register all mocks - IMPORTANT: Register parent package too
sys.modules["src"] = mock_src
sys.modules["src.trading"] = mock_trading
sys.modules["src.trading.circuit_breakers"] = mock_circuit_breakers
sys.modules["src.trading.circuit_breakers.manager"] = mock_cb_manager
sys.modules["src.trading.circuit_breakers.base"] = mock_cb_base
sys.modules["src.trading.safety"] = mock_safety
sys.modules["src.trading.safety.kill_switch"] = mock_kill_switch_mod
sys.modules["src.trading.risk"] = mock_risk
sys.modules["src.trading.risk.profiles"] = mock_profiles

# Load safety_manager module - Load as src.agent.safety_manager for relative imports
# Create src.agent mock and link safety_config
mock_src_agent = type(sys)("agent")
mock_src.agent = mock_src_agent
sys.modules["src.agent"] = mock_src_agent

# Load safety_config as src.agent.safety_config
safety_config_path = src_path / "agent" / "safety_config.py"
spec_config = importlib.util.spec_from_file_location("src.agent.safety_config", safety_config_path)
safety_config_module_src = importlib.util.module_from_spec(spec_config)
sys.modules["src.agent.safety_config"] = safety_config_module_src
spec_config.loader.exec_module(safety_config_module_src)

# Also link api module for database imports
mock_src.api = type(sys)("api")
mock_src.api.database = type(sys)("database")
mock_src.api.database.models = models_module
sys.modules["src.api"] = mock_src.api
sys.modules["src.api.database"] = mock_src.api.database
sys.modules["src.api.database.models"] = models_module

# Now load safety_manager as src.agent.safety_manager
safety_manager_path = src_path / "agent" / "safety_manager.py"
spec = importlib.util.spec_from_file_location("src.agent.safety_manager", safety_manager_path)
safety_manager_module = importlib.util.module_from_spec(spec)
sys.modules["src.agent.safety_manager"] = safety_manager_module
spec.loader.exec_module(safety_manager_module)

SafetyManager = safety_manager_module.SafetyManager
SafetyStatus = safety_manager_module.SafetyStatus

# Load config as src.agent.config for broker_manager imports
config_path = src_path / "agent" / "config.py"
spec_config_src = importlib.util.spec_from_file_location("src.agent.config", config_path)
config_module_src = importlib.util.module_from_spec(spec_config_src)
sys.modules["src.agent.config"] = config_module_src
spec_config_src.loader.exec_module(config_module_src)

# Mock trading.brokers dependencies for broker_manager
mock_brokers = type(sys)("brokers")
mock_brokers_base = type(sys)("base")
mock_brokers_mt5 = type(sys)("mt5")

# Create enum-like mock for BrokerType
class MockBrokerType:
    MT5 = "mt5"
    ALPACA = "alpaca"

# Create enum-like mock for ConnectionStatus
class MockConnectionStatus:
    CONNECTED = "connected"
    CONNECTING = "connecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"

# Create mock classes that broker_manager will import
mock_brokers_base.BrokerConfig = Mock
mock_brokers_base.BrokerType = MockBrokerType
mock_brokers_base.ConnectionStatus = MockConnectionStatus
mock_brokers_base.BrokerError = type("BrokerError", (Exception,), {})
mock_brokers_base.AuthenticationError = type("AuthenticationError", (Exception,), {})
mock_brokers_base.ConnectionError = type("ConnectionError", (Exception,), {})
mock_brokers_mt5.MT5Broker = Mock

# Assemble broker module hierarchy
mock_brokers.base = mock_brokers_base
mock_brokers.mt5 = mock_brokers_mt5
mock_trading.brokers = mock_brokers

# Register broker mocks
sys.modules["src.trading.brokers"] = mock_brokers
sys.modules["src.trading.brokers.base"] = mock_brokers_base
sys.modules["src.trading.brokers.mt5"] = mock_brokers_mt5

# Load circuit_breaker module as src.agent.circuit_breaker for broker_manager imports
circuit_breaker_path = src_path / "agent" / "circuit_breaker.py"
spec_cb = importlib.util.spec_from_file_location("src.agent.circuit_breaker", circuit_breaker_path)
circuit_breaker_module = importlib.util.module_from_spec(spec_cb)
sys.modules["src.agent.circuit_breaker"] = circuit_breaker_module
spec_cb.loader.exec_module(circuit_breaker_module)

CircuitBreaker = circuit_breaker_module.CircuitBreaker
CircuitBreakerConfig = circuit_breaker_module.CircuitBreakerConfig
CircuitOpenError = circuit_breaker_module.CircuitOpenError
CircuitState = circuit_breaker_module.CircuitState

# Now load broker_manager as src.agent.broker_manager
broker_manager_path = src_path / "agent" / "broker_manager.py"
spec_bm = importlib.util.spec_from_file_location("src.agent.broker_manager", broker_manager_path)
broker_manager_module = importlib.util.module_from_spec(spec_bm)
sys.modules["src.agent.broker_manager"] = broker_manager_module
spec_bm.loader.exec_module(broker_manager_module)

BrokerManager = broker_manager_module.BrokerManager

# Mock database.session for runner
mock_db_session = type(sys)("session")
mock_db_session.get_session = Mock
sys.modules["src.api.database.session"] = mock_db_session

# Mock agent modules needed by runner
# Load command_handler, state_manager, trading_cycle, trade_executor, metrics
# These are mocked for now as they have complex dependencies
mock_src_agent.command_handler = type(sys)("command_handler")
mock_src_agent.command_handler.CommandHandler = Mock
sys.modules["src.agent.command_handler"] = mock_src_agent.command_handler

mock_src_agent.state_manager = type(sys)("state_manager")
mock_src_agent.state_manager.StateManager = Mock
sys.modules["src.agent.state_manager"] = mock_src_agent.state_manager

mock_src_agent.trading_cycle = type(sys)("trading_cycle")
mock_src_agent.trading_cycle.TradingCycle = Mock
sys.modules["src.agent.trading_cycle"] = mock_src_agent.trading_cycle

mock_src_agent.trade_executor = type(sys)("trade_executor")
mock_src_agent.trade_executor.TradeExecutor = Mock
sys.modules["src.agent.trade_executor"] = mock_src_agent.trade_executor

mock_src_agent.metrics = type(sys)("metrics")
mock_src_agent.metrics.agent_metrics = Mock()
mock_src_agent.metrics.CycleMetrics = Mock
sys.modules["src.agent.metrics"] = mock_src_agent.metrics

# Now load runner as src.agent.runner
runner_path = src_path / "agent" / "runner.py"
spec_runner = importlib.util.spec_from_file_location("src.agent.runner", runner_path)
runner_module = importlib.util.module_from_spec(spec_runner)
sys.modules["src.agent.runner"] = runner_module
spec_runner.loader.exec_module(runner_module)

AgentRunner = runner_module.AgentRunner
AgentStatus = runner_module.AgentStatus

# Store model service mock as module variable with different name
_mock_model_service = mock_model_service


# Pytest fixture for mock_model_service (must be named mock_model_service for tests)
@pytest.fixture
def mock_model_service():
    """Provide a fresh mock_model_service for tests that need it.

    Resets the mock state for each test.
    """
    _mock_model_service.reset_mock()
    _mock_model_service.is_initialized = False
    _mock_model_service.is_loaded = False
    _mock_model_service.initialize = Mock(return_value=True)
    return _mock_model_service


# Export all for tests
__all__ = [
    "Base",
    "AgentState",
    "AgentCommand",
    "AgentConfig",
    "SafetyConfig",
    "SafetyManager",
    "SafetyStatus",
    "BrokerManager",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitOpenError",
    "CircuitState",
    "AgentRunner",
    "AgentStatus",
    "_mock_model_service",
]
