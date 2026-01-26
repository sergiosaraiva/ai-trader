"""Autonomous AI Trading Agent.

This module provides an autonomous trading agent that runs as a separate process
and imports backend services for predictions, trading, and data management.

The agent runs independently from the API server and can be controlled via
database commands.
"""

# Use lazy imports to avoid circular dependencies with API module
# Import only when needed to prevent eager loading of dependencies

__version__ = "1.0.0"

# Expose public API via __getattr__ for lazy loading
__all__ = [
    "AgentConfig",
    "AgentRunner",
    "AgentStatus",
    "CommandHandler",
    "StateManager",
    "TradingCycle",
    "CycleResult",
    "PredictionData",
    "SignalData",
]


def __getattr__(name):
    """Lazy import agent components."""
    if name == "AgentConfig":
        from .config import AgentConfig
        return AgentConfig
    elif name == "AgentRunner":
        from .runner import AgentRunner
        return AgentRunner
    elif name == "AgentStatus":
        from .runner import AgentStatus
        return AgentStatus
    elif name == "CommandHandler":
        from .command_handler import CommandHandler
        return CommandHandler
    elif name == "StateManager":
        from .state_manager import StateManager
        return StateManager
    elif name == "TradingCycle":
        from .trading_cycle import TradingCycle
        return TradingCycle
    elif name == "CycleResult":
        from .models import CycleResult
        return CycleResult
    elif name == "PredictionData":
        from .models import PredictionData
        return PredictionData
    elif name == "SignalData":
        from .models import SignalData
        return SignalData
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
