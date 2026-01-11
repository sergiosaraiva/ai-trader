"""
Trading Robot Module.

Provides the main TradingRobot class and configuration.
"""

from .config import RobotConfig, ModelConfig, BrokerConfig, SimulationConfig
from .core import (
    TradingRobot,
    RobotStatus,
    RobotState,
    TradingCycleResult,
    run_robot,
    setup_signal_handlers,
)

__all__ = [
    # Config
    'RobotConfig',
    'ModelConfig',
    'BrokerConfig',
    'SimulationConfig',
    # Core
    'TradingRobot',
    'RobotStatus',
    'RobotState',
    'TradingCycleResult',
    'run_robot',
    'setup_signal_handlers',
]
