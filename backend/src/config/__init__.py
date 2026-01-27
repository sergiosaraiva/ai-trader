"""Configuration management module."""

from .settings import Settings, get_settings
from .model_config import XGBoostHyperparameters, ModelHyperparameters
from .profile_loader import (
    ProfileLoader,
    ProfileConfig,
    load_profile,
    get_timeframe_config,
    get_indicator_config,
)
from .trading_config import (
    TradingConfig,
    TradingParameters,
    ModelParameters,
    RiskParameters,
    SystemParameters,
    trading_config,
)

__all__ = [
    "Settings",
    "get_settings",
    "XGBoostHyperparameters",
    "ModelHyperparameters",
    "ProfileLoader",
    "ProfileConfig",
    "load_profile",
    "get_timeframe_config",
    "get_indicator_config",
    "TradingConfig",
    "TradingParameters",
    "ModelParameters",
    "RiskParameters",
    "SystemParameters",
    "trading_config",
    "get_config",
    "get_trading_params",
    "get_model_params",
    "get_risk_params",
    "get_system_params",
]


def get_config() -> TradingConfig:
    """Get the global configuration instance.

    Returns:
        TradingConfig singleton instance
    """
    return trading_config


def get_trading_params() -> TradingParameters:
    """Get trading parameters.

    Returns:
        TradingParameters instance
    """
    return trading_config.trading


def get_model_params() -> ModelParameters:
    """Get model parameters.

    Returns:
        ModelParameters instance
    """
    return trading_config.model


def get_risk_params() -> RiskParameters:
    """Get risk parameters.

    Returns:
        RiskParameters instance
    """
    return trading_config.risk


def get_system_params() -> SystemParameters:
    """Get system parameters.

    Returns:
        SystemParameters instance
    """
    return trading_config.system
