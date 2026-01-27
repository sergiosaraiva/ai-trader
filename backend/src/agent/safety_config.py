"""Safety configuration for the trading agent.

DEPRECATED: This module now delegates to the centralized TradingConfig system.
Use trading_config.risk and trading_config.agent directly for new code.

This module provides backward compatibility by mapping the old SafetyConfig
interface to the new centralized configuration system.
"""

import logging
from typing import Dict, Any

from ..config import trading_config

logger = logging.getLogger(__name__)


def get_safety_config() -> Dict[str, Any]:
    """Get safety configuration from centralized TradingConfig.

    This function replaces the old SafetyConfig dataclass by reading
    from the centralized configuration system.

    Returns:
        Dictionary with all safety parameters

    Example:
        config = get_safety_config()
        max_dd = config["max_drawdown_percent"]  # 15.0 from TradingConfig.risk
    """
    # Map centralized config to legacy safety config format
    return {
        # Consecutive Loss Breaker
        "max_consecutive_losses": trading_config.risk.max_consecutive_losses,
        "consecutive_loss_action": "pause",  # Default action

        # Drawdown Breaker
        "max_drawdown_percent": trading_config.risk.max_drawdown_percent,  # CRITICAL: 15.0
        "drawdown_action": "stop",  # Default action

        # Daily Loss Limit (via Kill Switch)
        "max_daily_loss_percent": trading_config.risk.max_daily_loss_percent,
        "max_daily_loss_amount": trading_config.risk.max_daily_loss_amount,
        "daily_loss_action": "pause",  # Default action

        # Model Degradation Breaker
        "enable_model_degradation": trading_config.risk.enable_model_degradation,
        "min_win_rate": trading_config.risk.min_win_rate,
        "degradation_window": trading_config.risk.degradation_window,

        # Kill Switch
        "require_token_for_reset": True,  # Always require authorization
        "auto_reset_next_day": True,  # Auto-reset at new trading day
        "max_disconnection_seconds": trading_config.agent.max_reconnect_delay,

        # Daily Trade Limits
        "max_daily_trades": trading_config.risk.max_trades_per_day,
        "max_trades_per_hour": trading_config.risk.max_trades_per_hour,
    }


def validate_safety_config() -> None:
    """Validate safety configuration from centralized config.

    Raises:
        ValueError: If any configuration value is invalid
    """
    # Validate through centralized config system
    errors = trading_config.validate()
    if errors:
        error_msg = f"Safety configuration validation failed: {'; '.join(errors)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Safety configuration validated via centralized config")


def get_safety_config_dict() -> Dict[str, Any]:
    """Get safety config as dictionary (for serialization).

    Returns:
        Dictionary with all safety parameters
    """
    return get_safety_config()


# Backward compatibility: Allow importing safety config as dict
def load_safety_config_from_trading_config() -> Dict[str, Any]:
    """Load safety configuration from centralized TradingConfig.

    This is the recommended way to access safety configuration.

    Returns:
        Dictionary with safety parameters from TradingConfig
    """
    config = get_safety_config()
    validate_safety_config()
    return config


# Module-level documentation for migration
__doc_migration__ = """
Migration Guide: SafetyConfig → TradingConfig
==============================================

OLD (DEPRECATED):
    from .safety_config import SafetyConfig
    config = SafetyConfig(max_drawdown_percent=10.0)
    print(config.max_drawdown_percent)

NEW (RECOMMENDED):
    from ..config import trading_config
    print(trading_config.risk.max_drawdown_percent)  # 15.0

BACKWARD COMPATIBLE:
    from .safety_config import get_safety_config
    config = get_safety_config()
    print(config["max_drawdown_percent"])  # 15.0

Field Mappings:
---------------
SafetyConfig.max_consecutive_losses     → trading_config.risk.max_consecutive_losses
SafetyConfig.max_drawdown_percent       → trading_config.risk.max_drawdown_percent (15.0)
SafetyConfig.max_daily_loss_percent     → trading_config.risk.max_daily_loss_percent
SafetyConfig.max_daily_loss_amount      → trading_config.risk.max_daily_loss_amount
SafetyConfig.min_win_rate               → trading_config.risk.min_win_rate
SafetyConfig.degradation_window         → trading_config.risk.degradation_window
SafetyConfig.max_disconnection_seconds  → trading_config.agent.max_reconnect_delay
SafetyConfig.max_daily_trades           → trading_config.risk.max_trades_per_day
SafetyConfig.max_trades_per_hour        → trading_config.risk.max_trades_per_hour

Benefits:
---------
1. Single source of truth for all configuration
2. Hot reload capability without service restart
3. Database persistence with audit trail
4. Automatic validation across all config categories
5. Change callbacks for dependent services
"""
