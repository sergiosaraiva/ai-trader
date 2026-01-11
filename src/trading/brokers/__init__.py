"""
Broker Adapters Module.

Provides broker-specific implementations for trading execution.
"""

from .base import (
    BrokerAdapter,
    BrokerConfig,
    BrokerType,
    Quote,
    AccountInfo,
    BrokerOrder,
    BrokerPosition,
    ConnectionStatus,
    BrokerError,
    AuthenticationError,
    OrderRejectedError,
    InsufficientFundsError,
    ConnectionError,
)
from .alpaca import AlpacaBroker
from .mt5 import MT5Broker

__all__ = [
    # Base
    "BrokerAdapter",
    "BrokerConfig",
    "BrokerType",
    "Quote",
    "AccountInfo",
    "BrokerOrder",
    "BrokerPosition",
    "ConnectionStatus",
    "BrokerError",
    "AuthenticationError",
    "OrderRejectedError",
    "InsufficientFundsError",
    "ConnectionError",
    # Implementations
    "AlpacaBroker",
    "MT5Broker",
]


def create_broker(config: BrokerConfig) -> BrokerAdapter:
    """
    Factory function to create broker adapter.

    Args:
        config: Broker configuration

    Returns:
        Appropriate broker adapter instance

    Raises:
        ValueError: If broker type not supported
    """
    if config.broker_type == BrokerType.ALPACA:
        return AlpacaBroker(config)
    elif config.broker_type == BrokerType.MT5:
        return MT5Broker(config)
    else:
        raise ValueError(f"Unsupported broker type: {config.broker_type}")
