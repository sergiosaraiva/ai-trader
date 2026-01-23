"""Broker connection manager for the trading agent.

Manages MT5 broker connection lifecycle with automatic reconnection
and health monitoring.
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..trading.brokers.base import (
    BrokerConfig,
    BrokerType,
    ConnectionStatus,
    BrokerError,
    AuthenticationError,
    ConnectionError as BrokerConnectionError,
)
from ..trading.brokers.mt5 import MT5Broker
from .config import AgentConfig
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
)

logger = logging.getLogger(__name__)


class BrokerManager:
    """Manages MT5 broker connection for the agent.

    Features:
    - Automatic connection management
    - Reconnection on connection loss
    - Health monitoring
    - Account information access
    - Position querying
    """

    def __init__(self, config: AgentConfig):
        """Initialize broker manager.

        Args:
            config: Agent configuration with MT5 credentials
        """
        self.config = config
        self.broker: Optional[MT5Broker] = None
        self._connected = False
        self._reconnect_attempts = 0
        self._last_connection_time: Optional[datetime] = None
        self._current_reconnect_delay: float = config.initial_reconnect_delay

        # Circuit breaker for broker operations
        self._circuit_breaker = CircuitBreaker(
            CircuitBreakerConfig(
                name="broker",
                failure_threshold=3,
                success_threshold=2,
                timeout_seconds=config.broker_timeout_seconds,
                half_open_max_calls=2,
            )
        )

    async def connect(self) -> bool:
        """Connect to MT5 broker.

        Returns:
            True if connected successfully, False otherwise
        """
        if self._connected and self.broker and self.broker.is_connected:
            logger.info("Broker already connected")
            return True

        logger.info(f"Connecting to MT5 broker ({self.config.mode} mode)...")

        try:
            # Create broker configuration
            broker_config = BrokerConfig(
                broker_type=BrokerType.MT5,
                paper=(self.config.mode == "paper"),
                login=self.config.mt5_login or 0,
                password=self.config.mt5_password or "",
                server=self.config.mt5_server or "",
            )

            # Create broker instance
            self.broker = MT5Broker(broker_config)

            # Connect
            success = await self.broker.connect()

            if success:
                self._connected = True
                self._reconnect_attempts = 0
                self._current_reconnect_delay = self.config.initial_reconnect_delay  # Reset backoff
                self._last_connection_time = datetime.now()

                # Get account info for confirmation
                account = await self.broker.get_account()
                logger.info(
                    f"Connected to MT5 broker successfully - "
                    f"Account: {account.account_id}, "
                    f"Balance: {account.balance} {account.currency}, "
                    f"Equity: {account.equity}"
                )
                return True
            else:
                logger.error("Failed to connect to MT5 broker")
                return False

        except AuthenticationError as e:
            logger.error(f"MT5 authentication failed: {e}")
            return False

        except BrokerConnectionError as e:
            logger.error(f"MT5 connection error: {e}")
            return False

        except Exception as e:
            logger.error(f"Unexpected error connecting to MT5: {e}", exc_info=True)
            return False

    async def disconnect(self) -> None:
        """Disconnect from MT5 broker."""
        if not self.broker:
            return

        logger.info("Disconnecting from MT5 broker...")

        try:
            await self.broker.disconnect()
            self._connected = False
            logger.info("Disconnected from MT5 broker")

        except Exception as e:
            logger.error(f"Error disconnecting from MT5: {e}")

    async def reconnect(self) -> bool:
        """Attempt to reconnect to MT5 broker with exponential backoff.

        Uses configurable exponential backoff:
        - initial_reconnect_delay: Starting delay
        - reconnect_backoff_multiplier: Delay multiplier on each failure
        - max_reconnect_delay: Cap on maximum delay

        Returns:
            True if reconnected successfully, False otherwise
        """
        if self._reconnect_attempts >= self.config.max_reconnect_attempts:
            logger.error(
                f"Max reconnection attempts ({self.config.max_reconnect_attempts}) reached"
            )
            return False

        self._reconnect_attempts += 1

        # Calculate delay with exponential backoff (already set from previous failure)
        current_delay = self._current_reconnect_delay

        logger.info(
            f"Attempting reconnection {self._reconnect_attempts}/"
            f"{self.config.max_reconnect_attempts} "
            f"(delay={current_delay:.1f}s)..."
        )

        # Disconnect first if connected
        if self.broker:
            try:
                await self.broker.disconnect()
            except Exception:
                pass

        # Wait before reconnecting with current delay
        await asyncio.sleep(current_delay)

        # Try to reconnect
        success = await self.connect()

        if not success:
            # Increase delay for next attempt (exponential backoff)
            self._current_reconnect_delay = min(
                self._current_reconnect_delay * self.config.reconnect_backoff_multiplier,
                self.config.max_reconnect_delay
            )
            logger.debug(f"Next reconnect delay: {self._current_reconnect_delay:.1f}s")

        return success

    def reset_reconnect_state(self) -> None:
        """Reset reconnection state for fresh start.

        Call this after successful manual intervention or when starting fresh.
        """
        self._reconnect_attempts = 0
        self._current_reconnect_delay = self.config.initial_reconnect_delay
        logger.info("Reconnection state reset")

    def is_connected(self) -> bool:
        """Check if connected to MT5 broker.

        Returns:
            True if connected, False otherwise
        """
        if not self.broker:
            return False

        # Check broker connection status
        return self.broker.is_connected

    async def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get current account information.

        Returns:
            Dictionary with account info, or None if not connected
        """
        if not self.is_connected():
            logger.warning("Cannot get account info - not connected")
            return None

        try:
            account = await self.broker.get_account()
            return account.to_dict()

        except BrokerError as e:
            logger.error(f"Failed to get account info: {e}")
            return None

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get list of open positions.

        Returns:
            List of position dictionaries, empty list if none or error
        """
        if not self.is_connected():
            logger.warning("Cannot get positions - not connected")
            return []

        try:
            positions = await self.broker.get_positions()
            return [p.to_dict() for p in positions]

        except BrokerError as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def check_connection_health(self) -> bool:
        """Check if connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        if not self.is_connected():
            return False

        try:
            # Try to get account info as health check
            account = await self.broker.get_account()
            return account is not None

        except Exception as e:
            logger.warning(f"Connection health check failed: {e}")
            return False

    async def get_current_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current price tick for a symbol.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")

        Returns:
            Dictionary with bid/ask prices, or None if unavailable
        """
        if not self.is_connected():
            logger.warning("Cannot get price - not connected")
            return None

        try:
            tick = await self.broker.get_tick(symbol)
            if tick:
                return {
                    "symbol": symbol,
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "last": tick.last if hasattr(tick, "last") else tick.bid,
                    "time": tick.time.isoformat() if hasattr(tick, "time") and tick.time else None,
                }
            return None

        except BrokerError as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics.

        Returns:
            Dictionary with connection stats
        """
        return {
            "connected": self._connected,
            "broker_status": self.broker.status.value if self.broker else "no_broker",
            "reconnect_attempts": self._reconnect_attempts,
            "last_connection": self._last_connection_time.isoformat() if self._last_connection_time else None,
            "mode": self.config.mode,
            "circuit_breaker": self._circuit_breaker.get_status(),
        }

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status.

        Returns:
            Dictionary with circuit breaker status
        """
        return self._circuit_breaker.get_status()

    def reset_circuit_breaker(self) -> None:
        """Reset the broker circuit breaker.

        Call this after manual intervention to restore operations.
        """
        self._circuit_breaker.reset()
        logger.info("Broker circuit breaker reset")
