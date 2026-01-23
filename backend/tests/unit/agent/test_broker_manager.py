"""Unit tests for BrokerManager.

Tests broker connection management, health monitoring, and credential security.
Critical: Never expose MT5 credentials in logs.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import asyncio
import logging

# Import from conftest
from .conftest import AgentConfig


# Create mock classes for broker types
class MockBrokerConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockBrokerType:
    MT5 = "mt5"


class MockConnectionStatus:
    CONNECTED = Mock(value="connected")
    CONNECTING = Mock(value="connecting")
    DISCONNECTED = Mock(value="disconnected")

    def __init__(self, value):
        self.value = value


class MockBrokerError(Exception):
    pass


class MockAuthenticationError(MockBrokerError):
    pass


class MockConnectionError(MockBrokerError):
    pass


class MockAccountInfo:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return {
            "account_id": getattr(self, "account_id", ""),
            "currency": getattr(self, "currency", "USD"),
            "balance": getattr(self, "balance", 0.0),
            "equity": getattr(self, "equity", 0.0),
            "margin_used": getattr(self, "margin_used", 0.0),
            "margin_available": getattr(self, "margin_available", 0.0),
            "buying_power": getattr(self, "buying_power", 0.0),
            "cash": getattr(self, "cash", 0.0),
        }


class MockBrokerPosition:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self):
        return {
            "symbol": getattr(self, "symbol", ""),
            "quantity": getattr(self, "quantity", 0.0),
            "side": getattr(self, "side", "long"),
            "average_entry_price": getattr(self, "average_entry_price", 0.0),
            "current_price": getattr(self, "current_price", 0.0),
            "market_value": getattr(self, "market_value", 0.0),
            "cost_basis": getattr(self, "cost_basis", 0.0),
            "unrealized_pnl": getattr(self, "unrealized_pnl", 0.0),
            "unrealized_pnl_pct": getattr(self, "unrealized_pnl_pct", 0.0),
        }


@pytest.fixture
def agent_config():
    """Create agent config for testing."""
    return AgentConfig(
        mode="paper",
        mt5_login=12345678,
        mt5_password="test_password_secret",
        mt5_server="TestBroker-Demo",
    )


@pytest.fixture
def mock_mt5_broker():
    """Create a mock MT5 broker."""
    broker = AsyncMock()
    broker.is_connected = True
    broker.status = MockConnectionStatus.CONNECTED

    # Mock account info
    broker.get_account = AsyncMock(return_value=MockAccountInfo(
        account_id="12345678",
        currency="USD",
        balance=100000.0,
        equity=100000.0,
        margin_used=0.0,
        margin_available=100000.0,
        buying_power=100000.0,
        cash=100000.0,
    ))

    # Mock positions
    broker.get_positions = AsyncMock(return_value=[])

    # Mock connect/disconnect
    broker.connect = AsyncMock(return_value=True)
    broker.disconnect = AsyncMock()

    return broker


class TestBrokerManagerConnection:
    """Test broker connection management."""

    @pytest.mark.asyncio
    async def test_connect_successfully(self, agent_config, mock_mt5_broker):
        """Test successful connection to MT5."""
        with patch("src.agent.broker_manager.MT5Broker", return_value=mock_mt5_broker):
            from src.agent.broker_manager import BrokerManager
            manager = BrokerManager(agent_config)
            success = await manager.connect()

        assert success is True
        assert manager.is_connected() is True
        mock_mt5_broker.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, agent_config, mock_mt5_broker):
        """Test connect when already connected."""
        with patch("src.agent.broker_manager.MT5Broker", return_value=mock_mt5_broker):
            from src.agent.broker_manager import BrokerManager
            manager = BrokerManager(agent_config)
            manager.broker = mock_mt5_broker
            manager._connected = True

            success = await manager.connect()

        assert success is True
        # Should not call connect again
        mock_mt5_broker.connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_failure(self, agent_config):
        """Test connection failure handling."""
        mock_broker = AsyncMock()
        mock_broker.connect = AsyncMock(return_value=False)
        mock_broker.is_connected = False  # Set explicitly for is_connected() check

        with patch("src.agent.broker_manager.MT5Broker", return_value=mock_broker):
            from src.agent.broker_manager import BrokerManager
            manager = BrokerManager(agent_config)
            success = await manager.connect()

        assert success is False
        assert manager._connected is False

    @pytest.mark.asyncio
    async def test_connect_authentication_error(self, agent_config):
        """Test authentication error handling."""
        mock_broker = AsyncMock()
        mock_broker.connect = AsyncMock(side_effect=MockAuthenticationError("Invalid credentials"))
        mock_broker.is_connected = False  # Set explicitly for is_connected() check

        with patch("src.agent.broker_manager.MT5Broker", return_value=mock_broker):
            with patch("src.agent.broker_manager.AuthenticationError", MockAuthenticationError):
                from src.agent.broker_manager import BrokerManager
                manager = BrokerManager(agent_config)
                success = await manager.connect()

        assert success is False
        assert manager._connected is False

    @pytest.mark.asyncio
    async def test_disconnect(self, agent_config, mock_mt5_broker):
        """Test disconnection."""
        with patch("src.agent.broker_manager.MT5Broker", return_value=mock_mt5_broker):
            from src.agent.broker_manager import BrokerManager
            manager = BrokerManager(agent_config)
            manager.broker = mock_mt5_broker
            manager._connected = True

            await manager.disconnect()

        assert manager._connected is False
        mock_mt5_broker.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconnect_success(self, agent_config, mock_mt5_broker):
        """Test successful reconnection."""
        with patch("src.agent.broker_manager.MT5Broker", return_value=mock_mt5_broker):
            from src.agent.broker_manager import BrokerManager
            manager = BrokerManager(agent_config)
            manager._reconnect_delay = 0.01  # Speed up test

            success = await manager.reconnect()

        assert success is True
        assert manager._reconnect_attempts == 0  # Reset on success

    @pytest.mark.asyncio
    async def test_reconnect_max_attempts(self, agent_config):
        """Test max reconnection attempts exceeded."""
        # Set config to have low max reconnect attempts
        agent_config.max_reconnect_attempts = 3

        with patch("src.agent.broker_manager.MT5Broker"):
            from src.agent.broker_manager import BrokerManager
            manager = BrokerManager(agent_config)
            manager._reconnect_attempts = 3  # Already at max

            success = await manager.reconnect()

        assert success is False
        # Should not increment past max (returns early)

    @pytest.mark.asyncio
    async def test_get_account_info_success(self, agent_config, mock_mt5_broker):
        """Test getting account info successfully."""
        with patch("src.agent.broker_manager.MT5Broker", return_value=mock_mt5_broker):
            from src.agent.broker_manager import BrokerManager
            manager = BrokerManager(agent_config)
            manager.broker = mock_mt5_broker

            account_info = await manager.get_account_info()

        assert account_info is not None
        assert account_info["account_id"] == "12345678"
        assert account_info["balance"] == 100000.0
        assert account_info["equity"] == 100000.0

    @pytest.mark.asyncio
    async def test_get_account_info_not_connected(self, agent_config):
        """Test getting account info when not connected."""
        with patch("src.agent.broker_manager.MT5Broker"):
            from src.agent.broker_manager import BrokerManager
            manager = BrokerManager(agent_config)

            account_info = await manager.get_account_info()

        assert account_info is None

    @pytest.mark.asyncio
    async def test_get_open_positions_success(self, agent_config, mock_mt5_broker):
        """Test getting open positions successfully."""
        # Add mock position
        mock_position = MockBrokerPosition(
            symbol="EURUSD",
            quantity=0.1,
            side="long",
            average_entry_price=1.08500,
            current_price=1.08600,
            market_value=10860.0,
            cost_basis=10850.0,
            unrealized_pnl=10.0,
            unrealized_pnl_pct=0.092,
        )
        mock_mt5_broker.get_positions = AsyncMock(return_value=[mock_position])

        with patch("src.agent.broker_manager.MT5Broker", return_value=mock_mt5_broker):
            from src.agent.broker_manager import BrokerManager
            manager = BrokerManager(agent_config)
            manager.broker = mock_mt5_broker

            positions = await manager.get_open_positions()

        assert len(positions) == 1
        assert positions[0]["symbol"] == "EURUSD"
        assert positions[0]["quantity"] == 0.1

    @pytest.mark.asyncio
    async def test_health_check_when_connected(self, agent_config, mock_mt5_broker):
        """Test health check when connected."""
        with patch("src.agent.broker_manager.MT5Broker", return_value=mock_mt5_broker):
            from src.agent.broker_manager import BrokerManager
            manager = BrokerManager(agent_config)
            manager.broker = mock_mt5_broker

            healthy = await manager.check_connection_health()

        assert healthy is True

    @pytest.mark.asyncio
    async def test_health_check_when_disconnected(self, agent_config):
        """Test health check when disconnected."""
        with patch("src.agent.broker_manager.MT5Broker"):
            from src.agent.broker_manager import BrokerManager
            manager = BrokerManager(agent_config)

            healthy = await manager.check_connection_health()

        assert healthy is False

    @pytest.mark.asyncio
    async def test_credentials_not_in_normal_logs(self, agent_config, mock_mt5_broker, caplog):
        """CRITICAL: MT5 credentials must never appear in normal operation logs.

        Note: This tests that credentials from config don't appear in log messages
        during normal connect/disconnect operations. Exception messages from external
        libraries are a separate concern.
        """
        caplog.clear()

        with patch("src.agent.broker_manager.MT5Broker", return_value=mock_mt5_broker):
            from src.agent.broker_manager import BrokerManager
            manager = BrokerManager(agent_config)
            await manager.connect()
            await manager.disconnect()

        # Check all log messages
        for record in caplog.records:
            # Password should NOT appear in normal operation logs
            assert agent_config.mt5_password not in record.message

    def test_get_connection_stats(self, agent_config, mock_mt5_broker):
        """Test getting connection statistics."""
        with patch("src.agent.broker_manager.MT5Broker", return_value=mock_mt5_broker):
            from src.agent.broker_manager import BrokerManager
            manager = BrokerManager(agent_config)
            manager.broker = mock_mt5_broker
            manager._connected = True
            manager._reconnect_attempts = 2
            manager._last_connection_time = datetime(2024, 1, 15, 10, 0, 0)

            stats = manager.get_connection_stats()

        assert stats["connected"] is True
        assert stats["reconnect_attempts"] == 2
        assert stats["mode"] == "paper"
        assert "2024-01-15" in stats["last_connection"]
