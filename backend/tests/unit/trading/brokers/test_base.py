"""Tests for broker base classes and interfaces."""

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from src.trading.brokers.base import (
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
)


class TestBrokerConfig:
    """Tests for BrokerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BrokerConfig(broker_type=BrokerType.ALPACA)

        assert config.broker_type == BrokerType.ALPACA
        assert config.paper is True
        assert config.timeout_seconds == 30
        assert config.max_retries == 3

    def test_config_with_credentials(self):
        """Test configuration with credentials."""
        config = BrokerConfig(
            broker_type=BrokerType.ALPACA,
            api_key="test_key",
            secret_key="test_secret",
            paper=False,
        )

        assert config.api_key == "test_key"
        assert config.secret_key == "test_secret"
        assert config.paper is False

    def test_mt5_config(self):
        """Test MT5-specific configuration."""
        config = BrokerConfig(
            broker_type=BrokerType.MT5,
            login=12345678,
            password="password",
            server="Demo-Server",
        )

        assert config.broker_type == BrokerType.MT5
        assert config.login == 12345678
        assert config.server == "Demo-Server"


class TestQuote:
    """Tests for Quote dataclass."""

    def test_quote_creation(self):
        """Test quote creation."""
        quote = Quote(
            symbol="EURUSD",
            bid=1.1000,
            ask=1.1002,
            bid_size=100000.0,
            ask_size=100000.0,
            last=1.1001,
            last_size=10000.0,
            timestamp=datetime.now(timezone.utc),
        )

        assert quote.symbol == "EURUSD"
        assert quote.bid == 1.1000
        assert quote.ask == 1.1002

    def test_mid_price(self):
        """Test mid price calculation."""
        quote = Quote(
            symbol="EURUSD",
            bid=1.1000,
            ask=1.1002,
            bid_size=0.0,
            ask_size=0.0,
            last=0.0,
            last_size=0.0,
            timestamp=datetime.now(timezone.utc),
        )

        assert quote.mid == 1.1001

    def test_spread(self):
        """Test spread calculation."""
        quote = Quote(
            symbol="EURUSD",
            bid=1.1000,
            ask=1.1002,
            bid_size=0.0,
            ask_size=0.0,
            last=0.0,
            last_size=0.0,
            timestamp=datetime.now(timezone.utc),
        )

        assert quote.spread == pytest.approx(0.0002, rel=1e-5)

    def test_spread_pct(self):
        """Test spread percentage calculation."""
        quote = Quote(
            symbol="EURUSD",
            bid=1.1000,
            ask=1.1002,
            bid_size=0.0,
            ask_size=0.0,
            last=0.0,
            last_size=0.0,
            timestamp=datetime.now(timezone.utc),
        )

        expected = (0.0002 / 1.1001) * 100
        assert quote.spread_pct == pytest.approx(expected, rel=1e-3)

    def test_to_dict(self):
        """Test quote to dictionary conversion."""
        ts = datetime.now(timezone.utc)
        quote = Quote(
            symbol="EURUSD",
            bid=1.1000,
            ask=1.1002,
            bid_size=100.0,
            ask_size=100.0,
            last=1.1001,
            last_size=10.0,
            timestamp=ts,
        )

        d = quote.to_dict()

        assert d["symbol"] == "EURUSD"
        assert d["bid"] == 1.1000
        assert d["ask"] == 1.1002
        assert d["mid"] == 1.1001
        assert "spread" in d


class TestAccountInfo:
    """Tests for AccountInfo dataclass."""

    def test_account_creation(self):
        """Test account info creation."""
        account = AccountInfo(
            account_id="ACC123",
            currency="USD",
            balance=100000.0,
            equity=105000.0,
            margin_used=10000.0,
            margin_available=90000.0,
            buying_power=400000.0,
            cash=100000.0,
        )

        assert account.account_id == "ACC123"
        assert account.currency == "USD"
        assert account.balance == 100000.0
        assert account.equity == 105000.0

    def test_margin_level(self):
        """Test margin level calculation."""
        account = AccountInfo(
            account_id="ACC123",
            currency="USD",
            balance=100000.0,
            equity=105000.0,
            margin_used=10000.0,
            margin_available=90000.0,
            buying_power=400000.0,
            cash=100000.0,
        )

        assert account.margin_level == 1050.0  # (105000 / 10000) * 100

    def test_margin_level_no_margin(self):
        """Test margin level when no margin used."""
        account = AccountInfo(
            account_id="ACC123",
            currency="USD",
            balance=100000.0,
            equity=100000.0,
            margin_used=0.0,
            margin_available=100000.0,
            buying_power=400000.0,
            cash=100000.0,
        )

        assert account.margin_level == float('inf')

    def test_to_dict(self):
        """Test account info to dictionary conversion."""
        account = AccountInfo(
            account_id="ACC123",
            currency="USD",
            balance=100000.0,
            equity=105000.0,
            margin_used=10000.0,
            margin_available=90000.0,
            buying_power=400000.0,
            cash=100000.0,
        )

        d = account.to_dict()

        assert d["account_id"] == "ACC123"
        assert d["balance"] == 100000.0
        assert d["margin_level"] == 1050.0


class TestBrokerOrder:
    """Tests for BrokerOrder dataclass."""

    def test_order_creation(self):
        """Test broker order creation."""
        order = BrokerOrder(
            order_id="ORD123",
            client_order_id="CLIENT123",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=100.0,
            filled_quantity=0.0,
            remaining_quantity=100.0,
        )

        assert order.order_id == "ORD123"
        assert order.symbol == "EURUSD"
        assert order.side == "buy"

    def test_is_filled(self):
        """Test is_filled property."""
        order = BrokerOrder(
            order_id="ORD123",
            client_order_id="CLIENT123",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=100.0,
            filled_quantity=100.0,
            remaining_quantity=0.0,
            status="filled",
        )

        assert order.is_filled is True

    def test_is_cancelled(self):
        """Test is_cancelled property."""
        order = BrokerOrder(
            order_id="ORD123",
            client_order_id="CLIENT123",
            symbol="EURUSD",
            side="buy",
            order_type="market",
            quantity=100.0,
            filled_quantity=0.0,
            remaining_quantity=100.0,
            status="cancelled",
        )

        assert order.is_cancelled is True

    def test_is_active(self):
        """Test is_active property."""
        order = BrokerOrder(
            order_id="ORD123",
            client_order_id="CLIENT123",
            symbol="EURUSD",
            side="buy",
            order_type="limit",
            quantity=100.0,
            filled_quantity=0.0,
            remaining_quantity=100.0,
            status="accepted",
        )

        assert order.is_active is True


class TestBrokerPosition:
    """Tests for BrokerPosition dataclass."""

    def test_position_creation(self):
        """Test broker position creation."""
        position = BrokerPosition(
            symbol="EURUSD",
            quantity=10000.0,
            side="long",
            average_entry_price=1.1000,
            current_price=1.1050,
            market_value=11050.0,
            cost_basis=11000.0,
            unrealized_pnl=50.0,
            unrealized_pnl_pct=0.45,
        )

        assert position.symbol == "EURUSD"
        assert position.quantity == 10000.0
        assert position.side == "long"
        assert position.unrealized_pnl == 50.0

    def test_to_dict(self):
        """Test position to dictionary conversion."""
        position = BrokerPosition(
            symbol="EURUSD",
            quantity=10000.0,
            side="long",
            average_entry_price=1.1000,
            current_price=1.1050,
            market_value=11050.0,
            cost_basis=11000.0,
            unrealized_pnl=50.0,
            unrealized_pnl_pct=0.45,
        )

        d = position.to_dict()

        assert d["symbol"] == "EURUSD"
        assert d["quantity"] == 10000.0
        assert d["unrealized_pnl"] == 50.0


class TestBrokerErrors:
    """Tests for broker error classes."""

    def test_broker_error(self):
        """Test base broker error."""
        error = BrokerError("Test error")
        assert str(error) == "Test error"

    def test_authentication_error(self):
        """Test authentication error."""
        error = AuthenticationError("Invalid credentials")
        assert str(error) == "Invalid credentials"
        assert isinstance(error, BrokerError)

    def test_order_rejected_error(self):
        """Test order rejected error."""
        error = OrderRejectedError(
            "Order rejected",
            reason="Insufficient margin",
            order_id="ORD123",
        )
        assert str(error) == "Order rejected"
        assert error.reason == "Insufficient margin"
        assert error.order_id == "ORD123"

    def test_insufficient_funds_error(self):
        """Test insufficient funds error."""
        error = InsufficientFundsError(
            "Insufficient funds",
            required=10000.0,
            available=5000.0,
        )
        assert str(error) == "Insufficient funds"
        assert error.required == 10000.0
        assert error.available == 5000.0


class TestConnectionStatus:
    """Tests for ConnectionStatus enum."""

    def test_all_statuses(self):
        """Test all connection statuses exist."""
        assert ConnectionStatus.DISCONNECTED.value == "disconnected"
        assert ConnectionStatus.CONNECTING.value == "connecting"
        assert ConnectionStatus.CONNECTED.value == "connected"
        assert ConnectionStatus.RECONNECTING.value == "reconnecting"
        assert ConnectionStatus.ERROR.value == "error"


class TestBrokerType:
    """Tests for BrokerType enum."""

    def test_all_types(self):
        """Test all broker types exist."""
        assert BrokerType.ALPACA.value == "alpaca"
        assert BrokerType.MT5.value == "mt5"
        assert BrokerType.SIMULATION.value == "simulation"
