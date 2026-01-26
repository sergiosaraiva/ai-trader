"""
Broker Adapter Base Classes and Interfaces.

Defines the abstract interface for broker implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, AsyncIterator, Callable
import logging

logger = logging.getLogger(__name__)


class BrokerType(Enum):
    """Supported broker types."""
    ALPACA = "alpaca"
    MT5 = "mt5"
    INTERACTIVE_BROKERS = "interactive_brokers"
    SIMULATION = "simulation"


class ConnectionStatus(Enum):
    """Broker connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class BrokerError(Exception):
    """Base exception for broker errors."""
    pass


class AuthenticationError(BrokerError):
    """Authentication failed."""
    pass


class OrderRejectedError(BrokerError):
    """Order was rejected by broker."""
    def __init__(self, message: str, reason: str = "", order_id: str = ""):
        super().__init__(message)
        self.reason = reason
        self.order_id = order_id


class InsufficientFundsError(BrokerError):
    """Insufficient funds for order."""
    def __init__(self, message: str, required: float = 0.0, available: float = 0.0):
        super().__init__(message)
        self.required = required
        self.available = available


class ConnectionError(BrokerError):
    """Connection to broker failed."""
    pass


@dataclass
class BrokerConfig:
    """Configuration for broker connection."""
    broker_type: BrokerType
    api_key: str = ""
    secret_key: str = ""
    paper: bool = True
    base_url: Optional[str] = None

    # MT5 specific
    login: int = 0
    password: str = ""
    server: str = ""
    path: str = ""

    # Connection settings
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    # Rate limiting
    max_requests_per_second: int = 10

    # Additional settings
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Quote:
    """Real-time market quote."""
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    last: float
    last_size: float
    timestamp: datetime
    volume: float = 0.0

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid price."""
        if self.mid == 0:
            return 0.0
        return (self.spread / self.mid) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "last": self.last,
            "last_size": self.last_size,
            "timestamp": self.timestamp.isoformat(),
            "volume": self.volume,
            "mid": self.mid,
            "spread": self.spread,
        }


@dataclass
class AccountInfo:
    """Broker account information."""
    account_id: str
    currency: str
    balance: float
    equity: float
    margin_used: float
    margin_available: float
    buying_power: float
    cash: float

    # Day trading specific
    day_trades_remaining: int = -1  # -1 = unlimited
    pattern_day_trader: bool = False

    # Status
    trading_blocked: bool = False
    transfers_blocked: bool = False
    account_blocked: bool = False

    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def margin_level(self) -> float:
        """Margin level as percentage (equity/margin_used)."""
        if self.margin_used == 0:
            return float('inf')
        return (self.equity / self.margin_used) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "account_id": self.account_id,
            "currency": self.currency,
            "balance": self.balance,
            "equity": self.equity,
            "margin_used": self.margin_used,
            "margin_available": self.margin_available,
            "buying_power": self.buying_power,
            "cash": self.cash,
            "day_trades_remaining": self.day_trades_remaining,
            "pattern_day_trader": self.pattern_day_trader,
            "trading_blocked": self.trading_blocked,
            "margin_level": self.margin_level,
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class BrokerOrder:
    """Order representation from broker."""
    order_id: str
    client_order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    order_type: str  # "market", "limit", "stop", "stop_limit"
    quantity: float
    filled_quantity: float
    remaining_quantity: float

    # Prices
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    average_fill_price: Optional[float] = None

    # Status
    status: str = "new"  # new, accepted, pending_new, filled, partially_filled, cancelled, rejected

    # Timing
    time_in_force: str = "day"  # day, gtc, ioc, fok
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None

    # Extended hours
    extended_hours: bool = False

    # Rejection info
    reject_reason: str = ""

    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled."""
        return self.status == "filled"

    @property
    def is_cancelled(self) -> bool:
        """Check if order is cancelled."""
        return self.status in ("cancelled", "rejected")

    @property
    def is_active(self) -> bool:
        """Check if order is still active."""
        return self.status in ("new", "accepted", "pending_new", "partially_filled")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "average_fill_price": self.average_fill_price,
            "status": self.status,
            "time_in_force": self.time_in_force,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "is_filled": self.is_filled,
            "is_active": self.is_active,
        }


@dataclass
class BrokerPosition:
    """Position representation from broker."""
    symbol: str
    quantity: float
    side: str  # "long" or "short"
    average_entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    realized_pnl: float = 0.0

    # Exchange info
    exchange: str = ""
    asset_class: str = ""

    # Last update
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "side": self.side,
            "average_entry_price": self.average_entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "cost_basis": self.cost_basis,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "realized_pnl": self.realized_pnl,
            "last_updated": self.last_updated.isoformat(),
        }


class BrokerAdapter(ABC):
    """
    Abstract base class for broker implementations.

    All broker-specific adapters must implement this interface.
    """

    def __init__(self, config: BrokerConfig):
        """
        Initialize broker adapter.

        Args:
            config: Broker configuration
        """
        self.config = config
        self._status = ConnectionStatus.DISCONNECTED
        self._last_error: Optional[str] = None
        self._connected_at: Optional[datetime] = None

        # Callbacks
        self._on_quote: List[Callable[[Quote], None]] = []
        self._on_order_update: List[Callable[[BrokerOrder], None]] = []
        self._on_position_update: List[Callable[[BrokerPosition], None]] = []
        self._on_connection_change: List[Callable[[ConnectionStatus], None]] = []

    @property
    def status(self) -> ConnectionStatus:
        """Current connection status."""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        return self._status == ConnectionStatus.CONNECTED

    @property
    def last_error(self) -> Optional[str]:
        """Last error message."""
        return self._last_error

    def _set_status(self, status: ConnectionStatus) -> None:
        """Set connection status and notify callbacks."""
        old_status = self._status
        self._status = status

        if status == ConnectionStatus.CONNECTED:
            self._connected_at = datetime.now()

        if old_status != status:
            logger.info(f"Broker connection status changed: {old_status.value} -> {status.value}")
            for callback in self._on_connection_change:
                try:
                    callback(status)
                except Exception as e:
                    logger.error(f"Error in connection change callback: {e}")

    # ========================
    # Connection Methods
    # ========================

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to broker.

        Returns:
            True if connected successfully

        Raises:
            AuthenticationError: If authentication fails
            ConnectionError: If connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker."""
        pass

    @abstractmethod
    async def is_market_open(self, symbol: str = "") -> bool:
        """
        Check if market is open.

        Args:
            symbol: Optional symbol to check (some markets have different hours)

        Returns:
            True if market is open for trading
        """
        pass

    # ========================
    # Account Methods
    # ========================

    @abstractmethod
    async def get_account(self) -> AccountInfo:
        """
        Get account information.

        Returns:
            Account information

        Raises:
            BrokerError: If unable to get account info
        """
        pass

    # ========================
    # Order Methods
    # ========================

    @abstractmethod
    async def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
        extended_hours: bool = False,
    ) -> BrokerOrder:
        """
        Submit an order to the broker.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Order quantity
            order_type: "market", "limit", "stop", "stop_limit"
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: "day", "gtc", "ioc", "fok"
            client_order_id: Optional client-side order ID
            extended_hours: Allow extended hours trading

        Returns:
            BrokerOrder with order details

        Raises:
            OrderRejectedError: If order is rejected
            InsufficientFundsError: If insufficient buying power
            BrokerError: For other errors
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Broker order ID to cancel

        Returns:
            True if cancellation request accepted

        Raises:
            BrokerError: If cancellation fails
        """
        pass

    @abstractmethod
    async def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """
        Get order by ID.

        Args:
            order_id: Broker order ID

        Returns:
            BrokerOrder if found, None otherwise
        """
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[BrokerOrder]:
        """
        Get all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        pass

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            Number of orders cancelled
        """
        orders = await self.get_open_orders(symbol)
        cancelled = 0
        for order in orders:
            try:
                if await self.cancel_order(order.order_id):
                    cancelled += 1
            except BrokerError as e:
                logger.error(f"Failed to cancel order {order.order_id}: {e}")
        return cancelled

    # ========================
    # Position Methods
    # ========================

    @abstractmethod
    async def get_positions(self) -> List[BrokerPosition]:
        """
        Get all open positions.

        Returns:
            List of open positions
        """
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Trading symbol

        Returns:
            BrokerPosition if exists, None otherwise
        """
        pass

    async def close_position(
        self,
        symbol: str,
        quantity: Optional[float] = None,
    ) -> Optional[BrokerOrder]:
        """
        Close a position.

        Args:
            symbol: Symbol to close
            quantity: Optional quantity to close (None = close all)

        Returns:
            Closing order if successful
        """
        position = await self.get_position(symbol)
        if not position:
            return None

        close_qty = quantity or position.quantity
        side = "sell" if position.side == "long" else "buy"

        return await self.submit_order(
            symbol=symbol,
            side=side,
            quantity=close_qty,
            order_type="market",
        )

    async def close_all_positions(self) -> List[BrokerOrder]:
        """
        Close all open positions.

        Returns:
            List of closing orders
        """
        positions = await self.get_positions()
        orders = []

        for position in positions:
            try:
                order = await self.close_position(position.symbol)
                if order:
                    orders.append(order)
            except BrokerError as e:
                logger.error(f"Failed to close position {position.symbol}: {e}")

        return orders

    # ========================
    # Market Data Methods
    # ========================

    @abstractmethod
    async def get_quote(self, symbol: str) -> Quote:
        """
        Get current quote for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current quote
        """
        pass

    @abstractmethod
    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dictionary of symbol -> Quote
        """
        pass

    async def stream_quotes(
        self,
        symbols: List[str],
    ) -> AsyncIterator[Quote]:
        """
        Stream real-time quotes.

        Args:
            symbols: Symbols to stream

        Yields:
            Quote objects as they arrive

        Note:
            Default implementation polls. Override for websocket streaming.
        """
        import asyncio
        while self.is_connected:
            for symbol in symbols:
                try:
                    quote = await self.get_quote(symbol)
                    yield quote
                except BrokerError:
                    pass
            await asyncio.sleep(1.0)

    # ========================
    # Callback Registration
    # ========================

    def on_quote(self, callback: Callable[[Quote], None]) -> None:
        """Register callback for quote updates."""
        self._on_quote.append(callback)

    def on_order_update(self, callback: Callable[[BrokerOrder], None]) -> None:
        """Register callback for order updates."""
        self._on_order_update.append(callback)

    def on_position_update(self, callback: Callable[[BrokerPosition], None]) -> None:
        """Register callback for position updates."""
        self._on_position_update.append(callback)

    def on_connection_change(self, callback: Callable[[ConnectionStatus], None]) -> None:
        """Register callback for connection status changes."""
        self._on_connection_change.append(callback)

    def _notify_quote(self, quote: Quote) -> None:
        """Notify quote callbacks."""
        for callback in self._on_quote:
            try:
                callback(quote)
            except Exception as e:
                logger.error(f"Error in quote callback: {e}")

    def _notify_order_update(self, order: BrokerOrder) -> None:
        """Notify order update callbacks."""
        for callback in self._on_order_update:
            try:
                callback(order)
            except Exception as e:
                logger.error(f"Error in order update callback: {e}")

    def _notify_position_update(self, position: BrokerPosition) -> None:
        """Notify position update callbacks."""
        for callback in self._on_position_update:
            try:
                callback(position)
            except Exception as e:
                logger.error(f"Error in position update callback: {e}")

    # ========================
    # Utility Methods
    # ========================

    def get_stats(self) -> Dict[str, Any]:
        """Get broker adapter statistics."""
        return {
            "broker_type": self.config.broker_type.value,
            "status": self._status.value,
            "is_connected": self.is_connected,
            "paper_mode": self.config.paper,
            "connected_at": self._connected_at.isoformat() if self._connected_at else None,
            "last_error": self._last_error,
        }
