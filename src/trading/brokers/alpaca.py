"""
Alpaca Markets Broker Adapter.

Implements broker interface for Alpaca trading API.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, AsyncIterator
import logging

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
    ConnectionError as BrokerConnectionError,
)

logger = logging.getLogger(__name__)

# Optional imports - gracefully handle missing alpaca library
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopOrderRequest,
        StopLimitOrderRequest,
    )
    from alpaca.trading.enums import OrderSide, OrderType, TimeInForce, OrderStatus
    from alpaca.data.live import StockDataStream
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca SDK not installed. Install with: pip install alpaca-py")


class AlpacaBroker(BrokerAdapter):
    """
    Alpaca Markets broker implementation.

    Supports both paper and live trading via Alpaca API.

    Features:
    - Market, limit, stop, and stop-limit orders
    - Real-time quote streaming
    - Position and order management
    - Account information retrieval

    Example:
        config = BrokerConfig(
            broker_type=BrokerType.ALPACA,
            api_key="your_api_key",
            secret_key="your_secret_key",
            paper=True,
        )
        broker = AlpacaBroker(config)
        await broker.connect()
    """

    def __init__(self, config: BrokerConfig):
        """
        Initialize Alpaca broker adapter.

        Args:
            config: Broker configuration with API credentials
        """
        if not ALPACA_AVAILABLE:
            raise ImportError(
                "Alpaca SDK not installed. Install with: pip install alpaca-py"
            )

        super().__init__(config)
        self._trading_client: Optional[TradingClient] = None
        self._data_client: Optional[StockHistoricalDataClient] = None
        self._stream: Optional[StockDataStream] = None
        self._quote_subscriptions: Dict[str, bool] = {}
        self._latest_quotes: Dict[str, Quote] = {}

    async def connect(self) -> bool:
        """
        Connect to Alpaca API.

        Returns:
            True if connected successfully

        Raises:
            AuthenticationError: If API credentials are invalid
            ConnectionError: If connection fails
        """
        self._set_status(ConnectionStatus.CONNECTING)

        try:
            # Create trading client
            self._trading_client = TradingClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                paper=self.config.paper,
            )

            # Create data client
            self._data_client = StockHistoricalDataClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
            )

            # Verify connection by fetching account
            account = self._trading_client.get_account()
            if account is None:
                raise BrokerConnectionError("Failed to get account information")

            logger.info(
                f"Connected to Alpaca ({'paper' if self.config.paper else 'live'}) "
                f"account: {account.account_number}"
            )

            self._set_status(ConnectionStatus.CONNECTED)
            return True

        except Exception as e:
            self._last_error = str(e)
            self._set_status(ConnectionStatus.ERROR)

            error_str = str(e).lower()
            if "unauthorized" in error_str or "forbidden" in error_str:
                raise AuthenticationError(f"Authentication failed: {e}")
            raise BrokerConnectionError(f"Connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from Alpaca API."""
        if self._stream:
            try:
                await self._stream.close()
            except Exception as e:
                logger.warning(f"Error closing stream: {e}")
            self._stream = None

        self._trading_client = None
        self._data_client = None
        self._set_status(ConnectionStatus.DISCONNECTED)
        logger.info("Disconnected from Alpaca")

    async def is_market_open(self, symbol: str = "") -> bool:
        """
        Check if market is open.

        Args:
            symbol: Not used for Alpaca (uses market clock)

        Returns:
            True if US stock market is open
        """
        if not self._trading_client:
            return False

        try:
            clock = self._trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Failed to get market clock: {e}")
            return False

    async def get_account(self) -> AccountInfo:
        """
        Get Alpaca account information.

        Returns:
            AccountInfo with current account state

        Raises:
            BrokerError: If unable to get account
        """
        if not self._trading_client:
            raise BrokerError("Not connected to Alpaca")

        try:
            account = self._trading_client.get_account()

            return AccountInfo(
                account_id=str(account.account_number),
                currency=str(account.currency) if account.currency else "USD",
                balance=float(account.cash),
                equity=float(account.equity),
                margin_used=float(account.initial_margin) if account.initial_margin else 0.0,
                margin_available=float(account.buying_power) / 4,  # Approximate
                buying_power=float(account.buying_power),
                cash=float(account.cash),
                day_trades_remaining=int(account.daytrade_count) if account.daytrade_count else -1,
                pattern_day_trader=bool(account.pattern_day_trader),
                trading_blocked=bool(account.trading_blocked),
                transfers_blocked=bool(account.transfers_blocked),
                account_blocked=bool(account.account_blocked),
                last_updated=datetime.now(timezone.utc),
            )
        except Exception as e:
            raise BrokerError(f"Failed to get account: {e}")

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
        Submit order to Alpaca.

        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            quantity: Number of shares
            order_type: "market", "limit", "stop", "stop_limit"
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            time_in_force: "day", "gtc", "ioc", "fok"
            client_order_id: Custom order ID
            extended_hours: Allow extended hours

        Returns:
            BrokerOrder with order details

        Raises:
            OrderRejectedError: If order is rejected
            InsufficientFundsError: If buying power insufficient
        """
        if not self._trading_client:
            raise BrokerError("Not connected to Alpaca")

        try:
            # Map side
            alpaca_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            # Map time in force
            tif_map = {
                "day": TimeInForce.DAY,
                "gtc": TimeInForce.GTC,
                "ioc": TimeInForce.IOC,
                "fok": TimeInForce.FOK,
            }
            alpaca_tif = tif_map.get(time_in_force.lower(), TimeInForce.DAY)

            # Create appropriate order request
            order_type_lower = order_type.lower()

            if order_type_lower == "market":
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    client_order_id=client_order_id,
                    extended_hours=extended_hours,
                )
            elif order_type_lower == "limit":
                if limit_price is None:
                    raise BrokerError("Limit price required for limit order")
                request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    limit_price=limit_price,
                    client_order_id=client_order_id,
                    extended_hours=extended_hours,
                )
            elif order_type_lower == "stop":
                if stop_price is None:
                    raise BrokerError("Stop price required for stop order")
                request = StopOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    stop_price=stop_price,
                    client_order_id=client_order_id,
                    extended_hours=extended_hours,
                )
            elif order_type_lower == "stop_limit":
                if limit_price is None or stop_price is None:
                    raise BrokerError("Both limit and stop price required for stop-limit order")
                request = StopLimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=alpaca_side,
                    time_in_force=alpaca_tif,
                    limit_price=limit_price,
                    stop_price=stop_price,
                    client_order_id=client_order_id,
                    extended_hours=extended_hours,
                )
            else:
                raise BrokerError(f"Unsupported order type: {order_type}")

            # Submit order
            order = self._trading_client.submit_order(request)

            broker_order = self._convert_order(order)
            logger.info(
                f"Order submitted: {broker_order.order_id} {side} {quantity} {symbol} @ {order_type}"
            )

            return broker_order

        except Exception as e:
            error_str = str(e).lower()

            if "insufficient" in error_str or "buying power" in error_str:
                raise InsufficientFundsError(f"Insufficient funds: {e}")

            if "rejected" in error_str:
                raise OrderRejectedError(f"Order rejected: {e}", reason=str(e))

            raise BrokerError(f"Failed to submit order: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Alpaca order ID

        Returns:
            True if cancellation accepted
        """
        if not self._trading_client:
            raise BrokerError("Not connected to Alpaca")

        try:
            self._trading_client.cancel_order_by_id(order_id)
            logger.info(f"Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise BrokerError(f"Failed to cancel order: {e}")

    async def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """
        Get order by ID.

        Args:
            order_id: Alpaca order ID

        Returns:
            BrokerOrder if found
        """
        if not self._trading_client:
            raise BrokerError("Not connected to Alpaca")

        try:
            order = self._trading_client.get_order_by_id(order_id)
            return self._convert_order(order)
        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[BrokerOrder]:
        """
        Get all open orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open orders
        """
        if not self._trading_client:
            raise BrokerError("Not connected to Alpaca")

        try:
            orders = self._trading_client.get_orders(
                filter={"status": "open", "symbols": [symbol] if symbol else None}
            )
            return [self._convert_order(o) for o in orders]
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    async def get_positions(self) -> List[BrokerPosition]:
        """
        Get all open positions.

        Returns:
            List of open positions
        """
        if not self._trading_client:
            raise BrokerError("Not connected to Alpaca")

        try:
            positions = self._trading_client.get_all_positions()
            return [self._convert_position(p) for p in positions]
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """
        Get position for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            BrokerPosition if exists
        """
        if not self._trading_client:
            raise BrokerError("Not connected to Alpaca")

        try:
            position = self._trading_client.get_open_position(symbol)
            return self._convert_position(position)
        except Exception:
            return None

    async def get_quote(self, symbol: str) -> Quote:
        """
        Get current quote for symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Current quote
        """
        if not self._data_client:
            raise BrokerError("Not connected to Alpaca")

        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self._data_client.get_stock_latest_quote(request)

            if symbol not in quotes:
                raise BrokerError(f"No quote available for {symbol}")

            q = quotes[symbol]

            quote = Quote(
                symbol=symbol,
                bid=float(q.bid_price) if q.bid_price else 0.0,
                ask=float(q.ask_price) if q.ask_price else 0.0,
                bid_size=float(q.bid_size) if q.bid_size else 0.0,
                ask_size=float(q.ask_size) if q.ask_size else 0.0,
                last=float(q.ask_price) if q.ask_price else 0.0,  # Use ask as last
                last_size=0.0,
                timestamp=q.timestamp if q.timestamp else datetime.now(timezone.utc),
            )

            self._latest_quotes[symbol] = quote
            return quote

        except Exception as e:
            raise BrokerError(f"Failed to get quote for {symbol}: {e}")

    async def get_quotes(self, symbols: List[str]) -> Dict[str, Quote]:
        """
        Get quotes for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dict of symbol -> Quote
        """
        if not self._data_client:
            raise BrokerError("Not connected to Alpaca")

        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self._data_client.get_stock_latest_quote(request)

            result = {}
            for symbol, q in quotes.items():
                quote = Quote(
                    symbol=symbol,
                    bid=float(q.bid_price) if q.bid_price else 0.0,
                    ask=float(q.ask_price) if q.ask_price else 0.0,
                    bid_size=float(q.bid_size) if q.bid_size else 0.0,
                    ask_size=float(q.ask_size) if q.ask_size else 0.0,
                    last=float(q.ask_price) if q.ask_price else 0.0,
                    last_size=0.0,
                    timestamp=q.timestamp if q.timestamp else datetime.now(timezone.utc),
                )
                result[symbol] = quote
                self._latest_quotes[symbol] = quote

            return result

        except Exception as e:
            raise BrokerError(f"Failed to get quotes: {e}")

    async def stream_quotes(
        self,
        symbols: List[str],
    ) -> AsyncIterator[Quote]:
        """
        Stream real-time quotes via websocket.

        Args:
            symbols: Symbols to stream

        Yields:
            Quote objects as they arrive
        """
        if not ALPACA_AVAILABLE:
            raise BrokerError("Alpaca SDK not available")

        # Create stream if not exists
        if self._stream is None:
            self._stream = StockDataStream(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
            )

        # Queue for quotes
        quote_queue: asyncio.Queue[Quote] = asyncio.Queue()

        async def quote_handler(data):
            quote = Quote(
                symbol=data.symbol,
                bid=float(data.bid_price) if hasattr(data, 'bid_price') else 0.0,
                ask=float(data.ask_price) if hasattr(data, 'ask_price') else 0.0,
                bid_size=float(data.bid_size) if hasattr(data, 'bid_size') else 0.0,
                ask_size=float(data.ask_size) if hasattr(data, 'ask_size') else 0.0,
                last=float(data.price) if hasattr(data, 'price') else 0.0,
                last_size=float(data.size) if hasattr(data, 'size') else 0.0,
                timestamp=data.timestamp if hasattr(data, 'timestamp') else datetime.now(timezone.utc),
            )
            await quote_queue.put(quote)
            self._latest_quotes[quote.symbol] = quote
            self._notify_quote(quote)

        # Subscribe to quotes
        self._stream.subscribe_quotes(quote_handler, *symbols)

        # Start stream in background
        stream_task = asyncio.create_task(self._stream._run_forever())

        try:
            while self.is_connected:
                try:
                    quote = await asyncio.wait_for(quote_queue.get(), timeout=1.0)
                    yield quote
                except asyncio.TimeoutError:
                    continue
        finally:
            stream_task.cancel()
            try:
                await stream_task
            except asyncio.CancelledError:
                pass

    def _convert_order(self, order: Any) -> BrokerOrder:
        """Convert Alpaca order to BrokerOrder."""
        # Map status
        status_map = {
            OrderStatus.NEW: "new",
            OrderStatus.ACCEPTED: "accepted",
            OrderStatus.PENDING_NEW: "pending_new",
            OrderStatus.FILLED: "filled",
            OrderStatus.PARTIALLY_FILLED: "partially_filled",
            OrderStatus.CANCELED: "cancelled",
            OrderStatus.REJECTED: "rejected",
            OrderStatus.EXPIRED: "cancelled",
        }

        return BrokerOrder(
            order_id=str(order.id),
            client_order_id=str(order.client_order_id) if order.client_order_id else "",
            symbol=str(order.symbol),
            side=order.side.value.lower() if order.side else "buy",
            order_type=order.order_type.value.lower() if order.order_type else "market",
            quantity=float(order.qty) if order.qty else 0.0,
            filled_quantity=float(order.filled_qty) if order.filled_qty else 0.0,
            remaining_quantity=float(order.qty or 0) - float(order.filled_qty or 0),
            limit_price=float(order.limit_price) if order.limit_price else None,
            stop_price=float(order.stop_price) if order.stop_price else None,
            average_fill_price=float(order.filled_avg_price) if order.filled_avg_price else None,
            status=status_map.get(order.status, "new") if order.status else "new",
            time_in_force=order.time_in_force.value.lower() if order.time_in_force else "day",
            created_at=order.created_at if order.created_at else datetime.now(timezone.utc),
            updated_at=order.updated_at if order.updated_at else datetime.now(timezone.utc),
            filled_at=order.filled_at if order.filled_at else None,
            extended_hours=bool(order.extended_hours) if hasattr(order, 'extended_hours') else False,
        )

    def _convert_position(self, position: Any) -> BrokerPosition:
        """Convert Alpaca position to BrokerPosition."""
        qty = float(position.qty) if position.qty else 0.0
        side = "long" if qty > 0 else "short"
        qty = abs(qty)

        return BrokerPosition(
            symbol=str(position.symbol),
            quantity=qty,
            side=side,
            average_entry_price=float(position.avg_entry_price) if position.avg_entry_price else 0.0,
            current_price=float(position.current_price) if position.current_price else 0.0,
            market_value=float(position.market_value) if position.market_value else 0.0,
            cost_basis=float(position.cost_basis) if position.cost_basis else 0.0,
            unrealized_pnl=float(position.unrealized_pl) if position.unrealized_pl else 0.0,
            unrealized_pnl_pct=float(position.unrealized_plpc) * 100 if position.unrealized_plpc else 0.0,
            exchange=str(position.exchange) if hasattr(position, 'exchange') and position.exchange else "",
            asset_class=str(position.asset_class) if hasattr(position, 'asset_class') and position.asset_class else "",
            last_updated=datetime.now(timezone.utc),
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get broker statistics."""
        stats = super().get_stats()
        stats.update({
            "alpaca_paper": self.config.paper,
            "subscribed_symbols": list(self._quote_subscriptions.keys()),
            "cached_quotes": len(self._latest_quotes),
        })
        return stats
