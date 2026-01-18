"""
MetaTrader 5 Broker Adapter.

Implements broker interface for MT5 trading terminal.
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

# Optional imports - gracefully handle missing MT5 library
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None
    logger.warning("MetaTrader5 library not installed. Install with: pip install MetaTrader5")


class MT5Broker(BrokerAdapter):
    """
    MetaTrader 5 broker implementation.

    Provides trading functionality via MT5 terminal connection.

    Features:
    - Market and pending orders (limit, stop, stop-limit)
    - Real-time quote streaming
    - Position and order management
    - Multi-symbol support (Forex, CFDs, etc.)

    Requirements:
    - MetaTrader 5 terminal installed and running
    - Valid broker account credentials

    Example:
        config = BrokerConfig(
            broker_type=BrokerType.MT5,
            login=12345678,
            password="your_password",
            server="YourBroker-Demo",
            path="C:/Program Files/MetaTrader 5/terminal64.exe",
        )
        broker = MT5Broker(config)
        await broker.connect()
    """

    # MT5 Order type mapping
    ORDER_TYPE_MAP = {
        "market": "ORDER_TYPE_BUY",  # Will be adjusted for side
        "limit": "ORDER_TYPE_BUY_LIMIT",
        "stop": "ORDER_TYPE_BUY_STOP",
        "stop_limit": "ORDER_TYPE_BUY_STOP_LIMIT",
    }

    # MT5 Time in force mapping
    TIME_IN_FORCE_MAP = {
        "day": "ORDER_TIME_DAY",
        "gtc": "ORDER_TIME_GTC",
        "ioc": "ORDER_FILLING_IOC",
        "fok": "ORDER_FILLING_FOK",
    }

    def __init__(self, config: BrokerConfig):
        """
        Initialize MT5 broker adapter.

        Args:
            config: Broker configuration with MT5 credentials
        """
        if not MT5_AVAILABLE:
            raise ImportError(
                "MetaTrader5 library not installed. Install with: pip install MetaTrader5"
            )

        super().__init__(config)
        self._initialized = False
        self._order_counter = 0
        self._latest_quotes: Dict[str, Quote] = {}
        self._streaming_task: Optional[asyncio.Task] = None
        self._streaming_symbols: List[str] = []

    async def connect(self) -> bool:
        """
        Connect to MT5 terminal.

        Returns:
            True if connected successfully

        Raises:
            AuthenticationError: If login fails
            ConnectionError: If connection to terminal fails
        """
        self._set_status(ConnectionStatus.CONNECTING)

        try:
            # Initialize MT5
            init_kwargs = {}
            if self.config.path:
                init_kwargs["path"] = self.config.path

            if not mt5.initialize(**init_kwargs):
                error = mt5.last_error()
                raise BrokerConnectionError(f"MT5 initialization failed: {error}")

            # Login to account
            if self.config.login:
                authorized = mt5.login(
                    login=self.config.login,
                    password=self.config.password,
                    server=self.config.server,
                )

                if not authorized:
                    error = mt5.last_error()
                    mt5.shutdown()
                    raise AuthenticationError(f"MT5 login failed: {error}")

            # Verify connection
            account_info = mt5.account_info()
            if account_info is None:
                error = mt5.last_error()
                mt5.shutdown()
                raise BrokerConnectionError(f"Failed to get account info: {error}")

            self._initialized = True

            logger.info(
                f"Connected to MT5 account: {account_info.login} "
                f"({account_info.server})"
            )

            self._set_status(ConnectionStatus.CONNECTED)
            return True

        except (AuthenticationError, BrokerConnectionError):
            self._set_status(ConnectionStatus.ERROR)
            raise
        except Exception as e:
            self._last_error = str(e)
            self._set_status(ConnectionStatus.ERROR)
            raise BrokerConnectionError(f"MT5 connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        # Stop streaming
        if self._streaming_task:
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass
            self._streaming_task = None

        # Shutdown MT5
        if self._initialized:
            mt5.shutdown()
            self._initialized = False

        self._set_status(ConnectionStatus.DISCONNECTED)
        logger.info("Disconnected from MT5")

    async def is_market_open(self, symbol: str = "") -> bool:
        """
        Check if market is open for trading.

        Args:
            symbol: Symbol to check

        Returns:
            True if market is open
        """
        if not self._initialized:
            return False

        if not symbol:
            # Check general market status
            account = mt5.account_info()
            if account is None:
                return False
            return account.trade_allowed

        try:
            # Check specific symbol
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return False

            # Check if trading is enabled for symbol
            return symbol_info.trade_mode != 0  # 0 = disabled

        except Exception as e:
            logger.error(f"Failed to check market status: {e}")
            return False

    async def get_account(self) -> AccountInfo:
        """
        Get MT5 account information.

        Returns:
            AccountInfo with current account state
        """
        if not self._initialized:
            raise BrokerError("Not connected to MT5")

        try:
            account = mt5.account_info()
            if account is None:
                raise BrokerError(f"Failed to get account: {mt5.last_error()}")

            return AccountInfo(
                account_id=str(account.login),
                currency=account.currency,
                balance=float(account.balance),
                equity=float(account.equity),
                margin_used=float(account.margin),
                margin_available=float(account.margin_free),
                buying_power=float(account.margin_free),
                cash=float(account.balance),
                trading_blocked=not account.trade_allowed,
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
        Submit order to MT5.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Lot size
            order_type: "market", "limit", "stop", "stop_limit"
            limit_price: Limit price
            stop_price: Stop price
            time_in_force: Order duration
            client_order_id: Custom order ID (stored in comment)
            extended_hours: Not applicable for MT5

        Returns:
            BrokerOrder with order details

        Raises:
            OrderRejectedError: If order is rejected
            InsufficientFundsError: If margin insufficient
        """
        if not self._initialized:
            raise BrokerError("Not connected to MT5")

        try:
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                raise BrokerError(f"Symbol {symbol} not found")

            # Ensure symbol is enabled for trading
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    raise BrokerError(f"Failed to enable symbol {symbol}")

            # Get current price for market orders
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                raise BrokerError(f"Failed to get price for {symbol}")

            # Determine order type
            is_buy = side.lower() == "buy"
            order_type_lower = order_type.lower()

            if order_type_lower == "market":
                mt5_type = mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL
                price = tick.ask if is_buy else tick.bid
            elif order_type_lower == "limit":
                mt5_type = mt5.ORDER_TYPE_BUY_LIMIT if is_buy else mt5.ORDER_TYPE_SELL_LIMIT
                price = limit_price
            elif order_type_lower == "stop":
                mt5_type = mt5.ORDER_TYPE_BUY_STOP if is_buy else mt5.ORDER_TYPE_SELL_STOP
                price = stop_price
            elif order_type_lower == "stop_limit":
                mt5_type = mt5.ORDER_TYPE_BUY_STOP_LIMIT if is_buy else mt5.ORDER_TYPE_SELL_STOP_LIMIT
                price = stop_price
            else:
                raise BrokerError(f"Unsupported order type: {order_type}")

            if price is None:
                raise BrokerError("Price is required")

            # Map time in force
            if time_in_force.lower() == "gtc":
                type_time = mt5.ORDER_TIME_GTC
            else:
                type_time = mt5.ORDER_TIME_DAY

            # Generate order ID
            self._order_counter += 1
            magic_number = 100000 + self._order_counter

            # Build order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL if order_type_lower == "market" else mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": quantity,
                "type": mt5_type,
                "price": price,
                "deviation": 20,  # Slippage in points
                "magic": magic_number,
                "comment": client_order_id or f"AI_Trader_{self._order_counter}",
                "type_time": type_time,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Add stop limit price if needed
            if order_type_lower == "stop_limit" and limit_price:
                request["stoplimit"] = limit_price

            # Send order
            result = mt5.order_send(request)

            if result is None:
                raise BrokerError(f"Order send failed: {mt5.last_error()}")

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = self._get_error_description(result.retcode)

                if result.retcode == mt5.TRADE_RETCODE_NO_MONEY:
                    raise InsufficientFundsError(f"Insufficient margin: {error_msg}")

                raise OrderRejectedError(
                    f"Order rejected: {error_msg}",
                    reason=error_msg,
                    order_id=str(result.order),
                )

            # Create broker order
            broker_order = BrokerOrder(
                order_id=str(result.order),
                client_order_id=client_order_id or str(magic_number),
                symbol=symbol,
                side=side.lower(),
                order_type=order_type.lower(),
                quantity=quantity,
                filled_quantity=quantity if order_type_lower == "market" else 0.0,
                remaining_quantity=0.0 if order_type_lower == "market" else quantity,
                limit_price=limit_price,
                stop_price=stop_price,
                average_fill_price=result.price if order_type_lower == "market" else None,
                status="filled" if order_type_lower == "market" else "new",
                time_in_force=time_in_force,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            logger.info(
                f"Order submitted: {broker_order.order_id} {side} {quantity} {symbol} @ {order_type}"
            )

            return broker_order

        except (OrderRejectedError, InsufficientFundsError):
            raise
        except Exception as e:
            raise BrokerError(f"Failed to submit order: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: MT5 order ticket

        Returns:
            True if cancellation successful
        """
        if not self._initialized:
            raise BrokerError("Not connected to MT5")

        try:
            # Get order info
            order = mt5.orders_get(ticket=int(order_id))
            if not order:
                logger.warning(f"Order {order_id} not found")
                return False

            order = order[0]

            # Create cancel request
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": int(order_id),
            }

            result = mt5.order_send(request)

            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error = mt5.last_error() if result is None else self._get_error_description(result.retcode)
                raise BrokerError(f"Failed to cancel order: {error}")

            logger.info(f"Order cancelled: {order_id}")
            return True

        except Exception as e:
            raise BrokerError(f"Failed to cancel order: {e}")

    async def get_order(self, order_id: str) -> Optional[BrokerOrder]:
        """
        Get order by ticket.

        Args:
            order_id: MT5 order ticket

        Returns:
            BrokerOrder if found
        """
        if not self._initialized:
            raise BrokerError("Not connected to MT5")

        try:
            # Check pending orders
            order = mt5.orders_get(ticket=int(order_id))
            if order:
                return self._convert_mt5_order(order[0], is_pending=True)

            # Check history
            from_date = datetime(2020, 1, 1)
            order = mt5.history_orders_get(ticket=int(order_id))
            if order:
                return self._convert_mt5_order(order[0], is_pending=False)

            return None

        except Exception as e:
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[BrokerOrder]:
        """
        Get all pending orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of pending orders
        """
        if not self._initialized:
            raise BrokerError("Not connected to MT5")

        try:
            if symbol:
                orders = mt5.orders_get(symbol=symbol)
            else:
                orders = mt5.orders_get()

            if orders is None:
                return []

            return [self._convert_mt5_order(o, is_pending=True) for o in orders]

        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    async def get_positions(self) -> List[BrokerPosition]:
        """
        Get all open positions.

        Returns:
            List of open positions
        """
        if not self._initialized:
            raise BrokerError("Not connected to MT5")

        try:
            positions = mt5.positions_get()
            if positions is None:
                return []

            return [self._convert_mt5_position(p) for p in positions]

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    async def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        """
        Get position for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            BrokerPosition if exists
        """
        if not self._initialized:
            raise BrokerError("Not connected to MT5")

        try:
            positions = mt5.positions_get(symbol=symbol)
            if positions and len(positions) > 0:
                return self._convert_mt5_position(positions[0])
            return None

        except Exception:
            return None

    async def get_quote(self, symbol: str) -> Quote:
        """
        Get current quote for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Current quote
        """
        if not self._initialized:
            raise BrokerError("Not connected to MT5")

        try:
            # Ensure symbol is selected
            if not mt5.symbol_select(symbol, True):
                raise BrokerError(f"Failed to select symbol {symbol}")

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                raise BrokerError(f"Failed to get tick for {symbol}")

            quote = Quote(
                symbol=symbol,
                bid=float(tick.bid),
                ask=float(tick.ask),
                bid_size=0.0,  # MT5 doesn't provide size
                ask_size=0.0,
                last=float(tick.last) if tick.last else float(tick.bid),
                last_size=float(tick.volume) if tick.volume else 0.0,
                timestamp=datetime.fromtimestamp(tick.time, tz=timezone.utc),
                volume=float(tick.volume) if tick.volume else 0.0,
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
        result = {}
        for symbol in symbols:
            try:
                result[symbol] = await self.get_quote(symbol)
            except BrokerError:
                pass
        return result

    async def stream_quotes(
        self,
        symbols: List[str],
    ) -> AsyncIterator[Quote]:
        """
        Stream quotes by polling (MT5 doesn't have native streaming in Python).

        Args:
            symbols: Symbols to stream

        Yields:
            Quote objects
        """
        self._streaming_symbols = symbols

        # Select all symbols
        for symbol in symbols:
            mt5.symbol_select(symbol, True)

        while self.is_connected:
            for symbol in symbols:
                try:
                    quote = await self.get_quote(symbol)
                    self._notify_quote(quote)
                    yield quote
                except BrokerError:
                    pass

            await asyncio.sleep(0.1)  # 100ms polling interval

    def _convert_mt5_order(self, order: Any, is_pending: bool) -> BrokerOrder:
        """Convert MT5 order to BrokerOrder."""
        # Determine side from order type
        order_type_val = order.type
        is_buy = order_type_val in (
            mt5.ORDER_TYPE_BUY,
            mt5.ORDER_TYPE_BUY_LIMIT,
            mt5.ORDER_TYPE_BUY_STOP,
            mt5.ORDER_TYPE_BUY_STOP_LIMIT,
        )

        # Map order type
        if order_type_val in (mt5.ORDER_TYPE_BUY, mt5.ORDER_TYPE_SELL):
            order_type_str = "market"
        elif order_type_val in (mt5.ORDER_TYPE_BUY_LIMIT, mt5.ORDER_TYPE_SELL_LIMIT):
            order_type_str = "limit"
        elif order_type_val in (mt5.ORDER_TYPE_BUY_STOP, mt5.ORDER_TYPE_SELL_STOP):
            order_type_str = "stop"
        else:
            order_type_str = "stop_limit"

        # Determine status
        if is_pending:
            status = "new"
        elif hasattr(order, 'state') and order.state == mt5.ORDER_STATE_FILLED:
            status = "filled"
        elif hasattr(order, 'state') and order.state == mt5.ORDER_STATE_CANCELED:
            status = "cancelled"
        else:
            status = "filled"  # Historical orders are typically filled

        return BrokerOrder(
            order_id=str(order.ticket),
            client_order_id=str(order.magic) if hasattr(order, 'magic') else "",
            symbol=order.symbol,
            side="buy" if is_buy else "sell",
            order_type=order_type_str,
            quantity=float(order.volume_initial) if hasattr(order, 'volume_initial') else float(order.volume),
            filled_quantity=float(order.volume_initial - order.volume_current) if is_pending else float(order.volume),
            remaining_quantity=float(order.volume_current) if is_pending else 0.0,
            limit_price=float(order.price_open) if order.price_open else None,
            stop_price=float(order.price_stoplimit) if hasattr(order, 'price_stoplimit') and order.price_stoplimit else None,
            average_fill_price=float(order.price_open) if not is_pending else None,
            status=status,
            time_in_force="gtc" if hasattr(order, 'type_time') and order.type_time == mt5.ORDER_TIME_GTC else "day",
            created_at=datetime.fromtimestamp(order.time_setup, tz=timezone.utc) if hasattr(order, 'time_setup') else datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    def _convert_mt5_position(self, position: Any) -> BrokerPosition:
        """Convert MT5 position to BrokerPosition."""
        is_long = position.type == mt5.POSITION_TYPE_BUY

        return BrokerPosition(
            symbol=position.symbol,
            quantity=float(position.volume),
            side="long" if is_long else "short",
            average_entry_price=float(position.price_open),
            current_price=float(position.price_current),
            market_value=float(position.volume * position.price_current),
            cost_basis=float(position.volume * position.price_open),
            unrealized_pnl=float(position.profit),
            unrealized_pnl_pct=(position.profit / (position.volume * position.price_open)) * 100 if position.volume > 0 else 0.0,
            last_updated=datetime.now(timezone.utc),
        )

    def _get_error_description(self, retcode: int) -> str:
        """Get human-readable error description."""
        error_codes = {
            mt5.TRADE_RETCODE_REQUOTE: "Requote",
            mt5.TRADE_RETCODE_REJECT: "Request rejected",
            mt5.TRADE_RETCODE_CANCEL: "Request cancelled",
            mt5.TRADE_RETCODE_PLACED: "Order placed",
            mt5.TRADE_RETCODE_DONE: "Request completed",
            mt5.TRADE_RETCODE_DONE_PARTIAL: "Partially completed",
            mt5.TRADE_RETCODE_ERROR: "Request processing error",
            mt5.TRADE_RETCODE_TIMEOUT: "Request timeout",
            mt5.TRADE_RETCODE_INVALID: "Invalid request",
            mt5.TRADE_RETCODE_INVALID_VOLUME: "Invalid volume",
            mt5.TRADE_RETCODE_INVALID_PRICE: "Invalid price",
            mt5.TRADE_RETCODE_INVALID_STOPS: "Invalid stops",
            mt5.TRADE_RETCODE_TRADE_DISABLED: "Trade disabled",
            mt5.TRADE_RETCODE_MARKET_CLOSED: "Market closed",
            mt5.TRADE_RETCODE_NO_MONEY: "Insufficient funds",
            mt5.TRADE_RETCODE_PRICE_CHANGED: "Price changed",
            mt5.TRADE_RETCODE_PRICE_OFF: "Price off",
            mt5.TRADE_RETCODE_INVALID_EXPIRATION: "Invalid expiration",
            mt5.TRADE_RETCODE_ORDER_CHANGED: "Order changed",
            mt5.TRADE_RETCODE_TOO_MANY_REQUESTS: "Too many requests",
        }
        return error_codes.get(retcode, f"Unknown error ({retcode})")

    def get_stats(self) -> Dict[str, Any]:
        """Get broker statistics."""
        stats = super().get_stats()
        stats.update({
            "mt5_initialized": self._initialized,
            "streaming_symbols": self._streaming_symbols,
            "cached_quotes": len(self._latest_quotes),
        })
        return stats
