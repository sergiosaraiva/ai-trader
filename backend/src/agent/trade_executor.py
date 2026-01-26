"""Trade execution module for the trading agent.

Handles trade execution based on signals, including position sizing,
order submission, and trade tracking.
"""

import asyncio
import logging
import threading
from typing import Optional, List
from datetime import datetime

from sqlalchemy.orm import Session

from .broker_manager import BrokerManager
from .config import AgentConfig
from .models import TradeResult, PositionStatus
from ..trading.brokers.base import (
    BrokerError,
    OrderRejectedError,
    InsufficientFundsError,
)
from ..trading.signals.actions import TradingSignal, Action
from ..api.database.models import Trade

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Executes trades based on signals.

    Features:
    - Signal validation
    - Position sizing (Kelly Criterion or fixed)
    - Order submission to MT5
    - Trade tracking in database
    - Exit condition monitoring
    """

    def __init__(
        self,
        broker_manager: BrokerManager,
        config: AgentConfig,
        db_session_factory,
    ):
        """Initialize trade executor.

        Args:
            broker_manager: Broker connection manager
            config: Agent configuration
            db_session_factory: Factory to create database sessions
        """
        self.broker_manager = broker_manager
        self.config = config
        self.db_session_factory = db_session_factory

        # Track open trades
        self._open_trades: dict[int, dict] = {}  # trade_id -> trade info

        # Track orphaned trades (executed in MT5 but failed to record in DB)
        self._orphaned_trades: list[dict] = []
        self._orphaned_lock = threading.Lock()  # Thread-safe access to orphaned trades

        # Use configurable timeout from config
        self._db_timeout = config.db_timeout_seconds

        logger.info(f"TradeExecutor initialized (db_timeout={self._db_timeout}s)")

    async def execute_signal(self, signal: TradingSignal) -> TradeResult:
        """Execute a trading signal.

        Steps:
        1. Validate signal
        2. Calculate position size
        3. Submit order to MT5
        4. Wait for fill
        5. Store trade in database
        6. Return result

        Args:
            signal: Trading signal to execute

        Returns:
            TradeResult with execution details
        """
        start_time = datetime.now()

        # Step 1: Validate signal
        if signal.action not in [Action.BUY, Action.SELL]:
            return TradeResult(
                success=False,
                error="Invalid signal action (must be BUY or SELL)",
            )

        # Check broker connection
        if not self.broker_manager.is_connected():
            return TradeResult(
                success=False,
                error="Broker not connected",
            )

        # Step 2: Calculate position size
        try:
            account_info = await self.broker_manager.get_account_info()
            if not account_info:
                return TradeResult(
                    success=False,
                    error="Failed to get account information",
                )

            equity = account_info["equity"]
            quantity = await self._calculate_position_size(
                signal=signal,
                equity=equity,
            )

            if quantity <= 0:
                return TradeResult(
                    success=False,
                    error="Calculated position size is zero",
                )

            logger.info(
                f"Executing {signal.action.value} signal: "
                f"{quantity:.2f} lots @ confidence {signal.confidence:.1%}"
            )

        except ValueError as e:
            # Position sizing safety check failed - do not proceed
            logger.error(f"Position sizing safety check failed: {e}")
            return TradeResult(
                success=False,
                error=f"Position sizing safety error: {str(e)}",
            )

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return TradeResult(
                success=False,
                error=f"Position sizing error: {str(e)}",
            )

        # Step 3: Submit order to MT5
        try:
            side = "buy" if signal.action == Action.BUY else "sell"

            broker_order = await self.broker_manager.broker.submit_order(
                symbol=signal.symbol,
                side=side,
                quantity=quantity,
                order_type="market",
            )

            if not broker_order.is_filled:
                return TradeResult(
                    success=False,
                    error=f"Order not filled: {broker_order.status}",
                )

            entry_price = broker_order.average_fill_price
            mt5_ticket = int(broker_order.order_id)

            logger.info(
                f"Order filled: {mt5_ticket} - "
                f"{quantity:.2f} lots @ {entry_price:.5f}"
            )

        except OrderRejectedError as e:
            logger.error(f"Order rejected: {e}")
            return TradeResult(
                success=False,
                error=f"Order rejected: {e.reason}",
            )

        except InsufficientFundsError as e:
            logger.error(f"Insufficient funds: {e}")
            return TradeResult(
                success=False,
                error=f"Insufficient funds: required={e.required}, available={e.available}",
            )

        except BrokerError as e:
            logger.error(f"Broker error: {e}")
            return TradeResult(
                success=False,
                error=f"Broker error: {str(e)}",
            )

        # Step 4: Store trade in database
        try:
            trade_id = await asyncio.wait_for(
                asyncio.to_thread(
                    self._store_trade,
                    signal=signal,
                    entry_price=entry_price,
                    quantity=quantity,
                    mt5_ticket=mt5_ticket,
                ),
                timeout=self._db_timeout
            )

            # Track open trade
            self._open_trades[trade_id] = {
                "trade_id": trade_id,
                "mt5_ticket": mt5_ticket,
                "symbol": signal.symbol,
                "direction": "long" if signal.action == Action.BUY else "short",
                "entry_price": entry_price,
                "quantity": quantity,
                "stop_loss_price": signal.stop_loss_price,
                "take_profit_price": signal.take_profit_price,
                "entry_time": start_time,
                "max_bars": 24,  # Default max holding period
            }

            logger.info(f"Trade stored in database: ID={trade_id}")

        except asyncio.TimeoutError:
            logger.error(f"Database store operation timed out after {self._db_timeout}s")

            # Track orphaned trade for later reconciliation
            orphaned_trade = {
                "mt5_ticket": mt5_ticket,
                "symbol": signal.symbol,
                "side": side,
                "quantity": quantity,
                "entry_price": entry_price,
                "entry_time": start_time,
                "confidence": signal.confidence,
                "stop_loss": signal.stop_loss_price,
                "take_profit": signal.take_profit_price,
                "db_error": f"Timeout after {self._db_timeout}s",
                "retry_count": 0,
            }
            with self._orphaned_lock:
                self._orphaned_trades.append(orphaned_trade)

            # Log critical error
            logger.critical(
                f"ORPHANED TRADE DETECTED (TIMEOUT) - Added to reconciliation queue: "
                f"MT5 ticket={mt5_ticket}, "
                f"symbol={signal.symbol}, "
                f"side={side}, "
                f"quantity={quantity}, "
                f"price={entry_price}"
            )

            # Attempt immediate retry
            retry_trade_id = await self._retry_store_orphaned_trade(orphaned_trade)
            if retry_trade_id:
                logger.info(f"Orphaned trade recovered on retry: trade_id={retry_trade_id}")
                # Remove from orphaned list since it was recovered
                with self._orphaned_lock:
                    if orphaned_trade in self._orphaned_trades:
                        self._orphaned_trades.remove(orphaned_trade)
                trade_id = retry_trade_id
            else:
                return TradeResult(
                    success=True,  # Trade was executed in MT5
                    trade_id=None,
                    mt5_ticket=mt5_ticket,
                    entry_price=entry_price,
                    error=f"Database timeout (trade orphaned): {self._db_timeout}s",
                )

        except Exception as e:
            logger.error(f"Failed to store trade in database: {e}")

            # Track orphaned trade for later reconciliation
            orphaned_trade = {
                "mt5_ticket": mt5_ticket,
                "symbol": signal.symbol,
                "side": side,
                "quantity": quantity,
                "entry_price": entry_price,
                "entry_time": start_time,
                "confidence": signal.confidence,
                "stop_loss": signal.stop_loss_price,
                "take_profit": signal.take_profit_price,
                "db_error": str(e),
                "retry_count": 0,
            }
            with self._orphaned_lock:
                self._orphaned_trades.append(orphaned_trade)

            # Log critical error
            logger.critical(
                f"ORPHANED TRADE DETECTED - Added to reconciliation queue: "
                f"MT5 ticket={mt5_ticket}, "
                f"symbol={signal.symbol}, "
                f"side={side}, "
                f"quantity={quantity}, "
                f"price={entry_price}"
            )

            # Attempt immediate retry
            retry_trade_id = await self._retry_store_orphaned_trade(orphaned_trade)
            if retry_trade_id:
                logger.info(f"Orphaned trade recovered on retry: trade_id={retry_trade_id}")
                # Remove from orphaned list since it was recovered
                with self._orphaned_lock:
                    if orphaned_trade in self._orphaned_trades:
                        self._orphaned_trades.remove(orphaned_trade)
                trade_id = retry_trade_id
            else:
                return TradeResult(
                    success=True,  # Trade was executed in MT5
                    trade_id=None,
                    mt5_ticket=mt5_ticket,
                    entry_price=entry_price,
                    error=f"Database error (trade orphaned): {str(e)}",
                )

        # Step 5: Return result
        duration_ms = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(
            f"Trade executed successfully in {duration_ms:.1f}ms - "
            f"ID={trade_id}, ticket={mt5_ticket}"
        )

        return TradeResult(
            success=True,
            trade_id=trade_id,
            mt5_ticket=mt5_ticket,
            entry_price=entry_price,
        )

    async def check_open_positions(self) -> List[PositionStatus]:
        """Check status of open positions and identify exits.

        Checks for:
        - Take profit hit
        - Stop loss hit
        - Maximum bars/time exceeded

        Returns:
            List of positions that should be closed
        """
        if not self.broker_manager.is_connected():
            return []

        positions_to_close = []

        try:
            # Get current positions from broker
            broker_positions = await self.broker_manager.get_open_positions()

            # Check each tracked trade
            for trade_id, trade_info in list(self._open_trades.items()):
                mt5_ticket = trade_info["mt5_ticket"]
                symbol = trade_info["symbol"]

                # Find matching broker position
                broker_position = next(
                    (p for p in broker_positions if str(p.get("symbol")) == symbol),
                    None
                )

                if not broker_position:
                    # Position already closed in MT5
                    logger.info(f"Trade {trade_id} (ticket {mt5_ticket}) already closed")
                    del self._open_trades[trade_id]
                    continue

                current_price = broker_position["current_price"]
                unrealized_pnl = broker_position["unrealized_pnl"]

                # Check exit conditions
                should_close = False
                close_reason = None

                # Check take profit
                if trade_info.get("take_profit_price"):
                    if trade_info["direction"] == "long":
                        if current_price >= trade_info["take_profit_price"]:
                            should_close = True
                            close_reason = "take_profit"
                    else:  # short
                        if current_price <= trade_info["take_profit_price"]:
                            should_close = True
                            close_reason = "take_profit"

                # Check stop loss
                if trade_info.get("stop_loss_price") and not should_close:
                    if trade_info["direction"] == "long":
                        if current_price <= trade_info["stop_loss_price"]:
                            should_close = True
                            close_reason = "stop_loss"
                    else:  # short
                        if current_price >= trade_info["stop_loss_price"]:
                            should_close = True
                            close_reason = "stop_loss"

                # Check max holding time
                if not should_close:
                    bars_elapsed = (datetime.now() - trade_info["entry_time"]).total_seconds() / 3600
                    if bars_elapsed >= trade_info.get("max_bars", 24):
                        should_close = True
                        close_reason = "timeout"

                if should_close:
                    positions_to_close.append(
                        PositionStatus(
                            trade_id=trade_id,
                            mt5_ticket=mt5_ticket,
                            current_price=current_price,
                            unrealized_pnl=unrealized_pnl,
                            should_close=True,
                            close_reason=close_reason,
                        )
                    )

        except Exception as e:
            logger.error(f"Error checking open positions: {e}")

        return positions_to_close

    async def close_position(self, position_id: int, reason: str) -> bool:
        """Close a specific position.

        Args:
            position_id: Trade ID to close
            reason: Reason for closing

        Returns:
            True if closed successfully, False otherwise
        """
        if position_id not in self._open_trades:
            logger.warning(f"Trade {position_id} not found in open trades")
            return False

        trade_info = self._open_trades[position_id]

        try:
            # Close position in MT5
            symbol = trade_info["symbol"]
            close_order = await self.broker_manager.broker.close_position(symbol)

            if not close_order:
                logger.error(f"Failed to close position {position_id}")
                return False

            exit_price = close_order.average_fill_price

            # Update database
            await asyncio.wait_for(
                asyncio.to_thread(
                    self._update_trade_exit,
                    trade_id=position_id,
                    exit_price=exit_price,
                    exit_reason=reason,
                ),
                timeout=self._db_timeout
            )

            # Remove from tracking
            del self._open_trades[position_id]

            logger.info(
                f"Position closed: ID={position_id}, "
                f"reason={reason}, "
                f"exit_price={exit_price:.5f}"
            )

            return True

        except asyncio.TimeoutError:
            logger.error(f"Timeout updating trade exit after {self._db_timeout}s for position {position_id}")
            # Position was closed in MT5 but DB update failed - remove from tracking
            del self._open_trades[position_id]
            return True  # Trade was closed in MT5

        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
            return False

    async def close_all_positions(self, reason: str) -> int:
        """Close all open positions.

        Args:
            reason: Reason for closing all positions

        Returns:
            Number of positions closed
        """
        closed_count = 0

        for trade_id in list(self._open_trades.keys()):
            if await self.close_position(trade_id, reason):
                closed_count += 1

        logger.info(f"Closed {closed_count} positions (reason: {reason})")
        return closed_count

    async def _calculate_position_size(
        self,
        signal: TradingSignal,
        equity: float,
    ) -> float:
        """Calculate position size in lots.

        Uses Kelly Criterion if enabled, otherwise uses signal's position_size_pct.

        Args:
            signal: Trading signal
            equity: Account equity

        Returns:
            Position size in lots

        Raises:
            ValueError: If position size cannot be safely calculated
        """
        if self.config.use_kelly_sizing:
            # Kelly Criterion: f* = (bp - q) / b
            # where b = odds, p = win probability, q = loss probability
            # Simplified: use confidence as win probability
            win_prob = signal.confidence
            loss_prob = 1 - win_prob
            odds = signal.risk_reward_ratio if signal.risk_reward_ratio > 0 else 2.0

            kelly_fraction = (odds * win_prob - loss_prob) / odds
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%

            position_pct = kelly_fraction
        else:
            # Use signal's recommended position size
            position_pct = signal.position_size_pct

        # Cap at maximum
        position_pct = min(position_pct, self.config.max_position_size)

        # Calculate notional value
        notional = equity * position_pct

        # Get current price from broker (CRITICAL: never use fallback)
        current_price = None
        try:
            tick = await self.broker_manager.get_current_price(signal.symbol)
            if tick and tick.get("bid"):
                current_price = tick["bid"]
        except Exception as e:
            logger.warning(f"Failed to get current price from broker: {e}")

        # Fallback to signal's stop_loss_price only if it's a valid forex price
        if current_price is None and signal.stop_loss_price:
            # Get realistic price range for the currency pair
            valid_range = self._get_price_range_for_symbol(signal.symbol)
            if valid_range[0] <= signal.stop_loss_price <= valid_range[1]:
                current_price = signal.stop_loss_price
                logger.warning(
                    f"Using signal stop_loss_price as price fallback: {current_price}"
                )

        if current_price is None or current_price <= 0:
            raise ValueError(
                f"Cannot calculate position size: no valid price available "
                f"(stop_loss_price={signal.stop_loss_price})"
            )

        # Assume standard forex lot (100,000 units)
        lot_size = 100000

        # Calculate lots
        lots = notional / (current_price * lot_size)

        # Round to 0.01 (mini lot increments)
        lots = round(lots, 2)

        # Safety check: ensure lots is within reasonable bounds
        min_lots = 0.01
        max_lots = 10.0  # Hard cap to prevent catastrophic errors

        if lots < min_lots:
            logger.warning(f"Calculated lots {lots} below minimum, using {min_lots}")
            lots = min_lots
        elif lots > max_lots:
            logger.warning(f"Calculated lots {lots} above maximum, capping at {max_lots}")
            lots = max_lots

        return lots

    def _get_price_range_for_symbol(self, symbol: str) -> tuple[float, float]:
        """Get realistic price range for a currency pair.

        Args:
            symbol: Currency pair symbol (e.g., EURUSD, USDJPY)

        Returns:
            Tuple of (min_price, max_price) for the currency pair
        """
        # Normalize symbol to uppercase
        symbol = symbol.upper()

        # JPY pairs have much higher prices (around 100-150)
        if "JPY" in symbol:
            return (50.0, 200.0)

        # Major pairs with quotes near parity (EUR/USD, GBP/USD, etc.)
        major_pairs = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]
        if symbol in major_pairs:
            return (0.5, 2.0)

        # CHF pairs (typically 0.8-1.2 range)
        if "CHF" in symbol:
            return (0.5, 1.5)

        # CAD pairs (typically 1.0-1.5 range)
        if "CAD" in symbol:
            return (0.8, 1.8)

        # Default fallback for other pairs
        return (0.1, 10.0)

    def _store_trade(
        self,
        signal: TradingSignal,
        entry_price: float,
        quantity: float,
        mt5_ticket: int,
    ) -> int:
        """Store trade in database.

        Args:
            signal: Trading signal
            entry_price: Entry price
            quantity: Position size in lots
            mt5_ticket: MT5 order ticket number

        Returns:
            Trade ID

        Raises:
            Exception: If database operation fails
        """
        session = None
        try:
            session = self.db_session_factory()

            trade = Trade(
                symbol=signal.symbol,
                direction="long" if signal.action == Action.BUY else "short",
                entry_price=entry_price,
                entry_time=datetime.now(),
                lot_size=quantity,
                confidence=signal.confidence,
                stop_loss=signal.stop_loss_price,
                take_profit=signal.take_profit_price,
                execution_mode=self.config.mode,
                broker="mt5",
                mt5_ticket=mt5_ticket,
                status="open",
            )

            session.add(trade)
            session.commit()

            return trade.id

        except Exception as e:
            if session:
                session.rollback()
            raise e
        finally:
            if session:
                session.close()

    def _update_trade_exit(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        """Update trade with exit information.

        Args:
            trade_id: Trade ID
            exit_price: Exit price
            exit_reason: Reason for exit
        """
        session = None
        try:
            session = self.db_session_factory()

            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if trade:
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.exit_time = datetime.now()
                trade.status = "closed"

                # Calculate PnL in pips
                if trade.direction == "long":
                    pips = (exit_price - trade.entry_price) * 10000  # For 4-digit pairs
                else:
                    pips = (trade.entry_price - exit_price) * 10000

                trade.pips = pips
                trade.is_winner = pips > 0

                # Calculate PnL in USD (lot_size * pip_value * pips)
                # Standard lot = 100,000 units, pip value ~= $10 for EURUSD
                pip_value = 10.0
                trade.pnl_usd = trade.lot_size * pip_value * pips

                session.commit()

        except Exception as e:
            if session:
                session.rollback()
            logger.error(f"Failed to update trade exit: {e}")
        finally:
            if session:
                session.close()

    def get_open_trade_count(self) -> int:
        """Get number of open trades.

        Returns:
            Number of open trades
        """
        return len(self._open_trades)

    def get_open_trades(self) -> List[dict]:
        """Get list of open trades.

        Returns:
            List of open trade dictionaries
        """
        return list(self._open_trades.values())

    def get_orphaned_trades(self) -> List[dict]:
        """Get list of orphaned trades awaiting reconciliation.

        Thread-safe.

        Returns:
            List of orphaned trade dictionaries
        """
        with self._orphaned_lock:
            return list(self._orphaned_trades)

    def get_orphaned_trade_count(self) -> int:
        """Get count of orphaned trades.

        Thread-safe.

        Returns:
            Number of orphaned trades
        """
        with self._orphaned_lock:
            return len(self._orphaned_trades)

    async def _retry_store_orphaned_trade(self, orphaned_trade: dict) -> Optional[int]:
        """Retry storing an orphaned trade in the database.

        Args:
            orphaned_trade: Orphaned trade info

        Returns:
            Trade ID if successful, None otherwise
        """
        orphaned_trade["retry_count"] += 1
        max_retries = 3

        if orphaned_trade["retry_count"] > max_retries:
            logger.error(
                f"Max retries ({max_retries}) exceeded for orphaned trade: "
                f"MT5 ticket={orphaned_trade['mt5_ticket']}"
            )
            return None

        # Wait before retry (exponential backoff)
        await asyncio.sleep(2 ** orphaned_trade["retry_count"])

        # Wrap synchronous DB operation in thread to avoid blocking event loop
        def _store_in_db():
            session = None
            try:
                session = self.db_session_factory()

                trade = Trade(
                    symbol=orphaned_trade["symbol"],
                    direction="long" if orphaned_trade["side"] == "buy" else "short",
                    entry_price=orphaned_trade["entry_price"],
                    entry_time=orphaned_trade["entry_time"],
                    lot_size=orphaned_trade["quantity"],
                    confidence=orphaned_trade["confidence"],
                    stop_loss=orphaned_trade["stop_loss"],
                    take_profit=orphaned_trade["take_profit"],
                    execution_mode=self.config.mode,
                    broker="mt5",
                    mt5_ticket=orphaned_trade["mt5_ticket"],
                    status="open",
                )

                session.add(trade)
                session.commit()

                trade_id = trade.id

                logger.info(
                    f"Orphaned trade recovered: trade_id={trade_id}, "
                    f"MT5 ticket={orphaned_trade['mt5_ticket']}"
                )
                return trade_id

            except Exception as e:
                if session:
                    session.rollback()
                logger.warning(
                    f"Retry {orphaned_trade['retry_count']} failed for orphaned trade: {e}"
                )
                return None
            finally:
                if session:
                    session.close()

        try:
            trade_id = await asyncio.wait_for(
                asyncio.to_thread(_store_in_db),
                timeout=self._db_timeout
            )

            if trade_id:
                # Also add to open trades tracking
                self._open_trades[trade_id] = {
                    "trade_id": trade_id,
                    "mt5_ticket": orphaned_trade["mt5_ticket"],
                    "symbol": orphaned_trade["symbol"],
                    "direction": "long" if orphaned_trade["side"] == "buy" else "short",
                    "entry_price": orphaned_trade["entry_price"],
                    "quantity": orphaned_trade["quantity"],
                    "stop_loss_price": orphaned_trade["stop_loss"],
                    "take_profit_price": orphaned_trade["take_profit"],
                    "entry_time": orphaned_trade["entry_time"],
                    "max_bars": 24,
                }

            return trade_id

        except asyncio.TimeoutError:
            logger.warning(f"Retry timed out for orphaned trade after {self._db_timeout}s")
            return None

        except Exception as e:
            logger.warning(f"Retry failed for orphaned trade: {e}")
            return None

    async def reconcile_orphaned_trades(self, max_duration_seconds: float = 30.0) -> int:
        """Attempt to reconcile all orphaned trades.

        Should be called periodically (e.g., on agent startup or each cycle).
        Thread-safe with timeout protection.

        Args:
            max_duration_seconds: Maximum time to spend on reconciliation

        Returns:
            Number of trades successfully reconciled
        """
        with self._orphaned_lock:
            if not self._orphaned_trades:
                return 0
            # Copy list to avoid holding lock during async operations
            trades_to_reconcile = list(self._orphaned_trades)

        logger.info(f"Attempting to reconcile {len(trades_to_reconcile)} orphaned trades")
        reconciled = 0
        start_time = asyncio.get_event_loop().time()

        for orphaned_trade in trades_to_reconcile:
            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > max_duration_seconds:
                logger.warning(
                    f"Reconciliation timeout ({max_duration_seconds}s) reached, "
                    f"processed {reconciled} of {len(trades_to_reconcile)} trades"
                )
                break

            trade_id = await self._retry_store_orphaned_trade(orphaned_trade)
            if trade_id:
                with self._orphaned_lock:
                    if orphaned_trade in self._orphaned_trades:
                        self._orphaned_trades.remove(orphaned_trade)
                reconciled += 1

        if reconciled > 0:
            logger.info(f"Reconciled {reconciled} orphaned trades")

        with self._orphaned_lock:
            remaining = len(self._orphaned_trades)

        if remaining > 0:
            logger.warning(f"{remaining} orphaned trades still pending reconciliation")

        return reconciled
