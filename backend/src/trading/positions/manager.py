"""
Position Management System.

Handles position tracking, PnL calculation, and exposure management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import logging

from ..orders.manager import Order, OrderSide, OrderStatus

logger = logging.getLogger(__name__)


class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class PositionStatus(Enum):
    """Position status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


@dataclass
class Position:
    """
    Trading position representation.

    Tracks a position in a single symbol with full PnL accounting.
    """
    position_id: str
    symbol: str
    side: PositionSide
    status: PositionStatus = PositionStatus.OPEN

    # Quantity and price
    quantity: float = 0.0
    average_entry_price: float = 0.0
    current_price: float = 0.0

    # Cost basis for accurate PnL
    total_cost: float = 0.0

    # PnL tracking
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0

    # Trade tracking
    entry_orders: List[str] = field(default_factory=list)
    exit_orders: List[str] = field(default_factory=list)
    trade_count: int = 0

    # Timestamps
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.status == PositionStatus.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if position is closed."""
        return self.status == PositionStatus.CLOSED

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == PositionSide.LONG

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == PositionSide.SHORT

    @property
    def market_value(self) -> float:
        """Get current market value of position."""
        return self.quantity * self.current_price

    @property
    def net_pnl(self) -> float:
        """Get net PnL including unrealized and realized."""
        return self.unrealized_pnl + self.realized_pnl - self.total_commission

    @property
    def pnl_percentage(self) -> float:
        """Get PnL as percentage of cost basis."""
        if self.total_cost == 0:
            return 0.0
        return (self.unrealized_pnl / self.total_cost) * 100

    def update_price(self, price: float) -> None:
        """
        Update current price and recalculate unrealized PnL.

        Args:
            price: Current market price
        """
        self.current_price = price
        self.last_updated = datetime.now()

        if self.quantity == 0:
            self.unrealized_pnl = 0.0
            return

        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (price - self.average_entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.average_entry_price - price) * self.quantity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "status": self.status.value,
            "quantity": self.quantity,
            "average_entry_price": self.average_entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "total_commission": self.total_commission,
            "net_pnl": self.net_pnl,
            "pnl_percentage": self.pnl_percentage,
            "trade_count": self.trade_count,
            "opened_at": self.opened_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
        }


class PositionManager:
    """
    Position management system.

    Tracks positions, calculates PnL, and manages exposure.
    """

    def __init__(
        self,
        get_price_callback: Optional[Callable[[str], float]] = None,
    ):
        """
        Initialize position manager.

        Args:
            get_price_callback: Callback to get current price for a symbol
        """
        self.get_price = get_price_callback or (lambda s: 1.0)

        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []

        # Trade history for PnL reconciliation
        self._position_counter = 0

        # Callbacks
        self._on_position_opened: List[Callable[[Position], None]] = []
        self._on_position_closed: List[Callable[[Position], None]] = []

        logger.info("PositionManager initialized")

    def _generate_position_id(self, symbol: str) -> str:
        """Generate unique position ID."""
        self._position_counter += 1
        return f"POS_{symbol}_{self._position_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    def process_fill(self, order: Order) -> Optional[Position]:
        """
        Process order fill and update positions.

        Args:
            order: Filled order

        Returns:
            Updated or new Position, or None if no position change
        """
        if order.status != OrderStatus.FILLED:
            return None

        symbol = order.symbol
        fill_quantity = order.filled_quantity
        fill_price = order.average_fill_price
        commission = order.commission

        # Determine side from order
        is_buy = order.side == OrderSide.BUY

        # Get or create position
        position = self.positions.get(symbol)

        if position is None:
            # Open new position
            position = Position(
                position_id=self._generate_position_id(symbol),
                symbol=symbol,
                side=PositionSide.LONG if is_buy else PositionSide.SHORT,
                quantity=fill_quantity,
                average_entry_price=fill_price,
                current_price=fill_price,
                total_cost=fill_quantity * fill_price,
                total_commission=commission,
            )
            position.entry_orders.append(order.order_id)
            position.trade_count = 1

            self.positions[symbol] = position

            logger.info(
                f"Opened new {position.side.value} position in {symbol}: "
                f"{fill_quantity:.2f} @ {fill_price:.5f}"
            )

            # Trigger callback
            for callback in self._on_position_opened:
                try:
                    callback(position)
                except Exception as e:
                    logger.error(f"Error in position opened callback: {e}")

            return position

        # Handle existing position
        same_direction = (
            (position.is_long and is_buy) or
            (position.is_short and not is_buy)
        )

        if same_direction:
            # Add to position
            old_cost = position.average_entry_price * position.quantity
            new_cost = fill_price * fill_quantity
            total_quantity = position.quantity + fill_quantity

            position.average_entry_price = (old_cost + new_cost) / total_quantity
            position.quantity = total_quantity
            position.total_cost = position.average_entry_price * position.quantity
            position.total_commission += commission
            position.entry_orders.append(order.order_id)
            position.trade_count += 1

            logger.info(
                f"Added to {position.side.value} position in {symbol}: "
                f"+{fill_quantity:.2f} @ {fill_price:.5f}, "
                f"total={position.quantity:.2f}"
            )

        else:
            # Reduce or close position
            position.exit_orders.append(order.order_id)
            position.total_commission += commission
            position.trade_count += 1

            if fill_quantity >= position.quantity:
                # Close position (possibly with reversal)
                closed_quantity = position.quantity
                remaining_quantity = fill_quantity - closed_quantity

                # Calculate realized PnL
                if position.is_long:
                    realized = (fill_price - position.average_entry_price) * closed_quantity
                else:
                    realized = (position.average_entry_price - fill_price) * closed_quantity

                position.realized_pnl += realized
                position.quantity = 0
                position.unrealized_pnl = 0
                position.status = PositionStatus.CLOSED
                position.closed_at = datetime.now()

                logger.info(
                    f"Closed {position.side.value} position in {symbol}: "
                    f"Realized PnL={realized:.4f}"
                )

                # Move to closed positions
                self.closed_positions.append(position)
                del self.positions[symbol]

                # Trigger callback
                for callback in self._on_position_closed:
                    try:
                        callback(position)
                    except Exception as e:
                        logger.error(f"Error in position closed callback: {e}")

                # Handle reversal (open opposite position with remaining)
                if remaining_quantity > 0:
                    new_position = Position(
                        position_id=self._generate_position_id(symbol),
                        symbol=symbol,
                        side=PositionSide.LONG if is_buy else PositionSide.SHORT,
                        quantity=remaining_quantity,
                        average_entry_price=fill_price,
                        current_price=fill_price,
                        total_cost=remaining_quantity * fill_price,
                    )
                    new_position.entry_orders.append(order.order_id)
                    new_position.trade_count = 1

                    self.positions[symbol] = new_position

                    logger.info(
                        f"Reversal: Opened new {new_position.side.value} position in {symbol}: "
                        f"{remaining_quantity:.2f} @ {fill_price:.5f}"
                    )

                    # Trigger callback
                    for callback in self._on_position_opened:
                        try:
                            callback(new_position)
                        except Exception as e:
                            logger.error(f"Error in position opened callback: {e}")

                    return new_position

            else:
                # Partial close
                # Calculate realized PnL for closed portion
                if position.is_long:
                    realized = (fill_price - position.average_entry_price) * fill_quantity
                else:
                    realized = (position.average_entry_price - fill_price) * fill_quantity

                position.realized_pnl += realized
                position.quantity -= fill_quantity
                position.total_cost = position.average_entry_price * position.quantity
                position.status = PositionStatus.PARTIAL

                logger.info(
                    f"Reduced {position.side.value} position in {symbol}: "
                    f"-{fill_quantity:.2f} @ {fill_price:.5f}, "
                    f"remaining={position.quantity:.2f}, realized={realized:.4f}"
                )

        # Update current price
        position.update_price(fill_price)

        return position

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.positions.values())

    def update_positions(self) -> None:
        """Update all positions with current prices."""
        for symbol, position in self.positions.items():
            try:
                current_price = self.get_price(symbol)
                position.update_price(current_price)
            except Exception as e:
                logger.error(f"Error updating price for {symbol}: {e}")

    def calculate_total_pnl(self) -> Dict[str, float]:
        """
        Calculate total PnL across all positions.

        Returns:
            Dictionary with unrealized, realized, and net PnL
        """
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        total_realized = (
            sum(p.realized_pnl for p in self.positions.values()) +
            sum(p.realized_pnl for p in self.closed_positions)
        )
        total_commission = (
            sum(p.total_commission for p in self.positions.values()) +
            sum(p.total_commission for p in self.closed_positions)
        )

        return {
            "unrealized_pnl": total_unrealized,
            "realized_pnl": total_realized,
            "total_commission": total_commission,
            "net_pnl": total_unrealized + total_realized - total_commission,
        }

    def calculate_exposure(self) -> Dict[str, float]:
        """
        Calculate portfolio exposure.

        Returns:
            Dictionary with exposure metrics
        """
        long_exposure = sum(
            p.market_value for p in self.positions.values()
            if p.is_long
        )
        short_exposure = sum(
            p.market_value for p in self.positions.values()
            if p.is_short
        )

        return {
            "long_exposure": long_exposure,
            "short_exposure": short_exposure,
            "gross_exposure": long_exposure + short_exposure,
            "net_exposure": long_exposure - short_exposure,
            "position_count": len(self.positions),
        }

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        commission: float = 0.0,
    ) -> Optional[Position]:
        """
        Close a position manually.

        Args:
            symbol: Symbol to close
            exit_price: Exit price
            commission: Commission for close

        Returns:
            Closed position or None if not found
        """
        position = self.positions.get(symbol)
        if not position:
            return None

        # Calculate realized PnL
        if position.is_long:
            realized = (exit_price - position.average_entry_price) * position.quantity
        else:
            realized = (position.average_entry_price - exit_price) * position.quantity

        position.realized_pnl += realized
        position.total_commission += commission
        position.quantity = 0
        position.unrealized_pnl = 0
        position.status = PositionStatus.CLOSED
        position.closed_at = datetime.now()

        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[symbol]

        logger.info(
            f"Manually closed {position.side.value} position in {symbol}: "
            f"Realized PnL={realized:.4f}"
        )

        # Trigger callback
        for callback in self._on_position_closed:
            try:
                callback(position)
            except Exception as e:
                logger.error(f"Error in position closed callback: {e}")

        return position

    def close_all_positions(
        self,
        get_price: Optional[Callable[[str], float]] = None,
    ) -> List[Position]:
        """
        Close all open positions.

        Args:
            get_price: Optional price callback (uses stored callback if not provided)

        Returns:
            List of closed positions
        """
        price_func = get_price or self.get_price
        closed = []

        for symbol in list(self.positions.keys()):
            try:
                price = price_func(symbol)
                position = self.close_position(symbol, price)
                if position:
                    closed.append(position)
            except Exception as e:
                logger.error(f"Error closing position for {symbol}: {e}")

        return closed

    def on_position_opened(self, callback: Callable[[Position], None]) -> None:
        """Register callback for position opened events."""
        self._on_position_opened.append(callback)

    def on_position_closed(self, callback: Callable[[Position], None]) -> None:
        """Register callback for position closed events."""
        self._on_position_closed.append(callback)

    def get_stats(self) -> Dict[str, Any]:
        """Get position manager statistics."""
        pnl = self.calculate_total_pnl()
        exposure = self.calculate_exposure()

        winning_trades = len([p for p in self.closed_positions if p.realized_pnl > 0])
        losing_trades = len([p for p in self.closed_positions if p.realized_pnl <= 0])
        total_trades = winning_trades + losing_trades

        return {
            "open_positions": len(self.positions),
            "closed_positions": len(self.closed_positions),
            **pnl,
            **exposure,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0.0,
        }

    def reset(self) -> None:
        """Reset position manager state."""
        self.positions.clear()
        self.closed_positions.clear()
        self._position_counter = 0
        logger.info("PositionManager state reset")
