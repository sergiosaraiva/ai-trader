"""Position management module."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid


@dataclass
class Position:
    """Trading position."""

    symbol: str
    side: str  # BUY or SELL
    quantity: float
    entry_price: float
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    opened_at: datetime = field(default_factory=datetime.now)
    closed_at: Optional[datetime] = None
    is_open: bool = True

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.current_price == 0:
            return 0.0

        if self.side == "BUY":
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.entry_price == 0:
            return 0.0
        return self.unrealized_pnl / (self.entry_price * self.quantity)

    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return self.current_price * self.quantity

    def should_stop_loss(self) -> bool:
        """Check if stop loss should trigger."""
        if self.stop_loss is None:
            return False

        if self.side == "BUY":
            return self.current_price <= self.stop_loss
        else:
            return self.current_price >= self.stop_loss

    def should_take_profit(self) -> bool:
        """Check if take profit should trigger."""
        if self.take_profit is None:
            return False

        if self.side == "BUY":
            return self.current_price >= self.take_profit
        else:
            return self.current_price <= self.take_profit

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "market_value": self.market_value,
            "opened_at": self.opened_at.isoformat(),
            "is_open": self.is_open,
        }


class PositionManager:
    """
    Manages trading positions.

    Responsibilities:
    - Track open positions
    - Update position prices
    - Calculate portfolio metrics
    - Check stop loss / take profit
    """

    def __init__(self):
        """Initialize position manager."""
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []

    def open_position(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Position:
        """
        Open a new position.

        Args:
            symbol: Trading symbol
            side: Position side (BUY or SELL)
            quantity: Position size
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Created position
        """
        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self.positions[symbol] = position
        return position

    def close_position(
        self,
        symbol: str,
        close_price: Optional[float] = None,
    ) -> Optional[Position]:
        """
        Close a position.

        Args:
            symbol: Symbol to close
            close_price: Closing price

        Returns:
            Closed position or None
        """
        position = self.positions.get(symbol)
        if not position:
            return None

        if close_price:
            position.current_price = close_price

        position.is_open = False
        position.closed_at = datetime.now()

        self.closed_positions.append(position)
        del self.positions[symbol]

        return position

    def update_price(self, symbol: str, price: float) -> Optional[Position]:
        """Update position with current price."""
        position = self.positions.get(symbol)
        if position:
            position.current_price = price
        return position

    def update_all_prices(self, prices: Dict[str, float]) -> None:
        """Update all positions with current prices."""
        for symbol, price in prices.items():
            self.update_price(symbol, price)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol."""
        return self.positions.get(symbol)

    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.positions.values())

    def has_position(self, symbol: str) -> bool:
        """Check if position exists for symbol."""
        return symbol in self.positions

    def check_stop_loss_take_profit(self) -> List[Position]:
        """
        Check all positions for stop loss / take profit triggers.

        Returns:
            List of positions that should be closed
        """
        to_close = []

        for position in self.positions.values():
            if position.should_stop_loss():
                to_close.append(position)
            elif position.should_take_profit():
                to_close.append(position)

        return to_close

    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio-level metrics."""
        positions = list(self.positions.values())

        if not positions:
            return {
                "total_positions": 0,
                "total_market_value": 0.0,
                "total_unrealized_pnl": 0.0,
                "positions_by_side": {"BUY": 0, "SELL": 0},
            }

        total_value = sum(p.market_value for p in positions)
        total_pnl = sum(p.unrealized_pnl for p in positions)
        buy_count = sum(1 for p in positions if p.side == "BUY")
        sell_count = sum(1 for p in positions if p.side == "SELL")

        return {
            "total_positions": len(positions),
            "total_market_value": total_value,
            "total_unrealized_pnl": total_pnl,
            "average_pnl_pct": sum(p.unrealized_pnl_pct for p in positions) / len(positions),
            "positions_by_side": {"BUY": buy_count, "SELL": sell_count},
            "symbols": list(self.positions.keys()),
        }

    def get_closed_position_stats(self) -> Dict[str, Any]:
        """Calculate statistics for closed positions."""
        if not self.closed_positions:
            return {
                "total_closed": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_realized_pnl": 0.0,
            }

        winning = [p for p in self.closed_positions if p.unrealized_pnl > 0]
        losing = [p for p in self.closed_positions if p.unrealized_pnl < 0]
        total_pnl = sum(p.unrealized_pnl for p in self.closed_positions)

        return {
            "total_closed": len(self.closed_positions),
            "winning_trades": len(winning),
            "losing_trades": len(losing),
            "win_rate": len(winning) / len(self.closed_positions),
            "total_realized_pnl": total_pnl,
            "average_win": sum(p.unrealized_pnl for p in winning) / len(winning) if winning else 0,
            "average_loss": sum(p.unrealized_pnl for p in losing) / len(losing) if losing else 0,
        }
