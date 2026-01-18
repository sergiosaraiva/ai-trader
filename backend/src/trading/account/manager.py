"""
Account Management System.

Handles balance, equity, margin, and daily PnL tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Callable
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    starting_equity: float
    ending_equity: float
    high_water_mark: float
    low_water_mark: float
    realized_pnl: float
    unrealized_pnl: float
    trade_count: int
    winning_trades: int
    losing_trades: int
    gross_profit: float
    gross_loss: float
    commission: float

    @property
    def net_pnl(self) -> float:
        """Net PnL for the day."""
        return self.realized_pnl + self.unrealized_pnl - self.commission

    @property
    def daily_return(self) -> float:
        """Daily return percentage."""
        if self.starting_equity == 0:
            return 0.0
        return ((self.ending_equity - self.starting_equity) / self.starting_equity) * 100

    @property
    def win_rate(self) -> float:
        """Win rate for the day."""
        total = self.winning_trades + self.losing_trades
        return self.winning_trades / total if total > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        """Profit factor (gross profit / gross loss)."""
        return self.gross_profit / abs(self.gross_loss) if self.gross_loss != 0 else float('inf')

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown from high water mark."""
        if self.high_water_mark == 0:
            return 0.0
        return ((self.high_water_mark - self.low_water_mark) / self.high_water_mark) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat(),
            "starting_equity": self.starting_equity,
            "ending_equity": self.ending_equity,
            "high_water_mark": self.high_water_mark,
            "low_water_mark": self.low_water_mark,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "net_pnl": self.net_pnl,
            "trade_count": self.trade_count,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "gross_profit": self.gross_profit,
            "gross_loss": self.gross_loss,
            "commission": self.commission,
            "daily_return": self.daily_return,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
        }


@dataclass
class AccountState:
    """Current account state snapshot."""
    timestamp: datetime
    balance: float
    equity: float
    margin_used: float
    margin_available: float
    unrealized_pnl: float
    realized_pnl: float

    @property
    def margin_level(self) -> float:
        """Margin level percentage."""
        if self.margin_used == 0:
            return float('inf')
        return (self.equity / self.margin_used) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "balance": self.balance,
            "equity": self.equity,
            "margin_used": self.margin_used,
            "margin_available": self.margin_available,
            "margin_level": self.margin_level,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
        }


class AccountManager:
    """
    Account management system.

    Tracks balance, equity, margin, and daily statistics.
    """

    def __init__(
        self,
        initial_balance: float = 100000.0,
        leverage: float = 1.0,
        margin_requirement: float = 0.01,  # 1% margin requirement
        state_file: Optional[str] = None,
    ):
        """
        Initialize account manager.

        Args:
            initial_balance: Starting account balance
            leverage: Account leverage
            margin_requirement: Margin requirement percentage
            state_file: Optional file path for state persistence
        """
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.margin_requirement = margin_requirement
        self.state_file = Path(state_file) if state_file else None

        # Account values
        self._balance = initial_balance
        self._unrealized_pnl = 0.0
        self._realized_pnl_today = 0.0
        self._commission_today = 0.0
        self._margin_used = 0.0

        # High/low tracking
        self._high_water_mark = initial_balance
        self._low_water_mark = initial_balance
        self._daily_high = initial_balance
        self._daily_low = initial_balance

        # Trade counters
        self._trades_today = 0
        self._winning_trades_today = 0
        self._losing_trades_today = 0
        self._gross_profit_today = 0.0
        self._gross_loss_today = 0.0

        # Daily stats history
        self.daily_stats: List[DailyStats] = []
        self._current_day: Optional[date] = None

        # State snapshots
        self._snapshots: List[AccountState] = []

        # Initialize current day
        self._initialize_day()

        logger.info(
            f"AccountManager initialized: balance=${initial_balance:,.2f}, "
            f"leverage={leverage}x"
        )

    def _initialize_day(self) -> None:
        """Initialize a new trading day."""
        today = date.today()

        if self._current_day != today:
            # Save previous day stats if exists
            if self._current_day is not None:
                self._save_daily_stats()

            # Reset daily counters
            self._current_day = today
            self._realized_pnl_today = 0.0
            self._commission_today = 0.0
            self._trades_today = 0
            self._winning_trades_today = 0
            self._losing_trades_today = 0
            self._gross_profit_today = 0.0
            self._gross_loss_today = 0.0
            self._daily_high = self.equity
            self._daily_low = self.equity

            logger.info(f"New trading day initialized: {today}")

    def _save_daily_stats(self) -> None:
        """Save daily statistics."""
        stats = DailyStats(
            date=self._current_day,
            starting_equity=self._daily_high,  # Approximate
            ending_equity=self.equity,
            high_water_mark=self._daily_high,
            low_water_mark=self._daily_low,
            realized_pnl=self._realized_pnl_today,
            unrealized_pnl=self._unrealized_pnl,
            trade_count=self._trades_today,
            winning_trades=self._winning_trades_today,
            losing_trades=self._losing_trades_today,
            gross_profit=self._gross_profit_today,
            gross_loss=self._gross_loss_today,
            commission=self._commission_today,
        )
        self.daily_stats.append(stats)

    @property
    def balance(self) -> float:
        """Get current balance (excluding unrealized PnL)."""
        return self._balance

    @property
    def equity(self) -> float:
        """Get current equity (balance + unrealized PnL)."""
        return self._balance + self._unrealized_pnl

    @property
    def margin_used(self) -> float:
        """Get margin currently in use."""
        return self._margin_used

    @property
    def margin_available(self) -> float:
        """Get available margin."""
        max_margin = self._balance * self.leverage
        return max(0.0, max_margin - self._margin_used)

    @property
    def margin_level(self) -> float:
        """Get margin level percentage."""
        if self._margin_used == 0:
            return float('inf')
        return (self.equity / self._margin_used) * 100

    @property
    def daily_pnl(self) -> float:
        """Get PnL for current day."""
        return self._realized_pnl_today + self._unrealized_pnl - self._commission_today

    @property
    def total_realized_pnl(self) -> float:
        """Get total realized PnL since start."""
        return self._balance - self.initial_balance

    @property
    def drawdown(self) -> float:
        """Get current drawdown from high water mark."""
        if self._high_water_mark == 0:
            return 0.0
        return ((self._high_water_mark - self.equity) / self._high_water_mark) * 100

    def update_unrealized_pnl(self, pnl: float) -> None:
        """
        Update unrealized PnL from position values.

        Args:
            pnl: Current total unrealized PnL
        """
        self._initialize_day()
        self._unrealized_pnl = pnl

        # Update high/low water marks
        current_equity = self.equity
        self._high_water_mark = max(self._high_water_mark, current_equity)
        self._daily_high = max(self._daily_high, current_equity)
        self._daily_low = min(self._daily_low, current_equity)
        self._low_water_mark = min(self._low_water_mark, current_equity)

    def update_margin(self, margin: float) -> None:
        """
        Update margin used.

        Args:
            margin: Current margin used
        """
        self._margin_used = margin

    def record_trade_result(
        self,
        realized_pnl: float,
        commission: float = 0.0,
    ) -> None:
        """
        Record result of a closed trade.

        Args:
            realized_pnl: Realized PnL from trade
            commission: Commission paid
        """
        self._initialize_day()

        # Update balance
        self._balance += realized_pnl - commission

        # Update daily counters
        self._realized_pnl_today += realized_pnl
        self._commission_today += commission
        self._trades_today += 1

        if realized_pnl > 0:
            self._winning_trades_today += 1
            self._gross_profit_today += realized_pnl
        else:
            self._losing_trades_today += 1
            self._gross_loss_today += realized_pnl

        # Update high water mark
        current_equity = self.equity
        self._high_water_mark = max(self._high_water_mark, current_equity)
        self._daily_high = max(self._daily_high, current_equity)
        self._daily_low = min(self._daily_low, current_equity)

        logger.debug(
            f"Trade recorded: PnL={realized_pnl:.2f}, Commission={commission:.2f}, "
            f"Balance=${self._balance:,.2f}"
        )

    def can_open_position(
        self,
        position_value: float,
    ) -> bool:
        """
        Check if account has sufficient margin for new position.

        Args:
            position_value: Notional value of proposed position

        Returns:
            True if position can be opened
        """
        required_margin = position_value * self.margin_requirement
        return required_margin <= self.margin_available

    def get_max_position_value(self) -> float:
        """Get maximum position value based on available margin."""
        return self.margin_available / self.margin_requirement

    def reset_daily_counters(self) -> None:
        """Manually reset daily counters (for simulation)."""
        self._save_daily_stats()
        self._current_day = None
        self._initialize_day()

    def take_snapshot(self) -> AccountState:
        """
        Take a snapshot of current account state.

        Returns:
            AccountState snapshot
        """
        snapshot = AccountState(
            timestamp=datetime.now(),
            balance=self._balance,
            equity=self.equity,
            margin_used=self._margin_used,
            margin_available=self.margin_available,
            unrealized_pnl=self._unrealized_pnl,
            realized_pnl=self._realized_pnl_today,
        )
        self._snapshots.append(snapshot)
        return snapshot

    def get_state(self) -> AccountState:
        """Get current account state without storing snapshot."""
        return AccountState(
            timestamp=datetime.now(),
            balance=self._balance,
            equity=self.equity,
            margin_used=self._margin_used,
            margin_available=self.margin_available,
            unrealized_pnl=self._unrealized_pnl,
            realized_pnl=self._realized_pnl_today,
        )

    def get_daily_stats(self, days: int = 30) -> List[DailyStats]:
        """
        Get daily stats for recent days.

        Args:
            days: Number of days to retrieve

        Returns:
            List of DailyStats
        """
        return self.daily_stats[-days:]

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive account statistics."""
        return {
            "balance": self._balance,
            "equity": self.equity,
            "unrealized_pnl": self._unrealized_pnl,
            "margin_used": self._margin_used,
            "margin_available": self.margin_available,
            "margin_level": self.margin_level,
            "high_water_mark": self._high_water_mark,
            "low_water_mark": self._low_water_mark,
            "current_drawdown": self.drawdown,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self._trades_today,
            "daily_win_rate": (
                self._winning_trades_today / self._trades_today
                if self._trades_today > 0 else 0.0
            ),
            "total_realized_pnl": self.total_realized_pnl,
            "initial_balance": self.initial_balance,
            "total_return_pct": (
                (self.equity - self.initial_balance) / self.initial_balance * 100
            ),
        }

    def save_state(self, path: Optional[str] = None) -> None:
        """
        Save account state to file.

        Args:
            path: Optional path override
        """
        save_path = Path(path) if path else self.state_file
        if not save_path:
            logger.warning("No state file configured")
            return

        state = {
            "timestamp": datetime.now().isoformat(),
            "balance": self._balance,
            "unrealized_pnl": self._unrealized_pnl,
            "margin_used": self._margin_used,
            "high_water_mark": self._high_water_mark,
            "low_water_mark": self._low_water_mark,
            "daily_stats": [s.to_dict() for s in self.daily_stats],
        }

        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Account state saved to {save_path}")

    def load_state(self, path: Optional[str] = None) -> bool:
        """
        Load account state from file.

        Args:
            path: Optional path override

        Returns:
            True if loaded successfully
        """
        load_path = Path(path) if path else self.state_file
        if not load_path or not load_path.exists():
            logger.warning(f"State file not found: {load_path}")
            return False

        try:
            with open(load_path) as f:
                state = json.load(f)

            self._balance = state['balance']
            self._unrealized_pnl = state.get('unrealized_pnl', 0.0)
            self._margin_used = state.get('margin_used', 0.0)
            self._high_water_mark = state.get('high_water_mark', self._balance)
            self._low_water_mark = state.get('low_water_mark', self._balance)

            logger.info(f"Account state loaded from {load_path}")
            return True

        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return False

    def reset(self) -> None:
        """Reset account to initial state."""
        self._balance = self.initial_balance
        self._unrealized_pnl = 0.0
        self._realized_pnl_today = 0.0
        self._commission_today = 0.0
        self._margin_used = 0.0
        self._high_water_mark = self.initial_balance
        self._low_water_mark = self.initial_balance
        self._daily_high = self.initial_balance
        self._daily_low = self.initial_balance
        self._trades_today = 0
        self._winning_trades_today = 0
        self._losing_trades_today = 0
        self._gross_profit_today = 0.0
        self._gross_loss_today = 0.0
        self.daily_stats.clear()
        self._snapshots.clear()
        self._current_day = None

        logger.info(f"AccountManager reset to initial balance ${self.initial_balance:,.2f}")
