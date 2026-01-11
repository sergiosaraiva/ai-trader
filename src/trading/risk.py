"""Risk management module."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any


@dataclass
class RiskLimits:
    """Risk management limits."""

    # Position limits
    max_position_size: float = 0.02  # 2% of account per position
    max_total_exposure: float = 0.10  # 10% total exposure
    max_positions: int = 5  # Maximum concurrent positions

    # Loss limits
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_weekly_loss: float = 0.10  # 10% max weekly loss
    max_drawdown: float = 0.15  # 15% max drawdown

    # Trade limits
    max_trades_per_day: int = 20
    min_trade_interval: int = 300  # 5 minutes between trades

    # Signal filters
    min_confidence: float = 0.6
    min_signal_strength: float = 0.3


class RiskManager:
    """
    Manages trading risk across the system.

    Responsibilities:
    - Position sizing based on risk parameters
    - Pre-trade risk checks
    - Real-time exposure monitoring
    - Drawdown tracking and circuit breakers
    """

    def __init__(
        self,
        account_balance: float,
        limits: Optional[RiskLimits] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize risk manager.

        Args:
            account_balance: Initial account balance
            limits: Risk limits configuration
            config: Additional configuration
        """
        self.account_balance = account_balance
        self.initial_balance = account_balance
        self.limits = limits or RiskLimits()
        self.config = config or {}

        # Tracking
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.peak_balance = account_balance
        self.current_drawdown = 0.0
        self.trade_count_today = 0
        self.last_trade_time: Optional[datetime] = None
        self.positions: Dict[str, Dict] = {}

        # Circuit breaker state
        self.is_halted = False
        self.halt_reason = ""

    def check_signal(self, signal: Any) -> bool:
        """
        Check if signal passes risk filters.

        Args:
            signal: Trading signal

        Returns:
            True if signal is acceptable
        """
        if self.is_halted:
            return False

        # Check confidence threshold
        if signal.confidence < self.limits.min_confidence:
            return False

        # Check signal strength
        if signal.strength < self.limits.min_signal_strength:
            return False

        # Check trade count
        if self.trade_count_today >= self.limits.max_trades_per_day:
            return False

        # Check trade interval
        if self.last_trade_time:
            elapsed = (datetime.now() - self.last_trade_time).seconds
            if elapsed < self.limits.min_trade_interval:
                return False

        # Check position count
        if len(self.positions) >= self.limits.max_positions:
            if signal.symbol not in self.positions:
                return False

        return True

    def calculate_position_size(
        self,
        symbol: str,
        signal_strength: float,
        stop_loss_distance: float,
        current_price: Optional[float] = None,
    ) -> float:
        """
        Calculate position size based on risk parameters.

        Uses Kelly Criterion modified with risk limits.

        Args:
            symbol: Trading symbol
            signal_strength: Signal strength (0-1)
            stop_loss_distance: Distance to stop loss in price units
            current_price: Current price (optional)

        Returns:
            Position size (units/lots)
        """
        if self.is_halted or stop_loss_distance <= 0:
            return 0.0

        # Maximum risk per trade (% of account)
        max_risk_pct = self.limits.max_position_size

        # Adjust by signal strength
        risk_pct = max_risk_pct * signal_strength

        # Calculate position size
        risk_amount = self.account_balance * risk_pct

        if current_price and stop_loss_distance > 0:
            # Calculate units based on risk and stop loss
            stop_loss_pct = stop_loss_distance / current_price
            position_value = risk_amount / stop_loss_pct
            position_size = position_value / current_price
        else:
            # Default calculation
            position_size = risk_amount / (stop_loss_distance or 1)

        # Check total exposure limit
        current_exposure = self._calculate_current_exposure()
        max_new_exposure = self.limits.max_total_exposure - current_exposure

        if max_new_exposure <= 0:
            return 0.0

        max_position_value = self.account_balance * max_new_exposure
        if current_price:
            max_position_size = max_position_value / current_price
            position_size = min(position_size, max_position_size)

        return max(0.0, position_size)

    def _calculate_current_exposure(self) -> float:
        """Calculate current total exposure as fraction of account."""
        total_value = sum(
            pos.get("value", 0) for pos in self.positions.values()
        )
        return total_value / self.account_balance if self.account_balance > 0 else 0

    def update_position(
        self,
        symbol: str,
        quantity: float,
        entry_price: float,
        current_price: float,
    ) -> None:
        """Update position tracking."""
        if quantity > 0:
            self.positions[symbol] = {
                "quantity": quantity,
                "entry_price": entry_price,
                "current_price": current_price,
                "value": quantity * current_price,
                "pnl": quantity * (current_price - entry_price),
            }
        elif symbol in self.positions:
            del self.positions[symbol]

    def update_pnl(self, pnl: float) -> None:
        """Update P&L tracking."""
        self.daily_pnl += pnl
        self.weekly_pnl += pnl
        self.account_balance += pnl

        # Update peak and drawdown
        if self.account_balance > self.peak_balance:
            self.peak_balance = self.account_balance

        self.current_drawdown = (
            (self.peak_balance - self.account_balance) / self.peak_balance
            if self.peak_balance > 0
            else 0
        )

        # Check circuit breakers
        self._check_circuit_breakers()

    def _check_circuit_breakers(self) -> None:
        """Check and activate circuit breakers if needed."""
        # Daily loss limit
        daily_loss_pct = -self.daily_pnl / self.initial_balance
        if daily_loss_pct >= self.limits.max_daily_loss:
            self.halt_trading(f"Daily loss limit reached: {daily_loss_pct:.1%}")

        # Weekly loss limit
        weekly_loss_pct = -self.weekly_pnl / self.initial_balance
        if weekly_loss_pct >= self.limits.max_weekly_loss:
            self.halt_trading(f"Weekly loss limit reached: {weekly_loss_pct:.1%}")

        # Max drawdown
        if self.current_drawdown >= self.limits.max_drawdown:
            self.halt_trading(f"Max drawdown reached: {self.current_drawdown:.1%}")

    def halt_trading(self, reason: str) -> None:
        """Halt all trading."""
        self.is_halted = True
        self.halt_reason = reason
        print(f"TRADING HALTED: {reason}")

    def resume_trading(self) -> None:
        """Resume trading after halt."""
        self.is_halted = False
        self.halt_reason = ""
        print("Trading resumed")

    def reset_daily_counters(self) -> None:
        """Reset daily tracking counters."""
        self.daily_pnl = 0.0
        self.trade_count_today = 0

    def reset_weekly_counters(self) -> None:
        """Reset weekly tracking counters."""
        self.weekly_pnl = 0.0

    def record_trade(self) -> None:
        """Record a trade execution."""
        self.trade_count_today += 1
        self.last_trade_time = datetime.now()

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics."""
        return {
            "account_balance": self.account_balance,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_pct": self.daily_pnl / self.initial_balance if self.initial_balance > 0 else 0,
            "weekly_pnl": self.weekly_pnl,
            "current_drawdown": self.current_drawdown,
            "peak_balance": self.peak_balance,
            "position_count": len(self.positions),
            "total_exposure": self._calculate_current_exposure(),
            "trade_count_today": self.trade_count_today,
            "is_halted": self.is_halted,
            "halt_reason": self.halt_reason,
        }
