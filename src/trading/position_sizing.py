"""Kelly Criterion Position Sizing Module.

Implements optimal position sizing using the Kelly criterion and its variants:
- Full Kelly: Maximizes log utility (geometric growth rate)
- Fractional Kelly: Reduces variance at cost of lower expected growth
- Confidence-Adjusted Kelly: Scales position by model confidence
- Regime-Adjusted Kelly: Modifies sizing based on market conditions

Reference: Kelly, J. L. (1956). "A New Interpretation of Information Rate"
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


class SizingStrategy(Enum):
    """Position sizing strategies."""
    FIXED = "fixed"                    # Fixed percentage of account
    FULL_KELLY = "full_kelly"          # Full Kelly criterion
    HALF_KELLY = "half_kelly"          # 50% of Kelly (most common)
    QUARTER_KELLY = "quarter_kelly"    # 25% of Kelly (conservative)
    CONFIDENCE_KELLY = "confidence_kelly"  # Kelly scaled by confidence
    REGIME_KELLY = "regime_kelly"      # Kelly adjusted for market regime


@dataclass
class KellyParameters:
    """Parameters for Kelly criterion calculation.

    The Kelly criterion formula is:
        f* = W - (1-W)/R

    Where:
        f* = Optimal fraction of capital to risk
        W = Win probability
        R = Win/Loss ratio (avg_win / avg_loss)

    For profit factor (PF) based calculation:
        R = PF * (1-W) / W
    """

    # Core parameters
    win_rate: float = 0.53          # Historical win probability
    profit_factor: float = 1.89     # Total profits / Total losses
    avg_win_pips: float = 22.8      # Average winning trade in pips
    avg_loss_pips: float = 14.1     # Average losing trade in pips

    # Derived
    win_loss_ratio: float = field(init=False)
    full_kelly: float = field(init=False)

    def __post_init__(self):
        """Calculate derived values."""
        # Win/Loss ratio from profit factor or direct calculation
        if self.avg_win_pips > 0 and self.avg_loss_pips > 0:
            self.win_loss_ratio = self.avg_win_pips / self.avg_loss_pips
        else:
            # Derive from profit factor: PF = W*R / (1-W) => R = PF*(1-W)/W
            self.win_loss_ratio = self.profit_factor * (1 - self.win_rate) / self.win_rate

        # Full Kelly calculation
        self.full_kelly = self._calculate_kelly(self.win_rate, self.win_loss_ratio)

    @staticmethod
    def _calculate_kelly(win_rate: float, win_loss_ratio: float) -> float:
        """Calculate Kelly fraction.

        Kelly % = W - (1-W) / R

        Where:
            W = Win probability
            R = Win/Loss ratio
        """
        if win_loss_ratio <= 0:
            return 0.0

        kelly = win_rate - (1 - win_rate) / win_loss_ratio

        # Kelly can be negative if expected value is negative
        return max(0.0, kelly)

    @classmethod
    def from_wfo_results(cls, wfo_path: Path) -> "KellyParameters":
        """Create parameters from WFO results file."""
        with open(wfo_path) as f:
            wfo = json.load(f)

        # Calculate aggregate statistics
        total_trades = sum(w["total_trades"] for w in wfo["windows"])
        total_wins = sum(
            int(w["total_trades"] * w["win_rate"] / 100)
            for w in wfo["windows"]
        )

        win_rate = total_wins / total_trades if total_trades > 0 else 0.5

        # Average profit factor across windows
        avg_pf = np.mean([w["profit_factor"] for w in wfo["windows"]])

        return cls(
            win_rate=win_rate,
            profit_factor=avg_pf,
        )

    @classmethod
    def from_backtest(
        cls,
        trades: List[Dict],
    ) -> "KellyParameters":
        """Create parameters from backtest trade list."""
        if not trades:
            return cls()

        wins = [t for t in trades if t.get("pnl_pips", 0) > 0]
        losses = [t for t in trades if t.get("pnl_pips", 0) <= 0]

        win_rate = len(wins) / len(trades) if trades else 0.5

        avg_win = np.mean([t["pnl_pips"] for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t["pnl_pips"] for t in losses])) if losses else 1

        total_profit = sum(t["pnl_pips"] for t in wins)
        total_loss = abs(sum(t["pnl_pips"] for t in losses))

        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        return cls(
            win_rate=win_rate,
            profit_factor=profit_factor if profit_factor != float("inf") else 10.0,
            avg_win_pips=avg_win,
            avg_loss_pips=avg_loss,
        )


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing."""

    # Strategy
    strategy: SizingStrategy = SizingStrategy.HALF_KELLY

    # Kelly parameters
    kelly_params: KellyParameters = field(default_factory=KellyParameters)

    # Fractional Kelly multipliers
    kelly_fraction: float = 0.5     # Default to half Kelly

    # Fixed sizing parameters
    fixed_risk_pct: float = 0.02    # 2% fixed risk per trade

    # Limits
    max_position_pct: float = 0.05  # Maximum 5% per position
    min_position_pct: float = 0.005 # Minimum 0.5% per position
    max_total_exposure: float = 0.20  # Maximum 20% total exposure
    max_lot_size: float = 10.0      # Maximum 10 lots per trade (realistic broker limit)
    min_lot_size: float = 0.01      # Minimum micro lot

    # Confidence scaling
    confidence_min: float = 0.55    # Minimum confidence for trading
    confidence_max: float = 0.85    # Confidence at which full size is used

    # Regime adjustments
    regime_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "trending": 1.2,    # Increase size in trending markets
        "ranging": 0.8,     # Decrease size in ranging markets
        "volatile": 0.6,    # Decrease size in volatile markets
        "unknown": 1.0,     # Default
    })


class KellyPositionSizer:
    """Position sizing using Kelly criterion and variants.

    The Kelly criterion provides the optimal bet size to maximize long-term
    geometric growth rate. However, full Kelly can be volatile, so fractional
    Kelly (typically 25-50%) is often used in practice.

    Example:
        >>> sizer = KellyPositionSizer(account_balance=100000)
        >>> size = sizer.calculate_position_size(
        ...     confidence=0.65,
        ...     stop_loss_pips=15.0,
        ... )
        >>> print(f"Position size: {size:.2f} lots")
    """

    def __init__(
        self,
        account_balance: float,
        config: Optional[PositionSizingConfig] = None,
    ):
        """Initialize position sizer.

        Args:
            account_balance: Current account balance in base currency
            config: Position sizing configuration
        """
        self.account_balance = account_balance
        self.initial_balance = account_balance
        self.config = config or PositionSizingConfig()

        # Track current exposure
        self.current_positions: Dict[str, float] = {}

        # Performance tracking for adaptive Kelly
        self.recent_trades: List[Dict] = []
        self.max_recent_trades = 100

        logger.info(f"Initialized KellyPositionSizer with {self.config.strategy.value}")
        logger.info(f"Full Kelly: {self.config.kelly_params.full_kelly:.2%}")
        logger.info(f"Kelly fraction: {self.config.kelly_fraction:.0%}")

    def calculate_position_size(
        self,
        confidence: float,
        stop_loss_pips: float,
        pip_value: float = 10.0,  # Value per pip per lot
        market_regime: str = "unknown",
        symbol: str = "EURUSD",
    ) -> Tuple[float, Dict[str, Any]]:
        """Calculate position size based on configured strategy.

        Args:
            confidence: Model confidence (0-1)
            stop_loss_pips: Stop loss distance in pips
            pip_value: Value of 1 pip per standard lot (default $10 for EURUSD)
            market_regime: Current market regime
            symbol: Trading symbol

        Returns:
            Tuple of (position_size_in_lots, sizing_details)
        """
        details = {
            "strategy": self.config.strategy.value,
            "account_balance": self.account_balance,
            "confidence": confidence,
            "stop_loss_pips": stop_loss_pips,
            "market_regime": market_regime,
        }

        # Check minimum confidence
        if confidence < self.config.confidence_min:
            details["skip_reason"] = "confidence below minimum"
            return 0.0, details

        # Calculate base Kelly fraction
        kelly_pct = self._get_kelly_fraction(confidence, market_regime)
        details["kelly_fraction"] = kelly_pct

        # Calculate risk amount
        risk_amount = self.account_balance * kelly_pct
        details["risk_amount"] = risk_amount

        # Calculate position size from risk and stop loss
        # Risk = Position Size * Stop Loss * Pip Value
        # Position Size = Risk / (Stop Loss * Pip Value)
        if stop_loss_pips > 0 and pip_value > 0:
            position_size = risk_amount / (stop_loss_pips * pip_value)
        else:
            position_size = 0.0

        details["raw_position_size"] = position_size

        # Apply position limits
        position_size = self._apply_limits(position_size, pip_value, symbol)
        details["final_position_size"] = position_size

        return position_size, details

    def _get_kelly_fraction(
        self,
        confidence: float,
        market_regime: str,
    ) -> float:
        """Get Kelly fraction based on strategy and adjustments."""
        params = self.config.kelly_params

        if self.config.strategy == SizingStrategy.FIXED:
            return self.config.fixed_risk_pct

        elif self.config.strategy == SizingStrategy.FULL_KELLY:
            base_kelly = params.full_kelly

        elif self.config.strategy == SizingStrategy.HALF_KELLY:
            base_kelly = params.full_kelly * 0.5

        elif self.config.strategy == SizingStrategy.QUARTER_KELLY:
            base_kelly = params.full_kelly * 0.25

        elif self.config.strategy == SizingStrategy.CONFIDENCE_KELLY:
            # Scale Kelly by confidence level
            base_kelly = params.full_kelly * self.config.kelly_fraction
            confidence_scale = self._confidence_scale(confidence)
            base_kelly *= confidence_scale

        elif self.config.strategy == SizingStrategy.REGIME_KELLY:
            # Scale Kelly by regime
            base_kelly = params.full_kelly * self.config.kelly_fraction
            regime_mult = self.config.regime_multipliers.get(market_regime, 1.0)
            base_kelly *= regime_mult

        else:
            base_kelly = params.full_kelly * self.config.kelly_fraction

        return base_kelly

    def _confidence_scale(self, confidence: float) -> float:
        """Scale factor based on model confidence.

        Linear scaling from 0 at min confidence to 1 at max confidence.
        """
        conf_range = self.config.confidence_max - self.config.confidence_min
        if conf_range <= 0:
            return 1.0

        scale = (confidence - self.config.confidence_min) / conf_range
        return min(max(scale, 0.0), 1.0)

    def _apply_limits(
        self,
        position_size: float,
        pip_value: float,
        symbol: str,
    ) -> float:
        """Apply position size limits."""
        # Minimum position
        min_size = self.config.min_lot_size

        # Maximum position from percentage
        max_size_pct = (self.account_balance * self.config.max_position_pct) / (15 * pip_value)

        # Maximum absolute lot size (realistic broker limit)
        max_size = min(max_size_pct, self.config.max_lot_size)

        # Check total exposure
        current_exposure = sum(self.current_positions.values())
        max_new_exposure = self.config.max_total_exposure - current_exposure

        if max_new_exposure <= 0:
            return 0.0

        # Apply limits
        position_size = max(position_size, min_size) if position_size > 0 else 0.0
        position_size = min(position_size, max_size)

        # Round to standard lot sizes (0.01 for micro lots)
        position_size = round(position_size, 2)

        return position_size

    def update_balance(self, new_balance: float) -> None:
        """Update account balance."""
        self.account_balance = new_balance

    def record_trade(self, trade: Dict) -> None:
        """Record trade for adaptive Kelly calculation."""
        self.recent_trades.append(trade)
        if len(self.recent_trades) > self.max_recent_trades:
            self.recent_trades.pop(0)

    def update_kelly_params(self) -> None:
        """Update Kelly parameters from recent trades."""
        if len(self.recent_trades) >= 20:
            self.config.kelly_params = KellyParameters.from_backtest(
                self.recent_trades
            )
            logger.info(
                f"Updated Kelly params: W={self.config.kelly_params.win_rate:.2%}, "
                f"R={self.config.kelly_params.win_loss_ratio:.2f}, "
                f"Kelly={self.config.kelly_params.full_kelly:.2%}"
            )

    def get_sizing_summary(self) -> Dict[str, Any]:
        """Get summary of current sizing configuration."""
        params = self.config.kelly_params
        return {
            "strategy": self.config.strategy.value,
            "kelly_params": {
                "win_rate": params.win_rate,
                "profit_factor": params.profit_factor,
                "win_loss_ratio": params.win_loss_ratio,
                "full_kelly": params.full_kelly,
            },
            "kelly_fraction": self.config.kelly_fraction,
            "effective_kelly": params.full_kelly * self.config.kelly_fraction,
            "max_position_pct": self.config.max_position_pct,
            "account_balance": self.account_balance,
        }


def calculate_kelly_from_stats(
    win_rate: float,
    profit_factor: float,
) -> Dict[str, float]:
    """Calculate Kelly fractions from win rate and profit factor.

    Args:
        win_rate: Historical win probability (0-1)
        profit_factor: Total profits / Total losses

    Returns:
        Dict with various Kelly fractions
    """
    # Calculate win/loss ratio from profit factor
    # PF = (W * avg_win) / ((1-W) * avg_loss)
    # PF = W * R / (1-W) where R = avg_win/avg_loss
    # R = PF * (1-W) / W

    if win_rate <= 0 or win_rate >= 1:
        return {"error": "Invalid win rate"}

    win_loss_ratio = profit_factor * (1 - win_rate) / win_rate

    # Kelly: f* = W - (1-W)/R
    full_kelly = win_rate - (1 - win_rate) / win_loss_ratio
    full_kelly = max(0.0, full_kelly)

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "win_loss_ratio": win_loss_ratio,
        "full_kelly": full_kelly,
        "half_kelly": full_kelly * 0.5,
        "quarter_kelly": full_kelly * 0.25,
        "expected_value_per_trade": win_rate * win_loss_ratio - (1 - win_rate),
    }


def compare_sizing_strategies(
    account_balance: float = 100000,
    trades: int = 1000,
    win_rate: float = 0.53,
    avg_win: float = 22.8,
    avg_loss: float = 14.1,
    strategies: Optional[List[SizingStrategy]] = None,
) -> Dict[str, Dict]:
    """Compare different sizing strategies via Monte Carlo simulation.

    Args:
        account_balance: Starting account balance
        trades: Number of trades to simulate
        win_rate: Historical win probability
        avg_win: Average winning trade (pips)
        avg_loss: Average losing trade (pips)
        strategies: List of strategies to compare

    Returns:
        Dict mapping strategy name to performance metrics
    """
    if strategies is None:
        strategies = [
            SizingStrategy.FIXED,
            SizingStrategy.QUARTER_KELLY,
            SizingStrategy.HALF_KELLY,
            SizingStrategy.FULL_KELLY,
        ]

    results = {}
    np.random.seed(42)  # For reproducibility

    # Create Kelly parameters
    kelly_params = KellyParameters(
        win_rate=win_rate,
        avg_win_pips=avg_win,
        avg_loss_pips=avg_loss,
    )

    pip_value = 10.0  # $10 per pip per lot
    stop_loss = 15.0  # pips

    for strategy in strategies:
        config = PositionSizingConfig(
            strategy=strategy,
            kelly_params=kelly_params,
            kelly_fraction=0.5 if strategy == SizingStrategy.HALF_KELLY else
                          0.25 if strategy == SizingStrategy.QUARTER_KELLY else 1.0,
        )

        sizer = KellyPositionSizer(
            account_balance=account_balance,
            config=config,
        )

        balance = account_balance
        peak_balance = balance
        max_drawdown = 0.0
        balance_history = [balance]

        for _ in range(trades):
            # Get position size
            size, _ = sizer.calculate_position_size(
                confidence=0.65,  # Average confidence
                stop_loss_pips=stop_loss,
                pip_value=pip_value,
            )

            if size <= 0:
                continue

            # Simulate trade outcome
            is_win = np.random.random() < win_rate
            pnl = size * (avg_win if is_win else -avg_loss) * pip_value

            balance += pnl
            sizer.update_balance(balance)
            balance_history.append(balance)

            # Track drawdown
            if balance > peak_balance:
                peak_balance = balance
            drawdown = (peak_balance - balance) / peak_balance
            max_drawdown = max(max_drawdown, drawdown)

            # Stop if ruined
            if balance <= 0:
                break

        # Calculate metrics
        final_return = (balance - account_balance) / account_balance
        cagr = (balance / account_balance) ** (252 / trades) - 1 if trades > 0 else 0

        results[strategy.value] = {
            "final_balance": balance,
            "total_return": final_return,
            "cagr_estimate": cagr,
            "max_drawdown": max_drawdown,
            "sharpe_estimate": final_return / max_drawdown if max_drawdown > 0 else 0,
            "ruined": balance <= 0,
        }

    return results
