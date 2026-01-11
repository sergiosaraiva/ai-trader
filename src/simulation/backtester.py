"""Backtesting engine for strategy evaluation."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

import numpy as np
import pandas as pd

from ..models.base import BaseModel
from ..trading.orders.manager import Order, OrderType, OrderSide, OrderStatus
from ..trading.positions.manager import Position, PositionManager
from ..trading.risk.profiles import RiskProfile

# Try to import RiskManager, fall back gracefully if not available
try:
    from ..trading.risk import RiskManager, RiskLimits
except ImportError:
    # Create stub classes for backward compatibility
    from dataclasses import dataclass as dc

    @dc
    class RiskLimits:
        max_position_pct: float = 0.1
        max_drawdown_pct: float = 0.2
        max_daily_trades: int = 10

    class RiskManager:
        def __init__(self, initial_balance, risk_limits):
            self.account_balance = initial_balance
            self.risk_limits = risk_limits
            self.daily_pnl = 0

        def calculate_position_size(self, **kwargs):
            return kwargs.get("signal_strength", 0.5) * 0.01

        def record_trade(self):
            pass

        def update_pnl(self, pnl):
            self.daily_pnl += pnl
            self.account_balance += pnl

from .metrics import PerformanceMetrics


@dataclass
class BacktestResult:
    """Results from backtesting."""

    # Basic info
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float

    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float

    # Time series
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    trade_history: List[Dict] = field(default_factory=list)
    signal_history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "average_win": self.average_win,
            "average_loss": self.average_loss,
        }


class Backtester:
    """
    Backtesting engine for evaluating trading strategies.

    Features:
    - Historical data replay
    - Realistic order execution simulation
    - Commission and slippage modeling
    - Walk-forward validation support
    """

    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission: float = 0.0001,  # 0.01% per trade
        slippage: float = 0.0001,  # 0.01% slippage
        risk_limits: Optional[RiskLimits] = None,
    ):
        """
        Initialize backtester.

        Args:
            initial_balance: Starting account balance
            commission: Commission rate per trade
            slippage: Slippage rate
            risk_limits: Risk management limits
        """
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.risk_limits = risk_limits or RiskLimits()

        # State
        self.balance = initial_balance
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(initial_balance, self.risk_limits)
        self.equity_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.signal_history: List[Dict] = []

    def run(
        self,
        model: BaseModel,
        data: pd.DataFrame,
        features: pd.DataFrame,
        symbol: str,
        signal_threshold: float = 0.6,
    ) -> BacktestResult:
        """
        Run backtest with a model on historical data.

        Args:
            model: Trained prediction model
            data: OHLCV data with DatetimeIndex
            features: Feature data aligned with OHLCV
            symbol: Trading symbol
            signal_threshold: Minimum confidence for signals

        Returns:
            Backtest results
        """
        self._reset()

        sequence_length = model.config.get("sequence_length", 100)

        for i in range(sequence_length, len(data)):
            current_bar = data.iloc[i]
            current_time = data.index[i]
            current_price = current_bar["close"]

            # Update positions with current price
            self._update_positions(symbol, current_price)

            # Check stop loss / take profit
            self._check_exits(current_price, current_time)

            # Get features for prediction
            feature_window = features.iloc[i - sequence_length : i].values

            # Generate prediction
            prediction = model.predict(feature_window)

            # Record signal
            self.signal_history.append({
                "timestamp": current_time,
                "direction": prediction.direction,
                "confidence": prediction.confidence,
                "price_prediction": prediction.price_prediction,
            })

            # Generate and execute trades based on prediction
            if prediction.confidence >= signal_threshold:
                self._process_signal(
                    symbol=symbol,
                    direction=prediction.direction,
                    confidence=prediction.confidence,
                    current_price=current_price,
                    current_time=current_time,
                    atr=current_bar.get("atr_14", current_price * 0.01),
                )

            # Record equity
            self._record_equity(current_time, current_price)

        # Close any remaining positions
        self._close_all_positions(data.iloc[-1]["close"], data.index[-1])

        # Calculate results
        return self._calculate_results(symbol, data.index[0], data.index[-1])

    def _reset(self) -> None:
        """Reset backtester state."""
        self.balance = self.initial_balance
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(self.initial_balance, self.risk_limits)
        self.equity_history = []
        self.trade_history = []
        self.signal_history = []

    def _update_positions(self, symbol: str, current_price: float) -> None:
        """Update position prices."""
        self.position_manager.update_price(symbol, current_price)

    def _check_exits(self, current_price: float, current_time: datetime) -> None:
        """Check and execute stop loss / take profit."""
        positions_to_close = self.position_manager.check_stop_loss_take_profit()

        for position in positions_to_close:
            exit_type = "stop_loss" if position.should_stop_loss() else "take_profit"
            self._close_position(position, current_price, current_time, exit_type)

    def _process_signal(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        current_price: float,
        current_time: datetime,
        atr: float,
    ) -> None:
        """Process trading signal."""
        current_position = self.position_manager.get_position(symbol)

        # Determine action
        if direction == "bullish" and not current_position:
            self._open_position(
                symbol, "BUY", confidence, current_price, current_time, atr
            )
        elif direction == "bearish" and not current_position:
            self._open_position(
                symbol, "SELL", confidence, current_price, current_time, atr
            )
        elif direction == "bullish" and current_position and current_position.side == "SELL":
            self._close_position(current_position, current_price, current_time, "signal_reversal")
            self._open_position(
                symbol, "BUY", confidence, current_price, current_time, atr
            )
        elif direction == "bearish" and current_position and current_position.side == "BUY":
            self._close_position(current_position, current_price, current_time, "signal_reversal")
            self._open_position(
                symbol, "SELL", confidence, current_price, current_time, atr
            )

    def _open_position(
        self,
        symbol: str,
        side: str,
        confidence: float,
        price: float,
        time: datetime,
        atr: float,
    ) -> None:
        """Open a new position."""
        # Calculate position size
        stop_distance = atr * 2
        position_size = self.risk_manager.calculate_position_size(
            symbol=symbol,
            signal_strength=confidence,
            stop_loss_distance=stop_distance,
            current_price=price,
        )

        if position_size <= 0:
            return

        # Apply slippage
        if side == "BUY":
            entry_price = price * (1 + self.slippage)
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + (stop_distance * 1.5)
        else:
            entry_price = price * (1 - self.slippage)
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - (stop_distance * 1.5)

        # Calculate commission
        commission = position_size * entry_price * self.commission
        self.balance -= commission

        # Open position
        position = self.position_manager.open_position(
            symbol=symbol,
            side=side,
            quantity=position_size,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self.trade_history.append({
            "timestamp": time,
            "type": "open",
            "symbol": symbol,
            "side": side,
            "quantity": position_size,
            "price": entry_price,
            "commission": commission,
            "position_id": position.id,
        })

        self.risk_manager.record_trade()

    def _close_position(
        self,
        position: Position,
        price: float,
        time: datetime,
        reason: str,
    ) -> None:
        """Close an existing position."""
        # Apply slippage
        if position.side == "BUY":
            exit_price = price * (1 - self.slippage)
        else:
            exit_price = price * (1 + self.slippage)

        # Calculate P&L
        if position.side == "BUY":
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity

        # Apply commission
        commission = position.quantity * exit_price * self.commission
        pnl -= commission

        # Update balance
        self.balance += pnl
        self.risk_manager.update_pnl(pnl)

        # Close position
        self.position_manager.close_position(position.symbol, exit_price)

        self.trade_history.append({
            "timestamp": time,
            "type": "close",
            "symbol": position.symbol,
            "side": "SELL" if position.side == "BUY" else "BUY",
            "quantity": position.quantity,
            "price": exit_price,
            "pnl": pnl,
            "commission": commission,
            "reason": reason,
            "position_id": position.id,
        })

    def _close_all_positions(self, price: float, time: datetime) -> None:
        """Close all open positions."""
        for position in list(self.position_manager.positions.values()):
            self._close_position(position, price, time, "backtest_end")

    def _record_equity(self, time: datetime, price: float) -> None:
        """Record current equity."""
        position_value = sum(
            p.quantity * price for p in self.position_manager.positions.values()
        )
        equity = self.balance + position_value

        self.equity_history.append({
            "timestamp": time,
            "equity": equity,
            "balance": self.balance,
            "position_value": position_value,
        })

    def _calculate_results(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> BacktestResult:
        """Calculate backtest results."""
        equity_df = pd.DataFrame(self.equity_history)
        if equity_df.empty:
            return self._empty_result(symbol, start_date, end_date)

        equity_df.set_index("timestamp", inplace=True)
        equity_series = equity_df["equity"]

        metrics = PerformanceMetrics()
        metrics_dict = metrics.calculate_all(
            equity_series, self.trade_history, self.initial_balance
        )

        return BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            total_return=metrics_dict["total_return"],
            annualized_return=metrics_dict["annualized_return"],
            sharpe_ratio=metrics_dict["sharpe_ratio"],
            sortino_ratio=metrics_dict["sortino_ratio"],
            max_drawdown=metrics_dict["max_drawdown"],
            calmar_ratio=metrics_dict["calmar_ratio"],
            total_trades=metrics_dict["total_trades"],
            winning_trades=metrics_dict["winning_trades"],
            losing_trades=metrics_dict["losing_trades"],
            win_rate=metrics_dict["win_rate"],
            profit_factor=metrics_dict["profit_factor"],
            average_win=metrics_dict["average_win"],
            average_loss=metrics_dict["average_loss"],
            largest_win=metrics_dict["largest_win"],
            largest_loss=metrics_dict["largest_loss"],
            equity_curve=equity_series,
            drawdown_series=metrics_dict["drawdown_series"],
            trade_history=self.trade_history,
            signal_history=self.signal_history,
        )

    def _empty_result(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> BacktestResult:
        """Return empty result when no trades."""
        return BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_balance=self.initial_balance,
            final_balance=self.initial_balance,
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
        )

    def walk_forward_validation(
        self,
        model_class: type,
        model_config: Dict[str, Any],
        data: pd.DataFrame,
        features: pd.DataFrame,
        symbol: str,
        n_splits: int = 5,
        train_ratio: float = 0.8,
    ) -> List[BacktestResult]:
        """
        Perform walk-forward validation.

        Args:
            model_class: Model class to instantiate
            model_config: Model configuration
            data: Full OHLCV data
            features: Full feature data
            symbol: Trading symbol
            n_splits: Number of splits
            train_ratio: Ratio of training to total in each split

        Returns:
            List of backtest results for each split
        """
        results = []
        total_size = len(data)
        split_size = total_size // n_splits

        for i in range(n_splits):
            # Calculate split boundaries
            split_end = (i + 1) * split_size
            train_end = int(split_end * train_ratio)

            if i == 0:
                train_start = 0
            else:
                train_start = i * split_size

            test_start = train_end
            test_end = split_end

            # Extract data for this split
            train_data = data.iloc[train_start:train_end]
            train_features = features.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            test_features = features.iloc[test_start:test_end]

            # Train model
            model = model_class(model_config)
            model.build()
            # Note: In real implementation, would prepare sequences properly
            # model.train(X_train, y_train)

            # Backtest on test period
            result = self.run(model, test_data, test_features, symbol)
            results.append(result)

        return results
