"""Performance metrics calculation."""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


class PerformanceMetrics:
    """Calculate trading performance metrics."""

    def __init__(self, risk_free_rate: float = 0.02, trading_days: int = 252):
        """
        Initialize performance metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino
            trading_days: Trading days per year
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    def calculate_all(
        self,
        equity_series: pd.Series,
        trade_history: List[Dict],
        initial_balance: float,
    ) -> Dict[str, Any]:
        """
        Calculate all performance metrics.

        Args:
            equity_series: Time series of equity values
            trade_history: List of trade records
            initial_balance: Starting balance

        Returns:
            Dictionary of all metrics
        """
        # Return metrics
        total_return = self.total_return(equity_series, initial_balance)
        ann_return = self.annualized_return(equity_series, initial_balance)

        # Risk metrics
        returns = equity_series.pct_change().dropna()
        sharpe = self.sharpe_ratio(returns)
        sortino = self.sortino_ratio(returns)
        max_dd, dd_series = self.max_drawdown(equity_series)
        calmar = ann_return / max_dd if max_dd > 0 else 0

        # Trade metrics
        trade_stats = self.trade_statistics(trade_history)

        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "drawdown_series": dd_series,
            **trade_stats,
        }

    def total_return(self, equity_series: pd.Series, initial_balance: float) -> float:
        """Calculate total return."""
        if len(equity_series) == 0 or initial_balance == 0:
            return 0.0
        final = equity_series.iloc[-1]
        return (final - initial_balance) / initial_balance

    def annualized_return(
        self, equity_series: pd.Series, initial_balance: float
    ) -> float:
        """Calculate annualized return."""
        if len(equity_series) < 2:
            return 0.0

        total_return = self.total_return(equity_series, initial_balance)
        days = (equity_series.index[-1] - equity_series.index[0]).days
        if days <= 0:
            return 0.0

        years = days / 365.25
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    def sharpe_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Series of periodic returns

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / self.trading_days
        mean_excess = excess_returns.mean()
        std = excess_returns.std()

        if std == 0:
            return 0.0

        return (mean_excess / std) * np.sqrt(self.trading_days)

    def sortino_ratio(self, returns: pd.Series) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).

        Args:
            returns: Series of periodic returns

        Returns:
            Annualized Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - self.risk_free_rate / self.trading_days
        mean_excess = excess_returns.mean()

        # Downside deviation
        downside = returns[returns < 0]
        if len(downside) == 0:
            return float("inf") if mean_excess > 0 else 0.0

        downside_std = downside.std()
        if downside_std == 0:
            return 0.0

        return (mean_excess / downside_std) * np.sqrt(self.trading_days)

    def max_drawdown(self, equity_series: pd.Series) -> tuple:
        """
        Calculate maximum drawdown.

        Args:
            equity_series: Time series of equity values

        Returns:
            Tuple of (max drawdown percentage, drawdown series)
        """
        if len(equity_series) == 0:
            return 0.0, pd.Series()

        # Calculate running maximum
        running_max = equity_series.expanding().max()

        # Calculate drawdown
        drawdown = (equity_series - running_max) / running_max
        max_dd = abs(drawdown.min())

        return max_dd, drawdown

    def trade_statistics(self, trade_history: List[Dict]) -> Dict[str, Any]:
        """
        Calculate trade statistics.

        Args:
            trade_history: List of trade records

        Returns:
            Dictionary of trade statistics
        """
        # Filter closed trades with P&L
        closed_trades = [t for t in trade_history if t.get("type") == "close"]

        if not closed_trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "expectancy": 0.0,
            }

        pnls = [t.get("pnl", 0) for t in closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_trades = len(closed_trades)
        winning_trades = len(wins)
        losing_trades = len(losses)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        average_win = np.mean(wins) if wins else 0.0
        average_loss = np.mean(losses) if losses else 0.0

        largest_win = max(wins) if wins else 0.0
        largest_loss = min(losses) if losses else 0.0

        # Expectancy = (Win Rate × Average Win) - (Loss Rate × Average Loss)
        expectancy = (win_rate * average_win) + ((1 - win_rate) * average_loss)

        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "average_win": average_win,
            "average_loss": average_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "expectancy": expectancy,
        }

    def rolling_sharpe(
        self, returns: pd.Series, window: int = 60
    ) -> pd.Series:
        """Calculate rolling Sharpe ratio."""
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        daily_rf = self.risk_free_rate / self.trading_days

        return ((rolling_mean - daily_rf) / rolling_std) * np.sqrt(self.trading_days)

    def var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: Series of returns
            confidence: Confidence level (e.g., 0.95 for 95%)

        Returns:
            VaR as a positive number (potential loss)
        """
        if len(returns) == 0:
            return 0.0
        return abs(np.percentile(returns, (1 - confidence) * 100))

    def cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Args:
            returns: Series of returns
            confidence: Confidence level

        Returns:
            CVaR as a positive number
        """
        if len(returns) == 0:
            return 0.0

        var = self.var(returns, confidence)
        tail_losses = returns[returns <= -var]

        if len(tail_losses) == 0:
            return var

        return abs(tail_losses.mean())
