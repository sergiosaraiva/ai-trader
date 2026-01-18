---
name: analyzing-trading-performance
description: Calculates and interprets trading performance metrics including Sharpe ratio, Sortino ratio, drawdown, and trade statistics. Use when evaluating backtest results or live trading performance. Python metrics framework.
version: 1.0.0
---

# Analyzing Trading Performance

## Quick Reference

- Use `PerformanceMetrics` class from `src/simulation/metrics.py`
- Key risk-adjusted metrics: Sharpe (> 1.5), Sortino (> 2.0), Calmar
- Drawdown metrics: max_drawdown (< 15%), drawdown_series
- Trade statistics: win_rate, profit_factor, expectancy
- Call `metrics.calculate_all(equity_series, trade_history, initial_balance)`

## When to Use

- Evaluating backtest results
- Comparing strategy performance
- Monitoring live trading performance
- Reporting to stakeholders
- Optimizing strategy parameters

## When NOT to Use

- Real-time P&L calculation (use RiskManager)
- Per-trade analysis (use trade_history directly)
- Simple return calculation (use pct_change)

## Implementation Guide with Decision Tree

```
What metric category?
├─ Return metrics
│   ├─ total_return → (final - initial) / initial
│   └─ annualized_return → Compound annual growth rate
├─ Risk-adjusted metrics
│   ├─ sharpe_ratio → excess_return / volatility
│   └─ sortino_ratio → excess_return / downside_deviation
├─ Drawdown metrics
│   ├─ max_drawdown → Largest peak-to-trough decline
│   └─ calmar_ratio → annualized_return / max_drawdown
└─ Trade statistics
    ├─ win_rate → winning_trades / total_trades
    ├─ profit_factor → gross_profit / gross_loss
    └─ expectancy → (win_rate * avg_win) + ((1-win_rate) * avg_loss)
```

## Examples

**Example 1: PerformanceMetrics Class**

```python
# From: src/simulation/metrics.py:1-28
"""Performance metrics calculation."""

from typing import Dict, List, Any
import numpy as np
import pandas as pd


class PerformanceMetrics:
    """Calculate trading performance metrics."""

    def __init__(self, risk_free_rate: float = 0.02, trading_days: int = 252):
        """
        Initialize performance metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino (default 2%)
            trading_days: Trading days per year (252 for stocks, 365 for forex)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
```

**Explanation**: Configure risk-free rate (use current T-bill rate) and trading days (252 stocks, 365 forex 24/7).

**Example 2: Calculate All Metrics**

```python
# From: src/simulation/metrics.py:22-62
def calculate_all(
    self,
    equity_series: pd.Series,
    trade_history: List[Dict],
    initial_balance: float,
) -> Dict[str, Any]:
    """
    Calculate all performance metrics.

    Args:
        equity_series: Time series of equity values (DatetimeIndex)
        trade_history: List of trade records with 'pnl' field
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
```

**Explanation**: Single method calculates everything. Returns dict with all metrics. Drawdown series useful for plotting.

**Example 3: Sharpe Ratio Calculation**

```python
# From: src/simulation/metrics.py:86-106
def sharpe_ratio(self, returns: pd.Series) -> float:
    """
    Calculate Sharpe ratio.

    Formula: (mean_excess_return / std_return) * sqrt(trading_days)

    Args:
        returns: Series of periodic returns (daily, hourly, etc.)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    # Daily risk-free rate
    excess_returns = returns - self.risk_free_rate / self.trading_days
    mean_excess = excess_returns.mean()
    std = excess_returns.std()

    if std == 0:
        return 0.0

    # Annualize
    return (mean_excess / std) * np.sqrt(self.trading_days)
```

**Explanation**: Sharpe measures risk-adjusted return. Target: > 1.5 for good strategies, > 2.0 for excellent. Annualized by sqrt(trading_days).

**Example 4: Sortino Ratio (Downside Risk)**

```python
# From: src/simulation/metrics.py:108-133
def sortino_ratio(self, returns: pd.Series) -> float:
    """
    Calculate Sortino ratio (uses downside deviation only).

    Better than Sharpe for strategies with asymmetric returns.
    Only penalizes downside volatility, not upside.

    Args:
        returns: Series of periodic returns

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - self.risk_free_rate / self.trading_days
    mean_excess = excess_returns.mean()

    # Only negative returns (downside)
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf") if mean_excess > 0 else 0.0

    downside_std = downside.std()
    if downside_std == 0:
        return 0.0

    return (mean_excess / downside_std) * np.sqrt(self.trading_days)
```

**Explanation**: Sortino only penalizes downside volatility. Better for asymmetric return distributions. Target: > 2.0.

**Example 5: Maximum Drawdown**

```python
# From: src/simulation/metrics.py:135-155
def max_drawdown(self, equity_series: pd.Series) -> tuple:
    """
    Calculate maximum drawdown.

    Drawdown = (equity - running_max) / running_max

    Args:
        equity_series: Time series of equity values

    Returns:
        Tuple of (max drawdown percentage, drawdown series)
    """
    if len(equity_series) == 0:
        return 0.0, pd.Series()

    # Running maximum (high water mark)
    running_max = equity_series.expanding().max()

    # Drawdown at each point
    drawdown = (equity_series - running_max) / running_max

    # Max drawdown (most negative value, returned as positive)
    max_dd = abs(drawdown.min())

    return max_dd, drawdown
```

**Explanation**: Drawdown measures peak-to-trough decline. Target: < 15% for conservative, < 25% for aggressive. Series useful for visualization.

**Example 6: Trade Statistics**

```python
# From: src/simulation/metrics.py:157-218
def trade_statistics(self, trade_history: List[Dict]) -> Dict[str, Any]:
    """
    Calculate trade statistics.

    Args:
        trade_history: List of trade records with 'type' and 'pnl' fields

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

    # Expectancy = (win_rate * avg_win) + ((1-win_rate) * avg_loss)
    expectancy = (win_rate * average_win) + ((1 - win_rate) * average_loss)

    return {
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_win": average_win,
        "average_loss": average_loss,
        "largest_win": max(wins) if wins else 0.0,
        "largest_loss": min(losses) if losses else 0.0,
        "expectancy": expectancy,
    }
```

**Explanation**: Trade stats require closed trades with 'pnl' field. Profit factor: target > 1.5. Expectancy: expected $ per trade.

**Example 7: Complete Analysis Workflow**

```python
# Complete performance analysis
from src.simulation.metrics import PerformanceMetrics
from src.simulation.backtester import Backtester
import pandas as pd

# After running backtest
result = backtester.run(model, df, features_df, "EURUSD")

# Or calculate manually
metrics = PerformanceMetrics(risk_free_rate=0.02, trading_days=252)
performance = metrics.calculate_all(
    equity_series=result.equity_curve,
    trade_history=result.trade_history,
    initial_balance=10000.0,
)

# Print report
print("=" * 50)
print("PERFORMANCE REPORT")
print("=" * 50)
print(f"\nRETURN METRICS")
print(f"  Total Return:      {performance['total_return']:>10.2%}")
print(f"  Annualized Return: {performance['annualized_return']:>10.2%}")

print(f"\nRISK-ADJUSTED METRICS")
print(f"  Sharpe Ratio:      {performance['sharpe_ratio']:>10.2f}")
print(f"  Sortino Ratio:     {performance['sortino_ratio']:>10.2f}")
print(f"  Calmar Ratio:      {performance['calmar_ratio']:>10.2f}")

print(f"\nDRAWDOWN METRICS")
print(f"  Max Drawdown:      {performance['max_drawdown']:>10.2%}")

print(f"\nTRADE STATISTICS")
print(f"  Total Trades:      {performance['total_trades']:>10d}")
print(f"  Win Rate:          {performance['win_rate']:>10.2%}")
print(f"  Profit Factor:     {performance['profit_factor']:>10.2f}")
print(f"  Expectancy:        ${performance['expectancy']:>9.2f}")
print(f"  Avg Win:           ${performance['average_win']:>9.2f}")
print(f"  Avg Loss:          ${performance['average_loss']:>9.2f}")

# Check targets
targets = {
    "Sharpe > 1.5": performance['sharpe_ratio'] > 1.5,
    "Drawdown < 15%": performance['max_drawdown'] < 0.15,
    "Win Rate > 55%": performance['win_rate'] > 0.55,
    "Profit Factor > 1.5": performance['profit_factor'] > 1.5,
}

print(f"\nTARGET CHECKS")
for target, met in targets.items():
    status = "PASS" if met else "FAIL"
    print(f"  {target}: {status}")
```

**Explanation**: Complete workflow from backtest result to formatted report. Check against targets for systematic evaluation.

## Quality Checklist

- [ ] risk_free_rate set to current rate (~2-5%)
- [ ] trading_days correct for asset (252 stocks, 365 forex)
- [ ] equity_series has DatetimeIndex
- [ ] trade_history has 'type' and 'pnl' fields
- [ ] Checked Sharpe > 1.5 target
- [ ] Checked max_drawdown < 15% target
- [ ] Minimum 50 trades for statistical significance

## Common Mistakes

- **Wrong trading_days**: Incorrect annualization → Use 252 for stocks, 365 for 24/7 markets
- **Ignoring Sortino**: Missing asymmetry info → Use Sortino for non-normal returns
- **Only looking at returns**: Missing risk → Always check Sharpe and drawdown
- **Low trade count**: Not significant → Need 50+ trades minimum

## Validation

- [ ] Pattern confirmed in `src/simulation/metrics.py:1-266`
- [ ] Sharpe calculation at lines 86-106
- [ ] Trade statistics at lines 157-218

## Related Skills

- [running-backtests](./running-backtests.md) - Generates results to analyze
- [implementing-risk-management](./implementing-risk-management.md) - Risk limits based on metrics
