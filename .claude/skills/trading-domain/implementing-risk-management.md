---
name: implementing-risk-management
description: Implements risk management with position sizing, loss limits, and circuit breakers using the RiskManager class. Use when adding trading safeguards, calculating position sizes, or implementing risk controls. Python trading framework.
version: 1.0.0
---

# Implementing Risk Management

## Quick Reference

- Use `RiskManager` from `src/trading/risk.py` with `RiskLimits` dataclass
- Key limits: max_position_size (2%), max_daily_loss (5%), max_drawdown (15%)
- Position sizing based on stop loss distance and signal strength
- Circuit breakers auto-halt trading when limits exceeded
- Call `check_signal()` before every trade, `update_pnl()` after

## When to Use

- Calculating position sizes for new trades
- Enforcing loss limits (daily, weekly, drawdown)
- Implementing circuit breakers for automated trading
- Pre-trade risk checks
- Real-time exposure monitoring

## When NOT to Use

- Backtesting metrics (use PerformanceMetrics)
- Simple stop-loss orders (use broker's native)
- Historical risk analysis (use separate analysis)

## Implementation Guide with Decision Tree

```
Risk check flow:
├─ Pre-trade → check_signal(signal)
│   ├─ Passes? → Calculate position size
│   └─ Fails? → Reject trade, log reason
├─ Position sizing → calculate_position_size()
│   └─ Returns 0? → Don't trade (limits exceeded)
├─ During trade → update_pnl(pnl)
│   └─ Triggers circuit breaker? → halt_trading()
└─ Recovery → resume_trading() (manual only)

Key limits to configure:
├─ max_position_size: 1-3% of account per trade
├─ max_total_exposure: 5-15% total open positions
├─ max_daily_loss: 2-5% daily loss halt
└─ max_drawdown: 10-20% total drawdown halt
```

## Examples

**Example 1: RiskLimits Configuration**

```python
# From: src/trading/risk.py:8-29
@dataclass
class RiskLimits:
    """Risk management limits."""

    # Position limits
    max_position_size: float = 0.02   # 2% of account per position
    max_total_exposure: float = 0.10  # 10% total exposure
    max_positions: int = 5            # Maximum concurrent positions

    # Loss limits
    max_daily_loss: float = 0.05      # 5% max daily loss
    max_weekly_loss: float = 0.10     # 10% max weekly loss
    max_drawdown: float = 0.15        # 15% max drawdown

    # Trade limits
    max_trades_per_day: int = 20
    min_trade_interval: int = 300     # 5 minutes between trades (seconds)

    # Signal filters
    min_confidence: float = 0.6
    min_signal_strength: float = 0.3
```

**Explanation**: Dataclass with sensible defaults. Override specific fields as needed. Position limits prevent oversizing, loss limits trigger halt.

**Example 2: RiskManager Initialization**

```python
# From: src/trading/risk.py:31-72
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
```

**Explanation**: Tracks balance, P&L, drawdown, and circuit breaker state. Initialize with account balance and custom limits.

**Example 3: Pre-Trade Signal Check**

```python
# From: src/trading/risk.py:74-110
def check_signal(self, signal: Any) -> bool:
    """
    Check if signal passes risk filters.

    Args:
        signal: Trading signal with confidence and strength

    Returns:
        True if signal is acceptable, False to reject
    """
    # Circuit breaker check
    if self.is_halted:
        return False

    # Confidence threshold
    if signal.confidence < self.limits.min_confidence:
        return False

    # Signal strength threshold
    if signal.strength < self.limits.min_signal_strength:
        return False

    # Daily trade count
    if self.trade_count_today >= self.limits.max_trades_per_day:
        return False

    # Minimum interval between trades
    if self.last_trade_time:
        elapsed = (datetime.now() - self.last_trade_time).seconds
        if elapsed < self.limits.min_trade_interval:
            return False

    # Position count limit
    if len(self.positions) >= self.limits.max_positions:
        if signal.symbol not in self.positions:
            return False

    return True
```

**Explanation**: Multiple filters: halted state, confidence, strength, trade count, interval, position count. Return False to reject trade.

**Example 4: Position Sizing**

```python
# From: src/trading/risk.py:112-166
def calculate_position_size(
    self,
    symbol: str,
    signal_strength: float,
    stop_loss_distance: float,
    current_price: Optional[float] = None,
) -> float:
    """
    Calculate position size based on risk parameters.

    Uses modified Kelly Criterion with risk limits.

    Args:
        symbol: Trading symbol
        signal_strength: Signal strength (0-1)
        stop_loss_distance: Distance to stop loss in price units
        current_price: Current price (optional)

    Returns:
        Position size (units/lots). Returns 0 if trade rejected.
    """
    if self.is_halted or stop_loss_distance <= 0:
        return 0.0

    # Maximum risk per trade (% of account)
    max_risk_pct = self.limits.max_position_size

    # Scale by signal strength
    risk_pct = max_risk_pct * signal_strength

    # Dollar amount to risk
    risk_amount = self.account_balance * risk_pct

    # Calculate position size from risk and stop loss
    if current_price and stop_loss_distance > 0:
        stop_loss_pct = stop_loss_distance / current_price
        position_value = risk_amount / stop_loss_pct
        position_size = position_value / current_price
    else:
        position_size = risk_amount / (stop_loss_distance or 1)

    # Check total exposure limit
    current_exposure = self._calculate_current_exposure()
    max_new_exposure = self.limits.max_total_exposure - current_exposure

    if max_new_exposure <= 0:
        return 0.0

    # Cap position by remaining exposure
    if current_price:
        max_position_size = (self.account_balance * max_new_exposure) / current_price
        position_size = min(position_size, max_position_size)

    return max(0.0, position_size)
```

**Explanation**: Position size based on risk per trade and stop loss. Scaled by signal strength. Capped by total exposure limit. Returns 0 if limits exceeded.

**Example 5: P&L Update and Circuit Breakers**

```python
# From: src/trading/risk.py:194-228
def update_pnl(self, pnl: float) -> None:
    """Update P&L tracking and check circuit breakers."""
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
```

**Explanation**: Call `update_pnl()` after every trade close. Automatically checks all circuit breakers. Sets `is_halted = True` to stop new trades.

**Example 6: Complete Risk Management Workflow**

```python
# Complete risk management workflow
from src.trading.risk import RiskManager, RiskLimits
from src.trading.engine import TradingSignal

# 1. Configure limits
limits = RiskLimits(
    max_position_size=0.02,     # 2% per trade
    max_total_exposure=0.10,    # 10% total
    max_daily_loss=0.05,        # 5% daily halt
    max_drawdown=0.15,          # 15% total halt
    min_confidence=0.6,         # 60% confidence min
)

# 2. Initialize manager
risk_manager = RiskManager(
    account_balance=10000.0,
    limits=limits,
)

# 3. Pre-trade check
signal = TradingSignal(
    timestamp=datetime.now(),
    symbol="EURUSD",
    action="BUY",
    strength=0.7,
    confidence=0.65,
)

if not risk_manager.check_signal(signal):
    print("Signal rejected by risk manager")
else:
    # 4. Calculate position size
    current_price = 1.0850
    stop_loss_distance = 0.0050  # 50 pips

    position_size = risk_manager.calculate_position_size(
        symbol="EURUSD",
        signal_strength=signal.strength,
        stop_loss_distance=stop_loss_distance,
        current_price=current_price,
    )

    if position_size > 0:
        print(f"Position size: {position_size:.2f} units")

        # 5. Execute trade...
        # After trade closes:
        pnl = 150.0  # Example profit
        risk_manager.update_pnl(pnl)
        risk_manager.record_trade()

        # 6. Check status
        metrics = risk_manager.get_risk_metrics()
        print(f"Daily P&L: ${metrics['daily_pnl']:.2f}")
        print(f"Drawdown: {metrics['current_drawdown']:.2%}")
        print(f"Halted: {metrics['is_halted']}")
    else:
        print("Position size is 0 - limits exceeded")

# 7. Reset daily counters (call at start of each day)
risk_manager.reset_daily_counters()
```

**Explanation**: Full workflow: configure limits, check signal, size position, update P&L, monitor status. Reset counters daily.

**Example 7: Risk Metrics Report**

```python
# From: src/trading/risk.py:255-269
def get_risk_metrics(self) -> Dict[str, Any]:
    """Get current risk metrics."""
    return {
        "account_balance": self.account_balance,
        "daily_pnl": self.daily_pnl,
        "daily_pnl_pct": self.daily_pnl / self.initial_balance,
        "weekly_pnl": self.weekly_pnl,
        "current_drawdown": self.current_drawdown,
        "peak_balance": self.peak_balance,
        "position_count": len(self.positions),
        "total_exposure": self._calculate_current_exposure(),
        "trade_count_today": self.trade_count_today,
        "is_halted": self.is_halted,
        "halt_reason": self.halt_reason,
    }
```

**Explanation**: Returns all risk metrics for monitoring/logging. Use for dashboard or alerts.

## Quality Checklist

- [ ] RiskLimits configured with appropriate values
- [ ] `check_signal()` called before every trade
- [ ] `calculate_position_size()` returns 0 handled (don't trade)
- [ ] `update_pnl()` called after every trade close
- [ ] `record_trade()` called to track trade count
- [ ] Daily counters reset at start of each day
- [ ] Circuit breaker triggers logged/alerted
- [ ] `is_halted` checked in trading loop

## Common Mistakes

- **Skipping check_signal**: Trades without risk check → Always call first
- **Ignoring 0 position size**: Trades when should be rejected → Check for 0
- **Not updating P&L**: Circuit breakers don't trigger → Call update_pnl()
- **Too high limits**: Excessive risk → Start with 2% position, 5% daily
- **Manual resume without review**: Continues after halt → Review before resume

## Validation

- [ ] Pattern confirmed in `src/trading/risk.py:1-270`
- [ ] RiskLimits at lines 8-29
- [ ] Position sizing at lines 112-166
- [ ] Circuit breakers at lines 213-228

## Related Skills

- [running-backtests](./running-backtests.md) - Uses RiskLimits in simulation
- [analyzing-trading-performance](./analyzing-trading-performance.md) - For post-trade risk analysis
