---
name: running-backtests
description: Runs historical backtests using the Backtester class with realistic commission, slippage, and risk management. Use when evaluating trading strategies, comparing model performance, or validating signal generation. Python simulation framework.
version: 1.0.0
---

# Running Backtests

## Quick Reference

- Use `Backtester` class from `src/simulation/backtester.py`
- Configure: initial_balance, commission, slippage, risk_limits
- Run with: `backtester.run(model, data, features, symbol)`
- Returns `BacktestResult` dataclass with all metrics
- Walk-forward validation: `backtester.walk_forward_validation()`

## When to Use

- Evaluating model performance on historical data
- Comparing different model configurations
- Validating risk management rules
- Testing signal generation logic
- Walk-forward out-of-sample testing

## When NOT to Use

- Real-time trading (use TradingEngine)
- Paper trading simulation (use paper_trading.py)
- Simple indicator backtesting (use simpler loop)

## Implementation Guide with Decision Tree

```
What type of backtest?
├─ Single model evaluation → backtester.run()
│   └─ Returns BacktestResult with all metrics
├─ Walk-forward validation → backtester.walk_forward_validation()
│   └─ Returns List[BacktestResult] per split
└─ Parameter optimization → Loop over configs, compare results

Key metrics to check:
├─ Risk-adjusted: Sharpe ratio > 1.5, Sortino > 2.0
├─ Drawdown: max_drawdown < 15%
├─ Win rate: > 50% with positive profit_factor
└─ Trade count: Enough trades for statistical significance
```

## Examples

**Example 1: BacktestResult Dataclass Structure**

```python
# From: src/simulation/backtester.py:17-75
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
        """Convert to dictionary (excludes Series for JSON serialization)."""
        return {
            "symbol": self.symbol,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            # ... other fields
        }
```

**Explanation**: BacktestResult contains all metrics. Time series stored as pandas Series. Use `to_dict()` for JSON serialization.

**Example 2: Backtester Initialization**

```python
# From: src/simulation/backtester.py:77-116
class Backtester:
    """Backtesting engine for evaluating trading strategies."""

    def __init__(
        self,
        initial_balance: float = 10000.0,
        commission: float = 0.0001,  # 0.01% per trade
        slippage: float = 0.0001,    # 0.01% slippage
        risk_limits: Optional[RiskLimits] = None,
    ):
        """
        Initialize backtester.

        Args:
            initial_balance: Starting account balance
            commission: Commission rate per trade (0.0001 = 0.01%)
            slippage: Slippage rate (price impact)
            risk_limits: Risk management limits
        """
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.risk_limits = risk_limits or RiskLimits()

        # State (reset before each run)
        self.balance = initial_balance
        self.position_manager = PositionManager()
        self.risk_manager = RiskManager(initial_balance, self.risk_limits)
        self.equity_history: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.signal_history: List[Dict] = []
```

**Explanation**: Configure with realistic costs (commission, slippage). Risk limits control position sizing and loss limits.

**Example 3: Running a Backtest**

```python
# From: src/simulation/backtester.py:117-186
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

    return self._calculate_results(symbol, data.index[0], data.index[-1])
```

**Explanation**: Main loop iterates through data chronologically. Uses model.predict() for signals. Applies signal_threshold filter. Records equity at each step.

**Example 4: Walk-Forward Validation**

```python
# From: src/simulation/backtester.py:432-491
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

        train_start = i * split_size if i > 0 else 0
        test_start = train_end
        test_end = split_end

        # Extract data for this split
        train_data = data.iloc[train_start:train_end]
        train_features = features.iloc[train_start:train_end]
        test_data = data.iloc[test_start:test_end]
        test_features = features.iloc[test_start:test_end]

        # Train model on this split
        model = model_class(model_config)
        model.build()
        # model.train(X_train, y_train)  # Prepare sequences first

        # Backtest on test period
        result = self.run(model, test_data, test_features, symbol)
        results.append(result)

    return results
```

**Explanation**: Walk-forward tests on multiple out-of-sample periods. Each split: train on train_ratio, test on remainder. More realistic than single train/test split.

**Example 5: Complete Backtest Workflow**

```python
# Complete backtest workflow
from src.simulation.backtester import Backtester
from src.trading.risk import RiskLimits
from src.models import ModelRegistry
from src.data.processors.ohlcv import OHLCVProcessor
from src.features.technical import TechnicalIndicators
import pandas as pd

# 1. Load and prepare data
df = pd.read_csv("data/sample/EURUSD_daily.csv", parse_dates=["Date"], index_col="Date")
df.columns = [c.lower() for c in df.columns]

# 2. Add features
processor = OHLCVProcessor()
df = processor.clean(df)
df = processor.add_derived_features(df)

indicators = TechnicalIndicators()
features_df = indicators.calculate_all(df)
features_df = features_df.dropna()

# 3. Configure risk limits
risk_limits = RiskLimits(
    max_position_size=0.02,    # 2% per trade
    max_daily_loss=0.05,       # 5% daily loss limit
    max_drawdown=0.15,         # 15% max drawdown
    min_confidence=0.6,        # 60% confidence threshold
)

# 4. Initialize backtester
backtester = Backtester(
    initial_balance=10000.0,
    commission=0.0001,         # 1 pip commission
    slippage=0.0001,           # 1 pip slippage
    risk_limits=risk_limits,
)

# 5. Load trained model
model = ModelRegistry.create("short_term", {"sequence_length": 168})
model.build()
# model.load("path/to/weights")  # Load trained weights

# 6. Run backtest
result = backtester.run(
    model=model,
    data=df,
    features=features_df,
    symbol="EURUSD",
    signal_threshold=0.6,
)

# 7. Analyze results
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
print(f"Win Rate: {result.win_rate:.2%}")
print(f"Total Trades: {result.total_trades}")

# 8. Check if meets targets
targets_met = (
    result.sharpe_ratio > 1.5 and
    result.max_drawdown < 0.15 and
    result.win_rate > 0.55
)
print(f"Meets targets: {targets_met}")
```

**Explanation**: Full workflow from data to results. Configure realistic costs and risk limits. Check against performance targets.

**Example 6: Comparing Multiple Models**

```python
# Compare different model configurations
configs = [
    {"name": "short_term", "sequence_length": 168, "lstm_hidden_size": 128},
    {"name": "short_term", "sequence_length": 168, "lstm_hidden_size": 256},
    {"name": "short_term", "sequence_length": 336, "lstm_hidden_size": 256},
]

results = []
for config in configs:
    model = ModelRegistry.create("short_term", config)
    model.build()
    # model.train(...)  # Train each model

    result = backtester.run(model, df, features_df, "EURUSD")
    results.append({
        "config": config,
        "sharpe": result.sharpe_ratio,
        "return": result.total_return,
        "drawdown": result.max_drawdown,
    })

# Find best by Sharpe ratio
best = max(results, key=lambda x: x["sharpe"])
print(f"Best config: {best['config']}")
```

**Explanation**: Loop over configurations. Compare by Sharpe ratio (risk-adjusted return). Can extend to grid search.

## Quality Checklist

- [ ] Risk limits configured appropriately
- [ ] Commission and slippage set to realistic values
- [ ] signal_threshold matches model's confidence range
- [ ] Data and features aligned by index
- [ ] Model is trained before backtesting
- [ ] Results checked against performance targets
- [ ] Walk-forward validation used for robustness

## Common Mistakes

- **Zero commission/slippage**: Unrealistic results → Set to 0.0001+ each
- **No risk limits**: Oversized positions → Configure RiskLimits
- **Testing on training data**: Overfitting → Use walk-forward validation
- **Ignoring drawdown**: Risk of ruin → Check max_drawdown < 15%
- **Low trade count**: Not statistically significant → Need 50+ trades minimum

## Validation

- [ ] Pattern confirmed in `src/simulation/backtester.py:1-491`
- [ ] BacktestResult at lines 17-75
- [ ] Walk-forward at lines 432-491

## Related Skills

- [implementing-prediction-models](../backend/implementing-prediction-models.md) - For creating models to backtest
- [analyzing-trading-performance](./analyzing-trading-performance.md) - For interpreting backtest results
- [implementing-risk-management](./implementing-risk-management.md) - For risk limit configuration
