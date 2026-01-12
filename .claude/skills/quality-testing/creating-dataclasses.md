---
name: creating-dataclasses
description: Creates Python dataclasses for structured data transfer objects (DTOs) with type hints, defaults, and serialization. Use when defining model outputs, API responses, configuration objects, or domain entities. Python 3.11+ dataclasses.
---

# Creating Dataclasses

## Quick Reference

- Use `@dataclass` decorator from `dataclasses` module
- Required fields first, optional fields with defaults after
- Use `field(default_factory=list)` for mutable defaults (lists, dicts)
- Add `to_dict()` method for JSON serialization
- Use `@dataclass(frozen=True)` for immutable objects

## When to Use

- Model prediction outputs
- API request/response schemas (internal)
- Configuration objects
- Trading signals and orders
- Backtest results and metrics
- Any structured data passed between components

## When NOT to Use

- API contracts (use Pydantic BaseModel instead)
- Simple key-value storage (use dict)
- Database models (use SQLAlchemy)
- Objects needing validation (use Pydantic)

## Implementation Guide with Decision Tree

```
What type of dataclass?
├─ Immutable → @dataclass(frozen=True)
│   └─ Use for: configs, constants, keys
├─ Mutable → @dataclass
│   └─ Use for: results, signals, state
└─ Inheriting → class Child(Parent)
    └─ Use for: EnsemblePrediction(Prediction)

Has mutable defaults (list, dict)?
├─ Yes → field(default_factory=list)
└─ No → field(default=value)

Needs serialization?
├─ Yes → Add to_dict() method
└─ No → Use asdict() from dataclasses
```

## Examples

**Example 1: Basic Dataclass Structure**

```python
# From: src/models/base.py:14-55
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional


@dataclass
class Prediction:
    """Model prediction output."""

    # Required fields (no defaults) - must come first
    timestamp: datetime
    symbol: str

    # Price predictions
    price_prediction: float
    price_predictions_multi: Dict[str, float] = field(default_factory=dict)

    # Direction
    direction: str = "neutral"  # bullish, bearish, neutral
    direction_probability: float = 0.5

    # Confidence
    confidence: float = 0.5
    prediction_lower: float = 0.0
    prediction_upper: float = 0.0

    # Model info
    model_name: str = ""
    model_version: str = ""
    prediction_horizon: int = 1
```

**Explanation**: Required fields without defaults come first. Optional fields have defaults. Use type hints for all fields.

**Example 2: Mutable Default with field(default_factory)**

```python
# From: src/simulation/backtester.py:17-51
from dataclasses import dataclass, field
from typing import List, Dict
import pandas as pd


@dataclass
class BacktestResult:
    """Results from backtesting."""

    # Basic info (required)
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float

    # Metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float

    # Mutable defaults - MUST use field(default_factory)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_series: pd.Series = field(default_factory=pd.Series)
    trade_history: List[Dict] = field(default_factory=list)
    signal_history: List[Dict] = field(default_factory=list)
```

**Explanation**: Lists, dicts, and other mutable types MUST use `field(default_factory=...)`. Never use `= []` or `= {}` as defaults.

**Example 3: to_dict() Serialization Method**

```python
# From: src/simulation/backtester.py:53-75
def to_dict(self) -> Dict[str, Any]:
    """Convert to dictionary for JSON serialization."""
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
    # Note: Excludes pandas Series (not JSON serializable)
```

**Explanation**: Convert datetime to isoformat(). Exclude non-serializable fields (pandas Series, numpy arrays). Handle nested objects recursively.

**Example 4: Dataclass Inheritance**

```python
# From: src/models/ensemble/combiner.py:12-26
from dataclasses import dataclass
from ..base import Prediction


@dataclass
class EnsemblePrediction(Prediction):
    """Extended prediction with ensemble-specific fields."""

    component_predictions: Dict[str, Prediction] = None
    component_weights: Dict[str, float] = None
    agreement_score: float = 0.0
    market_regime: str = "unknown"

    def __post_init__(self):
        """Initialize mutable defaults after creation."""
        if self.component_predictions is None:
            self.component_predictions = {}
        if self.component_weights is None:
            self.component_weights = {}
```

**Explanation**: Child inherits all parent fields. Use `__post_init__` for complex initialization. `None` default with post_init assignment is alternative to default_factory.

**Example 5: Configuration Dataclass**

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
    min_trade_interval: int = 300     # 5 minutes between trades

    # Signal filters
    min_confidence: float = 0.6
    min_signal_strength: float = 0.3
```

**Explanation**: Configuration dataclass with all defaults. Comments document units and meaning. Group related fields.

**Example 6: Trading Signal Dataclass**

```python
# From: src/trading/engine.py:22-35
@dataclass
class TradingSignal:
    """Trading signal from model predictions."""

    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL, HOLD
    strength: float  # 0.0 to 1.0
    confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reason: str = ""
    prediction: Optional[Prediction] = None
```

**Explanation**: Required trading info first (timestamp, symbol, action). Optional risk levels. Can embed other dataclasses (Prediction).

**Example 7: Frozen (Immutable) Dataclass**

```python
# Example of frozen dataclass for immutable data
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelIdentifier:
    """Immutable model identifier for registry keys."""

    name: str
    version: str

    def __str__(self) -> str:
        return f"{self.name}@{self.version}"


# Usage:
model_id = ModelIdentifier("short_term", "1.0.0")
# model_id.name = "other"  # Raises FrozenInstanceError
```

**Explanation**: `frozen=True` makes instance immutable (hashable, usable as dict key). Use for identifiers, keys, or constants.

**Example 8: Using asdict() for Quick Serialization**

```python
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class SimpleResult:
    name: str
    value: float
    timestamp: datetime

result = SimpleResult("test", 42.0, datetime.now())

# Quick conversion (but doesn't handle datetime)
data = asdict(result)
# {'name': 'test', 'value': 42.0, 'timestamp': datetime(...)}

# For JSON, still need to handle datetime manually
import json
def json_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

json_str = json.dumps(asdict(result), default=json_serializer)
```

**Explanation**: `asdict()` converts to dict recursively. For JSON with datetime, use custom serializer or explicit `to_dict()` method.

## Quality Checklist

- [ ] `@dataclass` decorator applied
- [ ] Required fields (no defaults) listed first
- [ ] Mutable defaults use `field(default_factory=...)`
- [ ] All fields have type hints
- [ ] Docstring describes purpose
- [ ] `to_dict()` method for serialization if needed
- [ ] `__post_init__` used for complex initialization
- [ ] Comments explain non-obvious fields

## Common Mistakes

- **Mutable default**: `trades: List = []` → Use `field(default_factory=list)`
- **Required after optional**: Error → Put required fields first
- **Missing type hint**: Poor IDE support → Add type hints to all fields
- **Non-serializable in to_dict**: JSON error → Handle datetime, exclude Series

## Validation

- [ ] Prediction pattern in `src/models/base.py:14-55`
- [ ] BacktestResult pattern in `src/simulation/backtester.py:17-75`
- [ ] RiskLimits pattern in `src/trading/risk.py:8-29`
- [ ] TradingSignal pattern in `src/trading/engine.py:22-35`

## Related Skills

- [implementing-prediction-models](../backend/implementing-prediction-models.md) - Returns Prediction dataclass
- [creating-fastapi-endpoints](../backend/SKILL.md) - Pydantic models for API
- [running-backtests](../trading-domain/running-backtests.md) - Returns BacktestResult dataclass
