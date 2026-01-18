---
name: database
description: This skill should be used when the user asks to "add a database table", "create a model", "define a schema", "add database persistence". Creates SQLAlchemy ORM models with proper column types, indexes, and relationships for SQLite persistence.
version: 1.1.0
---

# Creating SQLAlchemy Models

## Quick Reference

- Use `declarative_base()` for model base class
- Add `__tablename__` for explicit table name
- Use `index=True` on frequently queried columns
- Add composite indexes with `__table_args__`
- Include `created_at` and `updated_at` timestamps

## When to Use

- Persisting data to SQLite database
- Creating entities that need querying
- Storing historical records (predictions, trades)
- Defining relational data structures

## When NOT to Use

- Temporary/cache data (use in-memory dict)
- Configuration objects (use Pydantic/dataclass)
- API request/response data (use Pydantic schemas)

## Implementation Guide

```
Is this a new entity type?
├─ Yes → Create new model class in database/models.py
│   └─ Inherit from Base
│   └─ Define __tablename__
└─ No → Modify existing model

Does entity need querying by specific columns?
├─ Yes → Add index=True or composite Index
│   └─ Consider query patterns for index design
└─ No → Skip indexes (small tables only)

Does entity track changes over time?
├─ Yes → Add created_at and updated_at columns
│   └─ Use onupdate=datetime.utcnow for updates
└─ No → Add created_at only
```

## Examples

**Example 1: Basic Model with Indexes**

```python
# From: src/api/database/models.py:22-61
class Prediction(Base):
    """Store model predictions."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, default="EURUSD")

    # Prediction details
    direction = Column(String(10), nullable=False)  # "long" or "short"
    confidence = Column(Float, nullable=False)
    prob_up = Column(Float, nullable=True)
    prob_down = Column(Float, nullable=True)

    # Component model predictions
    pred_1h = Column(Integer, nullable=True)  # 0=down, 1=up
    conf_1h = Column(Float, nullable=True)
    pred_4h = Column(Integer, nullable=True)
    conf_4h = Column(Float, nullable=True)
    pred_d = Column(Integer, nullable=True)
    conf_d = Column(Float, nullable=True)

    # Agreement info
    agreement_count = Column(Integer, nullable=True)
    agreement_score = Column(Float, nullable=True)

    # Market context
    market_regime = Column(String(20), nullable=True)
    market_price = Column(Float, nullable=True)
    vix_value = Column(Float, nullable=True)

    # Whether a trade was executed based on this prediction
    trade_executed = Column(Boolean, default=False)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_predictions_timestamp_symbol", "timestamp", "symbol"),
    )
```

**Explanation**: Primary key with autoincrement. Single column index on timestamp. Composite index for common query pattern. Explicit nullability on all columns.

**Example 2: Model with Relationships and Status**

```python
# From: src/api/database/models.py:64-107
class Trade(Base):
    """Store paper trade history."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, nullable=True)  # Link to prediction
    symbol = Column(String(20), nullable=False, default="EURUSD")

    # Trade details
    direction = Column(String(10), nullable=False)  # "long" or "short"
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False, index=True)

    # Exit details (null if still open)
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime, nullable=True)
    exit_reason = Column(String(20), nullable=True)  # "tp", "sl", "timeout", "manual"

    # Position sizing
    lot_size = Column(Float, nullable=False, default=0.1)
    position_value = Column(Float, nullable=True)  # USD value

    # Risk parameters
    take_profit = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    max_holding_bars = Column(Integer, nullable=True)

    # Result (null if still open)
    pips = Column(Float, nullable=True)
    pnl_usd = Column(Float, nullable=True)
    is_winner = Column(Boolean, nullable=True)

    # Trade metadata
    confidence = Column(Float, nullable=True)
    status = Column(String(20), nullable=False, default="open")  # "open", "closed"

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_trades_status", "status"),
        Index("idx_trades_entry_time", "entry_time"),
    )
```

**Explanation**: Foreign key reference (without FK constraint for flexibility). Status column with default. updated_at with onupdate. Multiple indexes for different query patterns.

**Example 3: Snapshot/Metrics Model**

```python
# From: src/api/database/models.py:110-146
class PerformanceSnapshot(Base):
    """Store periodic performance snapshots."""

    __tablename__ = "performance_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Account metrics
    balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)

    # Trading metrics
    total_trades = Column(Integer, nullable=False, default=0)
    winning_trades = Column(Integer, nullable=False, default=0)
    losing_trades = Column(Integer, nullable=False, default=0)
    win_rate = Column(Float, nullable=True)

    # P&L metrics
    total_pips = Column(Float, nullable=False, default=0.0)
    total_pnl_usd = Column(Float, nullable=False, default=0.0)
    profit_factor = Column(Float, nullable=True)
    avg_pips_per_trade = Column(Float, nullable=True)

    # Risk metrics
    max_drawdown = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)

    # Prediction accuracy
    predictions_made = Column(Integer, nullable=False, default=0)
    predictions_correct = Column(Integer, nullable=False, default=0)
    prediction_accuracy = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("idx_performance_timestamp", "timestamp"),)
```

**Explanation**: Aggregated metrics with defaults. Nullable for computed values that may not exist. Time-series data with timestamp index.

**Example 4: Cache/Data Table**

```python
# From: src/api/database/models.py:149-168
class MarketData(Base):
    """Cache for market data (optional, for quick lookup)."""

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)

    timeframe = Column(String(10), nullable=False, default="5min")

    __table_args__ = (
        Index("idx_market_data_symbol_timestamp", "symbol", "timestamp"),
    )
```

**Explanation**: OHLCV data structure. Composite index for symbol+timestamp queries. Timeframe column for multi-resolution data.

**Example 5: Module Setup**

```python
# From: src/api/database/models.py:1-19
"""SQLAlchemy database models for AI-Trader."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Boolean,
    JSON,
    Enum as SQLEnum,
    Index,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()
```

**Explanation**: Import all column types used. Create Base once, all models inherit. JSON and Enum available for complex types.

## Quality Checklist

- [ ] Explicit `nullable=` on all columns
- [ ] Index on frequently queried columns
- [ ] Pattern matches `src/api/database/models.py:22-61`
- [ ] `created_at` with `default=datetime.utcnow`
- [ ] `updated_at` with `onupdate=datetime.utcnow` if needed
- [ ] Composite index in `__table_args__` for multi-column queries
- [ ] Descriptive docstring on model class

## Common Mistakes

- **Missing nullable specification**: Unclear intent
  - Wrong: `name = Column(String(100))`
  - Correct: `name = Column(String(100), nullable=False)`

- **Missing index on query columns**: Slow queries
  - Wrong: `timestamp = Column(DateTime, nullable=False)`
  - Correct: `timestamp = Column(DateTime, nullable=False, index=True)`

- **Mutable default**: Shared across instances
  - Wrong: `created_at = Column(DateTime, default=datetime.utcnow())`
  - Correct: `created_at = Column(DateTime, default=datetime.utcnow)` (no parens)

## Validation

- [ ] Pattern confirmed in `src/api/database/models.py:22-61`
- [ ] Model imported in `src/api/database/__init__.py`
- [ ] Table created by `init_db()` in `src/api/database/session.py`

## Related Skills

- `creating-pydantic-schemas` - API schemas that map to models
- `creating-fastapi-endpoints` - Endpoints that query models
- `writing-pytest-tests` - Test database operations

---

*Version 1.0.0 | Last verified: 2026-01-16 | Source: src/api/database/models.py*
