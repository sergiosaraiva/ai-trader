---
name: database
description: Creates SQLAlchemy ORM models with column types, indexes, relationships, and cascading for SQLite/PostgreSQL.
version: 1.3.0
---

# Creating SQLAlchemy Models

## Quick Reference

- Inherit from `Base` (from `declarative_base()`)
- Set `__tablename__` explicitly
- Use `index=True` on frequently queried columns
- Add composite indexes in `__table_args__`
- Include `created_at` (and `updated_at` if needed)
- Use `ForeignKey` with `ondelete` for relationships

## Decision Tree

```
New entity? → Create model in database/models.py, inherit from Base
Query by column? → Add index=True or composite Index
Has relationships? → ForeignKey with ondelete (SET NULL optional, CASCADE required)
Tracks changes? → Add created_at and updated_at with onupdate
```

## Pattern: Model with Indexes and ForeignKey

```python
# Reference: backend/src/api/database/models.py
class Prediction(Base):
    """Store model predictions."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, default="EURUSD")
    direction = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=False)

    # Component predictions
    pred_1h = Column(Integer, nullable=True)
    conf_1h = Column(Float, nullable=True)

    # Status flags
    trade_executed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_predictions_timestamp_symbol", "timestamp", "symbol"),
    )

class Trade(Base):
    """Store paper trade history."""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id", ondelete="SET NULL"), nullable=True)
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime, nullable=False, index=True)
    status = Column(String(20), nullable=False, default="open")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

## Pattern: Module Setup

```python
from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, ForeignKey, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()
```

## Pattern: Session Dependency

```python
# backend/src/api/database/session.py
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

## Quality Checklist

- [ ] Explicit `nullable=` on all columns
- [ ] `index=True` on frequently queried columns
- [ ] `created_at = Column(DateTime, default=datetime.utcnow)` (no parens!)
- [ ] `ForeignKey(..., ondelete="SET NULL")` for optional refs
- [ ] Composite indexes in `__table_args__`

## Common Mistakes

| Wrong | Correct |
|-------|---------|
| `Column(String(100))` | `Column(String(100), nullable=False)` |
| `default=datetime.utcnow()` (evaluated once) | `default=datetime.utcnow` (callable) |
| `ForeignKey("predictions.id")` | `ForeignKey("predictions.id", ondelete="SET NULL")` |

## Related Skills

- `creating-pydantic-schemas` - API schemas that map to models
- `backend` - Endpoints that query models

---
<!-- v1.3.0 | 2026-01-24 -->
