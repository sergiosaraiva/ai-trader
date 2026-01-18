---
name: backend
description: This skill should be used when the user asks to "add an API endpoint", "create a REST route", "implement CRUD operations", "add a new route to the API". Creates FastAPI REST endpoints with proper error handling, Pydantic schemas, and service integration for the /api/v1 namespace.
version: 1.1.0
---

# Creating FastAPI Endpoints

## Quick Reference

- Use `APIRouter()` for modular routing, include in `main.py` with `app.include_router()`
- Always specify `response_model=` for type-safe responses and automatic documentation
- Use `Query()` for validated query parameters, `Depends()` for dependency injection
- Check service availability before operations (e.g., `model_service.is_loaded`)
- Re-raise `HTTPException`, catch and wrap other exceptions

## When to Use

- Adding new API endpoints under `/api/v1/`
- Creating CRUD operations for entities (predictions, trades, market data)
- Implementing paginated list endpoints
- Adding service status or health check endpoints
- Creating endpoints that integrate with singleton services

## When NOT to Use

- Internal service methods (use service classes instead)
- Background tasks (use scheduler or async tasks)
- WebSocket endpoints (different pattern)

## Implementation Guide

```
Is this a new route module?
├─ Yes → Create new file in src/api/routes/
│   └─ Add router = APIRouter() at top
│   └─ Include in src/api/main.py with app.include_router()
└─ No → Add to existing route file

Does endpoint need database access?
├─ Yes → Add db: Session = Depends(get_db) parameter
└─ No → Use service methods directly

Does endpoint need pagination?
├─ Yes → Add limit/offset Query parameters
│   └─ Return response with items, count, and total
└─ No → Return single response model
```

## Examples

**Example 1: GET Endpoint with Service Integration**

```python
# From: src/api/routes/predictions.py:26-78
@router.get("/predictions/latest", response_model=PredictionResponse)
async def get_latest_prediction() -> PredictionResponse:
    """Get the most recent prediction.

    Returns the latest prediction from the MTF Ensemble model.
    Predictions are generated every hour.
    """
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is initializing.",
        )

    try:
        # Get data for prediction
        df = data_service.get_data_for_prediction()
        if df is None or len(df) < 100:
            raise HTTPException(
                status_code=503,
                detail="Insufficient market data for prediction",
            )

        # Make prediction
        prediction = model_service.predict(df)

        return PredictionResponse(
            timestamp=prediction["timestamp"],
            symbol="EURUSD",
            direction=prediction["direction"],
            confidence=prediction["confidence"],
            # ... additional fields
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Explanation**: Service availability check first, then data validation, finally prediction. Re-raises HTTPException, wraps other errors.

**Example 2: GET Endpoint with Pagination**

```python
# From: src/api/routes/predictions.py:81-125
@router.get("/predictions/history", response_model=PredictionHistoryResponse)
async def get_prediction_history(
    limit: int = Query(default=50, le=500, description="Number of records"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
) -> PredictionHistoryResponse:
    """Get historical predictions.

    Returns past predictions stored in the database.
    """
    try:
        # Get total count
        total = db.query(Prediction).count()

        # Get predictions
        predictions = (
            db.query(Prediction)
            .order_by(Prediction.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        items = [
            PredictionHistoryItem(
                id=p.id,
                timestamp=p.timestamp.isoformat() if p.timestamp else "",
                symbol=p.symbol,
                direction=p.direction,
                confidence=p.confidence,
                market_price=p.market_price,
                trade_executed=p.trade_executed,
            )
            for p in predictions
        ]

        return PredictionHistoryResponse(
            predictions=items,
            count=len(items),
            total=total,
        )

    except Exception as e:
        logger.error(f"Error getting prediction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Explanation**: Uses `Query()` for pagination validation, `Depends(get_db)` for database session. Returns count and total for pagination UI.

**Example 3: Status/Info Endpoint**

```python
# From: src/api/routes/predictions.py:165-189
@router.get("/models/status", response_model=ModelStatusResponse)
async def get_model_status() -> ModelStatusResponse:
    """Get status of loaded models."""
    info = model_service.get_model_info()

    # Convert nested dicts for models
    models = {}
    if "models" in info and info["models"]:
        for tf, data in info["models"].items():
            models[tf] = {
                "trained": data.get("trained", False),
                "val_accuracy": data.get("val_accuracy"),
            }

    return ModelStatusResponse(
        loaded=info.get("loaded", False),
        model_dir=info.get("model_dir"),
        weights=info.get("weights"),
        agreement_bonus=info.get("agreement_bonus"),
        sentiment_enabled=info.get("sentiment_enabled", False),
        sentiment_by_timeframe=info.get("sentiment_by_timeframe", {}),
        models=models,
        initialized_at=info.get("initialized_at"),
        error=info.get("error"),
    )
```

**Explanation**: Simple status endpoint that transforms service info into response schema. Uses `.get()` with defaults for safety.

**Example 4: Router Setup Pattern**

```python
# From: src/api/routes/predictions.py:1-24
"""Prediction endpoints."""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session

from ..database.session import get_db
from ..database.models import Prediction
from ..services.data_service import data_service
from ..services.model_service import model_service
from ..schemas.prediction import (
    PredictionResponse,
    PredictionHistoryResponse,
    PredictionHistoryItem,
    ModelStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()
```

**Explanation**: Standard imports, logger setup, singleton service imports, schema imports, then router instantiation.

## Quality Checklist

- [ ] `response_model=` specified for all endpoints
- [ ] Query parameters validated with `Query(default=..., le=..., ge=...)`
- [ ] Service availability checked before operations
- [ ] Pattern matches `src/api/routes/predictions.py:26-78`
- [ ] HTTPException re-raised, other exceptions wrapped
- [ ] Descriptive docstring for Swagger documentation
- [ ] Router included in `src/api/main.py`

## Common Mistakes

- **Missing service check**: Always check `service.is_loaded` before calling methods
  - Wrong: `prediction = model_service.predict(df)`
  - Correct: Check `model_service.is_loaded` first, return 503 if not loaded

- **Catching HTTPException**: Don't wrap HTTPException in generic handler
  - Wrong: `except Exception as e: raise HTTPException(...)`
  - Correct: `except HTTPException: raise` then `except Exception as e: ...`

- **Missing response_model**: All endpoints should specify response type
  - Wrong: `async def get_data():`
  - Correct: `async def get_data() -> DataResponse:` with `response_model=DataResponse`

## Validation

- [ ] Pattern confirmed in `src/api/routes/predictions.py:26-78`
- [ ] Router included in `src/api/main.py:122`
- [ ] Tests exist in `tests/api/test_predictions.py`

## Related Skills

- `creating-pydantic-schemas` - Define request/response schemas
- `creating-python-services` - Implement business logic services
- `writing-pytest-tests` - Test API endpoints

---

*Version 1.0.0 | Last verified: 2026-01-16 | Source: src/api/routes/*
