---
name: backend
description: Creates FastAPI REST endpoints with proper error handling, Pydantic schemas, and service integration for /api/v1.
version: 1.3.0
---

# Creating FastAPI Endpoints

## Quick Reference

- Use `APIRouter()`, include in `main.py` with `app.include_router()`
- Specify `response_model=` for typed responses
- Check `service.is_loaded` before operations (return 503 if not ready)
- Use `log_exception()` for consistent error logging
- Re-raise `HTTPException`, catch and wrap other exceptions

## Decision Tree

```
New route module? → Create in src/api/routes/, add include_router() in main.py
Need pagination? → Add limit/offset Query params, return {items, count, total}
Service integration? → Check is_loaded first, 503 if not ready
```

## Pattern: GET with Service

```python
# Reference: backend/src/api/routes/performance.py
@router.get("/model/performance")
async def get_model_performance() -> Dict[str, Any]:
    """Get model performance metrics."""
    try:
        if not performance_service.is_loaded:
            if not performance_service.initialize():
                logger.warning("Failed to initialize performance service")
        return performance_service.get_performance_data()
    except Exception as e:
        log_exception(logger, "Error getting model performance", e)
        raise HTTPException(status_code=500, detail=str(e))
```

## Pattern: GET with Pagination

```python
# Reference: backend/src/api/routes/predictions.py
@router.get("/predictions/history", response_model=PredictionHistoryResponse)
async def get_prediction_history(
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> PredictionHistoryResponse:
    total = db.query(Prediction).count()
    predictions = db.query(Prediction).order_by(Prediction.timestamp.desc()).offset(offset).limit(limit).all()
    return PredictionHistoryResponse(predictions=[...], count=len(items), total=total)
```

## Pattern: Router Setup

```python
"""Module docstring."""
import logging
from fastapi import APIRouter, HTTPException
from ..services.my_service import my_service
from ..utils.logging import log_exception

logger = logging.getLogger(__name__)
router = APIRouter()
```

## Quality Checklist

- [ ] `response_model=` on typed endpoints
- [ ] Query params validated with `Query(default=, le=, ge=)`
- [ ] Service `is_loaded` checked
- [ ] Errors logged with `log_exception()`
- [ ] Router included in `main.py`

## Common Mistakes

| Wrong | Correct |
|-------|---------|
| `except Exception as e: raise HTTPException` (catches HTTPException) | Let HTTPException propagate, wrap only others |
| `data = service.get_data()` (no check) | Check `service.is_loaded` first |
| `logger.error(f"Error: {e}")` | `log_exception(logger, "Error", e)` |

## Related Skills

- `creating-pydantic-schemas` - Define request/response schemas
- `creating-python-services` - Implement business logic
- `testing` - Test API endpoints

---
<!-- v1.3.0 | 2026-01-24 -->
