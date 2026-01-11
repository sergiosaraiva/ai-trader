---
name: creating-api-endpoints
description: Creates FastAPI REST endpoints with Pydantic request/response models and proper validation. Use when adding new API routes for predictions, trading operations, or data retrieval. FastAPI/Pydantic stack.
---

# Creating API Endpoints

## Quick Reference

- Create router in `src/api/routes/` with `router = APIRouter()`
- Define Pydantic models for request/response with `Field()` validation
- Use `@router.post/get()` decorators with `response_model` parameter
- Include router in `src/api/main.py` with prefix and tags
- Async handlers: `async def endpoint_name(request: RequestModel) -> ResponseModel`

## When to Use

- Adding new prediction endpoints
- Creating trading operation APIs (orders, positions)
- Exposing model status or metrics
- Building data retrieval endpoints
- Adding health checks or monitoring endpoints

## When NOT to Use

- Internal service-to-service calls (use direct imports)
- Batch processing jobs (use CLI scripts)
- WebSocket real-time feeds (use different pattern)

## Implementation Guide with Decision Tree

```
What type of endpoint?
├─ POST (create/action) → Use request body with Pydantic model
│   └─ Needs validation? → Add Field(...) constraints
├─ GET (retrieve) → Use Query parameters
│   └─ Required param? → Field(..., description="...")
│   └─ Optional param? → Field(default=X, le=1000)
└─ Both request + query → Combine body model + Query params

Response type?
├─ Single object → response_model=ResponseModel
├─ List of objects → response_model=List[ResponseModel]
└─ Dict/status → response_model=Dict[str, Any]
```

## Examples

**Example 1: Router Setup and App Integration**

```python
# From: src/api/main.py:1-36
"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import predictions, trading, health


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="AI Assets Trader API",
        description="API for AI-powered trading predictions and management",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
    app.include_router(trading.router, prefix="/api/v1", tags=["Trading"])

    return app


app = create_app()
```

**Explanation**: Factory pattern for app creation. Group routers by domain with prefixes (e.g., `/api/v1`). Tags appear in OpenAPI docs.

**Example 2: Pydantic Request/Response Models**

```python
# From: src/api/routes/predictions.py:12-33
class PredictionRequest(BaseModel):
    """Request body for prediction."""

    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    timeframe: str = Field(default="1H", description="Timeframe for prediction")
    horizons: List[int] = Field(
        default=[1, 4, 12, 24], description="Prediction horizons"
    )


class PredictionResponse(BaseModel):
    """Response body for prediction."""

    symbol: str
    timestamp: datetime
    direction: str
    direction_probability: float
    confidence: float
    price_predictions: Dict[str, float]
    model_name: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""

    symbols: List[str]
    timeframe: str = "1H"
```

**Explanation**: Use `Field(...)` for required fields, `Field(default=X)` for optional. Add descriptions for OpenAPI docs. Response models define API contract.

**Example 3: POST Endpoint with Response Model**

```python
# From: src/api/routes/predictions.py:42-59
@router.post("/predictions", response_model=PredictionResponse)
async def get_prediction(request: PredictionRequest) -> PredictionResponse:
    """
    Get prediction for a symbol.

    Returns price direction and magnitude predictions.
    """
    # Placeholder - would load model and generate prediction
    return PredictionResponse(
        symbol=request.symbol,
        timestamp=datetime.now(),
        direction="neutral",
        direction_probability=0.5,
        confidence=0.0,
        price_predictions={},
        model_name="technical_ensemble",
        model_version="1.0.0",
    )
```

**Explanation**: `response_model` enables automatic validation and OpenAPI schema generation. Docstring becomes endpoint description.

**Example 4: Batch Endpoint Returning List**

```python
# From: src/api/routes/predictions.py:62-81
@router.post("/predictions/batch", response_model=List[PredictionResponse])
async def get_batch_predictions(
    request: BatchPredictionRequest,
) -> List[PredictionResponse]:
    """Get predictions for multiple symbols."""
    results = []
    for symbol in request.symbols:
        results.append(
            PredictionResponse(
                symbol=symbol,
                timestamp=datetime.now(),
                direction="neutral",
                direction_probability=0.5,
                confidence=0.0,
                price_predictions={},
                model_name="technical_ensemble",
                model_version="1.0.0",
            )
        )
    return results
```

**Explanation**: Use `List[ResponseModel]` for batch responses. Iterate over request items and build response list.

**Example 5: GET Endpoint with Query Parameters**

```python
# From: src/api/routes/predictions.py:84-95
@router.get("/predictions/history")
async def get_prediction_history(
    symbol: str = Query(..., description="Trading symbol"),
    limit: int = Query(default=100, le=1000, description="Number of records"),
) -> Dict[str, Any]:
    """Get historical predictions for a symbol."""
    return {
        "symbol": symbol,
        "predictions": [],
        "count": 0,
    }
```

**Explanation**: Use `Query(...)` for required query params, `Query(default=X, le=Y)` for optional with constraints. `le=1000` limits max value.

**Example 6: Status/Metrics Endpoint**

```python
# From: src/api/routes/predictions.py:97-108
@router.get("/models/status")
async def get_model_status() -> Dict[str, Any]:
    """Get status of loaded models."""
    return {
        "models": {
            "short_term": {"loaded": False, "version": None},
            "medium_term": {"loaded": False, "version": None},
            "long_term": {"loaded": False, "version": None},
            "ensemble": {"loaded": False, "version": None},
        },
    }
```

**Explanation**: Status endpoints return simple dicts. No need for complex response models for monitoring endpoints.

## Quality Checklist

- [ ] Router created with `router = APIRouter()`
- [ ] Request models use `Field(...)` for required, `Field(default=X)` for optional
- [ ] All fields have descriptions for OpenAPI docs
- [ ] Response model specified in decorator: `response_model=ResponseModel`
- [ ] Endpoint function is `async def`
- [ ] Function has type hints: `-> ResponseModel`
- [ ] Docstring describes endpoint purpose
- [ ] Router included in `src/api/main.py` with prefix and tags
- [ ] Query params use `Query()` with validation constraints

## Common Mistakes

- **Missing Field descriptions**: Poor API docs → Add `description="..."` to all Fields
- **Sync instead of async**: Blocks event loop → Use `async def` for all handlers
- **No response_model**: No validation/docs → Always specify `response_model=`
- **Hardcoded values**: Inflexible → Use request params or config

## Validation

- [ ] Pattern confirmed in `src/api/main.py:1-43`
- [ ] Request/Response models in `src/api/routes/predictions.py:12-40`
- [ ] Endpoint patterns in `src/api/routes/predictions.py:42-108`

## Related Skills

- [implementing-prediction-models](./implementing-prediction-models.md) - Models that power prediction endpoints
- [creating-dataclasses](../quality-testing/creating-dataclasses.md) - For internal DTOs
- [analyzing-trading-performance](../trading-domain/analyzing-trading-performance.md) - For metrics endpoints
