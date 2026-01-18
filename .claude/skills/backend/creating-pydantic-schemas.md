---
name: creating-pydantic-schemas
description: This skill should be used when the user asks to "define a schema", "create a request model", "add response validation", "document API contracts". Creates Pydantic request/response schemas with Field descriptions, validation, and examples for FastAPI integration.
version: 1.1.0
---

# Creating Pydantic Schemas

## Quick Reference

- Use `Field(...)` for required fields, `Field(default=...)` for optional
- Add `description=` to every Field for Swagger documentation
- Include `json_schema_extra` with example in Config class
- Create separate schemas for request, response, and list responses
- Use `Optional[T]` for nullable fields

## When to Use

- Defining API request/response contracts
- Validating incoming data
- Generating OpenAPI documentation
- Data transfer between layers
- Configuration objects with validation

## When NOT to Use

- Internal data structures without validation needs
- Database models (use SQLAlchemy)
- Complex nested configurations (use dataclasses)

## Implementation Guide

```
Is this a response schema?
├─ Yes → Include Field descriptions for all fields
│   └─ Add Config with json_schema_extra example
└─ No (request) → Add validation constraints

Is this a list response?
├─ Yes → Create wrapper with items, count, total
│   └─ Create separate item schema
└─ No → Single entity response

Does field have nested objects?
├─ Yes → Create separate schema for nested type
│   └─ Use Dict[str, T] or List[T] as appropriate
└─ No → Use primitive types
```

## Examples

**Example 1: Response Schema with Field Descriptions**

```python
# From: src/api/schemas/prediction.py:16-68
class PredictionResponse(BaseModel):
    """Response for latest prediction endpoint."""

    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")
    symbol: str = Field(default="EURUSD", description="Trading symbol")
    direction: str = Field(..., description="Predicted direction: 'long' or 'short'")
    confidence: float = Field(..., description="Ensemble confidence (0-1)")
    prob_up: float = Field(..., description="Probability of upward move")
    prob_down: float = Field(..., description="Probability of downward move")
    should_trade: bool = Field(..., description="Whether confidence >= 70% threshold")

    # Agreement info
    agreement_count: int = Field(..., description="Number of models agreeing (0-3)")
    agreement_score: float = Field(..., description="Agreement score (0-1)")
    all_agree: bool = Field(..., description="Whether all 3 models agree")

    # Market context
    market_regime: str = Field(..., description="Detected market regime")
    market_price: Optional[float] = Field(None, description="Current market price")
    vix_value: Optional[float] = Field(None, description="Current VIX value")

    # Component predictions
    component_directions: Dict[str, int] = Field(
        ..., description="Direction by timeframe"
    )
    component_confidences: Dict[str, float] = Field(
        ..., description="Confidence by timeframe"
    )
    component_weights: Dict[str, float] = Field(
        ..., description="Weights by timeframe"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T14:00:00",
                "symbol": "EURUSD",
                "direction": "long",
                "confidence": 0.72,
                "prob_up": 0.72,
                "prob_down": 0.28,
                "should_trade": True,
                "agreement_count": 3,
                "agreement_score": 1.0,
                "all_agree": True,
                "market_regime": "trending",
                "market_price": 1.08523,
                "vix_value": 15.32,
                "component_directions": {"1H": 1, "4H": 1, "D": 1},
                "component_confidences": {"1H": 0.68, "4H": 0.71, "D": 0.65},
                "component_weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
            }
        }
```

**Explanation**: Required fields use `Field(...)`, optional use `Field(None, ...)`. Grouped by category with comments. Example in Config for Swagger UI.

**Example 2: List Item Schema**

```python
# From: src/api/schemas/prediction.py:71-80
class PredictionHistoryItem(BaseModel):
    """Single prediction in history."""

    id: int
    timestamp: str
    symbol: str
    direction: str
    confidence: float
    market_price: Optional[float]
    trade_executed: bool
```

**Explanation**: Simple item schema for list responses. Minimal fields needed for display. Optional fields marked with `Optional[T]`.

**Example 3: Paginated List Response**

```python
# From: src/api/schemas/prediction.py:83-88
class PredictionHistoryResponse(BaseModel):
    """Response for prediction history endpoint."""

    predictions: List[PredictionHistoryItem]
    count: int
    total: int
```

**Explanation**: Standard pagination pattern: items list, count of returned items, total available. Enables client-side pagination UI.

**Example 4: Status Response with Nested Schemas**

```python
# From: src/api/schemas/prediction.py:91-113
class ModelInfo(BaseModel):
    """Information about a single model."""

    trained: bool
    val_accuracy: Optional[float]


class ModelStatusResponse(BaseModel):
    """Response for model status endpoint."""

    loaded: bool = Field(..., description="Whether models are loaded")
    model_dir: Optional[str] = Field(None, description="Model directory path")
    weights: Optional[Dict[str, float]] = Field(None, description="Ensemble weights")
    agreement_bonus: Optional[float] = Field(None, description="Agreement bonus")
    sentiment_enabled: bool = Field(False, description="Whether sentiment is enabled")
    sentiment_by_timeframe: Dict[str, bool] = Field(
        default_factory=dict, description="Sentiment enabled per timeframe"
    )
    models: Dict[str, ModelInfo] = Field(
        default_factory=dict, description="Individual model status"
    )
    initialized_at: Optional[str] = Field(None, description="Initialization timestamp")
    error: Optional[str] = Field(None, description="Error message if not loaded")
```

**Explanation**: Nested schema `ModelInfo` for complex structure. `default_factory=dict` for mutable defaults. Error field for debugging.

**Example 5: Schema File Organization**

```python
# From: src/api/schemas/prediction.py:1-7
"""Pydantic schemas for prediction endpoints."""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field
```

**Explanation**: Module docstring, standard imports, typing imports, then Pydantic imports. One schema file per route module.

## Quality Checklist

- [ ] All fields have `Field(description=...)`
- [ ] Config includes `json_schema_extra` with realistic example
- [ ] Pattern matches `src/api/schemas/prediction.py:16-68`
- [ ] Optional fields use `Optional[T]` with `Field(None, ...)`
- [ ] List responses have items, count, total
- [ ] Docstring describes schema purpose
- [ ] Schema file in `src/api/schemas/` directory

## Common Mistakes

- **Missing Field descriptions**: Swagger docs incomplete
  - Wrong: `confidence: float`
  - Correct: `confidence: float = Field(..., description="Ensemble confidence (0-1)")`

- **Mutable default in Field**: Causes shared state issues
  - Wrong: `items: Dict[str, int] = Field(default={})`
  - Correct: `items: Dict[str, int] = Field(default_factory=dict)`

- **No example in Config**: Swagger "Try it out" shows empty data
  - Wrong: Omit Config class
  - Correct: Add `json_schema_extra = {"example": {...}}`

## Validation

- [ ] Pattern confirmed in `src/api/schemas/prediction.py:16-68`
- [ ] Used in route with `response_model=` (see `src/api/routes/predictions.py:26`)
- [ ] Example renders in Swagger UI at `/docs`

## Related Skills

- `creating-fastapi-endpoints` - Use schemas as response_model
- `creating-sqlalchemy-models` - Database models that map to schemas
- `writing-pytest-tests` - Create test data matching schemas

---

*Version 1.0.0 | Last verified: 2026-01-16 | Source: src/api/schemas/*
