"""Pydantic schemas for threshold endpoints."""

from typing import List, Optional

from pydantic import BaseModel, Field


class ThresholdStatusResponse(BaseModel):
    """Response for threshold status endpoint."""

    initialized: bool = Field(..., description="Whether threshold service is initialized")
    use_dynamic: bool = Field(..., description="Whether dynamic threshold is enabled")
    current_threshold: Optional[float] = Field(
        None, ge=0, le=1, description="Current calculated threshold (0-1)"
    )
    last_calculation: Optional[str] = Field(
        None, description="Last calculation timestamp (ISO format)"
    )
    calculation_count: int = Field(..., ge=0, description="Total calculations performed")
    predictions_7d: int = Field(..., ge=0, description="Predictions in 7-day window")
    predictions_14d: int = Field(..., ge=0, description="Predictions in 14-day window")
    predictions_30d: int = Field(..., ge=0, description="Predictions in 30-day window")
    recent_trades: int = Field(..., ge=0, description="Recent trades tracked")
    static_fallback: float = Field(..., ge=0, le=1, description="Static fallback threshold")
    config: dict = Field(..., description="Configuration summary")

    model_config = {
        "json_schema_extra": {
            "example": {
                "initialized": True,
                "use_dynamic": True,
                "current_threshold": 0.6523,
                "last_calculation": "2024-01-15T14:30:00",
                "calculation_count": 42,
                "predictions_7d": 168,
                "predictions_14d": 336,
                "predictions_30d": 720,
                "recent_trades": 25,
                "static_fallback": 0.66,
                "config": {
                    "windows": "7d/14d/30d",
                    "weights": "25%/60%/15%",
                    "quantile": 0.6,
                    "bounds": "0.55-0.75",
                    "target_win_rate": 0.54,
                },
            }
        }
    }


class ThresholdHistoryItem(BaseModel):
    """Single threshold calculation in history."""

    timestamp: str = Field(..., description="Calculation timestamp (ISO format)")
    threshold: float = Field(..., ge=0, le=1, description="Calculated threshold value")
    short_term: Optional[float] = Field(None, description="Short-term component (7d)")
    medium_term: Optional[float] = Field(None, description="Medium-term component (14d)")
    long_term: Optional[float] = Field(None, description="Long-term component (30d)")
    blended: Optional[float] = Field(None, description="Blended value before adjustment")
    adjustment: Optional[float] = Field(None, description="Performance adjustment applied")
    predictions_7d: int = Field(..., ge=0, description="Predictions used (7d)")
    predictions_14d: int = Field(..., ge=0, description="Predictions used (14d)")
    predictions_30d: int = Field(..., ge=0, description="Predictions used (30d)")
    win_rate: Optional[float] = Field(None, ge=0, le=1, description="Recent win rate")
    trade_count: Optional[int] = Field(None, ge=0, description="Recent trade count")
    reason: Optional[str] = Field(None, description="Calculation reason")

    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": "2024-01-15T14:30:00",
                "threshold": 0.6523,
                "short_term": 0.6789,
                "medium_term": 0.6512,
                "long_term": 0.6234,
                "blended": 0.6543,
                "adjustment": -0.0020,
                "predictions_7d": 168,
                "predictions_14d": 336,
                "predictions_30d": 720,
                "win_rate": 0.52,
                "trade_count": 25,
                "reason": "dynamic",
            }
        }
    }


class ThresholdHistoryResponse(BaseModel):
    """Response for threshold history endpoint."""

    history: List[ThresholdHistoryItem] = Field(
        ..., description="List of threshold calculations"
    )
    count: int = Field(..., ge=0, description="Number of items in response")

    model_config = {
        "json_schema_extra": {
            "example": {
                "history": [
                    {
                        "timestamp": "2024-01-15T14:30:00",
                        "threshold": 0.6523,
                        "short_term": 0.6789,
                        "medium_term": 0.6512,
                        "long_term": 0.6234,
                        "blended": 0.6543,
                        "adjustment": -0.0020,
                        "predictions_7d": 168,
                        "predictions_14d": 336,
                        "predictions_30d": 720,
                        "win_rate": 0.52,
                        "trade_count": 25,
                        "reason": "dynamic",
                    }
                ],
                "count": 1,
            }
        }
    }


class ThresholdCalculateResponse(BaseModel):
    """Response for on-demand threshold calculation."""

    threshold: float = Field(..., ge=0, le=1, description="Calculated threshold value")
    timestamp: str = Field(..., description="Calculation timestamp (ISO format)")
    components: dict = Field(..., description="Threshold components breakdown")
    data_quality: dict = Field(..., description="Data quality metrics")

    model_config = {
        "json_schema_extra": {
            "example": {
                "threshold": 0.6523,
                "timestamp": "2024-01-15T14:30:00",
                "components": {
                    "short_term": 0.6789,
                    "medium_term": 0.6512,
                    "long_term": 0.6234,
                    "blended": 0.6543,
                    "adjustment": -0.0020,
                },
                "data_quality": {
                    "predictions_7d": 168,
                    "predictions_14d": 336,
                    "predictions_30d": 720,
                    "trade_count": 25,
                    "win_rate": 0.52,
                },
            }
        }
    }
