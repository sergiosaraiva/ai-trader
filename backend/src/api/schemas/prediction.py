"""Pydantic schemas for prediction endpoints."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .asset import AssetMetadata


class ComponentPrediction(BaseModel):
    """Prediction from a single timeframe model."""

    direction: int = Field(..., description="Direction: 0=down, 1=up")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence (0-1)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "direction": 1,
                "confidence": 0.72,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response for latest prediction endpoint."""

    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")
    symbol: str = Field(default="EURUSD", description="Trading symbol")
    direction: str = Field(..., description="Predicted direction: 'long' or 'short'")
    confidence: float = Field(..., ge=0, le=1, description="Ensemble confidence (0-1)")
    prob_up: float = Field(..., ge=0, le=1, description="Probability of upward move")
    prob_down: float = Field(..., ge=0, le=1, description="Probability of downward move")
    should_trade: bool = Field(..., description="Whether confidence >= 70% threshold")

    # Agreement info
    agreement_count: int = Field(..., ge=0, le=3, description="Number of models agreeing (0-3)")
    agreement_score: float = Field(..., ge=0, le=1, description="Agreement score (0-1)")
    all_agree: bool = Field(..., description="Whether all 3 models agree")

    # Market context
    market_regime: str = Field(..., description="Detected market regime")
    market_price: Optional[float] = Field(None, gt=0, description="Current market price")
    vix_value: Optional[float] = Field(None, ge=0, description="Current VIX value")

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

    # Asset metadata (optional for backward compatibility)
    asset_metadata: Optional[AssetMetadata] = Field(
        None, description="Asset-specific metadata (precision, units, formatting)"
    )

    model_config = {
        "json_schema_extra": {
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
    }


class PredictionHistoryItem(BaseModel):
    """Single prediction in history."""

    id: int = Field(..., description="Unique prediction ID")
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")
    symbol: str = Field(..., description="Trading symbol")
    direction: str = Field(..., description="Predicted direction: 'long' or 'short'")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence (0-1)")
    market_price: Optional[float] = Field(None, gt=0, description="Market price at prediction")
    trade_executed: bool = Field(..., description="Whether a trade was executed")
    should_trade: bool = Field(..., description="Whether confidence >= 70% threshold")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": 42,
                "timestamp": "2024-01-15T14:00:00",
                "symbol": "EURUSD",
                "direction": "long",
                "confidence": 0.72,
                "market_price": 1.08523,
                "trade_executed": True,
                "should_trade": True,
            }
        }
    }


class PredictionHistoryResponse(BaseModel):
    """Response for prediction history endpoint."""

    predictions: List[PredictionHistoryItem] = Field(
        ..., description="List of historical predictions"
    )
    count: int = Field(..., ge=0, description="Number of predictions in response")
    total: int = Field(..., ge=0, description="Total predictions available")

    model_config = {
        "json_schema_extra": {
            "example": {
                "predictions": [
                    {
                        "id": 42,
                        "timestamp": "2024-01-15T14:00:00",
                        "symbol": "EURUSD",
                        "direction": "long",
                        "confidence": 0.72,
                        "market_price": 1.08523,
                        "trade_executed": True,
                        "should_trade": True,
                    }
                ],
                "count": 1,
                "total": 100,
            }
        }
    }


class ModelInfo(BaseModel):
    """Information about a single model."""

    trained: bool = Field(..., description="Whether model is trained")
    val_accuracy: Optional[float] = Field(
        None, ge=0, le=1, description="Validation accuracy (0-1)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "trained": True,
                "val_accuracy": 0.67,
            }
        }
    }


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

    model_config = {
        "json_schema_extra": {
            "example": {
                "loaded": True,
                "model_dir": "models/mtf_ensemble",
                "weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
                "agreement_bonus": 0.05,
                "sentiment_enabled": True,
                "sentiment_by_timeframe": {"1H": False, "4H": False, "D": True},
                "models": {
                    "1H": {"trained": True, "val_accuracy": 0.67},
                    "4H": {"trained": True, "val_accuracy": 0.65},
                    "D": {"trained": True, "val_accuracy": 0.62},
                },
                "initialized_at": "2024-01-15T10:00:00",
                "error": None,
            }
        }
    }
