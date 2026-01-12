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

        # Get current price and VIX
        current_price = data_service.get_current_price()
        vix_value = data_service.get_latest_vix()

        # Make prediction
        prediction = model_service.predict(df)

        return PredictionResponse(
            timestamp=prediction["timestamp"],
            symbol="EURUSD",
            direction=prediction["direction"],
            confidence=prediction["confidence"],
            prob_up=prediction["prob_up"],
            prob_down=prediction["prob_down"],
            should_trade=prediction["should_trade"],
            agreement_count=prediction["agreement_count"],
            agreement_score=prediction["agreement_score"],
            all_agree=prediction["all_agree"],
            market_regime=prediction["market_regime"],
            market_price=current_price,
            vix_value=vix_value,
            component_directions=prediction["component_directions"],
            component_confidences=prediction["component_confidences"],
            component_weights=prediction["component_weights"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


@router.get("/predictions/stats")
async def get_prediction_stats(
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Get prediction accuracy statistics."""
    try:
        total = db.query(Prediction).count()

        if total == 0:
            return {
                "total_predictions": 0,
                "trades_executed": 0,
                "avg_confidence": 0.0,
            }

        # Count trades executed
        trades_executed = (
            db.query(Prediction).filter(Prediction.trade_executed == True).count()
        )

        # Average confidence
        from sqlalchemy import func

        avg_conf = db.query(func.avg(Prediction.confidence)).scalar() or 0.0

        return {
            "total_predictions": total,
            "trades_executed": trades_executed,
            "execution_rate": trades_executed / total if total > 0 else 0.0,
            "avg_confidence": float(avg_conf),
        }

    except Exception as e:
        logger.error(f"Error getting prediction stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


@router.post("/predictions/generate")
async def generate_prediction_now() -> Dict[str, Any]:
    """Manually trigger a prediction (for testing).

    This bypasses the scheduler and generates a prediction immediately.
    """
    if not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is initializing.",
        )

    try:
        from ..scheduler import run_prediction_now

        result = run_prediction_now()
        return result

    except Exception as e:
        logger.error(f"Error generating prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
