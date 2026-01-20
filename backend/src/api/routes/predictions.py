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
from ..services.asset_service import asset_service
from ..services.explanation_service import explanation_service
from ..schemas.prediction import (
    PredictionResponse,
    PredictionHistoryResponse,
    PredictionHistoryItem,
    ModelStatusResponse,
)
from ..utils.logging import log_exception

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/predictions/latest", response_model=PredictionResponse)
async def get_latest_prediction(
    symbol: str = Query(
        default="EURUSD",
        pattern="^[A-Za-z0-9\\-]{1,20}$",
        max_length=20,
        description="Trading symbol (alphanumeric with dash, max 20 chars)"
    )
) -> PredictionResponse:
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
        # Make prediction using pre-computed pipeline data (fast path)
        # This uses cached features from pipeline_service instead of recalculating
        prediction = model_service.predict_from_pipeline(symbol=symbol)

        # Get current price and VIX for response
        current_price = data_service.get_current_price(symbol)
        vix_value = data_service.get_latest_vix()

        # Get asset metadata
        asset_metadata = asset_service.get_asset_metadata(symbol)

        return PredictionResponse(
            timestamp=prediction["timestamp"],
            symbol=prediction["symbol"],
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
            asset_metadata=asset_metadata,
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

        items = []
        for p in predictions:
            try:
                # Defensive field extraction with None checks
                timestamp = p.timestamp.isoformat() if p.timestamp else ""
                symbol = p.symbol if p.symbol else "EURUSD"
                direction = p.direction if p.direction else "hold"
                confidence = max(0.0, min(1.0, p.confidence)) if p.confidence is not None else 0.0
                market_price = p.market_price if p.market_price is not None else 0.0
                trade_executed = p.trade_executed if p.trade_executed is not None else False
                # Use should_trade if available, otherwise fallback to confidence >= 0.70
                should_trade = p.should_trade if p.should_trade is not None else (p.confidence >= 0.70)

                items.append(PredictionHistoryItem(
                    id=p.id,
                    timestamp=timestamp,
                    symbol=symbol,
                    direction=direction,
                    confidence=confidence,
                    market_price=market_price,
                    trade_executed=trade_executed,
                    should_trade=should_trade,
                ))
            except Exception as e:
                # Log but don't fail entire request for single bad record
                log_exception(
                    logger,
                    f"Failed to serialize prediction {p.id}",
                    e,
                    prediction_id=p.id
                )
                continue

        return PredictionHistoryResponse(
            predictions=items,
            count=len(items),
            total=total,
        )

    except Exception as e:
        log_exception(logger, "Error getting prediction history", e)
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
    """Get status of loaded models.

    Returns information about the MTF Ensemble models including
    training status, accuracy, and configuration.
    """
    if not model_service.is_loaded:
        logger.warning("Model status requested but model not loaded")

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


@router.get("/predictions/explanation")
async def get_prediction_explanation(
    symbol: str = Query(
        default="EURUSD",
        pattern="^[A-Za-z0-9\\-]{1,20}$",
        max_length=20,
        description="Trading symbol"
    ),
    force_refresh: bool = Query(
        default=False,
        description="Force regenerate explanation even if cached"
    ),
) -> Dict[str, Any]:
    """Get a plain English explanation of the current recommendation.

    Uses GPT-4o-mini to generate an explanation of why the AI
    is recommending BUY, SELL, or HOLD based on the current
    technical and sentiment analysis.

    The explanation is cached and only regenerates when:
    - The direction changes (long/short)
    - Should_trade changes (above/below 70% threshold)
    - Confidence changes by more than 5%
    - VIX changes by more than 2 points
    - Timeframe agreement changes
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
                detail="Insufficient market data",
            )

        # Get current values
        current_price = data_service.get_current_price(symbol)
        vix_value = data_service.get_latest_vix()

        # Make prediction
        prediction = model_service.predict(df, symbol=symbol)

        # Get asset metadata
        asset_metadata = asset_service.get_asset_metadata(symbol)
        asset_type = asset_metadata.asset_type if asset_metadata else "forex"
        formatted_symbol = asset_metadata.formatted_symbol if asset_metadata else symbol

        # Generate explanation
        result = explanation_service.generate_explanation(
            prediction=prediction,
            vix=vix_value,
            current_price=current_price,
            symbol=formatted_symbol,
            asset_type=asset_type,
            force_refresh=force_refresh,
        )

        return {
            "explanation": result.get("explanation"),
            "generated_at": result.get("generated_at"),
            "cached": result.get("cached", False),
            "error": result.get("error"),
            "prediction_summary": {
                "direction": prediction.get("direction"),
                "confidence": float(prediction.get("confidence", 0)),
                "should_trade": bool(prediction.get("should_trade", False)),
                "agreement_count": int(prediction.get("agreement_count", 0)),
            },
            "vix": float(vix_value) if vix_value is not None else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
