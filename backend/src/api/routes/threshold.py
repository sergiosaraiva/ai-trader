"""API routes for dynamic confidence threshold monitoring.

Provides endpoints for:
- Threshold status and current value
- Calculation history
- On-demand threshold calculation
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session

from ..database.session import get_db
from ..schemas.threshold import (
    ThresholdStatusResponse,
    ThresholdHistoryResponse,
    ThresholdHistoryItem,
    ThresholdCalculateResponse,
)
from ..services.threshold_service import threshold_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/threshold", tags=["threshold"])


@router.get("/status", response_model=ThresholdStatusResponse)
async def get_threshold_status():
    """Get current threshold service status and metrics.

    Returns:
        Threshold service status including current threshold and configuration
    """
    if not threshold_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Threshold service not initialized"
        )

    try:
        status = threshold_service.get_status()
        return status

    except Exception as e:
        logger.error(f"Failed to get threshold status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get threshold status: {str(e)}"
        )


@router.get("/history", response_model=ThresholdHistoryResponse)
async def get_threshold_history(
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum records to return"),
    db: Session = Depends(get_db)
):
    """Get threshold calculation history.

    Args:
        limit: Maximum number of records to return (1-1000)
        db: Database session

    Returns:
        List of recent threshold calculations with components and metrics
    """
    if not threshold_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Threshold service not initialized"
        )

    try:
        history = threshold_service.get_recent_history(limit=limit, db=db)

        # Convert to response schema
        items = [ThresholdHistoryItem(**item) for item in history]

        return ThresholdHistoryResponse(
            history=items,
            count=len(items)
        )

    except Exception as e:
        logger.error(f"Failed to get threshold history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get threshold history: {str(e)}"
        )


@router.post("/calculate", response_model=ThresholdCalculateResponse)
async def calculate_threshold(
    record_history: bool = Query(default=True, description="Whether to record to database"),
    db: Session = Depends(get_db)
):
    """Calculate threshold on-demand.

    Performs a fresh threshold calculation and optionally records to history.
    Useful for testing and monitoring purposes.

    Args:
        record_history: Whether to persist calculation to database
        db: Database session

    Returns:
        Calculated threshold with detailed component breakdown
    """
    if not threshold_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Threshold service not initialized"
        )

    try:
        from datetime import datetime

        # Calculate threshold
        threshold = threshold_service.calculate_threshold(
            db=db if record_history else None,
            record_history=record_history
        )

        # Get detailed status for components
        status = threshold_service.get_status()

        # Get most recent history record for components
        if record_history:
            history = threshold_service.get_recent_history(limit=1, db=db)
            if history:
                latest = history[0]
                components = {
                    "short_term": latest.get("short_term"),
                    "medium_term": latest.get("medium_term"),
                    "long_term": latest.get("long_term"),
                    "blended": latest.get("blended"),
                    "adjustment": latest.get("adjustment"),
                }
                data_quality = {
                    "predictions_7d": latest.get("predictions_7d", 0),
                    "predictions_14d": latest.get("predictions_14d", 0),
                    "predictions_30d": latest.get("predictions_30d", 0),
                    "trade_count": latest.get("trade_count", 0),
                    "win_rate": latest.get("win_rate"),
                }
            else:
                components = {}
                data_quality = {}
        else:
            # No history recorded, use status data
            components = {}
            data_quality = {
                "predictions_7d": status["predictions_7d"],
                "predictions_14d": status["predictions_14d"],
                "predictions_30d": status["predictions_30d"],
                "trade_count": status["recent_trades"],
                "win_rate": None,
            }

        return ThresholdCalculateResponse(
            threshold=threshold,
            timestamp=datetime.utcnow().isoformat(),
            components=components,
            data_quality=data_quality
        )

    except Exception as e:
        logger.error(f"Failed to calculate threshold: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to calculate threshold: {str(e)}"
        )


@router.get("/current", response_model=float)
async def get_current_threshold():
    """Get the most recently calculated threshold value.

    Returns cached threshold without recalculating. Useful for quick checks.

    Returns:
        Current threshold value (float)
    """
    if not threshold_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Threshold service not initialized"
        )

    try:
        threshold = threshold_service.get_current_threshold()

        if threshold is None:
            # No cached threshold, calculate fresh
            threshold = threshold_service.calculate_threshold(record_history=False)

        return threshold

    except Exception as e:
        logger.error(f"Failed to get current threshold: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get current threshold: {str(e)}"
        )
