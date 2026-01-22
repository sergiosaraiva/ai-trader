"""Performance endpoints."""

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException

from ..services.performance_service import performance_service
from ..utils.logging import log_exception

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/model/performance")
async def get_model_performance() -> Dict[str, Any]:
    """Get model performance metrics and highlights.

    Returns comprehensive performance data including:
    - Overall metrics (total pips, win rate, profit factor, trades)
    - High-confidence performance (70% threshold)
    - Model consensus accuracy (full agreement)
    - Walk-forward validation results
    - Regime performance
    - Dynamic highlights
    - Summary headline and description

    The data is dynamically generated based on training metadata
    and backtest results, with intelligent fallbacks.
    """
    try:
        # Initialize service if not already done
        if not performance_service.is_loaded:
            logger.info("Performance service not loaded, initializing...")
            if not performance_service.initialize():
                logger.warning("Failed to initialize performance service, using defaults")

        # Get performance data
        data = performance_service.get_performance_data()

        return data

    except Exception as e:
        log_exception(logger, "Error getting model performance", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model/performance/reload")
async def reload_performance_data() -> Dict[str, Any]:
    """Reload performance data from disk.

    Useful after retraining models or updating backtest results.
    """
    try:
        success = performance_service.reload()

        if success:
            return {
                "status": "success",
                "message": "Performance data reloaded successfully",
            }
        else:
            return {
                "status": "warning",
                "message": "Performance data reload encountered issues, using defaults",
            }

    except Exception as e:
        log_exception(logger, "Error reloading performance data", e)
        raise HTTPException(status_code=500, detail=str(e))
