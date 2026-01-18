"""Pipeline management endpoints."""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks

from ..services.pipeline_service import pipeline_service
from ..scheduler import run_pipeline_now
from ..utils.validation import safe_iloc

router = APIRouter()


@router.get("/pipeline/status")
async def get_pipeline_status() -> Dict[str, Any]:
    """Get current pipeline status.

    Returns:
        Pipeline status including last update times and cache status
    """
    status = pipeline_service.get_status()

    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "pipeline": status,
    }


@router.post("/pipeline/run")
async def trigger_pipeline(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Manually trigger the data pipeline.

    This runs the full pipeline in the background:
    - Fetches new price data from yfinance
    - Updates the historical CSV
    - Recalculates technical indicators
    - Updates sentiment data

    Returns:
        Status message
    """
    # Run pipeline in background
    background_tasks.add_task(pipeline_service.run_full_pipeline)

    return {
        "status": "started",
        "message": "Pipeline started in background",
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/pipeline/run-sync")
async def trigger_pipeline_sync() -> Dict[str, Any]:
    """Manually trigger the data pipeline and wait for completion.

    This runs the full pipeline synchronously.

    Returns:
        Pipeline status after completion
    """
    try:
        success = pipeline_service.run_full_pipeline()

        if success:
            return {
                "status": "success",
                "message": "Pipeline completed successfully",
                "timestamp": datetime.now().isoformat(),
                "pipeline_status": pipeline_service.get_status(),
            }
        else:
            return {
                "status": "partial",
                "message": "Pipeline completed with issues",
                "timestamp": datetime.now().isoformat(),
                "pipeline_status": pipeline_service.get_status(),
            }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline failed: {str(e)}",
        )


@router.get("/pipeline/data/{timeframe}")
async def get_pipeline_data_summary(timeframe: str) -> Dict[str, Any]:
    """Get summary of cached data for a timeframe.

    Args:
        timeframe: One of "1h", "4h", "D" (daily)

    Returns:
        Data summary including row count, date range, and sample columns
    """
    valid_timeframes = {"1h", "4h", "D", "1H", "4H", "d"}
    if timeframe not in valid_timeframes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timeframe. Use one of: 1h, 4h, D",
        )

    df = pipeline_service.get_processed_data(timeframe)

    if df is None or df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No cached data for timeframe {timeframe}. Run the pipeline first.",
        )

    # Build response with safe DataFrame access
    response = {
        "timeframe": timeframe,
        "rows": len(df),
        "columns": len(df.columns),
        "date_range": {
            "start": df.index.min().isoformat() if hasattr(df.index.min(), 'isoformat') else str(df.index.min()),
            "end": df.index.max().isoformat() if hasattr(df.index.max(), 'isoformat') else str(df.index.max()),
        },
        "sample_columns": list(df.columns[:20]),
    }

    # Safely add latest bar if data exists
    if "close" in df.columns and len(df) > 0:
        # Use safe_iloc to prevent IndexError
        last_row = safe_iloc(df, -1)
        if last_row is not None:
            response["latest_bar"] = {
                "timestamp": df.index[-1].isoformat() if hasattr(df.index[-1], 'isoformat') else str(df.index[-1]),
                "open": float(last_row.get("open", 0)),
                "high": float(last_row.get("high", 0)),
                "low": float(last_row.get("low", 0)),
                "close": float(last_row.get("close", 0)),
            }
        else:
            response["latest_bar"] = None
    else:
        response["latest_bar"] = None

    return response
