"""Health check endpoints."""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter

from ..services.model_service import model_service
from ..services.data_service import data_service
from ..services.trading_service import trading_service

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
    }


@router.get("/health/detailed")
async def detailed_health() -> Dict[str, Any]:
    """Detailed health check with component status."""
    # Check model status
    model_status = "up" if model_service.is_loaded else "not_loaded"

    # Check data service
    try:
        price = data_service.get_current_price()
        data_status = "up" if price is not None else "degraded"
    except Exception:
        data_status = "down"

    # Check trading service
    try:
        _ = trading_service.get_status()
        trading_status = "up"
    except Exception:
        trading_status = "down"

    # Database status
    try:
        from sqlalchemy import text
        from ..database.session import get_session
        db = get_session()
        db.execute(text("SELECT 1"))
        db.close()
        db_status = "up"
    except Exception:
        db_status = "down"

    # Overall status
    all_up = all(
        s == "up"
        for s in [model_status, data_status, trading_status, db_status]
    )

    return {
        "status": "healthy" if all_up else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "api": {"status": "up"},
            "models": {"status": model_status},
            "database": {"status": db_status},
            "data_feed": {"status": data_status},
            "trading": {"status": trading_status},
        },
    }


@router.get("/health/ready")
async def readiness_check() -> Dict[str, Any]:
    """Readiness check - returns 200 only when all services are ready."""
    is_ready = (
        model_service.is_loaded
        and data_service._initialized
        and trading_service._initialized
    )

    return {
        "ready": is_ready,
        "timestamp": datetime.now().isoformat(),
        "services": {
            "model": model_service.is_loaded,
            "data": data_service._initialized,
            "trading": trading_service._initialized,
        },
    }
