"""Health check endpoints."""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.1.0",
    }


@router.get("/health/detailed")
async def detailed_health() -> Dict[str, Any]:
    """Detailed health check with component status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": {"status": "up"},
            "models": {"status": "not_loaded"},  # Update when models loaded
            "database": {"status": "not_connected"},  # Update when connected
            "data_feed": {"status": "not_connected"},  # Update when connected
        },
    }
