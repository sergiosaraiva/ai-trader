"""API routes for AI-Trader."""

from . import health, predictions, trading, market, pipeline, cron

__all__ = ["health", "predictions", "trading", "market", "pipeline", "cron"]
