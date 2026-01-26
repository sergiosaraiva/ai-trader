"""API routes for AI-Trader."""

from . import health, predictions, trading, market, pipeline, cron, performance, agent

__all__ = ["health", "predictions", "trading", "market", "pipeline", "cron", "performance", "agent"]
