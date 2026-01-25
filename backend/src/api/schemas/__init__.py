"""Pydantic schemas for API request/response models."""

from .asset import AssetMetadata
from .prediction import (
    PredictionResponse,
    PredictionHistoryResponse,
    ModelStatusResponse,
)
from .trading import (
    TradingStatusResponse,
    TradeResponse,
    TradeHistoryResponse,
    PerformanceResponse,
    EquityCurveResponse,
)
from .market import (
    MarketInfoResponse,
    CandleResponse,
    CandlesResponse,
)
from .agent import (
    AgentStartRequest,
    AgentStopRequest,
    AgentConfigUpdateRequest,
    KillSwitchRequest,
    CommandResponse,
    AgentStatusResponse,
    AgentMetricsResponse,
    CommandStatusResponse,
    CommandListResponse,
)

__all__ = [
    "AssetMetadata",
    "PredictionResponse",
    "PredictionHistoryResponse",
    "ModelStatusResponse",
    "TradingStatusResponse",
    "TradeResponse",
    "TradeHistoryResponse",
    "PerformanceResponse",
    "EquityCurveResponse",
    "MarketInfoResponse",
    "CandleResponse",
    "CandlesResponse",
    "AgentStartRequest",
    "AgentStopRequest",
    "AgentConfigUpdateRequest",
    "KillSwitchRequest",
    "CommandResponse",
    "AgentStatusResponse",
    "AgentMetricsResponse",
    "CommandStatusResponse",
    "CommandListResponse",
]
