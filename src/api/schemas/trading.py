"""Pydantic schemas for trading endpoints."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class TradeResponse(BaseModel):
    """Response for a single trade."""

    id: int
    symbol: str
    direction: str = Field(..., description="Trade direction: 'long' or 'short'")
    entry_price: float
    entry_time: str
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = Field(
        None, description="Exit reason: 'tp', 'sl', 'timeout', 'manual'"
    )
    lot_size: float
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    pips: Optional[float] = Field(None, description="Profit/loss in pips")
    pnl_usd: Optional[float] = Field(None, description="Profit/loss in USD")
    is_winner: Optional[bool] = None
    confidence: Optional[float] = None
    status: str = Field(..., description="Trade status: 'open' or 'closed'")

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "symbol": "EURUSD",
                "direction": "long",
                "entry_price": 1.08500,
                "entry_time": "2024-01-15T14:01:00",
                "exit_price": 1.08750,
                "exit_time": "2024-01-15T16:30:00",
                "exit_reason": "tp",
                "lot_size": 0.1,
                "take_profit": 1.08750,
                "stop_loss": 1.08350,
                "pips": 25.0,
                "pnl_usd": 250.0,
                "is_winner": True,
                "confidence": 0.72,
                "status": "closed",
            }
        }


class OpenPositionResponse(BaseModel):
    """Response for open position."""

    id: int
    symbol: str
    direction: str
    entry_price: float
    entry_time: str
    lot_size: float
    take_profit: float
    stop_loss: float
    confidence: Optional[float]
    current_price: Optional[float] = None
    unrealized_pips: Optional[float] = None
    unrealized_pnl: Optional[float] = None


class TradingStatusResponse(BaseModel):
    """Response for trading status endpoint."""

    mode: str = Field(default="paper", description="Trading mode")
    balance: float = Field(..., description="Account balance in USD")
    equity: float = Field(..., description="Account equity in USD")
    unrealized_pnl: float = Field(default=0.0, description="Unrealized P&L")
    has_position: bool = Field(..., description="Whether a position is open")
    open_position: Optional[OpenPositionResponse] = Field(
        None, description="Current open position"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "mode": "paper",
                "balance": 103450.00,
                "equity": 103575.00,
                "unrealized_pnl": 125.00,
                "has_position": True,
                "open_position": {
                    "id": 5,
                    "symbol": "EURUSD",
                    "direction": "long",
                    "entry_price": 1.08500,
                    "entry_time": "2024-01-15T14:01:00",
                    "lot_size": 0.1,
                    "take_profit": 1.08750,
                    "stop_loss": 1.08350,
                    "confidence": 0.72,
                    "current_price": 1.08625,
                    "unrealized_pips": 12.5,
                    "unrealized_pnl": 125.00,
                },
            }
        }


class TradeHistoryResponse(BaseModel):
    """Response for trade history endpoint."""

    trades: List[TradeResponse]
    count: int


class PerformanceResponse(BaseModel):
    """Response for performance endpoint."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float = Field(..., description="Win rate (0-1)")
    total_pips: float
    total_pnl_usd: float
    avg_pips_per_trade: float
    profit_factor: float
    initial_balance: float
    current_balance: float
    return_pct: float = Field(..., description="Return percentage")

    class Config:
        json_schema_extra = {
            "example": {
                "total_trades": 47,
                "winning_trades": 29,
                "losing_trades": 18,
                "win_rate": 0.617,
                "total_pips": 892.5,
                "total_pnl_usd": 8925.00,
                "avg_pips_per_trade": 19.0,
                "profit_factor": 2.45,
                "initial_balance": 100000.00,
                "current_balance": 108925.00,
                "return_pct": 8.925,
            }
        }


class EquityPoint(BaseModel):
    """Single point on equity curve."""

    timestamp: str
    balance: float
    equity: float


class EquityCurveResponse(BaseModel):
    """Response for equity curve endpoint."""

    data: List[EquityPoint]
    count: int
