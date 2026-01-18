"""Pydantic schemas for trading endpoints."""

from typing import List, Optional

from pydantic import BaseModel, Field

from .asset import AssetMetadata


class TradeResponse(BaseModel):
    """Response for a single trade."""

    id: int = Field(..., description="Unique trade ID")
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    direction: str = Field(..., description="Trade direction: 'long' or 'short'")
    entry_price: float = Field(..., gt=0, description="Entry price")
    entry_time: str = Field(..., description="Entry timestamp (ISO format)")
    exit_price: Optional[float] = Field(None, gt=0, description="Exit price")
    exit_time: Optional[str] = Field(None, description="Exit timestamp (ISO format)")
    exit_reason: Optional[str] = Field(
        None, description="Exit reason: 'tp', 'sl', 'timeout', 'manual'"
    )
    lot_size: float = Field(..., gt=0, description="Position size in lots")
    take_profit: Optional[float] = Field(None, gt=0, description="Take profit price")
    stop_loss: Optional[float] = Field(None, gt=0, description="Stop loss price")
    pips: Optional[float] = Field(None, description="Profit/loss in pips (DEPRECATED - use profit_points)")
    profit_points: Optional[float] = Field(None, description="Profit/loss in asset-specific units (pips/points/ticks/dollars)")
    pnl_usd: Optional[float] = Field(None, description="Profit/loss in USD")
    is_winner: Optional[bool] = Field(None, description="Whether trade was profitable")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Model confidence")
    status: str = Field(..., description="Trade status: 'open' or 'closed'")
    asset_metadata: Optional[AssetMetadata] = Field(
        None, description="Asset-specific metadata (precision, units, formatting)"
    )

    model_config = {
        "json_schema_extra": {
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
    }


class OpenPositionResponse(BaseModel):
    """Response for open position."""

    id: int = Field(..., description="Unique position ID")
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    direction: str = Field(..., description="Position direction: 'long' or 'short'")
    entry_price: float = Field(..., gt=0, description="Entry price")
    entry_time: str = Field(..., description="Entry timestamp (ISO format)")
    lot_size: float = Field(..., gt=0, description="Position size in lots")
    take_profit: float = Field(..., gt=0, description="Take profit price")
    stop_loss: float = Field(..., gt=0, description="Stop loss price")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Model confidence")
    current_price: Optional[float] = Field(None, gt=0, description="Current market price")
    unrealized_pips: Optional[float] = Field(None, description="Unrealized P&L in pips (DEPRECATED - use unrealized_profit_points)")
    unrealized_profit_points: Optional[float] = Field(None, description="Unrealized P&L in asset-specific units")
    unrealized_pnl: Optional[float] = Field(None, description="Unrealized P&L in USD")
    asset_metadata: Optional[AssetMetadata] = Field(
        None, description="Asset-specific metadata (precision, units, formatting)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
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
            }
        }
    }


class TradingStatusResponse(BaseModel):
    """Response for trading status endpoint."""

    mode: str = Field(default="paper", description="Trading mode: 'paper' or 'live'")
    balance: float = Field(..., ge=0, description="Account balance in USD")
    equity: float = Field(..., ge=0, description="Account equity in USD")
    unrealized_pnl: float = Field(default=0.0, description="Unrealized P&L in USD")
    has_position: bool = Field(..., description="Whether a position is open")
    open_position: Optional[OpenPositionResponse] = Field(
        None, description="Current open position details"
    )

    model_config = {
        "json_schema_extra": {
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
    }


class TradeHistoryResponse(BaseModel):
    """Response for trade history endpoint."""

    trades: List[TradeResponse] = Field(..., description="List of trades")
    count: int = Field(..., ge=0, description="Number of trades returned")

    model_config = {
        "json_schema_extra": {
            "example": {
                "trades": [
                    {
                        "id": 1,
                        "symbol": "EURUSD",
                        "direction": "long",
                        "entry_price": 1.08500,
                        "entry_time": "2024-01-15T14:01:00",
                        "exit_price": 1.08750,
                        "exit_time": "2024-01-15T16:30:00",
                        "exit_reason": "tp",
                        "lot_size": 0.1,
                        "pips": 25.0,
                        "pnl_usd": 250.0,
                        "is_winner": True,
                        "status": "closed",
                    }
                ],
                "count": 1,
            }
        }
    }


class PerformanceResponse(BaseModel):
    """Response for performance endpoint."""

    total_trades: int = Field(..., ge=0, description="Total number of closed trades")
    winning_trades: int = Field(..., ge=0, description="Number of winning trades")
    losing_trades: int = Field(..., ge=0, description="Number of losing trades")
    win_rate: float = Field(..., ge=0, le=1, description="Win rate (0-1)")
    total_pips: float = Field(..., description="Total profit/loss in pips (DEPRECATED - use total_profit_points)")
    total_profit_points: Optional[float] = Field(None, description="Total profit/loss in asset-specific units")
    total_pnl_usd: float = Field(..., description="Total profit/loss in USD")
    avg_pips_per_trade: float = Field(..., description="Average pips per trade (DEPRECATED - use avg_profit_points_per_trade)")
    avg_profit_points_per_trade: Optional[float] = Field(None, description="Average profit/loss per trade in asset-specific units")
    profit_factor: float = Field(..., ge=0, description="Profit factor (gross profit / gross loss)")
    initial_balance: float = Field(..., ge=0, description="Starting balance in USD")
    current_balance: float = Field(..., ge=0, description="Current balance in USD")
    return_pct: float = Field(..., description="Return percentage")
    asset_metadata: Optional[AssetMetadata] = Field(
        None, description="Asset-specific metadata (precision, units, formatting)"
    )

    model_config = {
        "json_schema_extra": {
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
    }


class EquityPoint(BaseModel):
    """Single point on equity curve."""

    timestamp: str = Field(..., description="Timestamp (ISO format)")
    balance: float = Field(..., ge=0, description="Account balance at this point")
    equity: float = Field(..., ge=0, description="Account equity at this point")

    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": "2024-01-15T14:00:00",
                "balance": 103450.00,
                "equity": 103575.00,
            }
        }
    }


class EquityCurveResponse(BaseModel):
    """Response for equity curve endpoint."""

    data: List[EquityPoint] = Field(..., description="Equity curve data points")
    count: int = Field(..., ge=0, description="Number of data points")

    model_config = {
        "json_schema_extra": {
            "example": {
                "data": [
                    {"timestamp": "2024-01-01T00:00:00", "balance": 100000.0, "equity": 100000.0},
                    {"timestamp": "2024-01-02T00:00:00", "balance": 100250.0, "equity": 100250.0},
                    {"timestamp": "2024-01-03T00:00:00", "balance": 103450.0, "equity": 103575.0},
                ],
                "count": 3,
            }
        }
    }
