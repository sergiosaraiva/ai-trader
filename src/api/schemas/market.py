"""Pydantic schemas for market data endpoints."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class MarketInfoResponse(BaseModel):
    """Response for current market info endpoint."""

    symbol: str
    price: Optional[float] = Field(None, description="Current price")
    change: Optional[float] = Field(None, description="Change from prev close")
    change_pct: Optional[float] = Field(None, description="Change percentage")
    day_high: Optional[float] = Field(None, description="Day high")
    day_low: Optional[float] = Field(None, description="Day low")
    prev_close: Optional[float] = Field(None, description="Previous close")
    timestamp: str = Field(..., description="Data timestamp")
    data_source: str = Field(default="yfinance", description="Data source")
    delay_minutes: int = Field(default=15, description="Approximate data delay")
    error: Optional[str] = Field(None, description="Error message if any")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "EURUSD",
                "price": 1.08523,
                "change": 0.00123,
                "change_pct": 0.113,
                "day_high": 1.08650,
                "day_low": 1.08320,
                "prev_close": 1.08400,
                "timestamp": "2024-01-15T14:30:00",
                "data_source": "yfinance",
                "delay_minutes": 15,
            }
        }


class CandleResponse(BaseModel):
    """Single OHLCV candle."""

    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None


class CandlesResponse(BaseModel):
    """Response for candles endpoint."""

    symbol: str
    timeframe: str
    candles: List[CandleResponse]
    count: int
