"""Pydantic schemas for market data endpoints."""

from typing import List, Optional

from pydantic import BaseModel, Field


class MarketInfoResponse(BaseModel):
    """Response for current market info endpoint."""

    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    price: Optional[float] = Field(None, gt=0, description="Current price")
    change: Optional[float] = Field(None, description="Change from previous close")
    change_pct: Optional[float] = Field(None, description="Change percentage")
    day_high: Optional[float] = Field(None, gt=0, description="Day high price")
    day_low: Optional[float] = Field(None, gt=0, description="Day low price")
    prev_close: Optional[float] = Field(None, gt=0, description="Previous close price")
    timestamp: str = Field(..., description="Data timestamp (ISO format)")
    data_source: str = Field(default="yfinance", description="Data source provider")
    delay_minutes: int = Field(default=15, ge=0, description="Approximate data delay in minutes")
    error: Optional[str] = Field(None, description="Error message if data unavailable")

    model_config = {
        "json_schema_extra": {
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
                "error": None,
            }
        }
    }


class CandleResponse(BaseModel):
    """Single OHLCV candle."""

    timestamp: str = Field(..., description="Candle timestamp (ISO format)")
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="Highest price")
    low: float = Field(..., gt=0, description="Lowest price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: Optional[float] = Field(None, ge=0, description="Trading volume")

    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": "2024-01-15T14:00:00",
                "open": 1.08500,
                "high": 1.08650,
                "low": 1.08450,
                "close": 1.08600,
                "volume": 15230.5,
            }
        }
    }


class CandlesResponse(BaseModel):
    """Response for candles endpoint."""

    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    timeframe: str = Field(..., description="Candle timeframe (e.g., '1H', '4H', '1D')")
    candles: List[CandleResponse] = Field(..., description="List of OHLCV candles")
    count: int = Field(..., ge=0, description="Number of candles returned")

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "EURUSD",
                "timeframe": "1H",
                "candles": [
                    {
                        "timestamp": "2024-01-15T13:00:00",
                        "open": 1.08450,
                        "high": 1.08550,
                        "low": 1.08400,
                        "close": 1.08500,
                        "volume": 12500.0,
                    },
                    {
                        "timestamp": "2024-01-15T14:00:00",
                        "open": 1.08500,
                        "high": 1.08650,
                        "low": 1.08450,
                        "close": 1.08600,
                        "volume": 15230.5,
                    },
                ],
                "count": 2,
            }
        }
    }


class VIXResponse(BaseModel):
    """Response for VIX data endpoint."""

    value: Optional[float] = Field(None, ge=0, description="Current VIX value")
    timestamp: str = Field(..., description="Data timestamp (ISO format)")
    data_source: str = Field(default="yfinance", description="Data source provider")
    error: Optional[str] = Field(None, description="Error message if data unavailable")

    model_config = {
        "json_schema_extra": {
            "example": {
                "value": 15.32,
                "timestamp": "2024-01-15T14:30:00",
                "data_source": "yfinance",
                "error": None,
            }
        }
    }
