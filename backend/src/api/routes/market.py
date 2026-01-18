"""Market data endpoints."""

import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query

from ..services.data_service import data_service
from ..schemas.market import (
    MarketInfoResponse,
    CandlesResponse,
    CandleResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/market/current", response_model=MarketInfoResponse)
async def get_current_market(
    symbol: str = Query(default="EURUSD", description="Trading symbol"),
) -> MarketInfoResponse:
    """Get current market information.

    Returns current price, daily change, and other market data.
    Data is from yfinance with ~15-20 minute delay.
    """
    try:
        info = data_service.get_market_info(symbol)

        return MarketInfoResponse(
            symbol=info.get("symbol", symbol),
            price=info.get("price"),
            change=info.get("change"),
            change_pct=info.get("change_pct"),
            day_high=info.get("day_high"),
            day_low=info.get("day_low"),
            prev_close=info.get("prev_close"),
            timestamp=info.get("timestamp", ""),
            data_source=info.get("data_source", "yfinance"),
            delay_minutes=info.get("delay_minutes", 15),
            error=info.get("error"),
        )

    except Exception as e:
        logger.error(f"Error getting market info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market/candles", response_model=CandlesResponse)
async def get_candles(
    symbol: str = Query(default="EURUSD", description="Trading symbol"),
    timeframe: str = Query(default="1H", description="Candle timeframe"),
    count: int = Query(default=24, le=100, description="Number of candles"),
) -> CandlesResponse:
    """Get recent OHLCV candles.

    Returns candle data for charting.
    """
    try:
        df = data_service.get_recent_candles(symbol, timeframe, count)

        if df is None or df.empty:
            raise HTTPException(
                status_code=503,
                detail="No candle data available",
            )

        candles = [
            CandleResponse(
                timestamp=str(idx),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]) if "volume" in row else None,
            )
            for idx, row in df.iterrows()
        ]

        return CandlesResponse(
            symbol=symbol,
            timeframe=timeframe,
            candles=candles,
            count=len(candles),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting candles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market/vix")
async def get_vix() -> Dict[str, Any]:
    """Get current VIX value.

    VIX is used as a sentiment indicator in the Daily model.
    """
    try:
        vix = data_service.get_latest_vix()

        return {
            "value": vix,
            "source": "yfinance",
            "symbol": "^VIX",
        }

    except Exception as e:
        logger.error(f"Error getting VIX: {e}")
        raise HTTPException(status_code=500, detail=str(e))
