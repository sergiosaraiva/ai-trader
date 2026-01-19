"""Trading endpoints."""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session

from ..database.session import get_db
from ..database.models import Trade
from ..services.data_service import data_service
from ..services.trading_service import trading_service
from ..schemas.trading import (
    TradingStatusResponse,
    TradeResponse,
    TradeHistoryResponse,
    PerformanceResponse,
    EquityCurveResponse,
    EquityPoint,
    OpenPositionResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/trading/status", response_model=TradingStatusResponse)
async def get_trading_status() -> TradingStatusResponse:
    """Get current trading status.

    Returns account balance, equity, and open position details.
    """
    try:
        status = trading_service.get_status()

        # Get current price for unrealized P&L calculation
        current_price = None
        open_position = None

        if status.get("open_position"):
            pos = status["open_position"]
            current_price = data_service.get_current_price()

            # Calculate unrealized P&L
            unrealized_pips = 0.0
            unrealized_pnl = 0.0

            if current_price:
                pip_size = 0.0001
                if pos["direction"] == "long":
                    unrealized_pips = (current_price - pos["entry_price"]) / pip_size
                else:
                    unrealized_pips = (pos["entry_price"] - current_price) / pip_size
                unrealized_pnl = unrealized_pips * 10.0  # $10 per pip for 0.1 lot

            open_position = OpenPositionResponse(
                id=pos["id"],
                symbol=pos["symbol"],
                direction=pos["direction"],
                entry_price=pos["entry_price"],
                entry_time=(
                    pos["entry_time"].isoformat()
                    if isinstance(pos["entry_time"], datetime)
                    else str(pos["entry_time"])
                ),
                lot_size=pos["lot_size"],
                take_profit=pos["take_profit"],
                stop_loss=pos["stop_loss"],
                confidence=pos.get("confidence"),
                current_price=current_price,
                unrealized_pips=unrealized_pips,
                unrealized_pnl=unrealized_pnl,
            )

        return TradingStatusResponse(
            mode="paper",
            balance=status["balance"],
            equity=status["equity"] + (open_position.unrealized_pnl if open_position else 0),
            unrealized_pnl=open_position.unrealized_pnl if open_position else 0.0,
            has_position=status["has_position"],
            open_position=open_position,
        )

    except Exception as e:
        logger.error(f"Error getting trading status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trading/history", response_model=TradeHistoryResponse)
async def get_trade_history(
    limit: int = Query(default=50, le=500, description="Number of trades"),
    status: Optional[str] = Query(
        default=None, description="Filter by status: 'open' or 'closed'"
    ),
    db: Session = Depends(get_db),
) -> TradeHistoryResponse:
    """Get trade history.

    Returns past trades with details including P&L.
    """
    try:
        query = db.query(Trade)

        if status:
            query = query.filter(Trade.status == status)

        trades = query.order_by(Trade.entry_time.desc()).limit(limit).all()

        items = [
            TradeResponse(
                id=t.id,
                symbol=t.symbol,
                direction=t.direction,
                entry_price=t.entry_price,
                entry_time=t.entry_time.isoformat() if t.entry_time else "",
                exit_price=t.exit_price,
                exit_time=t.exit_time.isoformat() if t.exit_time else None,
                exit_reason=t.exit_reason,
                lot_size=t.lot_size,
                take_profit=t.take_profit,
                stop_loss=t.stop_loss,
                pips=t.pips,
                pnl_usd=t.pnl_usd,
                is_winner=t.is_winner,
                confidence=t.confidence,
                status=t.status,
            )
            for t in trades
        ]

        return TradeHistoryResponse(trades=items, count=len(items))

    except Exception as e:
        logger.error(f"Error getting trade history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trading/performance", response_model=PerformanceResponse)
async def get_trading_performance() -> PerformanceResponse:
    """Get trading performance metrics.

    Returns win rate, profit factor, total P&L, and other statistics.
    """
    try:
        perf = trading_service.get_performance()

        return PerformanceResponse(
            total_trades=perf["total_trades"],
            winning_trades=perf["winning_trades"],
            losing_trades=perf["losing_trades"],
            win_rate=perf["win_rate"],
            total_pips=perf["total_pips"],
            total_pnl_usd=perf["total_pnl_usd"],
            avg_pips_per_trade=perf["avg_pips_per_trade"],
            profit_factor=perf["profit_factor"],
            initial_balance=perf["initial_balance"],
            current_balance=perf["current_balance"],
            return_pct=perf["return_pct"],
        )

    except Exception as e:
        logger.error(f"Error getting performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trading/equity-curve", response_model=EquityCurveResponse)
async def get_equity_curve(
    db: Session = Depends(get_db),
) -> EquityCurveResponse:
    """Get equity curve data.

    Returns balance/equity over time for charting.
    """
    try:
        curve_data = trading_service.get_equity_curve(db)

        points = [
            EquityPoint(
                timestamp=p["timestamp"],
                balance=p["balance"],
                equity=p["equity"],
            )
            for p in curve_data
        ]

        return EquityCurveResponse(data=points, count=len(points))

    except Exception as e:
        logger.error(f"Error getting equity curve: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trading/close-position")
async def close_position_manually(
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Manually close the open position.

    Closes at current market price with 'manual' exit reason.
    """
    try:
        status = trading_service.get_status()

        if not status.get("has_position"):
            raise HTTPException(status_code=400, detail="No open position to close")

        current_price = data_service.get_current_price()
        if current_price is None:
            raise HTTPException(status_code=503, detail="Cannot get current price")

        result = trading_service.close_position(current_price, "manual", db)

        if result is None:
            raise HTTPException(status_code=500, detail="Failed to close position")

        return {
            "status": "success",
            "message": f"Position closed at {current_price:.5f}",
            "trade": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Legacy endpoints for backward compatibility


@router.get("/positions", response_model=List[Dict[str, Any]])
async def get_positions() -> List[Dict[str, Any]]:
    """Get all open positions (legacy endpoint)."""
    try:
        if not trading_service.is_loaded:
            logger.warning("Positions requested but trading service not loaded")
            return []

        status = trading_service.get_status()
        if status.get("open_position"):
            return [status["open_position"]]
        return []

    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_legacy() -> Dict[str, Any]:
    """Get trading performance (legacy endpoint)."""
    try:
        if not trading_service.is_loaded:
            logger.warning("Performance requested but trading service not loaded")
            return {"error": "Service not initialized"}

        return trading_service.get_performance()

    except Exception as e:
        logger.error(f"Error getting performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk/metrics", response_model=Dict[str, Any])
async def get_risk_metrics() -> Dict[str, Any]:
    """Get current risk metrics."""
    try:
        if not trading_service.is_loaded:
            logger.warning("Risk metrics requested but trading service not loaded")
            return {"error": "Service not initialized"}

        status = trading_service.get_status()
        perf = trading_service.get_performance()

        return {
            "account_balance": status["balance"],
            "daily_pnl": 0.0,  # Would need daily tracking
            "weekly_pnl": 0.0,  # Would need weekly tracking
            "max_drawdown": 0.0,  # Would need peak tracking
            "current_exposure": 1.0 if status["has_position"] else 0.0,
            "is_halted": False,
        }

    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Backtest performance data by time period (from WFO validation with 70% confidence threshold)
# Data source: Walk-Forward Optimization windows in docs/02-walk-forward-optimization-results.md
BACKTEST_PERIODS = {
    "6m": {
        "label": "Last 6 Months",
        "total_pips": 2079,
        "win_rate": 0.477,
        "profit_factor": 1.48,
        "total_trades": 568,
        "period_start": "2025-01-01",
        "period_end": "2025-06-30",
        "period_years": 0.5,
        "period_months": 6,
    },
    "1y": {
        "label": "Last Year",
        "total_pips": 4317,  # Window 6 + 7
        "win_rate": 0.517,
        "profit_factor": 1.73,
        "total_trades": 948,
        "period_start": "2024-07-01",
        "period_end": "2025-06-30",
        "period_years": 1.0,
        "period_months": 12,
    },
    "2y": {
        "label": "Last 2 Years",
        "total_pips": 6749,  # Windows 5+6+7
        "win_rate": 0.551,
        "profit_factor": 1.93,
        "total_trades": 1282,
        "period_start": "2024-01-01",
        "period_end": "2025-06-30",
        "period_years": 2.0,
        "period_months": 24,
    },
    "3y": {
        "label": "Last 3 Years",
        "total_pips": 9839,  # Windows 3+4+5+6+7
        "win_rate": 0.524,
        "profit_factor": 1.77,
        "total_trades": 2185,
        "period_start": "2023-01-01",
        "period_end": "2025-06-30",
        "period_years": 3.0,
        "period_months": 36,
    },
    "5y": {
        "label": "All Time (5 Years)",
        "total_pips": 8693,
        "win_rate": 0.621,
        "profit_factor": 2.69,
        "total_trades": 966,
        "period_start": "2020-01-01",
        "period_end": "2025-12-31",
        "period_years": 5.0,
        "period_months": 60,
    },
}

# Leverage options for the calculator
LEVERAGE_OPTIONS = [
    {"value": 1, "label": "No Leverage (1:1)", "risk": "low"},
    {"value": 10, "label": "10:1", "risk": "medium"},
    {"value": 20, "label": "20:1", "risk": "high"},
    {"value": 30, "label": "30:1 (EU Retail)", "risk": "high"},
    {"value": 50, "label": "50:1", "risk": "extreme"},
]


@router.get("/trading/backtest-periods", response_model=Dict[str, Any])
async def get_backtest_periods() -> Dict[str, Any]:
    """Get backtest performance data by time period.

    Returns historical backtest performance metrics for different time periods,
    used by the What If Calculator to show potential returns.

    Data source: Walk-Forward Optimization validation results.
    """
    return {
        "periods": BACKTEST_PERIODS,
        "leverage_options": LEVERAGE_OPTIONS,
        "forex_constants": {
            "standard_lot_size": 100000,
            "pip_value_per_lot": 10,
        },
        "data_source": "WFO Validation (70% confidence threshold)",
    }
