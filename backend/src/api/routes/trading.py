"""Trading endpoints."""

import json
import logging
from datetime import datetime
from pathlib import Path
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


# Path to backtest results JSON file
BACKTEST_RESULTS_PATH = Path(__file__).parent.parent.parent.parent / "data" / "backtest_results.json"

# Cache for backtest results (reloaded when file changes)
_backtest_cache: Dict[str, Any] = {}
_backtest_cache_mtime: float = 0


def load_backtest_results() -> Dict[str, Any]:
    """Load backtest results from JSON file with caching.

    Results are cached and reloaded when the file is modified.
    This allows monthly updates without code changes or restarts.
    """
    global _backtest_cache, _backtest_cache_mtime

    # Check if file exists
    if not BACKTEST_RESULTS_PATH.exists():
        logger.warning(f"Backtest results file not found: {BACKTEST_RESULTS_PATH}")
        return _get_fallback_backtest_data()

    # Check if cache is still valid (file not modified)
    try:
        current_mtime = BACKTEST_RESULTS_PATH.stat().st_mtime
        if _backtest_cache and current_mtime == _backtest_cache_mtime:
            return _backtest_cache
    except OSError:
        pass

    # Load from file
    try:
        with open(BACKTEST_RESULTS_PATH, "r") as f:
            data = json.load(f)
        _backtest_cache = data
        _backtest_cache_mtime = BACKTEST_RESULTS_PATH.stat().st_mtime
        logger.info(f"Loaded backtest results from {BACKTEST_RESULTS_PATH}")
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Error loading backtest results: {e}")
        return _get_fallback_backtest_data()


def _get_fallback_backtest_data() -> Dict[str, Any]:
    """Return fallback data if JSON file is unavailable."""
    return {
        "metadata": {"last_updated": None, "error": "Backtest results file not found"},
        "periods": {
            "1y": {
                "label": "Last Year",
                "total_pips": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "total_trades": 0,
                "period_start": "N/A",
                "period_end": "N/A",
                "period_years": 1.0,
                "period_months": 12,
            }
        },
        "leverage_options": [
            {"value": 1, "label": "No Leverage (1:1)", "risk": "low"},
        ],
        "forex_constants": {
            "standard_lot_size": 100000,
            "pip_value_per_lot": 10,
        },
    }


@router.get("/trading/backtest-periods", response_model=Dict[str, Any])
async def get_backtest_periods() -> Dict[str, Any]:
    """Get backtest performance data by time period.

    Returns historical backtest performance metrics for different time periods,
    used by the What If Calculator to show potential returns.

    Data is loaded from data/backtest_results.json, which is updated by
    the walk_forward_optimization.py script during monthly backtest runs.
    """
    data = load_backtest_results()

    return {
        "periods": data.get("periods", {}),
        "leverage_options": data.get("leverage_options", []),
        "forex_constants": data.get("forex_constants", {}),
        "data_source": f"WFO Validation (updated: {data.get('metadata', {}).get('last_updated', 'unknown')})",
    }


@router.get("/trading/whatif-performance", response_model=Dict[str, Any])
async def get_whatif_performance(
    days: int = Query(default=30, ge=7, le=90, description="Number of days to simulate"),
    confidence_threshold: float = Query(default=0.70, ge=0.5, le=0.9, description="Minimum confidence to trade"),
    require_agreement: bool = Query(default=True, description="Require at least 2 models to agree on direction"),
) -> Dict[str, Any]:
    """Get simulated 'What If' performance for the last N days.

    Runs the model on historical data using proper triple barrier labeling
    (same methodology as training) to calculate realistic P&L.

    Triple barrier parameters (1H model):
    - Take Profit: 25 pips
    - Stop Loss: 15 pips
    - Max Holding: 12 bars (12 hours)

    Returns daily performance data suitable for the PerformanceChart component.
    """
    from ..services.model_service import model_service
    from ..services.pipeline_service import pipeline_service
    import pandas as pd
    import numpy as np

    # Triple barrier parameters matching training config (1H model)
    TAKE_PROFIT_PIPS = 25.0
    STOP_LOSS_PIPS = 15.0
    MAX_HOLDING_BARS = 12
    PIP_SIZE = 0.0001  # For EURUSD

    try:
        # Ensure model is loaded
        if not model_service.is_loaded:
            model_service.initialize()

        if not model_service.is_loaded:
            raise HTTPException(status_code=503, detail="Model not available")

        # Get historical feature data from pipeline cache
        df_1h = pipeline_service.get_processed_data("1h")
        df_4h = pipeline_service.get_processed_data("4h")
        df_daily = pipeline_service.get_processed_data("D")

        if df_1h is None or len(df_1h) < days * 24:
            raise HTTPException(status_code=503, detail="Insufficient historical data")

        # Get last N days of 1H bars (24 bars per day) + buffer for holding period
        n_bars = days * 24 + MAX_HOLDING_BARS
        df_1h_recent = df_1h.iloc[-n_bars:].copy()

        # Align 4H and Daily data
        start_time = df_1h_recent.index.min()
        df_4h_recent = df_4h[df_4h.index >= start_time].copy() if df_4h is not None else None
        df_daily_recent = df_daily[df_daily.index >= start_time].copy() if df_daily is not None else None

        # Run simulation with proper triple barrier logic
        completed_trades = []
        current_position = None
        pip_multiplier = 10000  # For EURUSD

        # Only process bars within the requested time window (not the buffer)
        simulation_start_idx = MAX_HOLDING_BARS

        for i in range(simulation_start_idx, len(df_1h_recent)):
            bar = df_1h_recent.iloc[i]
            bar_time = df_1h_recent.index[i]

            # Check if we have an open position that needs to be evaluated
            if current_position is not None:
                entry_price = current_position["entry_price"]
                direction = current_position["direction"]
                bars_held = current_position["bars_held"] + 1
                current_position["bars_held"] = bars_held

                # Calculate TP/SL levels
                if direction == "long":
                    tp_price = entry_price + TAKE_PROFIT_PIPS * PIP_SIZE
                    sl_price = entry_price - STOP_LOSS_PIPS * PIP_SIZE
                    # Check if TP or SL hit (using high/low)
                    hit_tp = bar["high"] >= tp_price
                    hit_sl = bar["low"] <= sl_price
                else:  # short
                    tp_price = entry_price - TAKE_PROFIT_PIPS * PIP_SIZE
                    sl_price = entry_price + STOP_LOSS_PIPS * PIP_SIZE
                    hit_tp = bar["low"] <= tp_price
                    hit_sl = bar["high"] >= sl_price

                # Determine exit
                exit_reason = None
                exit_pips = 0.0

                if hit_tp and hit_sl:
                    # Both hit in same bar - assume SL hit first (conservative)
                    exit_reason = "stop_loss"
                    exit_pips = -STOP_LOSS_PIPS
                elif hit_tp:
                    exit_reason = "take_profit"
                    exit_pips = TAKE_PROFIT_PIPS
                elif hit_sl:
                    exit_reason = "stop_loss"
                    exit_pips = -STOP_LOSS_PIPS
                elif bars_held >= MAX_HOLDING_BARS:
                    # Max holding period - exit at close
                    exit_reason = "max_holding"
                    if direction == "long":
                        exit_pips = (bar["close"] - entry_price) / PIP_SIZE
                    else:
                        exit_pips = (entry_price - bar["close"]) / PIP_SIZE

                if exit_reason:
                    # Close the position
                    completed_trades.append({
                        "entry_time": current_position["entry_time"],
                        "exit_time": bar_time,
                        "direction": direction,
                        "pips": round(exit_pips, 1),
                        "exit_reason": exit_reason,
                        "is_winner": exit_pips > 0,
                    })
                    current_position = None

            # If no position, check for new signal
            if current_position is None:
                # Get aligned data up to this point
                df_1h_slice = df_1h_recent.iloc[:i+1]
                df_4h_slice = df_4h_recent[df_4h_recent.index <= bar_time] if df_4h_recent is not None else None
                df_daily_slice = df_daily_recent[df_daily_recent.index <= bar_time] if df_daily_recent is not None else None

                if len(df_1h_slice) < 50:  # Need enough history for features
                    continue

                try:
                    # Get prediction for this bar
                    prediction = model_service.ensemble.predict_from_features(
                        df_1h_slice,
                        df_4h_slice if df_4h_slice is not None and len(df_4h_slice) > 0 else df_1h_slice,
                        df_daily_slice if df_daily_slice is not None and len(df_daily_slice) > 0 else df_1h_slice,
                    )

                    # Check if prediction meets threshold and has clear direction
                    # Direction can be int (1=long, -1/0=short) or string ("long"/"short")
                    direction = prediction.direction
                    if isinstance(direction, int):
                        direction = "long" if direction == 1 else "short"

                    # Check agreement requirement (at least 2 of 3 models agree)
                    meets_agreement = (
                        not require_agreement or
                        prediction.agreement_count >= 2
                    )

                    if (prediction.confidence >= confidence_threshold and
                        direction in ("long", "short") and
                        meets_agreement):
                        # Open new position at current bar's close
                        current_position = {
                            "entry_time": bar_time,
                            "entry_price": bar["close"],
                            "direction": direction,
                            "confidence": prediction.confidence,
                            "bars_held": 0,
                        }

                except Exception as e:
                    logger.debug(f"Prediction failed for bar {bar_time}: {e}")
                    continue

        # Aggregate trades into daily results
        daily_results = {}
        for trade in completed_trades:
            exit_date = trade["exit_time"].strftime("%Y-%m-%d")

            if exit_date not in daily_results:
                daily_results[exit_date] = {
                    "date": exit_date,
                    "trades": 0,
                    "wins": 0,
                    "daily_pnl": 0.0,
                }

            daily_results[exit_date]["trades"] += 1
            daily_results[exit_date]["daily_pnl"] += trade["pips"]
            if trade["is_winner"]:
                daily_results[exit_date]["wins"] += 1

        # Convert to sorted list and calculate cumulative P&L
        daily_list = sorted(daily_results.values(), key=lambda x: x["date"])

        cumulative_pnl = 0.0
        for day in daily_list:
            cumulative_pnl += day["daily_pnl"]
            day["cumulative_pnl"] = round(cumulative_pnl, 1)
            day["daily_pnl"] = round(day["daily_pnl"], 1)
            day["win_rate"] = round(day["wins"] / day["trades"] * 100, 0) if day["trades"] > 0 else 0

        # Calculate summary stats
        total_trades = sum(d["trades"] for d in daily_list)
        total_wins = sum(d["wins"] for d in daily_list)
        total_pnl = sum(d["daily_pnl"] for d in daily_list)
        profitable_days = sum(1 for d in daily_list if d["daily_pnl"] > 0)

        return {
            "daily_performance": daily_list,
            "summary": {
                "total_days": len(daily_list),
                "profitable_days": profitable_days,
                "total_trades": total_trades,
                "total_wins": total_wins,
                "win_rate": round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0,
                "total_pnl": round(total_pnl, 1),
                "avg_daily_pnl": round(total_pnl / len(daily_list), 1) if daily_list else 0,
                "confidence_threshold": confidence_threshold,
                "require_agreement": require_agreement,
                "tp_pips": TAKE_PROFIT_PIPS,
                "sl_pips": STOP_LOSS_PIPS,
                "max_holding_bars": MAX_HOLDING_BARS,
            },
            "data_source": "Historical simulation (Triple Barrier)",
            "generated_at": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating what-if performance: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
