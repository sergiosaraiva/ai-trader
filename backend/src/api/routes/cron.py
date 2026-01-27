"""Cron job endpoints for serverless/Railway cron deployments.

These endpoints replace the APScheduler when running in CRON mode.
Configure Railway cron jobs to call these endpoints at regular intervals.

Recommended Railway cron schedule:
- POST /api/v1/cron/tick     -> Every hour at :01 (0 1 * * *)
- POST /api/v1/cron/check    -> Every 5 minutes (*/5 * * * *)
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Header

from ..services.data_service import data_service
from ..services.model_service import model_service
from ..services.trading_service import trading_service
from ..services.pipeline_service import pipeline_service
from ..database.session import get_session
from ..database.models import Prediction

logger = logging.getLogger(__name__)

router = APIRouter()

# Optional API key for securing cron endpoints
CRON_API_KEY = os.getenv("CRON_API_KEY") or None  # Treat empty string as None


def verify_cron_auth(x_cron_key: Optional[str] = Header(None)) -> bool:
    """Verify cron API key if configured."""
    if CRON_API_KEY is None:
        return True  # No auth required if key not set

    if x_cron_key != CRON_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-Cron-Key header"
        )
    return True


@router.post("/cron/tick")
async def cron_tick(x_cron_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Main cron tick - runs the full hourly cycle.

    This endpoint should be called every hour (e.g., at :01).
    It performs:
    1. Data pipeline update (fetch prices, calculate features)
    2. Generate new prediction
    3. Execute paper trade if conditions met
    4. Save performance snapshot

    Configure Railway cron: `0 1 * * *` (every hour at :01)

    Returns:
        Status and results of the cron cycle
    """
    verify_cron_auth(x_cron_key)

    logger.info("Cron tick started")
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline": None,
        "prediction": None,
        "trade": None,
        "errors": [],
    }

    # Step 1: Run data pipeline
    try:
        logger.info("Running data pipeline...")
        success = pipeline_service.run_full_pipeline()
        results["pipeline"] = {
            "status": "success" if success else "partial",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        results["errors"].append(f"Pipeline: {str(e)}")
        results["pipeline"] = {"status": "error", "message": str(e)}

    # Step 2: Generate prediction
    try:
        logger.info("Generating prediction...")

        if not model_service.is_loaded:
            raise RuntimeError("Model not loaded")

        # Get prediction using pipeline data
        prediction = model_service.predict_from_pipeline()

        if prediction is None:
            raise RuntimeError("Prediction returned None")

        # Get current price and VIX
        current_price = data_service.get_current_price()
        vix_value = data_service.get_latest_vix()

        results["prediction"] = {
            "direction": prediction["direction"],
            "confidence": prediction["confidence"],
            "should_trade": prediction["should_trade"],
            "agreement_count": prediction["agreement_count"],
            "market_price": current_price,
        }

        # Save prediction to database
        db = get_session()
        try:
            pred_record = Prediction(
                timestamp=datetime.utcnow(),
                symbol="EURUSD",
                direction=prediction["direction"],
                confidence=prediction["confidence"],
                prob_up=prediction["prob_up"],
                prob_down=prediction["prob_down"],
                pred_1h=prediction["component_directions"].get("1H"),
                conf_1h=prediction["component_confidences"].get("1H"),
                pred_4h=prediction["component_directions"].get("4H"),
                conf_4h=prediction["component_confidences"].get("4H"),
                pred_d=prediction["component_directions"].get("D"),
                conf_d=prediction["component_confidences"].get("D"),
                agreement_count=prediction["agreement_count"],
                agreement_score=prediction["agreement_score"],
                market_regime=prediction["market_regime"],
                market_price=current_price,
                vix_value=vix_value,
                trade_executed=False,
                should_trade=prediction["should_trade"],
            )
            db.add(pred_record)
            db.commit()

            # Step 3: Execute trade if conditions met
            if prediction["should_trade"] and current_price:
                trade = trading_service.execute_trade(prediction, current_price, db)
                if trade:
                    pred_record.trade_executed = True
                    db.commit()
                    results["trade"] = {
                        "executed": True,
                        "direction": trade.get("direction"),
                        "entry_price": trade.get("entry_price"),
                    }
                else:
                    results["trade"] = {"executed": False, "reason": "Trade conditions not met"}
            else:
                results["trade"] = {"executed": False, "reason": "should_trade=False or no price"}

        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        results["errors"].append(f"Prediction: {str(e)}")
        results["prediction"] = {"status": "error", "message": str(e)}

    # Step 4: Save performance snapshot
    try:
        trading_service.save_performance_snapshot()
    except Exception as e:
        logger.error(f"Performance snapshot error: {e}")
        results["errors"].append(f"Performance: {str(e)}")

    logger.info(f"Cron tick completed: {len(results['errors'])} errors")

    return {
        "status": "success" if not results["errors"] else "partial",
        "results": results,
    }


@router.post("/cron/check")
async def cron_check_positions(x_cron_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Check positions and fetch market data.

    This endpoint should be called every 5 minutes.
    It performs:
    1. Fetch latest market data (price, VIX)
    2. Check open positions for TP/SL exit conditions

    Configure Railway cron: `*/5 * * * *` (every 5 minutes)

    Returns:
        Status of position check
    """
    verify_cron_auth(x_cron_key)

    logger.debug("Cron check started")
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "market_data": None,
        "position_check": None,
        "errors": [],
    }

    # Fetch market data
    try:
        price = data_service.get_current_price()
        vix = data_service.get_latest_vix()
        results["market_data"] = {
            "price": price,
            "vix": vix,
        }
    except Exception as e:
        logger.error(f"Market data error: {e}")
        results["errors"].append(f"Market data: {str(e)}")

    # Check positions
    try:
        status = trading_service.get_status()

        if status.get("has_position"):
            price = data_service.get_current_price()
            if price:
                result = trading_service.check_and_close_position(price)
                if result:
                    results["position_check"] = {
                        "closed": True,
                        "exit_reason": result.get("exit_reason"),
                        "pips": result.get("pips"),
                    }
                else:
                    results["position_check"] = {"closed": False, "reason": "No exit condition met"}
            else:
                results["position_check"] = {"closed": False, "reason": "No price data"}
        else:
            results["position_check"] = {"closed": False, "reason": "No open position"}

    except Exception as e:
        logger.error(f"Position check error: {e}")
        results["errors"].append(f"Position check: {str(e)}")

    return {
        "status": "success" if not results["errors"] else "partial",
        "results": results,
    }


@router.get("/cron/status")
async def get_cron_status() -> Dict[str, Any]:
    """Get current operating mode and scheduler status.

    Returns:
        Information about whether running in scheduler or cron mode
    """
    scheduler_enabled = os.getenv("SCHEDULER_ENABLED", "true").lower() in ("true", "1", "yes")

    # Try to get scheduler status if running
    scheduler_status = None
    if scheduler_enabled:
        try:
            from ..scheduler import get_scheduler_status
            scheduler_status = get_scheduler_status()
        except Exception:
            pass

    return {
        "mode": "scheduler" if scheduler_enabled else "cron",
        "scheduler_enabled": scheduler_enabled,
        "scheduler_status": scheduler_status,
        "cron_endpoints": {
            "tick": "POST /api/v1/cron/tick (hourly)",
            "check": "POST /api/v1/cron/check (every 5 min)",
        },
        "cron_auth_required": CRON_API_KEY is not None,
    }
