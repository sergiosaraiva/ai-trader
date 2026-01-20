"""Scheduler for periodic tasks.

Handles:
- Data pipeline (hourly: update price data, calculate features, update sentiment)
- Market data fetching (every 5 minutes for price display)
- Prediction generation (every hour, after pipeline)
- Paper trade execution (after each prediction)
- Position monitoring (every 5 minutes)
"""

import gc
import logging
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from .database.models import Prediction
from .database.session import get_session, init_db
from .services.data_service import data_service
from .services.model_service import model_service
from .services.trading_service import trading_service
from .services.pipeline_service import pipeline_service

logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler: Optional[BackgroundScheduler] = None


def run_data_pipeline() -> None:
    """Run the data pipeline (runs every hour at :55).

    This updates:
    - 5-min price data from yfinance
    - Resampled 1H, 4H, Daily data
    - Technical indicators for each timeframe
    - Enhanced features (patterns, cross-TF alignment)
    - Sentiment data (VIX, EPU)

    Runs before predictions at :01 to ensure fresh data.
    """
    try:
        logger.info("Running data pipeline...")

        success = pipeline_service.run_full_pipeline()

        if success:
            status = pipeline_service.get_status()
            logger.info(
                f"Pipeline complete: "
                f"price={status['cache_files']['1h']}, "
                f"sentiment={status['cache_files']['sentiment']}"
            )
        else:
            logger.warning("Pipeline run failed or incomplete")

    except Exception as e:
        logger.error(f"Error running data pipeline: {e}")


def fetch_market_data() -> None:
    """Fetch latest market data for display (runs every 5 minutes)."""
    try:
        logger.debug("Fetching market data...")

        # Refresh price cache
        price = data_service.get_current_price()
        vix = data_service.get_latest_vix()

        if price:
            vix_str = f"{vix:.2f}" if vix else "N/A"
            logger.debug(f"EUR/USD: {price:.5f}, VIX: {vix_str}")

    except Exception as e:
        logger.error(f"Error fetching market data: {e}")


def check_positions() -> None:
    """Check open positions for exit conditions (runs every 5 minutes)."""
    try:
        status = trading_service.get_status()

        if not status.get("has_position"):
            return

        # Get current price
        price = data_service.get_current_price()
        if price is None:
            logger.warning("Cannot check position: no price data")
            return

        # Check if position should be closed
        result = trading_service.check_and_close_position(price)

        if result:
            logger.info(f"Position closed: {result['exit_reason']}, {result['pips']:.1f} pips")

    except Exception as e:
        logger.error(f"Error checking positions: {e}")


def generate_prediction() -> None:
    """Generate prediction and optionally execute trade (runs every hour)."""
    try:
        logger.info("Generating hourly prediction...")

        # Try to get data from pipeline cache first
        df = pipeline_service.get_processed_data("1h")

        if df is None or len(df) < 100:
            # Fallback to data service if pipeline not ready
            logger.info("Pipeline cache not ready, using data service fallback")
            df = data_service.get_data_for_prediction()

        if df is None or len(df) < 100:
            logger.warning("Insufficient data for prediction")
            return

        # Get current price and VIX
        current_price = data_service.get_current_price()
        vix_value = data_service.get_latest_vix()

        if current_price is None:
            logger.warning("Cannot generate prediction: no price data")
            return

        # Make prediction using pipeline data for all timeframes
        prediction = model_service.predict_from_pipeline()

        logger.info(
            f"Prediction: {prediction['direction'].upper()} "
            f"(conf: {prediction['confidence']:.1%}, "
            f"agreement: {prediction['agreement_count']}/3)"
        )

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
            )

            db.add(pred_record)
            db.commit()

            # Execute trade if conditions met
            if prediction["should_trade"]:
                trade = trading_service.execute_trade(prediction, current_price, db)

                if trade:
                    # Mark prediction as having executed trade
                    pred_record.trade_executed = True
                    db.commit()

        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            db.rollback()
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error generating prediction: {e}")


def save_performance_snapshot() -> None:
    """Save performance snapshot (runs every hour)."""
    try:
        trading_service.save_performance_snapshot()
    except Exception as e:
        logger.error(f"Error saving performance snapshot: {e}")


def cleanup_memory() -> None:
    """Clean up service caches and run garbage collection.

    Runs periodically to prevent memory leaks from growing caches
    and to reclaim memory from temporary DataFrames.
    """
    try:
        logger.debug("Running memory cleanup...")

        # Clear service caches (keep historical data)
        data_service.clear_cache(release_historical=False)

        # Clear model cache
        model_service.clear_cache()

        # Run pipeline cleanup (closes session, clears temp references)
        pipeline_service.cleanup()

        # Force garbage collection
        collected = gc.collect()

        logger.info(f"Memory cleanup complete - collected {collected} objects")

    except Exception as e:
        logger.error(f"Error during memory cleanup: {e}")


def start_scheduler() -> BackgroundScheduler:
    """Start the background scheduler with all jobs.

    Returns:
        The scheduler instance
    """
    global scheduler

    if scheduler is not None and scheduler.running:
        logger.warning("Scheduler already running")
        return scheduler

    logger.info("Starting scheduler...")

    scheduler = BackgroundScheduler()

    # Job 1: Run data pipeline every hour at :55 (before predictions)
    scheduler.add_job(
        run_data_pipeline,
        trigger=CronTrigger(minute=55),  # Run at :55 each hour
        id="run_data_pipeline",
        name="Run Data Pipeline",
        replace_existing=True,
        max_instances=1,
    )

    # Job 2: Fetch market data every 5 minutes (for display)
    scheduler.add_job(
        fetch_market_data,
        trigger=IntervalTrigger(minutes=5),
        id="fetch_market_data",
        name="Fetch Market Data",
        replace_existing=True,
        max_instances=1,
    )

    # Job 3: Check positions every 5 minutes
    scheduler.add_job(
        check_positions,
        trigger=IntervalTrigger(minutes=5),
        id="check_positions",
        name="Check Positions",
        replace_existing=True,
        max_instances=1,
    )

    # Job 4: Generate prediction at the start of each hour
    scheduler.add_job(
        generate_prediction,
        trigger=CronTrigger(minute=1),  # Run at :01 each hour
        id="generate_prediction",
        name="Generate Prediction",
        replace_existing=True,
        max_instances=1,
    )

    # Job 5: Save performance snapshot every hour
    scheduler.add_job(
        save_performance_snapshot,
        trigger=CronTrigger(minute=5),  # Run at :05 each hour
        id="save_performance",
        name="Save Performance",
        replace_existing=True,
        max_instances=1,
    )

    # Job 6: Periodic memory cleanup every 4 hours
    scheduler.add_job(
        cleanup_memory,
        trigger=IntervalTrigger(hours=4),
        id="cleanup_memory",
        name="Memory Cleanup",
        replace_existing=True,
        max_instances=1,
    )

    scheduler.start()
    logger.info("Scheduler started with jobs:")
    for job in scheduler.get_jobs():
        logger.info(f"  - {job.name}: {job.trigger}")

    return scheduler


def stop_scheduler() -> None:
    """Stop the scheduler."""
    global scheduler

    if scheduler is not None and scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

    scheduler = None


def get_scheduler_status() -> dict:
    """Get scheduler status."""
    global scheduler

    if scheduler is None:
        return {"running": False, "jobs": []}

    return {
        "running": scheduler.running,
        "jobs": [
            {
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time) if job.next_run_time else None,
                "trigger": str(job.trigger),
            }
            for job in scheduler.get_jobs()
        ],
    }


def run_prediction_now() -> dict:
    """Manually trigger a prediction (for testing)."""
    try:
        generate_prediction()
        return {"status": "success", "message": "Prediction generated"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def run_pipeline_now() -> dict:
    """Manually trigger the data pipeline (for testing)."""
    try:
        run_data_pipeline()
        status = pipeline_service.get_status()
        return {
            "status": "success",
            "message": "Pipeline executed",
            "pipeline_status": status,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def seed_historical_predictions(min_predictions: int = 24) -> dict:
    """Seed historical predictions if database has fewer than min_predictions.

    This generates historical predictions using actual model inference on
    historical data points, giving users something to see immediately after
    deployment.

    Args:
        min_predictions: Minimum number of predictions to have in database

    Returns:
        dict with status and count of predictions seeded
    """
    from datetime import timedelta

    db = get_session()
    try:
        # Check current count
        current_count = db.query(Prediction).count()

        if current_count >= min_predictions:
            logger.info(f"Database has {current_count} predictions, skipping seed")
            return {"status": "skipped", "current_count": current_count}

        # Need to seed predictions
        predictions_to_add = min_predictions - current_count
        logger.info(f"Seeding {predictions_to_add} historical predictions...")

        if not model_service.is_loaded:
            logger.warning("Model not loaded, cannot seed predictions")
            return {"status": "error", "message": "Model not loaded"}

        # Get historical data from pipeline
        df = pipeline_service.get_processed_data("1h")
        if df is None or len(df) < 200:
            df = data_service.get_data_for_prediction()

        if df is None or len(df) < 200:
            logger.warning("Insufficient data to seed predictions")
            return {"status": "error", "message": "Insufficient data"}

        # Get VIX value for all predictions (use latest)
        vix_value = data_service.get_latest_vix()

        # Generate predictions at hourly intervals going back in time
        seeded = 0
        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

        for i in range(predictions_to_add):
            try:
                # Calculate timestamp for this prediction (going back in time)
                pred_time = now - timedelta(hours=i + 1)

                # Skip if prediction already exists for this hour
                existing = db.query(Prediction).filter(
                    Prediction.timestamp >= pred_time,
                    Prediction.timestamp < pred_time + timedelta(hours=1)
                ).first()

                if existing:
                    continue

                # Make prediction using current model
                prediction = model_service.predict_from_pipeline()

                # Get price from historical data if available
                market_price = None
                if 'close' in df.columns:
                    # Try to find price close to this timestamp
                    market_price = float(df['close'].iloc[-1 - i]) if len(df) > i else None

                if market_price is None:
                    market_price = data_service.get_current_price() or 1.0800

                # Create prediction record
                pred_record = Prediction(
                    timestamp=pred_time,
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
                    market_price=market_price,
                    vix_value=vix_value,
                    trade_executed=False,
                )

                db.add(pred_record)
                seeded += 1

            except Exception as e:
                logger.warning(f"Failed to seed prediction {i}: {e}")
                continue

        db.commit()
        logger.info(f"Seeded {seeded} historical predictions")

        return {"status": "success", "seeded": seeded, "total": current_count + seeded}

    except Exception as e:
        logger.error(f"Error seeding predictions: {e}")
        db.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        db.close()
