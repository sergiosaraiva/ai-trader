"""FastAPI application entry point."""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import predictions, trading, health, market, pipeline, cron, performance, agent
from .database.session import init_db
from .services.data_service import data_service
from .services.model_service import model_service
from .services.trading_service import trading_service
from .services.pipeline_service import pipeline_service
from .utils.logging import log_exception
from .utils.rate_limiter import setup_rate_limiting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting AI-Trader API...")

    # Initialize database
    logger.info("Initializing database...")
    init_db()

    # Initialize services
    logger.info("Initializing data service...")
    try:
        data_service.initialize()
    except FileNotFoundError as e:
        log_exception(logger, "Data service initialization failed: data files not found", e)
    except PermissionError as e:
        log_exception(logger, "Data service initialization failed: permission denied", e)
    except Exception as e:
        log_exception(logger, "Data service initialization failed with unexpected error", e)

    logger.info("Initializing pipeline service...")
    try:
        pipeline_service.initialize()
    except FileNotFoundError as e:
        log_exception(logger, "Pipeline service initialization failed: cache files not found", e)
    except PermissionError as e:
        log_exception(logger, "Pipeline service initialization failed: permission denied", e)
    except Exception as e:
        log_exception(logger, "Pipeline service initialization failed with unexpected error", e)

    logger.info("Initializing model service...")
    try:
        model_service.initialize(warm_up=True)
    except FileNotFoundError as e:
        log_exception(logger, "Model service initialization failed: model files not found", e)
    except ImportError as e:
        log_exception(logger, "Model service initialization failed: missing dependencies", e)
    except ValueError as e:
        log_exception(logger, "Model service initialization failed: invalid model configuration", e)
    except Exception as e:
        log_exception(logger, "Model service initialization failed with unexpected error", e)

    logger.info("Initializing trading service...")
    try:
        trading_service.initialize()
    except ConnectionError as e:
        log_exception(logger, "Trading service initialization failed: database connection error", e)
    except Exception as e:
        log_exception(logger, "Trading service initialization failed with unexpected error", e)

    # Seed historical predictions if database is empty
    logger.info("Checking prediction history...")
    try:
        from .scheduler import seed_historical_predictions
        result = seed_historical_predictions(min_predictions=24)
        if result["status"] == "success":
            logger.info(f"Seeded {result['seeded']} historical predictions")
        elif result["status"] == "skipped":
            logger.info(f"Prediction history OK ({result['current_count']} records)")
    except Exception as e:
        log_exception(logger, "Failed to seed historical predictions", e)

    # Start scheduler (optional - can be disabled for serverless/cron deployments)
    scheduler_enabled = os.getenv("SCHEDULER_ENABLED", "true").lower() in ("true", "1", "yes")

    if scheduler_enabled:
        logger.info("Starting scheduler (SCHEDULER_ENABLED=true)...")
        try:
            from .scheduler import start_scheduler
            start_scheduler()
            logger.info("Scheduler started - running in ALWAYS-ON mode")
        except ImportError as e:
            log_exception(logger, "Scheduler start failed: missing APScheduler", e)
        except RuntimeError as e:
            log_exception(logger, "Scheduler start failed: scheduler already running", e)
        except Exception as e:
            log_exception(logger, "Scheduler start failed with unexpected error", e)
    else:
        logger.info("Scheduler disabled (SCHEDULER_ENABLED=false) - running in CRON mode")
        logger.info("Use POST /api/v1/cron/tick to trigger updates via external cron")

    logger.info("AI-Trader API started successfully!")

    yield

    # Shutdown
    logger.info("Shutting down AI-Trader API...")

    try:
        from .scheduler import stop_scheduler
        stop_scheduler()
    except Exception:
        pass

    logger.info("AI-Trader API stopped.")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="AI Assets Trader API",
        description="""
        API for AI-powered forex trading predictions and paper trading simulation.

        ## Features
        - Real-time EUR/USD predictions using MTF Ensemble model
        - Paper trading with $100K virtual balance
        - Performance tracking and statistics
        - Market data from yfinance

        ## Model
        - Multi-Timeframe Ensemble (1H: 60%, 4H: 30%, Daily: 10%)
        - Confidence threshold: 70%
        - Triple barrier labeling (TP: 25 pips, SL: 15 pips)
        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS middleware
    # Allow origins from environment variable or use defaults
    cors_origins_env = os.getenv("CORS_ORIGINS", "")
    if cors_origins_env:
        cors_origins = [origin.strip() for origin in cors_origins_env.split(",")]
    else:
        # Default: allow common development and Railway origins
        cors_origins = [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:5173",
            "https://ai-assets-trader.up.railway.app",
            "https://ai-trader-frontend.up.railway.app",
        ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    logger.info(f"CORS enabled for origins: {cors_origins}")

    # Setup rate limiting
    setup_rate_limiting(app)

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
    app.include_router(trading.router, prefix="/api/v1", tags=["Trading"])
    app.include_router(market.router, prefix="/api/v1", tags=["Market"])
    app.include_router(pipeline.router, prefix="/api/v1", tags=["Pipeline"])
    app.include_router(cron.router, prefix="/api/v1", tags=["Cron"])
    app.include_router(performance.router, prefix="/api/v1", tags=["Performance"])
    app.include_router(agent.router, tags=["Agent"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
