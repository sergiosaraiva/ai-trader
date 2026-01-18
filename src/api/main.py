"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import predictions, trading, health, market, pipeline
from .database.session import init_db
from .services.data_service import data_service
from .services.model_service import model_service
from .services.trading_service import trading_service
from .services.pipeline_service import pipeline_service
from .utils.logging import log_exception

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

    # Start scheduler
    logger.info("Starting scheduler...")
    try:
        from .scheduler import start_scheduler
        start_scheduler()
    except ImportError as e:
        log_exception(logger, "Scheduler start failed: missing APScheduler", e)
    except RuntimeError as e:
        log_exception(logger, "Scheduler start failed: scheduler already running", e)
    except Exception as e:
        log_exception(logger, "Scheduler start failed with unexpected error", e)

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
    # Note: allow_credentials=False when using wildcard origins for security
    # For production, use explicit origins: allow_origins=["https://yourdomain.com"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=False,  # Must be False with wildcard origins
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
    app.include_router(trading.router, prefix="/api/v1", tags=["Trading"])
    app.include_router(market.router, prefix="/api/v1", tags=["Market"])
    app.include_router(pipeline.router, prefix="/api/v1", tags=["Pipeline"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
