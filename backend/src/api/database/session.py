"""Database session management for AI-Trader API."""

import os
import logging
from typing import Generator

from sqlalchemy import create_engine, text, inspect, DDL
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool, QueuePool

from .models import Base

logger = logging.getLogger(__name__)

# Get database URL from environment (PostgreSQL only)
DATABASE_URL = os.environ.get("DATABASE_URL")

# In development, allow a default but warn about it
if not DATABASE_URL:
    if os.environ.get("ENVIRONMENT", "development") == "development":
        DATABASE_URL = "postgresql://aitrader:aitrader_dev_password@localhost:5432/aitrader"
        print("WARNING: Using default DATABASE_URL for development")
    else:
        raise ValueError("DATABASE_URL environment variable must be set in production")

# Connection pool configuration (configurable via environment)
POOL_SIZE = int(os.environ.get("DB_POOL_SIZE", "5"))
MAX_OVERFLOW = int(os.environ.get("DB_MAX_OVERFLOW", "10"))
POOL_TIMEOUT = int(os.environ.get("DB_POOL_TIMEOUT", "30"))
POOL_RECYCLE = int(os.environ.get("DB_POOL_RECYCLE", "1800"))  # 30 minutes
SQL_ECHO = os.environ.get("DB_SQL_ECHO", "false").lower() == "true"

# Security warning for SQL echo in production
if SQL_ECHO and os.environ.get("ENVIRONMENT") == "production":
    logger.warning(
        "WARNING: DB_SQL_ECHO is enabled in production. "
        "This may expose sensitive query data in logs. Disable for production."
    )

# Create engine with appropriate connection pooling
# Use NullPool for serverless environments (Railway, etc.)
# Use QueuePool for local development with persistent containers
if os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("SERVERLESS"):
    engine = create_engine(
        DATABASE_URL,
        poolclass=NullPool,
        pool_pre_ping=True,  # Test connections before using them
        echo=SQL_ECHO,
    )
    logger.info("Database engine created with NullPool (serverless mode)")
else:
    # Local development or persistent environments with configurable pooling
    engine = create_engine(
        DATABASE_URL,
        pool_size=POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
        pool_timeout=POOL_TIMEOUT,
        pool_recycle=POOL_RECYCLE,  # Recycle connections to prevent stale connections
        pool_pre_ping=True,  # Test connections before using them
        echo=SQL_ECHO,
    )
    logger.info(
        f"Database engine created with QueuePool "
        f"(size={POOL_SIZE}, overflow={MAX_OVERFLOW}, timeout={POOL_TIMEOUT}s, "
        f"recycle={POOL_RECYCLE}s)"
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for getting database session.

    Usage in FastAPI:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables.

    Call this on application startup to create all tables.
    """
    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Run migrations for existing tables
    run_migrations()


def run_migrations() -> None:
    """Run database migrations for schema updates.

    This handles adding new columns to existing tables without data loss.
    All migrations are run in a transaction with automatic rollback on failure.
    """
    inspector = inspect(engine)

    # Check if predictions table exists
    if "predictions" not in inspector.get_table_names():
        return

    # Get existing columns for predictions table
    predictions_columns = {col["name"] for col in inspector.get_columns("predictions")}
    trades_columns = {col["name"] for col in inspector.get_columns("trades")} if "trades" in inspector.get_table_names() else set()

    # Use begin() for automatic commit/rollback
    with engine.begin() as conn:
        try:
            # Migration 1: Add should_trade column to predictions if missing
            if "should_trade" not in predictions_columns:
                # Use DDL for safer execution
                alter_stmt = DDL("ALTER TABLE predictions ADD COLUMN should_trade BOOLEAN DEFAULT TRUE")
                conn.execute(alter_stmt)

                # Update existing rows
                update_stmt = text(
                    "UPDATE predictions SET should_trade = (confidence >= :threshold)"
                ).bindparams(threshold=0.70)
                conn.execute(update_stmt)
                print("Migration: Added should_trade column to predictions")

            # Migration 2: Add new fields to predictions for agent tracking
            if "used_by_agent" not in predictions_columns:
                alter_stmt = DDL("ALTER TABLE predictions ADD COLUMN used_by_agent BOOLEAN DEFAULT FALSE")
                conn.execute(alter_stmt)
                print("Migration: Added used_by_agent column to predictions")

            if "agent_cycle_number" not in predictions_columns:
                alter_stmt = DDL("ALTER TABLE predictions ADD COLUMN agent_cycle_number INTEGER")
                conn.execute(alter_stmt)
                print("Migration: Added agent_cycle_number column to predictions")

            # Migration 3: Add new fields to trades for agent operations
            if "trades" in inspector.get_table_names():
                if "execution_mode" not in trades_columns:
                    alter_stmt = DDL("ALTER TABLE trades ADD COLUMN execution_mode VARCHAR(20) DEFAULT 'simulation'")
                    conn.execute(alter_stmt)
                    print("Migration: Added execution_mode column to trades")

                if "broker" not in trades_columns:
                    alter_stmt = DDL("ALTER TABLE trades ADD COLUMN broker VARCHAR(50)")
                    conn.execute(alter_stmt)
                    print("Migration: Added broker column to trades")

                if "mt5_ticket" not in trades_columns:
                    alter_stmt = DDL("ALTER TABLE trades ADD COLUMN mt5_ticket INTEGER")
                    conn.execute(alter_stmt)
                    print("Migration: Added mt5_ticket column to trades")

                if "explanation_id" not in trades_columns:
                    alter_stmt = DDL("ALTER TABLE trades ADD COLUMN explanation_id INTEGER")
                    conn.execute(alter_stmt)
                    print("Migration: Added explanation_id column to trades")

        except Exception as e:
            print(f"Migration failed: {e}")
            raise  # Re-raise to trigger automatic rollback


def get_session() -> Session:
    """Get a new database session (non-dependency version).

    Use this when not in FastAPI dependency injection context.
    Remember to close the session when done.
    """
    return SessionLocal()
