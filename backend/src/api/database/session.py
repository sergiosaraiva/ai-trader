"""Database session management for AI-Trader API."""

import os
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, Session

from .models import Base

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent.parent / "data" / "db" / "trading.db"

# Get database URL from environment or use default SQLite
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{DEFAULT_DB_PATH}"
)

# Create engine
# For SQLite, we need to set check_same_thread=False for multi-threaded access
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,  # Set to True for SQL debugging
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
    # Ensure directory exists
    db_path = DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Run migrations for existing tables
    run_migrations()


def run_migrations() -> None:
    """Run database migrations for schema updates.

    This handles adding new columns to existing tables without data loss.
    """
    inspector = inspect(engine)

    # Check if predictions table exists
    if "predictions" not in inspector.get_table_names():
        return

    # Get existing columns
    columns = {col["name"] for col in inspector.get_columns("predictions")}

    # Migration: Add should_trade column if missing
    if "should_trade" not in columns:
        with engine.connect() as conn:
            # Add column with default value
            conn.execute(text(
                "ALTER TABLE predictions ADD COLUMN should_trade BOOLEAN DEFAULT 1"
            ))
            # Update existing records based on confidence threshold (70%)
            conn.execute(text(
                "UPDATE predictions SET should_trade = (confidence >= 0.70)"
            ))
            conn.commit()


def get_session() -> Session:
    """Get a new database session (non-dependency version).

    Use this when not in FastAPI dependency injection context.
    Remember to close the session when done.
    """
    return SessionLocal()
