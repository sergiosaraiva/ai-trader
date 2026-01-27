#!/usr/bin/env python3
"""
Database migration script for Progressive Risk Reduction.

Adds:
1. risk_reduction_state table for tracking consecutive losses and risk factors
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from src.config import trading_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migration():
    """Run database migration for Progressive Risk Reduction features."""

    # Get database URL from config
    db_url = os.getenv("DATABASE_URL", "postgresql://aitrader:aitrader@postgres:5432/aitrader")

    logger.info(f"Connecting to database: {db_url.split('@')[1] if '@' in db_url else 'local'}")  # Hide password
    engine = create_engine(db_url)

    with engine.connect() as conn:
        # Start transaction
        trans = conn.begin()

        try:
            # Migration 1: Create risk_reduction_state table
            logger.info("Creating risk_reduction_state table...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS risk_reduction_state (
                    id SERIAL PRIMARY KEY,
                    consecutive_losses INTEGER NOT NULL DEFAULT 0,
                    risk_reduction_factor FLOAT NOT NULL DEFAULT 1.0,
                    last_trade_id INTEGER REFERENCES trades(id) ON DELETE SET NULL,
                    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
                );
            """))
            logger.info("✓ risk_reduction_state table created")

            # Migration 2: Add indexes for performance
            logger.info("Creating indexes...")
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_risk_reduction_updated
                ON risk_reduction_state(updated_at);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_risk_reduction_last_trade
                ON risk_reduction_state(last_trade_id);
            """))
            logger.info("✓ Indexes created")

            # Migration 3: Initialize state if table is empty
            logger.info("Initializing risk reduction state...")
            conn.execute(text("""
                INSERT INTO risk_reduction_state (consecutive_losses, risk_reduction_factor)
                SELECT 0, 1.0
                WHERE NOT EXISTS (SELECT 1 FROM risk_reduction_state);
            """))
            logger.info("✓ Initial state created")

            # Commit transaction
            trans.commit()
            logger.info("✅ Migration completed successfully!")

        except Exception as e:
            trans.rollback()
            logger.error(f"❌ Migration failed: {e}")
            raise

if __name__ == "__main__":
    run_migration()
