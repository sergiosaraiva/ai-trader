#!/usr/bin/env python3
"""Migration script for dynamic threshold system database tables.

Creates:
- threshold_history table for threshold calculations
- dynamic_threshold_used column in predictions table

Run this script to add the threshold tracking infrastructure to an existing database.
"""

import logging
import sqlite3
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "trading.db"


def create_threshold_history_table(conn):
    """Create threshold_history table."""
    logger.info("Creating threshold_history table...")

    cursor = conn.cursor()

    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS threshold_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            threshold_value FLOAT NOT NULL,
            short_term_component FLOAT,
            medium_term_component FLOAT,
            long_term_component FLOAT,
            blended_value FLOAT,
            performance_adjustment FLOAT,
            prediction_count_7d INTEGER NOT NULL DEFAULT 0,
            prediction_count_14d INTEGER NOT NULL DEFAULT 0,
            prediction_count_30d INTEGER NOT NULL DEFAULT 0,
            trade_win_rate_25 FLOAT,
            trade_count_25 INTEGER,
            reason VARCHAR(200),
            config_version INTEGER
        )
    """)

    # Create indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_threshold_history_timestamp
        ON threshold_history (timestamp)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_threshold_history_reason
        ON threshold_history (reason)
    """)

    conn.commit()
    logger.info("threshold_history table created successfully")


def add_dynamic_threshold_column(conn):
    """Add dynamic_threshold_used column to predictions table."""
    logger.info("Adding dynamic_threshold_used column to predictions table...")

    cursor = conn.cursor()

    # Check if column already exists
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [row[1] for row in cursor.fetchall()]

    if "dynamic_threshold_used" in columns:
        logger.info("dynamic_threshold_used column already exists, skipping")
        return

    # Add column
    cursor.execute("""
        ALTER TABLE predictions
        ADD COLUMN dynamic_threshold_used FLOAT
    """)

    conn.commit()
    logger.info("dynamic_threshold_used column added successfully")


def verify_migration(conn):
    """Verify that migration was successful."""
    logger.info("Verifying migration...")

    cursor = conn.cursor()

    # Check threshold_history table
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='threshold_history'
    """)
    if not cursor.fetchone():
        logger.error("threshold_history table not found!")
        return False
    logger.info("threshold_history table exists")

    # Check predictions.dynamic_threshold_used column
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [row[1] for row in cursor.fetchall()]

    if "dynamic_threshold_used" not in columns:
        logger.error("predictions.dynamic_threshold_used column not found!")
        return False
    logger.info("predictions.dynamic_threshold_used column exists")

    # Try to query threshold_history
    cursor.execute("SELECT COUNT(*) FROM threshold_history")
    count = cursor.fetchone()[0]
    logger.info(f"threshold_history table is accessible ({count} records)")

    logger.info("Migration verification successful!")
    return True


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Dynamic Threshold System Migration")
    logger.info("=" * 60)

    # Check if database exists
    if not DB_PATH.exists():
        logger.error(f"Database not found at: {DB_PATH}")
        logger.error("Please create the database first by running the API")
        return False

    try:
        # Connect to database
        logger.info(f"Connecting to database: {DB_PATH}")
        conn = sqlite3.connect(str(DB_PATH))

        # Create threshold_history table
        create_threshold_history_table(conn)

        # Add dynamic_threshold_used column
        add_dynamic_threshold_column(conn)

        # Verify migration
        success = verify_migration(conn)

        # Close connection
        conn.close()

        if success:
            logger.info("=" * 60)
            logger.info("Migration completed successfully!")
            logger.info("=" * 60)
            logger.info("")
            logger.info("Next steps:")
            logger.info("1. Restart the API server to initialize threshold service")
            logger.info("2. Check threshold status: GET /api/v1/threshold/status")
            logger.info("3. View threshold history: GET /api/v1/threshold/history")
            return True
        else:
            logger.error("Migration verification failed!")
            return False

    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
