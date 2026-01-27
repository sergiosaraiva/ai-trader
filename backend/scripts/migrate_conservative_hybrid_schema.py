#!/usr/bin/env python3
"""
Database migration script for Conservative Hybrid position sizing.

Adds:
1. risk_percentage_used column to trades table
2. circuit_breaker_events table for persistence
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
    """Run database migration for Conservative Hybrid features."""
    
    # Get database URL from config
    db_url = os.getenv("DATABASE_URL", "postgresql://aitrader:aitrader@postgres:5432/aitrader")
    
    logger.info(f"Connecting to database: {db_url.split('@')[1]}")  # Hide password
    engine = create_engine(db_url)
    
    with engine.connect() as conn:
        # Start transaction
        trans = conn.begin()
        
        try:
            # Migration 1: Add risk_percentage_used to trades table
            logger.info("Adding risk_percentage_used column to trades table...")
            conn.execute(text("""
                ALTER TABLE trades 
                ADD COLUMN IF NOT EXISTS risk_percentage_used FLOAT;
            """))
            logger.info("✓ risk_percentage_used column added")
            
            # Migration 2: Create circuit_breaker_events table
            logger.info("Creating circuit_breaker_events table...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS circuit_breaker_events (
                    id SERIAL PRIMARY KEY,
                    breaker_type VARCHAR NOT NULL,
                    action VARCHAR NOT NULL,
                    triggered_at TIMESTAMP WITH TIME ZONE NOT NULL,
                    recovered_at TIMESTAMP WITH TIME ZONE,
                    value FLOAT NOT NULL,
                    event_metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """))
            logger.info("✓ circuit_breaker_events table created")
            
            # Migration 3: Add indexes for performance
            logger.info("Creating indexes...")
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_circuit_breaker_type 
                ON circuit_breaker_events(breaker_type);
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_circuit_breaker_triggered_at 
                ON circuit_breaker_events(triggered_at);
            """))
            logger.info("✓ Indexes created")
            
            # Commit transaction
            trans.commit()
            logger.info("✅ Migration completed successfully!")
            
        except Exception as e:
            trans.rollback()
            logger.error(f"❌ Migration failed: {e}")
            raise

if __name__ == "__main__":
    run_migration()
