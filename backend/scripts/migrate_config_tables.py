"""Migration script to add configuration management tables.

Adds:
- configuration_settings: Centralized configuration with versioning
- configuration_history: Change history and audit trail
"""

import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.src.api.database.session import engine, init_db
from backend.src.api.database.models import Base, ConfigurationSetting, ConfigurationHistory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def migrate():
    """Create configuration tables."""
    logger.info("Starting configuration tables migration...")

    # Create all tables (will only create missing ones)
    try:
        init_db()
        logger.info("Database initialization complete")

        # Verify tables exist
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        if "configuration_settings" in tables:
            logger.info("✓ configuration_settings table exists")
        else:
            logger.error("✗ configuration_settings table not created")

        if "configuration_history" in tables:
            logger.info("✓ configuration_history table exists")
        else:
            logger.error("✗ configuration_history table not created")

        logger.info("Migration complete!")

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


if __name__ == "__main__":
    migrate()
