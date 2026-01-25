#!/usr/bin/env python3
"""
Migrate AI Trader from SQLite to PostgreSQL.

This script handles the complete migration process:
1. Creates PostgreSQL schema
2. Migrates existing data from SQLite (if exists)
3. Validates the migration

Usage:
    python scripts/migrate_to_postgres.py [--sqlite-path path/to/trading.db]
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

# Import models
from src.api.database.models import Base
from src.api.database.session import DATABASE_URL, engine as pg_engine


def create_postgres_schema():
    """Create all tables in PostgreSQL."""
    print("Creating PostgreSQL schema...")
    try:
        Base.metadata.create_all(bind=pg_engine)
        print("✓ PostgreSQL schema created successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to create schema: {e}")
        return False


def migrate_sqlite_data(sqlite_path: str):
    """Migrate data from SQLite to PostgreSQL."""
    if not os.path.exists(sqlite_path):
        print(f"SQLite database not found at {sqlite_path}, skipping data migration")
        return True

    print(f"Migrating data from {sqlite_path}...")

    try:
        # Create SQLite engine
        sqlite_engine = create_engine(f"sqlite:///{sqlite_path}")
        sqlite_session = sessionmaker(bind=sqlite_engine)()

        # Create PostgreSQL session
        pg_session = sessionmaker(bind=pg_engine)()

        # Get inspector for both databases
        sqlite_inspector = inspect(sqlite_engine)
        pg_inspector = inspect(pg_engine)

        # Get common tables
        sqlite_tables = set(sqlite_inspector.get_table_names())
        pg_tables = set(pg_inspector.get_table_names())
        common_tables = sqlite_tables & pg_tables

        print(f"Found {len(common_tables)} tables to migrate: {', '.join(common_tables)}")

        # Migrate each table
        for table_name in common_tables:
            print(f"  Migrating {table_name}...", end="")

            # Count records in SQLite
            count_result = sqlite_session.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            ).scalar()

            if count_result == 0:
                print(" (empty)")
                continue

            # Get data from SQLite
            # Note: We use raw SQL here to handle any table structure
            result = sqlite_session.execute(text(f"SELECT * FROM {table_name}"))
            rows = result.fetchall()
            columns = result.keys()

            # Prepare PostgreSQL insert
            # Check which columns exist in PostgreSQL
            pg_columns = {col["name"] for col in pg_inspector.get_columns(table_name)}

            # Filter to only columns that exist in both
            common_columns = [col for col in columns if col in pg_columns]

            if not common_columns:
                print(f" ✗ No common columns between SQLite and PostgreSQL")
                continue

            # Insert data into PostgreSQL
            inserted = 0
            for row in rows:
                # Build insert data with only common columns
                insert_data = {
                    col: getattr(row, col) for col in common_columns
                    if getattr(row, col) is not None
                }

                if insert_data:
                    # Use parameterized query
                    columns_str = ", ".join(insert_data.keys())
                    values_str = ", ".join([f":{key}" for key in insert_data.keys()])

                    try:
                        pg_session.execute(
                            text(f"INSERT INTO {table_name} ({columns_str}) VALUES ({values_str})"),
                            insert_data
                        )
                        inserted += 1
                    except Exception as e:
                        # Skip duplicates or constraint violations
                        pass

            pg_session.commit()
            print(f" ✓ ({inserted}/{count_result} records)")

        sqlite_session.close()
        pg_session.close()

        print("✓ Data migration completed successfully")
        return True

    except Exception as e:
        print(f"✗ Data migration failed: {e}")
        return False


def validate_migration():
    """Validate that the PostgreSQL database is properly set up."""
    print("\nValidating PostgreSQL setup...")

    try:
        # Test connection
        with pg_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
            print("✓ Connection successful")

            # Check tables
            inspector = inspect(pg_engine)
            tables = inspector.get_table_names()

            expected_tables = [
                "predictions", "trades", "performance_snapshots",
                "market_data", "agent_commands", "agent_state",
                "trade_explanations", "circuit_breaker_events"
            ]

            for table in expected_tables:
                if table in tables:
                    # Count records
                    count = conn.execute(
                        text(f"SELECT COUNT(*) FROM {table}")
                    ).scalar()
                    print(f"✓ Table '{table}' exists ({count} records)")
                else:
                    print(f"✗ Table '{table}' missing")

            # Test write capability
            test_time = datetime.utcnow()
            conn.execute(
                text("""
                    INSERT INTO predictions
                    (timestamp, symbol, direction, confidence, should_trade)
                    VALUES (:ts, :symbol, :dir, :conf, :should_trade)
                """),
                {
                    "ts": test_time,
                    "symbol": "EURUSD",
                    "dir": "test",
                    "conf": 0.5,
                    "should_trade": False
                }
            )
            conn.commit()

            # Clean up test data
            conn.execute(
                text("DELETE FROM predictions WHERE direction = 'test'")
            )
            conn.commit()

            print("✓ Write test successful")

        print("\n✓ PostgreSQL migration validated successfully!")
        return True

    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        return False


def main():
    """Main migration process."""
    parser = argparse.ArgumentParser(description="Migrate AI Trader to PostgreSQL")
    parser.add_argument(
        "--sqlite-path",
        default="data/trading.db",
        help="Path to existing SQLite database (default: data/trading.db)"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data migration, only create schema"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing PostgreSQL setup"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("AI TRADER - PostgreSQL Migration")
    print("=" * 60)
    print(f"PostgreSQL URL: {DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'configured'}")
    print()

    if args.validate_only:
        success = validate_migration()
    else:
        # Step 1: Create schema
        success = create_postgres_schema()

        # Step 2: Migrate data (if requested)
        if success and not args.skip_data:
            success = migrate_sqlite_data(args.sqlite_path)

        # Step 3: Validate
        if success:
            success = validate_migration()

    if success:
        print("\n" + "=" * 60)
        print("SUCCESS: PostgreSQL migration completed!")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("FAILED: Migration encountered errors")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())