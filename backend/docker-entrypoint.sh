#!/bin/bash
set -e

echo "=========================================="
echo "Starting AI-Trader Backend..."
echo "=========================================="

# Seed trades if database is empty
echo "Checking if trades need to be seeded..."
python /app/scripts/seed_trades_from_backtest.py --days 45 || echo "WARNING: Seeding failed or skipped"

echo "Starting uvicorn on port ${PORT:-8001}..."

# Start the application
exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8001}
