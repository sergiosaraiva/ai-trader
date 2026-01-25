#!/usr/bin/env python3
"""
Seed trades database with realistic historical trade data.

This script generates trade data based on actual backtest statistics
to populate the 30-Day Performance Chart.

Usage:
    python scripts/seed_trades_from_backtest.py [--days 45] [--clear]
"""

import argparse
import logging
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.database.models import Base, Trade

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "db" / "trading.db"
BACKTEST_RESULTS_PATH = DATA_DIR / "backtest_results.json"


def load_backtest_stats() -> dict:
    """Load backtest statistics from backtest_results.json (canonical source).

    Uses 70% confidence threshold data for trade generation.
    """
    import json

    default_stats = {
        "win_rate": 0.50,
        "avg_winner_pips": 20.0,
        "avg_loser_pips": -12.0,
        "trades_per_day": 1.5,
        "tp_pips": 25.0,
        "sl_pips": 15.0,
        "avg_confidence": 0.75,
    }

    if not BACKTEST_RESULTS_PATH.exists():
        logger.warning(f"Backtest results not found at {BACKTEST_RESULTS_PATH}, using defaults")
        return default_stats

    try:
        with open(BACKTEST_RESULTS_PATH) as f:
            data = json.load(f)

        # Get 70% threshold data (primary) or 5y all-time (fallback)
        threshold_70 = data.get("by_threshold", {}).get("0.70", {})
        all_time = data.get("periods", {}).get("5y", {})

        # Calculate win rate from threshold data or all-time
        win_rate = threshold_70.get("win_rate", all_time.get("win_rate", 50)) / 100

        # Calculate trades per day (total trades / trading days ~252 per year * years)
        total_trades = threshold_70.get("total_trades", all_time.get("total_trades", 1000))
        period_years = all_time.get("period_years", 4)
        trades_per_day = total_trades / (period_years * 252)

        return {
            "win_rate": win_rate,
            "avg_winner_pips": 20.0,  # Based on TP 25 pips minus slippage
            "avg_loser_pips": -12.0,  # Based on SL 15 pips minus slippage
            "trades_per_day": round(trades_per_day, 1),
            "tp_pips": 25.0,
            "sl_pips": 15.0,
            "avg_confidence": 0.75,
        }

    except Exception as e:
        logger.warning(f"Could not load backtest stats: {e}, using defaults")
        return default_stats


# Load backtest statistics dynamically from JSON
BACKTEST_STATS = load_backtest_stats()


def generate_trades(days: int = 45) -> list[dict]:
    """Generate realistic trade data based on backtest statistics."""
    trades = []
    random.seed(42)  # Reproducible results
    np.random.seed(42)

    # Base price around current EUR/USD levels
    base_price = 1.0300
    pip_size = 0.0001

    # Generate trades for each day (skip weekends)
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    current_date = end_date - timedelta(days=days)

    trade_id = 0

    while current_date < end_date:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue

        # Random number of trades this day (Poisson distribution)
        num_trades = np.random.poisson(BACKTEST_STATS["trades_per_day"])
        num_trades = max(0, min(num_trades, 5))  # Cap at 5 trades per day

        for _ in range(num_trades):
            trade_id += 1

            # Random entry time during trading hours (8:00 - 20:00 UTC)
            hour = random.randint(8, 19)
            minute = random.choice([0, 15, 30, 45])
            entry_time = current_date.replace(hour=hour, minute=minute)

            # Random direction
            direction = random.choice(["long", "short"])

            # Random price with some variation
            price_offset = random.uniform(-0.02, 0.02)
            entry_price = base_price + price_offset

            # Determine if winner based on win rate
            is_winner = random.random() < BACKTEST_STATS["win_rate"]

            # Calculate pips (with some variation)
            if is_winner:
                pips = BACKTEST_STATS["avg_winner_pips"] + random.gauss(0, 5)
                pips = max(5, min(pips, BACKTEST_STATS["tp_pips"]))  # Cap at TP
                exit_reason = "tp" if pips >= BACKTEST_STATS["tp_pips"] * 0.9 else "timeout"
            else:
                pips = BACKTEST_STATS["avg_loser_pips"] + random.gauss(0, 3)
                pips = max(-BACKTEST_STATS["sl_pips"], min(pips, -3))  # Cap at SL
                exit_reason = "sl" if abs(pips) >= BACKTEST_STATS["sl_pips"] * 0.9 else "timeout"

            pips = round(pips, 1)

            # Calculate exit price
            if direction == "long":
                exit_price = entry_price + (pips * pip_size)
                tp = entry_price + (BACKTEST_STATS["tp_pips"] * pip_size)
                sl = entry_price - (BACKTEST_STATS["sl_pips"] * pip_size)
            else:
                exit_price = entry_price - (pips * pip_size)
                tp = entry_price - (BACKTEST_STATS["tp_pips"] * pip_size)
                sl = entry_price + (BACKTEST_STATS["sl_pips"] * pip_size)

            # Exit time (1-12 hours after entry)
            holding_hours = random.randint(1, 12)
            exit_time = entry_time + timedelta(hours=holding_hours)

            # Confidence (around average with some variation)
            confidence = BACKTEST_STATS["avg_confidence"] + random.gauss(0, 0.05)
            confidence = max(0.70, min(confidence, 0.92))  # Keep in valid range

            # P&L in USD ($10 per pip for 0.1 lot)
            pnl_usd = pips * 10.0

            trade = {
                "symbol": "EURUSD",
                "direction": direction,
                "entry_price": round(entry_price, 5),
                "entry_time": entry_time,
                "exit_price": round(exit_price, 5),
                "exit_time": exit_time,
                "exit_reason": exit_reason,
                "lot_size": 0.1,
                "take_profit": round(tp, 5),
                "stop_loss": round(sl, 5),
                "pips": pips,
                "pnl_usd": round(pnl_usd, 2),
                "is_winner": is_winner,
                "confidence": round(confidence, 3),
                "status": "closed",
            }
            trades.append(trade)

        current_date += timedelta(days=1)

    # Sort by entry time
    trades.sort(key=lambda x: x["entry_time"])

    return trades


def seed_database(trades: list[dict], clear_existing: bool = False):
    """Save trades to the database."""
    # Ensure db directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(f"sqlite:///{DB_PATH}")
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        if clear_existing:
            deleted = session.query(Trade).delete()
            logger.info(f"Cleared {deleted} existing trades")

        # Add trades
        for trade_data in trades:
            trade = Trade(**trade_data)
            session.add(trade)

        session.commit()
        logger.info(f"Saved {len(trades)} trades to database")

        # Log summary
        total_pips = sum(t["pips"] for t in trades)
        winners = sum(1 for t in trades if t["is_winner"])
        win_rate = winners / len(trades) * 100 if trades else 0

        # Calculate by day for chart preview
        daily_pnl = {}
        for t in trades:
            day = t["exit_time"].strftime("%Y-%m-%d")
            daily_pnl[day] = daily_pnl.get(day, 0) + t["pips"]

        profitable_days = sum(1 for pnl in daily_pnl.values() if pnl > 0)

        logger.info(f"Summary:")
        logger.info(f"  - Total trades: {len(trades)}")
        logger.info(f"  - Winners: {winners} ({win_rate:.1f}%)")
        logger.info(f"  - Total pips: {total_pips:.1f}")
        logger.info(f"  - Trading days: {len(daily_pnl)}")
        logger.info(f"  - Profitable days: {profitable_days} ({profitable_days/len(daily_pnl)*100:.0f}%)")

    except Exception as e:
        session.rollback()
        logger.error(f"Error saving trades: {e}")
        raise
    finally:
        session.close()


def check_existing_closed_trades() -> int:
    """Check how many CLOSED trades exist in the database."""
    if not DB_PATH.exists():
        return 0

    engine = create_engine(f"sqlite:///{DB_PATH}")
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Only count closed trades - open trades from live trading don't count
        count = session.query(Trade).filter(Trade.status == "closed").count()
        return count
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description="Seed trades from backtest statistics")
    parser.add_argument("--days", type=int, default=45, help="Days of trades to generate")
    parser.add_argument("--clear", action="store_true", help="Clear existing trades first")
    parser.add_argument("--force", action="store_true", help="Force seeding even if trades exist")
    args = parser.parse_args()

    # Check if closed trades already exist (open trades from live trading don't count)
    existing = check_existing_closed_trades()
    if existing > 0 and not args.clear and not args.force:
        logger.info(f"Database already has {existing} closed trades. Use --clear or --force to reseed.")
        sys.exit(0)

    logger.info("=" * 60)
    logger.info("Seeding trades database from backtest statistics")
    logger.info("=" * 60)
    logger.info(f"Generating {args.days} days of trade data...")

    # Generate trades
    trades = generate_trades(days=args.days)

    if not trades:
        logger.warning("No trades generated")
        sys.exit(1)

    # Save to database
    seed_database(trades, clear_existing=args.clear)

    logger.info("=" * 60)
    logger.info("Database seeding complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
