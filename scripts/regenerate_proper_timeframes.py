#!/usr/bin/env python3
"""Regenerate timeframe data with proper OHLCV aggregation (not sliding window).

The original derived data used sliding window which creates many duplicate close prices,
causing severe class imbalance. This script creates properly aggregated timeframes.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent


def load_5min_data(symbol: str = "EURUSD") -> pd.DataFrame:
    """Load combined 5-minute data."""
    # First try to find the full combined file (2020-2025)
    full_combined = project_root / f"data/forex/{symbol}_20200101_20251231_5min_combined.csv"
    if full_combined.exists():
        logger.info(f"Found full combined file: {full_combined}")
        files = [full_combined]
    else:
        # Fall back to pattern matching, prefer larger date ranges
        patterns = [
            f"data/forex/{symbol}_*_5min_combined.csv",
            f"data/forex/raw/{symbol}_5min.parquet",
            f"data/forex/raw/{symbol}_5min.csv",
        ]
        files = []
        for pattern in patterns:
            files = list(project_root.glob(pattern))
            if files:
                # Sort by file size (largest first) to get the most complete file
                files = sorted(files, key=lambda f: f.stat().st_size, reverse=True)
                break

    if not files:
        raise FileNotFoundError(f"No 5-minute data found for {symbol}")

    filepath = files[0]  # Use largest file
    logger.info(f"Loading 5-min data from {filepath}")

    if filepath.suffix == '.parquet':
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]

    # Parse timestamp
    if 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
    elif 'datetime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['datetime'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df.set_index('timestamp')
    df = df.sort_index()

    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]

    # Ensure OHLCV columns exist
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    logger.info(f"Loaded {len(df)} rows from {df.index.min()} to {df.index.max()}")
    return df[required]


def aggregate_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Properly aggregate OHLCV data to target timeframe.

    Args:
        df: 5-minute OHLCV DataFrame with timestamp index
        timeframe: Target timeframe ('1H', '4H', '1D')

    Returns:
        Aggregated DataFrame
    """
    # Map timeframe to pandas resample rule
    resample_rules = {
        '1H': '1h',
        '4H': '4h',
        '1D': '1D',
        '1W': '1W',
    }

    rule = resample_rules.get(timeframe, timeframe.lower())

    logger.info(f"Aggregating to {timeframe} using rule '{rule}'")

    df_agg = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Verify data quality
    close = df_agg['close'].values
    unique_ratio = len(np.unique(close)) / len(close)

    returns = np.diff(close) / close[:-1]
    up_pct = np.sum(returns > 0) / len(returns)
    down_pct = np.sum(returns < 0) / len(returns)

    logger.info(f"  Rows: {len(df_agg)}")
    logger.info(f"  Unique close prices: {len(np.unique(close))} ({unique_ratio:.1%})")
    logger.info(f"  Direction balance: UP {up_pct:.1%}, DOWN {down_pct:.1%}")

    return df_agg


def save_timeframe_data(df: pd.DataFrame, symbol: str, timeframe: str, output_dir: Path):
    """Save aggregated data to parquet."""
    output_path = output_dir / timeframe
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / f"{symbol}_{timeframe}.parquet"

    # Reset index to include timestamp as column
    df_save = df.reset_index()
    df_save.to_parquet(filepath, index=False)

    logger.info(f"Saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Regenerate timeframe data with proper aggregation")
    parser.add_argument("--symbol", type=str, default="EURUSD", help="Currency pair")
    parser.add_argument("--timeframes", type=str, nargs="+", default=["1H", "4H", "1D"],
                        help="Timeframes to generate")
    parser.add_argument("--output-dir", type=str, default="data/forex/derived_proper",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = project_root / args.output_dir

    print("\n" + "=" * 70)
    print("REGENERATING TIMEFRAME DATA WITH PROPER AGGREGATION")
    print("=" * 70)
    print(f"Symbol: {args.symbol}")
    print(f"Timeframes: {args.timeframes}")
    print(f"Output: {output_dir}")
    print("=" * 70 + "\n")

    # Load 5-minute source data
    df_5min = load_5min_data(args.symbol)

    # Generate each timeframe
    for timeframe in args.timeframes:
        print(f"\n--- {timeframe} ---")
        df_tf = aggregate_to_timeframe(df_5min, timeframe)
        save_timeframe_data(df_tf, args.symbol, timeframe, output_dir)

    print("\n" + "=" * 70)
    print("DATA REGENERATION COMPLETE")
    print("=" * 70)
    print(f"\nTo use the new data, update the data path in training scripts:")
    print(f"  data_dir = project_root / '{args.output_dir}' / timeframe")
    print("=" * 70)


if __name__ == "__main__":
    main()
