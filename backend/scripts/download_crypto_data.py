#!/usr/bin/env python3
"""
Crypto Historical Data Downloader

Downloads historical OHLCV data from Binance Public Data (data.binance.vision).
Supports spot and futures markets with various timeframes.

Usage:
    python download_crypto_data.py --symbol BTCUSDT --interval 5m --start 2020-01-01 --end 2025-12-31
    python download_crypto_data.py --symbol ETHUSDT --interval 1h --start 2020-01-01 --end 2025-12-31
"""

import argparse
import os
import zipfile
import io
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from tqdm import tqdm

# Binance data base URL
BASE_URL = "https://data.binance.vision/data"

# Available intervals
VALID_INTERVALS = [
    "1s", "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1mo"
]

# Column names for kline data
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore"
]


def download_monthly_kline(
    symbol: str,
    interval: str,
    year: int,
    month: int,
    market_type: str = "spot",
    output_dir: Path = None,
    max_retries: int = 3
) -> pd.DataFrame | None:
    """
    Download monthly kline data from Binance.

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        interval: Candlestick interval (e.g., 5m, 1h)
        year: Year to download
        month: Month to download (1-12)
        market_type: 'spot' or 'futures'
        output_dir: Directory to save raw ZIP files
        max_retries: Number of retry attempts

    Returns:
        DataFrame with OHLCV data or None if download fails
    """
    filename = f"{symbol}-{interval}-{year}-{month:02d}.zip"

    if market_type == "spot":
        url = f"{BASE_URL}/spot/monthly/klines/{symbol}/{interval}/{filename}"
    else:
        url = f"{BASE_URL}/futures/um/monthly/klines/{symbol}/{interval}/{filename}"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=60)

            if response.status_code == 404:
                # Data not available for this period
                return None

            response.raise_for_status()

            # Extract CSV from ZIP
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                csv_filename = zf.namelist()[0]
                with zf.open(csv_filename) as f:
                    df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)

            # Convert timestamps - Binance uses microseconds for 2025+ data
            # Detect unit by checking timestamp magnitude
            sample_ts = df["open_time"].iloc[0]
            if sample_ts > 1e15:  # Microseconds (2025+)
                df["open_time"] = pd.to_datetime(df["open_time"], unit="us")
                df["close_time"] = pd.to_datetime(df["close_time"], unit="us")
            else:  # Milliseconds (pre-2025)
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

            # Convert numeric columns
            for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                continue
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            print(f"Error downloading {filename}: {e}")

    return None


def download_daily_kline(
    symbol: str,
    interval: str,
    date: datetime,
    market_type: str = "spot",
    max_retries: int = 3
) -> pd.DataFrame | None:
    """
    Download daily kline data from Binance.

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        interval: Candlestick interval (e.g., 5m, 1h)
        date: Date to download
        market_type: 'spot' or 'futures'
        max_retries: Number of retry attempts

    Returns:
        DataFrame with OHLCV data or None if download fails
    """
    date_str = date.strftime("%Y-%m-%d")
    filename = f"{symbol}-{interval}-{date_str}.zip"

    if market_type == "spot":
        url = f"{BASE_URL}/spot/daily/klines/{symbol}/{interval}/{filename}"
    else:
        url = f"{BASE_URL}/futures/um/daily/klines/{symbol}/{interval}/{filename}"

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=30)

            if response.status_code == 404:
                return None

            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                csv_filename = zf.namelist()[0]
                with zf.open(csv_filename) as f:
                    df = pd.read_csv(f, header=None, names=KLINE_COLUMNS)

            # Convert timestamps - Binance uses microseconds for 2025+ data
            sample_ts = df["open_time"].iloc[0]
            if sample_ts > 1e15:  # Microseconds (2025+)
                df["open_time"] = pd.to_datetime(df["open_time"], unit="us")
                df["close_time"] = pd.to_datetime(df["close_time"], unit="us")
            else:  # Milliseconds (pre-2025)
                df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

            for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return df

        except Exception:
            if attempt < max_retries - 1:
                continue

    return None


def generate_month_range(start_date: datetime, end_date: datetime) -> list[tuple[int, int]]:
    """Generate list of (year, month) tuples between start and end dates."""
    months = []
    current = start_date.replace(day=1)

    while current <= end_date:
        months.append((current.year, current.month))
        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return months


def download_crypto_data(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    output_dir: str = "data/crypto",
    market_type: str = "spot",
    workers: int = 4
) -> str:
    """
    Download cryptocurrency historical data from Binance.

    Args:
        symbol: Trading pair (e.g., BTCUSDT, ETHUSDT)
        interval: Candlestick interval (1m, 5m, 15m, 1h, 4h, 1d, etc.)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory for data
        market_type: 'spot' or 'futures'
        workers: Number of parallel download workers

    Returns:
        Path to the combined output file
    """
    # Validate interval
    if interval not in VALID_INTERVALS:
        raise ValueError(f"Invalid interval. Must be one of: {VALID_INTERVALS}")

    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate months to download
    months = generate_month_range(start, end)

    print(f"Downloading {symbol} {interval} data from {start_date} to {end_date}")
    print(f"Market type: {market_type}")
    print(f"Months to download: {len(months)}")

    # Download monthly data in parallel
    all_data = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_monthly_kline, symbol, interval, year, month, market_type
            ): (year, month)
            for year, month in months
        }

        with tqdm(total=len(futures), desc="Downloading months") as pbar:
            for future in as_completed(futures):
                year, month = futures[future]
                try:
                    df = future.result()
                    if df is not None and len(df) > 0:
                        all_data.append(df)
                except Exception as e:
                    print(f"Error for {year}-{month:02d}: {e}")
                pbar.update(1)

    if not all_data:
        print("No data downloaded!")
        return None

    # Combine all data
    print("Combining data...")
    combined = pd.concat(all_data, ignore_index=True)

    # Sort by time
    combined = combined.sort_values("open_time").reset_index(drop=True)

    # Remove duplicates
    combined = combined.drop_duplicates(subset=["open_time"], keep="first")

    # Filter to exact date range
    combined = combined[
        (combined["open_time"] >= start) &
        (combined["open_time"] <= end + timedelta(days=1))
    ]

    # Select and rename columns for consistency
    output = combined[["open_time", "open", "high", "low", "close", "volume"]].copy()
    output.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

    # Save to CSV
    start_str = start_date.replace("-", "")
    end_str = end_date.replace("-", "")
    output_file = output_path / f"{symbol}_{start_str}_{end_str}_{interval}.csv"

    output.to_csv(output_file, index=False)

    print(f"\nDownload complete!")
    print(f"Output file: {output_file}")
    print(f"Total candles: {len(output):,}")
    print(f"Date range: {output['Date'].min()} to {output['Date'].max()}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Download cryptocurrency historical data from Binance"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Trading pair (e.g., BTCUSDT, ETHUSDT)"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="5m",
        choices=VALID_INTERVALS,
        help="Candlestick interval (default: 5m)"
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/crypto",
        help="Output directory (default: data/crypto)"
    )
    parser.add_argument(
        "--market",
        type=str,
        default="spot",
        choices=["spot", "futures"],
        help="Market type (default: spot)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)"
    )

    args = parser.parse_args()

    download_crypto_data(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output,
        market_type=args.market,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
