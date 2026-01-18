#!/usr/bin/env python3
"""
Dukascopy Historical Forex Data Downloader

Downloads tick data from Dukascopy's free data feed and converts to OHLCV candles.
Handles large date ranges by downloading in chunks to avoid timeouts/rate limits.

Data Source: https://datafeed.dukascopy.com/datafeed/
URL Structure: https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YEAR}/{MONTH:00-11}/{DAY}/{HOUR}h_ticks.bi5

The .bi5 files are LZMA-compressed binary tick data with the following structure:
- Each tick is 20 bytes: timestamp(4) + ask(4) + bid(4) + ask_vol(4) + bid_vol(4)
- Timestamp is milliseconds since hour start
- Prices are in pips (multiply by point value for actual price)

Usage:
    python download_forex_data.py --symbol EURUSD --start 2020-01-01 --end 2024-01-01 --timeframe 5min
"""

import argparse
import io
import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

# Dukascopy data feed base URL
BASE_URL = "https://datafeed.dukascopy.com/datafeed"

# Point values for different currency pairs (for price calculation)
POINT_VALUES = {
    # JPY pairs have 3 decimal places, others have 5
    "USDJPY": 0.001,
    "EURJPY": 0.001,
    "GBPJPY": 0.001,
    "AUDJPY": 0.001,
    "NZDJPY": 0.001,
    "CADJPY": 0.001,
    "CHFJPY": 0.001,
    # Standard pairs
    "EURUSD": 0.00001,
    "GBPUSD": 0.00001,
    "AUDUSD": 0.00001,
    "NZDUSD": 0.00001,
    "USDCAD": 0.00001,
    "USDCHF": 0.00001,
    "EURGBP": 0.00001,
    "EURAUD": 0.00001,
    "EURCHF": 0.00001,
    "GBPCHF": 0.00001,
    "AUDCAD": 0.00001,
    "AUDNZD": 0.00001,
    # Metals
    "XAUUSD": 0.001,  # Gold
    "XAGUSD": 0.001,  # Silver
}


def get_point_value(symbol: str) -> float:
    """Get the point value for a symbol."""
    return POINT_VALUES.get(symbol.upper(), 0.00001)


def download_hour_ticks(
    symbol: str,
    year: int,
    month: int,  # 0-indexed (0=January, 11=December)
    day: int,
    hour: int,
    session: requests.Session,
    max_retries: int = 3,
) -> Optional[bytes]:
    """
    Download tick data for a specific hour from Dukascopy.

    Args:
        symbol: Currency pair (e.g., 'EURUSD')
        year: Year (e.g., 2024)
        month: Month (0-indexed: 0=Jan, 11=Dec)
        day: Day of month (1-31)
        hour: Hour (0-23)
        session: Requests session for connection pooling
        max_retries: Number of retry attempts

    Returns:
        Raw compressed bytes or None if no data
    """
    url = f"{BASE_URL}/{symbol}/{year}/{month:02d}/{day:02d}/{hour:02d}h_ticks.bi5"

    for attempt in range(max_retries):
        try:
            response = session.get(url, timeout=30)
            if response.status_code == 200 and len(response.content) > 0:
                return response.content
            elif response.status_code == 404:
                # No data for this hour (weekend, holiday, etc.)
                return None
            else:
                time.sleep(0.5 * (attempt + 1))
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                print(f"Failed to download {url}: {e}")
    return None


def decompress_bi5(data: bytes) -> bytes:
    """Decompress LZMA-compressed .bi5 data."""
    try:
        return lzma.decompress(data)
    except lzma.LZMAError:
        # Try with raw LZMA format
        try:
            decompressor = lzma.LZMADecompressor(format=lzma.FORMAT_AUTO)
            return decompressor.decompress(data)
        except Exception:
            return b""


def parse_ticks(
    data: bytes,
    base_timestamp: datetime,
    point_value: float,
) -> list[dict]:
    """
    Parse binary tick data into list of tick dictionaries.

    Binary format (20 bytes per tick):
    - timestamp: uint32 (milliseconds since hour start)
    - ask: uint32 (price in pips)
    - bid: uint32 (price in pips)
    - ask_volume: float32
    - bid_volume: float32
    """
    ticks = []
    tick_size = 20

    for i in range(0, len(data), tick_size):
        if i + tick_size > len(data):
            break

        chunk = data[i:i + tick_size]
        try:
            ms_offset, ask_pips, bid_pips, ask_vol, bid_vol = struct.unpack(">IIIff", chunk)

            timestamp = base_timestamp + timedelta(milliseconds=ms_offset)
            ask = ask_pips * point_value
            bid = bid_pips * point_value
            mid = (ask + bid) / 2

            ticks.append({
                "timestamp": timestamp,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "bid_volume": bid_vol,
                "ask_volume": ask_vol,
                "volume": bid_vol + ask_vol,
            })
        except struct.error:
            continue

    return ticks


def ticks_to_ohlcv(ticks: list[dict], timeframe: str = "5min") -> pd.DataFrame:
    """
    Convert tick data to OHLCV candles.

    Args:
        ticks: List of tick dictionaries
        timeframe: Pandas-compatible timeframe string (e.g., '5min', '1H', '1D')

    Returns:
        DataFrame with OHLCV data
    """
    if not ticks:
        return pd.DataFrame()

    df = pd.DataFrame(ticks)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # Resample to OHLCV
    ohlcv = df["mid"].resample(timeframe).ohlc()
    ohlcv.columns = ["open", "high", "low", "close"]

    # Add volume
    ohlcv["volume"] = df["volume"].resample(timeframe).sum()

    # Add bid/ask spread info (average spread per candle)
    ohlcv["spread"] = (df["ask"] - df["bid"]).resample(timeframe).mean()

    # Drop rows with no data
    ohlcv.dropna(subset=["open"], inplace=True)

    return ohlcv


def download_day_data(
    symbol: str,
    date: datetime,
    session: requests.Session,
    point_value: float,
) -> list[dict]:
    """Download all tick data for a specific day."""
    all_ticks = []

    for hour in range(24):
        base_timestamp = date.replace(hour=hour, minute=0, second=0, microsecond=0)

        # Download compressed data
        compressed = download_hour_ticks(
            symbol=symbol,
            year=date.year,
            month=date.month - 1,  # Convert to 0-indexed
            day=date.day,
            hour=hour,
            session=session,
        )

        if compressed:
            # Decompress and parse
            raw_data = decompress_bi5(compressed)
            if raw_data:
                ticks = parse_ticks(raw_data, base_timestamp, point_value)
                all_ticks.extend(ticks)

    return all_ticks


def download_forex_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: str = "5min",
    output_dir: str = "data/forex",
    chunk_days: int = 30,
    max_workers: int = 4,
) -> pd.DataFrame:
    """
    Download forex data from Dukascopy and convert to OHLCV candles.

    Args:
        symbol: Currency pair (e.g., 'EURUSD')
        start_date: Start date
        end_date: End date
        timeframe: Candle timeframe (e.g., '5min', '15min', '1H')
        output_dir: Directory to save data chunks
        chunk_days: Days per chunk for incremental saving
        max_workers: Number of parallel download threads

    Returns:
        Combined DataFrame with all OHLCV data
    """
    symbol = symbol.upper()
    point_value = get_point_value(symbol)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate list of all dates to download
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)

    print(f"Downloading {symbol} data from {start_date.date()} to {end_date.date()}")
    print(f"Total days: {len(dates)}, Timeframe: {timeframe}")
    print(f"Point value: {point_value}")
    print("-" * 60)

    # Create session for connection pooling
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })

    all_chunks = []
    chunk_files = []

    # Process in chunks
    for chunk_start in range(0, len(dates), chunk_days):
        chunk_end = min(chunk_start + chunk_days, len(dates))
        chunk_dates = dates[chunk_start:chunk_end]

        chunk_start_date = chunk_dates[0]
        chunk_end_date = chunk_dates[-1]
        chunk_filename = f"{symbol}_{chunk_start_date.strftime('%Y%m%d')}_{chunk_end_date.strftime('%Y%m%d')}_{timeframe}.csv"
        chunk_filepath = output_path / chunk_filename

        # Check if chunk already exists
        if chunk_filepath.exists():
            print(f"Loading existing chunk: {chunk_filename}")
            chunk_df = pd.read_csv(chunk_filepath, index_col=0, parse_dates=True)
            all_chunks.append(chunk_df)
            chunk_files.append(chunk_filepath)
            continue

        print(f"Downloading chunk: {chunk_start_date.date()} to {chunk_end_date.date()}")

        chunk_ticks = []

        # Download days in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_day_data, symbol, date, session, point_value): date
                for date in chunk_dates
            }

            with tqdm(total=len(chunk_dates), desc="Days", unit="day") as pbar:
                for future in as_completed(futures):
                    date = futures[future]
                    try:
                        day_ticks = future.result()
                        chunk_ticks.extend(day_ticks)
                    except Exception as e:
                        print(f"Error downloading {date}: {e}")
                    pbar.update(1)

        # Convert to OHLCV
        if chunk_ticks:
            chunk_df = ticks_to_ohlcv(chunk_ticks, timeframe)

            if not chunk_df.empty:
                # Save chunk
                chunk_df.to_csv(chunk_filepath)
                print(f"Saved chunk: {chunk_filename} ({len(chunk_df)} candles)")
                all_chunks.append(chunk_df)
                chunk_files.append(chunk_filepath)

        # Small delay between chunks to be nice to the server
        time.sleep(1)

    session.close()

    # Merge all chunks
    if all_chunks:
        print("-" * 60)
        print("Merging all chunks...")

        combined_df = pd.concat(all_chunks)
        combined_df.sort_index(inplace=True)
        combined_df = combined_df[~combined_df.index.duplicated(keep="first")]

        # Save combined file
        combined_filename = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{timeframe}_combined.csv"
        combined_filepath = output_path / combined_filename
        combined_df.to_csv(combined_filepath)

        print(f"Saved combined file: {combined_filename}")
        print(f"Total candles: {len(combined_df)}")
        print(f"Date range: {combined_df.index.min()} to {combined_df.index.max()}")

        # Clean up chunk files (optional - comment out to keep them)
        # for chunk_file in chunk_files:
        #     chunk_file.unlink()
        # print("Cleaned up chunk files")

        return combined_df

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(
        description="Download historical forex data from Dukascopy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download 4 years of EUR/USD 5-minute data
    python download_forex_data.py --symbol EURUSD --start 2020-01-01 --end 2024-01-01 --timeframe 5min

    # Download 1 year of GBP/USD hourly data
    python download_forex_data.py --symbol GBPUSD --start 2023-01-01 --end 2024-01-01 --timeframe 1H

    # Download with custom output directory
    python download_forex_data.py --symbol EURUSD --start 2020-01-01 --end 2024-01-01 -o data/forex/eurusd
        """,
    )

    parser.add_argument(
        "--symbol", "-s",
        type=str,
        default="EURUSD",
        help="Currency pair symbol (default: EURUSD)",
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--timeframe", "-t",
        type=str,
        default="5min",
        help="Candle timeframe: 1min, 5min, 15min, 30min, 1H, 4H, 1D (default: 5min)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/forex",
        help="Output directory (default: data/forex)",
    )
    parser.add_argument(
        "--chunk-days",
        type=int,
        default=30,
        help="Days per chunk for incremental saving (default: 30)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )

    args = parser.parse_args()

    start_date = datetime.strptime(args.start, "%Y-%m-%d")
    end_date = datetime.strptime(args.end, "%Y-%m-%d")

    # Validate dates
    if start_date >= end_date:
        print("Error: Start date must be before end date")
        return

    # Download data
    df = download_forex_data(
        symbol=args.symbol,
        start_date=start_date,
        end_date=end_date,
        timeframe=args.timeframe,
        output_dir=args.output,
        chunk_days=args.chunk_days,
        max_workers=args.workers,
    )

    if not df.empty:
        print("\n" + "=" * 60)
        print("Download complete!")
        print("=" * 60)
        print(f"\nData summary:")
        print(df.describe())
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nLast 5 rows:")
        print(df.tail())
    else:
        print("No data downloaded.")


if __name__ == "__main__":
    main()
