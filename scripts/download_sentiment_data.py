#!/usr/bin/env python3
"""
Economic Policy Uncertainty (EPU) Sentiment Data Downloader

Downloads EPU indices for countries relevant to forex/crypto trading:
- USA (daily from FRED)
- UK, Europe, Germany (monthly from FRED)
- Japan, Australia (monthly from policyuncertainty.com)

The data is combined into a daily dataset with forward-filled values for monthly data.

Usage:
    python download_sentiment_data.py --start 2020-01-01 --end 2025-12-31
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import io

import pandas as pd
import requests

# FRED API base URL (no API key needed for CSV download)
FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# FRED Series IDs
FRED_SERIES = {
    "US_daily": "USEPUINDXD",      # US Daily EPU
    "US_monthly": "USEPUINDXM",    # US Monthly EPU (backup)
    "UK": "UKEPUINDXM",            # UK Monthly EPU
    "Europe": "EUEPUINDXM",        # Europe Monthly EPU
    "Germany": "DEEPUINDXM",       # Germany Monthly EPU
    "China": "CHIEPUINDXM",        # China Monthly EPU
}

# PolicyUncertainty.com direct download URLs
POLICY_UNCERTAINTY_URLS = {
    "Japan": "https://www.policyuncertainty.com/media/Japan_Policy_Uncertainty_Data.xlsx",
    "Australia": "https://www.policyuncertainty.com/media/Australia_Policy_Uncertainty_Data.xlsx",
    "Global": "https://www.policyuncertainty.com/media/Global_Policy_Uncertainty_Data.xlsx",
}


def download_fred_series(series_id: str, start_date: str, end_date: str) -> pd.DataFrame | None:
    """
    Download a time series from FRED.

    Args:
        series_id: FRED series identifier
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with Date index and value column, or None if failed
    """
    url = f"{FRED_BASE_URL}?id={series_id}&cosd={start_date}&coed={end_date}"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(io.StringIO(response.text))
        df.columns = ["Date", "Value"]
        df["Date"] = pd.to_datetime(df["Date"])
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
        df = df.dropna()
        df = df.set_index("Date")

        return df

    except Exception as e:
        print(f"Error downloading {series_id}: {e}")
        return None


def download_policy_uncertainty_excel(country: str) -> pd.DataFrame | None:
    """
    Download EPU data from policyuncertainty.com Excel files.

    Args:
        country: Country name (Japan, Australia, Global)

    Returns:
        DataFrame with Date index and EPU value, or None if failed
    """
    if country not in POLICY_UNCERTAINTY_URLS:
        print(f"Unknown country: {country}")
        return None

    url = POLICY_UNCERTAINTY_URLS[country]

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        # Read Excel file
        df = pd.read_excel(io.BytesIO(response.content))

        # The Excel files have Year, Month, and EPU columns
        # Find the EPU column (usually named something like "Japan_EPU" or just the last numeric column)
        year_col = None
        month_col = None
        epu_col = None

        for col in df.columns:
            col_lower = str(col).lower()
            if "year" in col_lower:
                year_col = col
            elif "month" in col_lower:
                month_col = col
            elif "epu" in col_lower or "uncertainty" in col_lower:
                epu_col = col

        # If no EPU column found, try the last numeric column
        if epu_col is None:
            numeric_cols = df.select_dtypes(include=["number"]).columns
            if len(numeric_cols) > 0:
                epu_col = numeric_cols[-1]

        if year_col is None or month_col is None:
            # Try first two columns as year and month
            year_col = df.columns[0]
            month_col = df.columns[1]
            if epu_col is None:
                epu_col = df.columns[2] if len(df.columns) > 2 else df.columns[-1]

        # Create date from year and month
        df = df[[year_col, month_col, epu_col]].copy()
        df.columns = ["Year", "Month", "Value"]
        df = df.dropna()
        df["Year"] = df["Year"].astype(int)
        df["Month"] = df["Month"].astype(int)
        df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(Day=1))
        df = df[["Date", "Value"]].set_index("Date")
        df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

        return df

    except Exception as e:
        print(f"Error downloading {country} data: {e}")
        return None


def create_daily_sentiment_dataset(
    start_date: str,
    end_date: str,
    output_dir: str = "data/sentiment"
) -> str:
    """
    Create a combined daily sentiment dataset from multiple sources.

    Monthly data is forward-filled to create daily values.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory

    Returns:
        Path to output file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create date range for daily data
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    daily_df = pd.DataFrame(index=date_range)
    daily_df.index.name = "Date"

    print("Downloading Economic Policy Uncertainty data...")
    print("=" * 60)

    # Download FRED series
    for name, series_id in FRED_SERIES.items():
        print(f"Downloading {name} ({series_id})...")
        df = download_fred_series(series_id, start_date, end_date)

        if df is not None and len(df) > 0:
            col_name = f"EPU_{name.replace('_daily', '').replace('_monthly', '')}"

            if "daily" in name:
                # Daily data - just merge
                daily_df[col_name] = df["Value"]
            else:
                # Monthly data - resample to daily and forward-fill
                monthly_resampled = df["Value"].resample("D").ffill()
                daily_df[col_name] = monthly_resampled

            print(f"  ✓ {col_name}: {len(df)} records")
        else:
            print(f"  ✗ Failed to download {name}")

    # Download from policyuncertainty.com
    for country in ["Japan", "Australia", "Global"]:
        print(f"Downloading {country} from policyuncertainty.com...")
        df = download_policy_uncertainty_excel(country)

        if df is not None and len(df) > 0:
            col_name = f"EPU_{country}"
            # Monthly data - resample to daily and forward-fill
            monthly_resampled = df["Value"].resample("D").ffill()
            daily_df[col_name] = monthly_resampled
            print(f"  ✓ {col_name}: {len(df)} monthly records")
        else:
            print(f"  ✗ Failed to download {country}")

    # Forward fill and backward fill to handle any gaps
    daily_df = daily_df.ffill().bfill()

    # Filter to requested date range
    daily_df = daily_df.loc[start_date:end_date]

    # Calculate aggregate scores
    print("\nCalculating aggregate sentiment scores...")

    # Normalize each EPU to 0-100 scale based on historical range
    normalized_df = daily_df.copy()
    for col in daily_df.columns:
        min_val = daily_df[col].min()
        max_val = daily_df[col].max()
        if max_val > min_val:
            normalized_df[f"{col}_normalized"] = (
                (daily_df[col] - min_val) / (max_val - min_val) * 100
            )

    # Create sentiment multipliers (higher uncertainty = negative sentiment)
    # Convert EPU to sentiment: high EPU = low sentiment (bearish), low EPU = high sentiment (bullish)
    for col in daily_df.columns:
        if col.startswith("EPU_"):
            country = col.replace("EPU_", "")
            min_val = daily_df[col].min()
            max_val = daily_df[col].max()
            if max_val > min_val:
                # Invert: high EPU (100) -> -1, low EPU (0) -> +1
                # Scale to -0.2 to +0.2 for a 20% max adjustment
                normalized = (daily_df[col] - min_val) / (max_val - min_val)
                daily_df[f"Sentiment_{country}"] = (0.5 - normalized) * 0.4  # Range: -0.2 to +0.2

    # Currency pair specific sentiment (average of relevant countries)
    print("Creating currency pair sentiment scores...")

    # For forex pairs
    if "EPU_US" in daily_df.columns and "EPU_Europe" in daily_df.columns:
        daily_df["Sentiment_EURUSD"] = (
            daily_df.get("Sentiment_Europe", 0) - daily_df.get("Sentiment_US", 0)
        ) / 2

    if "EPU_US" in daily_df.columns and "EPU_UK" in daily_df.columns:
        daily_df["Sentiment_GBPUSD"] = (
            daily_df.get("Sentiment_UK", 0) - daily_df.get("Sentiment_US", 0)
        ) / 2

    if "EPU_US" in daily_df.columns and "EPU_Japan" in daily_df.columns:
        daily_df["Sentiment_USDJPY"] = (
            daily_df.get("Sentiment_US", 0) - daily_df.get("Sentiment_Japan", 0)
        ) / 2

    if "EPU_US" in daily_df.columns and "EPU_Australia" in daily_df.columns:
        daily_df["Sentiment_AUDUSD"] = (
            daily_df.get("Sentiment_Australia", 0) - daily_df.get("Sentiment_US", 0)
        ) / 2

    if "EPU_Europe" in daily_df.columns and "EPU_UK" in daily_df.columns:
        daily_df["Sentiment_EURGBP"] = (
            daily_df.get("Sentiment_Europe", 0) - daily_df.get("Sentiment_UK", 0)
        ) / 2

    # For crypto (use global/US sentiment as proxy)
    if "EPU_Global" in daily_df.columns:
        daily_df["Sentiment_Crypto"] = daily_df.get("Sentiment_Global", 0)
    elif "EPU_US" in daily_df.columns:
        daily_df["Sentiment_Crypto"] = daily_df.get("Sentiment_US", 0)

    # Save full dataset
    start_str = start_date.replace("-", "")
    end_str = end_date.replace("-", "")
    output_file = output_path / f"sentiment_epu_{start_str}_{end_str}_daily.csv"

    daily_df.to_csv(output_file)

    print("\n" + "=" * 60)
    print(f"Download complete!")
    print(f"Output file: {output_file}")
    print(f"Date range: {daily_df.index.min()} to {daily_df.index.max()}")
    print(f"Total days: {len(daily_df):,}")
    print(f"Columns: {len(daily_df.columns)}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

    print("\nColumns in dataset:")
    for col in sorted(daily_df.columns):
        print(f"  - {col}")

    # Also save a simplified version with just sentiment scores
    sentiment_cols = [col for col in daily_df.columns if col.startswith("Sentiment_")]
    if sentiment_cols:
        sentiment_file = output_path / f"sentiment_scores_{start_str}_{end_str}_daily.csv"
        daily_df[sentiment_cols].to_csv(sentiment_file)
        print(f"\nSimplified sentiment scores saved to: {sentiment_file}")

    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Download Economic Policy Uncertainty sentiment data"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2020-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default="2025-12-31",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sentiment",
        help="Output directory"
    )

    args = parser.parse_args()

    create_daily_sentiment_dataset(
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
