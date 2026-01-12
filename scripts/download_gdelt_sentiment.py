#!/usr/bin/env python3
"""
GDELT News Sentiment Data Downloader

Downloads 15-minute news sentiment data from GDELT via BigQuery for US and Europe.
Aggregates to hourly resolution for use with trading models.

GDELT provides global news monitoring with tone/sentiment scores updated every 15 minutes.
This is suitable for intraday trading models (1H, 4H) unlike daily VIX/EPU data.

Requirements:
    - Google Cloud account with BigQuery API enabled
    - Service account credentials JSON file
    - GOOGLE_APPLICATION_CREDENTIALS environment variable set

Usage:
    # Set credentials first
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

    # Download data
    python download_gdelt_sentiment.py --start 2020-01-01 --end 2025-12-31

    # Or specify credentials directly
    python download_gdelt_sentiment.py --credentials /path/to/credentials.json
"""

import argparse
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


def download_gdelt_sentiment(
    start_date: str,
    end_date: str,
    output_dir: str = "data/sentiment",
    credentials_path: str = None,
    aggregation: str = "hourly",
) -> str:
    """
    Download GDELT sentiment data for US and Europe.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory for CSV file
        credentials_path: Path to Google Cloud credentials JSON
        aggregation: Time aggregation ('15min', 'hourly', '4hourly', 'daily')

    Returns:
        Path to output CSV file
    """
    # Set credentials if provided
    if credentials_path:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

    # Import after setting credentials
    from google.cloud import bigquery

    print("=" * 60)
    print("GDELT NEWS SENTIMENT DOWNLOADER")
    print("=" * 60)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Aggregation: {aggregation}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Create BigQuery client
    print("\nConnecting to BigQuery...")
    client = bigquery.Client()
    print(f"Connected to project: {client.project}")

    # Determine time truncation based on aggregation
    # GDELT DATE format is YYYYMMDDHHMMSS (e.g., 20200503084500)
    # We parse it using PARSE_TIMESTAMP
    trunc_map = {
        "15min": "TIMESTAMP_TRUNC(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)), MINUTE)",
        "hourly": "TIMESTAMP_TRUNC(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)), HOUR)",
        "4hourly": "TIMESTAMP_TRUNC(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)), HOUR)",
        "daily": "TIMESTAMP_TRUNC(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING)), DAY)",
    }

    time_trunc = trunc_map.get(aggregation, trunc_map["hourly"])

    # GDELT GKG (Global Knowledge Graph) query for tone/sentiment
    # The V2tone field contains: Tone, Positive Score, Negative Score, Polarity,
    # Activity Reference Density, Self/Group Reference Density, Word Count
    query = f"""
    WITH parsed_data AS (
        SELECT
            {time_trunc} as timestamp,
            -- Extract tone (first value in V2Tone field)
            SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) as tone,
            -- Determine region from location
            CASE
                WHEN V2Locations LIKE '%United States%' THEN 'US'
                WHEN V2Locations LIKE '%Germany%' OR V2Locations LIKE '%France%'
                     OR V2Locations LIKE '%United Kingdom%'
                     OR V2Locations LIKE '%Italy%' OR V2Locations LIKE '%Spain%'
                     OR V2Locations LIKE '%Netherlands%' OR V2Locations LIKE '%Belgium%'
                     OR V2Locations LIKE '%Switzerland%' OR V2Locations LIKE '%Austria%'
                     OR V2Locations LIKE '%European Union%' THEN 'Europe'
                ELSE 'Other'
            END as region
        FROM `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE _PARTITIONTIME BETWEEN '{start_date}' AND '{end_date}'
            AND V2Tone IS NOT NULL
            AND V2Tone != ''
    )
    SELECT
        timestamp,
        -- US sentiment
        AVG(CASE WHEN region = 'US' THEN tone END) as tone_us,
        STDDEV(CASE WHEN region = 'US' THEN tone END) as tone_std_us,
        COUNT(CASE WHEN region = 'US' THEN 1 END) as article_count_us,
        -- Europe sentiment
        AVG(CASE WHEN region = 'Europe' THEN tone END) as tone_europe,
        STDDEV(CASE WHEN region = 'Europe' THEN tone END) as tone_std_europe,
        COUNT(CASE WHEN region = 'Europe' THEN 1 END) as article_count_europe,
        -- Global (for reference)
        AVG(tone) as tone_global,
        COUNT(*) as article_count_global
    FROM parsed_data
    WHERE region IN ('US', 'Europe')
    GROUP BY timestamp
    ORDER BY timestamp
    """

    print("\nExecuting BigQuery query...")
    print("This may take 1-3 minutes for 6 years of data...")

    # Execute query
    try:
        query_job = client.query(query)
        df = query_job.to_dataframe()

        bytes_processed = query_job.total_bytes_processed
        gb_processed = bytes_processed / (1024**3)
        print(f"Query processed: {gb_processed:.2f} GB")

    except Exception as e:
        print(f"Error executing query: {e}")
        raise

    print(f"Retrieved {len(df):,} rows")

    # Handle 4-hourly aggregation
    if aggregation == "4hourly":
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].dt.floor('4H')
        df = df.groupby('timestamp').agg({
            'tone_us': 'mean',
            'tone_std_us': 'mean',
            'article_count_us': 'sum',
            'tone_europe': 'mean',
            'tone_std_europe': 'mean',
            'article_count_europe': 'sum',
            'tone_global': 'mean',
            'article_count_global': 'sum',
        }).reset_index()
        print(f"Aggregated to 4-hourly: {len(df):,} rows")

    # Set timestamp as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df.sort_index()

    # Convert GDELT tone to sentiment score (-0.2 to +0.2 scale)
    # GDELT tone typically ranges from -10 to +10, with most values between -5 and +5
    def normalize_tone(tone_series, scale=0.2, clip_range=10):
        """Normalize GDELT tone to sentiment scale."""
        # Clip extreme values
        clipped = tone_series.clip(-clip_range, clip_range)
        # Normalize to -scale to +scale
        return (clipped / clip_range) * scale

    df['sentiment_us'] = normalize_tone(df['tone_us'])
    df['sentiment_europe'] = normalize_tone(df['tone_europe'])
    df['sentiment_global'] = normalize_tone(df['tone_global'])

    # Combined US-Europe sentiment (for EUR/USD)
    df['sentiment_eurusd'] = (df['sentiment_europe'] - df['sentiment_us']) / 2

    # Forward fill missing values (some hours may have no news)
    df = df.ffill().bfill()

    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    start_str = start_date.replace("-", "")
    end_str = end_date.replace("-", "")
    output_file = output_path / f"gdelt_sentiment_{start_str}_{end_str}_{aggregation}.csv"

    df.to_csv(output_file)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Output file: {output_file}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Total rows: {len(df):,}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")

    print("\nColumns:")
    for col in df.columns:
        non_null = df[col].notna().sum()
        print(f"  - {col}: {non_null:,} values")

    print("\nSentiment Statistics:")
    print(f"  US:     mean={df['sentiment_us'].mean():.4f}, std={df['sentiment_us'].std():.4f}")
    print(f"  Europe: mean={df['sentiment_europe'].mean():.4f}, std={df['sentiment_europe'].std():.4f}")
    print(f"  EURUSD: mean={df['sentiment_eurusd'].mean():.4f}, std={df['sentiment_eurusd'].std():.4f}")

    # Also save a simplified version
    simple_cols = ['sentiment_us', 'sentiment_europe', 'sentiment_global', 'sentiment_eurusd',
                   'article_count_us', 'article_count_europe']
    simple_file = output_path / f"gdelt_sentiment_{start_str}_{end_str}_{aggregation}_simple.csv"
    df[simple_cols].to_csv(simple_file)
    print(f"\nSimplified file: {simple_file}")

    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Download GDELT news sentiment data for US and Europe"
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
    parser.add_argument(
        "--credentials",
        type=str,
        default=None,
        help="Path to Google Cloud credentials JSON file"
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["15min", "hourly", "4hourly", "daily"],
        default="hourly",
        help="Time aggregation level"
    )

    args = parser.parse_args()

    # Default credentials path if not specified
    if args.credentials is None:
        default_creds = Path(__file__).parent.parent / "credentials" / "gcloud.json"
        if default_creds.exists():
            args.credentials = str(default_creds)

    download_gdelt_sentiment(
        start_date=args.start,
        end_date=args.end,
        output_dir=args.output,
        credentials_path=args.credentials,
        aggregation=args.aggregation,
    )


if __name__ == "__main__":
    main()
