"""Data Pipeline Service for continuous data updates.

This service manages the data pipeline that:
1. Fetches new 5-min OHLCV data from yfinance
2. Appends to the historical CSV
3. Resamples to 1H, 4H, Daily timeframes
4. Calculates technical indicators for each timeframe
5. Adds enhanced features (cross-TF, patterns, etc.)
6. Updates sentiment data (VIX, EPU from FRED)
7. Saves processed data to cache files for fast predictions

The pipeline runs periodically to keep data current without retraining models.
"""

import io
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple
import threading

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)


class PipelineService:
    """Service for managing the data update pipeline.

    This service continuously updates:
    - Raw 5-minute price data
    - Resampled timeframe data (1H, 4H, Daily)
    - Technical indicators for each timeframe
    - Enhanced features (patterns, cross-TF alignment)
    - Sentiment scores (VIX, EPU)

    All processed data is cached for fast predictions.
    """

    # FRED API for sentiment data
    FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    FRED_VIX_SERIES = "VIXCLS"
    FRED_EPU_SERIES = "USEPUINDXD"  # US Daily EPU

    def __init__(
        self,
        data_dir: str = "data",
        cache_dir: str = "data/cache",
        symbol: str = "EURUSD=X",
        historical_csv: Optional[str] = None,
    ):
        """Initialize pipeline service.

        Args:
            data_dir: Base data directory
            cache_dir: Directory for processed data cache
            symbol: yfinance symbol for live data
            historical_csv: Path to historical 5-min CSV
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.symbol = symbol

        # Historical data path
        self.historical_csv = Path(historical_csv) if historical_csv else (
            self.data_dir / "forex" / "EURUSD_20200101_20251231_5min_combined.csv"
        )

        # Sentiment data path
        self.sentiment_csv = self.data_dir / "sentiment" / "sentiment_scores_20200101_20251231_daily.csv"

        # Cache paths for processed data
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_5min = self.cache_dir / "eurusd_5min_updated.parquet"
        self.cache_1h = self.cache_dir / "eurusd_1h_features.parquet"
        self.cache_4h = self.cache_dir / "eurusd_4h_features.parquet"
        self.cache_daily = self.cache_dir / "eurusd_daily_features.parquet"
        self.cache_sentiment = self.cache_dir / "sentiment_updated.parquet"

        # Lock for thread safety
        self._lock = threading.Lock()
        self._initialized = False

        # Status tracking
        self.last_update: Optional[datetime] = None
        self.last_price_update: Optional[datetime] = None
        self.last_sentiment_update: Optional[datetime] = None

        # Lazy-loaded components
        self._technical_calculator = None
        self._feature_engine = None

    def initialize(self) -> bool:
        """Initialize the pipeline by loading historical data.

        Returns:
            True if successful
        """
        with self._lock:
            if self._initialized:
                return True

            try:
                logger.info("Initializing data pipeline...")

                # Check if historical CSV exists
                if not self.historical_csv.exists():
                    logger.warning(f"Historical CSV not found: {self.historical_csv}")
                    return False

                # Run initial pipeline
                self.run_full_pipeline()

                self._initialized = True
                logger.info("Data pipeline initialized successfully")
                return True

            except Exception as e:
                logger.error(f"Pipeline initialization failed: {e}")
                return False

    def run_full_pipeline(self) -> bool:
        """Run the complete data pipeline.

        This:
        1. Fetches and appends new 5-min data
        2. Resamples to all timeframes
        3. Calculates technical indicators
        4. Adds enhanced features
        5. Updates sentiment data
        6. Saves all to cache

        Returns:
            True if successful
        """
        try:
            logger.info("Running full data pipeline...")
            start_time = datetime.now()

            # Step 1: Update price data
            df_5min = self._update_price_data()
            if df_5min is None or df_5min.empty:
                logger.error("Failed to update price data")
                return False

            logger.info(f"  Price data: {len(df_5min):,} bars (5-min)")

            # Step 2: Update sentiment data
            df_sentiment = self._update_sentiment_data()
            if df_sentiment is not None:
                logger.info(f"  Sentiment data: {len(df_sentiment):,} days")

            # Step 3: Resample and calculate features for each timeframe
            for timeframe in ["1h", "4h", "D"]:
                self._process_timeframe(df_5min, timeframe, df_sentiment)

            self.last_update = datetime.now()
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Pipeline complete in {elapsed:.1f}s")

            return True

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _update_price_data(self) -> Optional[pd.DataFrame]:
        """Fetch new price data and append to historical CSV.

        Implements smart gap recovery:
        - Gap < 60 days: Use 5-minute data from yfinance (best resolution)
        - Gap 60-730 days: Use hourly data (2 year max)
        - Gap > 730 days: Use daily data (long history)

        The original CSV is updated (with backup) to persist changes.

        Returns:
            Updated 5-minute DataFrame
        """
        try:
            # Load historical data (prefer cache if exists and newer)
            logger.info("  Loading historical 5-min data...")

            df_hist = None
            source = "csv"

            # Check if cache exists and is newer than CSV
            if self.cache_5min.exists():
                cache_mtime = datetime.fromtimestamp(self.cache_5min.stat().st_mtime)
                csv_mtime = datetime.fromtimestamp(self.historical_csv.stat().st_mtime)

                if cache_mtime > csv_mtime:
                    df_hist = pd.read_parquet(self.cache_5min)
                    source = "cache"
                    logger.info(f"  Using cached data (newer than CSV)")

            if df_hist is None:
                df_hist = pd.read_csv(
                    self.historical_csv,
                    index_col=0,
                    parse_dates=True,
                )

            # Normalize column names
            df_hist.columns = df_hist.columns.str.lower()

            # Get last timestamp
            last_hist_time = df_hist.index.max()
            now = datetime.now()
            gap_days = (now - last_hist_time.to_pydatetime().replace(tzinfo=None)).days

            logger.info(f"  Historical data ends at: {last_hist_time}")
            logger.info(f"  Gap to recover: {gap_days} days")

            # Determine recovery strategy based on gap size
            df_new = self._fetch_gap_data(last_hist_time, gap_days)

            if df_new is None or df_new.empty:
                logger.warning("  No new data available from yfinance")
                self._save_price_data(df_hist, persist_csv=False)
                return df_hist

            # Filter to only new data (after last historical timestamp)
            df_new = df_new[df_new.index > last_hist_time]

            if df_new.empty:
                logger.info("  No new data to append (data is current)")
                self._save_price_data(df_hist, persist_csv=False)
                return df_hist

            logger.info(f"  Appending {len(df_new):,} new bars")

            # Combine historical + new
            df_combined = pd.concat([df_hist, df_new])
            df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
            df_combined = df_combined.sort_index()

            # Save to both cache AND update the CSV
            self._save_price_data(df_combined, persist_csv=True)

            self.last_price_update = datetime.now()
            return df_combined

        except Exception as e:
            logger.error(f"  Failed to update price data: {e}")
            import traceback
            traceback.print_exc()
            # Try to return cached data
            if self.cache_5min.exists():
                return pd.read_parquet(self.cache_5min)
            return None

    def _fetch_gap_data(
        self,
        last_time: pd.Timestamp,
        gap_days: int,
    ) -> Optional[pd.DataFrame]:
        """Fetch data to fill the gap using appropriate resolution.

        yfinance data availability:
        - 5-minute: Last 60 days
        - 15-minute: Last 60 days
        - 1-hour: Last 730 days (2 years)
        - Daily: Full history

        Args:
            last_time: Last timestamp in historical data
            gap_days: Number of days to recover

        Returns:
            DataFrame with new data (may be different resolution if gap > 60 days)
        """
        ticker = yf.Ticker(self.symbol)
        cols = ["open", "high", "low", "close", "volume"]

        if gap_days <= 0:
            logger.info("  Data is current, fetching last 7 days for safety")
            gap_days = 7

        # Strategy 1: Gap < 60 days - use 5-minute data (ideal)
        if gap_days < 60:
            logger.info(f"  Fetching 5-minute data for {gap_days + 2} days")
            df = ticker.history(period=f"{gap_days + 2}d", interval="5m")

            if not df.empty:
                df.columns = df.columns.str.lower()
                df.index = df.index.tz_localize(None) if df.index.tz else df.index
                df = df[[c for c in cols if c in df.columns]]
                logger.info(f"  Retrieved {len(df):,} 5-minute bars")
                return df

        # Strategy 2: Gap 60-730 days - use hourly data
        if gap_days < 730:
            logger.warning(f"  Gap > 60 days ({gap_days}d). Using HOURLY data (lower resolution)")
            logger.warning(f"  Note: 5-min granularity not available for data older than 60 days")

            # Fetch hourly data
            df_hourly = ticker.history(period=f"{min(gap_days + 7, 730)}d", interval="1h")

            if not df_hourly.empty:
                df_hourly.columns = df_hourly.columns.str.lower()
                df_hourly.index = df_hourly.index.tz_localize(None) if df_hourly.index.tz else df_hourly.index
                df_hourly = df_hourly[[c for c in cols if c in df_hourly.columns]]

                # Resample hourly to 5-minute by forward-filling
                # This creates synthetic 5-min bars (same OHLC for each 5-min within the hour)
                logger.info(f"  Retrieved {len(df_hourly):,} hourly bars, resampling to 5-min")
                df_5min = self._resample_to_5min(df_hourly)
                logger.info(f"  Created {len(df_5min):,} synthetic 5-minute bars")
                return df_5min

        # Strategy 3: Gap > 730 days - use daily data
        if gap_days >= 730:
            logger.warning(f"  Gap > 2 years ({gap_days}d). Using DAILY data (lowest resolution)")
            logger.warning(f"  Note: Consider running backfill script for proper 5-min data")

            df_daily = ticker.history(period="max", interval="1d")

            if not df_daily.empty:
                df_daily.columns = df_daily.columns.str.lower()
                df_daily.index = df_daily.index.tz_localize(None) if df_daily.index.tz else df_daily.index
                df_daily = df_daily[[c for c in cols if c in df_daily.columns]]

                # Filter to gap period
                df_daily = df_daily[df_daily.index > last_time]

                logger.info(f"  Retrieved {len(df_daily):,} daily bars")

                # Resample daily to 5-minute (very synthetic but better than gap)
                df_5min = self._resample_to_5min(df_daily, source_interval="daily")
                logger.info(f"  Created {len(df_5min):,} synthetic 5-minute bars from daily")
                return df_5min

        return None

    def _resample_to_5min(
        self,
        df: pd.DataFrame,
        source_interval: str = "hourly",
    ) -> pd.DataFrame:
        """Resample higher timeframe data to 5-minute bars.

        Creates synthetic 5-min bars by distributing OHLC values.
        This is not ideal but maintains data continuity.

        Args:
            df: DataFrame with higher timeframe data
            source_interval: "hourly" or "daily"

        Returns:
            DataFrame with 5-minute index
        """
        if df.empty:
            return df

        records = []

        # Number of 5-min bars per source bar
        bars_per_period = 12 if source_interval == "hourly" else 288  # 12 per hour, 288 per day

        for idx, row in df.iterrows():
            # Create 5-min timestamps within this period
            for i in range(bars_per_period):
                new_time = idx + timedelta(minutes=5 * i)

                # For first bar: use open, for last bar: use close
                # For middle bars: interpolate
                if i == 0:
                    bar_open = row["open"]
                    bar_close = row["open"] + (row["close"] - row["open"]) / bars_per_period
                elif i == bars_per_period - 1:
                    bar_open = row["close"] - (row["close"] - row["open"]) / bars_per_period
                    bar_close = row["close"]
                else:
                    progress = i / bars_per_period
                    bar_open = row["open"] + (row["close"] - row["open"]) * progress
                    bar_close = row["open"] + (row["close"] - row["open"]) * (progress + 1/bars_per_period)

                records.append({
                    "open": bar_open,
                    "high": max(bar_open, bar_close, row["high"] * (1 - abs(0.5 - i/bars_per_period) * 0.1)),
                    "low": min(bar_open, bar_close, row["low"] * (1 + abs(0.5 - i/bars_per_period) * 0.1)),
                    "close": bar_close,
                    "volume": row.get("volume", 0) / bars_per_period if "volume" in row else 0,
                    "timestamp": new_time,
                })

        df_5min = pd.DataFrame(records)
        df_5min = df_5min.set_index("timestamp")
        df_5min.index.name = None

        return df_5min

    def _save_price_data(self, df: pd.DataFrame, persist_csv: bool = True) -> None:
        """Save 5-min data to cache and optionally update the CSV.

        Args:
            df: DataFrame to save
            persist_csv: If True, also update the original CSV file
        """
        # Always save to cache (fast parquet)
        df.to_parquet(self.cache_5min)
        logger.info(f"  Saved 5-min cache: {len(df):,} bars")

        # Optionally persist to CSV (with backup)
        if persist_csv and len(df) > 0:
            try:
                # Create backup of original CSV
                backup_path = self.historical_csv.with_suffix('.csv.backup')
                if self.historical_csv.exists():
                    import shutil
                    shutil.copy2(self.historical_csv, backup_path)
                    logger.info(f"  Created backup: {backup_path}")

                # Update the CSV file
                df.to_csv(self.historical_csv)
                logger.info(f"  Updated CSV: {self.historical_csv}")

            except Exception as e:
                logger.error(f"  Failed to update CSV (cache is still valid): {e}")

    def _update_sentiment_data(self) -> Optional[pd.DataFrame]:
        """Update sentiment data from FRED.

        Fetches:
        - VIX (daily market fear)
        - US EPU (daily policy uncertainty)

        Returns:
            Updated sentiment DataFrame
        """
        try:
            # Load existing sentiment data
            if self.sentiment_csv.exists():
                df_sent = pd.read_csv(
                    self.sentiment_csv,
                    parse_dates=['Date'] if 'Date' in pd.read_csv(self.sentiment_csv, nrows=0).columns else [0],
                    index_col=0,
                )
                if not isinstance(df_sent.index, pd.DatetimeIndex):
                    df_sent.index = pd.to_datetime(df_sent.index)
                last_date = df_sent.index.max()
            else:
                df_sent = pd.DataFrame()
                last_date = datetime(2020, 1, 1)

            # Check if we need to update (update daily)
            if self.last_sentiment_update and (datetime.now() - self.last_sentiment_update).total_seconds() < 3600:
                logger.info("  Sentiment data recently updated, skipping")
                if self.cache_sentiment.exists():
                    return pd.read_parquet(self.cache_sentiment)
                return df_sent

            logger.info("  Fetching sentiment data from FRED...")

            # Fetch VIX
            start_date = (last_date - timedelta(days=7)).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")

            df_vix = self._fetch_fred_series(self.FRED_VIX_SERIES, start_date, end_date)
            df_epu = self._fetch_fred_series(self.FRED_EPU_SERIES, start_date, end_date)

            # Create/update sentiment dataframe
            if not df_sent.empty:
                df_result = df_sent.copy()
            else:
                # Create new sentiment dataframe
                date_range = pd.date_range(start="2020-01-01", end=end_date, freq="D")
                df_result = pd.DataFrame(index=date_range)
                df_result.index.name = "Date"

            # Update VIX values
            if df_vix is not None and not df_vix.empty:
                for idx, row in df_vix.iterrows():
                    if idx in df_result.index:
                        df_result.loc[idx, "VIX"] = row["Value"]
                    else:
                        df_result.loc[idx, "VIX"] = row["Value"]

            # Update EPU values
            if df_epu is not None and not df_epu.empty:
                for idx, row in df_epu.iterrows():
                    if idx in df_result.index:
                        df_result.loc[idx, "EPU_US"] = row["Value"]
                    else:
                        df_result.loc[idx, "EPU_US"] = row["Value"]

            # Sort and forward fill
            df_result = df_result.sort_index()
            df_result = df_result.ffill().bfill()

            # Calculate sentiment scores from raw values
            df_result = self._calculate_sentiment_scores(df_result)

            # Save to cache
            df_result.to_parquet(self.cache_sentiment)

            # Also update the original sentiment CSV
            try:
                # Create backup
                backup_path = self.sentiment_csv.with_suffix('.csv.backup')
                if self.sentiment_csv.exists():
                    import shutil
                    shutil.copy2(self.sentiment_csv, backup_path)

                # Save updated sentiment
                df_result.to_csv(self.sentiment_csv)
                logger.info(f"  Updated sentiment CSV: {self.sentiment_csv}")
            except Exception as e:
                logger.warning(f"  Failed to update sentiment CSV (cache is valid): {e}")

            self.last_sentiment_update = datetime.now()

            logger.info(f"  Sentiment updated: {len(df_result):,} days")
            return df_result

        except Exception as e:
            logger.error(f"  Failed to update sentiment: {e}")
            if self.cache_sentiment.exists():
                return pd.read_parquet(self.cache_sentiment)
            return None

    def _fetch_fred_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch a time series from FRED API."""
        try:
            url = f"{self.FRED_BASE_URL}?id={series_id}&cosd={start_date}&coed={end_date}"
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
            logger.warning(f"  Failed to fetch {series_id}: {e}")
            return None

    def _calculate_sentiment_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived sentiment scores from raw values."""
        # VIX sentiment: High VIX = Fear = Negative sentiment
        if "VIX" in df.columns:
            vix_min = df["VIX"].min()
            vix_max = df["VIX"].max()
            if vix_max > vix_min:
                vix_normalized = (df["VIX"] - vix_min) / (vix_max - vix_min)
                df["Sentiment_VIX"] = (0.5 - vix_normalized) * 0.4

        # EPU sentiment: High EPU = Uncertainty = Negative sentiment
        if "EPU_US" in df.columns:
            epu_min = df["EPU_US"].min()
            epu_max = df["EPU_US"].max()
            if epu_max > epu_min:
                epu_normalized = (df["EPU_US"] - epu_min) / (epu_max - epu_min)
                df["Sentiment_US"] = (0.5 - epu_normalized) * 0.4

        # Combined US sentiment
        if "Sentiment_US" in df.columns and "Sentiment_VIX" in df.columns:
            df["Sentiment_US_Combined"] = (df["Sentiment_US"] + df["Sentiment_VIX"]) / 2

        return df

    def _process_timeframe(
        self,
        df_5min: pd.DataFrame,
        timeframe: str,
        df_sentiment: Optional[pd.DataFrame] = None,
    ) -> None:
        """Process a single timeframe: resample, calculate indicators, save.

        Args:
            df_5min: 5-minute OHLCV data
            timeframe: Target timeframe ("1h", "4h", "D")
            df_sentiment: Sentiment data for daily model
        """
        logger.info(f"  Processing {timeframe} timeframe...")

        try:
            # Resample to target timeframe
            df_resampled = df_5min.resample(timeframe).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()

            logger.info(f"    Resampled: {len(df_resampled):,} bars")

            # Calculate technical indicators
            df_features = self._calculate_technical_indicators(df_resampled, timeframe)

            # Prepare higher timeframe data for cross-TF features
            higher_tf_data = self._prepare_higher_tf_data(df_5min, timeframe)

            # Add enhanced features
            df_features = self._add_enhanced_features(
                df_features,
                timeframe,
                higher_tf_data,
                include_sentiment=(timeframe == "D" and df_sentiment is not None),
            )

            # Add sentiment features for Daily model
            if timeframe == "D" and df_sentiment is not None:
                df_features = self._merge_sentiment(df_features, df_sentiment)

            logger.info(f"    Features: {len(df_features.columns)} columns")

            # Save to cache
            cache_path = {
                "1h": self.cache_1h,
                "4h": self.cache_4h,
                "D": self.cache_daily,
            }[timeframe]

            df_features.to_parquet(cache_path)
            logger.info(f"    Saved: {cache_path}")

        except Exception as e:
            logger.error(f"    Failed to process {timeframe}: {e}")
            import traceback
            traceback.print_exc()

    def _calculate_technical_indicators(
        self,
        df: pd.DataFrame,
        timeframe: str,
    ) -> pd.DataFrame:
        """Calculate technical indicators for a timeframe."""
        try:
            # Lazy load calculator
            if self._technical_calculator is None:
                from src.features.technical.calculator import TechnicalIndicatorCalculator
                self._technical_calculator = TechnicalIndicatorCalculator(
                    model_type="medium_term"
                )

            return self._technical_calculator.calculate(df)

        except Exception as e:
            logger.warning(f"    Technical indicators failed: {e}")
            # Return with basic derived features
            df["returns"] = df["close"].pct_change()
            df["range"] = df["high"] - df["low"]
            return df

    def _prepare_higher_tf_data(
        self,
        df_5min: pd.DataFrame,
        base_timeframe: str,
    ) -> Dict[str, pd.DataFrame]:
        """Prepare higher timeframe data for cross-TF features."""
        higher_tf_data = {}

        # Define higher timeframes for each base
        tf_hierarchy = {
            "1h": ["4h", "D"],
            "4h": ["D"],
            "D": [],
        }

        higher_tfs = tf_hierarchy.get(base_timeframe, [])

        for htf in higher_tfs:
            try:
                df_htf = df_5min.resample(htf).agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }).dropna()

                # Calculate basic indicators for HTF
                df_htf = self._calculate_technical_indicators(df_htf, htf)
                higher_tf_data[htf] = df_htf

            except Exception as e:
                logger.warning(f"    Failed to prepare {htf} data: {e}")

        return higher_tf_data

    def _add_enhanced_features(
        self,
        df: pd.DataFrame,
        timeframe: str,
        higher_tf_data: Dict[str, pd.DataFrame],
        include_sentiment: bool = False,
    ) -> pd.DataFrame:
        """Add enhanced features using EnhancedFeatureEngine."""
        try:
            # Lazy load feature engine
            if self._feature_engine is None:
                from src.models.multi_timeframe.enhanced_features import EnhancedFeatureEngine
                self._feature_engine = EnhancedFeatureEngine(
                    base_timeframe=timeframe,
                    include_time_features=True,
                    include_roc_features=True,
                    include_normalized_features=True,
                    include_pattern_features=True,
                    include_lag_features=True,
                    include_sentiment_features=False,  # We handle sentiment separately
                )

            # Update base timeframe
            self._feature_engine.base_timeframe = timeframe

            # Map timeframe names for higher_tf_data
            htf_mapped = {}
            for tf, tf_df in higher_tf_data.items():
                # Use uppercase for feature engine compatibility
                tf_upper = tf.upper() if tf != "D" else "D"
                htf_mapped[tf_upper] = tf_df

            return self._feature_engine.add_all_features(df, htf_mapped)

        except Exception as e:
            logger.warning(f"    Enhanced features failed: {e}")
            return df

    def _merge_sentiment(
        self,
        df: pd.DataFrame,
        df_sentiment: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge sentiment data into daily features."""
        try:
            # Extract date for merging
            df["_date"] = pd.to_datetime(df.index.date)

            # Prepare sentiment columns
            sent_cols = [c for c in df_sentiment.columns if "Sentiment" in c or "VIX" in c or "EPU" in c]

            if not sent_cols:
                df = df.drop("_date", axis=1)
                return df

            df_sent_merge = df_sentiment[sent_cols].copy()
            df_sent_merge["_date"] = pd.to_datetime(df_sent_merge.index.date)

            # Shift by 1 day to avoid look-ahead bias
            df_sent_merge = df_sent_merge.shift(1)

            # Merge
            df = df.reset_index()
            df = df.merge(df_sent_merge, on="_date", how="left")
            df = df.set_index(df.columns[0])
            df = df.drop("_date", axis=1, errors='ignore')

            # Forward fill sentiment
            for col in sent_cols:
                if col in df.columns:
                    df[col] = df[col].ffill().bfill()

            logger.info(f"    Added {len(sent_cols)} sentiment features")
            return df

        except Exception as e:
            logger.warning(f"    Sentiment merge failed: {e}")
            return df

    def get_processed_data(
        self,
        timeframe: str = "1h",
    ) -> Optional[pd.DataFrame]:
        """Get processed data for a timeframe from cache.

        Args:
            timeframe: "1h", "4h", or "D"

        Returns:
            Cached DataFrame with features, or None if not available
        """
        cache_path = {
            "1H": self.cache_1h,
            "1h": self.cache_1h,
            "4H": self.cache_4h,
            "4h": self.cache_4h,
            "D": self.cache_daily,
            "d": self.cache_daily,
        }.get(timeframe)

        if cache_path and cache_path.exists():
            return pd.read_parquet(cache_path)

        return None

    def get_latest_bar(self, timeframe: str = "1h") -> Optional[pd.Series]:
        """Get the latest bar with all features.

        Args:
            timeframe: "1h", "4h", or "D"

        Returns:
            Latest bar as Series, or None
        """
        df = self.get_processed_data(timeframe)
        if df is not None and not df.empty:
            return df.iloc[-1]
        return None

    def get_status(self) -> Dict:
        """Get detailed pipeline status including data quality info."""
        status = {
            "initialized": self._initialized,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "last_price_update": self.last_price_update.isoformat() if self.last_price_update else None,
            "last_sentiment_update": self.last_sentiment_update.isoformat() if self.last_sentiment_update else None,
            "cache_files": {
                "5min": self.cache_5min.exists(),
                "1h": self.cache_1h.exists(),
                "4h": self.cache_4h.exists(),
                "daily": self.cache_daily.exists(),
                "sentiment": self.cache_sentiment.exists(),
            },
        }

        # Add data quality information
        try:
            if self.cache_5min.exists():
                df = pd.read_parquet(self.cache_5min)
                data_start = df.index.min()
                data_end = df.index.max()
                gap_days = (datetime.now() - data_end.to_pydatetime().replace(tzinfo=None)).days

                status["data_info"] = {
                    "total_bars": len(df),
                    "date_range": {
                        "start": data_start.isoformat() if hasattr(data_start, 'isoformat') else str(data_start),
                        "end": data_end.isoformat() if hasattr(data_end, 'isoformat') else str(data_end),
                    },
                    "current_gap_days": gap_days,
                    "data_quality": "good" if gap_days < 1 else "stale" if gap_days < 7 else "outdated",
                }

            if self.historical_csv.exists():
                status["csv_info"] = {
                    "path": str(self.historical_csv),
                    "size_mb": round(self.historical_csv.stat().st_size / 1024 / 1024, 2),
                    "last_modified": datetime.fromtimestamp(
                        self.historical_csv.stat().st_mtime
                    ).isoformat(),
                }

                # Check if backup exists
                backup_path = self.historical_csv.with_suffix('.csv.backup')
                if backup_path.exists():
                    status["csv_info"]["backup_exists"] = True
                    status["csv_info"]["backup_time"] = datetime.fromtimestamp(
                        backup_path.stat().st_mtime
                    ).isoformat()

        except Exception as e:
            status["data_info_error"] = str(e)

        return status


# Singleton instance
pipeline_service = PipelineService()
