"""Data service for accessing market data from pipeline cache.

This service provides:
- Fast data access via pipeline_service cache (no yfinance calls during API requests)
- Current EUR/USD price from latest cached bar
- VIX data from sentiment cache
- OHLCV data for predictions (hourly resolution)
- In-memory caching for even faster repeated access
- Fallback to yfinance for get_ohlcv_data() and get_market_info() only

The pipeline_service handles all external data fetching (yfinance, FRED)
and runs on a schedule, keeping cache files up to date.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from ..utils.validation import safe_iloc
from ..utils.logging import log_exception
from ...config import trading_config

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FOREX_DATA_DIR = DATA_DIR / "forex"


class DataService:
    """Service for accessing market data from pipeline cache.

    Reads pre-processed data from pipeline_service cache files (parquet)
    for fast API responses. No yfinance calls during request handling.

    The pipeline_service is responsible for:
    - Periodic data fetching from external sources
    - Combining historical + live data
    - Calculating features and resampling
    - Updating cache files

    Cache configuration is centralized via TradingConfig:
    - price_cache_max_size: Maximum price cache entries (default: 50)
    - ohlcv_cache_max_size: Maximum OHLCV cache entries (default: 20)
    """

    # Yahoo Finance symbols
    EURUSD_SYMBOL = "EURUSD=X"
    VIX_SYMBOL = "^VIX"

    # Historical data file
    HISTORICAL_DATA_FILE = FOREX_DATA_DIR / "EURUSD_20200101_20251231_5min_combined.csv"

    # Cache durations (increased since we read from pipeline cache)
    PRICE_CACHE_TTL = timedelta(minutes=15)  # Pipeline runs hourly
    VIX_CACHE_TTL = timedelta(hours=4)
    OHLCV_CACHE_TTL = timedelta(minutes=15)
    HISTORICAL_CACHE_TTL = timedelta(hours=24)  # Historical data rarely changes

    def __init__(self):
        self._lock = Lock()

        # Price cache
        self._price_cache: Dict[str, Tuple[float, datetime]] = {}

        # OHLCV cache - stores DataFrames
        self._ohlcv_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}

        # VIX cache
        self._vix_cache: Optional[Tuple[pd.Series, datetime]] = None

        # Historical data cache (loaded once)
        self._historical_data: Optional[pd.DataFrame] = None
        self._historical_loaded_at: Optional[datetime] = None

        # Combined data cache for predictions
        self._prediction_data_cache: Optional[Tuple[pd.DataFrame, datetime]] = None

        # Initialized flag
        self._initialized = False

    @property
    def is_loaded(self) -> bool:
        """Check if service is initialized and ready."""
        return self._initialized

    def initialize(self) -> None:
        """Initialize service (lightweight since we use pipeline cache)."""
        if self._initialized:
            return

        logger.info("Initializing DataService...")

        try:
            # Load historical data (kept for backward compatibility)
            self._load_historical_data()

            # Mark as initialized
            # Note: We don't pre-fetch data anymore since we read from pipeline cache
            self._initialized = True
            logger.info("DataService initialized successfully (using pipeline cache)")
        except Exception as e:
            logger.error(f"Failed to initialize DataService: {e}")
            # Don't raise - service can still work with pipeline cache
            self._initialized = True

    def _load_historical_data(self) -> None:
        """Load historical 5-minute data from CSV."""
        if not self.HISTORICAL_DATA_FILE.exists():
            logger.warning(f"Historical data file not found: {self.HISTORICAL_DATA_FILE}")
            return

        logger.info(f"Loading historical data from {self.HISTORICAL_DATA_FILE}")

        try:
            df = pd.read_csv(
                self.HISTORICAL_DATA_FILE,
                index_col=0,  # First column is the datetime index
                parse_dates=True,
            )

            # Standardize column names (already lowercase in CSV)
            cols_to_keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
            df = df[cols_to_keep].copy()

            # Ensure proper datetime index
            df.index = pd.to_datetime(df.index)

            self._historical_data = df
            self._historical_loaded_at = datetime.now()

            logger.info(
                f"Loaded {len(df):,} historical bars "
                f"({df.index[0]} to {df.index[-1]})"
            )

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            self._historical_data = None

    def _get_live_data(
        self,
        symbol: str = "EURUSD",
        start_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Get data from pipeline cache (no longer makes yfinance calls).

        Args:
            symbol: Trading symbol
            start_date: Start date for data (optional, for filtering)

        Returns:
            DataFrame with hourly OHLCV data from pipeline cache
        """
        try:
            # Import pipeline_service to access cache
            from .pipeline_service import pipeline_service

            # Read from pipeline's 1h cache (already has all data processed)
            df = pipeline_service.get_processed_data("1h")

            if df is None or df.empty:
                logger.warning("Pipeline cache not available for 1h data")
                return None

            # Keep only OHLCV columns
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            available_cols = [c for c in ohlcv_cols if c in df.columns]
            df = df[available_cols].copy()

            # Filter by start_date if provided
            if start_date:
                df = df[df.index >= start_date]

            if df.empty:
                logger.warning(f"No data available after filtering (start_date={start_date})")
                return None

            logger.debug(f"Retrieved {len(df):,} bars from pipeline cache")
            return df

        except Exception as e:
            logger.error(f"Error reading from pipeline cache: {e}")
            return None

    def get_data_for_prediction(
        self,
        symbol: str = "EURUSD",
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Get data from pipeline cache for model prediction.

        The pipeline service handles:
        1. Loading historical 5-minute data
        2. Fetching new data periodically
        3. Combining and resampling to hourly
        4. Caching processed data

        This method simply reads from the pipeline's cache.

        Args:
            symbol: Trading symbol
            use_cache: Whether to use in-memory cache

        Returns:
            DataFrame with hourly OHLCV data from pipeline cache
        """
        # Check in-memory cache first
        with self._lock:
            if use_cache and self._prediction_data_cache is not None:
                df, cached_at = self._prediction_data_cache
                if datetime.now() - cached_at < self.OHLCV_CACHE_TTL:
                    logger.debug("Returning cached prediction data")
                    return df.copy()

        try:
            # Import pipeline_service
            from .pipeline_service import pipeline_service

            # Read hourly data from pipeline cache
            df = pipeline_service.get_processed_data("1h")

            if df is None or df.empty:
                logger.warning("Pipeline cache not available, falling back to live data")
                return self._get_live_data_for_prediction(symbol)

            # Keep only OHLCV columns
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            available_cols = [c for c in ohlcv_cols if c in df.columns]
            df = df[available_cols].copy()

            logger.debug(
                f"Retrieved data from pipeline: {len(df):,} hourly bars "
                f"({df.index[0]} to {df.index[-1]})"
            )

            # Cache the result in memory
            with self._lock:
                self._prediction_data_cache = (df.copy(), datetime.now())

            return df

        except Exception as e:
            logger.error(f"Error reading from pipeline cache: {e}")
            return self._get_live_data_for_prediction(symbol)

    def _get_live_data_for_prediction(
        self,
        symbol: str = "EURUSD",
    ) -> Optional[pd.DataFrame]:
        """Fallback: get only live data for prediction."""
        df = self._get_live_data(symbol)

        if df is None or len(df) < 200:
            logger.warning("Insufficient data for prediction")
            return None

        # Forward-fill and clean
        df = df.ffill().dropna()

        return df

    def get_current_price(self, symbol: str = "EURUSD") -> Optional[float]:
        """Get current price from pipeline cache.

        Args:
            symbol: Trading symbol (default: EURUSD)

        Returns:
            Current price (latest close) or None if unavailable
        """
        cache_key = f"{symbol}_price"

        with self._lock:
            # Check in-memory cache
            if cache_key in self._price_cache:
                price, timestamp = self._price_cache[cache_key]
                if datetime.now() - timestamp < self.PRICE_CACHE_TTL:
                    return price

        try:
            # Import pipeline_service
            from .pipeline_service import pipeline_service

            # Get latest bar from pipeline cache (1h data)
            df = pipeline_service.get_processed_data("1h")

            if df is None or df.empty:
                logger.warning("Pipeline cache not available for current price")
                return None

            # Get the latest close price
            if "close" not in df.columns:
                logger.warning("No 'close' column in pipeline data")
                return None

            price = float(df["close"].iloc[-1])

            # Cache the result (with size limit from centralized config)
            max_price_cache_size = trading_config.cache.price_cache_max_size
            with self._lock:
                # Evict oldest entries if cache is full
                if len(self._price_cache) >= max_price_cache_size:
                    oldest_key = min(
                        self._price_cache.keys(),
                        key=lambda k: self._price_cache[k][1]
                    )
                    del self._price_cache[oldest_key]
                self._price_cache[cache_key] = (price, datetime.now())

            logger.debug(f"Current price from pipeline cache: {price}")
            return price

        except Exception as e:
            logger.error(f"Error fetching current price from pipeline cache: {e}")
            return None

    def get_ohlcv_data(
        self,
        symbol: str = "EURUSD",
        period: str = "7d",
        interval: str = "5m",
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Get OHLCV data for a symbol.

        Args:
            symbol: Trading symbol
            period: Data period (e.g., "7d", "30d")
            interval: Data interval (e.g., "5m", "1h", "1d")
            force_refresh: Force refresh cache

        Returns:
            DataFrame with OHLCV columns or None
        """
        yf_symbol = self.EURUSD_SYMBOL if symbol == "EURUSD" else f"{symbol}=X"
        cache_key = f"{yf_symbol}_{period}_{interval}"

        with self._lock:
            if not force_refresh and cache_key in self._ohlcv_cache:
                df, timestamp = self._ohlcv_cache[cache_key]
                if datetime.now() - timestamp < self.OHLCV_CACHE_TTL:
                    return df.copy()

        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No OHLCV data for {yf_symbol}")
                return None

            # Standardize column names
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })

            # Ensure datetime index is timezone-naive for compatibility
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Keep only OHLCV columns
            df = df[["open", "high", "low", "close", "volume"]].copy()

            # Cache result (with size limit from centralized config)
            max_ohlcv_cache_size = trading_config.cache.ohlcv_cache_max_size
            with self._lock:
                # Evict oldest entries if cache is full
                if len(self._ohlcv_cache) >= max_ohlcv_cache_size:
                    oldest_key = min(
                        self._ohlcv_cache.keys(),
                        key=lambda k: self._ohlcv_cache[k][1]
                    )
                    del self._ohlcv_cache[oldest_key]
                self._ohlcv_cache[cache_key] = (df.copy(), datetime.now())

            logger.debug(f"Fetched {len(df)} bars for {yf_symbol} ({interval})")
            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {yf_symbol}: {e}")
            return None

    def get_vix_data(
        self,
        period: str = "60d",
        force_refresh: bool = False,
    ) -> Optional[pd.Series]:
        """Get VIX data from pipeline sentiment cache.

        Args:
            period: Data period (ignored, returns all available from cache)
            force_refresh: Force refresh cache

        Returns:
            Series with VIX values or None
        """
        with self._lock:
            if not force_refresh and self._vix_cache is not None:
                vix, timestamp = self._vix_cache
                if datetime.now() - timestamp < self.VIX_CACHE_TTL:
                    return vix.copy()

        try:
            # Import pipeline_service
            from .pipeline_service import pipeline_service

            # Read sentiment data from pipeline cache
            df_sentiment = None
            if pipeline_service.cache_sentiment.exists():
                df_sentiment = pd.read_parquet(pipeline_service.cache_sentiment)

            if df_sentiment is None or df_sentiment.empty:
                logger.warning("Pipeline sentiment cache not available")
                return None

            # Extract VIX column
            if "VIX" not in df_sentiment.columns:
                logger.warning("No VIX column in sentiment data")
                return None

            vix = df_sentiment["VIX"].copy()
            vix.name = "vix"

            # Ensure timezone-naive
            if vix.index.tz is not None:
                vix.index = vix.index.tz_localize(None)

            with self._lock:
                self._vix_cache = (vix.copy(), datetime.now())

            logger.debug(f"Retrieved {len(vix)} VIX values from pipeline cache")
            return vix

        except Exception as e:
            logger.error(f"Error fetching VIX data from pipeline cache: {e}")
            return None

    def get_latest_vix(self) -> Optional[float]:
        """Get most recent VIX value from pipeline cache."""
        vix = self.get_vix_data()
        if vix is not None and len(vix) > 0:
            return float(vix.iloc[-1])
        return None

    def resample_data(
        self,
        df: pd.DataFrame,
        timeframe: str,
    ) -> pd.DataFrame:
        """Resample data to target timeframe.

        Args:
            df: OHLCV DataFrame
            timeframe: Target timeframe ("1H", "4H", "D")

        Returns:
            Resampled DataFrame
        """
        if timeframe == "5min":
            return df.copy()

        # Use lowercase 'h' for hours (pandas >= 2.0 deprecation)
        tf_map = {"1H": "1h", "4H": "4h", "D": "D"}
        resample_tf = tf_map.get(timeframe, timeframe)

        resampled = df.resample(resample_tf).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in df.columns else "first",
        }).dropna()

        return resampled

    def get_market_info(self, symbol: str = "EURUSD") -> Dict:
        """Get current market information.

        Returns:
            Dict with price, change, high, low, etc.
        """
        yf_symbol = self.EURUSD_SYMBOL if symbol == "EURUSD" else f"{symbol}=X"

        try:
            ticker = yf.Ticker(yf_symbol)

            # Get today's data
            today = ticker.history(period="1d", interval="1m")
            yesterday = ticker.history(period="2d", interval="1d")

            # Use safe_iloc for bounds-checked DataFrame access
            current_price = safe_iloc(today, -1, "Close")
            if current_price is not None:
                current_price = float(current_price)

            day_high = float(today["High"].max()) if not today.empty else None
            day_low = float(today["Low"].min()) if not today.empty else None

            # Safe access to previous close with bounds checking
            prev_close = safe_iloc(yesterday, -2, "Close")
            if prev_close is not None:
                prev_close = float(prev_close)

            change = None
            change_pct = None
            if current_price and prev_close:
                change = current_price - prev_close
                change_pct = (change / prev_close) * 100

            return {
                "symbol": symbol,
                "price": current_price,
                "change": change,
                "change_pct": change_pct,
                "day_high": day_high,
                "day_low": day_low,
                "prev_close": prev_close,
                "timestamp": datetime.now().isoformat(),
                "data_source": "yfinance",
                "delay_minutes": 15,  # Approximate delay
            }

        except Exception as e:
            logger.error(f"Error getting market info: {e}")
            return {
                "symbol": symbol,
                "price": None,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def get_recent_candles(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "1H",
        count: int = 24,
    ) -> Optional[pd.DataFrame]:
        """Get recent candles for charting.

        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe (1H, 4H, D)
            count: Number of candles

        Returns:
            DataFrame with OHLCV data
        """
        # First try to use pipeline cache (most reliable)
        try:
            from .pipeline_service import pipeline_service

            tf_map = {"1H": "1h", "4H": "4h", "D": "D", "1h": "1h", "4h": "4h"}
            cache_tf = tf_map.get(timeframe, "1h")

            df = pipeline_service.get_processed_data(cache_tf)
            if df is not None and not df.empty:
                # Get only OHLCV columns
                ohlcv_cols = ["open", "high", "low", "close", "volume"]
                available_cols = [c for c in ohlcv_cols if c in df.columns]
                result = df[available_cols].tail(count)
                if not result.empty:
                    logger.debug(f"Got {len(result)} candles from pipeline cache")
                    return result
        except Exception as e:
            logger.warning(f"Pipeline cache not available: {e}")

        # Fallback to yfinance
        interval_map = {
            "5min": "5m",
            "15min": "15m",
            "1H": "1h",
            "4H": "1h",  # Will resample
            "D": "1d",
        }

        yf_interval = interval_map.get(timeframe, "1h")

        # Calculate period needed
        if timeframe == "D":
            period = f"{count + 5}d"
        elif timeframe in ("1H", "4H"):
            period = f"{max(count * 4 + 24, 168)}h"
        else:
            period = "7d"

        # Fetch data
        df = self.get_ohlcv_data(symbol=symbol, period=period, interval=yf_interval)

        if df is None:
            return None

        # Resample if needed (for 4H)
        if timeframe == "4H":
            df = self.resample_data(df, "4H")

        # Return last N candles
        return df.tail(count)

    def get_historical_data_info(self) -> Dict:
        """Get information about loaded historical data."""
        if self._historical_data is None:
            return {
                "loaded": False,
                "file": str(self.HISTORICAL_DATA_FILE),
            }

        return {
            "loaded": True,
            "file": str(self.HISTORICAL_DATA_FILE),
            "rows": len(self._historical_data),
            "start": str(self._historical_data.index[0]),
            "end": str(self._historical_data.index[-1]),
            "loaded_at": self._historical_loaded_at.isoformat() if self._historical_loaded_at else None,
        }

    def clear_cache(self, release_historical: bool = False) -> None:
        """Clear all cached data.

        Args:
            release_historical: If True, also release the historical DataFrame
                               to free ~50-100MB of memory
        """
        import gc

        with self._lock:
            self._price_cache.clear()
            self._ohlcv_cache.clear()
            self._vix_cache = None
            self._prediction_data_cache = None

            if release_historical and self._historical_data is not None:
                self._historical_data = None
                self._historical_loaded_at = None
                logger.info("Released historical data from memory")

        # Force garbage collection
        gc.collect()
        logger.info("DataService cache cleared")

    def get_memory_usage(self) -> Dict:
        """Get approximate memory usage of cached data.

        Returns:
            Dict with cache sizes and memory estimates
        """
        import sys

        with self._lock:
            historical_mb = (
                sys.getsizeof(self._historical_data) / 1024 / 1024
                if self._historical_data is not None
                else 0
            )
            # Estimate DataFrame memory more accurately if available
            if self._historical_data is not None:
                try:
                    historical_mb = self._historical_data.memory_usage(deep=True).sum() / 1024 / 1024
                except Exception:
                    pass

            return {
                "price_cache_entries": len(self._price_cache),
                "ohlcv_cache_entries": len(self._ohlcv_cache),
                "vix_cached": self._vix_cache is not None,
                "prediction_data_cached": self._prediction_data_cache is not None,
                "historical_data_mb": round(historical_mb, 2),
                "historical_loaded": self._historical_data is not None,
            }


# Singleton instance
data_service = DataService()
