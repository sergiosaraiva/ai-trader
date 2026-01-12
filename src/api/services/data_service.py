"""Data service for fetching market data via yfinance + historical CSV.

This service provides:
- Hybrid data: Historical 5-min CSV + live yfinance data
- Real-time EUR/USD price data (with ~15-20 min delay)
- VIX data for sentiment features
- Data caching to avoid excessive API calls
- Resampling to different timeframes
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FOREX_DATA_DIR = DATA_DIR / "forex"


class DataService:
    """Service for fetching and caching market data.

    Combines historical CSV data with live yfinance data for complete
    coverage needed by the MTF Ensemble model.
    """

    # Yahoo Finance symbols
    EURUSD_SYMBOL = "EURUSD=X"
    VIX_SYMBOL = "^VIX"

    # Historical data file
    HISTORICAL_DATA_FILE = FOREX_DATA_DIR / "EURUSD_20200101_20251231_5min_combined.csv"

    # Cache durations
    PRICE_CACHE_TTL = timedelta(minutes=5)
    VIX_CACHE_TTL = timedelta(hours=1)
    OHLCV_CACHE_TTL = timedelta(minutes=5)
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

    def initialize(self) -> None:
        """Initialize service by loading historical data and fetching live data."""
        if self._initialized:
            return

        logger.info("Initializing DataService...")

        try:
            # Load historical data first
            self._load_historical_data()

            # Pre-fetch live data
            self.get_current_price()
            self.get_vix_data()

            self._initialized = True
            logger.info("DataService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DataService: {e}")
            raise

    def _load_historical_data(self) -> None:
        """Load historical 5-minute data from CSV."""
        if not self.HISTORICAL_DATA_FILE.exists():
            logger.warning(f"Historical data file not found: {self.HISTORICAL_DATA_FILE}")
            return

        logger.info(f"Loading historical data from {self.HISTORICAL_DATA_FILE}")

        try:
            df = pd.read_csv(
                self.HISTORICAL_DATA_FILE,
                parse_dates=["timestamp"],
                index_col="timestamp",
            )

            # Standardize column names (already lowercase in CSV)
            df = df[["open", "high", "low", "close", "volume"]].copy()

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
        """Fetch live data from yfinance.

        Args:
            symbol: Trading symbol
            start_date: Start date for data (default: 7 days ago)

        Returns:
            DataFrame with 5-min or hourly OHLCV data
        """
        yf_symbol = self.EURUSD_SYMBOL if symbol == "EURUSD" else f"{symbol}=X"

        try:
            ticker = yf.Ticker(yf_symbol)

            # For forex, yfinance 5-min data can be unreliable
            # Try hourly first, then resample if needed
            if start_date:
                # Fetch from start_date to now
                df = ticker.history(start=start_date, interval="1h")
            else:
                # Default: last 60 days of hourly data
                df = ticker.history(period="60d", interval="1h")

            if df.empty:
                logger.warning(f"No live data available for {yf_symbol}")
                return None

            # Standardize column names
            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })

            # Ensure datetime index is timezone-naive
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Keep only OHLCV columns
            df = df[["open", "high", "low", "close", "volume"]].copy()

            return df

        except Exception as e:
            logger.error(f"Error fetching live data: {e}")
            return None

    def get_data_for_prediction(
        self,
        symbol: str = "EURUSD",
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Get combined historical + live data for model prediction.

        This method:
        1. Loads historical 5-minute data from CSV
        2. Fetches recent data from yfinance
        3. Combines them, using yfinance to fill the gap after CSV ends
        4. Returns data ready for the MTF Ensemble model

        Args:
            symbol: Trading symbol
            use_cache: Whether to use cached combined data

        Returns:
            DataFrame with 5-minute OHLCV data (historical + live)
        """
        # Check cache (5-minute TTL for combined data)
        with self._lock:
            if use_cache and self._prediction_data_cache is not None:
                df, cached_at = self._prediction_data_cache
                if datetime.now() - cached_at < timedelta(minutes=5):
                    logger.debug("Returning cached prediction data")
                    return df.copy()

        # Ensure historical data is loaded
        if self._historical_data is None:
            self._load_historical_data()

        if self._historical_data is None:
            logger.warning("No historical data available, falling back to live only")
            return self._get_live_data_for_prediction(symbol)

        try:
            # Get the end date of historical data
            hist_end = self._historical_data.index[-1]
            logger.debug(f"Historical data ends at: {hist_end}")

            # Fetch live data from yfinance
            # Start from a day before hist_end to ensure overlap
            live_start = hist_end - timedelta(days=1)
            live_df = self._get_live_data(symbol, start_date=live_start)

            if live_df is None or live_df.empty:
                logger.warning("No live data available, using historical only")
                # Return last portion of historical data
                return self._historical_data.tail(50000).copy()

            # The live data is hourly, we need to handle this properly
            # Option 1: Resample historical to hourly and combine
            # Option 2: Use historical as-is and append hourly data

            # We'll use Option 1 for consistency: resample historical to hourly
            # This ensures feature calculations work correctly
            hist_hourly = self._historical_data.resample("1h").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()

            # Find overlap point
            overlap_start = live_df.index[0]
            hist_before_overlap = hist_hourly[hist_hourly.index < overlap_start]

            # Combine: historical (before overlap) + live
            combined = pd.concat([hist_before_overlap, live_df])

            # Remove any duplicates (prefer live data)
            combined = combined[~combined.index.duplicated(keep="last")]

            # Sort by index
            combined = combined.sort_index()

            # Forward-fill any gaps
            combined = combined.ffill()

            # Drop any NaN rows
            combined = combined.dropna()

            logger.info(
                f"Combined data: {len(combined):,} hourly bars "
                f"({combined.index[0]} to {combined.index[-1]})"
            )

            # Cache the result
            with self._lock:
                self._prediction_data_cache = (combined.copy(), datetime.now())

            return combined

        except Exception as e:
            logger.error(f"Error combining data: {e}")
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
        """Get current price for a symbol.

        Args:
            symbol: Trading symbol (default: EURUSD)

        Returns:
            Current price or None if unavailable
        """
        yf_symbol = self.EURUSD_SYMBOL if symbol == "EURUSD" else f"{symbol}=X"

        with self._lock:
            # Check cache
            if yf_symbol in self._price_cache:
                price, timestamp = self._price_cache[yf_symbol]
                if datetime.now() - timestamp < self.PRICE_CACHE_TTL:
                    return price

        try:
            ticker = yf.Ticker(yf_symbol)
            # Get most recent data
            data = ticker.history(period="1d", interval="1m")

            if data.empty:
                logger.warning(f"No price data available for {yf_symbol}")
                return None

            price = float(data["Close"].iloc[-1])

            with self._lock:
                self._price_cache[yf_symbol] = (price, datetime.now())

            return price

        except Exception as e:
            logger.error(f"Error fetching price for {yf_symbol}: {e}")
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

            with self._lock:
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
        """Get VIX data for sentiment features.

        Args:
            period: Data period
            force_refresh: Force refresh cache

        Returns:
            Series with VIX closing values or None
        """
        with self._lock:
            if not force_refresh and self._vix_cache is not None:
                vix, timestamp = self._vix_cache
                if datetime.now() - timestamp < self.VIX_CACHE_TTL:
                    return vix.copy()

        try:
            ticker = yf.Ticker(self.VIX_SYMBOL)
            df = ticker.history(period=period, interval="1d")

            if df.empty:
                logger.warning("No VIX data available")
                return None

            # Get closing values
            vix = df["Close"].copy()
            vix.name = "vix"

            # Ensure timezone-naive
            if vix.index.tz is not None:
                vix.index = vix.index.tz_localize(None)

            with self._lock:
                self._vix_cache = (vix.copy(), datetime.now())

            logger.debug(f"Fetched {len(vix)} VIX values")
            return vix

        except Exception as e:
            logger.error(f"Error fetching VIX data: {e}")
            return None

    def get_latest_vix(self) -> Optional[float]:
        """Get most recent VIX value."""
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

            current_price = float(today["Close"].iloc[-1]) if not today.empty else None
            day_high = float(today["High"].max()) if not today.empty else None
            day_low = float(today["Low"].min()) if not today.empty else None

            prev_close = None
            if len(yesterday) >= 2:
                prev_close = float(yesterday["Close"].iloc[-2])

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

    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._price_cache.clear()
            self._ohlcv_cache.clear()
            self._vix_cache = None
            self._prediction_data_cache = None
        logger.info("DataService cache cleared")


# Singleton instance
data_service = DataService()
