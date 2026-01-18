"""CSV file data source for local development and backtesting.

This module provides a data source implementation that reads OHLCV data
from CSV files, supporting various date formats and column naming conventions.
"""

import gzip
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import pandas as pd

from .base import BaseDataSource, DataSourceFactory


logger = logging.getLogger(__name__)


class CSVDataSourceError(Exception):
    """Exception raised for CSV data source errors."""

    pass


class CSVDataSource(BaseDataSource):
    """Data source for reading OHLCV data from CSV files.

    Supports multiple file organization patterns:
    - Single file per symbol/timeframe
    - Directory-based organization: base_path/symbol/timeframe.csv
    - Date-range files: base_path/SYMBOL_YYYYMMDD_YYYYMMDD_timeframe.csv

    Attributes:
        base_path: Root directory containing CSV files.
        date_column: Name of the date/timestamp column.
        date_format: strptime format for parsing dates (auto-detected if None).
    """

    # Standard column name mappings
    COLUMN_MAPPINGS = {
        "timestamp": "timestamp",
        "datetime": "timestamp",
        "date": "timestamp",
        "time": "timestamp",
        "open": "open",
        "o": "open",
        "high": "high",
        "h": "high",
        "low": "low",
        "l": "low",
        "close": "close",
        "c": "close",
        "adj close": "close",
        "volume": "volume",
        "vol": "volume",
        "v": "volume",
        "spread": "spread",
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        base_path: Optional[Union[str, Path]] = None,
        date_column: Optional[str] = None,
        date_format: Optional[str] = None,
    ):
        """Initialize CSV data source.

        Args:
            config: Optional configuration dictionary with keys:
                - base_path: Root directory for CSV files
                - date_column: Column name for timestamps
                - date_format: Date parsing format
            base_path: Root directory (overrides config).
            date_column: Timestamp column name (overrides config).
            date_format: Date format string (overrides config).
        """
        super().__init__(config)

        # Extract from config, then apply direct parameter overrides
        cfg = config or {}
        self.base_path = Path(base_path or cfg.get("base_path", "data"))
        self.date_column = date_column or cfg.get("date_column", "timestamp")
        self.date_format = date_format or cfg.get("date_format")

        # Cache for loaded files to avoid repeated disk reads
        self._file_cache: Dict[str, pd.DataFrame] = {}
        self._cache_enabled = cfg.get("cache_enabled", True)

        # Index of available files
        self._file_index: Dict[str, Dict[str, List[Path]]] = {}

    def connect(self) -> bool:
        """Initialize the data source by indexing available files.

        Returns:
            True if base_path exists and contains files.
        """
        if not self.base_path.exists():
            logger.warning(f"Base path does not exist: {self.base_path}")
            self._connected = False
            return False

        self._build_file_index()
        self._connected = True
        logger.info(f"CSV data source connected: {self.base_path}")
        return True

    def disconnect(self) -> None:
        """Clear cache and disconnect."""
        self._file_cache.clear()
        self._file_index.clear()
        self._connected = False
        logger.info("CSV data source disconnected")

    def _build_file_index(self) -> None:
        """Build index of available CSV files."""
        self._file_index.clear()

        # Find all CSV files (including gzipped)
        csv_files = list(self.base_path.rglob("*.csv"))
        csv_files.extend(self.base_path.rglob("*.csv.gz"))

        for file_path in csv_files:
            symbol, timeframe = self._parse_filename(file_path)
            if symbol and timeframe:
                if symbol not in self._file_index:
                    self._file_index[symbol] = {}
                if timeframe not in self._file_index[symbol]:
                    self._file_index[symbol][timeframe] = []
                self._file_index[symbol][timeframe].append(file_path)

        # Sort files by name (typically chronological for date-range files)
        for symbol in self._file_index:
            for timeframe in self._file_index[symbol]:
                self._file_index[symbol][timeframe].sort()

        logger.debug(f"Indexed {len(csv_files)} CSV files")

    def _parse_filename(self, file_path: Path) -> tuple[Optional[str], Optional[str]]:
        """Extract symbol and timeframe from filename.

        Supports patterns:
        - EURUSD_20200101_20200131_5min.csv
        - EURUSD_1H.csv
        - EURUSD.csv (assumes daily)
        - symbol/1H/data.csv

        Args:
            file_path: Path to CSV file.

        Returns:
            Tuple of (symbol, timeframe) or (None, None) if cannot parse.
        """
        filename = file_path.stem.replace(".csv", "")  # Handle .csv.gz
        parts = filename.split("_")

        # Pattern: SYMBOL_YYYYMMDD_YYYYMMDD_timeframe
        if len(parts) >= 4:
            symbol = parts[0].upper()
            timeframe = parts[-1].lower()
            return symbol, self._normalize_timeframe(timeframe)

        # Pattern: SYMBOL_timeframe
        if len(parts) == 2:
            symbol = parts[0].upper()
            timeframe = parts[1].lower()
            return symbol, self._normalize_timeframe(timeframe)

        # Pattern: SYMBOL (assume daily)
        if len(parts) == 1:
            symbol = parts[0].upper()
            return symbol, "1D"

        # Try directory structure: base/symbol/timeframe/...
        try:
            rel_path = file_path.relative_to(self.base_path)
            path_parts = rel_path.parts
            if len(path_parts) >= 2:
                symbol = path_parts[0].upper()
                timeframe = path_parts[1].upper()
                return symbol, self._normalize_timeframe(timeframe)
        except ValueError:
            pass

        return None, None

    def _normalize_timeframe(self, timeframe: str) -> str:
        """Normalize timeframe string to standard format.

        Args:
            timeframe: Raw timeframe string.

        Returns:
            Normalized timeframe (e.g., '1H', '5M', '1D').
        """
        tf = timeframe.lower().strip()

        # Map common variations
        mappings = {
            "1min": "1M",
            "5min": "5M",
            "15min": "15M",
            "30min": "30M",
            "60min": "1H",
            "1hour": "1H",
            "4hour": "4H",
            "1day": "1D",
            "daily": "1D",
            "1week": "1W",
            "weekly": "1W",
            "1month": "1MO",
            "monthly": "1MO",
            "m1": "1M",
            "m5": "5M",
            "m15": "15M",
            "m30": "30M",
            "h1": "1H",
            "h4": "4H",
            "d1": "1D",
            "w1": "1W",
        }

        return mappings.get(tf, timeframe.upper())

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from CSV files.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD').
            timeframe: Data timeframe (e.g., '1H', '5M').
            start_date: Start date for data.
            end_date: End date for data (defaults to now).

        Returns:
            DataFrame with columns [open, high, low, close, volume]
            and DatetimeIndex.

        Raises:
            CSVDataSourceError: If data cannot be loaded.
        """
        if not self._connected:
            self.connect()

        symbol = symbol.upper()
        timeframe = self._normalize_timeframe(timeframe)
        end_date = end_date or datetime.now()

        # Find relevant files
        files = self._get_files_for_range(symbol, timeframe, start_date, end_date)

        if not files:
            raise CSVDataSourceError(
                f"No CSV files found for {symbol}/{timeframe} "
                f"in date range {start_date} to {end_date}"
            )

        # Load and concatenate files
        dfs = []
        for file_path in files:
            df = self._load_csv_file(file_path)
            if df is not None and not df.empty:
                dfs.append(df)

        if not dfs:
            raise CSVDataSourceError(f"Failed to load data from files: {files}")

        # Combine all dataframes
        combined = pd.concat(dfs, axis=0)
        combined = combined.sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]

        # Filter to requested date range
        mask = (combined.index >= pd.Timestamp(start_date)) & (
            combined.index <= pd.Timestamp(end_date)
        )
        result = combined.loc[mask]

        logger.info(
            f"Loaded {len(result)} rows for {symbol}/{timeframe} "
            f"from {len(files)} files"
        )

        return result

    def _get_files_for_range(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Path]:
        """Get list of files that may contain data for date range.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            start_date: Range start.
            end_date: Range end.

        Returns:
            List of file paths.
        """
        if symbol not in self._file_index:
            # Try case-insensitive search
            for s in self._file_index:
                if s.upper() == symbol.upper():
                    symbol = s
                    break
            else:
                return []

        if timeframe not in self._file_index.get(symbol, {}):
            # Try normalized timeframe variations
            normalized = self._normalize_timeframe(timeframe)
            if normalized not in self._file_index.get(symbol, {}):
                return []
            timeframe = normalized

        return self._file_index[symbol][timeframe]

    def _load_csv_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load a single CSV file.

        Args:
            file_path: Path to CSV file.

        Returns:
            DataFrame or None if loading fails.
        """
        cache_key = str(file_path)

        # Check cache
        if self._cache_enabled and cache_key in self._file_cache:
            return self._file_cache[cache_key].copy()

        try:
            # Handle gzipped files
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "rt") as f:
                    df = pd.read_csv(f)
            else:
                df = pd.read_csv(file_path)

            # Normalize column names
            df = self._normalize_columns(df)

            # Parse dates and set index
            df = self._parse_dates(df)

            # Validate required columns exist
            required = {"open", "high", "low", "close", "volume"}
            if not required.issubset(set(df.columns)):
                missing = required - set(df.columns)
                logger.warning(f"Missing columns in {file_path}: {missing}")
                # Try to add volume if missing
                if "volume" not in df.columns:
                    df["volume"] = 0

            # Cache result
            if self._cache_enabled:
                self._file_cache[cache_key] = df.copy()

            return df

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame column names to standard format.

        Args:
            df: Raw DataFrame.

        Returns:
            DataFrame with normalized column names.
        """
        # Create mapping for this DataFrame's columns
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in self.COLUMN_MAPPINGS:
                rename_map[col] = self.COLUMN_MAPPINGS[col_lower]

        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date column and set as index.

        Args:
            df: DataFrame with date column.

        Returns:
            DataFrame with DatetimeIndex.
        """
        # Find the timestamp column
        date_col = None
        for col in ["timestamp", "datetime", "date", "time"]:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            # Try first column if it looks like dates
            first_col = df.columns[0]
            try:
                pd.to_datetime(df[first_col].iloc[0])
                date_col = first_col
            except (ValueError, TypeError):
                raise CSVDataSourceError("Cannot find date column in CSV")

        # Parse dates
        if self.date_format:
            df[date_col] = pd.to_datetime(df[date_col], format=self.date_format)
        else:
            df[date_col] = pd.to_datetime(df[date_col])

        # Set as index
        df = df.set_index(date_col)
        df.index.name = None

        return df

    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols.

        Returns:
            List of symbol names found in CSV files.
        """
        if not self._connected:
            self.connect()
        return list(self._file_index.keys())

    def get_available_timeframes(self, symbol: Optional[str] = None) -> List[str]:
        """Get list of available timeframes.

        Args:
            symbol: Filter by symbol (optional).

        Returns:
            List of timeframe strings.
        """
        if not self._connected:
            self.connect()

        if symbol:
            symbol = symbol.upper()
            return list(self._file_index.get(symbol, {}).keys())

        # Get all unique timeframes
        timeframes = set()
        for symbol_data in self._file_index.values():
            timeframes.update(symbol_data.keys())
        return sorted(timeframes)

    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """Get most recent price from CSV data.

        Note: For CSV files, this returns the last available price,
        not a live price.

        Args:
            symbol: Trading symbol.

        Returns:
            Dictionary with 'bid', 'ask', 'last' prices.
        """
        if not self._connected:
            self.connect()

        symbol = symbol.upper()

        # Get the smallest available timeframe
        timeframes = self.get_available_timeframes(symbol)
        if not timeframes:
            raise CSVDataSourceError(f"No data available for {symbol}")

        # Prefer smaller timeframes for more recent data
        tf_order = ["1M", "5M", "15M", "30M", "1H", "4H", "1D", "1W"]
        timeframe = "1D"
        for tf in tf_order:
            if tf in timeframes:
                timeframe = tf
                break

        # Get the last row
        files = self._file_index[symbol][timeframe]
        df = self._load_csv_file(files[-1])

        if df is None or df.empty:
            raise CSVDataSourceError(f"Cannot load price data for {symbol}")

        last_row = df.iloc[-1]
        last_price = float(last_row["close"])

        # Estimate bid/ask from spread if available
        spread = float(last_row.get("spread", 0.0001))
        half_spread = spread / 2

        return {
            "bid": last_price - half_spread,
            "ask": last_price + half_spread,
            "last": last_price,
            "timestamp": df.index[-1].isoformat(),
        }

    def get_date_range(
        self, symbol: str, timeframe: str
    ) -> Optional[tuple[datetime, datetime]]:
        """Get the date range available for a symbol/timeframe.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            Tuple of (start_date, end_date) or None.
        """
        if not self._connected:
            self.connect()

        symbol = symbol.upper()
        timeframe = self._normalize_timeframe(timeframe)

        files = self._file_index.get(symbol, {}).get(timeframe, [])
        if not files:
            return None

        # Load first and last files to get date range
        first_df = self._load_csv_file(files[0])
        last_df = self._load_csv_file(files[-1])

        if first_df is None or last_df is None:
            return None

        start_date = first_df.index.min().to_pydatetime()
        end_date = last_df.index.max().to_pydatetime()

        return (start_date, end_date)

    def clear_cache(self) -> None:
        """Clear the file cache."""
        self._file_cache.clear()
        logger.debug("CSV cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache info.
        """
        return {
            "enabled": self._cache_enabled,
            "entries": len(self._file_cache),
            "files_indexed": sum(
                len(files)
                for symbol_data in self._file_index.values()
                for files in symbol_data.values()
            ),
        }


# Register with factory
DataSourceFactory.register("csv", CSVDataSource)
