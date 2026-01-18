"""Base interface for data storage implementations.

This module defines the abstract base class for all storage backends,
ensuring consistent interface for storing and retrieving OHLCV market data.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import pandas as pd


class DataStorageError(Exception):
    """Base exception for storage errors."""

    pass


class StorageNotFoundError(DataStorageError):
    """Raised when requested data is not found in storage."""

    pass


class StorageIntegrityError(DataStorageError):
    """Raised when data integrity check fails."""

    pass


class BaseStorage(ABC):
    """Abstract base class for data storage implementations.

    All storage backends (Parquet, SQLite, etc.) must implement this interface
    to ensure consistent data access patterns across the application.

    Attributes:
        base_path: Root directory for storage operations.
        metadata: Dictionary for storing metadata about stored data.
    """

    def __init__(self, base_path: Union[str, Path]):
        """Initialize storage with base path.

        Args:
            base_path: Root directory for storage operations.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        *,
        overwrite: bool = False,
    ) -> int:
        """Save OHLCV data to storage.

        Args:
            df: DataFrame with OHLCV data. Must have DatetimeIndex and columns:
                [open, high, low, close, volume]
            symbol: Trading symbol (e.g., 'EURUSD', 'AAPL').
            timeframe: Data timeframe (e.g., '1H', '1D').
            overwrite: If True, replace existing data. If False, append.

        Returns:
            Number of rows saved.

        Raises:
            DataStorageError: If save operation fails.
            ValueError: If DataFrame format is invalid.
        """
        pass

    @abstractmethod
    def load(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from storage.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            start_date: Start date filter (inclusive).
            end_date: End date filter (inclusive).

        Returns:
            DataFrame with OHLCV data and DatetimeIndex.

        Raises:
            StorageNotFoundError: If data not found for symbol/timeframe.
            DataStorageError: If load operation fails.
        """
        pass

    @abstractmethod
    def append(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> int:
        """Append new data to existing storage.

        Handles deduplication if new data overlaps with existing data.

        Args:
            df: DataFrame with new OHLCV data.
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            Number of new rows appended.

        Raises:
            DataStorageError: If append operation fails.
        """
        pass

    @abstractmethod
    def delete(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """Delete data from storage.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            start_date: Delete data from this date (inclusive).
            end_date: Delete data until this date (inclusive).
                If both dates are None, deletes all data for symbol/timeframe.

        Returns:
            Number of rows deleted.
        """
        pass

    @abstractmethod
    def exists(self, symbol: str, timeframe: str) -> bool:
        """Check if data exists for symbol/timeframe.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            True if data exists.
        """
        pass

    @abstractmethod
    def get_date_range(
        self,
        symbol: str,
        timeframe: str,
    ) -> Optional[tuple[datetime, datetime]]:
        """Get the date range of stored data.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            Tuple of (start_date, end_date) or None if no data.
        """
        pass

    @abstractmethod
    def list_symbols(self, timeframe: Optional[str] = None) -> List[str]:
        """List all available symbols.

        Args:
            timeframe: Filter by timeframe (optional).

        Returns:
            List of symbol names.
        """
        pass

    @abstractmethod
    def list_timeframes(self, symbol: Optional[str] = None) -> List[str]:
        """List all available timeframes.

        Args:
            symbol: Filter by symbol (optional).

        Returns:
            List of timeframe strings.
        """
        pass

    @abstractmethod
    def get_row_count(self, symbol: str, timeframe: str) -> int:
        """Get the number of rows stored.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            Number of rows, or 0 if not found.
        """
        pass

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame has required OHLCV structure.

        Args:
            df: DataFrame to validate.

        Returns:
            True if valid.

        Raises:
            ValueError: If validation fails with details.
        """
        required_columns = {"open", "high", "low", "close", "volume"}
        df_columns = set(df.columns.str.lower())

        missing = required_columns - df_columns
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        if df.empty:
            raise ValueError("DataFrame is empty")

        # Validate OHLC relationships
        open_col = df["open"] if "open" in df.columns else df["Open"]
        high_col = df["high"] if "high" in df.columns else df["High"]
        low_col = df["low"] if "low" in df.columns else df["Low"]
        close_col = df["close"] if "close" in df.columns else df["Close"]

        if (high_col < open_col).any() or (high_col < close_col).any():
            raise ValueError("High must be >= Open and Close")

        if (low_col > open_col).any() or (low_col > close_col).any():
            raise ValueError("Low must be <= Open and Close")

        return True

    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage statistics and information.

        Returns:
            Dictionary with storage stats (size, symbols, etc.).
        """
        symbols = self.list_symbols()
        timeframes = self.list_timeframes()

        total_rows = 0
        for symbol in symbols:
            for tf in self.list_timeframes(symbol):
                total_rows += self.get_row_count(symbol, tf)

        return {
            "base_path": str(self.base_path),
            "total_symbols": len(symbols),
            "total_timeframes": len(timeframes),
            "total_rows": total_rows,
            "symbols": symbols,
            "timeframes": timeframes,
        }

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(base_path='{self.base_path}')"
