"""Parquet-based storage for efficient OHLCV data persistence.

This module provides a high-performance storage backend using Apache Parquet
format, optimized for time series financial data with efficient compression
and fast date-range queries.
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .base import (
    BaseStorage,
    DataStorageError,
    StorageNotFoundError,
    StorageIntegrityError,
)


logger = logging.getLogger(__name__)


class ParquetStorage(BaseStorage):
    """Parquet-based storage for OHLCV market data.

    Features:
    - Efficient columnar storage with compression
    - Fast date-range queries using row group filtering
    - Partitioning by symbol and timeframe
    - Metadata tracking for data integrity
    - Incremental append support

    File organization:
        base_path/
        ├── EURUSD/
        │   ├── 1H/
        │   │   ├── data.parquet
        │   │   └── metadata.json
        │   └── 1D/
        │       ├── data.parquet
        │       └── metadata.json
        └── GBPUSD/
            └── ...

    Attributes:
        compression: Parquet compression algorithm ('snappy', 'gzip', 'zstd').
        row_group_size: Number of rows per row group for efficient queries.
    """

    def __init__(
        self,
        base_path: Union[str, Path],
        compression: str = "snappy",
        row_group_size: int = 100000,
    ):
        """Initialize Parquet storage.

        Args:
            base_path: Root directory for storage.
            compression: Compression algorithm ('snappy', 'gzip', 'zstd', 'none').
            row_group_size: Rows per row group (affects query performance).
        """
        super().__init__(base_path)
        self.compression = compression if compression != "none" else None
        self.row_group_size = row_group_size

        # Define schema for OHLCV data
        self._schema = pa.schema([
            ("timestamp", pa.timestamp("us")),
            ("open", pa.float64()),
            ("high", pa.float64()),
            ("low", pa.float64()),
            ("close", pa.float64()),
            ("volume", pa.float64()),
        ])

    def _get_data_path(self, symbol: str, timeframe: str) -> Path:
        """Get path to data file for symbol/timeframe.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            Path to parquet file.
        """
        return self.base_path / symbol.upper() / timeframe.upper() / "data.parquet"

    def _get_metadata_path(self, symbol: str, timeframe: str) -> Path:
        """Get path to metadata file.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            Path to metadata JSON file.
        """
        return self.base_path / symbol.upper() / timeframe.upper() / "metadata.json"

    def _load_metadata(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Load metadata for symbol/timeframe.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            Metadata dictionary.
        """
        meta_path = self._get_metadata_path(symbol, timeframe)
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(
        self, symbol: str, timeframe: str, metadata: Dict[str, Any]
    ) -> None:
        """Save metadata for symbol/timeframe.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            metadata: Metadata dictionary.
        """
        meta_path = self._get_metadata_path(symbol, timeframe)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def _compute_checksum(self, df: pd.DataFrame) -> str:
        """Compute checksum for data integrity.

        Args:
            df: DataFrame to checksum.

        Returns:
            MD5 hex digest.
        """
        # Use a subset of data for faster checksum
        sample_size = min(1000, len(df))
        sample = df.iloc[::max(1, len(df) // sample_size)]
        data_str = sample.to_json()
        return hashlib.md5(data_str.encode()).hexdigest()

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for storage.

        Normalizes column names and ensures proper types.

        Args:
            df: Input DataFrame.

        Returns:
            Prepared DataFrame.
        """
        df = df.copy()

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex")

        # Reset index to have timestamp as column
        df = df.reset_index()

        # Normalize column names
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ["datetime", "date", "time", "index"]:
                rename_map[col] = "timestamp"
            elif col_lower in ["open", "high", "low", "close", "volume"]:
                rename_map[col] = col_lower

        df = df.rename(columns=rename_map)

        # Ensure required columns exist
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                if col == "volume":
                    df["volume"] = 0.0
                else:
                    raise ValueError(f"Missing required column: {col}")

        # Select only required columns in order
        df = df[required]

        # Ensure proper types
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        *,
        overwrite: bool = False,
    ) -> int:
        """Save OHLCV data to Parquet storage.

        Args:
            df: DataFrame with OHLCV data and DatetimeIndex.
            symbol: Trading symbol.
            timeframe: Data timeframe.
            overwrite: If True, replace existing data.

        Returns:
            Number of rows saved.

        Raises:
            DataStorageError: If save fails.
            ValueError: If DataFrame is invalid.
        """
        if df.empty:
            logger.warning(f"Empty DataFrame provided for {symbol}/{timeframe}")
            return 0

        # Validate
        self.validate_dataframe(df)

        # Prepare data
        df_prepared = self._prepare_dataframe(df)

        # Get paths
        data_path = self._get_data_path(symbol, timeframe)
        data_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if data_path.exists() and not overwrite:
                # Append to existing
                return self.append(df, symbol, timeframe)

            # Write new file
            table = pa.Table.from_pandas(df_prepared, preserve_index=False)
            pq.write_table(
                table,
                data_path,
                compression=self.compression,
                row_group_size=self.row_group_size,
            )

            # Save metadata
            metadata = {
                "symbol": symbol.upper(),
                "timeframe": timeframe.upper(),
                "rows": len(df_prepared),
                "start_date": df_prepared["timestamp"].min().isoformat(),
                "end_date": df_prepared["timestamp"].max().isoformat(),
                "checksum": self._compute_checksum(df_prepared),
                "compression": self.compression or "none",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
            }
            self._save_metadata(symbol, timeframe, metadata)

            logger.info(f"Saved {len(df_prepared)} rows to {data_path}")
            return len(df_prepared)

        except Exception as e:
            raise DataStorageError(f"Failed to save data: {e}") from e

    def load(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Load OHLCV data from Parquet storage.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            start_date: Start date filter (inclusive).
            end_date: End date filter (inclusive).

        Returns:
            DataFrame with OHLCV data and DatetimeIndex.

        Raises:
            StorageNotFoundError: If data not found.
            DataStorageError: If load fails.
        """
        data_path = self._get_data_path(symbol, timeframe)

        if not data_path.exists():
            raise StorageNotFoundError(
                f"No data found for {symbol}/{timeframe} at {data_path}"
            )

        try:
            # Build filter for date range
            filters = None
            if start_date or end_date:
                filter_parts = []
                if start_date:
                    filter_parts.append(
                        ("timestamp", ">=", pd.Timestamp(start_date))
                    )
                if end_date:
                    filter_parts.append(
                        ("timestamp", "<=", pd.Timestamp(end_date))
                    )
                filters = filter_parts

            # Read with filters (row group filtering)
            table = pq.read_table(data_path, filters=filters)
            df = table.to_pandas()

            # Set timestamp as index
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            df = df.sort_index()

            logger.debug(f"Loaded {len(df)} rows from {data_path}")
            return df

        except Exception as e:
            raise DataStorageError(f"Failed to load data: {e}") from e

    def append(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> int:
        """Append new data to existing storage.

        Handles deduplication if new data overlaps with existing.

        Args:
            df: DataFrame with new OHLCV data.
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            Number of new rows appended.

        Raises:
            DataStorageError: If append fails.
        """
        if df.empty:
            return 0

        data_path = self._get_data_path(symbol, timeframe)

        # Prepare new data
        df_new = self._prepare_dataframe(df)

        try:
            if not data_path.exists():
                # No existing data, just save
                return self.save(df, symbol, timeframe, overwrite=True)

            # Load existing data
            existing_df = self.load(symbol, timeframe)
            existing_df = existing_df.reset_index()
            existing_df.columns = ["timestamp", "open", "high", "low", "close", "volume"]

            # Combine and deduplicate
            combined = pd.concat([existing_df, df_new], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
            combined = combined.sort_values("timestamp")

            # Calculate new rows
            new_rows = len(combined) - len(existing_df)

            # Write combined data
            table = pa.Table.from_pandas(combined, preserve_index=False)
            pq.write_table(
                table,
                data_path,
                compression=self.compression,
                row_group_size=self.row_group_size,
            )

            # Update metadata
            metadata = self._load_metadata(symbol, timeframe)
            metadata.update({
                "rows": len(combined),
                "start_date": combined["timestamp"].min().isoformat(),
                "end_date": combined["timestamp"].max().isoformat(),
                "checksum": self._compute_checksum(combined),
                "updated_at": datetime.now().isoformat(),
            })
            self._save_metadata(symbol, timeframe, metadata)

            logger.info(f"Appended {new_rows} rows to {data_path}")
            return new_rows

        except Exception as e:
            raise DataStorageError(f"Failed to append data: {e}") from e

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
            start_date: Delete from this date (inclusive).
            end_date: Delete until this date (inclusive).

        Returns:
            Number of rows deleted.
        """
        data_path = self._get_data_path(symbol, timeframe)
        meta_path = self._get_metadata_path(symbol, timeframe)

        if not data_path.exists():
            return 0

        try:
            if start_date is None and end_date is None:
                # Delete all data
                rows = self.get_row_count(symbol, timeframe)
                data_path.unlink()
                if meta_path.exists():
                    meta_path.unlink()
                # Clean up empty directories
                data_path.parent.rmdir() if not any(data_path.parent.iterdir()) else None
                logger.info(f"Deleted all data for {symbol}/{timeframe}")
                return rows

            # Partial delete - load, filter, save
            df = self.load(symbol, timeframe)
            original_rows = len(df)

            # Build mask for rows to keep
            mask = pd.Series(True, index=df.index)
            if start_date:
                mask &= df.index < pd.Timestamp(start_date)
            if end_date:
                mask |= df.index > pd.Timestamp(end_date)

            df_remaining = df[mask]
            deleted_rows = original_rows - len(df_remaining)

            if df_remaining.empty:
                # All data deleted
                data_path.unlink()
                if meta_path.exists():
                    meta_path.unlink()
            else:
                # Save remaining data
                self.save(df_remaining, symbol, timeframe, overwrite=True)

            logger.info(f"Deleted {deleted_rows} rows from {symbol}/{timeframe}")
            return deleted_rows

        except Exception as e:
            raise DataStorageError(f"Failed to delete data: {e}") from e

    def exists(self, symbol: str, timeframe: str) -> bool:
        """Check if data exists for symbol/timeframe.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            True if data exists.
        """
        return self._get_data_path(symbol, timeframe).exists()

    def get_date_range(
        self, symbol: str, timeframe: str
    ) -> Optional[tuple[datetime, datetime]]:
        """Get the date range of stored data.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            Tuple of (start_date, end_date) or None.
        """
        metadata = self._load_metadata(symbol, timeframe)
        if metadata and "start_date" in metadata and "end_date" in metadata:
            return (
                datetime.fromisoformat(metadata["start_date"]),
                datetime.fromisoformat(metadata["end_date"]),
            )

        # Fall back to reading from file
        if not self.exists(symbol, timeframe):
            return None

        try:
            df = self.load(symbol, timeframe)
            if df.empty:
                return None
            return (
                df.index.min().to_pydatetime(),
                df.index.max().to_pydatetime(),
            )
        except Exception:
            return None

    def list_symbols(self, timeframe: Optional[str] = None) -> List[str]:
        """List all available symbols.

        Args:
            timeframe: Filter by timeframe (optional).

        Returns:
            List of symbol names.
        """
        symbols = []
        for symbol_dir in self.base_path.iterdir():
            if symbol_dir.is_dir() and not symbol_dir.name.startswith("."):
                if timeframe:
                    tf_dir = symbol_dir / timeframe.upper()
                    if tf_dir.exists() and (tf_dir / "data.parquet").exists():
                        symbols.append(symbol_dir.name)
                else:
                    # Check if any timeframe has data
                    for tf_dir in symbol_dir.iterdir():
                        if tf_dir.is_dir() and (tf_dir / "data.parquet").exists():
                            symbols.append(symbol_dir.name)
                            break
        return sorted(symbols)

    def list_timeframes(self, symbol: Optional[str] = None) -> List[str]:
        """List all available timeframes.

        Args:
            symbol: Filter by symbol (optional).

        Returns:
            List of timeframe strings.
        """
        timeframes = set()

        if symbol:
            symbol_dir = self.base_path / symbol.upper()
            if symbol_dir.exists():
                for tf_dir in symbol_dir.iterdir():
                    if tf_dir.is_dir() and (tf_dir / "data.parquet").exists():
                        timeframes.add(tf_dir.name)
        else:
            for symbol_dir in self.base_path.iterdir():
                if symbol_dir.is_dir() and not symbol_dir.name.startswith("."):
                    for tf_dir in symbol_dir.iterdir():
                        if tf_dir.is_dir() and (tf_dir / "data.parquet").exists():
                            timeframes.add(tf_dir.name)

        return sorted(timeframes)

    def get_row_count(self, symbol: str, timeframe: str) -> int:
        """Get the number of rows stored.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            Number of rows, or 0 if not found.
        """
        # Try metadata first (faster)
        metadata = self._load_metadata(symbol, timeframe)
        if metadata and "rows" in metadata:
            return metadata["rows"]

        # Read from file
        data_path = self._get_data_path(symbol, timeframe)
        if not data_path.exists():
            return 0

        try:
            parquet_file = pq.ParquetFile(data_path)
            return parquet_file.metadata.num_rows
        except Exception:
            return 0

    def verify_integrity(self, symbol: str, timeframe: str) -> bool:
        """Verify data integrity using stored checksum.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            True if data is valid.

        Raises:
            StorageIntegrityError: If integrity check fails.
        """
        metadata = self._load_metadata(symbol, timeframe)
        if not metadata or "checksum" not in metadata:
            logger.warning(f"No checksum found for {symbol}/{timeframe}")
            return True  # Cannot verify, assume OK

        try:
            df = self.load(symbol, timeframe)
            df_prepared = self._prepare_dataframe(df)
            current_checksum = self._compute_checksum(df_prepared)

            if current_checksum != metadata["checksum"]:
                raise StorageIntegrityError(
                    f"Checksum mismatch for {symbol}/{timeframe}: "
                    f"expected {metadata['checksum']}, got {current_checksum}"
                )

            return True

        except StorageIntegrityError:
            raise
        except Exception as e:
            raise StorageIntegrityError(f"Integrity check failed: {e}") from e

    def optimize(self, symbol: str, timeframe: str) -> None:
        """Optimize storage by rewriting with optimal settings.

        Useful after many appends to consolidate row groups.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
        """
        if not self.exists(symbol, timeframe):
            return

        try:
            df = self.load(symbol, timeframe)
            self.save(df, symbol, timeframe, overwrite=True)
            logger.info(f"Optimized storage for {symbol}/{timeframe}")
        except Exception as e:
            logger.error(f"Failed to optimize {symbol}/{timeframe}: {e}")

    def get_storage_size(self, symbol: str, timeframe: str) -> int:
        """Get storage size in bytes.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            File size in bytes, or 0 if not found.
        """
        data_path = self._get_data_path(symbol, timeframe)
        if data_path.exists():
            return data_path.stat().st_size
        return 0
