"""Data Pipeline orchestrator for fetching, processing, and storing market data.

This module provides the main orchestration layer that coordinates data sources,
processors, and storage backends to create a unified data access interface.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

import pandas as pd

from .sources.base import BaseDataSource, DataSourceFactory
from .sources.csv_source import CSVDataSource
from .processors.ohlcv import OHLCVProcessor
from .storage.base import BaseStorage, StorageNotFoundError
from .storage.parquet_store import ParquetStorage


logger = logging.getLogger(__name__)


class DataPipelineError(Exception):
    """Exception raised for pipeline errors."""

    pass


@dataclass
class PipelineConfig:
    """Configuration for data pipeline.

    Attributes:
        source_type: Data source type ('csv', 'alpaca', 'mt5', 'yahoo').
        source_config: Configuration for data source.
        storage_path: Path for Parquet storage.
        cache_enabled: Enable in-memory caching.
        auto_validate: Validate data on load.
        fill_gaps: Attempt to fill data gaps.
        gap_threshold_hours: Threshold for detecting gaps.
    """

    source_type: str = "csv"
    source_config: Dict[str, Any] = field(default_factory=dict)
    storage_path: Union[str, Path] = "data/storage"
    cache_enabled: bool = True
    auto_validate: bool = True
    fill_gaps: bool = False
    gap_threshold_hours: float = 24.0


@dataclass
class DataQualityReport:
    """Report on data quality after loading.

    Attributes:
        symbol: Trading symbol.
        timeframe: Data timeframe.
        total_rows: Total number of rows.
        date_range: Tuple of (start, end) dates.
        gaps_found: Number of gaps detected.
        gap_details: List of gap date ranges.
        null_counts: Dictionary of null counts per column.
        validation_passed: Whether all validations passed.
        issues: List of issues found.
    """

    symbol: str
    timeframe: str
    total_rows: int
    date_range: tuple[datetime, datetime]
    gaps_found: int = 0
    gap_details: List[tuple[datetime, datetime]] = field(default_factory=list)
    null_counts: Dict[str, int] = field(default_factory=dict)
    validation_passed: bool = True
    issues: List[str] = field(default_factory=list)


class DataPipeline:
    """Main data pipeline orchestrator.

    Coordinates data fetching from sources, processing through OHLCV processor,
    and storage to Parquet files.

    Example:
        ```python
        config = PipelineConfig(
            source_type="csv",
            source_config={"base_path": "data/forex"},
            storage_path="data/storage"
        )
        pipeline = DataPipeline(config)

        # Fetch and store data
        rows = await pipeline.fetch_and_store("EURUSD", "1H", start, end)

        # Get processed data
        df = pipeline.get_data("EURUSD", "1H", start, end)
        ```

    Attributes:
        config: Pipeline configuration.
        source: Data source instance.
        storage: Storage backend instance.
        processor: OHLCV processor instance.
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        source: Optional[BaseDataSource] = None,
        storage: Optional[BaseStorage] = None,
    ):
        """Initialize data pipeline.

        Args:
            config: Pipeline configuration.
            source: Custom data source (overrides config).
            storage: Custom storage backend (overrides config).
        """
        self.config = config or PipelineConfig()

        # Initialize source
        if source:
            self.source = source
        else:
            self.source = self._create_source()

        # Initialize storage
        if storage:
            self.storage = storage
        else:
            self.storage = ParquetStorage(self.config.storage_path)

        # Initialize processor
        self.processor = OHLCVProcessor()

        # In-memory cache
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

    def _create_source(self) -> BaseDataSource:
        """Create data source from configuration.

        Returns:
            Data source instance.
        """
        source_type = self.config.source_type.lower()

        if source_type == "csv":
            return CSVDataSource(config=self.config.source_config)
        else:
            # Use factory for other sources
            return DataSourceFactory.create(source_type, self.config.source_config)

    async def fetch_and_store(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        *,
        overwrite: bool = False,
    ) -> int:
        """Fetch data from source and store to storage.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            start_date: Start date for fetching.
            end_date: End date (defaults to now).
            overwrite: If True, replace existing data.

        Returns:
            Number of rows stored.

        Raises:
            DataPipelineError: If operation fails.
        """
        end_date = end_date or datetime.now()
        symbol = symbol.upper()
        timeframe = timeframe.upper()

        logger.info(f"Fetching {symbol}/{timeframe} from {start_date} to {end_date}")

        try:
            # Ensure source is connected
            if not self.source.is_connected:
                self.source.connect()

            # Fetch from source
            df = self.source.fetch_ohlcv(symbol, timeframe, start_date, end_date)

            if df.empty:
                logger.warning(f"No data returned for {symbol}/{timeframe}")
                return 0

            # Process data
            df = self.processor.clean(df)

            if self.config.auto_validate:
                try:
                    self.processor.validate(df)
                except ValueError as e:
                    logger.warning(f"Validation warning: {e}")

            # Store data
            rows = self.storage.save(df, symbol, timeframe, overwrite=overwrite)

            # Invalidate cache
            cache_key = f"{symbol}_{timeframe}"
            self._cache.pop(cache_key, None)

            logger.info(f"Stored {rows} rows for {symbol}/{timeframe}")
            return rows

        except Exception as e:
            raise DataPipelineError(f"Failed to fetch and store: {e}") from e

    def fetch_and_store_sync(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        *,
        overwrite: bool = False,
    ) -> int:
        """Synchronous version of fetch_and_store.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            start_date: Start date for fetching.
            end_date: End date (defaults to now).
            overwrite: If True, replace existing data.

        Returns:
            Number of rows stored.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.fetch_and_store(symbol, timeframe, start_date, end_date, overwrite=overwrite)
        )

    async def update_latest(
        self,
        symbol: str,
        timeframe: str,
        lookback_hours: int = 24,
    ) -> int:
        """Update storage with latest data from source.

        Fetches only new data since last stored timestamp.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            lookback_hours: Hours to look back for overlap.

        Returns:
            Number of new rows added.
        """
        symbol = symbol.upper()
        timeframe = timeframe.upper()

        try:
            # Get current date range in storage
            date_range = self.storage.get_date_range(symbol, timeframe)

            if date_range:
                start_date = date_range[1] - timedelta(hours=lookback_hours)
            else:
                # No existing data, fetch last 30 days
                start_date = datetime.now() - timedelta(days=30)

            end_date = datetime.now()

            logger.info(f"Updating {symbol}/{timeframe} from {start_date}")

            # Ensure source is connected
            if not self.source.is_connected:
                self.source.connect()

            # Fetch new data
            df = self.source.fetch_ohlcv(symbol, timeframe, start_date, end_date)

            if df.empty:
                logger.info(f"No new data for {symbol}/{timeframe}")
                return 0

            # Process and append
            df = self.processor.clean(df)
            rows = self.storage.append(df, symbol, timeframe)

            # Invalidate cache
            cache_key = f"{symbol}_{timeframe}"
            self._cache.pop(cache_key, None)

            logger.info(f"Added {rows} new rows for {symbol}/{timeframe}")
            return rows

        except Exception as e:
            raise DataPipelineError(f"Failed to update latest: {e}") from e

    def get_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        *,
        use_cache: bool = True,
        add_derived_features: bool = False,
    ) -> pd.DataFrame:
        """Get data from storage.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            start_date: Start date filter.
            end_date: End date filter.
            use_cache: Use in-memory cache.
            add_derived_features: Add derived price features.

        Returns:
            DataFrame with OHLCV data.

        Raises:
            DataPipelineError: If data not found or load fails.
        """
        symbol = symbol.upper()
        timeframe = timeframe.upper()
        cache_key = f"{symbol}_{timeframe}"

        try:
            # Check cache
            if use_cache and self.config.cache_enabled:
                if cache_key in self._cache:
                    df = self._cache[cache_key]
                    # Filter to date range
                    if start_date:
                        df = df[df.index >= pd.Timestamp(start_date)]
                    if end_date:
                        df = df[df.index <= pd.Timestamp(end_date)]
                    if add_derived_features:
                        df = self.processor.add_derived_features(df)
                    return df.copy()

            # Load from storage
            df = self.storage.load(symbol, timeframe, start_date, end_date)

            # Cache full data (without date filter)
            if use_cache and self.config.cache_enabled:
                full_df = self.storage.load(symbol, timeframe)
                self._cache[cache_key] = full_df
                self._cache_timestamps[cache_key] = datetime.now()

            # Add derived features if requested
            if add_derived_features:
                df = self.processor.add_derived_features(df)

            return df

        except StorageNotFoundError as e:
            raise DataPipelineError(f"Data not found: {e}") from e
        except Exception as e:
            raise DataPipelineError(f"Failed to get data: {e}") from e

    def get_data_quality_report(
        self,
        symbol: str,
        timeframe: str,
    ) -> DataQualityReport:
        """Generate data quality report for stored data.

        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.

        Returns:
            DataQualityReport with quality metrics.
        """
        symbol = symbol.upper()
        timeframe = timeframe.upper()

        try:
            df = self.get_data(symbol, timeframe, use_cache=False)
        except DataPipelineError:
            return DataQualityReport(
                symbol=symbol,
                timeframe=timeframe,
                total_rows=0,
                date_range=(datetime.min, datetime.min),
                validation_passed=False,
                issues=["Data not found"],
            )

        issues = []

        # Basic stats
        total_rows = len(df)
        date_range = (
            df.index.min().to_pydatetime(),
            df.index.max().to_pydatetime(),
        )

        # Null counts
        null_counts = df.isnull().sum().to_dict()
        for col, count in null_counts.items():
            if count > 0:
                issues.append(f"Column '{col}' has {count} null values")

        # Detect gaps
        gaps = self._detect_gaps(df, timeframe)
        if gaps:
            issues.append(f"Found {len(gaps)} gaps in data")

        # Validate OHLC relationships
        try:
            self.processor.validate(df)
        except ValueError as e:
            issues.append(str(e))

        return DataQualityReport(
            symbol=symbol,
            timeframe=timeframe,
            total_rows=total_rows,
            date_range=date_range,
            gaps_found=len(gaps),
            gap_details=gaps,
            null_counts=null_counts,
            validation_passed=len(issues) == 0,
            issues=issues,
        )

    def _detect_gaps(
        self,
        df: pd.DataFrame,
        timeframe: str,
    ) -> List[tuple[datetime, datetime]]:
        """Detect gaps in time series data.

        Args:
            df: DataFrame with DatetimeIndex.
            timeframe: Expected timeframe.

        Returns:
            List of gap (start, end) tuples.
        """
        # Map timeframe to expected gap threshold
        gap_thresholds = {
            "1M": timedelta(minutes=5),
            "5M": timedelta(minutes=15),
            "15M": timedelta(minutes=45),
            "30M": timedelta(hours=1.5),
            "1H": timedelta(hours=3),
            "4H": timedelta(hours=12),
            "1D": timedelta(days=3),
            "1W": timedelta(weeks=2),
        }

        threshold = gap_thresholds.get(
            timeframe.upper(),
            timedelta(hours=self.config.gap_threshold_hours)
        )

        gaps = []
        if len(df) < 2:
            return gaps

        time_diffs = df.index.to_series().diff()

        for i, diff in enumerate(time_diffs):
            if pd.notna(diff) and diff > threshold:
                gap_start = df.index[i - 1].to_pydatetime()
                gap_end = df.index[i].to_pydatetime()
                gaps.append((gap_start, gap_end))

        return gaps

    def list_available_data(self) -> Dict[str, List[str]]:
        """List all available data in storage.

        Returns:
            Dictionary mapping symbols to list of timeframes.
        """
        result = {}
        for symbol in self.storage.list_symbols():
            result[symbol] = self.storage.list_timeframes(symbol)
        return result

    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with storage info.
        """
        info = self.storage.get_storage_info()
        info["cache_entries"] = len(self._cache)
        info["source_type"] = self.config.source_type
        return info

    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.debug("Pipeline cache cleared")

    def resample_and_store(
        self,
        symbol: str,
        source_timeframe: str,
        target_timeframe: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> int:
        """Resample data to different timeframe and store.

        Args:
            symbol: Trading symbol.
            source_timeframe: Source timeframe to read.
            target_timeframe: Target timeframe to create.
            start_date: Start date filter.
            end_date: End date filter.

        Returns:
            Number of rows stored.

        Raises:
            DataPipelineError: If operation fails.
        """
        symbol = symbol.upper()

        try:
            # Load source data
            df = self.get_data(
                symbol, source_timeframe, start_date, end_date, use_cache=False
            )

            if df.empty:
                logger.warning(f"No source data for {symbol}/{source_timeframe}")
                return 0

            # Resample
            df_resampled = self.processor.resample(df, target_timeframe)

            # Store
            rows = self.storage.save(df_resampled, symbol, target_timeframe, overwrite=True)

            logger.info(
                f"Resampled {symbol} from {source_timeframe} to {target_timeframe}: {rows} rows"
            )
            return rows

        except Exception as e:
            raise DataPipelineError(f"Failed to resample: {e}") from e

    def delete_data(
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
            start_date: Delete from this date.
            end_date: Delete until this date.

        Returns:
            Number of rows deleted.
        """
        symbol = symbol.upper()
        timeframe = timeframe.upper()

        rows = self.storage.delete(symbol, timeframe, start_date, end_date)

        # Invalidate cache
        cache_key = f"{symbol}_{timeframe}"
        self._cache.pop(cache_key, None)

        return rows

    def __enter__(self):
        """Context manager entry."""
        self.source.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.source.disconnect()
        self.clear_cache()


# Convenience function for quick data access
def load_data(
    symbol: str,
    timeframe: str,
    source_path: Union[str, Path] = "data/forex",
    storage_path: Union[str, Path] = "data/storage",
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Convenience function to load data with default pipeline.

    Args:
        symbol: Trading symbol.
        timeframe: Data timeframe.
        source_path: Path to source CSV files.
        storage_path: Path for Parquet storage.
        start_date: Start date filter.
        end_date: End date filter.

    Returns:
        DataFrame with OHLCV data.
    """
    config = PipelineConfig(
        source_type="csv",
        source_config={"base_path": str(source_path)},
        storage_path=str(storage_path),
    )

    pipeline = DataPipeline(config)

    # Check if data exists in storage
    if not pipeline.storage.exists(symbol, timeframe):
        # Load from source first
        pipeline.source.connect()
        source_start = start_date or datetime(2000, 1, 1)
        source_end = end_date or datetime.now()

        try:
            df = pipeline.source.fetch_ohlcv(symbol, timeframe, source_start, source_end)
            if not df.empty:
                pipeline.storage.save(df, symbol, timeframe)
        except Exception as e:
            logger.warning(f"Could not fetch from source: {e}")

    return pipeline.get_data(symbol, timeframe, start_date, end_date)
