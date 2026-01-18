"""Unit tests for data pipeline."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

from src.data.pipeline import (
    DataPipeline,
    PipelineConfig,
    DataQualityReport,
    DataPipelineError,
    load_data,
)
from src.data.sources.csv_source import CSVDataSource
from src.data.storage.parquet_store import ParquetStorage


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame."""
    dates = pd.date_range("2024-01-01", periods=500, freq="1h")
    return pd.DataFrame(
        {
            "open": [1.1 + i * 0.0001 for i in range(500)],
            "high": [1.11 + i * 0.0001 for i in range(500)],
            "low": [1.09 + i * 0.0001 for i in range(500)],
            "close": [1.105 + i * 0.0001 for i in range(500)],
            "volume": [1000 + i for i in range(500)],
        },
        index=dates,
    )


@pytest.fixture
def temp_csv_dir(tmp_path, sample_ohlcv_df):
    """Create temp directory with CSV file."""
    csv_data = sample_ohlcv_df.reset_index()
    csv_data.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    csv_path = tmp_path / "source" / "EURUSD_1H.csv"
    csv_path.parent.mkdir(parents=True)
    csv_data.to_csv(csv_path, index=False)
    return tmp_path / "source"


@pytest.fixture
def pipeline_config(temp_csv_dir, tmp_path):
    """Create pipeline config."""
    return PipelineConfig(
        source_type="csv",
        source_config={"base_path": str(temp_csv_dir)},
        storage_path=str(tmp_path / "storage"),
        cache_enabled=True,
        auto_validate=True,
    )


@pytest.fixture
def pipeline(pipeline_config):
    """Create pipeline instance."""
    return DataPipeline(pipeline_config)


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.source_type == "csv"
        assert config.storage_path == "data/storage"
        assert config.cache_enabled is True
        assert config.auto_validate is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            source_type="alpaca",
            source_config={"api_key": "test"},
            storage_path="/custom/path",
            cache_enabled=False,
        )

        assert config.source_type == "alpaca"
        assert config.source_config["api_key"] == "test"
        assert config.storage_path == "/custom/path"
        assert config.cache_enabled is False


class TestDataPipeline:
    """Tests for DataPipeline class."""

    def test_init_default(self):
        """Test initialization with defaults."""
        pipeline = DataPipeline()

        assert pipeline.config is not None
        assert pipeline.source is not None
        assert pipeline.storage is not None
        assert pipeline.processor is not None

    def test_init_with_config(self, pipeline_config):
        """Test initialization with config."""
        pipeline = DataPipeline(pipeline_config)

        assert isinstance(pipeline.source, CSVDataSource)
        assert isinstance(pipeline.storage, ParquetStorage)

    def test_init_with_custom_source_and_storage(self, tmp_path):
        """Test initialization with custom components."""
        source = Mock()
        storage = Mock()

        pipeline = DataPipeline(source=source, storage=storage)

        assert pipeline.source is source
        assert pipeline.storage is storage

    @pytest.mark.asyncio
    async def test_fetch_and_store(self, pipeline, sample_ohlcv_df):
        """Test fetch and store operation."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        rows = await pipeline.fetch_and_store("EURUSD", "1H", start, end)

        assert rows > 0
        assert pipeline.storage.exists("EURUSD", "1H")

    def test_fetch_and_store_sync(self, pipeline, sample_ohlcv_df):
        """Test synchronous fetch and store."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        rows = pipeline.fetch_and_store_sync("EURUSD", "1H", start, end)

        assert rows > 0

    @pytest.mark.asyncio
    async def test_fetch_and_store_empty_result(self, pipeline):
        """Test fetch with no data returned returns 0 rows."""
        # Use a date range with no data
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 10)

        # When no data is found in the date range, returns 0 rows
        rows = await pipeline.fetch_and_store("EURUSD", "1H", start, end)
        assert rows == 0

    @pytest.mark.asyncio
    async def test_fetch_and_store_overwrite(self, pipeline, sample_ohlcv_df):
        """Test fetch with overwrite flag."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)

        # First store
        await pipeline.fetch_and_store("EURUSD", "1H", start, end)
        initial_count = pipeline.storage.get_row_count("EURUSD", "1H")

        # Store again with overwrite
        await pipeline.fetch_and_store("EURUSD", "1H", start, end, overwrite=True)
        final_count = pipeline.storage.get_row_count("EURUSD", "1H")

        # Should be same (overwritten, not appended)
        assert initial_count == final_count

    def test_get_data(self, pipeline, sample_ohlcv_df):
        """Test getting data from storage."""
        # First store data
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 10)
        pipeline.fetch_and_store_sync("EURUSD", "1H", start, end)

        # Get data
        df = pipeline.get_data("EURUSD", "1H")

        assert not df.empty
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "open" in df.columns

    def test_get_data_with_date_filter(self, pipeline, sample_ohlcv_df):
        """Test getting data with date filter."""
        pipeline.fetch_and_store_sync(
            "EURUSD", "1H", datetime(2024, 1, 1), datetime(2024, 1, 20)
        )

        start = datetime(2024, 1, 5)
        end = datetime(2024, 1, 10)
        df = pipeline.get_data("EURUSD", "1H", start, end)

        assert df.index.min() >= pd.Timestamp(start)
        assert df.index.max() <= pd.Timestamp(end)

    def test_get_data_with_derived_features(self, pipeline, sample_ohlcv_df):
        """Test getting data with derived features."""
        pipeline.fetch_and_store_sync(
            "EURUSD", "1H", datetime(2024, 1, 1), datetime(2024, 1, 10)
        )

        df = pipeline.get_data("EURUSD", "1H", add_derived_features=True)

        assert "returns" in df.columns
        assert "log_returns" in df.columns
        assert "range" in df.columns

    def test_get_data_uses_cache(self, pipeline, sample_ohlcv_df):
        """Test data caching."""
        pipeline.fetch_and_store_sync(
            "EURUSD", "1H", datetime(2024, 1, 1), datetime(2024, 1, 10)
        )

        # First call populates cache
        df1 = pipeline.get_data("EURUSD", "1H")
        assert len(pipeline._cache) > 0

        # Second call uses cache
        df2 = pipeline.get_data("EURUSD", "1H")
        assert len(df1) == len(df2)

    def test_get_data_not_found(self, pipeline):
        """Test error when data not found."""
        with pytest.raises(DataPipelineError, match="Data not found"):
            pipeline.get_data("UNKNOWN", "1H")

    def test_clear_cache(self, pipeline, sample_ohlcv_df):
        """Test cache clearing."""
        pipeline.fetch_and_store_sync(
            "EURUSD", "1H", datetime(2024, 1, 1), datetime(2024, 1, 10)
        )
        pipeline.get_data("EURUSD", "1H")

        assert len(pipeline._cache) > 0
        pipeline.clear_cache()
        assert len(pipeline._cache) == 0

    def test_list_available_data(self, pipeline, sample_ohlcv_df):
        """Test listing available data."""
        pipeline.fetch_and_store_sync(
            "EURUSD", "1H", datetime(2024, 1, 1), datetime(2024, 1, 10)
        )

        available = pipeline.list_available_data()

        assert "EURUSD" in available
        assert "1H" in available["EURUSD"]

    def test_get_storage_info(self, pipeline, sample_ohlcv_df):
        """Test storage info retrieval."""
        pipeline.fetch_and_store_sync(
            "EURUSD", "1H", datetime(2024, 1, 1), datetime(2024, 1, 10)
        )

        info = pipeline.get_storage_info()

        assert "total_symbols" in info
        assert "total_rows" in info
        assert info["source_type"] == "csv"

    def test_resample_and_store(self, pipeline, sample_ohlcv_df):
        """Test resampling to different timeframe."""
        # Store 1H data
        pipeline.fetch_and_store_sync(
            "EURUSD", "1H", datetime(2024, 1, 1), datetime(2024, 1, 10)
        )

        # Resample to 4H
        rows = pipeline.resample_and_store("EURUSD", "1H", "4H")

        assert rows > 0
        assert pipeline.storage.exists("EURUSD", "4H")

        # 4H should have fewer rows than 1H
        df_1h = pipeline.get_data("EURUSD", "1H")
        df_4h = pipeline.get_data("EURUSD", "4H")
        assert len(df_4h) < len(df_1h)

    def test_delete_data(self, pipeline, sample_ohlcv_df):
        """Test data deletion."""
        pipeline.fetch_and_store_sync(
            "EURUSD", "1H", datetime(2024, 1, 1), datetime(2024, 1, 10)
        )

        deleted = pipeline.delete_data("EURUSD", "1H")

        assert deleted > 0
        assert not pipeline.storage.exists("EURUSD", "1H")

    def test_context_manager(self, pipeline_config):
        """Test context manager protocol."""
        with DataPipeline(pipeline_config) as pipeline:
            assert pipeline.source.is_connected

        assert not pipeline.source.is_connected


class TestDataQualityReport:
    """Tests for data quality reporting."""

    def test_quality_report_generation(self, pipeline, sample_ohlcv_df):
        """Test quality report generation."""
        pipeline.fetch_and_store_sync(
            "EURUSD", "1H", datetime(2024, 1, 1), datetime(2024, 1, 10)
        )

        report = pipeline.get_data_quality_report("EURUSD", "1H")

        assert report.symbol == "EURUSD"
        assert report.timeframe == "1H"
        assert report.total_rows > 0
        assert report.validation_passed is True

    def test_quality_report_not_found(self, pipeline):
        """Test quality report for non-existent data."""
        report = pipeline.get_data_quality_report("UNKNOWN", "1H")

        assert report.total_rows == 0
        assert report.validation_passed is False
        assert "Data not found" in report.issues


class TestGapDetection:
    """Tests for gap detection."""

    def test_detect_gaps_in_data(self, pipeline):
        """Test gap detection in time series."""
        # Create data with a gap
        dates1 = pd.date_range("2024-01-01", periods=100, freq="1h")
        dates2 = pd.date_range("2024-01-10", periods=100, freq="1h")  # Gap of ~5 days

        df = pd.DataFrame(
            {
                "open": [1.1] * 200,
                "high": [1.11] * 200,
                "low": [1.09] * 200,
                "close": [1.105] * 200,
                "volume": [1000] * 200,
            },
            index=dates1.append(dates2),
        )

        gaps = pipeline._detect_gaps(df, "1H")

        assert len(gaps) > 0


class TestLoadDataFunction:
    """Tests for convenience load_data function."""

    def test_load_data_creates_pipeline(self, temp_csv_dir, tmp_path):
        """Test load_data convenience function."""
        df = load_data(
            "EURUSD",
            "1H",
            source_path=temp_csv_dir,
            storage_path=tmp_path / "storage",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 10),
        )

        assert not df.empty
        assert isinstance(df.index, pd.DatetimeIndex)


class TestUpdateLatest:
    """Tests for incremental update functionality."""

    @pytest.mark.asyncio
    async def test_update_latest_new_data(self, pipeline, sample_ohlcv_df):
        """Test update latest when no existing data."""
        # Should fetch last 30 days by default
        rows = await pipeline.update_latest("EURUSD", "1H")

        # May return 0 if no recent data in CSV
        assert isinstance(rows, int)

    @pytest.mark.asyncio
    async def test_update_latest_with_existing(self, pipeline, sample_ohlcv_df):
        """Test update latest with existing data."""
        # First store some data
        await pipeline.fetch_and_store(
            "EURUSD", "1H", datetime(2024, 1, 1), datetime(2024, 1, 5)
        )

        # Update latest (will look back 24h from stored end date)
        rows = await pipeline.update_latest("EURUSD", "1H", lookback_hours=48)

        assert isinstance(rows, int)
