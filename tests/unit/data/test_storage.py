"""Unit tests for data storage classes."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.data.storage.base import (
    BaseStorage,
    DataStorageError,
    StorageNotFoundError,
    StorageIntegrityError,
)
from src.data.storage.parquet_store import ParquetStorage


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame."""
    dates = pd.date_range("2024-01-01", periods=1000, freq="1h")
    return pd.DataFrame(
        {
            "open": [1.1 + i * 0.0001 for i in range(1000)],
            "high": [1.11 + i * 0.0001 for i in range(1000)],
            "low": [1.09 + i * 0.0001 for i in range(1000)],
            "close": [1.105 + i * 0.0001 for i in range(1000)],
            "volume": [1000 + i for i in range(1000)],
        },
        index=dates,
    )


@pytest.fixture
def parquet_storage(tmp_path):
    """Create ParquetStorage instance."""
    return ParquetStorage(tmp_path)


class TestBaseStorage:
    """Tests for BaseStorage interface."""

    def test_validate_dataframe_valid(self, sample_ohlcv_df):
        """Test validation passes for valid DataFrame."""
        storage = ParquetStorage("temp")
        assert storage.validate_dataframe(sample_ohlcv_df) is True

    def test_validate_dataframe_missing_columns(self):
        """Test validation fails for missing columns."""
        storage = ParquetStorage("temp")
        df = pd.DataFrame(
            {"open": [1.0], "high": [1.1]},
            index=pd.DatetimeIndex([datetime.now()]),
        )

        with pytest.raises(ValueError, match="Missing required columns"):
            storage.validate_dataframe(df)

    def test_validate_dataframe_wrong_index(self):
        """Test validation fails for non-datetime index."""
        storage = ParquetStorage("temp")
        df = pd.DataFrame(
            {
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.05],
                "volume": [1000],
            }
        )

        with pytest.raises(ValueError, match="DatetimeIndex"):
            storage.validate_dataframe(df)

    def test_validate_dataframe_empty(self):
        """Test validation fails for empty DataFrame."""
        storage = ParquetStorage("temp")
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([]),
        )

        with pytest.raises(ValueError, match="empty"):
            storage.validate_dataframe(df)

    def test_validate_dataframe_invalid_high(self):
        """Test validation fails when high < open/close."""
        storage = ParquetStorage("temp")
        df = pd.DataFrame(
            {
                "open": [1.1],
                "high": [1.0],  # Invalid: high < open
                "low": [0.9],
                "close": [1.05],
                "volume": [1000],
            },
            index=pd.DatetimeIndex([datetime.now()]),
        )

        with pytest.raises(ValueError, match="High must be"):
            storage.validate_dataframe(df)


class TestParquetStorage:
    """Tests for ParquetStorage class."""

    def test_init(self, tmp_path):
        """Test initialization."""
        storage = ParquetStorage(tmp_path, compression="gzip")

        assert storage.base_path == tmp_path
        assert storage.compression == "gzip"
        assert tmp_path.exists()

    def test_save_and_load(self, parquet_storage, sample_ohlcv_df):
        """Test basic save and load."""
        rows = parquet_storage.save(sample_ohlcv_df, "EURUSD", "1H")

        assert rows == len(sample_ohlcv_df)

        loaded = parquet_storage.load("EURUSD", "1H")

        assert len(loaded) == len(sample_ohlcv_df)
        assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(loaded.index, pd.DatetimeIndex)

    def test_save_creates_metadata(self, parquet_storage, sample_ohlcv_df):
        """Test metadata file creation."""
        parquet_storage.save(sample_ohlcv_df, "EURUSD", "1H")

        meta_path = parquet_storage._get_metadata_path("EURUSD", "1H")
        assert meta_path.exists()

        with open(meta_path) as f:
            metadata = json.load(f)

        assert metadata["symbol"] == "EURUSD"
        assert metadata["timeframe"] == "1H"
        assert metadata["rows"] == 1000
        assert "checksum" in metadata

    def test_save_overwrite(self, parquet_storage, sample_ohlcv_df):
        """Test save with overwrite."""
        parquet_storage.save(sample_ohlcv_df, "EURUSD", "1H")

        # Create smaller dataframe
        small_df = sample_ohlcv_df.iloc[:100]
        parquet_storage.save(small_df, "EURUSD", "1H", overwrite=True)

        loaded = parquet_storage.load("EURUSD", "1H")
        assert len(loaded) == 100

    def test_load_with_date_filter(self, parquet_storage, sample_ohlcv_df):
        """Test load with date range filter."""
        parquet_storage.save(sample_ohlcv_df, "EURUSD", "1H")

        start = datetime(2024, 1, 10)
        end = datetime(2024, 1, 20)
        loaded = parquet_storage.load("EURUSD", "1H", start, end)

        assert loaded.index.min() >= pd.Timestamp(start)
        assert loaded.index.max() <= pd.Timestamp(end)

    def test_load_not_found(self, parquet_storage):
        """Test load raises error for non-existent data."""
        with pytest.raises(StorageNotFoundError):
            parquet_storage.load("UNKNOWN", "1H")

    def test_append_new_data(self, parquet_storage, sample_ohlcv_df):
        """Test appending new data."""
        # Save initial data
        initial = sample_ohlcv_df.iloc[:500]
        parquet_storage.save(initial, "EURUSD", "1H")

        # Append more data
        new_data = sample_ohlcv_df.iloc[400:700]  # Overlapping range
        added = parquet_storage.append(new_data, "EURUSD", "1H")

        # Load and verify
        loaded = parquet_storage.load("EURUSD", "1H")
        assert len(loaded) == 700  # Deduped: 500 + 200 new

    def test_append_to_empty(self, parquet_storage, sample_ohlcv_df):
        """Test append when no existing data."""
        small_df = sample_ohlcv_df.iloc[:100]
        rows = parquet_storage.append(small_df, "NEWPAIR", "1H")

        assert rows == 100
        assert parquet_storage.exists("NEWPAIR", "1H")

    def test_delete_all(self, parquet_storage, sample_ohlcv_df):
        """Test deleting all data."""
        parquet_storage.save(sample_ohlcv_df, "EURUSD", "1H")

        deleted = parquet_storage.delete("EURUSD", "1H")

        assert deleted == len(sample_ohlcv_df)
        assert not parquet_storage.exists("EURUSD", "1H")

    def test_delete_date_range(self, parquet_storage, sample_ohlcv_df):
        """Test partial delete by date range."""
        parquet_storage.save(sample_ohlcv_df, "EURUSD", "1H")
        initial_count = parquet_storage.get_row_count("EURUSD", "1H")

        # Delete middle portion
        start = datetime(2024, 1, 15)
        end = datetime(2024, 1, 25)
        deleted = parquet_storage.delete("EURUSD", "1H", start, end)

        remaining = parquet_storage.get_row_count("EURUSD", "1H")
        assert remaining == initial_count - deleted

    def test_exists(self, parquet_storage, sample_ohlcv_df):
        """Test exists check."""
        assert not parquet_storage.exists("EURUSD", "1H")

        parquet_storage.save(sample_ohlcv_df, "EURUSD", "1H")

        assert parquet_storage.exists("EURUSD", "1H")
        assert not parquet_storage.exists("EURUSD", "1D")

    def test_get_date_range(self, parquet_storage, sample_ohlcv_df):
        """Test getting date range."""
        parquet_storage.save(sample_ohlcv_df, "EURUSD", "1H")

        date_range = parquet_storage.get_date_range("EURUSD", "1H")

        assert date_range is not None
        assert date_range[0] == sample_ohlcv_df.index.min().to_pydatetime()
        assert date_range[1] == sample_ohlcv_df.index.max().to_pydatetime()

    def test_get_date_range_not_found(self, parquet_storage):
        """Test date range for non-existent data."""
        result = parquet_storage.get_date_range("UNKNOWN", "1H")
        assert result is None

    def test_list_symbols(self, parquet_storage, sample_ohlcv_df):
        """Test listing symbols."""
        parquet_storage.save(sample_ohlcv_df, "EURUSD", "1H")
        parquet_storage.save(sample_ohlcv_df, "GBPUSD", "1H")
        parquet_storage.save(sample_ohlcv_df, "EURUSD", "1D")

        symbols = parquet_storage.list_symbols()

        assert "EURUSD" in symbols
        assert "GBPUSD" in symbols

    def test_list_symbols_with_timeframe_filter(self, parquet_storage, sample_ohlcv_df):
        """Test listing symbols filtered by timeframe."""
        parquet_storage.save(sample_ohlcv_df, "EURUSD", "1H")
        parquet_storage.save(sample_ohlcv_df, "GBPUSD", "1D")

        symbols = parquet_storage.list_symbols(timeframe="1H")

        assert "EURUSD" in symbols
        assert "GBPUSD" not in symbols

    def test_list_timeframes(self, parquet_storage, sample_ohlcv_df):
        """Test listing timeframes."""
        parquet_storage.save(sample_ohlcv_df, "EURUSD", "1H")
        parquet_storage.save(sample_ohlcv_df, "EURUSD", "4H")
        parquet_storage.save(sample_ohlcv_df, "EURUSD", "1D")

        timeframes = parquet_storage.list_timeframes("EURUSD")

        assert "1H" in timeframes
        assert "4H" in timeframes
        assert "1D" in timeframes

    def test_get_row_count(self, parquet_storage, sample_ohlcv_df):
        """Test row count."""
        parquet_storage.save(sample_ohlcv_df, "EURUSD", "1H")

        count = parquet_storage.get_row_count("EURUSD", "1H")

        assert count == len(sample_ohlcv_df)

    def test_get_row_count_not_found(self, parquet_storage):
        """Test row count for non-existent data."""
        count = parquet_storage.get_row_count("UNKNOWN", "1H")
        assert count == 0

    def test_verify_integrity_valid(self, parquet_storage, sample_ohlcv_df):
        """Test integrity verification passes."""
        parquet_storage.save(sample_ohlcv_df, "EURUSD", "1H")

        result = parquet_storage.verify_integrity("EURUSD", "1H")

        assert result is True

    def test_get_storage_size(self, parquet_storage, sample_ohlcv_df):
        """Test storage size calculation."""
        parquet_storage.save(sample_ohlcv_df, "EURUSD", "1H")

        size = parquet_storage.get_storage_size("EURUSD", "1H")

        assert size > 0

    def test_get_storage_size_not_found(self, parquet_storage):
        """Test storage size for non-existent data."""
        size = parquet_storage.get_storage_size("UNKNOWN", "1H")
        assert size == 0

    def test_optimize(self, parquet_storage, sample_ohlcv_df):
        """Test storage optimization."""
        parquet_storage.save(sample_ohlcv_df, "EURUSD", "1H")

        # Should complete without error
        parquet_storage.optimize("EURUSD", "1H")

        # Data should still be valid
        loaded = parquet_storage.load("EURUSD", "1H")
        assert len(loaded) == len(sample_ohlcv_df)

    def test_get_storage_info(self, parquet_storage, sample_ohlcv_df):
        """Test storage info retrieval."""
        parquet_storage.save(sample_ohlcv_df, "EURUSD", "1H")
        parquet_storage.save(sample_ohlcv_df, "GBPUSD", "1H")

        info = parquet_storage.get_storage_info()

        assert info["total_symbols"] == 2
        assert info["total_rows"] == 2000
        assert "EURUSD" in info["symbols"]


class TestParquetStorageCompression:
    """Tests for different compression options."""

    @pytest.mark.parametrize("compression", ["snappy", "gzip", "zstd", None])
    def test_compression_options(self, tmp_path, sample_ohlcv_df, compression):
        """Test different compression algorithms."""
        storage = ParquetStorage(tmp_path, compression=compression)
        storage.save(sample_ohlcv_df, "TEST", "1H")

        loaded = storage.load("TEST", "1H")
        assert len(loaded) == len(sample_ohlcv_df)


class TestParquetStorageCaseNormalization:
    """Tests for symbol/timeframe case handling."""

    def test_symbol_case_normalization(self, parquet_storage, sample_ohlcv_df):
        """Test symbol is stored uppercase."""
        parquet_storage.save(sample_ohlcv_df, "eurusd", "1h")

        assert parquet_storage.exists("EURUSD", "1H")
        assert parquet_storage.exists("eurusd", "1h")

        loaded = parquet_storage.load("eurusd", "1h")
        assert len(loaded) == len(sample_ohlcv_df)
