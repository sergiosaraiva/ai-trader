"""Unit tests for CSV data source."""

import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from src.data.sources.csv_source import CSVDataSource, CSVDataSourceError


@pytest.fixture
def sample_csv_data():
    """Create sample OHLCV data."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "open": [1.1 + i * 0.001 for i in range(100)],
            "high": [1.11 + i * 0.001 for i in range(100)],
            "low": [1.09 + i * 0.001 for i in range(100)],
            "close": [1.105 + i * 0.001 for i in range(100)],
            "volume": [1000 + i * 10 for i in range(100)],
        }
    )


@pytest.fixture
def temp_csv_dir(sample_csv_data, tmp_path):
    """Create temporary directory with CSV files."""
    # Create EURUSD_1H.csv
    csv_path = tmp_path / "EURUSD_1H.csv"
    sample_csv_data.to_csv(csv_path, index=False)

    # Create another file with date range pattern
    csv_path2 = tmp_path / "GBPUSD_20240101_20240105_5M.csv"
    sample_csv_data.to_csv(csv_path2, index=False)

    return tmp_path


class TestCSVDataSource:
    """Tests for CSVDataSource class."""

    def test_init_default(self):
        """Test initialization with defaults."""
        source = CSVDataSource()
        assert source.base_path == Path("data")
        assert source.date_column == "timestamp"
        assert not source.is_connected

    def test_init_with_config(self, tmp_path):
        """Test initialization with config."""
        config = {
            "base_path": str(tmp_path),
            "date_column": "datetime",
            "cache_enabled": False,
        }
        source = CSVDataSource(config=config)
        assert source.base_path == tmp_path
        assert source.date_column == "datetime"
        assert not source._cache_enabled

    def test_connect_creates_index(self, temp_csv_dir):
        """Test connect indexes CSV files."""
        source = CSVDataSource(base_path=temp_csv_dir)
        result = source.connect()

        assert result is True
        assert source.is_connected
        assert "EURUSD" in source._file_index
        assert "GBPUSD" in source._file_index

    def test_connect_nonexistent_path(self, tmp_path):
        """Test connect with non-existent path."""
        source = CSVDataSource(base_path=tmp_path / "nonexistent")
        result = source.connect()

        assert result is False
        assert not source.is_connected

    def test_disconnect_clears_cache(self, temp_csv_dir):
        """Test disconnect clears caches."""
        source = CSVDataSource(base_path=temp_csv_dir)
        source.connect()
        source._file_cache["test"] = "data"

        source.disconnect()

        assert not source.is_connected
        assert len(source._file_cache) == 0
        assert len(source._file_index) == 0

    def test_fetch_ohlcv_success(self, temp_csv_dir, sample_csv_data):
        """Test successful OHLCV fetch."""
        source = CSVDataSource(base_path=temp_csv_dir)
        source.connect()

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)
        df = source.fetch_ohlcv("EURUSD", "1H", start, end)

        assert not df.empty
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_fetch_ohlcv_case_insensitive(self, temp_csv_dir):
        """Test symbol is case insensitive."""
        source = CSVDataSource(base_path=temp_csv_dir)
        source.connect()

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)
        df1 = source.fetch_ohlcv("eurusd", "1H", start, end)
        df2 = source.fetch_ohlcv("EURUSD", "1H", start, end)

        assert len(df1) == len(df2)

    def test_fetch_ohlcv_not_found(self, temp_csv_dir):
        """Test fetch with non-existent symbol."""
        source = CSVDataSource(base_path=temp_csv_dir)
        source.connect()

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)

        with pytest.raises(CSVDataSourceError, match="No CSV files found"):
            source.fetch_ohlcv("UNKNOWN", "1H", start, end)

    def test_fetch_ohlcv_filters_date_range(self, temp_csv_dir):
        """Test date range filtering."""
        source = CSVDataSource(base_path=temp_csv_dir)
        source.connect()

        start = datetime(2024, 1, 1, 12, 0)
        end = datetime(2024, 1, 2, 12, 0)
        df = source.fetch_ohlcv("EURUSD", "1H", start, end)

        assert df.index.min() >= pd.Timestamp(start)
        assert df.index.max() <= pd.Timestamp(end)

    def test_get_available_symbols(self, temp_csv_dir):
        """Test listing available symbols."""
        source = CSVDataSource(base_path=temp_csv_dir)
        source.connect()

        symbols = source.get_available_symbols()

        assert "EURUSD" in symbols
        assert "GBPUSD" in symbols

    def test_get_available_timeframes(self, temp_csv_dir):
        """Test listing available timeframes."""
        source = CSVDataSource(base_path=temp_csv_dir)
        source.connect()

        timeframes = source.get_available_timeframes("EURUSD")

        assert "1H" in timeframes

    def test_get_current_price(self, temp_csv_dir):
        """Test getting current price."""
        source = CSVDataSource(base_path=temp_csv_dir)
        source.connect()

        price = source.get_current_price("EURUSD")

        assert "bid" in price
        assert "ask" in price
        assert "last" in price
        assert price["last"] > 0

    def test_get_date_range(self, temp_csv_dir):
        """Test getting date range."""
        source = CSVDataSource(base_path=temp_csv_dir)
        source.connect()

        date_range = source.get_date_range("EURUSD", "1H")

        assert date_range is not None
        assert date_range[0] < date_range[1]

    def test_normalize_timeframe(self, temp_csv_dir):
        """Test timeframe normalization."""
        source = CSVDataSource(base_path=temp_csv_dir)

        assert source._normalize_timeframe("5min") == "5M"
        assert source._normalize_timeframe("1hour") == "1H"
        assert source._normalize_timeframe("daily") == "1D"
        assert source._normalize_timeframe("H1") == "1H"

    def test_cache_enabled(self, temp_csv_dir):
        """Test file caching."""
        source = CSVDataSource(base_path=temp_csv_dir)
        source.connect()

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)

        # First fetch - loads from file
        df1 = source.fetch_ohlcv("EURUSD", "1H", start, end)
        assert len(source._file_cache) > 0

        # Second fetch - should use cache
        df2 = source.fetch_ohlcv("EURUSD", "1H", start, end)
        assert len(df1) == len(df2)

    def test_clear_cache(self, temp_csv_dir):
        """Test cache clearing."""
        source = CSVDataSource(base_path=temp_csv_dir)
        source.connect()

        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 5)
        source.fetch_ohlcv("EURUSD", "1H", start, end)

        assert len(source._file_cache) > 0
        source.clear_cache()
        assert len(source._file_cache) == 0

    def test_context_manager(self, temp_csv_dir):
        """Test context manager protocol."""
        with CSVDataSource(base_path=temp_csv_dir) as source:
            assert source.is_connected
            symbols = source.get_available_symbols()
            assert len(symbols) > 0

        assert not source.is_connected


class TestCSVColumnNormalization:
    """Tests for column name normalization."""

    def test_various_column_names(self, tmp_path):
        """Test different column naming conventions."""
        # Create CSV with non-standard column names
        dates = pd.date_range("2024-01-01", periods=10, freq="1h")
        df = pd.DataFrame(
            {
                "DateTime": dates,
                "O": [1.1] * 10,
                "H": [1.2] * 10,
                "L": [1.0] * 10,
                "C": [1.15] * 10,
                "Vol": [1000] * 10,
            }
        )
        csv_path = tmp_path / "TEST_1H.csv"
        df.to_csv(csv_path, index=False)

        source = CSVDataSource(base_path=tmp_path)
        source.connect()

        result = source.fetch_ohlcv("TEST", "1H", datetime(2024, 1, 1), datetime(2024, 1, 2))

        assert "open" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "close" in result.columns
        assert "volume" in result.columns


class TestGzippedCSV:
    """Tests for gzipped CSV files."""

    def test_load_gzipped_file(self, tmp_path, sample_csv_data):
        """Test loading gzipped CSV."""
        import gzip

        csv_path = tmp_path / "EURUSD_1H.csv.gz"
        with gzip.open(csv_path, "wt") as f:
            sample_csv_data.to_csv(f, index=False)

        source = CSVDataSource(base_path=tmp_path)
        source.connect()

        df = source.fetch_ohlcv("EURUSD", "1H", datetime(2024, 1, 1), datetime(2024, 1, 5))

        assert not df.empty
        assert "open" in df.columns
