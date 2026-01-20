"""Unit tests for DataService pipeline cache integration.

Tests focus on the refactored DataService that reads from pipeline cache
(parquet files) instead of making yfinance API calls.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from pathlib import Path


@pytest.fixture
def sample_ohlcv_df():
    """Sample OHLCV DataFrame for testing."""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    return pd.DataFrame({
        "open": np.linspace(1.085, 1.090, 100),
        "high": np.linspace(1.086, 1.091, 100),
        "low": np.linspace(1.084, 1.089, 100),
        "close": np.linspace(1.0855, 1.0905, 100),
        "volume": np.random.randint(1000, 10000, 100),
    }, index=dates)


@pytest.fixture
def sample_sentiment_df():
    """Sample sentiment DataFrame for testing."""
    dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "VIX": np.linspace(15, 20, 30),
        "EPU_US": np.linspace(100, 150, 30),
        "Sentiment_VIX": np.linspace(-0.1, 0.1, 30),
        "Sentiment_US": np.linspace(-0.05, 0.05, 30),
    }, index=dates)


@pytest.fixture
def mock_pipeline_service(sample_ohlcv_df, sample_sentiment_df, tmp_path):
    """Mock pipeline_service with test cache files."""
    # Set up cache paths
    cache_1h = tmp_path / "eurusd_1h_features.parquet"
    cache_4h = tmp_path / "eurusd_4h_features.parquet"
    cache_daily = tmp_path / "eurusd_daily_features.parquet"
    cache_sentiment = tmp_path / "sentiment_updated.parquet"

    # Write sample data to cache files
    sample_ohlcv_df.to_parquet(cache_1h)
    sample_ohlcv_df.to_parquet(cache_4h)
    sample_ohlcv_df.to_parquet(cache_daily)
    sample_sentiment_df.to_parquet(cache_sentiment)

    # Create mock object
    mock = Mock()
    mock.cache_1h = cache_1h
    mock.cache_4h = cache_4h
    mock.cache_daily = cache_daily
    mock.cache_sentiment = cache_sentiment

    # Mock get_processed_data method
    mock.get_processed_data.side_effect = lambda tf: {
        "1h": sample_ohlcv_df.copy(),
        "4h": sample_ohlcv_df.copy(),
        "D": sample_ohlcv_df.copy(),
    }.get(tf)

    return mock


class TestDataServiceInitialization:
    """Test DataService initialization with pipeline cache."""

    def test_initial_state(self):
        """Test service starts with correct initial state."""
        from src.api.services.data_service import DataService

        service = DataService()

        assert service._initialized is False
        assert service._price_cache == {}
        assert service._ohlcv_cache == {}
        assert service._vix_cache is None
        assert service._historical_data is None
        assert service._prediction_data_cache is None

    def test_is_loaded_property(self):
        """Test is_loaded property reflects initialization state."""
        from src.api.services.data_service import DataService

        service = DataService()

        assert service.is_loaded is False

        service._initialized = True
        assert service.is_loaded is True

    def test_initialize_marks_as_initialized(self):
        """Test initialize sets initialized flag."""
        from src.api.services.data_service import DataService

        service = DataService()

        # Mock the historical data file to not exist
        with patch.object(service, 'HISTORICAL_DATA_FILE', Path('/nonexistent/file.csv')):
            service.initialize()

        assert service._initialized is True


class TestDataServicePipelineCacheReading:
    """Test DataService reading from pipeline cache."""

    def test_get_data_for_prediction_cache_hit(self, mock_pipeline_service):
        """Test get_data_for_prediction reads from pipeline cache."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        # Patch the pipeline_service at module level
        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            # First call - should read from pipeline cache
            df = service.get_data_for_prediction(use_cache=True)

            assert df is not None
            assert not df.empty
            assert len(df) == 100
            assert "open" in df.columns
            assert "close" in df.columns

            # Verify pipeline service was called
            mock_pipeline_service.get_processed_data.assert_called_with("1h")

    def test_get_data_for_prediction_in_memory_cache(self, mock_pipeline_service, sample_ohlcv_df):
        """Test get_data_for_prediction uses in-memory cache."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        # Pre-populate in-memory cache
        service._prediction_data_cache = (sample_ohlcv_df.copy(), datetime.now())

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            # Should return cached data without calling pipeline
            df = service.get_data_for_prediction(use_cache=True)

            assert df is not None
            assert len(df) == 100

            # Pipeline service should not be called (in-memory cache hit)
            mock_pipeline_service.get_processed_data.assert_not_called()

    def test_get_data_for_prediction_expired_in_memory_cache(self, mock_pipeline_service, sample_ohlcv_df):
        """Test get_data_for_prediction refreshes expired in-memory cache."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        # Pre-populate cache with old timestamp
        old_time = datetime.now() - timedelta(minutes=20)
        service._prediction_data_cache = (sample_ohlcv_df.copy(), old_time)

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            # Should fetch fresh data
            df = service.get_data_for_prediction(use_cache=True)

            assert df is not None

            # Pipeline service should be called (cache expired)
            mock_pipeline_service.get_processed_data.assert_called_once()

    def test_get_data_for_prediction_cache_miss(self):
        """Test get_data_for_prediction handles missing pipeline cache."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        # Create mock that returns None
        mock_pipeline = Mock()
        mock_pipeline.get_processed_data.return_value = None

        # Should fall back to live data method
        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            with patch.object(service, '_get_live_data_for_prediction') as mock_live:
                mock_live.return_value = None

                df = service.get_data_for_prediction()

                # Verify fallback was called
                mock_live.assert_called_once()

    def test_get_data_for_prediction_only_ohlcv_columns(self, mock_pipeline_service, sample_ohlcv_df):
        """Test get_data_for_prediction returns only OHLCV columns."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        # Add extra columns to mock data
        df_with_extras = sample_ohlcv_df.copy()
        df_with_extras["extra_feature"] = np.random.randn(len(df_with_extras))
        df_with_extras["another_feature"] = np.random.randn(len(df_with_extras))

        mock_pipeline_service.get_processed_data.return_value = df_with_extras

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            df = service.get_data_for_prediction()

            assert df is not None
            # Should only have OHLCV columns
            expected_cols = ["open", "high", "low", "close", "volume"]
            assert all(col in expected_cols for col in df.columns)


class TestDataServiceCurrentPrice:
    """Test DataService get_current_price from pipeline cache."""

    def test_get_current_price_from_pipeline_cache(self, mock_pipeline_service, sample_ohlcv_df):
        """Test get_current_price reads latest bar from pipeline cache."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            price = service.get_current_price("EURUSD")

            assert price is not None
            assert isinstance(price, float)
            # Should be the last close price
            expected_price = float(sample_ohlcv_df["close"].iloc[-1])
            assert price == expected_price

            # Verify pipeline service was called
            mock_pipeline_service.get_processed_data.assert_called_with("1h")

    def test_get_current_price_caching(self, mock_pipeline_service, sample_ohlcv_df):
        """Test get_current_price uses in-memory cache."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            # First call - should cache
            price1 = service.get_current_price("EURUSD")

            # Second call - should use cache
            price2 = service.get_current_price("EURUSD")

            assert price1 == price2

            # Pipeline service should only be called once (second call uses cache)
            assert mock_pipeline_service.get_processed_data.call_count == 1

    def test_get_current_price_cache_expired(self, mock_pipeline_service, sample_ohlcv_df):
        """Test get_current_price refreshes expired cache."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        # Pre-populate cache with old timestamp
        old_time = datetime.now() - timedelta(minutes=20)
        service._price_cache["EURUSD_price"] = (1.08000, old_time)

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            price = service.get_current_price("EURUSD")

            assert price is not None
            # Should get fresh price from pipeline
            expected_price = float(sample_ohlcv_df["close"].iloc[-1])
            assert price == expected_price

    def test_get_current_price_pipeline_unavailable(self):
        """Test get_current_price handles pipeline cache unavailable."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        # Mock pipeline service returning None
        mock_pipeline = Mock()
        mock_pipeline.get_processed_data.return_value = None

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            price = service.get_current_price("EURUSD")

            assert price is None

    def test_get_current_price_no_close_column(self):
        """Test get_current_price handles missing close column."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        # Mock pipeline returning data without close column
        mock_pipeline = Mock()
        df_no_close = pd.DataFrame({
            "open": [1.085],
            "high": [1.086],
            "low": [1.084],
        }, index=pd.date_range("2024-01-01", periods=1, freq="1h"))

        mock_pipeline.get_processed_data.return_value = df_no_close

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            price = service.get_current_price("EURUSD")

            assert price is None


class TestDataServiceVIXData:
    """Test DataService get_vix_data from pipeline sentiment cache."""

    def test_get_vix_data_from_sentiment_cache(self, mock_pipeline_service, sample_sentiment_df):
        """Test get_vix_data reads from pipeline sentiment cache."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            vix = service.get_vix_data()

            assert vix is not None
            assert len(vix) == 30
            assert vix.name == "vix"

            # Verify cache file was read
            assert mock_pipeline_service.cache_sentiment.exists()

    def test_get_vix_data_caching(self, mock_pipeline_service):
        """Test get_vix_data uses in-memory cache."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            # First call - should cache
            vix1 = service.get_vix_data()

            # Second call - should use cache (no file read)
            vix2 = service.get_vix_data()

            assert vix1 is not None
            assert vix2 is not None
            assert len(vix1) == len(vix2)

    def test_get_vix_data_force_refresh(self, mock_pipeline_service):
        """Test get_vix_data force_refresh bypasses cache."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            # First call - should cache
            vix1 = service.get_vix_data()

            # Modify cache
            service._vix_cache = (pd.Series([999.0]), datetime.now())

            # Force refresh - should read fresh data
            vix2 = service.get_vix_data(force_refresh=True)

            assert vix2 is not None
            # Should have fresh data, not the modified cache
            assert vix2.iloc[0] != 999.0

    def test_get_vix_data_sentiment_cache_unavailable(self, tmp_path):
        """Test get_vix_data handles missing sentiment cache."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        # Mock pipeline service with non-existent cache
        mock_pipeline = Mock()
        mock_pipeline.cache_sentiment = tmp_path / "nonexistent.parquet"

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            vix = service.get_vix_data()

            assert vix is None

    def test_get_vix_data_no_vix_column(self, mock_pipeline_service, tmp_path):
        """Test get_vix_data handles missing VIX column."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        # Create sentiment data without VIX
        df_no_vix = pd.DataFrame({
            "EPU_US": [100, 110, 120],
        }, index=pd.date_range("2024-01-01", periods=3, freq="D"))

        # Overwrite cache file
        df_no_vix.to_parquet(mock_pipeline_service.cache_sentiment)

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            vix = service.get_vix_data()

            assert vix is None

    def test_get_latest_vix(self, mock_pipeline_service, sample_sentiment_df):
        """Test get_latest_vix returns most recent VIX value."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            latest_vix = service.get_latest_vix()

            assert latest_vix is not None
            assert isinstance(latest_vix, float)
            # Should be the last VIX value
            expected_vix = float(sample_sentiment_df["VIX"].iloc[-1])
            assert latest_vix == expected_vix

    def test_get_latest_vix_no_data(self, tmp_path):
        """Test get_latest_vix returns None when no VIX data."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        mock_pipeline = Mock()
        mock_pipeline.cache_sentiment = tmp_path / "nonexistent.parquet"

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            latest_vix = service.get_latest_vix()

            assert latest_vix is None


class TestDataServiceRecentCandles:
    """Test DataService get_recent_candles from pipeline cache."""

    def test_get_recent_candles_from_pipeline_1h(self, mock_pipeline_service, sample_ohlcv_df):
        """Test get_recent_candles reads from pipeline cache for 1H."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            df = service.get_recent_candles("EURUSD", "1H", 24)

            assert df is not None
            assert len(df) == 24
            assert "open" in df.columns
            assert "close" in df.columns

            # Verify pipeline cache was used
            mock_pipeline_service.get_processed_data.assert_called_with("1h")

    def test_get_recent_candles_from_pipeline_4h(self, mock_pipeline_service):
        """Test get_recent_candles reads from pipeline cache for 4H."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            df = service.get_recent_candles("EURUSD", "4H", 24)

            assert df is not None

            # Verify 4h cache was used
            mock_pipeline_service.get_processed_data.assert_called_with("4h")

    def test_get_recent_candles_from_pipeline_daily(self, mock_pipeline_service):
        """Test get_recent_candles reads from pipeline cache for Daily."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            df = service.get_recent_candles("EURUSD", "D", 30)

            assert df is not None

            # Verify daily cache was used
            mock_pipeline_service.get_processed_data.assert_called_with("D")

    def test_get_recent_candles_only_ohlcv(self, mock_pipeline_service, sample_ohlcv_df):
        """Test get_recent_candles returns only OHLCV columns."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        # Add extra columns to mock data
        df_with_extras = sample_ohlcv_df.copy()
        df_with_extras["technical_indicator"] = np.random.randn(len(df_with_extras))

        mock_pipeline_service.get_processed_data.return_value = df_with_extras

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            df = service.get_recent_candles("EURUSD", "1H", 24)

            assert df is not None
            # Should only have OHLCV columns
            expected_cols = ["open", "high", "low", "close", "volume"]
            assert all(col in expected_cols for col in df.columns)

    def test_get_recent_candles_pipeline_unavailable_fallback(self, mock_pipeline_service):
        """Test get_recent_candles falls back to yfinance when pipeline unavailable."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        # Mock pipeline raising exception to trigger fallback
        mock_pipeline_service.get_processed_data.side_effect = Exception("Cache unavailable")

        # Mock yfinance fallback
        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            with patch.object(service, 'get_ohlcv_data') as mock_ohlcv:
                mock_ohlcv.return_value = pd.DataFrame({
                    "open": [1.085],
                    "high": [1.086],
                    "low": [1.084],
                    "close": [1.0855],
                    "volume": [1000],
                }, index=pd.date_range("2024-01-01", periods=1, freq="1h"))

                df = service.get_recent_candles("EURUSD", "1H", 24)

                # When pipeline fails, it should return data from yfinance fallback
                assert df is not None
                mock_ohlcv.assert_called()

    def test_get_recent_candles_count_limit(self, mock_pipeline_service, sample_ohlcv_df):
        """Test get_recent_candles respects count parameter."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            # Request only 10 candles
            df = service.get_recent_candles("EURUSD", "1H", 10)

            assert df is not None
            assert len(df) == 10

            # Should return the last 10 bars
            expected_last_close = sample_ohlcv_df["close"].iloc[-1]
            actual_last_close = df["close"].iloc[-1]
            assert actual_last_close == expected_last_close


class TestDataServiceCacheIntegration:
    """Test DataService integration with pipeline_service."""

    def test_data_service_and_pipeline_service_coordination(self, mock_pipeline_service):
        """Test that data_service and pipeline_service work together."""
        from src.api.services.data_service import DataService

        service = DataService()
        service.initialize()

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            # Get prediction data
            df_prediction = service.get_data_for_prediction()

            # Get current price
            current_price = service.get_current_price()

            # Get VIX
            vix = service.get_vix_data()

            # Get recent candles
            df_candles = service.get_recent_candles("EURUSD", "1H", 24)

            # All should succeed
            assert df_prediction is not None
            assert current_price is not None
            assert vix is not None
            assert df_candles is not None

            # Verify pipeline service was used
            assert mock_pipeline_service.get_processed_data.called

    def test_cache_ttl_respected(self, mock_pipeline_service, sample_ohlcv_df):
        """Test that cache TTLs are respected."""
        from src.api.services.data_service import DataService
        import time

        service = DataService()
        service._initialized = True

        # Set short TTL for testing
        service.PRICE_CACHE_TTL = timedelta(seconds=1)

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            # First call
            price1 = service.get_current_price("EURUSD")
            call_count_1 = mock_pipeline_service.get_processed_data.call_count

            # Second call immediately - should use cache
            price2 = service.get_current_price("EURUSD")
            call_count_2 = mock_pipeline_service.get_processed_data.call_count

            assert call_count_1 == call_count_2  # No additional call

            # Wait for TTL to expire
            time.sleep(1.1)

            # Third call - should refresh
            price3 = service.get_current_price("EURUSD")
            call_count_3 = mock_pipeline_service.get_processed_data.call_count

            assert call_count_3 > call_count_2  # Additional call made

    def test_initialize_sequence(self, mock_pipeline_service):
        """Test initialization sequence loads correctly."""
        from src.api.services.data_service import DataService

        service = DataService()

        assert not service.is_loaded

        # Mock historical data file
        with patch.object(service, 'HISTORICAL_DATA_FILE', Path('/nonexistent/file.csv')):
            service.initialize()

        assert service.is_loaded

        # Service should be ready to use
        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            df = service.get_data_for_prediction()
            assert df is not None


class TestDataServiceCacheMiss:
    """Test DataService behavior when pipeline cache is missing."""

    def test_prediction_data_cache_miss_empty_df(self):
        """Test get_data_for_prediction when cache returns empty DataFrame."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        mock_pipeline = Mock()
        mock_pipeline.get_processed_data.return_value = pd.DataFrame()

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            with patch.object(service, '_get_live_data_for_prediction') as mock_live:
                mock_live.return_value = None

                df = service.get_data_for_prediction()

                # Should fall back and return None
                assert df is None

    def test_current_price_cache_miss(self):
        """Test get_current_price when cache is unavailable."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        mock_pipeline = Mock()
        mock_pipeline.get_processed_data.return_value = None

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            price = service.get_current_price("EURUSD")

            assert price is None

    def test_vix_cache_miss(self, tmp_path):
        """Test get_vix_data when sentiment cache is unavailable."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        mock_pipeline = Mock()
        mock_pipeline.cache_sentiment = tmp_path / "nonexistent.parquet"

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            vix = service.get_vix_data()

            assert vix is None


class TestDataServiceCacheCorruption:
    """Test DataService error handling when cache files are corrupted."""

    def test_corrupted_cache_file(self, tmp_path):
        """Test graceful handling of corrupted cache file."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        # Create a corrupted parquet file
        cache_file = tmp_path / "corrupted.parquet"
        cache_file.write_text("This is not a valid parquet file")

        mock_pipeline = Mock()
        mock_pipeline.cache_1h = cache_file
        mock_pipeline.get_processed_data.side_effect = Exception("Corrupted file")

        # Should handle error gracefully
        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            with patch.object(service, '_get_live_data_for_prediction') as mock_live:
                mock_live.return_value = None

                df = service.get_data_for_prediction()

                # Should fall back or return None
                assert df is None

    def test_empty_cache_file(self, tmp_path):
        """Test handling of empty cache file."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        # Create an empty parquet file
        cache_file = tmp_path / "empty.parquet"
        empty_df = pd.DataFrame()
        empty_df.to_parquet(cache_file)

        mock_pipeline = Mock()
        mock_pipeline.get_processed_data.return_value = empty_df

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline):
            with patch.object(service, '_get_live_data_for_prediction') as mock_live:
                mock_live.return_value = None

                df = service.get_data_for_prediction()

                # Should handle empty data
                assert df is None


class TestDataServicePerformance:
    """Test DataService performance characteristics."""

    def test_cache_read_performance(self, mock_pipeline_service):
        """Test that cache reads are fast (<100ms for typical operation)."""
        from src.api.services.data_service import DataService
        import time

        service = DataService()
        service._initialized = True

        with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
            # Warm up
            service.get_data_for_prediction()

            # Measure cached read
            start = time.time()
            service.get_data_for_prediction()
            elapsed = time.time() - start

            # Should be very fast (in-memory cache)
            assert elapsed < 0.1  # 100ms

    def test_multiple_concurrent_reads(self, mock_pipeline_service):
        """Test thread safety of cache reads."""
        from src.api.services.data_service import DataService
        import threading

        service = DataService()
        service._initialized = True

        results = []
        errors = []

        def read_data():
            try:
                with patch('src.api.services.pipeline_service.pipeline_service', mock_pipeline_service):
                    df = service.get_data_for_prediction()
                    results.append(df is not None)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=read_data) for _ in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all to complete
        for t in threads:
            t.join()

        # All should succeed
        assert len(errors) == 0
        assert all(results)


class TestDataServiceClearCache:
    """Test DataService cache clearing."""

    def test_clear_cache(self):
        """Test clear_cache clears all in-memory caches."""
        from src.api.services.data_service import DataService

        service = DataService()

        # Populate caches
        service._price_cache = {"test": (1.0, datetime.now())}
        service._ohlcv_cache = {"test": (pd.DataFrame(), datetime.now())}
        service._vix_cache = (pd.Series(), datetime.now())
        service._prediction_data_cache = (pd.DataFrame(), datetime.now())

        service.clear_cache()

        # All should be cleared
        assert service._price_cache == {}
        assert service._ohlcv_cache == {}
        assert service._vix_cache is None
        assert service._prediction_data_cache is None
