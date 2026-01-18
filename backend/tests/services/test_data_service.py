"""Unit tests for DataService."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path


class TestDataServiceInitialization:
    """Test DataService initialization."""

    def test_initial_state(self):
        """Test service starts with correct initial state."""
        from src.api.services.data_service import DataService

        service = DataService()

        assert service._initialized is False
        assert service._price_cache == {}
        assert service._ohlcv_cache == {}
        assert service._vix_cache is None
        assert service._historical_data is None

    def test_is_loaded_property(self):
        """Test is_loaded property reflects initialization state."""
        from src.api.services.data_service import DataService

        service = DataService()

        assert service.is_loaded is False

        service._initialized = True
        assert service.is_loaded is True

    def test_initialize_already_initialized(self):
        """Test initialize returns early if already initialized."""
        from src.api.services.data_service import DataService

        service = DataService()
        service._initialized = True

        # Should not raise or do anything
        service.initialize()

        assert service._initialized is True


class TestDataServicePriceCache:
    """Test DataService price caching."""

    def test_price_cache_hit(self):
        """Test that cached prices are returned within TTL."""
        from src.api.services.data_service import DataService

        service = DataService()

        # Pre-populate cache
        cached_price = 1.08543
        service._price_cache["EURUSD=X"] = (cached_price, datetime.now())

        # Should return cached value
        with patch.object(service, '_get_live_data') as mock_live:
            price = service.get_current_price("EURUSD")

            # Mock should not be called - cache hit
            assert price == cached_price

    def test_price_cache_expired(self):
        """Test that expired cache triggers new fetch."""
        from src.api.services.data_service import DataService

        service = DataService()

        # Pre-populate cache with old timestamp
        old_time = datetime.now() - timedelta(hours=1)
        service._price_cache["EURUSD=X"] = (1.08000, old_time)

        # Mock yfinance
        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker.return_value = mock_ticker_instance

            # Create mock price data
            mock_data = pd.DataFrame({
                "Close": [1.08543],
            })
            mock_ticker_instance.history.return_value = mock_data

            price = service.get_current_price("EURUSD")

            assert price == 1.08543
            # New price should be in cache
            assert service._price_cache["EURUSD=X"][0] == 1.08543


class TestDataServiceOHLCV:
    """Test DataService OHLCV data methods."""

    def test_get_ohlcv_data_empty(self):
        """Test get_ohlcv_data returns None when no data."""
        from src.api.services.data_service import DataService

        service = DataService()

        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker.return_value = mock_ticker_instance
            mock_ticker_instance.history.return_value = pd.DataFrame()

            result = service.get_ohlcv_data("EURUSD", "7d", "1h")

            assert result is None

    def test_get_ohlcv_data_success(self):
        """Test get_ohlcv_data returns data when available."""
        from src.api.services.data_service import DataService

        service = DataService()

        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker.return_value = mock_ticker_instance

            # Create mock OHLCV data
            mock_data = pd.DataFrame({
                "Open": [1.08500, 1.08550],
                "High": [1.08600, 1.08650],
                "Low": [1.08450, 1.08500],
                "Close": [1.08550, 1.08600],
                "Volume": [1000, 1100],
            }, index=pd.date_range("2024-01-01", periods=2, freq="1h"))
            mock_ticker_instance.history.return_value = mock_data

            result = service.get_ohlcv_data("EURUSD", "7d", "1h")

            assert result is not None
            assert len(result) == 2
            assert "open" in result.columns
            assert "close" in result.columns


class TestDataServiceVIX:
    """Test DataService VIX data methods."""

    def test_get_vix_data_empty(self):
        """Test get_vix_data returns None when no data."""
        from src.api.services.data_service import DataService

        service = DataService()

        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker.return_value = mock_ticker_instance
            mock_ticker_instance.history.return_value = pd.DataFrame()

            result = service.get_vix_data()

            assert result is None

    def test_get_latest_vix(self):
        """Test get_latest_vix returns most recent VIX value."""
        from src.api.services.data_service import DataService

        service = DataService()

        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker.return_value = mock_ticker_instance

            # Create mock VIX data
            mock_data = pd.DataFrame({
                "Close": [14.5, 15.0, 15.5],
            }, index=pd.date_range("2024-01-01", periods=3, freq="1d"))
            mock_ticker_instance.history.return_value = mock_data

            result = service.get_latest_vix()

            assert result == 15.5


class TestDataServiceResampling:
    """Test DataService data resampling."""

    def test_resample_5min_returns_copy(self):
        """Test resampling to 5min returns original data copy."""
        from src.api.services.data_service import DataService

        service = DataService()

        df = pd.DataFrame({
            "open": [1.08500],
            "high": [1.08600],
            "low": [1.08450],
            "close": [1.08550],
            "volume": [1000],
        }, index=pd.date_range("2024-01-01", periods=1, freq="5min"))

        result = service.resample_data(df, "5min")

        assert len(result) == 1
        assert result is not df  # Should be a copy

    def test_resample_to_1h(self):
        """Test resampling 5-min data to 1H."""
        from src.api.services.data_service import DataService

        service = DataService()

        # Create 12 5-min bars (1 hour worth)
        df = pd.DataFrame({
            "open": np.linspace(1.085, 1.086, 12),
            "high": np.linspace(1.0855, 1.0865, 12),
            "low": np.linspace(1.0845, 1.0855, 12),
            "close": np.linspace(1.0852, 1.0862, 12),
            "volume": [100] * 12,
        }, index=pd.date_range("2024-01-01 10:00", periods=12, freq="5min"))

        result = service.resample_data(df, "1H")

        assert len(result) == 1
        assert result.iloc[0]["open"] == df.iloc[0]["open"]
        assert result.iloc[0]["close"] == df.iloc[-1]["close"]


class TestDataServiceMarketInfo:
    """Test DataService market info methods."""

    def test_get_market_info_success(self):
        """Test get_market_info returns correct structure."""
        from src.api.services.data_service import DataService

        service = DataService()

        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker_instance = Mock()
            mock_ticker.return_value = mock_ticker_instance

            # Mock today's minute data
            today_data = pd.DataFrame({
                "Open": [1.08500],
                "High": [1.08600],
                "Low": [1.08450],
                "Close": [1.08550],
            }, index=pd.date_range("2024-01-01", periods=1, freq="1min"))

            # Mock yesterday's daily data
            yesterday_data = pd.DataFrame({
                "Close": [1.08400, 1.08500],
            }, index=pd.date_range("2023-12-31", periods=2, freq="1d"))

            mock_ticker_instance.history.side_effect = [today_data, yesterday_data]

            result = service.get_market_info("EURUSD")

            assert "symbol" in result
            assert result["symbol"] == "EURUSD"
            assert "price" in result
            assert "timestamp" in result

    def test_get_market_info_error(self):
        """Test get_market_info handles errors gracefully."""
        from src.api.services.data_service import DataService

        service = DataService()

        with patch('yfinance.Ticker') as mock_ticker:
            mock_ticker.side_effect = Exception("Network error")

            result = service.get_market_info("EURUSD")

            assert "error" in result
            assert result["symbol"] == "EURUSD"


class TestDataServiceCacheManagement:
    """Test DataService cache management."""

    def test_clear_cache(self):
        """Test clear_cache clears all caches."""
        from src.api.services.data_service import DataService

        service = DataService()

        # Populate caches
        service._price_cache = {"test": (1.0, datetime.now())}
        service._ohlcv_cache = {"test": (pd.DataFrame(), datetime.now())}
        service._vix_cache = (pd.Series(), datetime.now())
        service._prediction_data_cache = (pd.DataFrame(), datetime.now())

        service.clear_cache()

        assert service._price_cache == {}
        assert service._ohlcv_cache == {}
        assert service._vix_cache is None
        assert service._prediction_data_cache is None


class TestDataServiceHistoricalData:
    """Test DataService historical data handling."""

    def test_get_historical_data_info_not_loaded(self):
        """Test historical data info when not loaded."""
        from src.api.services.data_service import DataService

        service = DataService()

        info = service.get_historical_data_info()

        assert info["loaded"] is False
        assert "file" in info

    def test_get_historical_data_info_loaded(self):
        """Test historical data info when loaded."""
        from src.api.services.data_service import DataService

        service = DataService()

        # Simulate loaded data
        service._historical_data = pd.DataFrame({
            "open": [1.08500],
            "close": [1.08550],
        }, index=pd.date_range("2024-01-01", periods=1, freq="5min"))
        service._historical_loaded_at = datetime.now()

        info = service.get_historical_data_info()

        assert info["loaded"] is True
        assert info["rows"] == 1
        assert "start" in info
        assert "end" in info
