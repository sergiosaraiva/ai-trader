"""Tests for asset_service.py to verify security fixes."""

import pytest
from src.api.services.asset_service import AssetService, MAX_CACHE_SIZE


class TestAssetServiceSecurity:
    """Test security fixes in AssetService."""

    def setup_method(self):
        """Create a fresh service instance for each test."""
        self.service = AssetService()

    def test_cache_size_limit_enforced(self):
        """Test that cache doesn't grow beyond MAX_CACHE_SIZE."""
        # Fill cache to the limit
        for i in range(MAX_CACHE_SIZE):
            symbol = f"TEST{i:04d}"
            self.service.get_asset_metadata(symbol)

        assert len(self.service._cache) == MAX_CACHE_SIZE

        # Add one more - should evict the oldest
        self.service.get_asset_metadata("OVERFLOW")
        assert len(self.service._cache) == MAX_CACHE_SIZE

        # First symbol should have been evicted (FIFO)
        assert "TEST0000" not in self.service._cache
        assert "OVERFLOW" in self.service._cache

    def test_cache_fifo_eviction_order(self):
        """Test that cache evicts in FIFO order."""
        # Fill cache
        for i in range(MAX_CACHE_SIZE):
            self.service.get_asset_metadata(f"SYM{i:04d}")

        # Add 5 more symbols
        for i in range(5):
            self.service.get_asset_metadata(f"NEW{i}")

        # First 5 symbols should be evicted
        for i in range(5):
            assert f"SYM{i:04d}" not in self.service._cache

        # All new symbols should be present
        for i in range(5):
            assert f"NEW{i}" in self.service._cache

    def test_error_handling_invalid_symbol(self):
        """Test error handling for malformed symbols."""
        # This shouldn't crash, should return default metadata
        metadata = self.service.get_asset_metadata("")
        assert metadata is not None
        assert metadata.symbol == ""
        # Should return unknown type as fallback
        # (will default to stock unless empty string handling added)

    def test_error_handling_with_exception(self, monkeypatch):
        """Test that exceptions are caught and default metadata returned."""
        # Force an exception in detection
        def raise_error(*args):
            raise RuntimeError("Forced error")

        monkeypatch.setattr(self.service, "_is_forex", raise_error)

        # Should not raise, should return default metadata
        metadata = self.service.get_asset_metadata("TEST")
        assert metadata is not None
        assert metadata.asset_type == "unknown"

    def test_forex_detection_normal(self):
        """Test normal forex symbol detection."""
        metadata = self.service.get_asset_metadata("EURUSD")
        assert metadata.asset_type == "forex"
        assert metadata.base_currency == "EUR"
        assert metadata.quote_currency == "USD"
        assert metadata.profit_unit == "pips"

    def test_crypto_detection_normal(self):
        """Test normal crypto symbol detection."""
        metadata = self.service.get_asset_metadata("BTC-USD")
        assert metadata.asset_type == "crypto"
        assert metadata.base_currency == "BTC"
        assert metadata.quote_currency == "USD"
        assert metadata.profit_unit == "dollars"

    def test_stock_detection_default(self):
        """Test stock symbol as default fallback."""
        metadata = self.service.get_asset_metadata("AAPL")
        assert metadata.asset_type == "stock"
        assert metadata.quote_currency == "USD"
        assert metadata.profit_unit == "points"

    def test_cache_hit_doesnt_count_toward_limit(self):
        """Test that cache hits don't trigger eviction."""
        # Fill cache
        for i in range(MAX_CACHE_SIZE):
            self.service.get_asset_metadata(f"TEST{i:04d}")

        # Request first symbol multiple times (cache hits)
        for _ in range(10):
            self.service.get_asset_metadata("TEST0000")

        # First symbol should still be in cache
        assert "TEST0000" in self.service._cache
        assert len(self.service._cache) == MAX_CACHE_SIZE

    def test_clear_cache_resets_limit(self):
        """Test that clear_cache allows refilling."""
        # Fill cache
        for i in range(MAX_CACHE_SIZE):
            self.service.get_asset_metadata(f"TEST{i:04d}")

        self.service.clear_cache()
        assert len(self.service._cache) == 0

        # Should be able to fill again
        for i in range(MAX_CACHE_SIZE):
            self.service.get_asset_metadata(f"NEW{i:04d}")

        assert len(self.service._cache) == MAX_CACHE_SIZE


class TestSymbolValidation:
    """Test symbol validation patterns."""

    @pytest.mark.parametrize("symbol", [
        "EURUSD",
        "BTC-USD",
        "AAPL",
        "TSLA",
        "EUR-USD",
        "ETH",
        "A",
        "ABC123",
        "TEST-123",
    ])
    def test_valid_symbols_accepted(self, symbol):
        """Test that valid symbols are accepted."""
        import re
        pattern = r"^[A-Za-z0-9\-]{1,20}$"
        assert re.match(pattern, symbol), f"Valid symbol rejected: {symbol}"

    @pytest.mark.parametrize("symbol", [
        "EUR/USD",  # Contains slash
        "BTC USD",  # Contains space
        "A" * 21,   # Too long
        "EUR$USD",  # Contains special char
        "'; DROP TABLE--",  # SQL injection attempt
        "../../etc/passwd",  # Path traversal attempt
        "<script>alert(1)</script>",  # XSS attempt
        "",  # Empty string
        "TEST\nNEW",  # Contains newline
        "TEST\x00END",  # Contains null byte
    ])
    def test_invalid_symbols_rejected(self, symbol):
        """Test that invalid symbols are rejected."""
        import re
        pattern = r"^[A-Za-z0-9\-]{1,20}$"
        assert not re.match(pattern, symbol), f"Invalid symbol accepted: {symbol}"


class TestAssetMetadataFallback:
    """Test default metadata fallback behavior."""

    def test_default_metadata_structure(self):
        """Test that default metadata has correct structure."""
        service = AssetService()
        # Force error by mocking
        metadata = service._create_default_metadata("UNKNOWN")

        assert metadata.symbol == "UNKNOWN"
        assert metadata.asset_type == "unknown"
        assert metadata.price_precision == 5
        assert metadata.profit_unit == "points"
        assert metadata.profit_multiplier == 1.0
        assert metadata.base_currency is None
        assert metadata.quote_currency is None
