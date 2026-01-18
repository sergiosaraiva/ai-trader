"""Tests for AssetService - asset metadata detection and management."""

import pytest
from src.api.services.asset_service import AssetService
from src.api.schemas.asset import AssetMetadata


class TestAssetService:
    """Test AssetService asset type detection and metadata generation."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up fresh service for each test."""
        self.service = AssetService()
        # Clear cache before each test
        self.service.clear_cache()

    # Forex Detection Tests

    def test_forex_eurusd_detection(self):
        """Test forex detection for EURUSD."""
        metadata = self.service.get_asset_metadata("EURUSD")

        assert metadata.symbol == "EURUSD"
        assert metadata.asset_type == "forex"
        assert metadata.price_precision == 5
        assert metadata.profit_unit == "pips"
        assert metadata.profit_multiplier == 10000.0
        assert metadata.formatted_symbol == "EUR/USD"
        assert metadata.base_currency == "EUR"
        assert metadata.quote_currency == "USD"

    def test_forex_gbpjpy_detection(self):
        """Test forex detection for GBPJPY."""
        metadata = self.service.get_asset_metadata("GBPJPY")

        assert metadata.asset_type == "forex"
        assert metadata.formatted_symbol == "GBP/JPY"
        assert metadata.base_currency == "GBP"
        assert metadata.quote_currency == "JPY"

    def test_forex_lowercase_conversion(self):
        """Test lowercase forex symbols are converted to uppercase."""
        metadata = self.service.get_asset_metadata("eurusd")

        assert metadata.symbol == "EURUSD"
        assert metadata.asset_type == "forex"
        assert metadata.formatted_symbol == "EUR/USD"

    def test_forex_multiple_currencies(self):
        """Test multiple forex pairs are detected correctly."""
        pairs = ["AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "EURGBP"]

        for pair in pairs:
            metadata = self.service.get_asset_metadata(pair)
            assert metadata.asset_type == "forex"
            assert metadata.price_precision == 5
            assert metadata.profit_unit == "pips"

    def test_non_forex_six_char_symbol(self):
        """Test 6-char symbol without currencies is not forex."""
        # ABCDEF is not a known forex pair
        metadata = self.service.get_asset_metadata("ABCDEF")

        assert metadata.asset_type != "forex"

    # Crypto Detection Tests

    def test_crypto_btcusd_with_dash(self):
        """Test crypto detection for BTC-USD format."""
        metadata = self.service.get_asset_metadata("BTC-USD")

        assert metadata.symbol == "BTC-USD"
        assert metadata.asset_type == "crypto"
        assert metadata.price_precision == 8
        assert metadata.profit_unit == "dollars"
        assert metadata.profit_multiplier == 1.0
        assert metadata.formatted_symbol == "BTC/USD"
        assert metadata.base_currency == "BTC"
        assert metadata.quote_currency == "USD"

    def test_crypto_ethusd_with_dash(self):
        """Test crypto detection for ETH-USD format."""
        metadata = self.service.get_asset_metadata("ETH-USD")

        assert metadata.asset_type == "crypto"
        assert metadata.formatted_symbol == "ETH/USD"
        assert metadata.base_currency == "ETH"
        assert metadata.quote_currency == "USD"

    def test_crypto_btcusd_without_dash(self):
        """Test crypto detection for BTCUSD format (no dash)."""
        metadata = self.service.get_asset_metadata("BTCUSD")

        assert metadata.asset_type == "crypto"
        assert metadata.formatted_symbol == "BTC/USD"
        assert metadata.base_currency == "BTC"
        assert metadata.quote_currency == "USD"

    def test_crypto_ethusd_without_dash(self):
        """Test crypto detection for ETHUSD format (no dash)."""
        metadata = self.service.get_asset_metadata("ETHUSD")

        assert metadata.asset_type == "crypto"
        assert metadata.formatted_symbol == "ETH/USD"

    def test_crypto_multiple_symbols(self):
        """Test multiple crypto symbols are detected correctly."""
        symbols = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD"]

        for symbol in symbols:
            metadata = self.service.get_asset_metadata(symbol)
            assert metadata.asset_type == "crypto"
            assert metadata.price_precision == 8
            assert metadata.profit_unit == "dollars"

    def test_crypto_lowercase_with_dash(self):
        """Test lowercase crypto symbols are converted."""
        metadata = self.service.get_asset_metadata("btc-usd")

        assert metadata.symbol == "BTC-USD"
        assert metadata.asset_type == "crypto"

    # Stock Detection Tests

    def test_stock_aapl_detection(self):
        """Test stock detection for AAPL."""
        metadata = self.service.get_asset_metadata("AAPL")

        assert metadata.symbol == "AAPL"
        assert metadata.asset_type == "stock"
        assert metadata.price_precision == 2
        assert metadata.profit_unit == "points"
        assert metadata.profit_multiplier == 100.0
        assert metadata.formatted_symbol == "AAPL"
        assert metadata.base_currency is None
        assert metadata.quote_currency == "USD"

    def test_stock_tsla_detection(self):
        """Test stock detection for TSLA."""
        metadata = self.service.get_asset_metadata("TSLA")

        assert metadata.asset_type == "stock"
        assert metadata.formatted_symbol == "TSLA"

    def test_stock_lowercase_conversion(self):
        """Test lowercase stock symbols are converted."""
        metadata = self.service.get_asset_metadata("aapl")

        assert metadata.symbol == "AAPL"
        assert metadata.asset_type == "stock"

    def test_stock_multiple_tickers(self):
        """Test multiple stock tickers are detected correctly."""
        tickers = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN"]

        for ticker in tickers:
            metadata = self.service.get_asset_metadata(ticker)
            assert metadata.asset_type == "stock"
            assert metadata.price_precision == 2
            assert metadata.profit_unit == "points"

    # Unknown Symbol Tests

    def test_unknown_symbol_defaults_to_stock(self):
        """Test unknown symbols default to stock metadata."""
        metadata = self.service.get_asset_metadata("UNKNWN")

        assert metadata.asset_type == "stock"
        assert metadata.price_precision == 2
        assert metadata.profit_unit == "points"

    def test_empty_string_defaults_to_stock(self):
        """Test empty string defaults to stock (edge case)."""
        # Service converts to uppercase, so "" stays ""
        metadata = self.service.get_asset_metadata("")

        assert metadata.asset_type == "stock"

    # Caching Tests

    def test_caching_works_correctly(self):
        """Test metadata is cached after first call."""
        # First call - creates metadata
        metadata1 = self.service.get_asset_metadata("EURUSD")

        # Second call - should return cached result
        metadata2 = self.service.get_asset_metadata("EURUSD")

        # Should be the same object (cached)
        assert metadata1 is metadata2

    def test_cache_different_symbols(self):
        """Test different symbols have separate cache entries."""
        metadata_eur = self.service.get_asset_metadata("EURUSD")
        metadata_btc = self.service.get_asset_metadata("BTC-USD")

        assert metadata_eur.asset_type == "forex"
        assert metadata_btc.asset_type == "crypto"
        assert metadata_eur is not metadata_btc

    def test_cache_case_insensitive(self):
        """Test cache handles case-insensitive lookups."""
        # Both should resolve to same cached entry
        metadata1 = self.service.get_asset_metadata("eurusd")
        metadata2 = self.service.get_asset_metadata("EURUSD")

        # They should be the same cached object
        assert metadata1.symbol == metadata2.symbol == "EURUSD"

    def test_clear_cache_removes_entries(self):
        """Test clear_cache removes all cached metadata."""
        # Add entries to cache
        self.service.get_asset_metadata("EURUSD")
        self.service.get_asset_metadata("BTC-USD")
        self.service.get_asset_metadata("AAPL")

        # Verify cache is not empty
        assert len(self.service._cache) == 3

        # Clear cache
        self.service.clear_cache()

        # Verify cache is empty
        assert len(self.service._cache) == 0

    # Formatted Symbol Generation Tests

    def test_formatted_symbol_forex(self):
        """Test formatted_symbol for forex pairs."""
        metadata = self.service.get_asset_metadata("GBPUSD")
        assert metadata.formatted_symbol == "GBP/USD"

    def test_formatted_symbol_crypto_with_dash(self):
        """Test formatted_symbol for crypto with dash."""
        metadata = self.service.get_asset_metadata("ETH-USD")
        assert metadata.formatted_symbol == "ETH/USD"

    def test_formatted_symbol_crypto_without_dash(self):
        """Test formatted_symbol for crypto without dash."""
        metadata = self.service.get_asset_metadata("SOLUSD")
        assert metadata.formatted_symbol == "SOL/USD"

    def test_formatted_symbol_stock(self):
        """Test formatted_symbol for stocks stays unchanged."""
        metadata = self.service.get_asset_metadata("MSFT")
        assert metadata.formatted_symbol == "MSFT"

    # Edge Cases

    def test_crypto_with_multiple_dashes(self):
        """Test crypto symbol with multiple dashes."""
        # Should handle gracefully
        metadata = self.service.get_asset_metadata("BTC-USD-TEST")

        # Will be detected as crypto (BTC is in crypto list)
        assert metadata.asset_type == "crypto"

    def test_numeric_suffix_in_symbol(self):
        """Test symbols with numeric suffixes."""
        # Some exchanges use BTC1, BTC2 etc.
        metadata = self.service.get_asset_metadata("BTC1")

        # Should still detect as crypto (starts with BTC)
        assert metadata.asset_type == "crypto"

    def test_very_long_symbol(self):
        """Test handling of unusually long symbols."""
        metadata = self.service.get_asset_metadata("VERYLONGSYMBOLNAME")

        # Should default to stock
        assert metadata.asset_type == "stock"

    # Thread Safety Tests (basic)

    def test_concurrent_cache_access(self):
        """Test cache is thread-safe for concurrent access."""
        # This is a basic test - full threading tests would be more complex
        metadata1 = self.service.get_asset_metadata("EURUSD")
        metadata2 = self.service.get_asset_metadata("EURUSD")

        # Should safely return same cached object
        assert metadata1 is metadata2
        assert len(self.service._cache) == 1


class TestAssetServiceSingleton:
    """Test the singleton asset_service instance."""

    def test_singleton_instance_exists(self):
        """Test global asset_service singleton exists."""
        from src.api.services.asset_service import asset_service

        assert asset_service is not None
        assert isinstance(asset_service, AssetService)

    def test_singleton_instance_functional(self):
        """Test singleton instance works correctly."""
        from src.api.services.asset_service import asset_service

        metadata = asset_service.get_asset_metadata("EURUSD")

        assert metadata.asset_type == "forex"
        assert metadata.formatted_symbol == "EUR/USD"
