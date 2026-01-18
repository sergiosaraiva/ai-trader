"""Service for asset metadata detection and management.

This service provides asset-specific metadata (precision, profit units, formatting)
based on symbol patterns and asset type detection.
"""

import logging
import re
from threading import Lock
from typing import Dict, Optional

from ..schemas.asset import AssetMetadata

logger = logging.getLogger(__name__)

# Maximum number of cached asset metadata entries
MAX_CACHE_SIZE = 100


class AssetService:
    """Service for detecting asset types and providing metadata.

    Uses singleton pattern - metadata is cached and shared across requests.
    Provides thread-safe asset detection with pattern matching.
    """

    def __init__(self):
        self._lock = Lock()
        self._cache: Dict[str, AssetMetadata] = {}

        # Common currency codes for forex detection
        self._forex_currencies = {
            "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD",
            "SEK", "NOK", "DKK", "PLN", "HUF", "CZK", "TRY", "ZAR",
        }

        # Common crypto symbols
        self._crypto_symbols = {
            "BTC", "ETH", "USDT", "BNB", "SOL", "XRP", "ADA", "DOGE",
            "MATIC", "DOT", "AVAX", "LINK", "UNI", "ATOM", "LTC",
        }

    def get_asset_metadata(self, symbol: str) -> AssetMetadata:
        """Get metadata for a trading symbol.

        Detects asset type based on symbol pattern and returns appropriate
        formatting information. Results are cached for performance.

        Args:
            symbol: Trading symbol (e.g., EURUSD, BTC-USD, AAPL)

        Returns:
            AssetMetadata with type-specific formatting

        Examples:
            >>> service.get_asset_metadata("EURUSD")
            AssetMetadata(asset_type="forex", profit_unit="pips", ...)

            >>> service.get_asset_metadata("BTC-USD")
            AssetMetadata(asset_type="crypto", profit_unit="dollars", ...)
        """
        # Check cache first
        with self._lock:
            if symbol in self._cache:
                return self._cache[symbol]

            # Limit cache size (FIFO eviction)
            if len(self._cache) >= MAX_CACHE_SIZE:
                # Remove oldest entry (first inserted)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"Cache full, evicted {oldest_key}")

        # Detect asset type and create metadata (outside lock)
        metadata = self._detect_and_create_metadata(symbol)

        # Cache result
        with self._lock:
            self._cache[symbol] = metadata

        logger.debug(f"Created metadata for {symbol}: {metadata.asset_type}")
        return metadata

    def _detect_and_create_metadata(self, symbol: str) -> AssetMetadata:
        """Detect asset type and create appropriate metadata.

        Detection logic:
        1. Forex: 6-char symbol with recognized currency codes (EURUSD)
        2. Crypto: Symbol contains known crypto codes or has dash separator
        3. Stock: Default fallback

        Args:
            symbol: Trading symbol

        Returns:
            AssetMetadata with detected type and formatting
        """
        try:
            symbol_upper = symbol.upper()

            # Forex detection: 6-character pairs like EURUSD
            if self._is_forex(symbol_upper):
                return self._create_forex_metadata(symbol_upper)

            # Crypto detection: BTC-USD, ETH-USD, or contains crypto symbol
            if self._is_crypto(symbol_upper):
                return self._create_crypto_metadata(symbol_upper)

            # Default to stock
            return self._create_stock_metadata(symbol_upper)
        except Exception as e:
            logger.error(f"Error detecting asset type for {symbol}: {e}")
            return self._create_default_metadata(symbol)

    def _is_forex(self, symbol: str) -> bool:
        """Check if symbol is a forex pair.

        Args:
            symbol: Trading symbol (uppercase)

        Returns:
            True if detected as forex pair
        """
        # Forex pairs are typically 6 characters (EURUSD)
        if len(symbol) != 6:
            return False

        # Check if both base and quote are known currencies
        base = symbol[:3]
        quote = symbol[3:]

        return base in self._forex_currencies and quote in self._forex_currencies

    def _is_crypto(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency.

        Args:
            symbol: Trading symbol (uppercase)

        Returns:
            True if detected as crypto
        """
        # Check for dash separator (BTC-USD)
        if "-" in symbol:
            parts = symbol.split("-")
            if len(parts) == 2 and parts[0] in self._crypto_symbols:
                return True

        # Check for known crypto symbols
        for crypto in self._crypto_symbols:
            if symbol.startswith(crypto):
                return True

        return False

    def _create_forex_metadata(self, symbol: str) -> AssetMetadata:
        """Create metadata for forex pair.

        Args:
            symbol: Forex symbol (e.g., EURUSD)

        Returns:
            AssetMetadata with forex-specific settings
        """
        base = symbol[:3]
        quote = symbol[3:]

        return AssetMetadata(
            symbol=symbol,
            asset_type="forex",
            price_precision=5,
            profit_unit="pips",
            profit_multiplier=10000.0,
            formatted_symbol=f"{base}/{quote}",
            base_currency=base,
            quote_currency=quote,
        )

    def _create_crypto_metadata(self, symbol: str) -> AssetMetadata:
        """Create metadata for cryptocurrency.

        Args:
            symbol: Crypto symbol (e.g., BTC-USD, BTCUSD)

        Returns:
            AssetMetadata with crypto-specific settings
        """
        # Parse base and quote from symbol
        if "-" in symbol:
            parts = symbol.split("-")
            base = parts[0]
            quote = parts[1] if len(parts) > 1 else "USD"
            formatted = f"{base}/{quote}"
        else:
            # Try to extract crypto prefix
            base = None
            for crypto in self._crypto_symbols:
                if symbol.startswith(crypto):
                    base = crypto
                    quote = symbol[len(crypto):] or "USD"
                    break

            if not base:
                base = symbol
                quote = "USD"

            formatted = f"{base}/{quote}"

        return AssetMetadata(
            symbol=symbol,
            asset_type="crypto",
            price_precision=8,
            profit_unit="dollars",
            profit_multiplier=1.0,
            formatted_symbol=formatted,
            base_currency=base,
            quote_currency=quote,
        )

    def _create_stock_metadata(self, symbol: str) -> AssetMetadata:
        """Create metadata for stock.

        Args:
            symbol: Stock symbol (e.g., AAPL, TSLA)

        Returns:
            AssetMetadata with stock-specific settings
        """
        return AssetMetadata(
            symbol=symbol,
            asset_type="stock",
            price_precision=2,
            profit_unit="points",
            profit_multiplier=100.0,
            formatted_symbol=symbol,
            base_currency=None,
            quote_currency="USD",
        )

    def _create_default_metadata(self, symbol: str) -> AssetMetadata:
        """Create default metadata when detection fails.

        Args:
            symbol: Trading symbol

        Returns:
            AssetMetadata with safe default settings
        """
        return AssetMetadata(
            symbol=symbol,
            asset_type="unknown",
            price_precision=5,
            profit_unit="points",
            profit_multiplier=1.0,
            formatted_symbol=symbol,
            base_currency=None,
            quote_currency=None,
        )

    def clear_cache(self) -> None:
        """Clear the metadata cache.

        Useful for testing or when asset definitions change.
        """
        with self._lock:
            self._cache.clear()
        logger.info("Asset metadata cache cleared")


# Singleton instance
asset_service = AssetService()
