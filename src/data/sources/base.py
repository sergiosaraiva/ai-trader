"""Base class for data sources."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, Any, Type

import pandas as pd


class BaseDataSource(ABC):
    """Abstract base class for all data sources."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data source with optional configuration."""
        self.config = config or {}
        self._connected = False

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to data source."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to data source."""
        pass

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'AAPL')
            timeframe: Candle timeframe (e.g., '1H', '1D', '1W')
            start_date: Start date for data
            end_date: End date for data (defaults to now)

        Returns:
            DataFrame with columns: [open, high, low, close, volume]
            Index: datetime
        """
        pass

    @abstractmethod
    def get_available_symbols(self) -> list:
        """Get list of available trading symbols."""
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """Get current bid/ask price for a symbol."""
        pass

    @property
    def is_connected(self) -> bool:
        """Check if connected to data source."""
        return self._connected

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()


class DataSourceFactory:
    """Factory for creating data source instances."""

    _sources: Dict[str, Type[BaseDataSource]] = {}

    @classmethod
    def register(cls, name: str, source_class: Type[BaseDataSource]) -> None:
        """Register a data source class."""
        cls._sources[name.lower()] = source_class

    @classmethod
    def create(cls, name: str, config: Optional[Dict[str, Any]] = None) -> BaseDataSource:
        """Create a data source instance by name."""
        source_class = cls._sources.get(name.lower())
        if source_class is None:
            available = ", ".join(cls._sources.keys())
            raise ValueError(f"Unknown data source: {name}. Available: {available}")
        return source_class(config)

    @classmethod
    def available_sources(cls) -> list:
        """Get list of available data source names."""
        return list(cls._sources.keys())
