---
name: adding-data-sources
description: Adds new data source connectors following the BaseDataSource abstract class pattern with factory registration. Use when integrating new brokers (Alpaca, IBKR), data providers (Yahoo, Polygon), or exchange APIs. Python async pattern.
---

# Adding Data Sources

## Quick Reference

- Inherit from `BaseDataSource` in `src/data/sources/base.py`
- Implement 5 abstract methods: `connect()`, `disconnect()`, `fetch_ohlcv()`, `get_available_symbols()`, `get_current_price()`
- Register with `DataSourceFactory.register("name", ClassName)`
- Support context manager: `__enter__` and `__exit__` provided by base class
- Track connection state with `self._connected`

## When to Use

- Adding a new broker API (Alpaca, Interactive Brokers, TD Ameritrade)
- Integrating market data providers (Yahoo Finance, Polygon, Alpha Vantage)
- Connecting to exchange APIs (Binance, Coinbase, MT5)
- Creating mock/test data sources
- Building custom data adapters

## When NOT to Use

- CSV file loading (use pandas directly)
- One-time data downloads (use scripts)
- Real-time streaming (different pattern needed)

## Implementation Guide with Decision Tree

```
What type of data source?
├─ REST API → Implement with requests/httpx
│   └─ Needs auth? → Store in config, load from .env
├─ Native SDK → Import broker's Python SDK
│   └─ MT5 → MetaTrader5 package (Windows only)
│   └─ Alpaca → alpaca-py package
└─ WebSocket → Not covered by this pattern

Connection management:
├─ Persistent connection → Set _connected in connect()
├─ Per-request → Set _connected = True always
└─ Context manager → Use `with source:` pattern
```

## Examples

**Example 1: BaseDataSource Abstract Class**

```python
# From: src/data/sources/base.py:1-74
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
```

**Explanation**: Base class defines contract. Context manager support built-in. `_connected` tracks state.

**Example 2: DataSourceFactory Pattern**

```python
# From: src/data/sources/base.py:76-99
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
```

**Explanation**: Factory enables creation by name. Names are case-insensitive. Register at module end.

**Example 3: Yahoo Finance Implementation**

```python
# From: src/data/sources/yahoo.py (structure)
"""Yahoo Finance data source."""

from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
import yfinance as yf

from .base import BaseDataSource, DataSourceFactory


class YahooDataSource(BaseDataSource):
    """Yahoo Finance data source using yfinance library."""

    TIMEFRAME_MAP = {
        "1M": "1m",
        "5M": "5m",
        "15M": "15m",
        "30M": "30m",
        "1H": "1h",
        "1D": "1d",
        "1W": "1wk",
        "1MO": "1mo",
    }

    def connect(self) -> bool:
        """Yahoo doesn't require explicit connection."""
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Yahoo doesn't require explicit disconnection."""
        self._connected = False

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        yf_interval = self.TIMEFRAME_MAP.get(timeframe.upper())
        if yf_interval is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=yf_interval,
        )

        # Standardize column names
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]]

        return df

    def get_available_symbols(self) -> list:
        """Return common forex pairs."""
        return ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]

    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """Get current price from Yahoo."""
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "bid": info.get("bid", 0),
            "ask": info.get("ask", 0),
            "last": info.get("regularMarketPrice", 0),
        }


# Register data source
DataSourceFactory.register("yahoo", YahooDataSource)
```

**Explanation**: Yahoo is stateless, so connect/disconnect just track state. Timeframe mapping converts our format to yfinance format. Standardize column names to lowercase.

**Example 4: Alpaca Implementation**

```python
# From: src/data/sources/alpaca.py (structure)
"""Alpaca Markets data source."""

from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .base import BaseDataSource, DataSourceFactory


class AlpacaDataSource(BaseDataSource):
    """Alpaca Markets data source."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.api_key = self.config.get("api_key")
        self.secret_key = self.config.get("secret_key")
        self.client = None

    def connect(self) -> bool:
        """Initialize Alpaca client."""
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API key and secret required in config")

        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Clear client reference."""
        self.client = None
        self._connected = False

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch stock bars from Alpaca."""
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        tf_map = {
            "1M": TimeFrame.Minute,
            "1H": TimeFrame.Hour,
            "1D": TimeFrame.Day,
        }

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf_map.get(timeframe.upper(), TimeFrame.Day),
            start=start_date,
            end=end_date,
        )

        bars = self.client.get_stock_bars(request)
        df = bars.df

        # Standardize column names
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]


# Register data source
DataSourceFactory.register("alpaca", AlpacaDataSource)
```

**Explanation**: Alpaca requires API credentials from config. Client created in connect(). Check `_connected` before operations.

**Example 5: Context Manager Usage**

```python
# Usage pattern
from src.data.sources import DataSourceFactory

# Create source by name
source = DataSourceFactory.create("yahoo")

# Use as context manager
with source:
    df = source.fetch_ohlcv(
        symbol="EURUSD=X",
        timeframe="1D",
        start_date=datetime(2024, 1, 1),
    )
    print(f"Fetched {len(df)} rows")

# Or manual connect/disconnect
source = DataSourceFactory.create("alpaca", {
    "api_key": "YOUR_KEY",
    "secret_key": "YOUR_SECRET",
})
source.connect()
try:
    df = source.fetch_ohlcv("AAPL", "1D", datetime(2024, 1, 1))
finally:
    source.disconnect()
```

**Explanation**: Context manager is preferred. Manual connect/disconnect needs try/finally for cleanup.

## Quality Checklist

- [ ] Class inherits from `BaseDataSource`
- [ ] All 5 abstract methods implemented
- [ ] `_connected` flag managed correctly
- [ ] Credentials loaded from config (not hardcoded)
- [ ] OHLCV DataFrame has standardized columns: [open, high, low, close, volume]
- [ ] OHLCV DataFrame has DatetimeIndex
- [ ] Registered with `DataSourceFactory.register()` at module end
- [ ] Timeframe mapping from our format to API format

## Common Mistakes

- **Hardcoded credentials**: Security risk → Load from config, set via .env
- **Missing connection check**: RuntimeError → Check `_connected` in fetch methods
- **Non-standard columns**: Processing fails → Lowercase and standardize to [open, high, low, close, volume]
- **Missing index**: Time series issues → Ensure DatetimeIndex

## Validation

- [ ] Pattern confirmed in `src/data/sources/base.py:1-99`
- [ ] Yahoo implementation in `src/data/sources/yahoo.py`
- [ ] Alpaca implementation in `src/data/sources/alpaca.py`

## Related Skills

- [creating-data-processors](../backend/creating-data-processors.md) - For processing fetched data
- [creating-technical-indicators](../feature-engineering/creating-technical-indicators.md) - For adding indicators to fetched data
