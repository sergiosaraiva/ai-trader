"""Yahoo Finance data source connector."""

from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd

from .base import BaseDataSource, DataSourceFactory


class YahooDataSource(BaseDataSource):
    """Yahoo Finance data source for stocks and forex."""

    TIMEFRAME_MAP = {
        "1M": "1m",
        "5M": "5m",
        "15M": "15m",
        "30M": "30m",
        "1H": "1h",
        "4H": "4h",
        "1D": "1d",
        "1W": "1wk",
        "1MO": "1mo",
    }

    # Forex symbols need =X suffix in Yahoo
    FOREX_PAIRS = [
        "EURUSD", "GBPUSD", "USDJPY", "USDCHF",
        "AUDUSD", "USDCAD", "NZDUSD", "EURGBP",
        "EURJPY", "GBPJPY", "AUDJPY", "CADJPY",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Yahoo Finance data source."""
        super().__init__(config)
        self._yf = None

    def connect(self) -> bool:
        """Connect to Yahoo Finance (no auth required)."""
        try:
            import yfinance as yf

            self._yf = yf
            self._connected = True
            return True

        except ImportError:
            raise ImportError(
                "yfinance package not installed. Install with: pip install yfinance"
            )

    def disconnect(self) -> None:
        """Disconnect from Yahoo Finance."""
        self._yf = None
        self._connected = False

    def _convert_symbol(self, symbol: str) -> str:
        """Convert symbol to Yahoo Finance format."""
        # Forex pairs need =X suffix
        if symbol.upper() in self.FOREX_PAIRS:
            return f"{symbol.upper()}=X"
        return symbol

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        if not self._connected:
            raise ConnectionError("Not connected to Yahoo Finance")

        interval = self.TIMEFRAME_MAP.get(timeframe.upper())
        if interval is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        yf_symbol = self._convert_symbol(symbol)
        end_date = end_date or datetime.now()

        ticker = self._yf.Ticker(yf_symbol)
        df = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval,
        )

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            inplace=True,
        )

        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)  # Remove timezone

        return df[["open", "high", "low", "close", "volume"]]

    def get_available_symbols(self) -> List[str]:
        """Get commonly used forex symbols."""
        # Yahoo doesn't have a symbol list endpoint
        return self.FOREX_PAIRS.copy()

    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """Get current price from Yahoo Finance."""
        if not self._connected:
            raise ConnectionError("Not connected to Yahoo Finance")

        yf_symbol = self._convert_symbol(symbol)
        ticker = self._yf.Ticker(yf_symbol)
        info = ticker.info

        # Yahoo provides last price, not bid/ask for most symbols
        last_price = info.get("regularMarketPrice", info.get("previousClose", 0))
        bid = info.get("bid", last_price)
        ask = info.get("ask", last_price)

        return {
            "bid": bid if bid else last_price,
            "ask": ask if ask else last_price,
            "last": last_price,
            "time": datetime.now(),
        }


# Register with factory
DataSourceFactory.register("yahoo", YahooDataSource)
