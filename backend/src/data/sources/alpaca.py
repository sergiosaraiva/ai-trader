"""Alpaca Markets data source connector."""

from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd

from .base import BaseDataSource, DataSourceFactory


class AlpacaDataSource(BaseDataSource):
    """Alpaca Markets data source for stocks and crypto."""

    TIMEFRAME_MAP = {
        "1M": "1Min",
        "5M": "5Min",
        "15M": "15Min",
        "30M": "30Min",
        "1H": "1Hour",
        "4H": "4Hour",
        "1D": "1Day",
        "1W": "1Week",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Alpaca data source."""
        super().__init__(config)
        self._api = None
        self._data_client = None

    def connect(self) -> bool:
        """Connect to Alpaca API."""
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient

            api_key = self.config.get("api_key")
            secret_key = self.config.get("secret_key")
            paper = self.config.get("paper", True)

            if not api_key or not secret_key:
                raise ValueError("Alpaca API key and secret required")

            self._api = TradingClient(api_key, secret_key, paper=paper)
            self._data_client = StockHistoricalDataClient(api_key, secret_key)
            self._connected = True
            return True

        except ImportError:
            raise ImportError(
                "alpaca-py package not installed. Install with: pip install alpaca-py"
            )

    def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        self._api = None
        self._data_client = None
        self._connected = False

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Alpaca."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        tf_str = self.TIMEFRAME_MAP.get(timeframe.upper())
        if tf_str is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        # Parse timeframe
        if "Min" in tf_str:
            amount = int(tf_str.replace("Min", ""))
            tf = TimeFrame(amount, TimeFrameUnit.Minute)
        elif "Hour" in tf_str:
            amount = int(tf_str.replace("Hour", ""))
            tf = TimeFrame(amount, TimeFrameUnit.Hour)
        elif "Day" in tf_str:
            tf = TimeFrame(1, TimeFrameUnit.Day)
        elif "Week" in tf_str:
            tf = TimeFrame(1, TimeFrameUnit.Week)
        else:
            raise ValueError(f"Cannot parse timeframe: {tf_str}")

        end_date = end_date or datetime.now()

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start_date,
            end=end_date,
        )

        bars = self._data_client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        # Reset multi-index if present
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index(level=0, drop=True)

        df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            },
            inplace=True,
        )

        return df[["open", "high", "low", "close", "volume"]]

    def get_available_symbols(self) -> List[str]:
        """Get available symbols from Alpaca."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        assets = self._api.get_all_assets()
        return [a.symbol for a in assets if a.tradable]

    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """Get current price from Alpaca."""
        if not self._connected:
            raise ConnectionError("Not connected to Alpaca")

        from alpaca.data.requests import StockLatestQuoteRequest

        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = self._data_client.get_stock_latest_quote(request)[symbol]

        return {
            "bid": quote.bid_price,
            "ask": quote.ask_price,
            "last": (quote.bid_price + quote.ask_price) / 2,
            "time": quote.timestamp,
        }


# Register with factory
DataSourceFactory.register("alpaca", AlpacaDataSource)
