"""MetaTrader 5 data source connector."""

from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd

from .base import BaseDataSource, DataSourceFactory


class MT5DataSource(BaseDataSource):
    """MetaTrader 5 data source for forex data."""

    TIMEFRAME_MAP = {
        "1M": "TIMEFRAME_M1",
        "5M": "TIMEFRAME_M5",
        "15M": "TIMEFRAME_M15",
        "30M": "TIMEFRAME_M30",
        "1H": "TIMEFRAME_H1",
        "4H": "TIMEFRAME_H4",
        "1D": "TIMEFRAME_D1",
        "1W": "TIMEFRAME_W1",
        "1MO": "TIMEFRAME_MN1",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize MT5 data source."""
        super().__init__(config)
        self._mt5 = None

    def connect(self) -> bool:
        """Connect to MetaTrader 5 terminal."""
        try:
            import MetaTrader5 as mt5

            self._mt5 = mt5

            login = self.config.get("login")
            password = self.config.get("password")
            server = self.config.get("server")

            if login and password and server:
                initialized = mt5.initialize(
                    login=login,
                    password=password,
                    server=server,
                )
            else:
                initialized = mt5.initialize()

            if not initialized:
                error = mt5.last_error()
                raise ConnectionError(f"MT5 initialization failed: {error}")

            self._connected = True
            return True

        except ImportError:
            raise ImportError(
                "MetaTrader5 package not installed. Install with: pip install MetaTrader5"
            )

    def disconnect(self) -> None:
        """Disconnect from MetaTrader 5."""
        if self._mt5 and self._connected:
            self._mt5.shutdown()
            self._connected = False

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from MT5."""
        if not self._connected:
            raise ConnectionError("Not connected to MT5")

        tf_name = self.TIMEFRAME_MAP.get(timeframe.upper())
        if tf_name is None:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        tf = getattr(self._mt5, tf_name)
        end_date = end_date or datetime.now()

        rates = self._mt5.copy_rates_range(symbol, tf, start_date, end_date)

        if rates is None or len(rates) == 0:
            raise ValueError(f"No data returned for {symbol}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "tick_volume": "volume",
            },
            inplace=True,
        )

        return df[["open", "high", "low", "close", "volume"]]

    def get_available_symbols(self) -> List[str]:
        """Get available symbols from MT5."""
        if not self._connected:
            raise ConnectionError("Not connected to MT5")

        symbols = self._mt5.symbols_get()
        return [s.name for s in symbols] if symbols else []

    def get_current_price(self, symbol: str) -> Dict[str, float]:
        """Get current bid/ask price."""
        if not self._connected:
            raise ConnectionError("Not connected to MT5")

        tick = self._mt5.symbol_info_tick(symbol)
        if tick is None:
            raise ValueError(f"Cannot get price for {symbol}")

        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "time": datetime.fromtimestamp(tick.time),
        }


# Register with factory
DataSourceFactory.register("mt5", MT5DataSource)
