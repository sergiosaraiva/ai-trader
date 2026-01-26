"""
Market Simulator for Backtesting.

Provides market data replay and simulation for backtesting trading strategies.
Supports event-driven data streaming and realistic market conditions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Iterator, Callable
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class MarketStatus(Enum):
    """Market status enumeration."""
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    OPEN = "open"
    POST_MARKET = "post_market"


@dataclass
class MarketBar:
    """
    Single market data bar.

    Represents OHLCV data for a single time period.
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    # Optional technical data
    atr: Optional[float] = None
    spread: Optional[float] = None

    # Metadata
    bar_index: int = 0
    timeframe: str = "1D"

    @property
    def mid_price(self) -> float:
        """Get mid price (OHLC average)."""
        return (self.open + self.high + self.low + self.close) / 4

    @property
    def typical_price(self) -> float:
        """Get typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> float:
        """Get bar range."""
        return self.high - self.low

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "atr": self.atr,
            "spread": self.spread,
            "bar_index": self.bar_index,
            "timeframe": self.timeframe,
        }


@dataclass
class MarketSnapshot:
    """
    Market snapshot at a point in time.

    Contains current prices and market data for multiple symbols.
    """
    timestamp: datetime
    bars: Dict[str, MarketBar] = field(default_factory=dict)
    market_status: MarketStatus = MarketStatus.OPEN

    def get_price(self, symbol: str) -> Optional[float]:
        """Get close price for symbol."""
        bar = self.bars.get(symbol)
        return bar.close if bar else None

    def get_bar(self, symbol: str) -> Optional[MarketBar]:
        """Get bar for symbol."""
        return self.bars.get(symbol)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "market_status": self.market_status.value,
            "bars": {s: b.to_dict() for s, b in self.bars.items()},
        }


@dataclass
class MarketSession:
    """Market trading session configuration."""
    name: str
    open_time: str  # HH:MM format
    close_time: str  # HH:MM format
    timezone: str = "UTC"
    trading_days: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon-Fri

    def is_open(self, dt: datetime) -> bool:
        """Check if market is open at given time."""
        # Check trading day
        if dt.weekday() not in self.trading_days:
            return False

        # Parse times
        open_h, open_m = map(int, self.open_time.split(":"))
        close_h, close_m = map(int, self.close_time.split(":"))

        current_minutes = dt.hour * 60 + dt.minute
        open_minutes = open_h * 60 + open_m
        close_minutes = close_h * 60 + close_m

        # Handle overnight sessions
        if close_minutes < open_minutes:
            return current_minutes >= open_minutes or current_minutes < close_minutes
        else:
            return open_minutes <= current_minutes < close_minutes


# Default forex session (24/5)
FOREX_SESSION = MarketSession(
    name="Forex",
    open_time="00:00",
    close_time="23:59",
    trading_days=[0, 1, 2, 3, 4],  # Mon-Fri
)

# US stock market session
US_STOCK_SESSION = MarketSession(
    name="US_Stock",
    open_time="09:30",
    close_time="16:00",
    timezone="America/New_York",
    trading_days=[0, 1, 2, 3, 4],
)


class MarketSimulator:
    """
    Market simulator for backtesting.

    Features:
    - Historical data replay
    - Event-driven bar generation
    - Multiple symbol support
    - Market session handling
    - ATR and spread calculation
    """

    def __init__(
        self,
        session: Optional[MarketSession] = None,
        atr_period: int = 14,
        default_spread_pips: float = 1.0,
    ):
        """
        Initialize market simulator.

        Args:
            session: Market session configuration
            atr_period: Period for ATR calculation
            default_spread_pips: Default spread in pips
        """
        self.session = session or FOREX_SESSION
        self.atr_period = atr_period
        self.default_spread_pips = default_spread_pips

        # Data storage
        self._data: Dict[str, pd.DataFrame] = {}
        self._features: Dict[str, pd.DataFrame] = {}

        # Simulation state
        self._current_index: int = 0
        self._start_index: int = 0
        self._end_index: int = 0
        self._current_time: Optional[datetime] = None
        self._is_running: bool = False

        # Subscribers for market events
        self._on_bar_callbacks: List[Callable[[MarketSnapshot], None]] = []

    def load_data(
        self,
        symbol: str,
        data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Load historical data for a symbol.

        Args:
            symbol: Trading symbol
            data: OHLCV DataFrame with DatetimeIndex
            features: Optional pre-computed features
        """
        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if "timestamp" in data.columns:
                data = data.set_index("timestamp")
            elif "date" in data.columns:
                data = data.set_index("date")
            data.index = pd.to_datetime(data.index)

        # Standardize column names
        data.columns = [c.lower() for c in data.columns]

        # Calculate ATR if not present
        if "atr" not in data.columns:
            data["atr"] = self._calculate_atr(data)

        self._data[symbol] = data.sort_index()

        if features is not None:
            self._features[symbol] = features.sort_index()

        logger.info(f"Loaded {len(data)} bars for {symbol}")

    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Average True Range."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean()

        return atr

    def start(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        """
        Start simulation from specified time.

        Args:
            start_time: Start timestamp (default: first available)
            end_time: End timestamp (default: last available)
        """
        if not self._data:
            raise ValueError("No data loaded. Call load_data() first.")

        # Determine common time range across all symbols
        min_time = max(df.index.min() for df in self._data.values())
        max_time = min(df.index.max() for df in self._data.values())

        start_time = start_time or min_time
        end_time = end_time or max_time

        # Ensure start_time is within range
        start_time = max(start_time, min_time)
        end_time = min(end_time, max_time)

        # Get reference data (first symbol)
        ref_symbol = list(self._data.keys())[0]
        ref_data = self._data[ref_symbol]

        # Find indices
        self._start_index = ref_data.index.get_indexer([start_time], method="bfill")[0]
        self._end_index = ref_data.index.get_indexer([end_time], method="ffill")[0]

        self._current_index = self._start_index
        self._current_time = ref_data.index[self._current_index]
        self._is_running = True

        logger.info(
            f"Simulation started: {start_time} to {end_time} "
            f"({self._end_index - self._start_index + 1} bars)"
        )

    def stop(self) -> None:
        """Stop the simulation."""
        self._is_running = False
        logger.info("Simulation stopped")

    def reset(self) -> None:
        """Reset simulation to start."""
        self._current_index = self._start_index
        ref_symbol = list(self._data.keys())[0]
        self._current_time = self._data[ref_symbol].index[self._current_index]
        self._is_running = True

    def advance(self, bars: int = 1) -> Optional[MarketSnapshot]:
        """
        Advance simulation by N bars.

        Args:
            bars: Number of bars to advance

        Returns:
            MarketSnapshot at new position, or None if at end
        """
        if not self._is_running:
            return None

        new_index = self._current_index + bars

        if new_index > self._end_index:
            self._is_running = False
            return None

        self._current_index = new_index

        # Get current time from reference data
        ref_symbol = list(self._data.keys())[0]
        self._current_time = self._data[ref_symbol].index[self._current_index]

        snapshot = self.get_current_snapshot()

        # Notify subscribers
        for callback in self._on_bar_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Error in bar callback: {e}")

        return snapshot

    def get_current_snapshot(self) -> MarketSnapshot:
        """
        Get current market snapshot.

        Returns:
            MarketSnapshot with current bars for all symbols
        """
        bars = {}

        for symbol, data in self._data.items():
            if self._current_index < len(data):
                row = data.iloc[self._current_index]

                bars[symbol] = MarketBar(
                    symbol=symbol,
                    timestamp=self._current_time,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row.get("volume", 0),
                    atr=row.get("atr"),
                    spread=row.get("spread", self.default_spread_pips * 0.0001),
                    bar_index=self._current_index,
                )

        market_status = (
            MarketStatus.OPEN if self.session.is_open(self._current_time)
            else MarketStatus.CLOSED
        )

        return MarketSnapshot(
            timestamp=self._current_time,
            bars=bars,
            market_status=market_status,
        )

    def get_current_bar(self, symbol: str) -> Optional[MarketBar]:
        """Get current bar for a symbol."""
        snapshot = self.get_current_snapshot()
        return snapshot.get_bar(symbol)

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current close price for a symbol."""
        bar = self.get_current_bar(symbol)
        return bar.close if bar else None

    def get_features(self, symbol: str) -> Optional[pd.Series]:
        """
        Get features for current bar.

        Args:
            symbol: Trading symbol

        Returns:
            Feature values as Series, or None
        """
        if symbol not in self._features:
            return None

        features = self._features[symbol]
        if self._current_index < len(features):
            return features.iloc[self._current_index]

        return None

    def get_historical_bars(
        self,
        symbol: str,
        n_bars: int,
    ) -> List[MarketBar]:
        """
        Get historical bars up to current position.

        Args:
            symbol: Trading symbol
            n_bars: Number of bars to retrieve

        Returns:
            List of MarketBar objects
        """
        if symbol not in self._data:
            return []

        data = self._data[symbol]
        start_idx = max(0, self._current_index - n_bars + 1)
        end_idx = self._current_index + 1

        bars = []
        for i in range(start_idx, end_idx):
            row = data.iloc[i]
            bars.append(MarketBar(
                symbol=symbol,
                timestamp=data.index[i],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row.get("volume", 0),
                atr=row.get("atr"),
                bar_index=i,
            ))

        return bars

    def get_historical_data(
        self,
        symbol: str,
        n_bars: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get historical DataFrame up to current position.

        Args:
            symbol: Trading symbol
            n_bars: Number of bars (default: all history)

        Returns:
            DataFrame with historical data
        """
        if symbol not in self._data:
            return pd.DataFrame()

        data = self._data[symbol]
        end_idx = self._current_index + 1

        if n_bars:
            start_idx = max(0, end_idx - n_bars)
        else:
            start_idx = 0

        return data.iloc[start_idx:end_idx].copy()

    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        if self._current_time is None:
            return False
        return self.session.is_open(self._current_time)

    def on_bar(self, callback: Callable[[MarketSnapshot], None]) -> None:
        """
        Register callback for bar events.

        Args:
            callback: Function to call on each new bar
        """
        self._on_bar_callbacks.append(callback)

    def iter_bars(self) -> Iterator[MarketSnapshot]:
        """
        Iterate through all bars.

        Yields:
            MarketSnapshot for each bar
        """
        # Start if not already started
        if self._end_index == 0:
            self.start()
        else:
            self.reset()

        while self._is_running:
            snapshot = self.get_current_snapshot()
            yield snapshot

            if self.advance() is None:
                break

    @property
    def current_time(self) -> Optional[datetime]:
        """Get current simulation time."""
        return self._current_time

    @property
    def current_index(self) -> int:
        """Get current bar index."""
        return self._current_index

    @property
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._is_running

    @property
    def progress(self) -> float:
        """Get simulation progress (0-1)."""
        total_bars = self._end_index - self._start_index + 1
        current_bar = self._current_index - self._start_index
        return current_bar / total_bars if total_bars > 0 else 0

    @property
    def remaining_bars(self) -> int:
        """Get number of remaining bars."""
        return self._end_index - self._current_index

    @property
    def symbols(self) -> List[str]:
        """Get list of loaded symbols."""
        return list(self._data.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get simulator statistics."""
        return {
            "symbols": self.symbols,
            "is_running": self._is_running,
            "current_time": self._current_time.isoformat() if self._current_time else None,
            "current_index": self._current_index,
            "progress": self.progress,
            "remaining_bars": self.remaining_bars,
            "total_bars": self._end_index - self._start_index + 1,
        }
