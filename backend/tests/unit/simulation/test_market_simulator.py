"""Tests for Market Simulator."""

import pytest
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from src.simulation.market_simulator import (
    MarketSimulator,
    MarketBar,
    MarketSnapshot,
    MarketSession,
    MarketStatus,
    FOREX_SESSION,
    US_STOCK_SESSION,
)


class TestMarketBar:
    """Tests for MarketBar dataclass."""

    def test_bar_creation(self):
        """Test bar creation."""
        bar = MarketBar(
            symbol="EURUSD",
            timestamp=datetime.now(),
            open=1.1000,
            high=1.1050,
            low=1.0950,
            close=1.1020,
            volume=10000,
        )

        assert bar.symbol == "EURUSD"
        assert bar.open == 1.1000
        assert bar.close == 1.1020

    def test_mid_price(self):
        """Test mid price calculation."""
        bar = MarketBar(
            symbol="EURUSD",
            timestamp=datetime.now(),
            open=1.1000,
            high=1.1100,
            low=1.0900,
            close=1.1000,
        )

        # (1.1 + 1.11 + 1.09 + 1.1) / 4 = 1.1
        assert bar.mid_price == pytest.approx(1.1, rel=0.01)

    def test_typical_price(self):
        """Test typical price calculation."""
        bar = MarketBar(
            symbol="EURUSD",
            timestamp=datetime.now(),
            open=1.1000,
            high=1.1100,
            low=1.0900,
            close=1.1000,
        )

        # (1.11 + 1.09 + 1.1) / 3 = 1.1
        assert bar.typical_price == pytest.approx(1.1, rel=0.01)

    def test_range(self):
        """Test bar range calculation."""
        bar = MarketBar(
            symbol="EURUSD",
            timestamp=datetime.now(),
            open=1.1000,
            high=1.1100,
            low=1.0900,
            close=1.1000,
        )

        assert bar.range == pytest.approx(0.02, rel=0.01)

    def test_to_dict(self):
        """Test bar serialization."""
        bar = MarketBar(
            symbol="EURUSD",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            open=1.1000,
            high=1.1050,
            low=1.0950,
            close=1.1020,
            volume=10000,
        )

        d = bar.to_dict()

        assert d["symbol"] == "EURUSD"
        assert d["open"] == 1.1000
        assert "timestamp" in d


class TestMarketSnapshot:
    """Tests for MarketSnapshot dataclass."""

    def test_snapshot_creation(self):
        """Test snapshot creation."""
        bar = MarketBar(
            symbol="EURUSD",
            timestamp=datetime.now(),
            open=1.1000,
            high=1.1050,
            low=1.0950,
            close=1.1020,
        )
        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            bars={"EURUSD": bar},
            market_status=MarketStatus.OPEN,
        )

        assert snapshot.market_status == MarketStatus.OPEN
        assert "EURUSD" in snapshot.bars

    def test_get_price(self):
        """Test getting price from snapshot."""
        bar = MarketBar(
            symbol="EURUSD",
            timestamp=datetime.now(),
            open=1.1000,
            high=1.1050,
            low=1.0950,
            close=1.1020,
        )
        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            bars={"EURUSD": bar},
        )

        assert snapshot.get_price("EURUSD") == 1.1020
        assert snapshot.get_price("GBPUSD") is None

    def test_get_bar(self):
        """Test getting bar from snapshot."""
        bar = MarketBar(
            symbol="EURUSD",
            timestamp=datetime.now(),
            open=1.1000,
            high=1.1050,
            low=1.0950,
            close=1.1020,
        )
        snapshot = MarketSnapshot(
            timestamp=datetime.now(),
            bars={"EURUSD": bar},
        )

        assert snapshot.get_bar("EURUSD") == bar
        assert snapshot.get_bar("GBPUSD") is None


class TestMarketSession:
    """Tests for MarketSession."""

    def test_forex_session_always_open_weekday(self):
        """Test forex session is open on weekdays."""
        # Monday at noon
        monday = datetime(2024, 1, 15, 12, 0, 0)  # Monday
        assert FOREX_SESSION.is_open(monday) is True

    def test_forex_session_closed_weekend(self):
        """Test forex session is closed on weekends."""
        # Saturday
        saturday = datetime(2024, 1, 13, 12, 0, 0)
        assert FOREX_SESSION.is_open(saturday) is False

    def test_us_stock_session_hours(self):
        """Test US stock session hours."""
        session = US_STOCK_SESSION

        # 10:00 AM - should be open
        open_time = datetime(2024, 1, 15, 10, 0, 0)  # Monday
        assert session.is_open(open_time) is True

        # 8:00 AM - before open
        early = datetime(2024, 1, 15, 8, 0, 0)
        assert session.is_open(early) is False

        # 5:00 PM - after close
        late = datetime(2024, 1, 15, 17, 0, 0)
        assert session.is_open(late) is False


class TestMarketSimulator:
    """Tests for MarketSimulator."""

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        np.random.seed(42)

        # Generate random walk prices
        returns = np.random.randn(100) * 0.01
        prices = 1.1 * (1 + returns).cumprod()

        df = pd.DataFrame({
            "open": prices * (1 + np.random.randn(100) * 0.001),
            "high": prices * (1 + abs(np.random.randn(100) * 0.005)),
            "low": prices * (1 - abs(np.random.randn(100) * 0.005)),
            "close": prices,
            "volume": np.random.randint(1000, 10000, 100),
        }, index=dates)

        return df

    @pytest.fixture
    def simulator(self, sample_data):
        """Create market simulator with sample data."""
        sim = MarketSimulator()
        sim.load_data("EURUSD", sample_data)
        return sim

    def test_load_data(self, sample_data):
        """Test loading data."""
        sim = MarketSimulator()
        sim.load_data("EURUSD", sample_data)

        assert "EURUSD" in sim.symbols
        assert len(sim._data["EURUSD"]) == 100

    def test_start_simulation(self, simulator):
        """Test starting simulation."""
        simulator.start()

        assert simulator.is_running is True
        assert simulator.current_time is not None

    def test_advance_bars(self, simulator):
        """Test advancing simulation."""
        simulator.start()
        initial_index = simulator.current_index

        snapshot = simulator.advance(bars=5)

        assert simulator.current_index == initial_index + 5
        assert snapshot is not None
        assert "EURUSD" in snapshot.bars

    def test_iter_bars(self, simulator):
        """Test iterating through bars."""
        bar_count = 0

        for snapshot in simulator.iter_bars():
            bar_count += 1
            assert snapshot is not None
            if bar_count > 10:
                break

        assert bar_count > 10

    def test_get_current_snapshot(self, simulator):
        """Test getting current snapshot."""
        simulator.start()

        snapshot = simulator.get_current_snapshot()

        assert snapshot.timestamp is not None
        assert "EURUSD" in snapshot.bars
        assert snapshot.bars["EURUSD"].close > 0

    def test_get_current_bar(self, simulator):
        """Test getting current bar."""
        simulator.start()

        bar = simulator.get_current_bar("EURUSD")

        assert bar is not None
        assert bar.symbol == "EURUSD"
        assert bar.close > 0

    def test_get_current_price(self, simulator):
        """Test getting current price."""
        simulator.start()

        price = simulator.get_current_price("EURUSD")

        assert price is not None
        assert price > 0

    def test_get_historical_bars(self, simulator):
        """Test getting historical bars."""
        simulator.start()
        simulator.advance(bars=20)

        bars = simulator.get_historical_bars("EURUSD", n_bars=10)

        assert len(bars) == 10
        # Check bars are in chronological order
        for i in range(1, len(bars)):
            assert bars[i].timestamp >= bars[i-1].timestamp

    def test_get_historical_data(self, simulator):
        """Test getting historical DataFrame."""
        simulator.start()
        simulator.advance(bars=20)

        df = simulator.get_historical_data("EURUSD", n_bars=10)

        assert len(df) == 10
        assert "close" in df.columns

    def test_progress(self, simulator):
        """Test progress tracking."""
        simulator.start()

        initial_progress = simulator.progress
        simulator.advance(bars=50)
        later_progress = simulator.progress

        assert later_progress > initial_progress
        assert 0 <= later_progress <= 1

    def test_remaining_bars(self, simulator):
        """Test remaining bars count."""
        simulator.start()
        initial_remaining = simulator.remaining_bars

        simulator.advance(bars=10)
        later_remaining = simulator.remaining_bars

        assert later_remaining == initial_remaining - 10

    def test_stop_and_reset(self, simulator):
        """Test stopping and resetting simulation."""
        simulator.start()
        simulator.advance(bars=20)

        simulator.stop()
        assert simulator.is_running is False

        simulator.reset()
        assert simulator.is_running is True
        assert simulator.current_index == simulator._start_index

    def test_date_range_filtering(self, sample_data):
        """Test starting simulation with date range."""
        sim = MarketSimulator()
        sim.load_data("EURUSD", sample_data)

        start_date = sample_data.index[20]
        end_date = sample_data.index[80]

        sim.start(start_time=start_date, end_time=end_date)

        # First bar should be at or after start_date
        assert sim.current_time >= start_date

    def test_on_bar_callback(self, simulator):
        """Test bar callback functionality."""
        callbacks_received = []

        def on_bar(snapshot):
            callbacks_received.append(snapshot)

        simulator.on_bar(on_bar)
        simulator.start()

        # Advance to trigger callbacks
        for _ in range(5):
            simulator.advance()

        assert len(callbacks_received) == 5

    def test_atr_calculation(self, sample_data):
        """Test ATR is calculated when not present."""
        # Remove ATR if present
        if "atr" in sample_data.columns:
            sample_data = sample_data.drop(columns=["atr"])

        sim = MarketSimulator(atr_period=14)
        sim.load_data("EURUSD", sample_data)

        assert "atr" in sim._data["EURUSD"].columns

    def test_multiple_symbols(self, sample_data):
        """Test simulator with multiple symbols."""
        sim = MarketSimulator()

        # Create second dataset
        sample_data2 = sample_data.copy()
        sample_data2["close"] = sample_data["close"] * 1.2

        sim.load_data("EURUSD", sample_data)
        sim.load_data("GBPUSD", sample_data2)

        sim.start()
        snapshot = sim.get_current_snapshot()

        assert "EURUSD" in snapshot.bars
        assert "GBPUSD" in snapshot.bars
        assert snapshot.bars["GBPUSD"].close > snapshot.bars["EURUSD"].close

    def test_get_stats(self, simulator):
        """Test getting simulator statistics."""
        simulator.start()
        simulator.advance(bars=10)

        stats = simulator.get_stats()

        assert "symbols" in stats
        assert "current_time" in stats
        assert "progress" in stats
        assert stats["is_running"] is True
