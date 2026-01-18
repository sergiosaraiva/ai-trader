---
name: generating-test-data
description: Creates test fixtures, mock data builders, and synthetic datasets for testing. Generates realistic OHLCV data, edge cases, and scenario-specific test inputs. Use when planning-test-scenarios identifies data requirements, or when tests need consistent reproducible data.
version: 1.0.0
---

# Generating Test Data

Create reliable, reproducible test data for unit tests, integration tests, and scenario validation.

## Quick Reference

```
1. Identify data requirements from test plan
2. Choose generation strategy (fixture, builder, synthetic)
3. Generate data with edge cases
4. Store in appropriate location
5. Document data characteristics
```

## When to Use

- Test plan requires specific data scenarios
- Need reproducible test data across runs
- Testing edge cases (NaN, zeros, extremes)
- Creating mock objects for dependencies
- Building synthetic datasets for ML testing

## When NOT to Use

- Production data available and appropriate
- Simple literal values sufficient
- Random data acceptable (fuzz testing)

---

## Data Generation Strategies

### Strategy 1: Static Fixtures

Pre-defined data files for consistent tests.

```
tests/fixtures/
├── ohlcv/
│   ├── eurusd_100_candles.csv
│   ├── eurusd_with_gaps.csv
│   └── eurusd_high_volatility.csv
├── predictions/
│   ├── valid_prediction.json
│   └── edge_case_predictions.json
└── configs/
    └── test_indicator_config.yaml
```

**Best for**: Reference data, golden outputs, configuration

### Strategy 2: Builder Pattern

Programmatic data construction with defaults.

```python
class OHLCVBuilder:
    def __init__(self):
        self.reset()

    def reset(self):
        self._data = {
            "timestamp": [],
            "open": [],
            "high": [],
            "low": [],
            "close": [],
            "volume": [],
        }
        self._default_price = 100.0
        self._default_volume = 1000
        return self

    def with_candles(self, n: int):
        """Add n candles with default values."""
        for i in range(n):
            self._add_candle(self._default_price, self._default_volume)
        return self

    def with_trend(self, direction: str, candles: int, magnitude: float):
        """Add trending candles."""
        price = self._data["close"][-1] if self._data["close"] else self._default_price
        step = magnitude / candles * (1 if direction == "up" else -1)
        for _ in range(candles):
            price += step
            self._add_candle(price, self._default_volume)
        return self

    def with_nan_at(self, positions: List[int]):
        """Insert NaN at specific positions."""
        for pos in positions:
            if pos < len(self._data["close"]):
                self._data["close"][pos] = np.nan
        return self

    def build(self) -> pd.DataFrame:
        """Return the constructed DataFrame."""
        df = pd.DataFrame(self._data)
        df["timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="1h")
        df.set_index("timestamp", inplace=True)
        return df
```

**Best for**: Configurable data, many variations, readable tests

### Strategy 3: Synthetic Generation

Algorithm-generated data with controlled properties.

```python
class SyntheticOHLCV:
    @staticmethod
    def random_walk(
        n_candles: int,
        start_price: float = 100.0,
        volatility: float = 0.02,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate random walk price series."""
        np.random.seed(seed)
        returns = np.random.normal(0, volatility, n_candles)
        prices = start_price * np.cumprod(1 + returns)

        # Generate OHLC from close prices
        df = pd.DataFrame({
            "close": prices,
            "open": np.roll(prices, 1),
            "high": prices * (1 + np.abs(np.random.normal(0, volatility/2, n_candles))),
            "low": prices * (1 - np.abs(np.random.normal(0, volatility/2, n_candles))),
            "volume": np.random.randint(1000, 10000, n_candles),
        })
        df.loc[0, "open"] = start_price
        return df

    @staticmethod
    def with_pattern(pattern: str, n_candles: int) -> pd.DataFrame:
        """Generate data with specific pattern."""
        if pattern == "squeeze":
            # Low volatility consolidation
            return SyntheticOHLCV._generate_squeeze(n_candles)
        elif pattern == "breakout":
            # Consolidation then expansion
            return SyntheticOHLCV._generate_breakout(n_candles)
        elif pattern == "trend_up":
            return SyntheticOHLCV._generate_trend(n_candles, direction=1)
        elif pattern == "trend_down":
            return SyntheticOHLCV._generate_trend(n_candles, direction=-1)
```

**Best for**: Large datasets, statistical properties, ML training

---

## Examples

### Example 1: OHLCV Test Fixtures

**Requirement**: Test data for technical indicator calculations

**Implementation**:

```python
# tests/conftest.py
import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_ohlcv():
    """Standard 100-candle OHLCV fixture."""
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)

    return pd.DataFrame({
        "open": np.roll(close, 1),
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 10000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))


@pytest.fixture
def ohlcv_with_nan():
    """OHLCV data with NaN values for edge case testing."""
    df = sample_ohlcv()
    df.loc[df.index[10], "close"] = np.nan
    df.loc[df.index[50], "volume"] = np.nan
    return df


@pytest.fixture
def ohlcv_insufficient():
    """Only 5 candles - insufficient for most indicators."""
    return sample_ohlcv().head(5)


@pytest.fixture
def ohlcv_flat():
    """Flat price - tests division by zero cases."""
    n = 50
    return pd.DataFrame({
        "open": [100.0] * n,
        "high": [100.0] * n,
        "low": [100.0] * n,
        "close": [100.0] * n,
        "volume": [1000] * n,
    }, index=pd.date_range("2024-01-01", periods=n, freq="1h"))
```

### Example 2: Prediction Builder

**Requirement**: Create Prediction objects for testing

**Implementation**:

```python
# tests/builders/prediction_builder.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from src.models.base import Prediction


class PredictionBuilder:
    """Builder for creating test Prediction objects."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._timestamp = datetime.now()
        self._symbol = "EURUSD"
        self._price_prediction = 1.0850
        self._price_predictions_multi = {"1h": 1.0850, "4h": 1.0855}
        self._direction = "bullish"
        self._direction_probability = 0.65
        self._confidence = 0.72
        self._model_name = "test_model"
        self._model_version = "1.0.0"
        return self

    def with_symbol(self, symbol: str):
        self._symbol = symbol
        return self

    def with_direction(self, direction: str, probability: float = 0.65):
        self._direction = direction
        self._direction_probability = probability
        return self

    def with_confidence(self, confidence: float):
        self._confidence = confidence
        return self

    def with_price(self, price: float):
        self._price_prediction = price
        return self

    def bullish(self):
        return self.with_direction("bullish", 0.75)

    def bearish(self):
        return self.with_direction("bearish", 0.70)

    def neutral(self):
        return self.with_direction("neutral", 0.50)

    def high_confidence(self):
        return self.with_confidence(0.90)

    def low_confidence(self):
        return self.with_confidence(0.35)

    def build(self) -> Prediction:
        return Prediction(
            timestamp=self._timestamp,
            symbol=self._symbol,
            price_prediction=self._price_prediction,
            price_predictions_multi=self._price_predictions_multi,
            direction=self._direction,
            direction_probability=self._direction_probability,
            confidence=self._confidence,
            model_name=self._model_name,
            model_version=self._model_version,
        )


# Usage in tests:
def test_high_confidence_bullish():
    prediction = (
        PredictionBuilder()
        .with_symbol("GBPUSD")
        .bullish()
        .high_confidence()
        .build()
    )
    assert prediction.direction == "bullish"
    assert prediction.confidence == 0.90
```

### Example 3: Trade Signal Generator

**Requirement**: Generate trade signals for backtesting tests

**Implementation**:

```python
# tests/builders/trade_builder.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
import random


@dataclass
class TradeSignal:
    timestamp: datetime
    symbol: str
    direction: str  # "long" or "short"
    size: float
    entry_price: float
    stop_loss: float
    take_profit: float


class TradeSignalGenerator:
    """Generate trade signals for testing."""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.symbols = ["EURUSD", "GBPUSD", "USDJPY"]

    def generate_random_signals(
        self,
        n_signals: int,
        start_date: datetime = None,
    ) -> List[TradeSignal]:
        """Generate n random trade signals."""
        start = start_date or datetime(2024, 1, 1)
        signals = []

        for i in range(n_signals):
            symbol = random.choice(self.symbols)
            direction = random.choice(["long", "short"])
            entry = 100 + random.uniform(-5, 5)
            stop_distance = entry * 0.02  # 2% stop
            tp_distance = entry * 0.04    # 4% take profit

            signals.append(TradeSignal(
                timestamp=start + timedelta(hours=i),
                symbol=symbol,
                direction=direction,
                size=random.uniform(0.01, 0.1),
                entry_price=entry,
                stop_loss=entry - stop_distance if direction == "long" else entry + stop_distance,
                take_profit=entry + tp_distance if direction == "long" else entry - tp_distance,
            ))

        return signals

    def generate_winning_sequence(self, n_signals: int) -> List[TradeSignal]:
        """Generate signals that would all be winners (for testing)."""
        # Implementation for deterministic winning trades
        pass

    def generate_losing_sequence(self, n_signals: int) -> List[TradeSignal]:
        """Generate signals that would all be losers (for testing)."""
        # Implementation for deterministic losing trades
        pass

    def generate_drawdown_scenario(
        self,
        initial_wins: int,
        consecutive_losses: int,
    ) -> List[TradeSignal]:
        """Generate scenario that triggers drawdown."""
        signals = []
        signals.extend(self.generate_winning_sequence(initial_wins))
        signals.extend(self.generate_losing_sequence(consecutive_losses))
        return signals
```

### Example 4: Mock Data Source

**Requirement**: Mock data source for testing without API calls

**Implementation**:

```python
# tests/mocks/mock_data_source.py
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

from src.data.sources.base import BaseDataSource


class MockDataSource(BaseDataSource):
    """Mock data source for testing."""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._data: Dict[str, pd.DataFrame] = {}
        self._should_fail = False
        self._latency_ms = 0

    def set_data(self, symbol: str, df: pd.DataFrame):
        """Pre-load data for a symbol."""
        self._data[symbol] = df

    def set_should_fail(self, should_fail: bool):
        """Configure mock to raise errors."""
        self._should_fail = should_fail

    def set_latency(self, ms: int):
        """Simulate network latency."""
        self._latency_ms = ms

    def connect(self) -> bool:
        if self._should_fail:
            raise ConnectionError("Mock connection failure")
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        if self._should_fail:
            raise RuntimeError("Mock fetch failure")

        if self._latency_ms:
            import time
            time.sleep(self._latency_ms / 1000)

        if symbol in self._data:
            df = self._data[symbol]
            # Filter by date range
            mask = df.index >= start_date
            if end_date:
                mask &= df.index <= end_date
            return df[mask]

        raise ValueError(f"No mock data for {symbol}")

    def get_available_symbols(self) -> List[str]:
        return list(self._data.keys())


# Usage in tests:
@pytest.fixture
def mock_data_source(sample_ohlcv):
    source = MockDataSource()
    source.set_data("EURUSD", sample_ohlcv)
    source.connect()
    yield source
    source.disconnect()


def test_fetch_with_mock(mock_data_source):
    df = mock_data_source.fetch_ohlcv(
        "EURUSD", "1H",
        datetime(2024, 1, 1),
        datetime(2024, 1, 5),
    )
    assert len(df) > 0
```

### Example 5: Edge Case Data Factory

**Requirement**: Systematic edge case data generation

**Implementation**:

```python
# tests/factories/edge_cases.py
import pandas as pd
import numpy as np
from typing import Callable, Dict, List


class EdgeCaseFactory:
    """Factory for generating edge case test data."""

    @staticmethod
    def all_nan(n: int = 50) -> pd.DataFrame:
        """All values are NaN."""
        return pd.DataFrame({
            "open": [np.nan] * n,
            "high": [np.nan] * n,
            "low": [np.nan] * n,
            "close": [np.nan] * n,
            "volume": [np.nan] * n,
        })

    @staticmethod
    def all_zeros(n: int = 50) -> pd.DataFrame:
        """All values are zero."""
        return pd.DataFrame({
            "open": [0.0] * n,
            "high": [0.0] * n,
            "low": [0.0] * n,
            "close": [0.0] * n,
            "volume": [0] * n,
        })

    @staticmethod
    def negative_prices(n: int = 50) -> pd.DataFrame:
        """Contains negative prices (invalid)."""
        return pd.DataFrame({
            "open": [-100.0] * n,
            "high": [-99.0] * n,
            "low": [-101.0] * n,
            "close": [-100.0] * n,
            "volume": [1000] * n,
        })

    @staticmethod
    def extreme_values(n: int = 50) -> pd.DataFrame:
        """Extreme large and small values."""
        return pd.DataFrame({
            "open": [1e10, 1e-10] * (n // 2),
            "high": [1e10, 1e-10] * (n // 2),
            "low": [1e10, 1e-10] * (n // 2),
            "close": [1e10, 1e-10] * (n // 2),
            "volume": [1e15, 1] * (n // 2),
        })

    @staticmethod
    def high_low_inverted(n: int = 50) -> pd.DataFrame:
        """High < Low (invalid OHLCV)."""
        return pd.DataFrame({
            "open": [100.0] * n,
            "high": [99.0] * n,   # Less than low!
            "low": [101.0] * n,
            "close": [100.0] * n,
            "volume": [1000] * n,
        })

    @staticmethod
    def single_row() -> pd.DataFrame:
        """Only one data point."""
        return pd.DataFrame({
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000],
        })

    @staticmethod
    def empty() -> pd.DataFrame:
        """Empty DataFrame."""
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    @classmethod
    def all_cases(cls) -> Dict[str, pd.DataFrame]:
        """Return all edge cases as a dictionary."""
        return {
            "all_nan": cls.all_nan(),
            "all_zeros": cls.all_zeros(),
            "negative_prices": cls.negative_prices(),
            "extreme_values": cls.extreme_values(),
            "high_low_inverted": cls.high_low_inverted(),
            "single_row": cls.single_row(),
            "empty": cls.empty(),
        }


# Usage: parametrized testing
@pytest.mark.parametrize("case_name,data", EdgeCaseFactory.all_cases().items())
def test_indicator_handles_edge_cases(case_name, data):
    """Test indicator doesn't crash on edge cases."""
    indicators = TechnicalIndicators()
    try:
        result = indicators.calculate_all(data)
        # Should either return valid result or raise clear error
    except ValueError as e:
        # Expected for invalid data
        assert "invalid" in str(e).lower() or "empty" in str(e).lower()
```

---

## File Organization

```
tests/
├── conftest.py           # Shared fixtures
├── fixtures/             # Static data files
│   ├── ohlcv/
│   ├── predictions/
│   └── configs/
├── builders/             # Builder classes
│   ├── __init__.py
│   ├── ohlcv_builder.py
│   ├── prediction_builder.py
│   └── trade_builder.py
├── factories/            # Factory classes
│   ├── __init__.py
│   ├── edge_cases.py
│   └── synthetic.py
├── mocks/                # Mock implementations
│   ├── __init__.py
│   ├── mock_data_source.py
│   └── mock_model.py
└── unit/                 # Unit tests
    └── ...
```

---

## Quality Checklist

- [ ] All test data is deterministic (seeded random)
- [ ] Edge cases covered (NaN, zero, empty, extreme)
- [ ] Data characteristics documented
- [ ] Fixtures are minimal (only needed data)
- [ ] Builders have sensible defaults
- [ ] Mocks implement full interface
- [ ] No production data in test fixtures

## Common Mistakes

- **Non-deterministic data**: Tests flaky → Always seed random generators
- **Too much data**: Tests slow → Use minimal data that covers case
- **Missing edge cases**: Bugs in production → Systematically generate edge cases
- **Coupled to implementation**: Tests break on refactor → Test behavior not internals
- **Production data in tests**: Security risk → Generate synthetic data

## Related Skills

- [planning-test-scenarios](planning-test-scenarios.md) - Identify data requirements
- [validating-time-series-data](validating-time-series-data.md) - Time series specific considerations
- [creating-dataclasses](creating-dataclasses.md) - Data structure patterns
