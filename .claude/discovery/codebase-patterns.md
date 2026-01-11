# AI-Trader Codebase Patterns Discovery Report

> Generated: 2026-01-07
> Purpose: Foundation for creating Claude Code Skills

---

## Table of Contents

1. [Skills Best Practices Summary](#skills-best-practices-summary)
2. [Technology Stack](#technology-stack)
3. [Architecture Patterns](#architecture-patterns)
4. [Code Patterns by Layer](#code-patterns-by-layer)
5. [Domain-Specific Patterns](#domain-specific-patterns)
6. [Naming Conventions](#naming-conventions)
7. [Configuration Patterns](#configuration-patterns)
8. [Recommended Skills](#recommended-skills)

---

## Skills Best Practices Summary

Based on Anthropic's documentation, effective Skills should follow these principles:

### Core Principles

1. **Progressive Disclosure**: Skills load content in stages (metadata -> instructions -> resources)
2. **Conciseness**: Only include context Claude doesn't already have
3. **Appropriate Freedom**: Match specificity to task fragility
4. **Filesystem-Based**: Skills are directories with SKILL.md and optional resources

### Skill Structure

```
skill-name/
├── SKILL.md              # Required: Instructions + YAML frontmatter
├── REFERENCE.md          # Optional: Detailed reference docs
├── examples/             # Optional: Code examples
└── scripts/              # Optional: Executable utilities
```

### SKILL.md Requirements

```yaml
---
name: skill-name          # lowercase, hyphens, max 64 chars
description: Description  # max 1024 chars, third person
---

# Skill Name

## Instructions
[Clear, step-by-step guidance]

## Examples
[Concrete examples]
```

### Key Recommendations

- **Description**: Write in third person, include what the skill does AND when to use it
- **Token Budget**: Keep SKILL.md body under 500 lines
- **File References**: Keep one level deep from SKILL.md
- **Naming**: Use gerund form (verb + -ing) for skill names
- **Testing**: Test with different Claude models (Haiku, Sonnet, Opus)
- **Workflows**: Include checklists for complex multi-step tasks
- **Feedback Loops**: Implement validate -> fix -> repeat patterns

---

## Technology Stack

### Core Technologies

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| Language | Python | 3.11+ | Core development |
| Deep Learning | PyTorch | >=2.0.0 | Model implementation |
| ML Framework | PyTorch Lightning | >=2.0.0 | Training utilities |
| Time Series | Darts | >=0.26.0 | Forecasting library |
| Technical Analysis | pandas-ta, TA-Lib | >=0.3.14, >=0.4.28 | Indicators |
| Data Processing | pandas, numpy, polars | >=2.0.0, >=1.24.0, >=0.19.0 | Data manipulation |
| API Framework | FastAPI | >=0.100.0 | REST API |
| Validation | Pydantic | >=2.0.0 | Data validation |
| Database | SQLAlchemy, PostgreSQL, Redis | >=2.0.0 | Persistence |
| ML Tracking | MLflow | >=2.8.0 | Experiment tracking |
| Config | pydantic-settings, python-dotenv | >=2.0.0, >=1.0.0 | Configuration |
| Logging | loguru | >=0.7.0 | Structured logging |

### Development Tools

| Tool | Version | Purpose |
|------|---------|---------|
| black | >=23.0.0 | Code formatting (100 char line) |
| isort | >=5.12.0 | Import sorting |
| flake8 | >=6.1.0 | Linting |
| mypy | >=1.5.0 | Type checking |
| pytest | >=7.4.0 | Testing |
| pytest-asyncio | >=0.21.0 | Async test support |
| pytest-cov | >=4.1.0 | Coverage |

---

## Architecture Patterns

### Layer Architecture

```
ai-trader/
├── src/
│   ├── config/          # Configuration management
│   ├── data/            # Data layer
│   │   ├── sources/     # Data source adapters
│   │   ├── processors/  # Data transformation
│   │   └── storage/     # Database/cache
│   ├── features/        # Feature engineering
│   │   └── technical/   # Technical indicators (FOCUS)
│   ├── models/          # ML models
│   │   ├── base.py      # Abstract base classes
│   │   ├── technical/   # Time-horizon models
│   │   └── ensemble/    # Model combination
│   ├── simulation/      # Backtesting
│   ├── trading/         # Trading engine
│   └── api/             # REST API
│       └── routes/      # API endpoints
├── configs/             # YAML configurations
│   └── indicators/      # Per-model indicator configs
├── data/sample/         # Sample data (CSV)
└── tests/               # Test suite
    ├── unit/
    └── integration/
```

### Pattern: Layered Architecture

**When to use**: Organizing code by technical concern (data, models, API)

**When NOT to use**: Simple scripts, single-file utilities

**Example Files**:
- `src/data/sources/` - Data layer
- `src/models/` - Model layer
- `src/api/` - Presentation layer
- `src/trading/` - Business logic layer

---

## Code Patterns by Layer

### Pattern 1: Abstract Base Class with Registry

**Description**: Base class defines interface, registry enables factory creation

**Frequency**: High (used across all model types)

**Files**:
- `src/models/base.py:57-259` - BaseModel + ModelRegistry
- `src/data/sources/base.py:10-99` - BaseDataSource + DataSourceFactory

**Example**:
```python
# src/models/base.py
class BaseModel(ABC):
    """Abstract base class for all prediction models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = self.config.get("name", self.__class__.__name__)
        self.model = None
        self.is_trained = False

    @abstractmethod
    def build(self) -> None:
        """Build model architecture."""
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Prediction:
        pass


class ModelRegistry:
    """Registry for managing model classes."""
    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        cls._models[name.lower()] = model_class

    @classmethod
    def create(cls, name: str, config=None) -> BaseModel:
        model_class = cls._models.get(name.lower())
        if model_class is None:
            raise ValueError(f"Unknown model: {name}")
        return model_class(config)


# Usage in implementations
ModelRegistry.register("short_term", ShortTermModel)
```

**When to use**:
- Multiple implementations of same interface
- Factory pattern for object creation
- Plugin-style architecture

**Quality Criteria**:
- Abstract methods clearly defined
- Registry uses lowercase names for case-insensitivity
- Helpful error messages listing available options

---

### Pattern 2: Dataclass for Data Transfer Objects

**Description**: Use `@dataclass` for structured data with type hints

**Frequency**: Very High

**Files**:
- `src/models/base.py:14-55` - Prediction dataclass
- `src/simulation/backtester.py:17-75` - BacktestResult dataclass
- `src/trading/risk.py:8-29` - RiskLimits dataclass
- `src/trading/engine.py:22-35` - TradingSignal dataclass

**Example**:
```python
# src/models/base.py
@dataclass
class Prediction:
    """Model prediction output."""
    timestamp: datetime
    symbol: str
    price_prediction: float
    price_predictions_multi: Dict[str, float] = field(default_factory=dict)
    direction: str = "neutral"
    direction_probability: float = 0.5
    confidence: float = 0.5
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            # ...
        }
```

**When to use**:
- Structured data with multiple fields
- Model outputs, API responses, configuration
- When you need `to_dict()` serialization

**Variations**:
- `field(default_factory=list)` for mutable defaults
- `@dataclass(frozen=True)` for immutable objects
- Inheritance: `class EnsemblePrediction(Prediction)`

---

### Pattern 3: Pydantic Settings for Configuration

**Description**: Use `pydantic_settings.BaseSettings` for type-safe config with .env support

**Frequency**: Core pattern (single instance)

**Files**:
- `src/config/settings.py:1-64`

**Example**:
```python
# src/config/settings.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Global application settings."""
    app_name: str = "AI Assets Trader"
    debug: bool = False
    database_url: str = "postgresql://localhost:5432/ai_trader"
    max_position_size: float = 0.02
    learning_rate: float = 1e-4

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
```

**When to use**:
- Application configuration
- Environment-based settings
- Type validation for config values

**Quality Criteria**:
- Use `@lru_cache()` for singleton pattern
- Group related settings (paths, database, trading)
- Provide sensible defaults

---

### Pattern 4: Pydantic Models for API Request/Response

**Description**: Use Pydantic `BaseModel` for API contracts with automatic validation

**Frequency**: High (all API endpoints)

**Files**:
- `src/api/routes/predictions.py:12-33`

**Example**:
```python
# src/api/routes/predictions.py
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    """Request body for prediction."""
    symbol: str = Field(..., description="Trading symbol (e.g., EURUSD)")
    timeframe: str = Field(default="1H", description="Timeframe for prediction")
    horizons: List[int] = Field(default=[1, 4, 12, 24])


class PredictionResponse(BaseModel):
    """Response body for prediction."""
    symbol: str
    timestamp: datetime
    direction: str
    confidence: float
    price_predictions: Dict[str, float]
```

**When to use**:
- API request/response schemas
- Data validation with helpful errors
- OpenAPI/Swagger documentation

**Quality Criteria**:
- Use `Field(...)` for required fields
- Use `Field(default=X, description="...")` for optional
- Add Query validation for path params: `Query(default=100, le=1000)`

---

### Pattern 5: FastAPI Router Pattern

**Description**: Organize API endpoints using APIRouter with tags

**Frequency**: Standard (all API routes)

**Files**:
- `src/api/main.py:1-43`
- `src/api/routes/predictions.py:1-108`

**Example**:
```python
# src/api/main.py
from fastapi import FastAPI
from .routes import predictions, trading, health

def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Assets Trader API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(health.router, tags=["Health"])
    app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
    return app

# src/api/routes/predictions.py
router = APIRouter()

@router.post("/predictions", response_model=PredictionResponse)
async def get_prediction(request: PredictionRequest) -> PredictionResponse:
    """Get prediction for a symbol."""
    return PredictionResponse(...)
```

**When to use**:
- REST API organization
- Grouping related endpoints
- API versioning with prefixes

---

### Pattern 6: Indicator Calculator Class

**Description**: Calculator class with `calculate_all()` and `get_feature_names()` methods

**Frequency**: High (all indicator types)

**Files**:
- `src/features/technical/trend.py:1-224`
- `src/features/technical/momentum.py:1-200`
- `src/features/technical/volatility.py`
- `src/features/technical/volume.py`
- `src/features/technical/indicators.py:1-89` (combines all)

**Example**:
```python
# src/features/technical/trend.py
class TrendIndicators:
    """Calculate trend-based technical indicators."""

    def __init__(self):
        self._feature_names: List[str] = []

    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self._feature_names.copy()

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all trend indicators."""
        df = df.copy()
        self._feature_names = []

        df = self.sma(df, periods=[5, 10, 20, 50])
        df = self.ema(df, periods=[5, 10, 20, 50])
        df = self.adx(df, period=14)
        return df

    def sma(self, df: pd.DataFrame, periods: List[int], column: str = "close"):
        """Simple Moving Average."""
        for period in periods:
            col_name = f"sma_{period}"
            df[col_name] = df[column].rolling(window=period).mean()
            self._feature_names.append(col_name)
        return df
```

**When to use**:
- Feature engineering pipelines
- When you need to track feature names
- Composable indicator groups

**Quality Criteria**:
- Always `df.copy()` at start to avoid mutation
- Track feature names in `_feature_names` list
- Return modified dataframe for chaining

---

### Pattern 7: Model with DEFAULT_CONFIG and Merged Config

**Description**: Models define defaults, constructor merges with user config

**Frequency**: High (all model implementations)

**Files**:
- `src/models/technical/short_term.py:22-54`
- `src/models/ensemble/combiner.py:163-180`

**Example**:
```python
# src/models/technical/short_term.py
class ShortTermModel(TechnicalBaseModel):
    """Short-term prediction model."""

    DEFAULT_CONFIG = {
        "name": "short_term",
        "version": "1.0.0",
        "sequence_length": 168,
        "prediction_horizon": [1, 4, 12, 24],
        "cnn_filters": [64, 128, 256],
        "lstm_hidden_size": 256,
        "batch_size": 64,
        "learning_rate": 1e-4,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)
```

**When to use**:
- Models with many hyperparameters
- When you need sensible defaults
- Override-friendly configuration

---

### Pattern 8: Processor with Validate/Clean/Transform Methods

**Description**: Data processor with validation, cleaning, and transformation pipeline

**Frequency**: Standard

**Files**:
- `src/data/processors/ohlcv.py:1-258`

**Example**:
```python
# src/data/processors/ohlcv.py
class OHLCVProcessor:
    """Process and transform OHLCV data."""

    def validate(self, df: pd.DataFrame) -> bool:
        """Validate OHLCV dataframe."""
        required_columns = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean OHLCV data."""
        df = df.copy()
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.ffill()
        return df

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived price features."""
        df = df.copy()
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        return df

    def create_sequences(self, df, sequence_length, target_column="close"):
        """Create sequences for time series modeling."""
        # Returns (X, y) numpy arrays
```

**When to use**:
- Data preprocessing pipelines
- ETL operations
- Time series data preparation

---

### Pattern 9: Risk Manager with Circuit Breakers

**Description**: Risk management class with limits, position sizing, and automatic halt

**Frequency**: Core trading pattern

**Files**:
- `src/trading/risk.py:1-270`

**Example**:
```python
# src/trading/risk.py
@dataclass
class RiskLimits:
    """Risk management limits."""
    max_position_size: float = 0.02
    max_daily_loss: float = 0.05
    max_drawdown: float = 0.15
    max_trades_per_day: int = 20


class RiskManager:
    def __init__(self, account_balance: float, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.is_halted = False

    def check_signal(self, signal) -> bool:
        """Check if signal passes risk filters."""
        if self.is_halted:
            return False
        if signal.confidence < self.limits.min_confidence:
            return False
        return True

    def calculate_position_size(self, symbol, signal_strength, stop_loss_distance):
        """Calculate position size based on risk parameters."""
        # Kelly-criterion-inspired position sizing
        pass

    def _check_circuit_breakers(self):
        """Check and activate circuit breakers if needed."""
        if daily_loss_pct >= self.limits.max_daily_loss:
            self.halt_trading("Daily loss limit reached")
```

**When to use**:
- Trading systems requiring risk controls
- Systems with automatic safety stops
- Position sizing logic

---

### Pattern 10: Enum for State Management

**Description**: Use `Enum` for finite state values

**Frequency**: Standard

**Files**:
- `src/trading/engine.py:15-19`
- `src/trading/execution.py` (OrderType, OrderSide, OrderStatus)

**Example**:
```python
# src/trading/engine.py
from enum import Enum

class TradingMode(Enum):
    """Trading operation mode."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

# Usage
engine = TradingEngine(mode=TradingMode.PAPER)
if engine.mode == TradingMode.LIVE:
    # Handle live trading
```

**When to use**:
- Finite set of states/modes
- Order types, statuses
- Configuration options

---

## Domain-Specific Patterns

### Trading Domain Patterns

#### 1. Multi-Horizon Prediction

**Description**: Models predict multiple time horizons simultaneously

**Files**: `src/models/technical/short_term.py:27-28`

```python
"prediction_horizon": [1, 4, 12, 24]  # 1H, 4H, 12H, 24H
```

**Convention**: Store multi-horizon predictions in dict with string keys:
```python
price_predictions_multi = {"1h": 1.1234, "4h": 1.1245, "24h": 1.1300}
```

#### 2. Technical Indicator Priority System

**Description**: Indicators marked with priority levels (P0-P3) for importance

**Files**: `configs/indicators/short_term_indicators.yaml`

```yaml
momentum:
  rsi:
    enabled: true
    periods: [7, 14]
    priority: P0  # Critical

  cci:
    enabled: false
    priority: P3  # Optional - disable first if needed
```

**Priority Levels**:
- `P0`: Critical - must have (RSI, MACD, ATR)
- `P1`: Important - significantly improves accuracy
- `P2`: Useful - adds value, can be omitted
- `P3`: Optional - nice to have

#### 3. Backtesting Result Structure

**Description**: Comprehensive metrics structure for strategy evaluation

**Files**: `src/simulation/backtester.py:17-75`, `src/simulation/metrics.py`

**Key Metrics**:
- Return metrics: total_return, annualized_return
- Risk metrics: sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio
- Trade statistics: win_rate, profit_factor, average_win/loss

#### 4. Signal to Order Conversion

**Description**: Predictions -> TradingSignal -> Order pipeline

**Files**: `src/trading/engine.py:101-189`

```python
# Pipeline
prediction = model.predict(features)     # Prediction dataclass
signal = engine.on_market_data(...)      # TradingSignal dataclass
order = engine.process_signal(signal)    # Order dataclass
```

---

## Naming Conventions

### File Naming

| Pattern | Example | Usage |
|---------|---------|-------|
| snake_case | `short_term.py` | All Python files |
| Singular | `backtester.py` | Classes/modules |
| Descriptive | `short_term_indicators.yaml` | Config files |

### Class Naming

| Pattern | Example | Usage |
|---------|---------|-------|
| PascalCase | `TechnicalEnsemble` | All classes |
| *Model suffix | `ShortTermModel` | Model classes |
| *Indicators suffix | `TrendIndicators` | Indicator calculators |
| *Manager suffix | `RiskManager`, `PositionManager` | Manager classes |
| *Processor suffix | `OHLCVProcessor` | Data processors |
| Base* prefix | `BaseModel`, `BaseDataSource` | Abstract base classes |

### Method Naming

| Pattern | Example | Usage |
|---------|---------|-------|
| calculate_* | `calculate_all`, `calculate_position_size` | Computation methods |
| get_* | `get_feature_names`, `get_settings` | Getter methods |
| check_* | `check_signal`, `check_circuit_breakers` | Validation methods |
| to_dict | `prediction.to_dict()` | Serialization |
| _private | `_normalize_weights`, `_check_exits` | Internal methods |

### Variable Naming

| Pattern | Example | Usage |
|---------|---------|-------|
| df | `df`, `train_df` | DataFrames |
| *_series | `equity_series` | pandas Series |
| *_config | `model_config` | Configuration dicts |
| *_history | `trade_history` | Lists of records |
| is_* | `is_trained`, `is_halted` | Boolean flags |

### Column Naming (Technical Indicators)

| Pattern | Example | Description |
|---------|---------|-------------|
| indicator_period | `sma_20`, `rsi_14` | Single period indicators |
| indicator_period_suffix | `stoch_k_14`, `stoch_d_14` | Multi-output indicators |
| indicator_param1_param2 | `bb_20_2` | Multiple parameters |

---

## Configuration Patterns

### YAML Model Configuration Structure

**Files**: `configs/short_term.yaml`, `configs/medium_term.yaml`

```yaml
model:
  name: short_term
  version: "1.0.0"

data:
  sequence_length: 168
  prediction_horizon: [1, 4, 12, 24]
  train_ratio: 0.7

architecture:
  cnn_filters: [64, 128, 256]
  lstm_hidden_size: 256

training:
  batch_size: 64
  learning_rate: 0.0001
  epochs: 100
  early_stopping_patience: 15
  optimizer: AdamW

  loss:
    price: HuberLoss
    direction: CrossEntropyLoss

features:
  indicators_config: configs/indicators/short_term_indicators.yaml
```

### YAML Indicator Configuration Structure

**Files**: `configs/indicators/short_term_indicators.yaml`

```yaml
version: "1.0"
model_type: short_term

indicators:
  enabled_categories:
    - trend
    - momentum

  trend:
    ema:
      enabled: true
      periods: [8, 13, 21, 55]
      priority: P0

    adx:
      enabled: false
      period: 14
      priority: P2
```

---

## Recommended Skills

Based on the codebase patterns, the following Skills would be valuable:

### 1. `creating-technical-indicators`

**Description**: Create new technical indicators following the project's calculator pattern with feature tracking.

**Key Patterns to Encode**:
- Indicator calculator class structure
- `_feature_names` tracking
- `calculate_all()` method pattern
- DataFrame copy and chaining

### 2. `implementing-prediction-models`

**Description**: Implement new prediction models inheriting from BaseModel with proper registration.

**Key Patterns to Encode**:
- DEFAULT_CONFIG pattern
- Abstract method implementations (build, train, predict, predict_batch)
- ModelRegistry registration
- Prediction dataclass usage

### 3. `creating-api-endpoints`

**Description**: Create FastAPI endpoints with Pydantic request/response models.

**Key Patterns to Encode**:
- Pydantic BaseModel for requests/responses
- Field validation with descriptions
- APIRouter organization
- Response model typing

### 4. `running-backtests`

**Description**: Execute backtests and interpret results using the simulation module.

**Key Patterns to Encode**:
- BacktestResult structure
- Metric interpretation (Sharpe, Sortino, drawdown)
- Walk-forward validation

### 5. `configuring-model-training`

**Description**: Configure and run model training with proper YAML configuration.

**Key Patterns to Encode**:
- YAML configuration structure
- Indicator configuration by priority
- Training hyperparameters

### 6. `adding-data-sources`

**Description**: Add new data source connectors following the BaseDataSource pattern.

**Key Patterns to Encode**:
- BaseDataSource abstract methods
- DataSourceFactory registration
- Context manager pattern (__enter__, __exit__)

### 7. `analyzing-trading-performance`

**Description**: Calculate and interpret trading performance metrics.

**Key Patterns to Encode**:
- PerformanceMetrics class methods
- Risk-adjusted returns (Sharpe, Sortino)
- Trade statistics calculation

---

## Appendix: Quick Reference

### Creating a New Model

```python
# src/models/technical/new_model.py
from ..base import BaseModel, Prediction, ModelRegistry

class NewModel(BaseModel):
    DEFAULT_CONFIG = {"name": "new_model", "version": "1.0.0"}

    def __init__(self, config=None):
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

    def build(self): pass
    def train(self, X_train, y_train, X_val=None, y_val=None): pass
    def predict(self, X): pass
    def predict_batch(self, X): pass

ModelRegistry.register("new_model", NewModel)
```

### Creating a New Indicator Group

```python
# src/features/technical/new_indicators.py
class NewIndicators:
    def __init__(self):
        self._feature_names: List[str] = []

    def get_feature_names(self) -> List[str]:
        return self._feature_names.copy()

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        self._feature_names = []
        # Add calculations
        return df
```

### Creating a New API Endpoint

```python
# src/api/routes/new_route.py
from fastapi import APIRouter
from pydantic import BaseModel, Field

router = APIRouter()

class NewRequest(BaseModel):
    param: str = Field(..., description="Required param")

class NewResponse(BaseModel):
    result: str

@router.post("/new-endpoint", response_model=NewResponse)
async def new_endpoint(request: NewRequest) -> NewResponse:
    return NewResponse(result="...")
```
