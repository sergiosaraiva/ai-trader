# AI-Trader Codebase Patterns

> Updated: 2026-01-12
> Purpose: Foundation for creating Claude Code Skills
> Source: Consolidated from Anthropic Skills best practices + codebase analysis

---

## Table of Contents

1. [Skills Best Practices Summary](#1-skills-best-practices-summary)
2. [Technology Stack](#2-technology-stack)
3. [Layer Architecture](#3-layer-architecture)
4. [Python Backend Patterns](#4-python-backend-patterns)
5. [React Frontend Patterns](#5-react-frontend-patterns)
6. [Testing Patterns](#6-testing-patterns)
7. [Development Workflow Patterns](#7-development-workflow-patterns)
8. [Domain-Specific Patterns](#8-domain-specific-patterns)
9. [Naming Conventions](#9-naming-conventions)
10. [Recommended Skills](#10-recommended-skills)

---

## 1. Skills Best Practices Summary

Based on Anthropic's documentation, effective Skills follow these principles:

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

## 2. Technology Stack

### Backend (Python 3.12+)

| Category | Technologies | Versions |
|----------|-------------|----------|
| **API Framework** | FastAPI, uvicorn | >=0.100.0, >=0.23.0 |
| **ML Models** | XGBoost, scikit-learn | >=2.0.0, >=1.3.0 |
| **Data Processing** | pandas, numpy | >=2.0.0, >=1.24.0 |
| **Technical Analysis** | pandas-ta | latest |
| **Database** | SQLAlchemy | >=2.0.0 |
| **Scheduling** | APScheduler | >=3.10.0 |
| **Validation** | Pydantic | >=2.0.0 |
| **Testing** | pytest, pytest-asyncio | >=7.4.0 |

### Frontend (Node.js 20+)

| Category | Technologies | Versions |
|----------|-------------|----------|
| **Framework** | React | 19.2.0 |
| **Build Tool** | Vite | 7.2.4 |
| **Styling** | TailwindCSS | 4.1.18 |
| **Charts** | Recharts | 3.6.0 |
| **Icons** | lucide-react | 0.562.0 |
| **Testing** | Vitest, Testing Library | 4.0.17 |

### Architecture

- **Pattern**: Layered monolith with service architecture
- **API Style**: REST with FastAPI
- **State**: SQLite for persistence, in-memory caching
- **ML Pipeline**: XGBoost ensemble with multi-timeframe models

---

## 3. Layer Architecture

```
ai-trader/
├── src/                      # Python source code
│   ├── api/                  # FastAPI web layer
│   │   ├── main.py          # App entry point
│   │   ├── routes/          # API endpoints
│   │   ├── services/        # Business logic (singletons)
│   │   ├── schemas/         # Pydantic models
│   │   └── database/        # SQLAlchemy models
│   ├── models/              # ML models
│   │   └── multi_timeframe/ # MTF Ensemble (PRIMARY)
│   ├── features/            # Feature engineering
│   │   ├── technical/       # Technical indicators
│   │   └── sentiment/       # Sentiment features
│   ├── trading/             # Trading logic
│   └── simulation/          # Backtesting
├── frontend/                # React application
│   └── src/
│       ├── components/      # React components
│       ├── api/             # API client
│       └── hooks/           # Custom hooks
├── scripts/                 # CLI tools
├── tests/                   # Test suites
└── docs/                    # Documentation
```

---

## 4. Python Backend Patterns

### Pattern 4.1: FastAPI Application Structure

**Description**: Factory pattern for FastAPI application with lifespan management, CORS, and router composition.

**Examples**:
- `src/api/main.py:87-130` - Application factory
- `src/api/main.py:25-84` - Lifespan context manager

**Code Pattern**:
```python
# src/api/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting API...")
    init_db()
    service.initialize()
    yield
    # Shutdown
    logger.info("Shutting down...")

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="AI Assets Trader API",
        description="...",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
    app.include_router(health.router, tags=["Health"])
    app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])

    return app

app = create_app()
```

**When to Use**: Creating new FastAPI applications, adding route modules

**When NOT to Use**: Simple scripts, CLI-only tools

**Quality Criteria**:
- Lifespan handles startup/shutdown gracefully
- Services initialized with error handling
- CORS configured appropriately

---

### Pattern 4.2: Service Singleton Pattern

**Description**: Thread-safe singleton services with lazy initialization, caching, and status tracking.

**Examples**:
- `src/api/services/model_service.py:28-391` - ModelService class
- `src/api/services/model_service.py:389-391` - Singleton instantiation

**Code Pattern**:
```python
# src/api/services/model_service.py
class ModelService:
    """Singleton service with thread-safe operations."""

    CACHE_TTL = timedelta(minutes=1)

    def __init__(self, model_dir: Optional[Path] = None):
        self._lock = Lock()
        self._model = None
        self._cache: Dict[str, Dict] = {}
        self._initialized = False
        self._error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def initialize(self, warm_up: bool = True) -> bool:
        if self._initialized:
            return True
        try:
            self._load_model()
            if warm_up:
                self._warm_up()
            self._initialized = True
            return True
        except Exception as e:
            self._error = str(e)
            return False

    def predict(self, df: pd.DataFrame, use_cache: bool = True) -> Dict:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        cache_key = str(df.index[-1])
        with self._lock:
            if use_cache and cache_key in self._cache:
                # Return cached result if fresh
                ...

        with self._lock:
            result = self._model.predict(df)
        return result

# Singleton instance
model_service = ModelService()
```

**When to Use**: Services managing expensive resources (ML models, DB), shared state

**When NOT to Use**: Stateless utilities, request-scoped operations

**Quality Criteria**:
- Thread-safe with proper locking
- Lazy initialization
- Clear status properties (is_loaded, is_initialized)
- Cache with TTL

---

### Pattern 4.3: FastAPI Route Pattern

**Description**: Route handlers with dependency injection, proper error handling, and Pydantic response models.

**Examples**:
- `src/api/routes/predictions.py:26-78` - GET endpoint with error handling
- `src/api/routes/predictions.py:81-125` - GET with pagination and DB dependency

**Code Pattern**:
```python
# src/api/routes/predictions.py
from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session

router = APIRouter()

@router.get("/predictions/latest", response_model=PredictionResponse)
async def get_latest_prediction() -> PredictionResponse:
    """Get the most recent prediction."""
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = data_service.get_data_for_prediction()
        if df is None or len(df) < 100:
            raise HTTPException(status_code=503, detail="Insufficient data")

        prediction = model_service.predict(df)
        return PredictionResponse(**prediction)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predictions/history", response_model=PredictionHistoryResponse)
async def get_prediction_history(
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> PredictionHistoryResponse:
    """Get historical predictions."""
    predictions = db.query(Prediction).order_by(...).offset(offset).limit(limit).all()
    return PredictionHistoryResponse(predictions=predictions, total=total)
```

**When to Use**: All API endpoint definitions, CRUD operations

**Quality Criteria**:
- Typed response_model
- Query validation with Query()
- Service availability checks first
- Re-raise HTTPException, catch others
- Descriptive docstrings

---

### Pattern 4.4: Pydantic Schema Pattern

**Description**: Request/response schemas with Field descriptions, examples, and nested models.

**Examples**:
- `src/api/schemas/prediction.py:16-68` - Response schema with Config
- `src/api/schemas/prediction.py:71-89` - List response schema

**Code Pattern**:
```python
# src/api/schemas/prediction.py
from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    """Response for prediction endpoint."""

    timestamp: str = Field(..., description="ISO format timestamp")
    symbol: str = Field(default="EURUSD", description="Trading symbol")
    direction: str = Field(..., description="'long' or 'short'")
    confidence: float = Field(..., description="Confidence (0-1)")
    should_trade: bool = Field(..., description="Whether confidence >= threshold")

    # Nested data
    component_directions: Dict[str, int] = Field(..., description="Direction by TF")
    component_confidences: Dict[str, float] = Field(..., description="Confidence by TF")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T14:00:00",
                "direction": "long",
                "confidence": 0.72,
            }
        }

class PredictionHistoryResponse(BaseModel):
    """Paginated list response."""
    predictions: List[PredictionHistoryItem]
    count: int
    total: int
```

**When to Use**: All API request/response definitions, configuration objects

**Quality Criteria**:
- Field descriptions for documentation
- Example in Config for Swagger UI
- Proper typing with Optional where needed

---

### Pattern 4.5: SQLAlchemy Model Pattern

**Description**: Database models with proper types, indexes, and relationships.

**Examples**:
- `src/api/database/models.py:22-61` - Prediction model
- `src/api/database/models.py:64-107` - Trade model with indexes

**Code Pattern**:
```python
# src/api/database/models.py
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Prediction(Base):
    """Store model predictions."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, default="EURUSD")
    direction = Column(String(10), nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_predictions_timestamp_symbol", "timestamp", "symbol"),
    )
```

**When to Use**: Persistent data storage, entities that need querying

**Quality Criteria**:
- Explicit nullability
- Indexes on frequently queried columns
- created_at/updated_at timestamps

---

### Pattern 4.6: Dataclass Configuration Pattern

**Description**: Configuration objects using dataclasses with defaults, factory methods, and post-init.

**Examples**:
- `src/models/multi_timeframe/mtf_ensemble.py:28-119` - MTFEnsembleConfig
- `src/trading/position_sizing.py:144-176` - PositionSizingConfig

**Code Pattern**:
```python
# src/models/multi_timeframe/mtf_ensemble.py
from dataclasses import dataclass, field

@dataclass
class MTFEnsembleConfig:
    """Configuration for MTF Ensemble."""

    weights: Dict[str, float] = field(default_factory=lambda: {
        "1H": 0.6, "4H": 0.3, "D": 0.1
    })
    min_confidence: float = 0.55
    agreement_bonus: float = 0.05
    include_sentiment: bool = False

    @classmethod
    def default(cls) -> "MTFEnsembleConfig":
        """Default configuration."""
        return cls()

    @classmethod
    def with_sentiment(cls, pair: str = "EURUSD") -> "MTFEnsembleConfig":
        """Config with sentiment enabled."""
        return cls(
            include_sentiment=True,
            trading_pair=pair,
            sentiment_by_timeframe={"1H": False, "4H": False, "D": True}
        )
```

**When to Use**: Complex configuration with many parameters, factory methods needed

**Quality Criteria**:
- Sensible defaults
- Factory methods for common configurations
- `field(default_factory=...)` for mutable defaults

---

### Pattern 4.7: ML Model Class Pattern

**Description**: Machine learning model classes with train/predict/save/load interface.

**Examples**:
- `src/models/multi_timeframe/mtf_ensemble.py:154-653` - MTFEnsemble class

**Code Pattern**:
```python
class MTFEnsemble:
    """Multi-Timeframe Ensemble model."""

    def __init__(
        self,
        config: Optional[MTFEnsembleConfig] = None,
        model_dir: Optional[Path] = None,
    ):
        self.config = config or MTFEnsembleConfig.default()
        self.model_dir = Path(model_dir) if model_dir else Path("models/mtf_ensemble")
        self.models: Dict[str, Model] = {}
        self.is_trained = False
        self.training_results: Dict = {}

    def train(self, df_5min: pd.DataFrame, ...) -> Dict[str, Dict]:
        """Train all timeframe models."""
        results = {}
        for tf, model in self.models.items():
            results[tf] = model.train(...)
        self.is_trained = all(m.is_trained for m in self.models.values())
        return results

    def predict(self, df_5min: pd.DataFrame) -> MTFPrediction:
        """Make ensemble prediction."""
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return MTFPrediction(...)

    def save(self, path: Optional[Path] = None) -> None:
        """Save ensemble to disk."""
        ...

    def load(self, path: Optional[Path] = None) -> None:
        """Load ensemble from disk."""
        ...
```

**When to Use**: ML model wrappers, ensemble models

**Quality Criteria**:
- Config object for parameters
- `is_trained` property
- `train()` returns metrics dict
- `predict()` checks is_trained
- save/load for persistence

---

### Pattern 4.8: CLI Script Pattern

**Description**: Command-line scripts with argparse, logging, and progress output.

**Examples**:
- `scripts/train_mtf_ensemble.py:1-456` - Full training script

**Code Pattern**:
```python
#!/usr/bin/env python3
"""Train Multi-Timeframe Ensemble model."""

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train MTF Ensemble")
    parser.add_argument("--data", type=str, default="data/forex/EURUSD.csv")
    parser.add_argument("--output", type=str, default="models/mtf_ensemble")
    parser.add_argument("--sentiment", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("MTF ENSEMBLE TRAINING")
    print("=" * 70)

    # ... training logic ...

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
```

**When to Use**: Training scripts, data processing, backtesting tools

**Quality Criteria**:
- Docstring with usage
- argparse with help text
- Progress output with separators

---

## 5. React Frontend Patterns

### Pattern 5.1: Dashboard Layout Pattern

**Description**: Main dashboard orchestrating data fetching and child components.

**Examples**:
- `frontend/src/components/Dashboard.jsx:1-206` - Complete dashboard
- `frontend/src/components/Dashboard.jsx:25-93` - usePolling hooks

**Code Pattern**:
```jsx
// frontend/src/components/Dashboard.jsx
import { useCallback } from 'react';
import { api } from '../api/client';
import { usePolling } from '../hooks/usePolling';

const INTERVALS = {
  prediction: 30000,    // 30 seconds
  candles: 60000,       // 1 minute
};

export function Dashboard() {
  const {
    data: prediction,
    loading: predictionLoading,
    error: predictionError,
    refetch: refetchPrediction,
  } = usePolling(
    useCallback(() => api.getPrediction(), []),
    INTERVALS.prediction
  );

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <header>...</header>
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <PredictionCard prediction={prediction} loading={predictionLoading} error={predictionError} />
          <PriceChart candles={candlesData} />
        </div>
      </main>
    </div>
  );
}
```

**Quality Criteria**:
- Polling intervals as constants
- useCallback for fetch functions
- Pass loading/error to children
- Responsive grid layout

---

### Pattern 5.2: Card Component Pattern

**Description**: Display card with loading, error, and data states.

**Examples**:
- `frontend/src/components/PredictionCard.jsx:1-139`

**Code Pattern**:
```jsx
// frontend/src/components/PredictionCard.jsx
import { AlertCircle, TrendingUp, TrendingDown } from 'lucide-react';

export function PredictionCard({ prediction, loading, error }) {
  // Loading state
  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="h-16 bg-gray-700 rounded mb-4"></div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 border border-red-500/30">
        <div className="flex items-center gap-2 text-red-400">
          <AlertCircle size={20} />
          <span>Error loading prediction</span>
        </div>
      </div>
    );
  }

  // Empty state
  if (!prediction) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <p className="text-gray-500">No prediction available</p>
      </div>
    );
  }

  // Data state
  const { signal, confidence } = prediction;

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-lg font-semibold">Current Prediction</h2>
      <span className={`text-4xl font-bold ${getSignalColor(signal)}`}>
        {signal}
      </span>
    </div>
  );
}
```

**Quality Criteria**:
- Handle loading/error/empty/data states
- Skeleton loader for loading
- Clear error message with icon
- TailwindCSS for styling

---

### Pattern 5.3: API Client Pattern

**Description**: Centralized API client with error handling.

**Examples**:
- `frontend/src/api/client.js:1-81`

**Code Pattern**:
```javascript
// frontend/src/api/client.js
const API_BASE = '/api';

class APIError extends Error {
  constructor(message, status, data) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.data = data;
  }
}

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const config = {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  };

  try {
    const response = await fetch(url, config);
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        errorData.detail || `HTTP error ${response.status}`,
        response.status,
        errorData
      );
    }
    return await response.json();
  } catch (error) {
    if (error instanceof APIError) throw error;
    throw new APIError(error.message || 'Network error', 0, null);
  }
}

export const api = {
  health: () => fetch('/health').then(r => r.json()),
  getPrediction: () => request('/predict'),
  getCandles: (symbol, timeframe, count) =>
    request(`/candles?symbol=${symbol}&timeframe=${timeframe}&count=${count}`),
};
```

**Quality Criteria**:
- Custom error class with status
- Base URL constant
- Each endpoint as method

---

## 6. Testing Patterns

### Pattern 6.1: Python API Test Pattern

**Description**: FastAPI endpoint tests with TestClient and mocked services.

**Examples**:
- `tests/api/test_predictions.py:1-134`

**Code Pattern**:
```python
# tests/api/test_predictions.py
import pytest
from unittest.mock import Mock
from fastapi.testclient import TestClient
from fastapi import FastAPI

class TestPredictionEndpoints:
    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up service mocks before each test."""
        self.mock_model_service = Mock()
        self.mock_model_service.is_loaded = True
        self.mock_model_service.get_model_info.return_value = {
            "loaded": True,
            "weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
        }

    def test_model_status_endpoint(self):
        from src.api.routes import predictions

        original = predictions.model_service
        predictions.model_service = self.mock_model_service

        try:
            app = FastAPI()
            app.include_router(predictions.router)
            client = TestClient(app)

            response = client.get("/models/status")
            assert response.status_code == 200
            assert response.json()["loaded"] is True
        finally:
            predictions.model_service = original
```

**Quality Criteria**:
- autouse fixture for common setup
- Proper service restoration in finally
- Test both success and error paths

---

### Pattern 6.2: React Component Test Pattern

**Description**: Vitest tests with Testing Library.

**Examples**:
- `frontend/src/components/PredictionCard.test.jsx:1-84`

**Code Pattern**:
```jsx
// frontend/src/components/PredictionCard.test.jsx
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PredictionCard } from './PredictionCard';

describe('PredictionCard', () => {
  it('renders loading state', () => {
    render(<PredictionCard loading={true} />);
    const skeleton = document.querySelector('.animate-pulse');
    expect(skeleton).toBeInTheDocument();
  });

  it('renders error state', () => {
    render(<PredictionCard error="Test error" />);
    expect(screen.getByText('Error loading prediction')).toBeInTheDocument();
  });

  it('renders BUY prediction correctly', () => {
    const prediction = { signal: 'BUY', confidence: 0.72 };
    render(<PredictionCard prediction={prediction} />);
    expect(screen.getByText('BUY')).toBeInTheDocument();
    expect(screen.getByText('72.0%')).toBeInTheDocument();
  });
});
```

**Quality Criteria**:
- Test all states (loading, error, empty, data)
- Use screen.getByText for assertions
- Descriptive test names

---

## 7. Development Workflow Patterns

### Feature Implementation Workflow

Based on git history, features follow this pattern:

1. **Documentation First**: Create/update `docs/XX-feature-name.md`
2. **Core Implementation**: Add to `src/` with proper module structure
3. **Script Creation**: Add `scripts/feature_script.py` for CLI access
4. **CLAUDE.md Update**: Update project guide with new capabilities
5. **Results Storage**: Add results to `results/` directory

**Recent Examples**:
```
12060ed feat: Implement market regime detection
  - docs/20-regime-detection-analysis.md
  - src/features/regime/regime_detector.py
  - scripts/analyze_regime_performance.py

dc17a92 feat: Implement confidence threshold optimization
  - docs/19-confidence-threshold-optimization.md
  - scripts/optimize_confidence_threshold.py
```

### Commit Message Convention

```
feat: <description>           # New feature
fix: <description>            # Bug fix
docs: <description>           # Documentation
refactor: <description>       # Refactoring
chore: <description>          # Maintenance

# With scopes:
feat(sentiment): Add sentiment analysis
feat(api): Add prediction endpoint
```

---

## 8. Domain-Specific Patterns

### Technical Indicator Calculator

**Description**: Configuration-driven indicator calculation.

**Examples**:
- `src/features/technical/calculator.py:47-411`

**Key Features**:
- YAML configuration loading
- Registry-based or legacy calculation
- Model-type specific configs
- Feature grouping by category

### Kelly Criterion Position Sizing

**Description**: Position sizing with Kelly variants.

**Examples**:
- `src/trading/position_sizing.py:179-397`

**Key Features**:
- Multiple strategies (Fixed, Half Kelly, etc.)
- Confidence-based scaling
- Regime-based adjustments
- Position limits

---

## 9. Naming Conventions

### Files

| Type | Convention | Example |
|------|------------|---------|
| Python modules | snake_case | `model_service.py` |
| Python classes | PascalCase | `ModelService` |
| React components | PascalCase | `PredictionCard.jsx` |
| Test files | `test_*.py` or `*.test.jsx` | `test_predictions.py` |

### Code

| Type | Convention | Example |
|------|------------|---------|
| Functions | snake_case | `get_prediction()` |
| Variables | snake_case | `model_service` |
| Constants | UPPER_SNAKE | `CACHE_TTL` |
| Classes | PascalCase | `MTFEnsemble` |
| Private | _prefix | `_load_model()` |

---

## 10. Recommended Skills

Based on these patterns, the following Claude Code Skills would be valuable:

### High Priority Skills

| Skill Name | Description | Key Patterns |
|------------|-------------|--------------|
| **creating-fastapi-endpoints** | Creates API endpoints with proper patterns | 4.3, 4.4 |
| **creating-react-components** | Creates React components with state handling | 5.2, 5.3 |
| **creating-python-services** | Creates singleton service classes | 4.2 |
| **creating-ml-models** | Creates ML model classes with train/predict/save | 4.7 |
| **writing-pytest-tests** | Writes API tests with mocked services | 6.1 |
| **creating-cli-scripts** | Creates CLI scripts with argparse | 4.8 |

### Skill Structure Template

```
skill-name/
├── SKILL.md           # Main instructions (< 500 lines)
├── PATTERNS.md        # Code pattern templates
└── CHECKLIST.md       # Quality checklist
```

Each skill should include:
- SKILL.md with pattern templates
- Example code snippets
- When to use/not use guidance
- Quality checklist
