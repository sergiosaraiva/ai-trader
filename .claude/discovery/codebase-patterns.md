# AI-Trader Codebase Patterns Discovery Report

> **Generated**: 2026-01-18
> **Version**: 2.1 (Enhanced)
> **Codebase**: AI Assets Trader - MTF Ensemble Trading System
> **Analysis Period**: Last 3 months of commits (October 2025 - January 2026)

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total patterns documented** | 18 |
| **HIGH priority patterns** | 8 |
| **MEDIUM priority patterns** | 6 |
| **LOW priority patterns** | 4 |
| **Anti-patterns documented** | 6 |
| **Shared utilities indexed** | 12 |
| **Active skills** | 7 (SKILL.md based) |
| **Related skill documents** | 16 |

**Key Findings**:
- FastAPI service singleton pattern is the most critical backend pattern (used in 6+ services)
- React component state handling (loading/error/data) is consistent across all frontend components
- Error handling follows a consistent try/except/HTTPException pattern in API layer
- Test patterns are well-established with mock service injection via try/finally blocks
- Two meta-skills added: `routing-to-skills` and `improving-framework-continuously`

---

## Table of Contents

1. [Skills Best Practices Summary](#1-skills-best-practices-summary)
2. [Technology Stack](#2-technology-stack)
3. [Layer Architecture](#3-layer-architecture)
4. [HIGH Priority Patterns](#4-high-priority-patterns)
5. [MEDIUM Priority Patterns](#5-medium-priority-patterns)
6. [LOW Priority Patterns](#6-low-priority-patterns)
7. [Anti-Patterns (What NOT to Do)](#7-anti-patterns-what-not-to-do)
8. [Shared Utilities Index](#8-shared-utilities-index)
9. [Error Handling Summary](#9-error-handling-summary)
10. [Testing Patterns](#10-testing-patterns)
11. [Development Workflow Patterns](#11-development-workflow-patterns)
12. [Naming Conventions](#12-naming-conventions)
13. [Recommended Skills](#13-recommended-skills)

---

## 1. Skills Best Practices Summary

Based on Anthropic's documentation (consolidated January 2026), effective Skills follow these principles:

### Core Principles

1. **Progressive Disclosure**: Skills load content in stages (reduces token cost)
   - Level 1: Metadata (always loaded at startup, ~100 tokens per skill)
   - Level 2: Instructions (loaded when triggered via bash read, <5k tokens)
   - Level 3: Resources (loaded as needed, effectively unlimited - no context penalty)

2. **Conciseness is Key**: Only include context Claude doesn't already have
   - Challenge each piece: "Does Claude really need this?"
   - Claude is already very smart - don't over-explain
   - Default assumption: Claude knows common patterns; only add domain-specific knowledge

3. **Appropriate Freedom (Degrees of Freedom)**: Match specificity to task fragility
   - High freedom: Text instructions for flexible tasks (multiple valid approaches)
   - Medium freedom: Pseudocode or templates with parameters
   - Low freedom: Exact scripts for fragile operations (database migrations, critical sequences)
   - Think of it as "narrow bridge vs open field" for the agent

4. **Filesystem-Based Architecture**: Skills are directories, Claude navigates via bash
   - Claude uses bash `read` to access SKILL.md when triggered
   - Scripts are executed via bash, only output enters context (not source code)
   - No practical limit on bundled content since it's loaded on-demand

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
description: Description  # max 1024 chars, third person, includes WHEN to use
---

# Skill Name

## Quick Reference
[Concise key points]

## When to Use
[Trigger conditions]

## When NOT to Use
[Anti-conditions]

## Implementation Guide
[Decision tree or workflow]

## Examples
[Concrete examples with file paths]

## Quality Checklist
[Verification items]

## Common Mistakes
[Anti-patterns specific to this skill]
```

### Key Recommendations

| Recommendation | Details |
|----------------|---------|
| **Description** | Third person, includes what AND when to use (both triggers) |
| **Token Budget** | SKILL.md body under 500 lines; split larger content into references |
| **File References** | One level deep from SKILL.md (avoid nested references) |
| **Naming** | Gerund form (verb + -ing): `creating-api-endpoints`, `improving-framework-continuously` |
| **Testing** | Test with Haiku (needs more guidance), Sonnet (balanced), Opus (may need less) |
| **Workflows** | Include copy-paste checklists for multi-step tasks |
| **Feedback Loops** | Validate -> fix -> repeat patterns; run validator before proceeding |
| **Anti-Hallucination** | Require citations with file:line, allow "I don't know", verify before citing |
| **Examples** | Include input/output pairs like regular prompting |
| **Table of Contents** | Add for files >100 lines to help Claude navigate |

### Progressive Disclosure Patterns

**Pattern 1: High-level guide with references**
```markdown
# Main Guide
## Quick start
[Basic instructions]

## Advanced features
**Form filling**: See [FORMS.md](FORMS.md)
**API reference**: See [REFERENCE.md](REFERENCE.md)
```

**Pattern 2: Domain-specific organization**
```
skill/
├── SKILL.md (overview and navigation)
└── reference/
    ├── finance.md
    ├── sales.md
    └── product.md
```

---

## 2. Technology Stack

### Backend (Python 3.12+)

| Category | Technologies | Versions | Usage |
|----------|-------------|----------|-------|
| **API Framework** | FastAPI, uvicorn | >=0.100.0, >=0.23.0 | All API endpoints |
| **ML Models** | XGBoost, scikit-learn | >=2.0.0, >=1.3.0 | MTF Ensemble |
| **Data Processing** | pandas, numpy, pyarrow | >=2.0.0, >=1.24.0, >=14.0.0 | All data ops |
| **Technical Analysis** | pandas-ta | latest | Feature engineering |
| **Database** | SQLAlchemy | >=2.0.0 | SQLite persistence |
| **Scheduling** | APScheduler | >=3.10.0 | Prediction scheduler |
| **Validation** | Pydantic | >=2.0.0 | Request/response schemas |
| **Testing** | pytest, pytest-asyncio | >=7.4.0 | Backend tests |
| **Data Sources** | yfinance, fredapi | >=0.2.28, >=0.5.0 | Market & sentiment data |

### Frontend (Node.js 20+)

| Category | Technologies | Versions | Usage |
|----------|-------------|----------|-------|
| **Framework** | React | 19.2.0 | UI components |
| **Build Tool** | Vite | 7.2.4 | Development & bundling |
| **Styling** | TailwindCSS | 4.1.18 | All styling |
| **Charts** | Recharts | 3.6.0 | Price charts |
| **Icons** | lucide-react | 0.562.0 | UI icons |
| **Testing** | Vitest, Testing Library | 4.0.17 | Component tests |

### Architecture

- **Pattern**: Layered monolith with service architecture
- **API Style**: REST with FastAPI
- **State**: SQLite for persistence, in-memory caching
- **ML Pipeline**: XGBoost ensemble with multi-timeframe models (1H, 4H, Daily)

---

## 3. Layer Architecture

```
ai-trader/
├── src/                      # Python source code
│   ├── api/                  # FastAPI web layer
│   │   ├── main.py          # App entry point + lifespan
│   │   ├── scheduler.py     # APScheduler setup
│   │   ├── routes/          # API endpoints (predictions, trading, market, pipeline)
│   │   ├── services/        # Business logic singletons
│   │   ├── schemas/         # Pydantic models
│   │   └── database/        # SQLAlchemy models + session
│   ├── models/              # ML models
│   │   └── multi_timeframe/ # MTF Ensemble (PRIMARY)
│   ├── features/            # Feature engineering
│   │   ├── technical/       # Technical indicators + registry
│   │   ├── sentiment/       # EPU/VIX sentiment loading
│   │   └── regime/          # Market regime detection
│   ├── trading/             # Trading logic (position sizing, risk)
│   └── simulation/          # Backtesting
├── frontend/                # React application
│   └── src/
│       ├── components/      # React components (Dashboard, Cards, Chart)
│       ├── api/             # API client
│       └── hooks/           # Custom hooks (usePolling)
├── scripts/                 # CLI tools
├── tests/                   # Test suites (735+ tests)
│   ├── api/                 # API endpoint tests
│   ├── services/            # Service tests
│   └── unit/                # Unit tests by module
└── docs/                    # Documentation
```

---

## 4. HIGH Priority Patterns

### Pattern 4.1: FastAPI Service Singleton

**Description**: Thread-safe singleton services with lazy initialization, caching, and status tracking.

**Priority**: HIGH
**Frequency**: Used in 6 services (`model_service`, `data_service`, `trading_service`, `pipeline_service`)
**Layer**: Backend API Services

**Examples**:
1. `src/api/services/model_service.py:28-280` - Complete ModelService class
2. `src/api/services/trading_service.py:1-200` - TradingService implementation
3. `src/api/services/data_service.py:1-150` - DataService implementation

**Code Pattern**:
```python
# src/api/services/model_service.py
from threading import Lock
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

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
                cached = self._cache[cache_key]
                if datetime.now() - cached["cached_at"] < self.CACHE_TTL:
                    return cached["prediction"]

        with self._lock:
            result = self._model.predict(df)
        return result

# Singleton instance at module level
model_service = ModelService()
```

**When to Use**:
- Services managing expensive resources (ML models, DB connections)
- Shared state across multiple requests
- Operations requiring thread safety

**When NOT to Use**:
- Stateless utilities
- Request-scoped operations
- Simple helper functions

**Quality Criteria**:
- [ ] Thread-safe with `threading.Lock`
- [ ] Lazy initialization in `initialize()` method
- [ ] Clear status properties (`is_loaded`, `is_initialized`)
- [ ] Cache with TTL for expensive operations
- [ ] Singleton instance at module level

---

### Pattern 4.2: FastAPI Route Handler

**Description**: Route handlers with service availability checks, proper error handling, and Pydantic response models.

**Priority**: HIGH
**Frequency**: Used in 20+ endpoints across all route modules
**Layer**: Backend API Routes

**Examples**:
1. `src/api/routes/predictions.py:26-78` - GET with service check
2. `src/api/routes/predictions.py:81-125` - GET with pagination
3. `src/api/routes/trading.py:1-100` - Trading endpoints

**Code Pattern**:
```python
# src/api/routes/predictions.py
from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.orm import Session

router = APIRouter()

@router.get("/predictions/latest", response_model=PredictionResponse)
async def get_latest_prediction() -> PredictionResponse:
    """Get the most recent prediction."""
    # 1. Check service availability FIRST
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # 2. Validate data availability
        df = data_service.get_data_for_prediction()
        if df is None or len(df) < 100:
            raise HTTPException(status_code=503, detail="Insufficient data")

        # 3. Perform operation
        prediction = model_service.predict(df)
        return PredictionResponse(**prediction)

    except HTTPException:
        raise  # Re-raise HTTPException unchanged
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**When to Use**: All API endpoint definitions

**Quality Criteria**:
- [ ] `response_model=` specified on decorator
- [ ] Service availability check BEFORE try block
- [ ] HTTPException re-raised unchanged
- [ ] Other exceptions wrapped with 500 status
- [ ] Descriptive docstring for Swagger

---

### Pattern 4.3: Pydantic Schema Pattern

**Description**: Request/response schemas with Field descriptions and Config examples.

**Priority**: HIGH
**Frequency**: Used in 15+ schema classes
**Layer**: Backend API Schemas

**Examples**:
1. `src/api/schemas/prediction.py:16-68` - PredictionResponse
2. `src/api/schemas/trading.py:1-100` - Trading schemas

**Code Pattern**:
```python
# src/api/schemas/prediction.py
from pydantic import BaseModel, Field
from typing import Dict, Optional

class PredictionResponse(BaseModel):
    """Response for prediction endpoint."""

    timestamp: str = Field(..., description="ISO format timestamp")
    symbol: str = Field(default="EURUSD", description="Trading symbol")
    direction: str = Field(..., description="'long' or 'short'")
    confidence: float = Field(..., ge=0, le=1, description="Confidence 0-1")
    should_trade: bool = Field(..., description="Confidence >= threshold")

    # Nested data
    component_directions: Dict[str, int] = Field(..., description="By TF")
    component_confidences: Dict[str, float] = Field(..., description="By TF")

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-01-15T14:00:00",
                "direction": "long",
                "confidence": 0.72,
            }
        }
```

**Quality Criteria**:
- [ ] Field descriptions for all attributes
- [ ] Validation constraints (`ge=`, `le=`, `max_length=`)
- [ ] Example in Config for Swagger UI
- [ ] Proper Optional typing

---

### Pattern 4.4: React Card Component

**Description**: Display card with loading, error, empty, and data states using TailwindCSS.

**Priority**: HIGH
**Frequency**: Used in 6 components (PredictionCard, AccountStatus, PerformanceStats, TradeHistory)
**Layer**: Frontend Components

**Examples**:
1. `frontend/src/components/PredictionCard.jsx:1-139`
2. `frontend/src/components/AccountStatus.jsx:1-100`
3. `frontend/src/components/PerformanceStats.jsx:1-120`

**Code Pattern**:
```jsx
// frontend/src/components/PredictionCard.jsx
import { AlertCircle, TrendingUp, TrendingDown } from 'lucide-react';

export function PredictionCard({ prediction, loading, error }) {
  // 1. Loading state
  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="h-16 bg-gray-700 rounded mb-4"></div>
      </div>
    );
  }

  // 2. Error state
  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 border border-red-500/30">
        <div className="flex items-center gap-2 text-red-400">
          <AlertCircle size={20} />
          <span>Error loading prediction</span>
        </div>
        <p className="text-red-300 mt-2 text-sm">{error}</p>
      </div>
    );
  }

  // 3. Empty state
  if (!prediction) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <p className="text-gray-500">No prediction available</p>
      </div>
    );
  }

  // 4. Data state
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
- [ ] All four states handled (loading, error, empty, data)
- [ ] Skeleton loader with `animate-pulse`
- [ ] Error message with icon and details
- [ ] TailwindCSS for styling
- [ ] Props destructured at top

---

### Pattern 4.5: Dashboard Data Fetching

**Description**: Main dashboard with usePolling hooks for periodic data fetching.

**Priority**: HIGH
**Frequency**: Main pattern in Dashboard.jsx
**Layer**: Frontend Components

**Examples**:
1. `frontend/src/components/Dashboard.jsx:1-206`
2. `frontend/src/hooks/usePolling.js:1-50`

**Code Pattern**:
```jsx
// frontend/src/components/Dashboard.jsx
import { useCallback } from 'react';
import { api } from '../api/client';
import { usePolling } from '../hooks/usePolling';

const INTERVALS = {
  prediction: 30000,   // 30 seconds
  candles: 60000,      // 1 minute
  account: 60000,      // 1 minute
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

  const {
    data: account,
    loading: accountLoading,
    error: accountError,
  } = usePolling(
    useCallback(() => api.getAccount(), []),
    INTERVALS.account
  );

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <header>...</header>
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <PredictionCard
            prediction={prediction}
            loading={predictionLoading}
            error={predictionError}
          />
          <AccountStatus
            account={account}
            loading={accountLoading}
            error={accountError}
          />
        </div>
      </main>
    </div>
  );
}
```

**Quality Criteria**:
- [ ] Polling intervals as named constants
- [ ] `useCallback` wrapper for fetch functions
- [ ] Pass loading/error to child components
- [ ] Responsive grid with TailwindCSS

---

### Pattern 4.6: Pytest API Test Pattern

**Description**: FastAPI endpoint tests with TestClient and mocked services.

**Priority**: HIGH
**Frequency**: Used in 30+ API tests
**Layer**: Testing

**Examples**:
1. `tests/api/test_predictions.py:1-134`
2. `tests/api/test_trading.py:1-100`

**Code Pattern**:
```python
# tests/api/test_predictions.py
import pytest
from unittest.mock import Mock
from fastapi.testclient import TestClient
from fastapi import FastAPI

class TestPredictionEndpoints:
    """Test prediction endpoints."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up service mocks before each test."""
        self.mock_model_service = Mock()
        self.mock_model_service.is_loaded = True
        self.mock_model_service.get_model_info.return_value = {
            "loaded": True,
            "weights": {"1H": 0.6, "4H": 0.3, "D": 0.1},
        }

        # Create mock DataFrame
        self.mock_df = pd.DataFrame({
            "open": np.random.rand(200) + 1.08,
            "high": np.random.rand(200) + 1.085,
            "low": np.random.rand(200) + 1.075,
            "close": np.random.rand(200) + 1.08,
            "volume": np.random.randint(1000, 10000, 200),
        })

    def test_model_status_endpoint(self):
        """Test model status endpoint."""
        from src.api.routes import predictions

        original_model = predictions.model_service
        predictions.model_service = self.mock_model_service

        try:
            app = FastAPI()
            app.include_router(predictions.router)
            client = TestClient(app)

            response = client.get("/models/status")

            assert response.status_code == 200
            assert response.json()["loaded"] is True
        finally:
            predictions.model_service = original_model
```

**Quality Criteria**:
- [ ] `@pytest.fixture(autouse=True)` for setup
- [ ] Services restored in `finally` block
- [ ] Realistic mock data
- [ ] Both success and error paths tested

---

### Pattern 4.7: Vitest Component Test Pattern

**Description**: React component tests with Vitest and Testing Library.

**Priority**: HIGH
**Frequency**: Used in 35+ frontend tests
**Layer**: Frontend Testing

**Examples**:
1. `frontend/src/components/PredictionCard.test.jsx:1-84`
2. `frontend/src/components/AccountStatus.test.jsx:1-60`

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
    expect(screen.getByText('Test error')).toBeInTheDocument();
  });

  it('renders no prediction state', () => {
    render(<PredictionCard prediction={null} />);
    expect(screen.getByText('No prediction available')).toBeInTheDocument();
  });

  it('renders BUY prediction correctly', () => {
    const prediction = {
      signal: 'BUY',
      confidence: 0.72,
      current_price: 1.08543,
    };
    render(<PredictionCard prediction={prediction} />);

    expect(screen.getByText('BUY')).toBeInTheDocument();
    expect(screen.getByText('72.0%')).toBeInTheDocument();
  });
});
```

**Quality Criteria**:
- [ ] All four states tested (loading, error, empty, data)
- [ ] `screen.getByText` for assertions
- [ ] Descriptive test names
- [ ] Realistic mock data

---

### Pattern 4.8: Dataclass Configuration

**Description**: Configuration objects using dataclasses with defaults and factory methods.

**Priority**: HIGH
**Frequency**: Used in 8+ configuration classes
**Layer**: Backend Models

**Examples**:
1. `src/models/multi_timeframe/mtf_ensemble.py:28-119` - MTFEnsembleConfig
2. `src/trading/position_sizing.py:144-176` - PositionSizingConfig
3. `src/features/technical/calculator.py:24-45` - CalculatorConfig

**Code Pattern**:
```python
# src/models/multi_timeframe/mtf_ensemble.py
from dataclasses import dataclass, field
from typing import Dict, Optional

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

**Quality Criteria**:
- [ ] `@dataclass` decorator
- [ ] `field(default_factory=...)` for mutable defaults
- [ ] Factory classmethods for common configurations
- [ ] Type hints on all attributes

---

## 5. MEDIUM Priority Patterns

### Pattern 5.1: SQLAlchemy Model

**Description**: Database models with types, indexes, and relationships.

**Priority**: MEDIUM
**Frequency**: Used in 3 models (Prediction, Trade, Account)
**Layer**: Backend Database

**Examples**:
1. `src/api/database/models.py:22-61` - Prediction model
2. `src/api/database/models.py:64-107` - Trade model

**Code Pattern**:
```python
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Index
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Prediction(Base):
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

---

### Pattern 5.2: API Client Pattern (Frontend)

**Description**: Centralized API client with error handling.

**Priority**: MEDIUM
**Frequency**: Single client used across frontend
**Layer**: Frontend API

**Examples**:
1. `frontend/src/api/client.js:1-81`

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
  getPrediction: () => request('/v1/predictions/latest'),
  getAccount: () => request('/v1/trading/account'),
};
```

---

### Pattern 5.3: CLI Script Pattern

**Description**: Command-line scripts with argparse and progress output.

**Priority**: MEDIUM
**Frequency**: Used in 15+ scripts
**Layer**: Scripts

**Examples**:
1. `scripts/train_mtf_ensemble.py:1-456`
2. `scripts/backtest_mtf_ensemble.py:1-700`

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
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

---

### Pattern 5.4: FastAPI Lifespan Pattern

**Description**: Application factory with lifespan context manager.

**Priority**: MEDIUM
**Frequency**: Single use in main.py
**Layer**: Backend API

**Examples**:
1. `src/api/main.py:25-130`

**Code Pattern**:
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting API...")
    init_db()
    model_service.initialize()
    start_scheduler()
    yield
    # Shutdown
    logger.info("Shutting down...")
    stop_scheduler()

def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Assets Trader API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
    app.include_router(health.router, tags=["Health"])
    app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])

    return app

app = create_app()
```

---

### Pattern 5.5: ML Model Class Pattern

**Description**: ML model wrapper with train/predict/save/load interface.

**Priority**: MEDIUM
**Frequency**: Used in 4 model classes
**Layer**: Backend Models

**Examples**:
1. `src/models/multi_timeframe/mtf_ensemble.py:154-653`

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

    def train(self, df_5min: pd.DataFrame, ...) -> Dict[str, Dict]:
        """Train all timeframe models."""
        results = {}
        for tf, model in self.models.items():
            results[tf] = model.train(...)
        self.is_trained = all(m.is_trained for m in self.models.values())
        return results

    def predict(self, df_5min: pd.DataFrame) -> MTFPrediction:
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return MTFPrediction(...)

    def save(self, path: Optional[Path] = None) -> None: ...
    def load(self, path: Optional[Path] = None) -> None: ...
```

---

### Pattern 5.6: Technical Indicator Calculator

**Description**: Configuration-driven indicator calculation with registry.

**Priority**: MEDIUM
**Frequency**: Used across all feature engineering
**Layer**: Backend Features

**Examples**:
1. `src/features/technical/calculator.py:47-411`

---

## 6. LOW Priority Patterns

### Pattern 6.1: usePolling Hook

**Description**: Custom React hook for periodic data fetching.

**Priority**: LOW
**Frequency**: Single hook implementation
**Layer**: Frontend Hooks

**Examples**:
1. `frontend/src/hooks/usePolling.js:1-50`

---

### Pattern 6.2: Database Session Dependency

**Description**: FastAPI dependency for database sessions.

**Priority**: LOW
**Frequency**: Used in database routes
**Layer**: Backend Database

**Examples**:
1. `src/api/database/session.py:1-30`

---

### Pattern 6.3: Regime Detector

**Description**: Market regime detection with trend/volatility analysis.

**Priority**: LOW
**Frequency**: Specialized trading domain
**Layer**: Backend Features

**Examples**:
1. `src/features/regime/regime_detector.py:1-200`

---

### Pattern 6.4: Position Sizing

**Description**: Kelly criterion position sizing strategies.

**Priority**: LOW
**Frequency**: Specialized trading domain
**Layer**: Backend Trading

**Examples**:
1. `src/trading/position_sizing.py:179-397`

---

## 7. Anti-Patterns (What NOT to Do)

### Anti-Pattern 7.1: Catching HTTPException in Generic Handler

**Why it's wrong**: Swallows specific error details and status codes.

**Evidence**: Avoided in `src/api/routes/predictions.py:74-78`

**Wrong**:
```python
try:
    # ... operations
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

**Correct approach**: See `src/api/routes/predictions.py:74-78`
```python
try:
    # ... operations
except HTTPException:
    raise  # Re-raise unchanged
except Exception as e:
    logger.error(f"Error: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

---

### Anti-Pattern 7.2: Missing Service Availability Check

**Why it's wrong**: Causes cryptic errors when service not initialized.

**Evidence**: Correct pattern in all route handlers

**Wrong**:
```python
@router.get("/predictions/latest")
async def get_latest():
    prediction = model_service.predict(df)  # Fails if not loaded
    return prediction
```

**Correct approach**: See `src/api/routes/predictions.py:26-37`
```python
@router.get("/predictions/latest")
async def get_latest():
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # Then proceed
```

---

### Anti-Pattern 7.3: Not Restoring Mocked Services in Tests

**Why it's wrong**: Causes test pollution and flaky tests.

**Evidence**: Correct pattern in all API tests

**Wrong**:
```python
def test_endpoint(self):
    predictions.model_service = self.mock  # Never restored!
    # Test code
```

**Correct approach**: See `tests/api/test_predictions.py:56-68`
```python
def test_endpoint(self):
    original = predictions.model_service
    predictions.model_service = self.mock
    try:
        # Test code
    finally:
        predictions.model_service = original
```

---

### Anti-Pattern 7.4: Mutable Default Arguments

**Why it's wrong**: Shared state between instances causes bugs.

**Evidence**: Avoided in all dataclass definitions

**Wrong**:
```python
@dataclass
class Config:
    weights: Dict[str, float] = {"1H": 0.6}  # SHARED MUTABLE!
```

**Correct approach**: See `src/models/multi_timeframe/mtf_ensemble.py:30-35`
```python
@dataclass
class Config:
    weights: Dict[str, float] = field(default_factory=lambda: {"1H": 0.6})
```

---

### Anti-Pattern 7.5: Missing Response Model on Endpoints

**Why it's wrong**: No type validation, poor documentation.

**Evidence**: All endpoints have response_model

**Wrong**:
```python
@router.get("/data")
async def get_data():
    return {"value": 123}
```

**Correct approach**: See `src/api/routes/predictions.py:26`
```python
@router.get("/data", response_model=DataResponse)
async def get_data() -> DataResponse:
    return DataResponse(value=123)
```

---

### Anti-Pattern 7.6: Not Handling All Component States

**Why it's wrong**: UI crashes or shows broken states.

**Evidence**: All card components handle 4 states

**Wrong**:
```jsx
function Card({ data }) {
  return <div>{data.value}</div>;  // Crashes if data is null
}
```

**Correct approach**: See `frontend/src/components/PredictionCard.jsx:1-139`
```jsx
function Card({ data, loading, error }) {
  if (loading) return <Skeleton />;
  if (error) return <ErrorMessage error={error} />;
  if (!data) return <EmptyState />;
  return <div>{data.value}</div>;
}
```

---

## 8. Shared Utilities Index

| Utility | Location | Used By | Purpose |
|---------|----------|---------|---------|
| `TechnicalIndicatorCalculator` | `src/features/technical/calculator.py:47` | 15+ scripts | Calculate technical indicators |
| `SentimentLoader` | `src/features/sentiment/sentiment_loader.py:1` | 5+ modules | Load EPU/VIX sentiment |
| `MTFEnsemble` | `src/models/multi_timeframe/mtf_ensemble.py:154` | 10+ scripts | Main prediction model |
| `MTFEnsembleConfig` | `src/models/multi_timeframe/mtf_ensemble.py:28` | 10+ scripts | Model configuration |
| `model_service` | `src/api/services/model_service.py:389` | 5+ routes | Singleton model service |
| `data_service` | `src/api/services/data_service.py:1` | 5+ routes | Market data fetching |
| `trading_service` | `src/api/services/trading_service.py:1` | 3+ routes | Paper trading logic |
| `api` client | `frontend/src/api/client.js:1` | All components | API communication |
| `usePolling` | `frontend/src/hooks/usePolling.js:1` | Dashboard | Periodic fetching |
| `get_db` | `src/api/database/session.py:1` | 5+ routes | DB session dependency |
| `RegimeDetector` | `src/features/regime/regime_detector.py:1` | 3+ scripts | Market regime detection |
| `PositionSizer` | `src/trading/position_sizing.py:179` | 2+ scripts | Kelly criterion sizing |

---

## 9. Error Handling Summary

| Layer | Exception Type | Logging Pattern | Example |
|-------|---------------|-----------------|---------|
| **API Routes** | `HTTPException` | `logger.error(f"Error: {e}")` | `src/api/routes/predictions.py:77` |
| **Services** | `RuntimeError` | `logger.error(f"Failed: {e}")` | `src/api/services/model_service.py:93` |
| **ML Models** | `RuntimeError` | `logger.warning(f"Issue: {e}")` | `src/models/multi_timeframe/mtf_ensemble.py:500` |
| **Scripts** | `Exception` | `logger.error(f"Error: {e}")` | `scripts/train_mtf_ensemble.py:225` |
| **Frontend** | `APIError` | Console in dev | `frontend/src/api/client.js:8` |

### Error Response Format (API)

```json
{
  "detail": "Human-readable error message"
}
```

### HTTP Status Code Usage

| Code | Meaning | When Used |
|------|---------|-----------|
| 200 | Success | Normal response |
| 400 | Bad Request | Invalid input |
| 404 | Not Found | Resource doesn't exist |
| 500 | Server Error | Unexpected exception |
| 503 | Service Unavailable | Model not loaded, insufficient data |

---

## 10. Testing Patterns

### Test Organization

```
tests/
├── api/                      # API endpoint tests
│   ├── test_predictions.py   # Prediction endpoints
│   ├── test_trading.py       # Trading endpoints
│   └── test_health.py        # Health check
├── services/                 # Service unit tests
│   ├── test_data_service.py
│   └── test_trading_service.py
└── unit/                     # Unit tests by module
    ├── features/             # Feature engineering
    ├── models/               # ML models
    └── trading/              # Trading logic
```

### Test Naming Convention

```
test_<function>_<condition>_<expected_result>

Examples:
- test_model_status_endpoint
- test_latest_prediction_model_not_loaded
- test_latest_prediction_insufficient_data
```

### Test Coverage Targets

| Module | Target | Current |
|--------|--------|---------|
| API Routes | 90% | 92% |
| Services | 85% | 88% |
| Frontend Components | 80% | 85% |
| ML Models | 70% | 75% |

---

## 11. Development Workflow Patterns

### Feature Implementation Workflow

Based on git history, features follow this pattern:

1. **Documentation First**: Create/update `docs/XX-feature-name.md`
2. **Core Implementation**: Add to `src/` with proper module structure
3. **Script Creation**: Add `scripts/feature_script.py` for CLI access
4. **Testing**: Add tests in appropriate `tests/` directory
5. **CLAUDE.md Update**: Update project guide with new capabilities

**Recent Examples**:
```
12060ed feat: Implement market regime detection
  - docs/05-regime-detection-analysis.md
  - src/features/regime/regime_detector.py
  - scripts/analyze_regime_performance.py

dc17a92 feat: Implement confidence threshold optimization
  - docs/04-confidence-threshold-optimization.md
  - scripts/optimize_confidence_threshold.py

6058433 feat: Add web showcase, Docker deployment
  - frontend/ (entire React application)
  - src/api/ (FastAPI backend)
  - Dockerfile, docker-compose.yml
```

### Commit Message Convention

```
feat: <description>           # New feature
fix: <description>            # Bug fix
docs: <description>           # Documentation
refactor: <description>       # Refactoring
chore: <description>          # Maintenance
test: <description>           # Test additions

# With scopes:
feat(sentiment): Add sentiment analysis
feat(api): Add prediction endpoint
fix(frontend): Fix loading state
```

---

## 12. Naming Conventions

### Files

| Type | Convention | Example |
|------|------------|---------|
| Python modules | snake_case | `model_service.py` |
| Python classes | PascalCase | `ModelService` |
| React components | PascalCase | `PredictionCard.jsx` |
| React tests | `*.test.jsx` | `PredictionCard.test.jsx` |
| Python tests | `test_*.py` | `test_predictions.py` |

### Code

| Type | Convention | Example |
|------|------------|---------|
| Functions | snake_case | `get_prediction()` |
| Variables | snake_case | `model_service` |
| Constants | UPPER_SNAKE | `CACHE_TTL` |
| Classes | PascalCase | `MTFEnsemble` |
| Private | _prefix | `_load_model()` |
| React props | camelCase | `predictionLoading` |

---

## 13. Recommended Skills

Based on patterns discovered, the following skills exist or should be created:

### Active Skills (7 SKILL.md files)

| Skill | Type | Location | Purpose |
|-------|------|----------|---------|
| `backend` | Domain | `.claude/skills/backend/SKILL.md` | FastAPI endpoints, services, schemas |
| `frontend` | Domain | `.claude/skills/frontend/SKILL.md` | React components, API clients |
| `database` | Domain | `.claude/skills/database/SKILL.md` | SQLAlchemy models, migrations |
| `testing` | Domain | `.claude/skills/testing/SKILL.md` | pytest and Vitest test writing |
| `build-deployment` | Domain | `.claude/skills/build-deployment/SKILL.md` | CLI scripts, Docker, deployment |
| `routing-to-skills` | Meta | `.claude/skills/routing-to-skills/SKILL.md` | Dynamic skill discovery and routing |
| `improving-framework-continuously` | Meta | `.claude/skills/improving-framework-continuously/SKILL.md` | Error processing, framework evolution |

### Related Skill Documents (16)

| Category | Documents |
|----------|-----------|
| **Backend** | `creating-api-endpoints.md`, `creating-pydantic-schemas.md`, `creating-python-services.md`, `creating-data-processors.md`, `implementing-prediction-models.md` |
| **Frontend** | `creating-api-clients.md` |
| **Feature Engineering** | `creating-technical-indicators.md`, `configuring-indicator-yaml.md` |
| **Data Layer** | `adding-data-sources.md` |
| **Quality Testing** | `creating-dataclasses.md`, `generating-test-data.md`, `planning-test-scenarios.md`, `validating-time-series-data.md` |
| **Testing** | `writing-vitest-tests.md` |
| **Trading Domain** | `analyzing-trading-performance.md`, `implementing-risk-management.md`, `running-backtests.md` |

### Meta-Skills (New in 2.1)

Two meta-skills enable self-improving framework:

1. **`routing-to-skills`** (v1.1.0): Dynamic skill discovery
   - Analyzes task context (layer, type, keywords)
   - Scores available skills (0-100 points)
   - Returns top 3 recommendations with confidence
   - Handles multi-skill scenarios and fallbacks

2. **`improving-framework-continuously`** (v1.1.0): Framework evolution
   - Captures errors in structured reports
   - YAML validation before every commit
   - Weekly triage and root cause analysis
   - Metrics tracking (error rate, recurrence, resolution time)

### Recommended New Skills

| Skill Name | Priority | Key Patterns |
|------------|----------|--------------|
| `creating-ml-models` | HIGH | 5.5 - ML Model Class Pattern |
| `creating-cli-scripts` | MEDIUM | 5.3 - CLI Script Pattern |
| `creating-react-hooks` | LOW | 6.1 - usePolling Hook Pattern |

### Skill Structure Template

```
skill-name/
├── SKILL.md           # Main instructions (< 500 lines, YAML frontmatter)
├── EXAMPLES.md        # Extended code examples (if needed)
├── REFERENCE.md       # Detailed reference docs (if needed)
└── scripts/           # Executable utilities (optional)
```

### YAML Frontmatter Requirements

```yaml
---
name: skill-name       # lowercase, hyphens, max 64 chars (must match folder)
description: ...       # max 1024 chars, third person, what + when to use
version: 1.0.0         # recommended for tracking
---
```

---

## Validation Checklist

- [x] At least 15 patterns identified and documented (18 total)
- [x] Each HIGH priority pattern has 2+ concrete code examples with file paths
- [x] Anti-patterns section documents 6 things to avoid
- [x] Shared utilities indexed (12 utilities)
- [x] Error handling patterns documented by layer
- [x] All file path references verified to exist
- [x] Patterns organized by priority (HIGH -> MEDIUM -> LOW)
- [x] Skills best practices from Anthropic docs consolidated
- [x] Recent git commits analyzed for workflow patterns
- [x] Active skills inventory updated (7 SKILL.md files)
- [x] Meta-skills documented (routing-to-skills, improving-framework-continuously)
- [x] YAML frontmatter requirements documented

---

## Changes in Version 2.1

- Updated skills inventory from v2.0 (14 skills) to accurate count (7 SKILL.md + 16 related docs)
- Added meta-skills documentation (routing-to-skills, improving-framework-continuously)
- Enhanced Skills Best Practices with progressive disclosure details
- Added anti-hallucination and examples recommendations
- Added YAML frontmatter requirements section
- Updated validation checklist

---

*Last updated: 2026-01-18 | Version 2.1 Enhanced*
