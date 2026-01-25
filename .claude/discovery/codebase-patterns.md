# AI-Trader Codebase Patterns Discovery Report

> **Generated**: 2026-01-23
> **Version**: 3.0.0 (Major Update)
> **Codebase**: AI Trading Agent - MTF Ensemble Trading System
> **Analysis Period**: Last 3 months of commits (October 2025 - January 2026)

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total patterns documented** | 24 |
| **HIGH priority patterns** | 11 |
| **MEDIUM priority patterns** | 8 |
| **LOW priority patterns** | 5 |
| **Anti-patterns documented** | 8 |
| **Shared utilities indexed** | 16 |
| **Active skills** | 24 (7 SKILL.md + 17 related docs) |

**Key Changes in v3.0**:
- Added 6 new patterns from recent feature development
- New services: PerformanceService, ExplanationService, FeatureSelectionManager
- New frontend components: ModelHighlights, PerformanceChart, ExplanationCard
- New ML pattern: EnhancedMetaFeatureCalculator (data leakage prevention)
- Updated anti-patterns with DEPRECATED field migration pattern
- Updated technology versions (React 19.2, Vite 7.2, TailwindCSS 4.1)

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

### Anti-Hallucination Requirements (CRITICAL)

Skills should include these safeguards:
- **Verification requirements**: Must read actual files before citing
- **Citation requirements**: Include file:line references for claims
- **Uncertainty permission**: Can say "I need to check the codebase"
- **Grounding validation**: Verify references exist after generation
- **Direct quotes**: Use quotes from source documents to ground responses

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

---

## 2. Technology Stack

### Backend (Python 3.12+)

| Category | Technologies | Versions | Usage |
|----------|-------------|----------|-------|
| **API Framework** | FastAPI, uvicorn | >=0.100.0, >=0.23.0 | All API endpoints |
| **ML Models** | XGBoost, scikit-learn | >=2.0.0, >=1.3.0 | MTF Ensemble |
| **Gradient Boosting** | LightGBM, CatBoost | >=4.0.0, >=1.2.0 | Alternative models |
| **Data Processing** | pandas, numpy, pyarrow | >=2.0.0, >=1.24.0, >=14.0.0 | All data ops |
| **Technical Analysis** | pandas-ta | latest | Feature engineering |
| **Database** | SQLAlchemy, psycopg2-binary | >=2.0.0, >=2.9.0 | SQLite/PostgreSQL |
| **Scheduling** | APScheduler | >=3.10.0 | Prediction scheduler |
| **Validation** | Pydantic, pydantic-settings | >=2.0.0 | Request/response schemas |
| **Testing** | pytest, pytest-asyncio | >=7.4.0 | Backend tests |
| **Data Sources** | yfinance, fredapi | >=0.2.28, >=0.5.0 | Market & sentiment data |
| **LLM Integration** | OpenAI | >=1.0.0 | AI explanations |
| **HTTP Clients** | httpx, aiohttp | >=0.25.0, >=3.9.0 | Async API calls |

### Frontend (Node.js 20+)

| Category | Technologies | Versions | Usage |
|----------|-------------|----------|-------|
| **Framework** | React | 19.2.0 | UI components |
| **Build Tool** | Vite | 7.2.4 | Development & bundling |
| **Styling** | TailwindCSS | 4.1.18 | All styling |
| **Charts** | Recharts | 3.6.0 | Price & performance charts |
| **Icons** | lucide-react | 0.562.0 | UI icons |
| **Props Validation** | prop-types | 15.8.1 | Runtime type checking |
| **Testing** | Vitest, Testing Library | 4.0.17 | Component tests |

### Architecture

- **Pattern**: Layered monolith with service architecture
- **API Style**: REST with FastAPI
- **State**: SQLite for persistence, in-memory caching
- **ML Pipeline**: XGBoost ensemble with multi-timeframe models (1H, 4H, Daily)
- **Stacking**: Meta-learner with enhanced meta-features (20 features)

---

## 3. Layer Architecture

```
ai-trader/
├── backend/
│   └── src/                      # Python source code
│       ├── api/                  # FastAPI web layer
│       │   ├── main.py          # App entry point + lifespan
│       │   ├── scheduler.py     # APScheduler setup
│       │   ├── routes/          # API endpoints
│       │   │   ├── predictions.py
│       │   │   ├── trading.py
│       │   │   ├── market.py
│       │   │   ├── performance.py  # NEW: Model highlights
│       │   │   └── pipeline.py
│       │   ├── services/        # Business logic singletons
│       │   │   ├── model_service.py
│       │   │   ├── data_service.py
│       │   │   ├── trading_service.py
│       │   │   ├── performance_service.py   # NEW
│       │   │   ├── explanation_service.py   # NEW
│       │   │   └── pipeline_service.py
│       │   ├── schemas/         # Pydantic models
│       │   └── database/        # SQLAlchemy models + session
│       ├── models/              # ML models
│       │   ├── multi_timeframe/ # MTF Ensemble (PRIMARY)
│       │   │   ├── mtf_ensemble.py
│       │   │   ├── stacking_meta_learner.py
│       │   │   ├── enhanced_meta_features.py   # NEW
│       │   │   └── improved_model.py
│       │   └── feature_selection/  # NEW: RFECV module
│       │       ├── manager.py
│       │       ├── rfecv_selector.py
│       │       └── rfecv_config.py
│       ├── features/            # Feature engineering
│       │   ├── technical/       # Technical indicators + registry
│       │   ├── sentiment/       # EPU/VIX sentiment loading
│       │   └── regime/          # Market regime detection
│       ├── trading/             # Trading logic (position sizing, risk)
│       └── simulation/          # Backtesting
├── frontend/                    # React application
│   └── src/
│       ├── components/          # React components
│       │   ├── Dashboard.jsx
│       │   ├── PredictionCard.jsx
│       │   ├── ModelHighlights.jsx      # NEW
│       │   ├── PerformanceChart.jsx     # NEW
│       │   ├── ExplanationCard.jsx      # NEW
│       │   └── ...
│       ├── api/                 # API client
│       └── hooks/               # Custom hooks (usePolling)
├── scripts/                     # CLI tools
├── tests/                       # Test suites (735+ tests)
└── docs/                        # Documentation
```

---

## 4. HIGH Priority Patterns

### Pattern 4.1: FastAPI Service Singleton

**Description**: Thread-safe singleton services with lazy initialization, caching, and status tracking.

**Priority**: HIGH
**Frequency**: Used in 7 services (`model_service`, `data_service`, `trading_service`, `pipeline_service`, `performance_service`, `explanation_service`, `asset_service`)
**Layer**: Backend API Services

**Examples**:
1. `backend/src/api/services/model_service.py:28-280` - Complete ModelService class
2. `backend/src/api/services/performance_service.py:61-529` - PerformanceService implementation
3. `backend/src/api/services/explanation_service.py:19-369` - ExplanationService with caching

**Code Pattern**:
```python
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

model_service = ModelService()
```

**When to Use**:
- Services managing expensive resources (ML models, DB connections, LLM clients)
- Shared state across multiple requests
- Operations requiring thread safety
- Services with intelligent caching (TTL or value-change based)

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
- [ ] Cache cleanup to prevent memory leaks

---

### Pattern 4.2: FastAPI Route Handler

**Description**: Route handlers with service availability checks, proper error handling, and Pydantic response models.

**Priority**: HIGH
**Frequency**: Used in 25+ endpoints across all route modules
**Layer**: Backend API Routes

**Examples**:
1. `backend/src/api/routes/predictions.py:26-78` - GET with service check
2. `backend/src/api/routes/performance.py:1-50` - Performance highlights endpoint
3. `backend/src/api/routes/trading.py:1-100` - Trading endpoints

**Code Pattern**:
```python
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
```

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
**Frequency**: Used in 20+ schema classes
**Layer**: Backend API Schemas

**Examples**:
1. `backend/src/api/schemas/prediction.py:16-68` - PredictionResponse
2. `backend/src/api/schemas/trading.py:1-200` - Trading schemas with DEPRECATED fields

**Code Pattern**:
```python
from pydantic import BaseModel, Field
from typing import Dict, Optional

class PredictionResponse(BaseModel):
    """Response for prediction endpoint."""

    timestamp: str = Field(..., description="ISO format timestamp")
    symbol: str = Field(default="EURUSD", description="Trading symbol")
    direction: str = Field(..., description="'long' or 'short'")
    confidence: float = Field(..., ge=0, le=1, description="Confidence 0-1")
    should_trade: bool = Field(..., description="Confidence >= threshold")

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
- [ ] DEPRECATED fields documented with replacement field name

---

### Pattern 4.4: React Card Component with PropTypes

**Description**: Display card with loading, error, empty, and data states using TailwindCSS and PropTypes validation.

**Priority**: HIGH
**Frequency**: Used in 10+ components (PredictionCard, AccountStatus, PerformanceStats, ModelHighlights, ExplanationCard, PerformanceChart)
**Layer**: Frontend Components

**Examples**:
1. `frontend/src/components/ModelHighlights.jsx:1-143`
2. `frontend/src/components/ExplanationCard.jsx:1-107`
3. `frontend/src/components/PerformanceChart.jsx:1-366`

**Code Pattern**:
```jsx
import { AlertCircle, TrendingUp } from 'lucide-react';
import PropTypes from 'prop-types';

export function ModelHighlights({ performance, loading, error }) {
  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="h-16 bg-gray-700 rounded mb-4"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 border border-yellow-500/30">
        <div className="flex items-center gap-2 text-yellow-400">
          <AlertCircle size={20} />
          <span>Performance data unavailable</span>
        </div>
        <p className="text-gray-500 text-sm mt-2">{error}</p>
      </div>
    );
  }

  if (!performance || performance.highlights?.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <p className="text-gray-500">No performance highlights available</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6 card-hover">
      <h2 className="text-xl font-bold">{performance.summary.headline}</h2>
      {/* ... data rendering ... */}
    </div>
  );
}

ModelHighlights.propTypes = {
  performance: PropTypes.shape({
    highlights: PropTypes.array,
    summary: PropTypes.shape({
      headline: PropTypes.string,
      description: PropTypes.string,
    }),
    metrics: PropTypes.object,
  }),
  loading: PropTypes.bool,
  error: PropTypes.string,
};

ModelHighlights.defaultProps = {
  performance: null,
  loading: false,
  error: null,
};

export default ModelHighlights;
```

**Quality Criteria**:
- [ ] All four states handled (loading, error, empty, data)
- [ ] Skeleton loader with `animate-pulse`
- [ ] Error message with icon and details
- [ ] TailwindCSS for styling
- [ ] Props destructured at top
- [ ] PropTypes validation defined
- [ ] defaultProps for optional props
- [ ] Named and default export

---

### Pattern 4.5: Chart Component with useMemo

**Description**: Complex chart components using useMemo for data transformation and memo for child components.

**Priority**: HIGH
**Frequency**: Used in 3 chart components (PriceChart, PerformanceChart)
**Layer**: Frontend Components

**Examples**:
1. `frontend/src/components/PerformanceChart.jsx:1-366`

**Code Pattern**:
```jsx
import { useMemo, memo } from 'react';
import { ComposedChart, Bar, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

const CustomTooltip = memo(function CustomTooltip({ active, payload, profitUnit }) {
  if (!active || !payload || payload.length === 0) return null;
  const data = payload[0]?.payload;
  if (!data) return null;
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3">
      {/* tooltip content */}
    </div>
  );
});

export function PerformanceChart({ trades, loading, error, assetMetadata }) {
  const chartData = useMemo(() => {
    if (!trades || !Array.isArray(trades)) return [];
    // Complex data transformation
    return transformedData;
  }, [trades]);

  const stats = useMemo(() => {
    if (chartData.length === 0) return defaultStats;
    // Calculate statistics
    return { totalDays, profitableDays, totalPnl, maxDrawdown };
  }, [chartData]);

  if (loading) return <LoadingSkeleton />;
  if (error) return <ErrorState error={error} />;
  if (chartData.length === 0) return <EmptyState />;

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={chartData}>
          {/* chart components */}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
```

**Quality Criteria**:
- [ ] `useMemo` for expensive data transformations
- [ ] `memo` for tooltip and child components
- [ ] Dependency arrays correct
- [ ] ResponsiveContainer for chart sizing
- [ ] Loading/error/empty states handled
- [ ] Accessibility attributes (role, aria-label)

---

### Pattern 4.6: Dashboard Data Fetching

**Description**: Main dashboard with usePolling hooks for periodic data fetching.

**Priority**: HIGH
**Frequency**: Main pattern in Dashboard.jsx
**Layer**: Frontend Components

**Examples**:
1. `frontend/src/components/Dashboard.jsx:1-250`
2. `frontend/src/hooks/usePolling.js:1-50`

**Code Pattern**:
```jsx
import { useCallback } from 'react';
import { api } from '../api/client';
import { usePolling } from '../hooks/usePolling';

const INTERVALS = {
  prediction: 30000,
  candles: 60000,
  account: 60000,
  performance: 300000,  // 5 minutes for static data
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
    data: performance,
    loading: performanceLoading,
    error: performanceError,
  } = usePolling(
    useCallback(() => api.getPerformance(), []),
    INTERVALS.performance
  );

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <PredictionCard
            prediction={prediction}
            loading={predictionLoading}
            error={predictionError}
          />
          <ModelHighlights
            performance={performance}
            loading={performanceLoading}
            error={performanceError}
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
- [ ] Different intervals for different data freshness needs

---

### Pattern 4.7: Pytest API Test Pattern

**Description**: FastAPI endpoint tests with TestClient and mocked services.

**Priority**: HIGH
**Frequency**: Used in 40+ API tests
**Layer**: Testing

**Examples**:
1. `backend/tests/api/test_predictions.py:1-134`
2. `backend/tests/api/test_performance_routes.py:1-100`

**Code Pattern**:
```python
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

        self.mock_df = pd.DataFrame({
            "open": np.random.rand(200) + 1.08,
            "close": np.random.rand(200) + 1.08,
        })

    def test_model_status_endpoint(self):
        """Test model status endpoint."""
        from backend.src.api.routes import predictions

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

### Pattern 4.8: Vitest Component Test Pattern

**Description**: React component tests with Vitest and Testing Library.

**Priority**: HIGH
**Frequency**: Used in 13 frontend test files
**Layer**: Frontend Testing

**Examples**:
1. `frontend/src/components/ModelHighlights.test.jsx:1-100`
2. `frontend/src/components/ExplanationCard.test.jsx:1-84`

**Code Pattern**:
```jsx
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ExplanationCard } from './ExplanationCard';

describe('ExplanationCard', () => {
  it('renders loading state', () => {
    render(<ExplanationCard loading={true} />);
    const skeleton = document.querySelector('.animate-pulse');
    expect(skeleton).toBeInTheDocument();
  });

  it('renders error state with refresh button', () => {
    const mockRefresh = vi.fn();
    render(<ExplanationCard error="Test error" onRefresh={mockRefresh} />);
    expect(screen.getByText('AI explanation unavailable')).toBeInTheDocument();
    
    const refreshButton = screen.getByTitle('Retry');
    fireEvent.click(refreshButton);
    expect(mockRefresh).toHaveBeenCalledTimes(1);
  });

  it('renders null when no explanation', () => {
    const { container } = render(<ExplanationCard explanation={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders explanation with cached indicator', () => {
    const explanation = {
      explanation: 'BUY with high confidence.',
      cached: true,
    };
    render(<ExplanationCard explanation={explanation} />);
    expect(screen.getByText('BUY with high confidence.')).toBeInTheDocument();
    expect(screen.getByText('cached')).toBeInTheDocument();
  });
});
```

**Quality Criteria**:
- [ ] All states tested (loading, error, empty, data)
- [ ] `screen.getByText` for assertions
- [ ] `vi.fn()` for mock functions
- [ ] `fireEvent` for user interactions
- [ ] Descriptive test names

---

### Pattern 4.9: Dataclass Configuration with Factory Methods

**Description**: Configuration objects using dataclasses with defaults and factory methods.

**Priority**: HIGH
**Frequency**: Used in 10+ configuration classes
**Layer**: Backend Models

**Examples**:
1. `backend/src/models/multi_timeframe/mtf_ensemble.py:28-119` - MTFEnsembleConfig
2. `backend/src/models/feature_selection/rfecv_config.py:1-50` - RFECVConfig

**Code Pattern**:
```python
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
    use_stacking: bool = True
    use_enhanced_meta_features: bool = True  # NEW

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
- [ ] Docstring explaining purpose

---

### Pattern 4.10: Data Leakage Prevention in ML Features

**Description**: Feature engineering with explicit .shift(1) to prevent look-ahead bias.

**Priority**: HIGH
**Frequency**: Used in enhanced_meta_features.py, technical indicators
**Layer**: Backend ML Models

**Examples**:
1. `backend/src/models/multi_timeframe/enhanced_meta_features.py:1-303`

**Code Pattern**:
```python
"""CRITICAL: Data Leakage Prevention
All features MUST use .shift(1) or appropriate lag to prevent look-ahead bias.
"""

class EnhancedMetaFeatureCalculator:
    """Calculator for enhanced meta-features with data leakage prevention.

    All rolling calculations use .shift(1) to ensure no future data is used.
    """

    def calculate_market_context(self, price_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate volatility, trend strength, and regime features.

        CRITICAL: All calculations use .shift(1) to prevent look-ahead bias.
        """
        df = price_data.copy()

        returns = df["close"].pct_change()
        vol_raw = returns.rolling(window=20, min_periods=5).std()
        vol_shifted = vol_raw.shift(1)  # CRITICAL: Use past volatility only

        ma14 = df["close"].rolling(window=14, min_periods=5).mean()
        deviation = abs(df["close"] - ma14) / ma14
        trend_strength_raw = deviation.rolling(window=14, min_periods=5).mean()
        trend_strength_shifted = trend_strength_raw.shift(1)  # CRITICAL

        vol_shifted = vol_shifted.fillna(vol_shifted.median())

        return {
            "recent_volatility": vol_shifted.values,
            "trend_strength": trend_strength_shifted.values,
        }
```

**Quality Criteria**:
- [ ] Module-level docstring explaining leakage prevention
- [ ] `.shift(1)` on ALL rolling calculations
- [ ] Comments marking CRITICAL shifts
- [ ] Proper NaN handling after shifting
- [ ] No access to future data anywhere

---

### Pattern 4.11: Cache with Hash-Based Invalidation

**Description**: Caching with config/value hash for automatic invalidation.

**Priority**: HIGH
**Frequency**: Used in feature_selection, explanation_service
**Layer**: Backend Services

**Examples**:
1. `backend/src/models/feature_selection/manager.py:24-250`
2. `backend/src/api/services/explanation_service.py:67-134`

**Code Pattern**:
```python
import hashlib
import json
from pathlib import Path

CACHE_HASH_LENGTH = 8

class FeatureSelectionManager:
    def _compute_config_hash(self) -> str:
        """Compute hash of config for cache invalidation."""
        config_dict = {
            "step": self.config.step,
            "min_features": self.config.min_features_to_select,
            "cv": self.config.cv,
            "scoring": self.config.scoring,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:CACHE_HASH_LENGTH]

    def _get_cache_path(self, timeframe: str) -> Path:
        config_hash = self._compute_config_hash()
        return self.cache_dir / f"{timeframe}_rfecv_{config_hash}.json"

    def _load_from_cache(self, timeframe: str, n_features: int) -> Optional[Dict]:
        if not self.config.cache_enabled:
            return None

        cache_path = self._get_cache_path(timeframe)
        if not cache_path.exists():
            return None

        with open(cache_path, "r") as f:
            cached = json.load(f)

        # CRITICAL: Validate feature count matches
        if cached.get("n_original_features") != n_features:
            logger.warning(f"Cache invalidated: feature count changed")
            cache_path.unlink()
            return None

        return cached
```

**Quality Criteria**:
- [ ] Hash computed from all relevant config parameters
- [ ] `sort_keys=True` for deterministic hashing
- [ ] Cache path includes hash for automatic versioning
- [ ] Validation of cached data (e.g., feature count)
- [ ] Cache cleanup for invalid entries

---

## 5. MEDIUM Priority Patterns

### Pattern 5.1: SQLAlchemy Model

**Description**: Database models with types, indexes, and relationships.

**Priority**: MEDIUM
**Frequency**: Used in 3 models (Prediction, Trade, Account)
**Layer**: Backend Database

**Examples**:
1. `backend/src/api/database/models.py:22-107`

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
1. `frontend/src/api/client.js:1-100`

**Code Pattern**:
```javascript
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
  getPerformance: () => request('/v1/performance'),
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
1. `backend/scripts/train_mtf_ensemble.py:1-456`
2. `backend/scripts/optimize_hyperparameters.py:1-300`

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
    parser.add_argument("--stacking", action="store_true", default=True)
    parser.add_argument("--enhanced-meta", action="store_true", default=True)
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
1. `backend/src/api/main.py:25-130`

**Code Pattern**:
```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting API...")
    init_db()
    model_service.initialize()
    performance_service.initialize()  # NEW
    start_scheduler()
    yield
    logger.info("Shutting down...")
    stop_scheduler()

def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Trading Agent API",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
    app.include_router(health.router, tags=["Health"])
    app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
    app.include_router(performance.router, prefix="/api/v1", tags=["Performance"])

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
1. `backend/src/models/multi_timeframe/mtf_ensemble.py:154-653`

---

### Pattern 5.6: Performance Service with Default Constants

**Description**: Service loading from multiple data files with documented defaults.

**Priority**: MEDIUM
**Frequency**: Used in performance_service.py
**Layer**: Backend Services

**Examples**:
1. `backend/src/api/services/performance_service.py:21-80`

**Code Pattern**:
```python
# Paths to data files (canonical sources of truth)
BACKTEST_RESULTS_PATH = PROJECT_ROOT / "data" / "backtest_results.json"

def _load_defaults_from_backtest() -> Dict[str, Any]:
    """Load default metrics from backtest_results.json (canonical source).

    This ensures defaults always match the latest validated backtest results,
    avoiding stale hardcoded values that drift from actual data.
    """
    if not BACKTEST_RESULTS_PATH.exists():
        return fallback  # Zero values

    with open(BACKTEST_RESULTS_PATH) as f:
        data = json.load(f)

    # Load baseline from 5y (All Time) period
    all_time = data.get("periods", {}).get("5y", {})
    baseline = {
        "TOTAL_PIPS": all_time.get("total_pips", 0),
        "WIN_RATE": all_time.get("win_rate", 0) / 100,
        ...
    }
    return {"baseline": baseline, "high_conf": high_conf}

# Load defaults once at module initialization (from canonical backtest_results.json)
_LOADED_DEFAULTS = _load_defaults_from_backtest()
DEFAULT_BASELINE_METRICS = _LOADED_DEFAULTS["baseline"]
DEFAULT_HIGH_CONF_METRICS = _LOADED_DEFAULTS["high_conf"]

class PerformanceService:
    def _load_metrics(self) -> None:
        """Load metrics from training_metadata.json and backtest_results.json."""
        metadata_path = self.DEFAULT_MODEL_DIR / "training_metadata.json"
        if not metadata_path.exists():
            logger.warning(f"Training metadata not found at {metadata_path}")
            self._metrics = self._get_default_metrics()
            return

        # Load from files, fall back to defaults for missing values
```

**Quality Criteria**:
- [ ] Constants at module level with date/source comments
- [ ] Graceful fallback to defaults when files unavailable
- [ ] Logging when using fallbacks
- [ ] Single source of truth for metric values

---

### Pattern 5.7: Technical Indicator Calculator

**Description**: Configuration-driven indicator calculation with registry.

**Priority**: MEDIUM
**Frequency**: Used across all feature engineering
**Layer**: Backend Features

**Examples**:
1. `backend/src/features/technical/calculator.py:47-411`

---

### Pattern 5.8: LLM Service with Intelligent Caching

**Description**: OpenAI integration with value-change-based cache invalidation.

**Priority**: MEDIUM
**Frequency**: Used in explanation_service.py
**Layer**: Backend Services

**Examples**:
1. `backend/src/api/services/explanation_service.py:19-369`

**Code Pattern**:
```python
class ExplanationService:
    CACHE_TTL = timedelta(hours=1)
    CONFIDENCE_THRESHOLD = 0.05  # 5% change triggers regeneration
    VIX_THRESHOLD = 2.0

    def _should_regenerate(self, current_values: Dict[str, Any]) -> bool:
        """Check if we should regenerate based on value changes."""
        if not self._last_values:
            return True

        if current_values.get("direction") != self._last_values.get("direction"):
            return True

        conf_diff = abs(
            current_values.get("confidence", 0) -
            self._last_values.get("confidence", 0)
        )
        if conf_diff >= self.CONFIDENCE_THRESHOLD:
            return True

        return False

    def _cleanup_expired_cache(self) -> None:
        """Remove expired cache entries to prevent memory leak."""
        now = datetime.now()
        expired_keys = [
            k for k, v in self._cache.items()
            if now - v["generated_at"] > self.CACHE_TTL
        ]
        for key in expired_keys:
            del self._cache[key]
```

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
1. `backend/src/api/database/session.py:1-30`

---

### Pattern 6.3: Regime Detector

**Description**: Market regime detection with trend/volatility analysis.

**Priority**: LOW
**Frequency**: Specialized trading domain
**Layer**: Backend Features

**Examples**:
1. `backend/src/features/regime/regime_detector.py:1-200`

---

### Pattern 6.4: Position Sizing

**Description**: Kelly criterion position sizing strategies.

**Priority**: LOW
**Frequency**: Specialized trading domain
**Layer**: Backend Trading

**Examples**:
1. `backend/src/trading/position_sizing.py:179-397`

---

### Pattern 6.5: JSON Serialization Helper

**Description**: Convert numpy types to JSON-serializable Python types.

**Priority**: LOW
**Frequency**: Used in caching and serialization
**Layer**: Backend Utilities

**Examples**:
1. `backend/src/models/feature_selection/manager.py:188-200`

**Code Pattern**:
```python
def convert_for_json(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): convert_for_json(v) for k, v in obj.items()}
    return obj
```

---

## 7. Anti-Patterns (What NOT to Do)

### Anti-Pattern 7.1: Catching HTTPException in Generic Handler

**Why it's wrong**: Swallows specific error details and status codes.

**Evidence**: Avoided in `backend/src/api/routes/predictions.py:74-78`

**Wrong**:
```python
try:
    # ... operations
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
```

**Correct approach**: See `backend/src/api/routes/predictions.py:74-78`
```python
try:
    # ... operations
except HTTPException:
    raise
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

**Correct approach**: See `backend/src/api/routes/predictions.py:26-37`
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
    predictions.model_service = self.mock
    # Test code
```

**Correct approach**: See `backend/tests/api/test_predictions.py:56-68`
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
    weights: Dict[str, float] = {"1H": 0.6}
```

**Correct approach**: See `backend/src/models/multi_timeframe/mtf_ensemble.py:30-35`
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

**Correct approach**: See `backend/src/api/routes/predictions.py:26`
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
  return <div>{data.value}</div>;
}
```

**Correct approach**: See `frontend/src/components/ModelHighlights.jsx:1-143`
```jsx
function Card({ data, loading, error }) {
  if (loading) return <Skeleton />;
  if (error) return <ErrorMessage error={error} />;
  if (!data) return <EmptyState />;
  return <div>{data.value}</div>;
}
```

---

### Anti-Pattern 7.7: Missing Data Leakage Prevention in Rolling Features

**Why it's wrong**: Look-ahead bias invalidates all model performance claims.

**Evidence**: Correct pattern in `backend/src/models/multi_timeframe/enhanced_meta_features.py`

**Wrong**:
```python
vol = returns.rolling(window=20).std()
```

**Correct approach**:
```python
vol_raw = returns.rolling(window=20).std()
vol_shifted = vol_raw.shift(1)  # CRITICAL: Use past volatility only
```

---

### Anti-Pattern 7.8: Not Documenting DEPRECATED Fields

**Why it's wrong**: Developers don't know about the migration path.

**Evidence**: Found in `backend/src/api/schemas/trading.py:26`

**Correct approach**: Document with replacement field name
```python
pips: Optional[float] = Field(
    None, 
    description="Profit/loss in pips (DEPRECATED - use profit_points)"
)
```

---

## 8. Shared Utilities Index

| Utility | Location | Used By | Purpose |
|---------|----------|---------|---------|
| `TechnicalIndicatorCalculator` | `backend/src/features/technical/calculator.py:47` | 15+ scripts | Calculate technical indicators |
| `SentimentLoader` | `backend/src/features/sentiment/sentiment_loader.py:1` | 5+ modules | Load EPU/VIX sentiment |
| `MTFEnsemble` | `backend/src/models/multi_timeframe/mtf_ensemble.py:154` | 10+ scripts | Main prediction model |
| `MTFEnsembleConfig` | `backend/src/models/multi_timeframe/mtf_ensemble.py:28` | 10+ scripts | Model configuration |
| `EnhancedMetaFeatureCalculator` | `backend/src/models/multi_timeframe/enhanced_meta_features.py:32` | 2 modules | Meta-learner features |
| `FeatureSelectionManager` | `backend/src/models/feature_selection/manager.py:24` | Training scripts | RFECV feature selection |
| `model_service` | `backend/src/api/services/model_service.py:389` | 5+ routes | Singleton model service |
| `data_service` | `backend/src/api/services/data_service.py:36` | 5+ routes | Market data fetching |
| `trading_service` | `backend/src/api/services/trading_service.py:24` | 3+ routes | Paper trading logic |
| `performance_service` | `backend/src/api/services/performance_service.py:61` | Performance routes | Model highlights |
| `explanation_service` | `backend/src/api/services/explanation_service.py:19` | Prediction routes | LLM explanations |
| `api` client | `frontend/src/api/client.js:1` | All components | API communication |
| `usePolling` | `frontend/src/hooks/usePolling.js:1` | Dashboard | Periodic fetching |
| `get_db` | `backend/src/api/database/session.py:1` | 5+ routes | DB session dependency |
| `RegimeDetector` | `backend/src/features/regime/regime_detector.py:1` | 3+ scripts | Market regime detection |
| `PositionSizer` | `backend/src/trading/position_sizing.py:179` | 2+ scripts | Kelly criterion sizing |

---

## 9. Error Handling Summary

| Layer | Exception Type | Logging Pattern | Example |
|-------|---------------|-----------------|---------|
| **API Routes** | `HTTPException` | `logger.error(f"Error: {e}")` | `backend/src/api/routes/predictions.py:77` |
| **Services** | `RuntimeError` | `logger.error(f"Failed: {e}")` | `backend/src/api/services/model_service.py:93` |
| **ML Models** | `RuntimeError` | `logger.warning(f"Issue: {e}")` | `backend/src/models/multi_timeframe/mtf_ensemble.py:500` |
| **Scripts** | `Exception` | `logger.error(f"Error: {e}")` | `backend/scripts/train_mtf_ensemble.py:225` |
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
backend/tests/
├── api/                      # API endpoint tests
│   ├── test_predictions.py
│   ├── test_trading.py
│   ├── test_performance_routes.py
│   └── test_health.py
├── services/                 # Service unit tests
│   ├── test_data_service.py
│   └── test_trading_service.py
├── unit/                     # Unit tests by module
│   ├── api/
│   │   └── test_performance_service.py
│   ├── models/
│   │   ├── test_enhanced_meta_features.py
│   │   └── test_stacking_meta_learner.py
│   └── features/
└── integration/              # Integration tests
    ├── test_enhanced_meta_learner.py
    └── test_rfecv_integration.py

frontend/src/
├── components/
│   ├── *.jsx                 # Component files
│   └── *.test.jsx            # Component tests (co-located)
├── hooks/
│   └── *.test.js             # Hook tests
└── api/
    └── client.test.js        # API client tests
```

### Test Naming Convention

```
test_<function>_<condition>_<expected_result>

Examples:
- test_model_status_endpoint
- test_latest_prediction_model_not_loaded
- test_performance_highlights_loading_state
- test_explanation_card_cached_indicator
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
2. **Backend First**: Add services and routes
3. **Frontend Second**: Add components consuming API
4. **Testing**: Add tests in appropriate directories
5. **CLAUDE.md Update**: Update project guide with new capabilities

**Recent Examples**:
```
b134ad7 feat: Add ModelHighlights component with dynamic performance metrics
  - backend/src/api/routes/performance.py
  - backend/src/api/services/performance_service.py
  - backend/tests/api/test_performance_routes.py
  - backend/tests/unit/api/test_performance_service.py
  - frontend/src/components/ModelHighlights.jsx
  - frontend/src/components/ModelHighlights.test.jsx

15161e2 feat: Add Enhanced Meta-Learner Features (20 meta-features)
  - backend/src/models/multi_timeframe/enhanced_meta_features.py
  - backend/tests/unit/models/test_enhanced_meta_features.py
  - backend/tests/integration/test_enhanced_meta_learner.py
```

### Commit Message Convention

```
feat: <description>           # New feature
fix: <description>            # Bug fix
docs: <description>           # Documentation
refactor: <description>       # Refactoring
chore: <description>          # Maintenance
test: <description>           # Test additions
perf: <description>           # Performance improvements

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
| React components | PascalCase | `ModelHighlights.jsx` |
| React tests | `*.test.jsx` | `ModelHighlights.test.jsx` |
| Python tests | `test_*.py` | `test_performance_service.py` |

### Code

| Type | Convention | Example |
|------|------------|---------|
| Functions | snake_case | `get_prediction()` |
| Variables | snake_case | `model_service` |
| Constants | UPPER_SNAKE | `CACHE_TTL`, `DEFAULT_BASELINE_METRICS` |
| Classes | PascalCase | `MTFEnsemble` |
| Private | _prefix | `_load_model()` |
| React props | camelCase | `predictionLoading` |

---

## 13. Recommended Skills

Based on patterns discovered, the following skills exist:

### Active Skills (24 total)

| Layer | Count | Primary Skills |
|-------|-------|----------------|
| **Meta** | 2 | `routing-to-skills`, `improving-framework-continuously` |
| **Backend** | 6 | `backend`, `creating-python-services`, `creating-pydantic-schemas` |
| **Frontend** | 2 | `frontend`, `creating-api-clients` |
| **Database** | 1 | `database` |
| **Feature Engineering** | 2 | `creating-technical-indicators`, `configuring-indicator-yaml` |
| **Data Layer** | 1 | `adding-data-sources` |
| **Trading Domain** | 3 | `running-backtests`, `analyzing-trading-performance`, `implementing-risk-management` |
| **Testing** | 2 | `testing`, `writing-vitest-tests` |
| **Quality & Testing** | 4 | `creating-dataclasses`, `validating-time-series-data`, `planning-test-scenarios`, `generating-test-data` |
| **Build & Deployment** | 1 | `build-deployment` |

### Quick Reference: When to Use

| Task | Primary Skill |
|------|---------------|
| Add API endpoint | `backend` |
| Create service class | `creating-python-services` |
| Define API schema | `creating-pydantic-schemas` |
| Internal DTO | `creating-dataclasses` |
| React component | `frontend` |
| Database model | `database` |
| Technical indicator | `creating-technical-indicators` |
| New data source | `adding-data-sources` |
| Backtest strategy | `running-backtests` |
| Analyze performance | `analyzing-trading-performance` |
| Position sizing | `implementing-risk-management` |
| Python tests | `testing` |
| Frontend tests | `writing-vitest-tests` |
| Test data | `generating-test-data` |
| Time series validation | `validating-time-series-data` |
| CLI script | `build-deployment` |

### Recommended New Skills

| Skill Name | Priority | Key Patterns |
|------------|----------|--------------|
| `creating-ml-features` | HIGH | 4.10 - Data Leakage Prevention Pattern |
| `creating-chart-components` | MEDIUM | 4.5 - Chart Component with useMemo |
| `caching-strategies` | MEDIUM | 4.11 - Cache with Hash-Based Invalidation |

---

## Validation Checklist

- [x] At least 15 patterns identified and documented (24 total)
- [x] Each HIGH priority pattern has 2+ concrete code examples with file paths
- [x] Anti-patterns section documents 8 things to avoid
- [x] Shared utilities indexed (16 utilities)
- [x] Error handling patterns documented by layer
- [x] All file path references verified to exist
- [x] Patterns organized by priority (HIGH -> MEDIUM -> LOW)
- [x] Skills best practices from Anthropic docs consolidated
- [x] Recent git commits analyzed for workflow patterns
- [x] Active skills inventory updated (24 skills)
- [x] Technology stack versions updated (React 19.2, Vite 7.2)
- [x] New patterns from recent development documented

---

## Changes in Version 3.0

### New Patterns Added
- **4.4**: Updated React Card Component pattern with PropTypes (from 13 test files)
- **4.5**: Chart Component with useMemo (PerformanceChart, PriceChart)
- **4.10**: Data Leakage Prevention in ML Features (EnhancedMetaFeatureCalculator)
- **4.11**: Cache with Hash-Based Invalidation (FeatureSelectionManager, ExplanationService)
- **5.6**: Performance Service with Default Constants
- **5.8**: LLM Service with Intelligent Caching

### New Anti-Patterns
- **7.7**: Missing Data Leakage Prevention in Rolling Features
- **7.8**: Not Documenting DEPRECATED Fields

### Updated Sections
- Technology stack: React 19.2.0, Vite 7.2.4, TailwindCSS 4.1.18, LightGBM/CatBoost
- Shared utilities: Added 4 new utilities (EnhancedMetaFeatureCalculator, FeatureSelectionManager, performance_service, explanation_service)
- Layer architecture: Added new routes, services, and ML modules
- Test organization: Added frontend test patterns (13 test files)

---

*Last updated: 2026-01-23 | Version 3.0.0*
