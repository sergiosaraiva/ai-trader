# Backend Implementation Continuation Prompt

Copy and paste the following into a new Claude Code session:

---

## Task: Implement Web Showcase Backend

I need you to implement the backend services for the AI-Trader web showcase. This is a production-ready trading system that needs to be deployed online to demonstrate live predictions.

### Project Location
`/home/sergio/ai-trader`

### Key Documentation
Read these files first:
1. `CLAUDE.md` - Project guide and conventions
2. `docs/06-web-showcase-implementation-plan.md` - Full implementation plan
3. `docs/01-current-state-of-the-art.md` - Current system state

### What to Implement

Create the backend services to enable live predictions via API:

**1. Model Service** (`src/api/services/model_service.py`)
- Singleton pattern for MTF Ensemble model loading
- Load models from `models/mtf_ensemble/`
- Warm-up on startup
- Thread-safe prediction method
- Prediction caching (1-minute TTL)

**2. Data Service** (`src/api/services/data_service.py`)
- Fetch EUR/USD data via yfinance (no API key needed)
- Fetch VIX data via yfinance for sentiment
- Cache OHLCV data (minimum 2 days of 5-min bars)
- Resample to 1H, 4H, Daily for model inference

**3. Trading Service** (`src/api/services/trading_service.py`)
- Paper trading simulation with $100K virtual balance
- Execute trades based on predictions (confidence >= 70%)
- Track open positions and P&L
- Record trade history

**4. Database Layer** (`src/api/database/`)
- SQLite database (`data/db/trading.db`)
- Tables: predictions, trades, performance
- SQLAlchemy models

**5. Scheduler** (`src/api/scheduler.py`)
- APScheduler setup
- Every 5 min: Fetch latest market data
- Every 1 hour: Generate prediction + execute paper trade

**6. API Endpoints** (update existing routes)
- `GET /api/v1/predictions/latest` - Current prediction
- `GET /api/v1/predictions/history` - Last N predictions
- `GET /api/v1/trading/status` - Current position, balance
- `GET /api/v1/trading/history` - Trade history
- `GET /api/v1/trading/performance` - Win rate, P&L stats
- `GET /api/v1/market/current` - Current EUR/USD price

### Technical Requirements

- Use yfinance for market data (free, no API key)
- Models are XGBoost, loaded via joblib
- Existing code: `src/models/multi_timeframe/mtf_ensemble.py`
- Existing API skeleton: `src/api/main.py`, `src/api/routes/`
- Confidence threshold: 70% (only trade when ensemble confidence >= 0.70)
- Triple barrier: TP=25 pips, SL=15 pips for 1H trades

### Data Flow

```
yfinance (EUR/USD, VIX)
    ↓
Data Service (cache + resample)
    ↓
Model Service (MTFEnsemble.predict)
    ↓
Trading Service (paper trade execution)
    ↓
Database (persist predictions + trades)
    ↓
API Endpoints (serve to frontend)
```

### Implementation Order

1. Database models (SQLAlchemy)
2. Data service (yfinance integration)
3. Model service (load and predict)
4. Trading service (paper trading logic)
5. Scheduler (periodic tasks)
6. Update API routes (connect everything)
7. Test the full flow

### Existing MTF Ensemble Usage

```python
from src.models.multi_timeframe.mtf_ensemble import MTFEnsemble, MTFEnsembleConfig

config = MTFEnsembleConfig(
    weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
    include_sentiment=True,
    sentiment_by_timeframe={"1H": False, "4H": False, "D": True},
)
ensemble = MTFEnsemble(config=config, model_dir="models/mtf_ensemble")
ensemble.load()

# Predict returns: {"direction": "long"/"short", "confidence": 0.72, ...}
prediction = ensemble.predict(df_5min)
```

### Success Criteria

- `GET /health` returns healthy status
- `GET /api/v1/predictions/latest` returns real prediction from model
- `GET /api/v1/market/current` returns live EUR/USD price
- Scheduler runs predictions every hour
- Paper trades are recorded in SQLite

Proceed autonomously. Implement all services and test them.
