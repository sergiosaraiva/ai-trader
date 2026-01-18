# CLAUDE.md - AI Assets Trader Project Guide

## Interaction Mode

**ALWAYS PROCEED WITHOUT ASKING FOR CONFIRMATION.**

- Do not ask for permission before taking actions
- Do not wait for user confirmation on implementation choices
- Make autonomous decisions based on best practices and project context
- Execute tasks fully without pausing for approval
- If multiple valid approaches exist, choose the most appropriate one and proceed
- Only ask questions if there is genuine ambiguity that cannot be resolved from context

The user trusts Claude to make good decisions. Act decisively and complete tasks end-to-end.

## Project Overview

AI Assets Trader is a **production-ready Multi-Timeframe (MTF) Ensemble trading system** for forex. The system uses XGBoost models across three timeframes (1H, 4H, Daily) combined with sentiment analysis to generate trading predictions. It includes a **web showcase** with a React frontend and FastAPI backend for live demonstration.

**Current Status: Production-Ready (WFO Validated, Threshold Optimized, Web Showcase)**
- +8,693 pips profit with optimized 70% confidence threshold
- +18,136 pips across 7 WFO windows (100% profitable)
- 62.1% win rate, 2.69 profit factor (at 70% threshold)
- VIX/EPU sentiment integration (Daily model only)
- Walk-Forward Optimization: PASSED (100% consistency)
- Confidence Threshold: OPTIMIZED (70% recommended)
- Web Showcase: React dashboard + FastAPI backend
- Docker deployment ready for Railway cloud

## Current Performance (Optimized)

| Metric | Baseline (55%) | Optimized (70%) | Improvement |
|--------|----------------|-----------------|-------------|
| **Total Profit** | +7,987 pips | **+8,693 pips** | +8.8% |
| **Win Rate** | 57.8% | **62.1%** | +4.3% |
| **Profit Factor** | 2.22 | **2.69** | +21% |
| **Total Trades** | 1,103 | 966 | -12.4% |
| **Avg Pips/Trade** | +7.2 | **+9.0** | +25% |
| **Sharpe Ratio** | 6.09 | **7.67** | +26% |

### Model Accuracy

| Model | Weight | Val Accuracy | High-Conf (≥60%) |
|-------|--------|--------------|------------------|
| 1H | 60% | 67.07% | 72.14% |
| 4H | 30% | 65.43% | 71.12% |
| Daily | 10% | 61.54% | 64.21% |

## Project Structure

```
ai-trader/
├── .claude/                       # Agent-Skill Framework (v1.2.0)
│   ├── agents/                    # 6 specialized agents
│   ├── skills/                    # 24 active skills by layer
│   ├── improvement/               # Error reporting & continuous improvement
│   ├── scripts/                   # Validation scripts
│   ├── hooks/                     # Git pre-commit hooks
│   ├── metrics/                   # Weekly health reports
│   └── optimization/              # Monthly consolidation reports
├── data/
│   ├── forex/                     # EUR/USD 5-minute data (448K bars, 2020-2025)
│   │   └── EURUSD_*_5min_combined.csv
│   ├── sentiment/                 # Sentiment data
│   │   ├── sentiment_scores_*.csv # EPU + VIX daily sentiment
│   │   └── gdelt_sentiment_*.csv  # GDELT hourly (available, not used)
│   └── sample/                    # Sample data for development
├── docs/
│   ├── 01-current-state-of-the-art.md  # Comprehensive current state
│   └── ...                        # Analysis results and plans
├── frontend/                      # React Web Showcase
│   ├── src/
│   │   ├── components/            # React components
│   │   │   ├── Dashboard.jsx      # Main dashboard layout
│   │   │   ├── PredictionCard.jsx # Current prediction display
│   │   │   ├── AccountStatus.jsx  # Paper trading balance
│   │   │   ├── PerformanceStats.jsx # Trading statistics
│   │   │   ├── TradeHistory.jsx   # Recent trades list
│   │   │   └── PriceChart.jsx     # EUR/USD price chart
│   │   ├── api/
│   │   │   └── client.js          # API client with fetch
│   │   └── hooks/                 # Custom React hooks
│   ├── Dockerfile                 # Multi-stage build (Node → nginx)
│   ├── nginx.conf.template        # Nginx config with API proxy
│   ├── docker-entrypoint.sh       # Runtime env substitution
│   └── railway.json               # Railway deployment config
├── models/
│   └── mtf_ensemble/              # Production models
│       ├── 1H_model.pkl           # 1-hour XGBoost model
│       ├── 4H_model.pkl           # 4-hour XGBoost model
│       ├── D_model.pkl            # Daily XGBoost model (with sentiment)
│       └── training_metadata.json # Configuration and results
├── scripts/
│   ├── train_mtf_ensemble.py      # Training script
│   ├── backtest_mtf_ensemble.py   # Backtesting script
│   ├── download_sentiment_data.py # EPU + VIX download
│   └── download_gdelt_sentiment.py # GDELT download (BigQuery)
├── src/
│   ├── api/                       # FastAPI Backend Service
│   │   ├── main.py                # Application entry point
│   │   ├── scheduler.py           # APScheduler for data updates
│   │   ├── routes/
│   │   │   ├── health.py          # Health check endpoint
│   │   │   ├── predictions.py     # Prediction endpoints
│   │   │   ├── trading.py         # Paper trading endpoints
│   │   │   ├── market.py          # Market data endpoints
│   │   │   └── pipeline.py        # Data pipeline endpoints
│   │   ├── services/
│   │   │   ├── model_service.py   # MTF Ensemble predictions
│   │   │   ├── trading_service.py # Paper trading logic
│   │   │   ├── data_service.py    # Market data fetching
│   │   │   └── pipeline_service.py # Data pipeline management
│   │   ├── schemas/               # Pydantic schemas
│   │   └── database/              # SQLAlchemy models
│   ├── features/
│   │   ├── technical/             # Technical indicators
│   │   │   ├── calculator.py      # TechnicalIndicatorCalculator
│   │   │   └── indicator_registry.py
│   │   └── sentiment/             # Sentiment features
│   │       ├── sentiment_loader.py # EPU/VIX loader
│   │       └── gdelt_loader.py    # GDELT loader
│   ├── models/
│   │   └── multi_timeframe/       # MTF Ensemble (PRIMARY)
│   │       ├── mtf_ensemble.py    # MTFEnsemble class
│   │       ├── improved_model.py  # ImprovedTimeframeModel
│   │       └── enhanced_features.py # EnhancedFeatureEngine
│   ├── simulation/                # Backtesting
│   └── trading/                   # Risk management
├── tests/                         # Python tests (735+ tests)
├── Dockerfile                     # Backend API container
├── docker-compose.yml             # Local orchestration
├── requirements-api.txt           # Production API dependencies
└── railway.json                   # Railway backend config
```

## MTF Ensemble Architecture

The production system uses a Multi-Timeframe Ensemble with weighted averaging:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MTF ENSEMBLE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   5-min Data ──┬──► 1H Model (60% weight) ───────┐              │
│                │    - 115 features               │              │
│                │    - No sentiment               │              │
│                │                                  │              │
│                ├──► 4H Model (30% weight) ───────┼──► Ensemble  │
│                │    - 113 features               │   Prediction │
│                │    - No sentiment               │              │
│                │                                  │              │
│                └──► Daily Model (10% weight) ────┘              │
│                     - 134 features                              │
│                     - VIX + EPU sentiment                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Optimal Configuration

```python
MTFEnsembleConfig(
    weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
    agreement_bonus=0.05,
    use_regime_adjustment=True,
    include_sentiment=True,
    sentiment_source="epu",
    sentiment_by_timeframe={"1H": False, "4H": False, "D": True},
    trading_pair="EURUSD",
)
```

### Key Finding: Resolution Matching Rule

**Sentiment data resolution must match or be finer than the trading timeframe.**

| Configuration | Total Pips | Profit Factor |
|---------------|------------|---------------|
| Baseline (No Sentiment) | +7,596 | 2.12 |
| **EPU Daily-Only (Optimal)** | **+7,987** | **2.22** |
| GDELT All Timeframes | +7,273 | 2.09 |

Daily EPU/VIX sentiment works for Daily model; adding it to 1H/4H models degrades performance.

## Data Sources

### Price Data
- **Source**: MetaTrader 5
- **Pair**: EUR/USD
- **Resolution**: 5-minute (resampled to 1H, 4H, Daily)
- **Period**: 2020-01-01 to 2025-12-31
- **Records**: 448,586 bars
- **Location**: `data/forex/EURUSD_20200101_20251231_5min_combined.csv`

### Sentiment Data
| Dataset | Source | Resolution | Location |
|---------|--------|------------|----------|
| US EPU | FRED | Daily | `data/sentiment/sentiment_scores_*.csv` |
| VIX | FRED | Daily | `data/sentiment/sentiment_scores_*.csv` |
| GDELT | BigQuery | Hourly | `data/sentiment/gdelt_sentiment_*.csv` |

## Walk-Forward Optimization (WFO)

The model has been validated using walk-forward optimization with 7 rolling windows:

| Window | Test Period | Trades | Win Rate | Pips | PF |
|--------|-------------|--------|----------|------|-----|
| 1 | 2022-H1 | 533 | 57.2% | +3,939 | 2.20 |
| 2 | 2022-H2 | 662 | 54.7% | +4,358 | 1.98 |
| 3 | 2023-H1 | 478 | 47.7% | +1,455 | 1.41 |
| 4 | 2023-H2 | 425 | 48.7% | +1,635 | 1.56 |
| 5 | 2024-H1 | 334 | 62.0% | +2,432 | 2.55 |
| 6 | 2024-H2 | 380 | 56.6% | +2,238 | 2.08 |
| 7 | 2025-H1 | 568 | 47.7% | +2,079 | 1.48 |

**Summary:** 100% consistency (7/7 windows profitable), +18,136 total pips, CV=0.40 (stable)

See `docs/02-walk-forward-optimization-results.md` for full analysis.

## Confidence Threshold Optimization

Testing confidence thresholds from 55% to 75% reveals optimal trade filtering:

| Threshold | Trades | Win Rate | Pips | PF | Sharpe |
|-----------|--------|----------|------|-----|--------|
| 55% | 1,103 | 57.8% | +7,987 | 2.22 | 6.09 |
| 60% | 1,055 | 60.3% | +8,447 | 2.43 | 6.80 |
| 65% | 1,016 | 60.8% | +8,561 | 2.54 | 7.16 |
| **70%** | **966** | **62.1%** | **+8,693** | **2.69** | **7.67** |
| 75% | 899 | 63.1% | +8,526 | 2.82 | 8.08 |

**Recommendation:** 70% threshold maximizes total pips while maintaining excellent quality metrics.

See `docs/04-confidence-threshold-optimization.md` for full analysis.

## Market Regime Detection

The model is robust across all market conditions. Regime analysis shows:

| Regime | Trades | Win Rate | Pips | PF | Status |
|--------|--------|----------|------|-----|--------|
| Ranging High Vol | 25 | **84.0%** | +443 | 8.38 | TRADE |
| Ranging Low Vol | 67 | 74.6% | +939 | 5.01 | TRADE |
| Ranging Normal | 84 | 73.8% | +1,142 | 4.89 | TRADE |
| Trending High Vol | 96 | 71.9% | +1,241 | 4.15 | TRADE |
| Trending Low Vol | 96 | 69.8% | +1,239 | 4.09 | TRADE |
| Trending Normal | 211 | 65.9% | +2,303 | 3.31 | TRADE |

**Finding:** All regimes profitable - no regime filtering needed. Model adapts to all market conditions.

See `docs/05-regime-detection-analysis.md` for full analysis.

## Web Showcase

The web showcase provides a live demonstration of the MTF Ensemble trading system with paper trading capabilities.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    WEB SHOWCASE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   React Frontend (Port 3001)                                    │
│   ├── Dashboard.jsx          Main layout with all components    │
│   ├── PredictionCard.jsx     Current BUY/SELL/HOLD signal       │
│   ├── AccountStatus.jsx      $100K paper trading balance        │
│   ├── PerformanceStats.jsx   Win rate, profit factor, etc.      │
│   ├── TradeHistory.jsx       Recent trades table                │
│   └── PriceChart.jsx         EUR/USD price with Recharts        │
│                                                                  │
│   nginx (reverse proxy)                                          │
│   └── /api/* → Backend (Port 8001)                              │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   FastAPI Backend (Port 8001)                                   │
│   ├── /health               Health check                        │
│   ├── /api/v1/predictions   Current and historical predictions  │
│   ├── /api/v1/trading       Paper trading operations            │
│   ├── /api/v1/market        EUR/USD market data (yfinance)      │
│   └── /api/v1/pipeline      Data update management              │
│                                                                  │
│   Services:                                                      │
│   ├── model_service         MTF Ensemble predictions            │
│   ├── trading_service       Paper trading ($100K balance)       │
│   ├── data_service          Market data from yfinance           │
│   └── pipeline_service      Scheduled data updates              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check for container orchestration |
| GET | `/api/v1/predictions/current` | Get current trading prediction |
| GET | `/api/v1/predictions/history` | Get prediction history |
| GET | `/api/v1/trading/account` | Get paper trading account status |
| GET | `/api/v1/trading/positions` | Get open positions |
| GET | `/api/v1/trading/history` | Get trade history |
| POST | `/api/v1/trading/execute` | Execute a trade |
| GET | `/api/v1/market/price` | Get current EUR/USD price |
| GET | `/api/v1/market/history` | Get price history |
| GET | `/api/v1/pipeline/status` | Get data pipeline status |
| POST | `/api/v1/pipeline/refresh` | Trigger data refresh |

### Frontend Components

| Component | Description |
|-----------|-------------|
| `Dashboard.jsx` | Main layout, orchestrates all components |
| `PredictionCard.jsx` | Displays current BUY/SELL/HOLD signal with confidence |
| `AccountStatus.jsx` | Shows paper trading balance ($100K starting) |
| `PerformanceStats.jsx` | Win rate, profit factor, total trades, total pips |
| `TradeHistory.jsx` | Table of recent trades with outcomes |
| `PriceChart.jsx` | EUR/USD price chart using Recharts |

## Docker Deployment

The application is containerized for Railway cloud deployment.

### Container Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DOCKER ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ai-trader-backend (Port 8001)                                 │
│   ├── Base: python:3.12-slim                                    │
│   ├── Dependencies: requirements-api.txt                        │
│   ├── Models: /app/models (mounted read-only)                   │
│   ├── Data: /app/data (mounted for persistence)                 │
│   └── Entry: uvicorn src.api.main:app                           │
│                                                                  │
│   ai-trader-frontend (Port 3001 → 80 internal)                  │
│   ├── Build: node:20-alpine (multi-stage)                       │
│   ├── Serve: nginx:alpine                                       │
│   ├── Proxy: /api/* → backend:8001                              │
│   └── Entry: docker-entrypoint.sh (envsubst)                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Local Development with Docker

```bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down

# Rebuild a specific service
docker-compose build backend
docker-compose build frontend
```

### Railway Deployment

Both services have `railway.json` configuration files:

**Backend (`railway.json`)**:
- Dockerfile build
- Start: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`
- Health: `/health` (60s timeout)

**Frontend (`frontend/railway.json`)**:
- Dockerfile build
- Health: `/health` (30s timeout)
- Environment: `BACKEND_URL` must point to backend service

### Environment Variables

| Variable | Service | Default | Description |
|----------|---------|---------|-------------|
| `PORT` | Backend | 8001 | API server port |
| `DATABASE_URL` | Backend | sqlite:///./data/trading.db | Database connection |
| `FRED_API_KEY` | Backend | - | FRED API key for sentiment |
| `BACKEND_URL` | Frontend | http://backend:8001 | Backend API URL |

## Common Commands

### Training

```bash
# Train with optimal configuration (sentiment on Daily only)
python scripts/train_mtf_ensemble.py --sentiment

# Train without sentiment (baseline)
python scripts/train_mtf_ensemble.py

# Custom sentiment configuration
python scripts/train_mtf_ensemble.py --sentiment-tf "D" --sentiment-source epu

# Custom weights
python scripts/train_mtf_ensemble.py --weights "0.6,0.3,0.1"
```

### Backtesting

```bash
# Backtest with optimal 70% confidence threshold
python scripts/backtest_mtf_ensemble.py --model-dir models/mtf_ensemble --confidence 0.70

# Backtest with default 55% threshold (baseline)
python scripts/backtest_mtf_ensemble.py --model-dir models/mtf_ensemble

# With comparison to individual models
python scripts/backtest_mtf_ensemble.py --compare
```

### Confidence Optimization

```bash
# Run confidence threshold optimization
python scripts/optimize_confidence_threshold.py

# Custom thresholds
python scripts/optimize_confidence_threshold.py --thresholds "0.55,0.60,0.65,0.70,0.75"
```

### Regime Analysis

```bash
# Run regime performance analysis
python scripts/analyze_regime_performance.py --confidence 0.70
```

### Walk-Forward Optimization

```bash
# Run WFO validation (24-month train, 6-month test windows)
python scripts/walk_forward_optimization.py --sentiment

# Custom window configuration
python scripts/walk_forward_optimization.py --sentiment --train-months 24 --test-months 6 --step-months 6
```

### Data Download

```bash
# Download EPU + VIX sentiment data
python scripts/download_sentiment_data.py --start 2020-01-01 --end 2025-12-31

# Download GDELT sentiment (requires Google Cloud credentials)
export GOOGLE_APPLICATION_CREDENTIALS="credentials/gcloud.json"
python scripts/download_gdelt_sentiment.py --start 2020-01-01 --end 2025-12-31
```

### Docker Commands

```bash
# Build and run all services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop all services
docker-compose down

# Build individual services
docker build -t ai-trader-backend .
docker build -t ai-trader-frontend ./frontend
```

### Frontend Development

```bash
# Install dependencies
cd frontend && npm install

# Development server (http://localhost:5173)
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Run tests with UI
npm run test:ui

# Run tests with coverage
npm run test:coverage
```

### Backend API

```bash
# Run API locally (without Docker)
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8001

# Access API documentation
# Swagger UI: http://localhost:8001/docs
# ReDoc: http://localhost:8001/redoc

# Health check
curl http://localhost:8001/health
```

### Testing

```bash
# Run all backend tests (735+ tests)
pytest tests/ -v

# Run frontend tests (35 tests)
cd frontend && npm test

# Run specific test file
pytest tests/api/test_predictions.py -v

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html
```

## Feature Engineering

### Feature Categories (per model)

| Category | 1H | 4H | Daily | Description |
|----------|-----|-----|-------|-------------|
| Trend | 15 | 15 | 15 | EMA, SMA, Supertrend, ADX |
| Momentum | 12 | 12 | 12 | RSI, MACD, Stochastic, MFI |
| Volatility | 10 | 10 | 10 | ATR, Bollinger, Keltner |
| Time | 12 | 8 | 6 | Hour, day, session flags |
| Pattern | 15 | 15 | 15 | Highs, lows, engulfing |
| Cross-TF | 8 | 5 | 0 | HTF trend, HTF RSI |
| Sentiment | 0 | 0 | 24 | VIX, EPU, MAs, z-scores |
| **Total** | **115** | **113** | **134** | |

### Top Features by Model

**1H Model**: is_newyork, htf_4H_trend, supertrend_dir, trend_alignment, dist_ema_21
**4H Model**: price_pctl_50, dist_ema_55, htf_D_rsi, macd, rsi_14_pctl
**Daily Model**: sentiment_ma_30, ema_21, sma_50, ema_55, supertrend_10

## Triple Barrier Labeling

| Timeframe | Take Profit | Stop Loss | Max Holding |
|-----------|-------------|-----------|-------------|
| 1H | 25 pips | 15 pips | 12 bars |
| 4H | 50 pips | 25 pips | 18 bars |
| Daily | 150 pips | 75 pips | 15 bars |

## Key Files Reference

### Production Models
| File | Purpose |
|------|---------|
| `models/mtf_ensemble/1H_model.pkl` | 1-hour XGBoost model |
| `models/mtf_ensemble/4H_model.pkl` | 4-hour XGBoost model |
| `models/mtf_ensemble/D_model.pkl` | Daily XGBoost model (with sentiment) |
| `models/mtf_ensemble/training_metadata.json` | Configuration and results |

### Core Implementation
| File | Purpose |
|------|---------|
| `src/models/multi_timeframe/mtf_ensemble.py` | MTFEnsemble class, MTFEnsembleConfig |
| `src/models/multi_timeframe/improved_model.py` | ImprovedTimeframeModel, labeling |
| `src/models/multi_timeframe/enhanced_features.py` | EnhancedFeatureEngine |
| `src/features/sentiment/sentiment_loader.py` | EPU/VIX sentiment loading |
| `src/features/sentiment/gdelt_loader.py` | GDELT hourly sentiment |
| `src/features/technical/calculator.py` | Technical indicator calculation |

### Web Showcase - Backend API
| File | Purpose |
|------|---------|
| `src/api/main.py` | FastAPI application entry point |
| `src/api/scheduler.py` | APScheduler for data pipeline updates |
| `src/api/routes/predictions.py` | Prediction endpoints |
| `src/api/routes/trading.py` | Paper trading endpoints |
| `src/api/routes/market.py` | Market data endpoints |
| `src/api/routes/pipeline.py` | Data pipeline endpoints |
| `src/api/services/model_service.py` | MTF Ensemble prediction service |
| `src/api/services/trading_service.py` | Paper trading logic |
| `src/api/services/data_service.py` | Market data fetching (yfinance) |
| `src/api/services/pipeline_service.py` | Data pipeline management |

### Web Showcase - Frontend
| File | Purpose |
|------|---------|
| `frontend/src/components/Dashboard.jsx` | Main dashboard layout |
| `frontend/src/components/PredictionCard.jsx` | Current prediction display |
| `frontend/src/components/AccountStatus.jsx` | Paper trading balance |
| `frontend/src/components/PerformanceStats.jsx` | Trading statistics |
| `frontend/src/components/TradeHistory.jsx` | Recent trades table |
| `frontend/src/components/PriceChart.jsx` | EUR/USD price chart |
| `frontend/src/api/client.js` | API client (fetch wrapper) |

### Docker & Deployment
| File | Purpose |
|------|---------|
| `Dockerfile` | Backend API container |
| `frontend/Dockerfile` | Frontend multi-stage build |
| `docker-compose.yml` | Local orchestration |
| `railway.json` | Railway backend config |
| `frontend/railway.json` | Railway frontend config |
| `frontend/nginx.conf.template` | Nginx reverse proxy config |
| `frontend/docker-entrypoint.sh` | Runtime env substitution |
| `requirements-api.txt` | Production API dependencies |
| `.dockerignore` | Backend build exclusions |
| `frontend/.dockerignore` | Frontend build exclusions |

### Scripts
| File | Purpose |
|------|---------|
| `scripts/train_mtf_ensemble.py` | Training with all options |
| `scripts/backtest_mtf_ensemble.py` | Backtesting simulation |
| `scripts/walk_forward_optimization.py` | WFO validation (robustness testing) |
| `scripts/optimize_confidence_threshold.py` | Confidence threshold optimization |
| `scripts/analyze_regime_performance.py` | Market regime analysis |
| `scripts/backtest_position_sizing.py` | Kelly criterion position sizing comparison |
| `scripts/download_sentiment_data.py` | EPU + VIX download |
| `scripts/download_gdelt_sentiment.py` | GDELT BigQuery download |

### Documentation
| File | Purpose |
|------|---------|
| `docs/01-current-state-of-the-art.md` | **Comprehensive current state** |
| `docs/02-walk-forward-optimization-results.md` | **WFO validation results** |
| `docs/03-kelly-criterion-position-sizing.md` | **Kelly position sizing** |
| `docs/04-confidence-threshold-optimization.md` | **Confidence optimization results** |
| `docs/05-regime-detection-analysis.md` | **Market regime analysis** |
| `docs/06-web-showcase-implementation-plan.md` | **Web showcase roadmap** |

## Technology Stack

| Component | Technology |
|-----------|------------|
| **Backend** | |
| Language | Python 3.12 |
| ML Models | XGBoost |
| Indicators | pandas-ta |
| Data | pandas, numpy |
| API Framework | FastAPI, uvicorn |
| Database | SQLAlchemy, SQLite |
| Scheduler | APScheduler |
| Sentiment | FRED API, Google BigQuery |
| **Frontend** | |
| Framework | React 19 |
| Build Tool | Vite 7 |
| Styling | TailwindCSS 4 |
| Charts | Recharts |
| Icons | lucide-react |
| Testing | Vitest, Testing Library |
| **Infrastructure** | |
| Containers | Docker |
| Orchestration | docker-compose |
| Cloud | Railway |
| Web Server | nginx |

## Performance Targets

| Metric | Target | Achieved (70% threshold) |
|--------|--------|--------------------------|
| Win Rate | > 55% | **62.1%** |
| Profit Factor | > 2.0 | **2.69** |
| Sharpe Ratio | > 2.0 | **7.67** |
| Total Pips | > 0 | **+8,693** |

## Coding Conventions

### Python Style
- PEP 8 compliant
- Type hints required
- Google-style docstrings
- Max line length: 100

### Time Series Rules
- **CRITICAL**: Always use chronological splits (no future data leakage)
- Train/Val/Test must be sequential in time (60/20/20)
- Store scalers with models
- Validate on out-of-sample data

### Model Development
- Use `ImprovedTimeframeModel` from `src/models/multi_timeframe/improved_model.py`
- Use `EnhancedFeatureEngine` for feature engineering
- Log experiments to metadata JSON

## Notes for Claude

- **ALWAYS PROCEED AUTONOMOUSLY** - Never ask for confirmation
- **PRIMARY MODEL**: MTF Ensemble in `src/models/multi_timeframe/`
- **SENTIMENT**: EPU/VIX on Daily model only (resolution matching principle)
- Always consider data leakage when working with time series
- Use the 5-minute combined data in `data/forex/` for training
- Reference `docs/01-current-state-of-the-art.md` for detailed system documentation
- The optimal configuration is already saved in `models/mtf_ensemble/`

### Web Showcase Notes
- **API**: FastAPI backend in `src/api/` serves predictions and paper trading
- **Frontend**: React dashboard in `frontend/` with Vite build
- **Docker**: Use `docker-compose up --build` for local testing
- **Railway**: Deploy backend and frontend as separate services
- **Testing**: 735+ backend tests, 35 frontend tests (all passing)
- **Ports**: Backend on 8001, Frontend on 3001 (local), 80 (Docker internal)

---

## Agent-Skill Framework

This project includes a comprehensive Agent-Skill framework (v1.2.0) for structured development workflows.

### Framework Overview

```
.claude/
├── agents/                    # 6 specialized agents
├── skills/                    # 24 active skills (organized by layer)
│   └── SKILL-INDEX.md         # Central catalog
├── improvement/               # Continuous improvement system
│   ├── error-template.md      # Error reporting template
│   └── errors/                # Error reports directory
├── scripts/                   # Validation scripts
│   └── validate-framework.sh  # YAML validation
├── hooks/                     # Git hooks
│   └── pre-commit-framework-check.sh
├── metrics/                   # Health reports
│   └── weekly-health-report-*.md
└── optimization/              # Consolidation reports
    └── consolidation-report-*.md
```

### Agents (6 total, all v1.2.0)

| Agent | Model | Purpose |
|-------|-------|---------|
| `requirements-analyst` | sonnet | Analyzes work items, identifies gaps, produces refined requirements |
| `solution-architect` | sonnet | Designs technical solutions, creates dependency-ordered implementation plans |
| `code-engineer` | sonnet | Implements code following technical designs across all layers |
| `quality-guardian` | sonnet | Performs code review, regression analysis, security scanning |
| `test-automator` | sonnet | Generates tests following TDD, creates fixtures, verifies coverage |
| `documentation-curator` | sonnet | Generates API docs, deployment guides, release notes |

**Agent Workflow:**
```
requirements-analyst → solution-architect → code-engineer → quality-guardian → test-automator → documentation-curator
```

### Skills by Layer (24 active)

| Layer | Skills | Primary Skill |
|-------|--------|---------------|
| **Meta** | 2 | `routing-to-skills`, `improving-framework-continuously` |
| **Backend** | 6 | `backend` (FastAPI endpoints) |
| **Frontend** | 2 | `frontend` (React components) |
| **Database** | 1 | `database` (SQLAlchemy models) |
| **Testing** | 2 | `testing` (pytest), `writing-vitest-tests` |
| **Quality & Testing** | 4 | `planning-test-scenarios`, `generating-test-data` |
| **Feature Engineering** | 2 | `creating-technical-indicators` |
| **Data Layer** | 1 | `adding-data-sources` |
| **Trading Domain** | 3 | `running-backtests`, `implementing-risk-management` |
| **Build & Deployment** | 1 | `build-deployment` (CLI scripts) |

**Full skill index:** `.claude/skills/SKILL-INDEX.md`

### Quick Reference: When to Use Skills

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
| Python tests | `testing` |
| Frontend tests | `writing-vitest-tests` |
| CLI script | `build-deployment` |

### Framework Commands

```bash
# Validate all skills and agents (run before commits)
.claude/scripts/validate-framework.sh

# Install pre-commit hook (one-time)
cp .claude/hooks/pre-commit-framework-check.sh .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### Framework Health

| Metric | Current | Target |
|--------|---------|--------|
| Health Score | 97/100 | >90 |
| Skills | 24 | 15-25 |
| Agents | 6 | 6 |
| YAML Validation | 100% | 100% |
| Error Backlog | 0 | <10 |

**Reports:**
- Weekly health: `.claude/metrics/weekly-health-report-*.md`
- Monthly consolidation: `.claude/optimization/consolidation-report-*.md`

### Skill Invocation

Skills are invoked via the `Skill` tool or slash commands:

```
/backend     - Create FastAPI endpoints
/frontend    - Create React components
/database    - Create SQLAlchemy models
/testing     - Write pytest tests
```

### Anti-Hallucination Features (v1.2.0)

The `routing-to-skills` meta-skill includes:
- **Verification requirements**: Must read actual files before citing
- **Citation requirements**: Include file:line references for claims
- **Uncertainty permission**: Can say "I need to check the codebase"
- **Grounding validation**: Verify references exist after generation
