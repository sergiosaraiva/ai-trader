# Web Showcase Implementation Plan

**Date:** January 12, 2026
**Status:** Backend COMPLETE, Frontend PENDING
**Objective:** Deploy AI-Trader as an online web showcase with near real-time predictions and simulated trading statistics.

---

## Executive Summary

This document outlines the implementation plan for the AI-Trader web showcase. The **backend is fully implemented** and ready for frontend development.

### Current Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| **API Server** | COMPLETE | FastAPI with all endpoints working |
| **Data Pipeline** | COMPLETE | Automatic CSV updates, gap recovery |
| **Model Service** | COMPLETE | Singleton pattern, MTF Ensemble loaded |
| **Data Service** | COMPLETE | yfinance integration + historical CSV |
| **Trading Service** | COMPLETE | Paper trading with $100K virtual balance |
| **Database** | COMPLETE | SQLite with predictions, trades, performance |
| **Scheduler** | COMPLETE | APScheduler for hourly updates |
| **Frontend** | PENDING | React dashboard to be built |
| **Docker** | PENDING | Containerization to be done |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AI-TRADER WEB SHOWCASE ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                         BACKEND (COMPLETE)                          │ │
│  │                                                                      │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │ │
│  │  │  Pipeline Service │  │  Model Service   │  │ Trading Service  │  │ │
│  │  │                   │  │                  │  │                  │  │ │
│  │  │ • Update CSV      │  │ • MTF Ensemble   │  │ • Paper trading  │  │ │
│  │  │ • Fetch yfinance  │  │ • 1H/4H/D models │  │ • $100K balance  │  │ │
│  │  │ • Calc indicators │  │ • 70% threshold  │  │ • P&L tracking   │  │ │
│  │  │ • Update sentiment│  │ • Predictions    │  │ • Trade history  │  │ │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘  │ │
│  │                                                                      │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │ │
│  │  │   Data Service   │  │    Scheduler     │  │     Database     │  │ │
│  │  │                   │  │                  │  │                  │  │ │
│  │  │ • Historical CSV  │  │ • :55 Pipeline   │  │ • SQLite         │  │ │
│  │  │ • Live yfinance   │  │ • :01 Predict    │  │ • Predictions    │  │ │
│  │  │ • VIX data        │  │ • :05 Snapshot   │  │ • Trades         │  │ │
│  │  │ • Hybrid approach │  │ • 5min checks    │  │ • Performance    │  │ │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘  │ │
│  │                                                                      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        FRONTEND (PENDING)                           │ │
│  │                                                                      │ │
│  │  React + Vite + TailwindCSS                                         │ │
│  │  • Dashboard with prediction cards                                   │ │
│  │  • Price chart with candlesticks                                    │ │
│  │  • Equity curve                                                      │ │
│  │  • Trade history table                                               │ │
│  │                                                                      │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: Backend Implementation (COMPLETE)

### 1.1 Files Created

| File | Purpose | Status |
|------|---------|--------|
| `src/api/main.py` | FastAPI app with lifespan | COMPLETE |
| `src/api/scheduler.py` | APScheduler for periodic tasks | COMPLETE |
| `src/api/services/pipeline_service.py` | Data pipeline with CSV updates | COMPLETE |
| `src/api/services/model_service.py` | MTF Ensemble inference | COMPLETE |
| `src/api/services/data_service.py` | yfinance + historical data | COMPLETE |
| `src/api/services/trading_service.py` | Paper trading simulation | COMPLETE |
| `src/api/database/models.py` | SQLAlchemy models | COMPLETE |
| `src/api/database/session.py` | DB session management | COMPLETE |
| `src/api/routes/predictions.py` | Prediction endpoints | COMPLETE |
| `src/api/routes/trading.py` | Trading endpoints | COMPLETE |
| `src/api/routes/market.py` | Market data endpoints | COMPLETE |
| `src/api/routes/pipeline.py` | Pipeline management endpoints | COMPLETE |
| `src/api/routes/health.py` | Health check endpoints | COMPLETE |
| `src/api/schemas/*.py` | Pydantic request/response schemas | COMPLETE |

### 1.2 API Endpoints (All Working)

#### Health Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Basic health check |
| `/health/detailed` | GET | Component status |
| `/health/ready` | GET | Readiness check |

#### Pipeline Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/pipeline/status` | GET | Pipeline status, data quality |
| `/api/v1/pipeline/run` | POST | Trigger pipeline (async) |
| `/api/v1/pipeline/run-sync` | POST | Trigger pipeline (wait) |
| `/api/v1/pipeline/data/{tf}` | GET | Data summary for timeframe |

#### Prediction Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/predictions/latest` | GET | Current prediction |
| `/api/v1/predictions/history` | GET | Prediction history |

#### Trading Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/trading/status` | GET | Balance, open position |
| `/api/v1/trading/history` | GET | Trade history |
| `/api/v1/trading/performance` | GET | Win rate, P&L stats |
| `/api/v1/trading/equity-curve` | GET | Balance over time |

#### Market Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/market/current` | GET | Current EUR/USD price |
| `/api/v1/market/candles` | GET | OHLCV candlestick data |

### 1.3 Scheduler Jobs

| Job | Schedule | Description |
|-----|----------|-------------|
| Run Data Pipeline | `:55` each hour | Update CSV, features, sentiment |
| Generate Prediction | `:01` each hour | Make prediction with fresh data |
| Save Performance | `:05` each hour | Record performance snapshot |
| Fetch Market Data | Every 5 min | Update current price display |
| Check Positions | Every 5 min | Monitor open trades for TP/SL |

---

## Part 2: Data Pipeline (COMPLETE)

### 2.1 Pipeline Features

The data pipeline keeps all CSV files continuously updated:

1. **Price Data Updates**
   - Fetches new 5-min bars from yfinance
   - Appends to historical CSV (450,000+ bars)
   - Creates backup before each update
   - Smart gap recovery for extended downtime

2. **Technical Indicators**
   - Resamples to 1H, 4H, Daily timeframes
   - Calculates 50+ technical indicators
   - Adds enhanced features (128 total for 1H)

3. **Sentiment Data**
   - Fetches VIX and EPU from FRED API
   - Calculates sentiment scores
   - Updates sentiment CSV

### 2.2 Gap Recovery Strategy

The pipeline handles service downtime intelligently:

| Gap Duration | Data Source | Resolution |
|--------------|-------------|------------|
| < 60 days | yfinance 5-min | Native 5-minute bars |
| 60-730 days | yfinance hourly | Interpolated to 5-min |
| > 730 days | yfinance daily | Interpolated to 5-min |

### 2.3 Data Files

| File | Description | Size |
|------|-------------|------|
| `data/forex/EURUSD_*_5min_combined.csv` | 5-min OHLCV data | ~46 MB |
| `data/forex/EURUSD_*_5min_combined.csv.backup` | Backup before update | ~46 MB |
| `data/sentiment/sentiment_scores_*_daily.csv` | VIX + EPU sentiment | ~650 KB |
| `data/cache/eurusd_1h_features.parquet` | 1H with all features | Cache |
| `data/cache/eurusd_4h_features.parquet` | 4H with all features | Cache |
| `data/cache/eurusd_daily_features.parquet` | Daily with features + sentiment | Cache |

### 2.4 Pipeline Status API Response

```json
{
  "status": "ok",
  "pipeline": {
    "initialized": true,
    "last_update": "2026-01-12T17:40:17",
    "cache_files": {
      "5min": true,
      "1h": true,
      "4h": true,
      "daily": true,
      "sentiment": true
    },
    "data_info": {
      "total_bars": 450548,
      "date_range": {
        "start": "2020-01-01T22:00:00",
        "end": "2026-01-12T17:35:00"
      },
      "current_gap_days": 0,
      "data_quality": "good"
    },
    "csv_info": {
      "path": "data/forex/EURUSD_*_5min_combined.csv",
      "size_mb": 45.57,
      "backup_exists": true
    }
  }
}
```

---

## Part 3: Running the Backend

### 3.1 Quick Start

```bash
# Navigate to project
cd /home/sergio/ai-trader

# Activate virtual environment
source .venv/bin/activate

# Start the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8001

# Server will:
# 1. Initialize database
# 2. Run data pipeline (updates CSV, ~15-20 seconds)
# 3. Load MTF Ensemble model
# 4. Start scheduler for automatic updates
```

### 3.2 Testing Endpoints

```bash
# Health check
curl http://localhost:8001/health

# Pipeline status
curl http://localhost:8001/api/v1/pipeline/status

# Trigger pipeline update
curl -X POST http://localhost:8001/api/v1/pipeline/run-sync

# Get prediction
curl http://localhost:8001/api/v1/predictions/latest

# Get trading status
curl http://localhost:8001/api/v1/trading/status

# Interactive API docs
open http://localhost:8001/docs
```

### 3.3 Sample Prediction Response

```json
{
  "timestamp": "2026-01-12T17:39:34",
  "symbol": "EURUSD",
  "direction": "long",
  "confidence": 0.771,
  "should_trade": true,
  "agreement_count": 3,
  "all_agree": true,
  "market_regime": "ranging",
  "market_price": 1.16727,
  "vix_value": 15.17,
  "component_directions": {"1H": 1, "4H": 1, "D": 1},
  "component_confidences": {"1H": 0.584, "4H": 0.650, "D": 0.785},
  "component_weights": {"1H": 0.7, "4H": 0.25, "D": 0.05}
}
```

---

## Part 4: Frontend Implementation (PENDING)

### 4.1 Technology Stack

**Recommended:** React + Vite + TailwindCSS

### 4.2 Dashboard Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AI-TRADER DASHBOARD                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────┐  ┌────────────────────────────────────┐ │
│  │    CURRENT PREDICTION      │  │         ACCOUNT STATUS             │ │
│  │                            │  │                                    │ │
│  │    Direction: ▲ LONG       │  │  Balance: $100,000.00             │ │
│  │    Confidence: 77%         │  │  P&L Today: $0.00                 │ │
│  │    Agreement: 3/3 ✓        │  │  Open Position: None              │ │
│  │    Should Trade: YES       │  │                                    │ │
│  │                            │  │                                    │ │
│  └────────────────────────────┘  └────────────────────────────────────┘ │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        PRICE CHART                                 │  │
│  │  [Candlestick chart with prediction signals]                      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌────────────────────────────┐  ┌────────────────────────────────────┐ │
│  │    PERFORMANCE STATS       │  │        RECENT TRADES              │ │
│  │                            │  │                                    │ │
│  │  Win Rate: --              │  │  No trades yet                    │ │
│  │  Profit Factor: --         │  │                                    │ │
│  │  Total Trades: 0           │  │                                    │ │
│  │                            │  │                                    │ │
│  └────────────────────────────┘  └────────────────────────────────────┘ │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        MODEL INFO                                  │  │
│  │  • MTF Ensemble (1H: 60%, 4H: 30%, Daily: 10%)                   │  │
│  │  • Backtest: +8,693 pips, 62.1% win rate, 2.69 PF                │  │
│  │  • Confidence threshold: 70%                                      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Frontend File Structure

```
frontend/
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
├── src/
│   ├── main.jsx
│   ├── App.jsx
│   ├── api/
│   │   └── client.js           # API client
│   ├── components/
│   │   ├── Dashboard.jsx
│   │   ├── PredictionCard.jsx
│   │   ├── AccountStatus.jsx
│   │   ├── PriceChart.jsx
│   │   ├── PerformanceStats.jsx
│   │   ├── TradeHistory.jsx
│   │   └── ModelInfo.jsx
│   └── hooks/
│       └── usePolling.js       # Auto-refresh
└── public/
    └── favicon.ico
```

### 4.4 Data Refresh Strategy

| Data Type | Refresh Interval |
|-----------|-----------------|
| Current price | 30 seconds |
| Prediction | 1 minute (check for new) |
| Pipeline status | 1 minute |
| Trade history | 1 minute |
| Performance stats | 5 minutes |

---

## Part 5: Docker Deployment (PENDING)

### 5.1 Backend Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY models/mtf_ensemble/ ./models/mtf_ensemble/
COPY data/ ./data/

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.2 Deployment Options

| Provider | Cost | Notes |
|----------|------|-------|
| Railway | ~$5/month | Easy deployment |
| Render | Free tier | Good for demo |
| Google Cloud Run | ~$5-20/month | Scalable |
| DigitalOcean | ~$5-12/month | Simple VPS |

---

## Part 6: Next Steps

### Immediate (Frontend Development)

1. Initialize React + Vite project in `frontend/`
2. Create API client to call backend endpoints
3. Build dashboard components
4. Add polling for auto-refresh
5. Style with TailwindCSS

### Backend Requirements for Frontend

The backend **must be running** for the frontend to work:
- Provides all data via REST API
- Updates CSV files automatically via scheduler
- Generates predictions hourly
- Tracks paper trading

### Start Backend for Development

```bash
# Terminal 1: Start backend
cd /home/sergio/ai-trader
source .venv/bin/activate
uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2: Start frontend (once built)
cd /home/sergio/ai-trader/frontend
npm run dev
```

---

## Summary

### What's Complete
- All backend services (pipeline, model, data, trading)
- All API endpoints
- Automatic data updates via scheduler
- CSV persistence with gap recovery
- Paper trading simulation

### What's Pending
- Frontend React application
- Docker containerization
- Cloud deployment

### Key Files
| File | Description |
|------|-------------|
| `src/api/main.py` | Main FastAPI application |
| `src/api/services/pipeline_service.py` | Data pipeline (CSV updates) |
| `src/api/services/model_service.py` | MTF Ensemble predictions |
| `src/api/scheduler.py` | Automatic hourly updates |

### Running the Backend

```bash
source .venv/bin/activate
uvicorn src.api.main:app --port 8001
```

The backend will automatically:
1. Update CSV files with latest market data
2. Calculate technical indicators
3. Generate predictions hourly
4. Track paper trades

**Yes, a running backend service is required** to keep the data updated and serve the frontend.
