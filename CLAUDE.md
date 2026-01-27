# CLAUDE.md - AI Assets Trader Project Guide

## Interaction Mode

**ALWAYS PROCEED WITHOUT ASKING FOR CONFIRMATION.** Act decisively and complete tasks end-to-end. Only ask questions for genuine ambiguity that cannot be resolved from context.

## Mandatory Agent Workflow

**ALL code changes MUST use the agent pipeline:**

| Request Type | Pipeline |
|-------------|----------|
| Bug fix, simple endpoint | `code-engineer` → `quality-guardian` → `test-automator` |
| Feature implementation | `solution-architect` → `code-engineer` → `quality-guardian` → `test-automator` |
| Complex feature | `requirements-analyst` → `solution-architect` → `code-engineer` → `quality-guardian` → `test-automator` |
| API changes | Add `documentation-curator` at end |

**Non-code tasks (no agents):** Research, documentation reading, explaining code, running scripts

## Project Overview

Production-ready **Multi-Timeframe (MTF) Ensemble** forex trading system using XGBoost models (1H, 4H, Daily) with sentiment analysis. Includes React frontend + FastAPI backend.

**Status:** WFO Validated | Docker/Railway Ready | Config C Active

| Metric | WFO (9 Windows) | Config C Details |
|--------|-----------------|------------------|
| Total Trades | 1,257 | 60% confidence threshold |
| Win Rate | 53.9% | Consistent across windows |
| Total Pips | +6,202 | All test periods |
| Max Drawdown | 15.1% | Circuit breaker limit |
| Profitable Windows | 9/9 (100%) | Perfect consistency |
| Test Period | 4.5 years | 2021-2025 validation |

**Active Configuration (Config C):**
- Confidence Threshold: 60%
- Training Window: 18 months
- Model Directory: `models/wfo_conf60_18mo/window_9`
- Validation Method: Walk-Forward Optimization (18mo train, 6mo test, 6mo roll)
- Risk Management: Progressive risk reduction + 15% circuit breaker

## Architecture

```
5-min Data ─┬─► 1H Model (115 features, no sentiment) ─┐
            ├─► 4H Model (113 features, no sentiment) ─┼─► Meta-Learner ─► Prediction
            └─► Daily Model (134 features, VIX+EPU) ───┘    (XGBoost)
```

**Key Rule:** Sentiment data resolution must match trading timeframe → EPU/VIX on Daily only.

## Project Structure

```
ai-trader/
├── .claude/                    # Agent-Skill Framework v1.2.0
│   ├── agents/                 # 6 specialized agents
│   ├── skills/                 # 27 active skills
│   └── improvement/            # Error reporting & metrics
├── backend/
│   ├── src/api/                # FastAPI (routes, services, schemas, database)
│   ├── src/models/             # ML models (multi_timeframe/)
│   ├── src/features/           # Feature engineering (technical, sentiment)
│   ├── models/mtf_ensemble/    # Production models (*.pkl)
│   ├── data/forex/             # EUR/USD 5-min data (448K bars)
│   ├── data/sentiment/         # EPU + VIX data
│   ├── scripts/                # Training & data scripts
│   └── tests/                  # 735+ tests
├── frontend/
│   ├── src/components/         # React components
│   └── src/api/                # API client
├── docker-compose.yml          # Local orchestration
└── Dockerfile(s)               # Railway deployment
```

## Common Commands

All commands from `backend/` directory unless noted.

```bash
# WFO Validation (MANDATORY for model validation) - Config C defaults
python scripts/walk_forward_optimization.py --sentiment --stacking    # 18mo train, 60% conf
python scripts/walk_forward_optimization.py --sentiment --stacking --confidence 0.65  # Test higher threshold

# Training (for production deployment only - NOT for validation)
python scripts/train_mtf_ensemble.py --sentiment --stacking          # Train on full dataset
# ⚠️ WARNING: Always validate with WFO first before training production model

# Backtesting - Config C
python scripts/backtest_mtf_ensemble.py --model-dir models/wfo_conf60_18mo/window_9 --confidence 0.60
# ⚠️ Only use for specific analysis - WFO is the authoritative validation

# Docker (from project root)
docker-compose up --build       # Build and start
docker-compose logs -f          # View logs

# Frontend (from frontend/)
npm run dev                     # Dev server (localhost:5173)
npm test                        # Run tests

# Backend API
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8001
# Docs: http://localhost:8001/docs
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /api/v1/predictions/current` | Current prediction |
| `GET /api/v1/predictions/history` | Prediction history |
| `GET /api/v1/trading/account` | Paper trading account ($100K) |
| `GET /api/v1/trading/positions` | Open positions |
| `POST /api/v1/trading/execute` | Execute trade |
| `GET /api/v1/market/price` | Current EUR/USD price |
| `GET /api/v1/pipeline/status` | Data pipeline status |

## Key Files

| Category | Files |
|----------|-------|
| **Models** | `models/mtf_ensemble/{1H,4H,D}_model.pkl`, `stacking_meta_learner.pkl` |
| **Core** | `src/models/multi_timeframe/mtf_ensemble.py`, `stacking_meta_learner.py`, `improved_model.py` |
| **Features** | `src/features/sentiment/sentiment_loader.py`, `src/features/technical/calculator.py` |
| **API** | `src/api/main.py`, `src/api/services/model_service.py`, `src/api/routes/` |
| **Frontend** | `frontend/src/components/Dashboard.jsx`, `PredictionCard.jsx`, `PriceChart.jsx` |

## Configuration

**Config C (Active Production Settings):**
```python
# Trading Parameters
confidence_threshold = 0.60  # Minimum confidence to trade
training_window = 18  # months
model_directory = "models/wfo_conf60_18mo/window_9"

# Model Ensemble
MTFEnsembleConfig(
    weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
    use_stacking=True,
    include_sentiment=True,
    sentiment_by_timeframe={"1H": False, "4H": False, "D": True},
)
```

**Triple Barrier:** 1H (25/15 pips, 12 bars) | 4H (50/25 pips, 18 bars) | Daily (150/75 pips, 15 bars)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8001 | API server port |
| `DATABASE_URL` | sqlite:///./data/trading.db | Database connection |
| `FRED_API_KEY` | - | FRED API for sentiment |
| `BACKEND_URL` | http://backend:8001 | Backend URL (frontend) |

## Coding Conventions

- **Python:** PEP 8, type hints, Google docstrings, max 100 chars
- **Time Series:** CRITICAL - See "Data Leakage Prevention" section below
- **Models:** Use `ImprovedTimeframeModel` and `EnhancedFeatureEngine`

## Data Leakage Prevention

**ZERO TOLERANCE FOR DATA LEAKAGE.** This is the most critical aspect of the project.

### Mandatory Validation Method: Walk-Forward Optimization (WFO)

**NEVER train with simple 60/20/20 split.** Always use WFO for model validation:

```bash
# CORRECT: WFO validation (mandatory)
cd backend
python scripts/walk_forward_optimization.py --sentiment --stacking

# WRONG: Single train/test split (DO NOT USE for validation)
python scripts/train_mtf_ensemble.py --sentiment --stacking
```

### WFO Configuration

- **Training window:** 24 months
- **Test window:** 6 months (out-of-sample, never seen during training)
- **Step size:** 6 months (roll forward)
- **Result:** 8 independent test windows across different market regimes

### Pre-Backtest Checklist

Before running ANY backtest, verify:

1. ✅ **Check model training dates:**
   ```bash
   cat models/mtf_ensemble/training_metadata.json
   # Note: train_end date
   ```

2. ✅ **Verify backtest uses ONLY test data:**
   - Backtest start date MUST be >= train_end date
   - NO overlap between training and testing data
   - Example: If trained on 2020-2023, backtest ONLY on 2024+

3. ✅ **Use WFO results for validation:**
   ```bash
   cat models/wfo_validation/wfo_results.json
   # Review all 8 windows for consistency
   ```

4. ✅ **Check for data availability:**
   ```bash
   tail data/forex/EURUSD_*.csv  # Verify date range
   tail data/sentiment/*.csv      # Verify sentiment coverage
   ```

### Data Leakage Red Flags

🚨 **STOP IMMEDIATELY** if you see:

1. Backtest returns > 1000% over 2-3 years
2. Win rate > 70% consistently
3. Max drawdown = 0.0% or exactly 15.0%
4. Sharpe ratio > 5.0
5. Profit factor > 5.0

These are signs of data leakage. Verify train/test split immediately.

### Training Data Splits

**Within each WFO window (for model training only):**
- Training: 80% of window data
- Validation: 20% of window data
- **NEVER touch test data during training**

**For chronological data:**
```python
# CORRECT: Chronological split (no shuffling)
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_val = df.iloc[train_size:]

# WRONG: Random split (causes data leakage in time series)
train_test_split(df, shuffle=True)  # DO NOT USE
```

### WFO Results Location

- **WFO models:** `backend/models/wfo_validation/window_*/`
- **WFO results:** `backend/models/wfo_validation/wfo_results.json`
- **Window 7 analysis:** `backend/docs/WFO_WINDOW_7_ANALYSIS.md` (only 3 trades - regime change)

### Documentation

All WFO validation details are in:
- `backend/scripts/walk_forward_optimization.py` - WFO implementation
- `backend/docs/WFO_WINDOW_7_ANALYSIS.md` - Window anomaly analysis
- `backend/models/wfo_validation/wfo_results.json` - 8-window results

## Technology Stack

**Backend:** Python 3.12, XGBoost, FastAPI, SQLAlchemy, pandas-ta, APScheduler
**Frontend:** React 19, Vite 7, TailwindCSS 4, Recharts, Vitest
**Infrastructure:** Docker, nginx, Railway

---

## Agent-Skill Framework

**CRITICAL:** Use agents for ALL code tasks. DO NOT implement manually.

### Agent Registry

| Agent | Purpose | Triggers |
|-------|---------|----------|
| `requirements-analyst` | Analyze requirements, identify gaps | "analyze requirements", "what's missing" |
| `solution-architect` | Design solutions, create implementation plans | "design", "plan", "how should we implement" |
| `code-engineer` | Implement code changes | "implement", "add", "create", "fix" |
| `quality-guardian` | Code review, security scan, regression check | "review", "check quality", "security scan" |
| `test-automator` | Generate tests, verify coverage | "write tests", "add coverage" |
| `documentation-curator` | API docs, release notes | "document", "API docs" |

### Agent Selection

```
Requirements analysis? → requirements-analyst
Design/planning?       → solution-architect
Code implementation?   → code-engineer
Review/quality?        → quality-guardian
Testing?               → test-automator
Documentation?         → documentation-curator
```

### Skills Quick Reference

| Task | Skill |
|------|-------|
| API endpoint | `backend` |
| React component | `frontend` |
| Charts | `creating-chart-components` |
| Database model | `database` |
| ML features | `creating-ml-features` |
| Caching | `implementing-caching-strategies` |
| Tests (Python) | `testing` |
| Tests (Frontend) | `writing-vitest-tests` |

**Full index:** `.claude/skills/SKILL-INDEX.md`

### Framework Validation

```bash
.claude/scripts/validate-framework.sh    # Validate before commits
```
