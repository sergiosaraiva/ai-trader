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

**Status:** WFO Validated | Docker/Railway Ready

| Metric | Value (All Time) | Value (70% Conf) |
|--------|------------------|------------------|
| Total Pips | +14,637 | +13,718 |
| Win Rate | 50.8% | 53.1% |
| Profit Factor | 1.58x | 1.75x |
| Total Trades | 3,801 | 2,853 |
| WFO Windows | 8/8 profitable (100% consistency) |

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
# Training
python scripts/train_mtf_ensemble.py --sentiment --stacking          # Production (recommended)
python scripts/train_mtf_ensemble.py --sentiment                     # Without stacking

# Backtesting
python scripts/backtest_mtf_ensemble.py --model-dir models/mtf_ensemble --confidence 0.70

# WFO Validation
python scripts/walk_forward_optimization.py --sentiment --stacking

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

```python
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
- **Time Series:** CRITICAL - chronological splits only (no future data leakage), Train/Val/Test 60/20/20
- **Models:** Use `ImprovedTimeframeModel` and `EnhancedFeatureEngine`

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
