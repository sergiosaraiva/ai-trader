# AI Trader

A Multi-Timeframe Ensemble Trading System using XGBoost for technical analysis-based forex trading predictions with sentiment analysis integration.

## Performance (Current Production Model)

| Metric | Value |
|--------|-------|
| **Total Profit** | **+7,987 pips** |
| **Win Rate** | 57.8% |
| **Profit Factor** | 2.22 |
| **Total Trades** | 1,103 |
| **Avg Pips/Trade** | +7.2 |

### Model Accuracy

| Model | Val Accuracy | High-Conf (≥60%) |
|-------|--------------|------------------|
| 1H (60% weight) | 67.07% | 72.14% |
| 4H (30% weight) | 65.43% | 71.12% |
| Daily (10% weight) | 61.54% | 64.21% |

## Features

- **Multi-Timeframe Ensemble**: Combines 1H, 4H, and Daily XGBoost models
- **Sentiment Integration**: VIX + EPU sentiment on Daily model (research-optimized)
- **115+ Technical Features**: Trend, momentum, volatility, patterns, cross-TF alignment
- **Triple Barrier Labeling**: Realistic TP/SL/Timeout trade outcomes
- **Confidence-Based Trading**: Higher accuracy on high-confidence signals
- **6 Years of Data**: Trained on EUR/USD 2020-2025 (448K 5-minute bars)
- **AI Trading Agent**: Autonomous trading with safety systems and real-time monitoring
- **Web Dashboard**: React frontend for live predictions and performance tracking
- **Docker Deployment**: Production-ready containerized deployment

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sergiosaraiva/ai-trader.git
cd ai-trader

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r backend/requirements.txt
```

### Train the Model

```bash
cd backend

# Train with optimal configuration (sentiment on Daily only)
python scripts/train_mtf_ensemble.py --sentiment

# Train without sentiment
python scripts/train_mtf_ensemble.py
```

### Run Backtest

```bash
cd backend

# Backtest the trained model
python scripts/backtest_mtf_ensemble.py --model-dir models/mtf_ensemble
```

### Run with Docker (Web Showcase)

```bash
# Build and run all services
docker-compose up --build

# Access the dashboard at http://localhost:3001
# Access the API at http://localhost:8001/docs
```

## AI Trading Agent

This project includes an autonomous AI Trading Agent that can execute trades based on MTF Ensemble predictions.

### Quick Start with Agent

```bash
# 1. Start all services (includes agent in simulation mode)
docker-compose up -d

# 2. Check agent health
curl http://localhost:8002/health

# 3. View agent status
curl http://localhost:8001/api/v1/agent/status

# 4. View agent logs
docker logs -f ai-trader-agent

# 5. Monitor via web dashboard
open http://localhost:3001
```

### Agent Features

- **Three Trading Modes**: Simulation (backtesting), Paper (MT5 demo), Live (MT5 real)
- **Safety Systems**: Circuit breakers, kill switch, daily loss limits
- **Command Queue Pattern**: API queues commands; agent polls and executes asynchronously
- **Crash Recovery**: Automatic recovery with state persistence
- **Real-time Monitoring**: Performance metrics, win rate, profit factor
- **Health Checks**: HTTP endpoints for container orchestration

### Configuration

Configure the agent via environment variables in `.env`:

```bash
# Agent Mode (simulation, paper, live)
AGENT_MODE=simulation

# Trading Parameters
AGENT_CONFIDENCE_THRESHOLD=0.70
AGENT_CYCLE_INTERVAL=60
AGENT_MAX_POSITION_SIZE=0.1

# Safety Settings
AGENT_MAX_CONSECUTIVE_LOSSES=5
AGENT_MAX_DRAWDOWN_PERCENT=10.0
AGENT_MAX_DAILY_LOSS_PERCENT=5.0

# MT5 Credentials (required for paper/live modes)
AGENT_MT5_LOGIN=
AGENT_MT5_PASSWORD=
AGENT_MT5_SERVER=
```

**IMPORTANT**: MT5 requires Windows. Docker (Linux) only supports simulation mode. For paper/live trading, run the agent on Windows.

### Agent Operations

**Start Agent:**

```bash
curl -X POST http://localhost:8001/api/v1/agent/start \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "simulation",
    "confidence_threshold": 0.70,
    "cycle_interval_seconds": 60,
    "max_position_size": 0.1,
    "use_kelly_sizing": true
  }'
```

**Stop Agent:**

```bash
curl -X POST http://localhost:8001/api/v1/agent/stop \
  -H "Content-Type: application/json" \
  -d '{
    "force": false,
    "close_positions": true
  }'
```

**Get Performance Metrics:**

```bash
curl "http://localhost:8001/api/v1/agent/metrics?period=24h"
```

### Complete Agent Documentation

- **[AI Trading Agent](docs/AI-TRADING-AGENT.md)** - Complete system documentation (~39KB)
  - Architecture overview with ASCII diagrams
  - Execution modes (simulation, paper, live)
  - Configuration options and safety systems
  - Trading cycle workflow
  - MT5 integration details
  - Database schema
  - Deployment instructions

- **[Agent Operations Guide](docs/AGENT-OPERATIONS-GUIDE.md)** - Operations runbook (~22KB)
  - Startup and shutdown procedures
  - Monitoring and logging
  - Troubleshooting common issues
  - Emergency procedures (kill switch, circuit breaker recovery)
  - Maintenance tasks

- **[Agent API Reference](docs/AGENT-API-REFERENCE.md)** - Complete API documentation (~18KB)
  - All 11+ endpoints with request/response examples
  - Schema definitions
  - Error codes and status responses
  - Safety endpoints

- **[Agent Quick Reference](docs/AGENT-QUICK-REFERENCE.md)** - Operator cheat sheet (~11KB)
  - Quick commands
  - Status codes
  - Common operations
  - Emergency procedures

- **[Changelog](docs/CHANGELOG.md)** - Version history (~11KB)
  - v1.0.0 - Initial agent release
  - All phases implemented

See [AI Trading Agent Documentation](docs/AI-TRADING-AGENT.md) for complete details.

### Safety Systems

The agent includes comprehensive safety mechanisms:

| Safety Feature | Default Threshold | Action |
|----------------|-------------------|--------|
| Consecutive Loss Breaker | 5 losses | Pause trading |
| Drawdown Breaker | 10% from peak | Stop agent |
| Daily Loss Limit | 5% OR $5,000 | Kill switch |
| Model Degradation | Win rate < 45% | Pause trading (optional) |

**Emergency Stop (Kill Switch):**

```bash
curl -X POST http://localhost:8001/api/v1/agent/kill-switch \
  -H "Content-Type: application/json" \
  -d '{
    "action": "trigger",
    "reason": "Emergency stop"
  }'
```

### Make Predictions

```python
from src.models.multi_timeframe import MTFEnsemble, MTFEnsembleConfig

# Load trained ensemble (run from backend/ directory)
config = MTFEnsembleConfig.with_sentiment("EURUSD")
ensemble = MTFEnsemble(config=config, model_dir="models/mtf_ensemble")
ensemble.load()

# Predict on recent 5-minute data
prediction = ensemble.predict(df_5min_recent)

print(f"Direction: {'LONG' if prediction.direction == 1 else 'SHORT'}")
print(f"Confidence: {prediction.confidence:.1%}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MTF ENSEMBLE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   5-min Data ──┬──► 1H Model (60% weight) ───────┐              │
│                │                                  │              │
│                ├──► 4H Model (30% weight) ───────┼──► Ensemble  │
│                │                                  │   Prediction │
│                └──► Daily Model (10% weight) ────┘              │
│                     + VIX/EPU Sentiment                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
ai-trader/
├── backend/                 # FastAPI Backend & ML Pipeline
│   ├── configs/            # Model and indicator configurations
│   ├── data/
│   │   ├── forex/          # EUR/USD 5-minute data (2020-2025)
│   │   └── sentiment/      # VIX, EPU, GDELT sentiment data
│   ├── models/
│   │   └── mtf_ensemble/   # Trained production models
│   ├── scripts/            # Training & data scripts
│   ├── src/
│   │   ├── api/            # FastAPI routes & services
│   │   ├── features/       # Technical indicators & sentiment
│   │   ├── models/         # MTF ensemble implementation
│   │   ├── simulation/     # Backtesting engine
│   │   └── trading/        # Risk management
│   └── tests/              # Python tests (735+ tests)
├── frontend/               # React Web Dashboard
│   └── src/
│       ├── components/     # Dashboard, PredictionCard, Charts
│       └── api/            # API client
├── docs/                   # Documentation
└── docker-compose.yml      # Local orchestration
```

## Data

### Price Data
- **Source**: MetaTrader 5
- **Pair**: EUR/USD
- **Resolution**: 5-minute (resampled to 1H, 4H, Daily)
- **Period**: 2020-01-01 to 2025-12-31
- **Records**: 448,586 bars

### Sentiment Data
- **VIX**: CBOE Volatility Index (daily) from FRED
- **EPU**: Economic Policy Uncertainty Index (daily) from FRED
- **GDELT**: Hourly news sentiment (available but not used in production)

## Key Findings

### Sentiment Integration Research

| Configuration | Total Pips | Profit Factor |
|---------------|------------|---------------|
| Baseline (No Sentiment) | +7,596 | 2.12 |
| **EPU Daily-Only (Optimal)** | **+7,987** | **2.22** |
| GDELT All Timeframes | +7,273 | 2.09 |

**Resolution Matching Rule**: Sentiment data resolution must match or be finer than the trading timeframe. Daily EPU/VIX works for Daily model; adding it to 1H/4H models degrades performance.

### What Works
- Multi-timeframe ensemble (60/30/10 weights)
- VIX + EPU sentiment on Daily model only
- Cross-timeframe trend alignment features
- Triple barrier labeling for realistic simulation

### What Doesn't Work
- Sentiment on intraday models (resolution mismatch)
- GDELT hourly sentiment (too noisy despite proper resolution)
- Equal timeframe weights

## Configuration

### Training Options

```bash
cd backend

# With sentiment (recommended)
python scripts/train_mtf_ensemble.py --sentiment

# Custom sentiment configuration
python scripts/train_mtf_ensemble.py --sentiment-tf "D" --sentiment-source epu

# Custom weights
python scripts/train_mtf_ensemble.py --weights "0.6,0.3,0.1"
```

### Triple Barrier Parameters

| Timeframe | Take Profit | Stop Loss | Max Holding |
|-----------|-------------|-----------|-------------|
| 1H | 25 pips | 15 pips | 12 bars |
| 4H | 50 pips | 25 pips | 18 bars |
| Daily | 150 pips | 75 pips | 15 bars |

## Documentation

- [Current State of the Art](docs/15-current-state-of-the-art.md) - Comprehensive system documentation
- [Architecture Overview](docs/01-architecture-overview.md)
- [Technical Analysis Model Design](docs/02-technical-analysis-model-design.md)
- [Technical Indicators Configuration](docs/03-technical-indicators-configuration.md)
- [Confidence & Uncertainty System](docs/04-confidence-uncertainty-system.md)
- [Trading Robot Design](docs/05-trading-robot-design.md)
- [MTF Ensemble Implementation](docs/08-multi-timeframe-ensemble-implementation.md)
- [Sentiment Analysis Results](docs/13-sentiment-analysis-test-results.md)

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| ML Models | XGBoost |
| Indicators | pandas-ta |
| Data | pandas, numpy |
| Sentiment | FRED API, Google BigQuery |

## Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Win Rate | > 55% | 57.8% |
| Profit Factor | > 2.0 | 2.22 |
| High-Conf Accuracy | > 65% | 72.14% (1H @ ≥60%) |

## License

This project is for educational and research purposes.

## Acknowledgments

Developed with assistance from Claude (Anthropic).
