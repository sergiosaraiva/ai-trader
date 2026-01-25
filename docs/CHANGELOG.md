# Changelog

All notable changes to the AI Trader project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-22 - AI Trading Agent Release

This major release introduces an autonomous AI Trading Agent with comprehensive safety systems, real-time monitoring, and production-ready deployment capabilities.

### Added

#### AI Trading Agent (Phase 1-8)

- **Agent Module** (`backend/src/agent/`)
  - `AgentRunner`: Main orchestrator managing agent lifecycle
  - `CommandHandler`: Polls and processes commands from database queue
  - `StateManager`: Persists agent state for crash recovery
  - `TradingCycle`: Executes prediction → signal → trade flow
  - `BrokerManager`: MT5 connection and order management (Windows only)
  - `TradeExecutor`: Position sizing and trade execution
  - `SafetyManager`: Circuit breakers and kill switch implementation
  - `config.py`: Configuration management from environment variables
  - `safety_config.py`: Safety system configuration
  - `main.py`: Agent entry point with health server

#### Safety Systems

- **Circuit Breakers**:
  - Consecutive Loss Breaker (5 losses → pause)
  - Drawdown Breaker (10% → stop)
  - Daily Loss Limit (5% OR $5,000 → kill switch)
  - Model Degradation Breaker (win rate < 45% → pause, optional)
- **Kill Switch**: Emergency stop with authorization code reset
- **Trade Limits**: Max 50 trades/day, 20 trades/hour
- **Connection Monitoring**: Auto-reconnect on MT5 disconnection

#### Backend API for Agent Control

- **Agent Control Endpoints** (`backend/src/api/routes/agent.py`):
  - `POST /api/v1/agent/start`: Start agent with configuration
  - `POST /api/v1/agent/stop`: Stop agent (graceful or force)
  - `POST /api/v1/agent/pause`: Pause trading
  - `POST /api/v1/agent/resume`: Resume trading
  - `PUT /api/v1/agent/config`: Update configuration while running
  - `POST /api/v1/agent/kill-switch`: Trigger or reset kill switch

- **Status & Metrics Endpoints**:
  - `GET /api/v1/agent/status`: Current agent status
  - `GET /api/v1/agent/metrics`: Performance metrics (all, 24h, 7d, 30d)
  - `GET /api/v1/agent/commands`: List commands
  - `GET /api/v1/agent/commands/{id}`: Get command status

- **Safety Endpoints**:
  - `GET /api/v1/agent/safety`: Safety system status
  - `GET /api/v1/agent/safety/events`: Safety event audit trail
  - `POST /api/v1/agent/safety/kill-switch/reset-code`: Generate reset code
  - `POST /api/v1/agent/safety/circuit-breakers/reset`: Reset circuit breaker

#### Database Migration (PostgreSQL)

- **Migrated from SQLite to PostgreSQL** for production readiness
- **New Tables**:
  - `agent_state`: Current agent operational state
  - `agent_commands`: Command queue for agent operations
  - `circuit_breaker_events`: Audit trail for safety system activations
- **Migration Script**: `backend/scripts/migrate_to_postgres.py`

#### Docker Deployment

- **Agent Container** (`backend/Dockerfile.agent`):
  - Dedicated container for autonomous agent
  - Health check endpoint on port 8002
  - Automatic restart on crash
  - Log volume mounting
  - Resource limits (2G RAM, 1 CPU)

- **Updated docker-compose.yml**:
  - PostgreSQL service with persistent volume
  - Agent service with dependencies on postgres and backend
  - Health checks for all services
  - Network isolation

#### Frontend Updates

- **Agent Control Panel** (planned for future release):
  - Start/stop/pause/resume controls
  - Real-time status display
  - Performance metrics dashboard
  - Safety event monitoring

#### Documentation

- **[AI-TRADING-AGENT.md](AI-TRADING-AGENT.md)**: Comprehensive agent documentation
  - Architecture overview with diagrams
  - Quick start guide
  - Configuration reference
  - API reference
  - Safety systems explanation
  - Monitoring guide
  - Troubleshooting FAQ

- **[AGENT-OPERATIONS-GUIDE.md](AGENT-OPERATIONS-GUIDE.md)**: Operations runbook
  - Pre-flight checks
  - Start/stop procedures
  - Health monitoring
  - Incident response procedures
  - Maintenance tasks
  - Recovery procedures

- **[AGENT-API-REFERENCE.md](AGENT-API-REFERENCE.md)**: Complete API documentation
  - All endpoint specifications
  - Request/response examples
  - Error codes
  - Rate limiting guidelines

- **Updated [README.md](../README.md)**: Added agent quick start section

### Changed

- **Database**: Migrated from SQLite to PostgreSQL for better concurrency and production reliability
- **docker-compose.yml**: Added agent service and PostgreSQL database
- **Backend Requirements**: Added PostgreSQL driver (`psycopg2-binary`)
- **Environment Variables**: Added agent configuration variables (see `.env.example`)

### Security

- **Multiple Safety Layers**:
  - Circuit breakers prevent runaway losses
  - Kill switch requires authorization code to reset
  - Daily loss limits protect capital
  - MT5 disconnection monitoring
  - Trade frequency limits

- **Command Queue Pattern**:
  - API cannot directly control agent (prevents malicious commands)
  - Agent polls commands from database
  - All commands logged for audit trail

- **State Persistence**:
  - Agent state saved after each cycle
  - Enables crash recovery
  - Prevents data loss

### Fixed

- N/A (new feature release)

### Known Limitations

- **MT5 Windows-Only**: MetaTrader 5 only works on Windows. Docker deployment (Linux) only supports simulation mode. For paper/live trading:
  - Run agent on Windows host
  - Use WSL2 with Windows MT5 access
  - Deploy to Windows-based cloud service

- **Manual Position Reconciliation**: After crash, positions may need manual reconciliation between agent database and MT5 terminal

- **No Authentication**: Agent API endpoints do not require authentication. Add authentication in production deployments.

## [1.0.0] - 2024-01-08 - Initial Release

### Added

- **Multi-Timeframe Ensemble Model**:
  - 1H, 4H, and Daily XGBoost models
  - Stacking meta-learner for dynamic combination
  - 115+ technical features per timeframe
  - Sentiment integration (VIX + EPU on Daily model)

- **Web Showcase**:
  - React frontend dashboard
  - FastAPI backend API
  - Docker deployment
  - Live predictions and performance tracking

- **Backtesting System**:
  - Triple barrier labeling
  - Realistic TP/SL simulation
  - Walk-forward optimization
  - Confidence threshold optimization

- **Data Pipeline**:
  - MT5 data download scripts
  - FRED sentiment data integration
  - Feature engineering pipeline
  - Data validation

- **Documentation**:
  - Architecture overview
  - Model design documentation
  - Sentiment analysis results
  - Deployment guides

### Performance

- **Backtest Results** (simulation, 55% confidence threshold):
  - Total Profit: +7,987 pips
  - Win Rate: 57.8%
  - Profit Factor: 2.22
  - Total Trades: 1,103

- **Model Accuracy**:
  - 1H Model: 67.07% validation accuracy (72.14% high-confidence)
  - 4H Model: 65.43% validation accuracy (71.12% high-confidence)
  - Daily Model: 61.54% validation accuracy (64.21% high-confidence)

- **Walk-Forward Optimization**:
  - 7 rolling windows tested
  - 100% consistency (all windows profitable)
  - +18,136 total pips across all windows

---

## Upgrade Guide

### From 1.0.0 to 2.0.0

#### 1. Database Migration

The project now uses PostgreSQL instead of SQLite.

**Steps:**

```bash
# 1. Backup existing SQLite database
cp backend/data/db/trading.db backend/data/db/trading.db.backup

# 2. Update docker-compose.yml
docker-compose pull postgres

# 3. Start PostgreSQL
docker-compose up -d postgres

# 4. Run migration script
docker exec ai-trader-backend python scripts/migrate_to_postgres.py

# 5. Verify migration
docker exec -it ai-trader-postgres psql -U trader trading -c "SELECT COUNT(*) FROM trades;"
```

#### 2. Environment Variables

Add new agent-related environment variables to `.env`:

```bash
# Agent Configuration
AGENT_MODE=simulation
AGENT_CONFIDENCE_THRESHOLD=0.70
AGENT_CYCLE_INTERVAL=60
AGENT_MAX_POSITION_SIZE=0.1
AGENT_USE_KELLY_SIZING=true
AGENT_HEALTH_PORT=8002
AGENT_INITIAL_CAPITAL=100000.0

# Safety Settings
AGENT_MAX_CONSECUTIVE_LOSSES=5
AGENT_MAX_DRAWDOWN_PERCENT=10.0
AGENT_MAX_DAILY_LOSS_PERCENT=5.0
AGENT_ENABLE_MODEL_DEGRADATION=false

# MT5 Credentials (optional, only for paper/live)
AGENT_MT5_LOGIN=
AGENT_MT5_PASSWORD=
AGENT_MT5_SERVER=

# Live Trading Confirmation (MUST be true for live mode)
AGENT_LIVE_TRADING_CONFIRMED=false
```

#### 3. Docker Compose

Update `docker-compose.yml` to include new services:

```bash
# Stop all services
docker-compose down

# Pull updated images
docker-compose pull

# Rebuild containers
docker-compose build --no-cache

# Start all services
docker-compose up -d
```

#### 4. Verify Agent

```bash
# Check agent health
curl http://localhost:8002/health

# Check agent status
curl http://localhost:8001/api/v1/agent/status
```

#### Breaking Changes

- **Database**: SQLite is no longer supported. Migrate to PostgreSQL.
- **Docker**: New agent service requires updated docker-compose.yml
- **Environment**: New environment variables required for agent

#### Non-Breaking Changes

- **API**: All existing API endpoints remain unchanged
- **Models**: Existing trained models are compatible
- **Frontend**: No changes required

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 2.0.0 | 2024-01-22 | AI Trading Agent release with safety systems |
| 1.0.0 | 2024-01-08 | Initial release with MTF Ensemble and web showcase |

---

## Roadmap

### Version 2.1.0 (Q1 2024)

- [ ] Frontend agent control panel
- [ ] Webhook notifications for events
- [ ] Advanced position sizing strategies
- [ ] Multi-pair support (GBPUSD, USDJPY, etc.)

### Version 2.2.0 (Q2 2024)

- [ ] Real-time market regime detection
- [ ] Dynamic confidence threshold adjustment
- [ ] Machine learning for optimal circuit breaker thresholds
- [ ] Trade analytics dashboard

### Version 3.0.0 (Q3 2024)

- [ ] Multi-agent orchestration (multiple strategies)
- [ ] Portfolio management across pairs
- [ ] Advanced risk management (VaR, CVaR)
- [ ] Automated model retraining pipeline

---

**Maintained by**: Sergio Saraiva

**Support**: For issues and questions, open a GitHub issue or refer to the documentation.

**License**: Educational and research purposes only. Not financial advice.
