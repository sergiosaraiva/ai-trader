# AI Trading Agent

**Version:** 1.0.0
**Last Updated:** 2026-01-22
**Status:** Production-Ready

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Execution Modes](#execution-modes)
- [Configuration](#configuration)
- [Safety Systems](#safety-systems)
- [Trading Cycle](#trading-cycle)
- [API Integration](#api-integration)
- [MT5 Integration](#mt5-integration)
- [Database Schema](#database-schema)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

---

## Overview

The AI Trading Agent is an autonomous trading system that executes trades based on predictions from the Multi-Timeframe (MTF) Ensemble model. It operates continuously, monitoring market conditions, generating predictions, and executing trades while maintaining comprehensive safety controls.

### Key Features

- **Autonomous Operation**: Runs continuously with configurable cycle intervals (default: 60s)
- **Three Execution Modes**: Simulation (backtesting), Paper (MT5 demo), Live (MT5 real)
- **Command Queue Pattern**: API queues commands; agent polls and executes asynchronously
- **Safety Systems**: Circuit breakers, kill switch, daily loss limits
- **Crash Recovery**: Automatic recovery with state persistence
- **Real-time Monitoring**: Performance metrics, win rate, profit factor
- **Health Checks**: HTTP endpoints for container orchestration
- **Database Audit Trail**: Complete history of commands, states, and safety events

### Production Status

- **Status**: Production-ready (v1.0.0)
- **Container**: `ai-trader-agent` (Docker)
- **Health Port**: 8002
- **Dependencies**: PostgreSQL, Backend API, MT5 (paper/live modes only)
- **Resource Limits**: 2GB RAM, 1 CPU core

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          AI TRADING AGENT ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │                        AgentRunner                             │         │
│  │  ┌──────────────────────────────────────────────────────────┐ │         │
│  │  │ Main Loop (every cycle_interval_seconds)                 │ │         │
│  │  │                                                           │ │         │
│  │  │  1. Check for commands (CommandHandler)                  │ │         │
│  │  │  2. Check broker connection health                       │ │         │
│  │  │  3. Check safety status (SafetyManager)                  │ │         │
│  │  │  4. Execute trading cycle if safe                        │ │         │
│  │  │  5. Update agent state (StateManager)                    │ │         │
│  │  │                                                           │ │         │
│  │  └──────────────────────────────────────────────────────────┘ │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │                      TradingCycle                              │         │
│  │  ┌──────────────────────────────────────────────────────────┐ │         │
│  │  │ Trading Workflow                                          │ │         │
│  │  │                                                           │ │         │
│  │  │  1. Check model service is loaded                        │ │         │
│  │  │  2. Generate prediction (ModelService)                   │ │         │
│  │  │  3. Store prediction in database                         │ │         │
│  │  │  4. Check safety status                                  │ │         │
│  │  │  5. Check confidence threshold                           │ │         │
│  │  │  6. Check open positions (TradeExecutor)                 │ │         │
│  │  │  7. Generate & execute signal                            │ │         │
│  │  │     - Simulation: Log signal                             │ │         │
│  │  │     - Paper/Live: Execute via MT5 (BrokerManager)        │ │         │
│  │  │                                                           │ │         │
│  │  └──────────────────────────────────────────────────────────┘ │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │                      SafetyManager                             │         │
│  │  ┌──────────────────────────────────────────────────────────┐ │         │
│  │  │ Safety Checks                                             │ │         │
│  │  │                                                           │ │         │
│  │  │  Kill Switch (highest priority)                          │ │         │
│  │  │    - Daily loss limit: 5% OR $5,000                      │ │         │
│  │  │    - Daily trade limit                                   │ │         │
│  │  │    - Broker disconnection                                │ │         │
│  │  │                                                           │ │         │
│  │  │  Circuit Breakers                                        │ │         │
│  │  │    - Consecutive Loss (5 losses → pause)                 │ │         │
│  │  │    - Drawdown (10% from peak → halt)                     │ │         │
│  │  │    - Model Degradation (win rate < 45% → pause)         │ │         │
│  │  │                                                           │ │         │
│  │  └──────────────────────────────────────────────────────────┘ │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Diagram

```
┌─────────────┐         ┌──────────────┐         ┌──────────────┐
│   Backend   │◄────────│  AgentRunner │────────►│  PostgreSQL  │
│     API     │ Commands│              │  State  │   Database   │
└─────────────┘         └──────┬───────┘         └──────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
            ┌───────▼───────┐     ┌──────▼──────┐
            │ TradingCycle  │     │SafetyManager│
            └───────┬───────┘     └─────────────┘
                    │
         ┌──────────┼──────────┐
         │          │          │
    ┌────▼────┐ ┌──▼──────┐ ┌─▼───────────┐
    │  Model  │ │ Broker  │ │    Trade    │
    │ Service │ │ Manager │ │  Executor   │
    └─────────┘ └────┬────┘ └─────────────┘
                     │
                ┌────▼────┐
                │   MT5   │
                │ Broker  │
                └─────────┘
```

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **AgentRunner** | `src/agent/runner.py` | Main orchestrator; manages lifecycle, commands, main loop |
| **TradingCycle** | `src/agent/trading_cycle.py` | Executes predict → signal → trade workflow |
| **SafetyManager** | `src/agent/safety_manager.py` | Coordinates circuit breakers and kill switch |
| **BrokerManager** | `src/agent/broker_manager.py` | Manages MT5 connection with auto-reconnect |
| **TradeExecutor** | `src/agent/trade_executor.py` | Executes trades and manages positions |
| **CommandHandler** | `src/agent/command_handler.py` | Polls and processes commands from database |
| **StateManager** | `src/agent/state_manager.py` | Persists agent state for crash recovery |
| **PositionTracker** | `src/agent/position_tracker.py` | Tracks open positions and P&L |

---

## Execution Modes

The agent supports three execution modes, configured via `AGENT_MODE` environment variable.

### 1. Simulation Mode (Default)

**Use Case**: Backtesting, development, testing without broker connection.

```bash
AGENT_MODE=simulation
```

**Behavior**:
- No MT5 broker connection required
- Predictions are made but no real trades executed
- Signals are logged for analysis
- Ideal for testing configuration changes
- No capital at risk

**Pros**:
- Safe for testing
- Fast execution
- No broker dependencies
- Works in Docker/Linux

**Cons**:
- No real market execution feedback
- Cannot test broker connectivity issues

### 2. Paper Trading Mode

**Use Case**: Testing with real market data but demo account (no real money).

```bash
AGENT_MODE=paper
AGENT_MT5_LOGIN=12345678
AGENT_MT5_PASSWORD=your_password
AGENT_MT5_SERVER=ICMarkets-Demo
```

**Behavior**:
- Connects to MT5 demo account
- Executes real trades on demo account
- Real market prices and spreads
- Realistic slippage and execution delays
- Full broker integration testing

**Requirements**:
- MT5 demo account credentials
- MT5 installed on Windows (not supported in Docker/Linux)
- Network connectivity to broker

**Pros**:
- Realistic market simulation
- Tests full execution pipeline
- No capital at risk
- Real broker feedback

**Cons**:
- Requires Windows or WSL2 with MT5
- Demo account conditions may differ from live
- Network dependency

### 3. Live Trading Mode

**Use Case**: Production trading with real money.

```bash
AGENT_MODE=live
AGENT_MT5_LOGIN=87654321
AGENT_MT5_PASSWORD=your_password
AGENT_MT5_SERVER=ICMarkets-Live01
AGENT_LIVE_TRADING_CONFIRMED=true  # Required safety flag
```

**Behavior**:
- Connects to MT5 live account
- Executes real trades with real money
- All safety systems active
- Full audit trail in database

**Requirements**:
- MT5 live account with sufficient balance
- `AGENT_LIVE_TRADING_CONFIRMED=true` must be set
- All safety systems configured
- Monitoring in place

**Pros**:
- Real trading with real profits/losses
- Full production deployment

**Cons**:
- **CAPITAL AT RISK**
- Requires careful configuration
- Mandatory safety system validation
- Requires Windows for MT5

**⚠️ IMPORTANT**: Live mode requires explicit confirmation via `AGENT_LIVE_TRADING_CONFIRMED=true`. This is a safety measure to prevent accidental live trading.

---

## Configuration

The agent is configured entirely via environment variables with validation and defaults.

### Core Configuration

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `AGENT_MODE` | `simulation` | `simulation`, `paper`, `live` | Execution mode |
| `AGENT_SYMBOL` | `EURUSD` | Any forex pair | Trading symbol |
| `AGENT_CONFIDENCE_THRESHOLD` | `0.70` | 0.50-0.95 | Minimum confidence to trade |
| `AGENT_CYCLE_INTERVAL` | `60` | 1-3600 | Seconds between cycles |
| `AGENT_MAX_POSITION_SIZE` | `0.1` | >0 | Maximum lot size |
| `AGENT_USE_KELLY_SIZING` | `true` | `true`, `false` | Use Kelly Criterion for sizing |
| `AGENT_INITIAL_CAPITAL` | `100000.0` | >0 | Starting capital (simulation/paper) |
| `AGENT_HEALTH_PORT` | `8002` | 1024-65535 | Health check HTTP port |

### MT5 Broker Configuration

Required for `paper` and `live` modes only:

| Variable | Required | Description |
|----------|----------|-------------|
| `AGENT_MT5_LOGIN` | Yes (paper/live) | MT5 account login number |
| `AGENT_MT5_PASSWORD` | Yes (paper/live) | MT5 account password |
| `AGENT_MT5_SERVER` | Yes (paper/live) | MT5 server name (e.g., `ICMarkets-Demo`) |

### Safety Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MAX_CONSECUTIVE_LOSSES` | `5` | Max losses before consecutive loss breaker triggers |
| `AGENT_MAX_DRAWDOWN_PERCENT` | `10.0` | Max drawdown % from peak before halt |
| `AGENT_MAX_DAILY_LOSS_PERCENT` | `5.0` | Max daily loss % before kill switch |
| `AGENT_ENABLE_MODEL_DEGRADATION` | `false` | Enable model degradation monitoring |

### Database Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection string |

**Format**: `postgresql://user:password@host:port/database`

**Example**: `postgresql://aitrader:secret@postgres:5432/trading`

### Complete Example (.env)

```bash
# ============================================
# AI Trading Agent Configuration
# ============================================

# Execution Mode
AGENT_MODE=simulation
AGENT_SYMBOL=EURUSD

# Trading Parameters
AGENT_CONFIDENCE_THRESHOLD=0.70
AGENT_CYCLE_INTERVAL=60
AGENT_MAX_POSITION_SIZE=0.1
AGENT_USE_KELLY_SIZING=true
AGENT_INITIAL_CAPITAL=100000.0

# Safety Settings
AGENT_MAX_CONSECUTIVE_LOSSES=5
AGENT_MAX_DRAWDOWN_PERCENT=10.0
AGENT_MAX_DAILY_LOSS_PERCENT=5.0
AGENT_ENABLE_MODEL_DEGRADATION=false

# MT5 Credentials (required for paper/live modes)
# NOTE: MT5 requires Windows. Docker (Linux) only supports simulation mode.
AGENT_MT5_LOGIN=
AGENT_MT5_PASSWORD=
AGENT_MT5_SERVER=

# Live Trading Confirmation (REQUIRED for live mode)
AGENT_LIVE_TRADING_CONFIRMED=false

# Database
DATABASE_URL=postgresql://aitrader:yourpassword@postgres:5432/trading

# Backend API
BACKEND_URL=http://backend:8001

# Health Check
AGENT_HEALTH_PORT=8002
```

---

## Safety Systems

The agent includes multiple layers of safety mechanisms to protect capital from catastrophic losses.

### Safety Hierarchy

```
┌─────────────────────────────────────────────────┐
│  Kill Switch (Highest Priority)                 │
│  └─> Halts ALL trading immediately              │
│      - Daily loss limit exceeded                │
│      - Daily trade limit exceeded               │
│      - Broker disconnection timeout             │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  Circuit Breakers (Medium Priority)             │
│  └─> Pause or reduce trading activity           │
│      - Consecutive losses                       │
│      - Drawdown from peak                       │
│      - Model degradation (optional)             │
└─────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  Confidence Threshold (Basic Filter)            │
│  └─> Only trade signals above threshold         │
└─────────────────────────────────────────────────┘
```

### 1. Kill Switch

**Purpose**: Emergency stop for catastrophic conditions.

**Triggers**:

| Trigger | Default Threshold | Action |
|---------|-------------------|--------|
| Daily Loss % | 5% of starting equity | Halt all trading |
| Daily Loss $ | $5,000 absolute | Halt all trading |
| Daily Trades | 100 trades | Halt all trading |
| Broker Disconnection | 300 seconds | Halt all trading |
| Manual Trigger | User command | Halt all trading |

**Behavior**:
- Immediately stops agent main loop
- Closes all open positions
- Blocks all new trades
- Requires manual reset with authorization code

**Reset Process**:

1. Get reset authorization code:
   ```bash
   curl http://localhost:8001/api/v1/agent/safety/kill-switch/reset-code
   ```

2. Reset kill switch:
   ```bash
   curl -X POST http://localhost:8001/api/v1/agent/kill-switch \
     -H "Content-Type: application/json" \
     -d '{"action": "reset"}'
   ```

3. Agent must be manually restarted after reset

**Database Audit**: All kill switch events logged to `circuit_breaker_events` table.

### 2. Circuit Breakers

**Purpose**: Gradual risk reduction before kill switch activation.

#### A. Consecutive Loss Breaker

**Trigger**: 5 consecutive losing trades (default).

**States**:
- **Active**: Normal trading
- **Reduced**: After 3 losses, reduce position size by 50%
- **Halted**: After 5 losses, pause trading

**Recovery**: Automatic after successful winning trade.

**Configuration**:
```bash
AGENT_MAX_CONSECUTIVE_LOSSES=5
```

#### B. Drawdown Breaker

**Trigger**: 10% drawdown from peak equity (default).

**States**:
- **Active**: Drawdown < 5%
- **Warning**: Drawdown 5-10%, reduce size by 25%
- **Halted**: Drawdown > 10%, stop trading

**Recovery**: Automatic when drawdown falls below 5%.

**Configuration**:
```bash
AGENT_MAX_DRAWDOWN_PERCENT=10.0
```

#### C. Model Degradation Breaker (Optional)

**Trigger**: Win rate falls below 45% over last 20 trades.

**States**:
- **Active**: Win rate >= 50%
- **Reduced**: Win rate 45-50%, increase min confidence to 75%
- **Halted**: Win rate < 45%, pause trading

**Recovery**: Automatic when win rate improves above 50%.

**Configuration**:
```bash
AGENT_ENABLE_MODEL_DEGRADATION=true  # Disabled by default
```

### 3. Risk Limits

**Daily Loss Limit**:
- Percentage: 5% of starting daily equity
- Absolute: $5,000 USD equivalent
- Whichever is reached first triggers kill switch

**Position Size Limits**:
- Maximum: `AGENT_MAX_POSITION_SIZE` lots
- Reduced by safety multiplier from circuit breakers
- Kelly Criterion scaling (if enabled)

**Daily Trade Limit**:
- Default: 100 trades per day
- Prevents runaway trading loops
- Resets at UTC midnight

### Safety Status API

Check comprehensive safety status:

```bash
curl http://localhost:8001/api/v1/agent/safety
```

Response:
```json
{
  "is_safe_to_trade": true,
  "circuit_breakers": {
    "overall_state": "active",
    "active_breakers": [],
    "can_trade": true
  },
  "kill_switch": {
    "is_active": false,
    "reason": null,
    "triggered_at": null
  },
  "daily_metrics": {
    "trades": 12,
    "loss_pct": 1.2,
    "loss_amount": 1200.0
  },
  "account_metrics": {
    "current_equity": 101200.0,
    "peak_equity": 103500.0,
    "drawdown_pct": 2.2
  }
}
```

---

## Trading Cycle

The trading cycle executes every `cycle_interval_seconds` (default: 60 seconds).

### Cycle Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                     Trading Cycle                            │
└─────────────────────────────────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            │  1. Check Model Service   │
            │     - Is model loaded?    │
            └─────────────┬─────────────┘
                          │ Yes
            ┌─────────────▼─────────────┐
            │  2. Generate Prediction   │
            │     - From ModelService   │
            │     - 115+ features       │
            │     - MTF ensemble        │
            └─────────────┬─────────────┘
                          │
            ┌─────────────▼─────────────┐
            │  3. Store in Database     │
            │     - predictions table   │
            └─────────────┬─────────────┘
                          │
            ┌─────────────▼─────────────┐
            │  4. Check Safety Status   │
            │     - Kill switch?        │
            │     - Circuit breakers?   │
            │     - If unsafe → HOLD    │
            └─────────────┬─────────────┘
                          │ Safe
            ┌─────────────▼─────────────┐
            │  5. Check Confidence      │
            │     - Above threshold?    │
            │     - If low → HOLD       │
            └─────────────┬─────────────┘
                          │ High
            ┌─────────────▼─────────────┐
            │  6. Check Open Positions  │
            │     - Close expired       │
            │     - Update trailing SL  │
            └─────────────┬─────────────┘
                          │
            ┌─────────────▼─────────────┐
            │  7. Execute Signal        │
            │                           │
            │  Simulation: Log signal   │
            │  Paper/Live: Execute MT5  │
            └───────────────────────────┘
```

### Cycle Duration

Typical cycle duration: 100-500ms

**Breakdown**:
- Prediction generation: 50-200ms
- Database operations: 20-50ms
- Safety checks: 10-20ms
- Trade execution: 20-200ms (if executed)

**Monitoring**: Cycle duration logged in `agent_state` table.

### Cycle States

| State | Description |
|-------|-------------|
| **Prediction Made** | Successfully generated prediction |
| **Signal Generated** | Confidence threshold met, signal created |
| **Trade Executed** | Signal executed on broker (paper/live) |
| **Trade Failed** | Execution failed (logged for analysis) |
| **Hold** | No action taken (low confidence or unsafe) |
| **Error** | Cycle failed (model error, database error, etc.) |

---

## API Integration

The agent is controlled via REST API using a command queue pattern.

### Command Queue Pattern

```
┌──────────┐                ┌──────────┐                ┌──────────┐
│   API    │   Queue         │ Database │  Poll          │  Agent   │
│  Client  │─────────►───────│  Table   │───────►────────│  Runner  │
└──────────┘   Command       └──────────┘  Execute       └──────────┘
                                                │
                                                │ Result
                                                ▼
                                          ┌──────────┐
                                          │ Database │
                                          │  Update  │
                                          └──────────┘
```

**Benefits**:
- Decouples API from agent execution
- Async operation (no blocking)
- Command history and audit trail
- Crash recovery (commands in database)
- Multiple API instances supported

See [AGENT-API-REFERENCE.md](AGENT-API-REFERENCE.md) for complete API documentation.

---

## MT5 Integration

MetaTrader 5 integration for paper and live trading modes.

### Requirements

- **Windows OS**: MT5 is Windows-only (Wine not supported)
- **MT5 Terminal**: Installed and configured
- **Account**: Demo (paper) or Live account
- **Python Package**: `MetaTrader5` (included in `requirements-api.txt`)

### Docker Limitation

**IMPORTANT**: MT5 does not work in Docker (Linux containers). The agent container only supports **simulation mode** in Docker.

**Options for Paper/Live Trading**:

1. **Run agent on Windows host** (not in Docker)
2. **Use WSL2 with Windows MT5** (requires special setup)
3. **Deploy agent on Windows server** (outside Docker)

### MT5 Configuration

**Demo Account** (Paper Trading):

```bash
AGENT_MODE=paper
AGENT_MT5_LOGIN=12345678          # Demo account number
AGENT_MT5_PASSWORD=YourPassword
AGENT_MT5_SERVER=ICMarkets-Demo   # Demo server
```

**Live Account** (Real Trading):

```bash
AGENT_MODE=live
AGENT_MT5_LOGIN=87654321          # Live account number
AGENT_MT5_PASSWORD=YourPassword
AGENT_MT5_SERVER=ICMarkets-Live01 # Live server
AGENT_LIVE_TRADING_CONFIRMED=true # Required safety flag
```

### Connection Management

The `BrokerManager` handles MT5 connection lifecycle:

**Features**:
- Automatic connection on agent start
- Health monitoring every cycle
- Automatic reconnection on connection loss
- Max 5 reconnection attempts with exponential backoff
- Graceful disconnection on agent stop

**Connection States**:
- `DISCONNECTED`: Not connected
- `CONNECTING`: Connection in progress
- `CONNECTED`: Active connection
- `RECONNECTING`: Attempting to reconnect
- `ERROR`: Connection failed

**Health Checks**:
- Executed every cycle
- Queries account info to verify connectivity
- Triggers reconnection if health check fails

### Trade Execution

The `TradeExecutor` manages trade execution:

**Order Flow**:

1. **Signal Generation**: TradingCycle generates trading signal
2. **Position Sizing**: Kelly Criterion or fixed size
3. **Order Creation**: MT5 market order with SL/TP
4. **Order Execution**: Submit to MT5 broker
5. **Confirmation**: Wait for MT5 response
6. **Database Logging**: Store trade in `trades` table
7. **Position Tracking**: Add to open positions

**Order Parameters**:
- **Type**: Market order (immediate execution)
- **Symbol**: From agent config (default: EURUSD)
- **Volume**: Calculated by position sizing
- **Stop Loss**: 2% default (configurable)
- **Take Profit**: 4% default (2:1 R:R)
- **Magic Number**: Unique identifier for agent trades
- **Comment**: Agent cycle number and prediction ID

**Position Management**:
- Open positions tracked in memory and database
- Checked every cycle for:
  - Stop loss hit
  - Take profit hit
  - Maximum holding time exceeded
  - Manual close via API
- Automatic closure on agent stop (configurable)

---

## Database Schema

The agent uses PostgreSQL for state persistence and audit trails.

### Tables

#### 1. `agent_commands`

Stores queued commands from API.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PRIMARY KEY | Command ID |
| `command` | VARCHAR | Command name (start, stop, pause, resume, kill, update_config) |
| `payload` | JSONB | Command parameters |
| `status` | VARCHAR | pending, processing, completed, failed |
| `created_at` | TIMESTAMP | When command was queued |
| `processed_at` | TIMESTAMP | When command was processed |
| `result` | JSONB | Execution result |
| `error_message` | TEXT | Error if failed |

**Indexes**:
- `idx_agent_commands_status` on `(status, created_at)`

#### 2. `agent_state`

Stores current agent state for crash recovery.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PRIMARY KEY | State ID |
| `status` | VARCHAR | stopped, starting, running, paused, stopping, error |
| `mode` | VARCHAR | simulation, paper, live |
| `cycle_count` | INTEGER | Number of cycles executed |
| `last_cycle_at` | TIMESTAMP | Timestamp of last cycle |
| `account_equity` | FLOAT | Current account equity |
| `open_positions` | INTEGER | Number of open positions |
| `circuit_breaker_state` | VARCHAR | Circuit breaker status |
| `kill_switch_active` | BOOLEAN | Whether kill switch is triggered |
| `error_message` | TEXT | Error message if status=error |
| `config` | JSONB | Current agent configuration |
| `last_prediction` | JSONB | Last prediction made |
| `started_at` | TIMESTAMP | When agent started |
| `stopped_at` | TIMESTAMP | When agent stopped |
| `updated_at` | TIMESTAMP | Last state update |

**Indexes**:
- `idx_agent_state_updated` on `(updated_at DESC)`

#### 3. `predictions`

Stores all predictions generated by agent.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PRIMARY KEY | Prediction ID |
| `timestamp` | TIMESTAMP | Prediction timestamp |
| `symbol` | VARCHAR | Trading symbol |
| `direction` | VARCHAR | long or short |
| `confidence` | FLOAT | Ensemble confidence (0-1) |
| `prob_up` | FLOAT | Probability of up move |
| `prob_down` | FLOAT | Probability of down move |
| `pred_1h` | VARCHAR | 1H model prediction |
| `conf_1h` | FLOAT | 1H model confidence |
| `pred_4h` | VARCHAR | 4H model prediction |
| `conf_4h` | FLOAT | 4H model confidence |
| `pred_d` | VARCHAR | Daily model prediction |
| `conf_d` | FLOAT | Daily model confidence |
| `agreement_count` | INTEGER | Number of models agreeing |
| `agreement_score` | FLOAT | Agreement score (0-1) |
| `market_regime` | VARCHAR | Market regime classification |
| `should_trade` | BOOLEAN | Whether prediction met threshold |
| `used_by_agent` | BOOLEAN | Whether prediction was used by agent |
| `agent_cycle_number` | INTEGER | Agent cycle number (if used) |

**Indexes**:
- `idx_predictions_timestamp` on `(timestamp DESC)`
- `idx_predictions_agent` on `(used_by_agent, agent_cycle_number)`

#### 4. `trades`

Stores all executed trades.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PRIMARY KEY | Trade ID |
| `trade_id` | VARCHAR UNIQUE | Unique trade identifier |
| `mt5_ticket` | BIGINT | MT5 order ticket (null for simulation) |
| `symbol` | VARCHAR | Trading symbol |
| `side` | VARCHAR | buy or sell |
| `entry_time` | TIMESTAMP | Trade entry time |
| `entry_price` | FLOAT | Entry price |
| `volume` | FLOAT | Trade volume (lots) |
| `stop_loss` | FLOAT | Stop loss price |
| `take_profit` | FLOAT | Take profit price |
| `exit_time` | TIMESTAMP | Trade exit time |
| `exit_price` | FLOAT | Exit price |
| `pips` | FLOAT | Profit/loss in pips |
| `profit_usd` | FLOAT | Profit/loss in USD |
| `status` | VARCHAR | open, closed, cancelled |
| `close_reason` | VARCHAR | Why trade was closed |
| `is_winner` | BOOLEAN | Whether trade was profitable |
| `prediction_id` | INTEGER | Reference to prediction |
| `agent_cycle_number` | INTEGER | Agent cycle number |

**Indexes**:
- `idx_trades_entry_time` on `(entry_time DESC)`
- `idx_trades_status` on `(status)`
- `idx_trades_prediction` on `(prediction_id)`

#### 5. `circuit_breaker_events`

Audit trail of safety system activations.

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PRIMARY KEY | Event ID |
| `breaker_type` | VARCHAR | consecutive_loss, drawdown, model_degradation, kill_switch |
| `severity` | VARCHAR | warning, critical |
| `action` | VARCHAR | triggered, recovered, reset, warning |
| `reason` | TEXT | Human-readable reason |
| `value` | FLOAT | Current value that triggered |
| `threshold` | FLOAT | Threshold that was exceeded |
| `triggered_at` | TIMESTAMP | When event occurred |
| `recovered_at` | TIMESTAMP | When breaker recovered (if applicable) |

**Indexes**:
- `idx_cb_events_triggered` on `(triggered_at DESC)`
- `idx_cb_events_type` on `(breaker_type, severity)`

#### 6. `trade_explanations`

Stores LLM-generated trade explanations (if OpenAI API configured).

| Column | Type | Description |
|--------|------|-------------|
| `id` | SERIAL PRIMARY KEY | Explanation ID |
| `trade_id` | VARCHAR | Reference to trade |
| `explanation` | TEXT | LLM explanation of trade decision |
| `confidence_factors` | JSONB | Factors contributing to confidence |
| `risk_assessment` | TEXT | Risk analysis |
| `created_at` | TIMESTAMP | When explanation was generated |

**Indexes**:
- `idx_trade_explanations_trade` on `(trade_id)`

---

## Deployment

### Docker Deployment (Recommended)

The agent runs as a separate container in the Docker Compose stack.

**Start All Services**:

```bash
# From project root
docker-compose up -d

# View logs
docker logs -f ai-trader-agent

# Check health
curl http://localhost:8002/health
```

**Agent Container Specs**:

- **Image**: Custom Python 3.12-slim
- **Dockerfile**: `backend/Dockerfile.agent`
- **Health Port**: 8002
- **Memory Limit**: 2GB
- **CPU Limit**: 1 core
- **Restart Policy**: `unless-stopped`
- **Volumes**:
  - `./backend/models:/app/models:ro` (read-only)
  - `./backend/data/forex:/app/data/forex:ro` (read-only)
  - `./backend/data/sentiment:/app/data/sentiment:ro` (read-only)
  - `./backend/logs:/app/logs` (writable for logs)

### Standalone Deployment (Windows for Paper/Live)

For paper/live trading, run agent on Windows with MT5:

**1. Install Dependencies**:

```bash
pip install -r backend/requirements-api.txt
```

**2. Configure Environment**:

Create `.env.agent`:

```bash
# Database (remote PostgreSQL)
DATABASE_URL=postgresql://user:password@db-host:5432/trading

# Backend API
BACKEND_URL=http://backend-host:8001

# Agent Configuration
AGENT_MODE=paper
AGENT_CONFIDENCE_THRESHOLD=0.70
AGENT_CYCLE_INTERVAL=60

# MT5 Credentials
AGENT_MT5_LOGIN=12345678
AGENT_MT5_PASSWORD=YourPassword
AGENT_MT5_SERVER=ICMarkets-Demo
```

**3. Run Agent**:

```bash
cd backend
python -m src.agent.main
```

**4. Monitor Logs**:

Logs written to `backend/logs/agent.log`.

### Health Checks

**HTTP Health Endpoint**:

```bash
curl http://localhost:8002/health
```

Response:
```json
{
  "status": "healthy",
  "uptime_seconds": 8520.0,
  "agent_status": "running",
  "cycle_count": 142,
  "last_cycle": "2026-01-22T14:30:00Z"
}
```

**Docker Health Check**:

Configured in `docker-compose.yml`:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

---

## Monitoring

### Log Monitoring

**View Agent Logs**:

```bash
# Docker
docker logs -f ai-trader-agent

# Standalone
tail -f backend/logs/agent.log
```

**Log Levels**:
- `INFO`: Normal operations (cycle start/end, trades)
- `WARNING`: Non-critical issues (low confidence, paused)
- `ERROR`: Errors that don't crash agent
- `CRITICAL`: Severe errors (kill switch triggered)

### Metrics Collection

All performance metrics stored in database and accessible via API:

```bash
curl "http://localhost:8001/api/v1/agent/metrics?period=24h"
```

### Alert Integration

Circuit breaker events can trigger external alerts by monitoring `circuit_breaker_events` table:

```sql
SELECT *
FROM circuit_breaker_events
WHERE severity = 'critical'
  AND triggered_at >= NOW() - INTERVAL '1 hour';
```

---

## Troubleshooting

See [AGENT-OPERATIONS-GUIDE.md](AGENT-OPERATIONS-GUIDE.md) for comprehensive troubleshooting procedures.

### Quick Issues

**Agent Won't Start**: Check database connection and model files

**Model Not Loading**: Restart backend service, wait 30-60 seconds

**MT5 Connection Fails**: Use simulation mode in Docker (MT5 requires Windows)

**High Memory Usage**: Increase memory limit in docker-compose.yml

---

## Related Documentation

- [AGENT-OPERATIONS-GUIDE.md](AGENT-OPERATIONS-GUIDE.md) - Detailed operations manual
- [AGENT-API-REFERENCE.md](AGENT-API-REFERENCE.md) - Complete API documentation
- [AGENT-QUICK-REFERENCE.md](AGENT-QUICK-REFERENCE.md) - Cheat sheet for operators
- [CHANGELOG.md](CHANGELOG.md) - Version history and release notes

---

## License

This project is for educational and research purposes. See LICENSE file for details.

---

## Acknowledgments

Developed with assistance from Claude (Anthropic).
