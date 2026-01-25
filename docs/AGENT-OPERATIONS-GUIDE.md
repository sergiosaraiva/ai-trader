# Agent Operations Guide

This guide provides operational procedures for managing the AI Trading Agent in production environments.

## Table of Contents

1. [Starting the Agent](#starting-the-agent)
2. [Stopping the Agent](#stopping-the-agent)
3. [Monitoring](#monitoring)
4. [Incident Response](#incident-response)
5. [Maintenance](#maintenance)
6. [Recovery Procedures](#recovery-procedures)

## Starting the Agent

### Pre-Flight Checks

Before starting the agent, verify the following:

#### 1. Check System Health

```bash
# Check all containers are running
docker ps | grep ai-trader

# Expected output:
# ai-trader-postgres   (healthy)
# ai-trader-backend    (healthy)
# ai-trader-agent      (up)
# ai-trader-frontend   (healthy)

# Check backend health
curl http://localhost:8001/health

# Check agent health
curl http://localhost:8002/health
```

#### 2. Verify Model Files

```bash
# Check model files exist
docker exec ai-trader-agent ls -la /app/models/mtf_ensemble/

# Expected files:
# 1H_model.pkl
# 4H_model.pkl
# D_model.pkl
# stacking_meta_learner.pkl
# training_metadata.json
```

#### 3. Check Database Connection

```bash
# Test database connection
docker exec ai-trader-backend python -c "
from src.api.database.session import get_session
session = next(get_session())
print('Database connection OK')
session.close()
"
```

#### 4. Verify Configuration

Review environment variables:

```bash
# Check agent configuration
docker exec ai-trader-agent env | grep AGENT_

# Verify critical settings:
# - AGENT_MODE (simulation/paper/live)
# - AGENT_CONFIDENCE_THRESHOLD (0.70 recommended)
# - AGENT_MAX_CONSECUTIVE_LOSSES (5 recommended)
# - AGENT_MAX_DRAWDOWN_PERCENT (10.0 recommended)
```

#### 5. Review Market Conditions

Before starting in paper/live mode:

- Check if markets are open (forex trades 24/5)
- Review current volatility (VIX levels)
- Check for scheduled news events (high impact)
- Verify sufficient margin available

### Start Command

#### Via API (Recommended)

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

**Response:**

```json
{
  "status": "queued",
  "command_id": 1,
  "message": "Start command queued successfully in simulation mode. Agent will process shortly."
}
```

#### Via Frontend Dashboard

1. Open dashboard at `http://localhost:3001`
2. Navigate to Agent Control panel
3. Select mode and parameters
4. Click "Start Agent"

### Verification Steps

#### 1. Check Command Status

```bash
# Poll command status (replace 1 with actual command_id)
curl http://localhost:8001/api/v1/agent/commands/1

# Wait for status: "completed"
# If status is "failed", check error_message
```

#### 2. Verify Agent is Running

```bash
# Check agent status
curl http://localhost:8001/api/v1/agent/status

# Expected response:
# {
#   "status": "running",
#   "mode": "simulation",
#   "cycle_count": 1,
#   ...
# }
```

#### 3. Monitor First Cycles

```bash
# Watch agent logs
docker logs -f ai-trader-agent

# Look for:
# - "Agent started successfully"
# - "Executing cycle 1"
# - Prediction made
# - Safety checks passed
```

#### 4. Verify Trades (if any)

```bash
# Check if any trades were executed
curl http://localhost:8001/api/v1/trading/history?limit=10

# Check open positions
curl http://localhost:8001/api/v1/trading/positions
```

### Start Checklist

- [ ] All containers healthy
- [ ] Model files verified
- [ ] Database connection OK
- [ ] Configuration reviewed
- [ ] Market conditions checked (paper/live only)
- [ ] Start command issued
- [ ] Command status = completed
- [ ] Agent status = running
- [ ] First cycle executed successfully
- [ ] No safety triggers in first 5 cycles

## Stopping the Agent

### Graceful Shutdown

Use graceful shutdown to allow the agent to finish its current cycle and optionally close positions.

#### Stop Command

```bash
curl -X POST http://localhost:8001/api/v1/agent/stop \
  -H "Content-Type: application/json" \
  -d '{
    "force": false,
    "close_positions": false
  }'
```

**Parameters:**

- `force`: If true, stop immediately without waiting for current cycle
- `close_positions`: If true, close all open positions before stopping

#### Typical Stop Scenarios

**1. Normal Shutdown (keep positions open):**

```bash
curl -X POST http://localhost:8001/api/v1/agent/stop \
  -H "Content-Type: application/json" \
  -d '{
    "force": false,
    "close_positions": false
  }'
```

**2. Full Shutdown (close positions):**

```bash
curl -X POST http://localhost:8001/api/v1/agent/stop \
  -H "Content-Type: application/json" \
  -d '{
    "force": false,
    "close_positions": true
  }'
```

**3. Emergency Stop (immediate):**

```bash
curl -X POST http://localhost:8001/api/v1/agent/stop \
  -H "Content-Type: application/json" \
  -d '{
    "force": true,
    "close_positions": true
  }'
```

### Force Stop

If the agent is not responding to stop commands:

```bash
# Kill agent container
docker kill ai-trader-agent

# Restart agent container (it will start in stopped state)
docker start ai-trader-agent
```

**WARNING**: Force stopping may leave positions open and state inconsistent. Verify manually after restart.

### Emergency Procedures

#### Kill Switch (Emergency Stop)

Use the kill switch for immediate halt with position closure:

```bash
curl -X POST http://localhost:8001/api/v1/agent/kill-switch \
  -H "Content-Type: application/json" \
  -d '{
    "action": "trigger",
    "reason": "Emergency stop - operator initiated"
  }'
```

The kill switch:
- Stops trading immediately
- Closes all open positions
- Requires authorization to reset
- Logs event to safety audit trail

### Stop Verification

```bash
# Check agent status
curl http://localhost:8001/api/v1/agent/status

# Expected response:
# {
#   "status": "stopped",
#   ...
# }

# Verify no open positions
curl http://localhost:8001/api/v1/trading/positions
# Expected: []

# Check last trades closed properly
curl http://localhost:8001/api/v1/trading/history?limit=5
```

## Monitoring

### Health Checks

#### Agent Health Endpoint

```bash
# Check agent health
curl http://localhost:8002/health

# Healthy response (status 200):
{
  "status": "healthy",
  "agent_status": "running",
  "cycle_count": 142,
  "model_loaded": true
}

# Unhealthy response (status 503):
{
  "status": "unhealthy",
  "agent_status": "error",
  "last_error": "Failed to connect to broker"
}
```

**Monitoring Setup:**

Configure your monitoring system (Prometheus, DataDog, etc.) to:

1. Poll `/health` every 30 seconds
2. Alert if status 503 for > 2 minutes
3. Alert if `cycle_count` not increasing for > 5 minutes
4. Alert if `model_loaded: false`

#### Backend Health Endpoint

```bash
# Check backend health
curl http://localhost:8001/health
```

### Performance Metrics

#### Real-Time Metrics

```bash
# Get current metrics
curl "http://localhost:8001/api/v1/agent/metrics?period=all"

# Get 24-hour metrics
curl "http://localhost:8001/api/v1/agent/metrics?period=24h"
```

#### Key Metrics to Monitor

| Metric | Green | Yellow | Red | Action |
|--------|-------|--------|-----|--------|
| Win Rate | > 55% | 50-55% | < 50% | Review if < 50% for > 100 trades |
| Profit Factor | > 2.0 | 1.5-2.0 | < 1.5 | Pause if < 1.5 for > 100 trades |
| Max Drawdown | < 5% | 5-8% | > 8% | Review positions if > 8% |
| Circuit Breaker Triggers | 0 | 1-2/week | > 3/week | Investigate root cause |
| Consecutive Losses | < 3 | 3-4 | 5+ | Halt triggered automatically |

### Alert Thresholds

Configure alerts for:

#### Critical Alerts (Immediate Response)

1. **Kill Switch Activated**
   - Event: `kill_switch_active: true`
   - Action: Review immediately, check positions

2. **Circuit Breaker Triggered**
   - Event: `circuit_breaker_state` not null
   - Action: Review trades, check market conditions

3. **Agent Crash**
   - Event: Agent health status 503 for > 2 minutes
   - Action: Check logs, verify database connection

4. **Drawdown > 10%**
   - Event: `max_drawdown < -10.0`
   - Action: Stop agent immediately, review strategy

#### Warning Alerts (Review Within 1 Hour)

1. **Win Rate Decline**
   - Event: Win rate < 50% over last 50 trades
   - Action: Review recent trades, check market conditions

2. **High Frequency Trading**
   - Event: > 20 trades in 1 hour
   - Action: Check for signal spam, review cycle interval

3. **Position Stuck**
   - Event: Position open for > 24 hours
   - Action: Review position, consider manual close

4. **Model Confidence Low**
   - Event: All predictions < 60% confidence for > 10 cycles
   - Action: Review market conditions, consider pause

### Log Monitoring

```bash
# Follow agent logs in real-time
docker logs -f ai-trader-agent

# Search for errors
docker logs ai-trader-agent 2>&1 | grep -i error

# Search for safety events
docker logs ai-trader-agent 2>&1 | grep -i "circuit breaker\|kill switch"

# Export logs to file
docker logs ai-trader-agent > agent-logs-$(date +%Y%m%d).log
```

**Log Patterns to Watch:**

- `ERROR`: Failed operations, exceptions
- `WARNING.*circuit breaker`: Safety system activations
- `WARNING.*reconnect`: Broker connection issues
- `Cycle.*failed`: Trading cycle failures

## Incident Response

### Circuit Breaker Triggered

**Symptoms:**
- Agent status = paused
- `circuit_breaker_state` not null
- Trading stopped

**Response Procedure:**

#### 1. Identify Trigger

```bash
# Check safety events
curl http://localhost:8001/api/v1/agent/safety/events?limit=10

# Look for most recent event with action="triggered"
```

#### 2. Analyze Root Cause

**Consecutive Loss Breaker:**

```bash
# Review recent trades
curl http://localhost:8001/api/v1/trading/history?limit=10

# Check for pattern:
# - All similar direction (all longs or all shorts)?
# - Low confidence trades?
# - Volatile market conditions?
```

**Drawdown Breaker:**

```bash
# Check metrics
curl "http://localhost:8001/api/v1/agent/metrics?period=24h"

# Calculate current drawdown:
# drawdown_pct = (current_equity - peak_equity) / peak_equity * 100

# Review largest losing trades
```

**Model Degradation Breaker:**

```bash
# Check win rate over rolling window
curl "http://localhost:8001/api/v1/agent/metrics?period=24h"

# If win_rate < 45%, investigate:
# - Market regime change?
# - Model no longer effective?
# - Data quality issues?
```

#### 3. Take Corrective Action

Based on root cause:

- **Market Conditions**: Wait for conditions to improve
- **Model Issues**: Retrain or update model
- **Configuration**: Adjust confidence threshold or position sizing
- **Bug**: Fix code, redeploy

#### 4. Reset Circuit Breaker

```bash
# Reset specific breaker
curl -X POST http://localhost:8001/api/v1/agent/safety/circuit-breakers/reset \
  -H "Content-Type: application/json" \
  -d '{"breaker_name": "consecutive_loss"}'

# Verify reset
curl http://localhost:8001/api/v1/agent/safety
```

#### 5. Resume Agent

```bash
# Resume trading
curl -X POST http://localhost:8001/api/v1/agent/resume

# Monitor closely for next 10 cycles
docker logs -f ai-trader-agent
```

### Kill Switch Activated

**Symptoms:**
- `kill_switch_active: true`
- All trading halted
- Positions closed (if configured)

**Response Procedure:**

#### 1. Review Trigger Reason

```bash
# Check safety events
curl http://localhost:8001/api/v1/agent/safety/events?limit=5

# Look for kill_switch event with reason
```

#### 2. Assess Damage

```bash
# Check final metrics
curl "http://localhost:8001/api/v1/agent/metrics?period=all"

# Review last trades
curl http://localhost:8001/api/v1/trading/history?limit=20

# Check account equity
curl http://localhost:8001/api/v1/trading/account
```

#### 3. Investigate Root Cause

**Daily Loss Limit:**
- Was there a market event (news, volatility spike)?
- Were all trades low confidence?
- Was position sizing too aggressive?

**MT5 Disconnection:**
- Network issues?
- MT5 server maintenance?
- Credentials expired?

**Manual Trigger:**
- Who triggered it and why?
- What market conditions prompted it?

#### 4. Get Reset Authorization

```bash
# Generate reset code
curl -X POST http://localhost:8001/api/v1/agent/safety/kill-switch/reset-code

# Response:
{
  "reset_code": "A3B7C9D2",
  "expires_at": "2024-01-15T14:35:00Z",
  "message": "Use this code to reset the kill switch within 5 minutes."
}
```

**IMPORTANT**: This code expires in 5 minutes. Get approval from supervisor/team lead before resetting.

#### 5. Reset Kill Switch

```bash
# Reset with authorization
curl -X POST http://localhost:8001/api/v1/agent/kill-switch \
  -H "Content-Type: application/json" \
  -d '{
    "action": "reset"
  }'
```

#### 6. Restart Agent with Adjusted Parameters

```bash
# Start with more conservative settings
curl -X POST http://localhost:8001/api/v1/agent/start \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "simulation",
    "confidence_threshold": 0.75,
    "max_position_size": 0.05,
    "use_kelly_sizing": true
  }'
```

### Connection Loss

**Symptoms:**
- Broker status = disconnected
- No trades executing
- Health check warnings

**Response Procedure:**

#### 1. Verify Network Connectivity

```bash
# Check if backend can reach external services
docker exec ai-trader-backend curl -I https://www.google.com

# Check database connectivity
docker exec ai-trader-backend nc -zv postgres 5432
```

#### 2. Check MT5 Status (Paper/Live Mode)

```bash
# Check agent logs for connection errors
docker logs --tail 50 ai-trader-agent | grep -i "mt5\|broker"

# Verify MT5 credentials are correct
docker exec ai-trader-agent env | grep MT5
```

#### 3. Attempt Reconnection

The agent automatically attempts to reconnect every cycle. Monitor logs:

```bash
docker logs -f ai-trader-agent | grep -i reconnect
```

#### 4. Manual Restart if Needed

If auto-reconnect fails:

```bash
# Pause agent
curl -X POST http://localhost:8001/api/v1/agent/pause

# Stop agent
curl -X POST http://localhost:8001/api/v1/agent/stop \
  -H "Content-Type: application/json" \
  -d '{"force": false, "close_positions": false}'

# Verify connectivity
# (check MT5 terminal, network, credentials)

# Restart agent
curl -X POST http://localhost:8001/api/v1/agent/start \
  -H "Content-Type: application/json" \
  -d '{...}'
```

## Maintenance

### Database Backup

Perform regular backups of the PostgreSQL database:

```bash
# Backup database
docker exec ai-trader-postgres pg_dump -U trader trading > backup-$(date +%Y%m%d).sql

# Verify backup
ls -lh backup-*.sql

# Compress backup
gzip backup-$(date +%Y%m%d).sql
```

**Backup Schedule:**
- Daily: Retain last 7 days
- Weekly: Retain last 4 weeks
- Monthly: Retain last 12 months

**Backup Content:**
- Trade history (`trades` table)
- Position history (`positions` table)
- Agent state (`agent_state` table)
- Command history (`agent_commands` table)
- Safety events (`circuit_breaker_events` table)

### Log Rotation

Agent logs can grow large over time. Implement log rotation:

```bash
# Check current log size
docker logs ai-trader-agent 2>&1 | wc -l

# Export and clear logs monthly
docker logs ai-trader-agent > logs/agent-$(date +%Y%m).log 2>&1

# Configure Docker log rotation in docker-compose.yml:
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### Model Updates

When retraining or updating the model:

#### 1. Train New Model

```bash
cd backend
python scripts/train_mtf_ensemble.py --sentiment --stacking
```

#### 2. Validate New Model

```bash
# Run backtest on new model
python scripts/backtest_mtf_ensemble.py --model-dir models/mtf_ensemble_new

# Compare results to current model
python scripts/compare_models.py models/mtf_ensemble models/mtf_ensemble_new
```

#### 3. Stop Agent

```bash
curl -X POST http://localhost:8001/api/v1/agent/stop \
  -H "Content-Type: application/json" \
  -d '{"force": false, "close_positions": true}'
```

#### 4. Deploy New Model

```bash
# Backup current model
mv models/mtf_ensemble models/mtf_ensemble_backup_$(date +%Y%m%d)

# Deploy new model
mv models/mtf_ensemble_new models/mtf_ensemble

# Restart agent container to reload model
docker restart ai-trader-agent
```

#### 5. Restart Agent and Monitor

```bash
# Start agent
curl -X POST http://localhost:8001/api/v1/agent/start \
  -H "Content-Type: application/json" \
  -d '{...}'

# Monitor first 20 cycles closely
docker logs -f ai-trader-agent
```

#### 6. Rollback if Needed

```bash
# Stop agent
curl -X POST http://localhost:8001/api/v1/agent/stop \
  -H "Content-Type: application/json" \
  -d '{"force": true, "close_positions": true}'

# Restore backup model
rm -rf models/mtf_ensemble
mv models/mtf_ensemble_backup_$(date +%Y%m%d) models/mtf_ensemble

# Restart agent container
docker restart ai-trader-agent

# Restart agent
curl -X POST http://localhost:8001/api/v1/agent/start \
  -H "Content-Type: application/json" \
  -d '{...}'
```

### System Updates

When updating the system (code, dependencies, etc.):

#### 1. Schedule Maintenance Window

- Choose low-activity period (e.g., weekends)
- Close all positions before starting
- Notify team of downtime

#### 2. Backup Everything

```bash
# Backup database
docker exec ai-trader-postgres pg_dump -U trader trading > backup-pre-update.sql

# Backup model files
tar -czf models-backup-$(date +%Y%m%d).tar.gz backend/models/

# Export current configuration
docker-compose config > docker-compose-backup.yml
```

#### 3. Stop All Services

```bash
docker-compose down
```

#### 4. Update Code

```bash
git pull origin main
```

#### 5. Rebuild Containers

```bash
docker-compose build --no-cache
```

#### 6. Start Services

```bash
docker-compose up -d
```

#### 7. Verify All Services

```bash
# Check health
curl http://localhost:8001/health
curl http://localhost:8002/health

# Run smoke tests
curl http://localhost:8001/api/v1/predictions/current
```

#### 8. Restart Agent

```bash
curl -X POST http://localhost:8001/api/v1/agent/start \
  -H "Content-Type: application/json" \
  -d '{...}'
```

## Recovery Procedures

### Crash Recovery

The agent is designed for automatic crash recovery:

**What Happens on Crash:**

1. Agent container restarts automatically (Docker `restart: unless-stopped`)
2. Agent loads state from database (`agent_state` table)
3. Agent resumes from last successful cycle
4. Open positions are preserved
5. Pending commands are reprocessed

**Verification After Crash:**

```bash
# Check agent status
curl http://localhost:8001/api/v1/agent/status

# Verify cycle_count resumed correctly
# Verify last_cycle_at matches expectation

# Check for state corruption
curl http://localhost:8001/api/v1/trading/positions
# Compare to MT5 (if paper/live mode)

# Review logs for crash cause
docker logs --tail 100 ai-trader-agent | grep -i error
```

**Manual Recovery Steps (if auto-recovery fails):**

```bash
# 1. Stop agent completely
docker stop ai-trader-agent

# 2. Check database state
docker exec -it ai-trader-postgres psql -U trader trading -c "SELECT * FROM agent_state ORDER BY updated_at DESC LIMIT 1;"

# 3. Clear corrupted state (if needed)
docker exec -it ai-trader-postgres psql -U trader trading -c "UPDATE agent_state SET status = 'stopped' WHERE id = 1;"

# 4. Start agent container
docker start ai-trader-agent

# 5. Restart agent via API
curl -X POST http://localhost:8001/api/v1/agent/start \
  -H "Content-Type: application/json" \
  -d '{...}'
```

### Data Reconciliation

After crash or extended downtime, reconcile agent data with MT5:

#### 1. Export MT5 Positions

```bash
# In MT5 Terminal:
# - Right-click on Trade tab
# - Export to CSV
# - Save as mt5_positions.csv
```

#### 2. Get Agent Positions

```bash
curl http://localhost:8001/api/v1/trading/positions > agent_positions.json
```

#### 3. Compare and Reconcile

```python
# reconcile.py
import json
import pandas as pd

# Load data
mt5_positions = pd.read_csv('mt5_positions.csv')
agent_positions = json.load(open('agent_positions.json'))

# Compare ticket numbers
mt5_tickets = set(mt5_positions['ticket'])
agent_tickets = set(p['ticket'] for p in agent_positions)

# Find mismatches
missing_in_agent = mt5_tickets - agent_tickets
missing_in_mt5 = agent_tickets - mt5_tickets

print(f"Positions in MT5 but not in agent: {missing_in_agent}")
print(f"Positions in agent but not in MT5: {missing_in_mt5}")
```

#### 4. Manual Corrections

If mismatches found:

```bash
# Close phantom positions in database
docker exec -it ai-trader-postgres psql -U trader trading -c "UPDATE positions SET status = 'closed', exit_time = NOW() WHERE ticket IN (12345, 67890);"

# Or add missing positions (rare - usually indicates data corruption)
```

### Manual Intervention

When manual intervention is required:

#### Close All Positions

```bash
# Via agent API (preferred)
curl -X POST http://localhost:8001/api/v1/agent/stop \
  -H "Content-Type: application/json" \
  -d '{"force": false, "close_positions": true}'

# Via MT5 (if agent unresponsive)
# - Open MT5 Terminal
# - Right-click on position
# - Close Order
```

#### Manual Trade Execution

```bash
# Execute trade manually (use with caution)
curl -X POST http://localhost:8001/api/v1/trading/execute \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "EURUSD",
    "direction": "long",
    "volume": 0.1,
    "reason": "Manual trade - operator override"
  }'
```

#### Reset Agent State

```bash
# Clear agent state (DANGEROUS - loses all state)
docker exec -it ai-trader-postgres psql -U trader trading -c "DELETE FROM agent_state;"

# Restart agent - will initialize fresh state
docker restart ai-trader-agent
```

---

**Version**: 2.0.0
**Last Updated**: 2024-01-22
**Related Documentation**:
- [AI Trading Agent](AI-TRADING-AGENT.md)
- [Agent API Reference](AGENT-API-REFERENCE.md)
