# Agent Quick Reference

Quick reference card for common agent operations. Keep this handy for daily operations.

## Status Checks

```bash
# Agent health (container health check)
curl http://localhost:8002/health

# Agent status (detailed)
curl http://localhost:8001/api/v1/agent/status

# Performance metrics (last 24h)
curl "http://localhost:8001/api/v1/agent/metrics?period=24h"

# Safety status
curl http://localhost:8001/api/v1/agent/safety

# View logs
docker logs -f ai-trader-agent
docker logs --tail 100 ai-trader-agent
```

## Agent Control

```bash
# Start (simulation mode)
curl -X POST http://localhost:8001/api/v1/agent/start \
  -H "Content-Type: application/json" \
  -d '{"mode": "simulation", "confidence_threshold": 0.70}'

# Stop (close positions)
curl -X POST http://localhost:8001/api/v1/agent/stop \
  -H "Content-Type: application/json" \
  -d '{"close_positions": true}'

# Pause
curl -X POST http://localhost:8001/api/v1/agent/pause

# Resume
curl -X POST http://localhost:8001/api/v1/agent/resume

# Update confidence threshold
curl -X PUT http://localhost:8001/api/v1/agent/config \
  -H "Content-Type: application/json" \
  -d '{"confidence_threshold": 0.75}'
```

## Safety Operations

```bash
# Trigger kill switch
curl -X POST http://localhost:8001/api/v1/agent/kill-switch \
  -H "Content-Type: application/json" \
  -d '{"action": "trigger", "reason": "Emergency stop"}'

# Get reset code
curl -X POST http://localhost:8001/api/v1/agent/safety/kill-switch/reset-code

# Reset kill switch
curl -X POST http://localhost:8001/api/v1/agent/kill-switch \
  -H "Content-Type: application/json" \
  -d '{"action": "reset"}'

# Reset circuit breaker
curl -X POST http://localhost:8001/api/v1/agent/safety/circuit-breakers/reset \
  -H "Content-Type: application/json" \
  -d '{"breaker_name": "consecutive_loss"}'

# View safety events
curl "http://localhost:8001/api/v1/agent/safety/events?limit=20"
```

## Command Tracking

```bash
# Check command status (replace 1 with command_id)
curl http://localhost:8001/api/v1/agent/commands/1

# List recent commands
curl "http://localhost:8001/api/v1/agent/commands?limit=10"

# List pending commands
curl "http://localhost:8001/api/v1/agent/commands?status=pending"
```

## Trading Data

```bash
# Get open positions
curl http://localhost:8001/api/v1/trading/positions

# Get trade history
curl "http://localhost:8001/api/v1/trading/history?limit=20"

# Get account status
curl http://localhost:8001/api/v1/trading/account
```

## Docker Operations

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# Restart agent only
docker restart ai-trader-agent

# View agent logs
docker logs -f ai-trader-agent

# Check container status
docker ps | grep ai-trader

# Execute command in agent container
docker exec ai-trader-agent <command>

# Shell into agent container
docker exec -it ai-trader-agent bash
```

## Database Operations

```bash
# Connect to database
docker exec -it ai-trader-postgres psql -U trader trading

# Backup database
docker exec ai-trader-postgres pg_dump -U trader trading > backup.sql

# View agent state
docker exec -it ai-trader-postgres psql -U trader trading \
  -c "SELECT status, mode, cycle_count, kill_switch_active FROM agent_state ORDER BY updated_at DESC LIMIT 1;"

# View recent commands
docker exec -it ai-trader-postgres psql -U trader trading \
  -c "SELECT id, command, status, created_at FROM agent_commands ORDER BY created_at DESC LIMIT 10;"

# View circuit breaker events
docker exec -it ai-trader-postgres psql -U trader trading \
  -c "SELECT breaker_type, severity, action, reason, triggered_at FROM circuit_breaker_events ORDER BY triggered_at DESC LIMIT 10;"

# View recent trades
docker exec -it ai-trader-postgres psql -U trader trading \
  -c "SELECT entry_time, exit_time, direction, pips, is_winner FROM trades ORDER BY entry_time DESC LIMIT 10;"
```

## Troubleshooting

```bash
# Check all service health
curl http://localhost:8001/health  # Backend
curl http://localhost:8002/health  # Agent
curl http://localhost:3001/health  # Frontend (if available)

# Check if model loaded
docker exec ai-trader-agent ls -la /app/models/mtf_ensemble/

# Test database connection
docker exec ai-trader-backend python -c "from src.api.database.session import get_session; next(get_session()); print('DB OK')"

# Check agent configuration
docker exec ai-trader-agent env | grep AGENT_

# Search logs for errors
docker logs ai-trader-agent 2>&1 | grep -i error

# Search logs for circuit breaker triggers
docker logs ai-trader-agent 2>&1 | grep -i "circuit breaker"

# Check last 50 log lines
docker logs --tail 50 ai-trader-agent
```

## Environment Variables

```bash
# Trading Configuration
AGENT_MODE=simulation                    # simulation, paper, live
AGENT_CONFIDENCE_THRESHOLD=0.70         # 0.5-0.95
AGENT_CYCLE_INTERVAL=60                 # seconds
AGENT_MAX_POSITION_SIZE=0.1             # 0-1 (fraction of equity)
AGENT_USE_KELLY_SIZING=true             # true/false

# Safety Settings
AGENT_MAX_CONSECUTIVE_LOSSES=5          # integer
AGENT_MAX_DRAWDOWN_PERCENT=10.0         # percent
AGENT_MAX_DAILY_LOSS_PERCENT=5.0        # percent
AGENT_ENABLE_MODEL_DEGRADATION=false    # true/false

# MT5 (paper/live only)
AGENT_MT5_LOGIN=                        # integer
AGENT_MT5_PASSWORD=                     # string
AGENT_MT5_SERVER=                       # string

# Database
DATABASE_URL=postgresql://trader:password@postgres:5432/trading
```

## Useful SQL Queries

```sql
-- Agent current status
SELECT status, mode, cycle_count, kill_switch_active, updated_at
FROM agent_state
ORDER BY updated_at DESC
LIMIT 1;

-- Pending commands
SELECT id, command, payload, created_at
FROM agent_commands
WHERE status = 'pending'
ORDER BY created_at ASC;

-- Recent circuit breaker events
SELECT breaker_type, severity, action, reason, triggered_at
FROM circuit_breaker_events
ORDER BY triggered_at DESC
LIMIT 10;

-- Trading performance (last 24h)
SELECT
    COUNT(*) as total_trades,
    SUM(CASE WHEN is_winner THEN 1 ELSE 0 END) as winning_trades,
    AVG(CASE WHEN is_winner THEN 1.0 ELSE 0.0 END) as win_rate,
    SUM(pips) as total_pips
FROM trades
WHERE exit_time >= NOW() - INTERVAL '24 hours';

-- Open positions
SELECT ticket, symbol, direction, volume, entry_price, entry_time, unrealized_pips
FROM positions
WHERE status = 'open'
ORDER BY entry_time DESC;

-- Largest losses
SELECT entry_time, exit_time, direction, volume, pips
FROM trades
WHERE is_winner = false
ORDER BY pips ASC
LIMIT 10;

-- Longest running trades
SELECT entry_time, exit_time, direction, volume, pips,
       EXTRACT(EPOCH FROM (exit_time - entry_time))/3600 as duration_hours
FROM trades
ORDER BY duration_hours DESC
LIMIT 10;
```

## Common Workflows

### Start Agent Workflow

```bash
# 1. Check prerequisites
docker ps | grep ai-trader
curl http://localhost:8001/health
docker exec ai-trader-agent ls /app/models/mtf_ensemble/

# 2. Start agent
curl -X POST http://localhost:8001/api/v1/agent/start \
  -H "Content-Type: application/json" \
  -d '{"mode": "simulation", "confidence_threshold": 0.70}'

# 3. Wait for command to process (10 seconds)
sleep 10

# 4. Verify agent running
curl http://localhost:8001/api/v1/agent/status | grep '"status": "running"'

# 5. Monitor first cycles
docker logs -f ai-trader-agent
```

### Stop Agent Workflow

```bash
# 1. Stop agent with position close
curl -X POST http://localhost:8001/api/v1/agent/stop \
  -H "Content-Type: application/json" \
  -d '{"close_positions": true}'

# 2. Wait for command to process
sleep 10

# 3. Verify agent stopped
curl http://localhost:8001/api/v1/agent/status | grep '"status": "stopped"'

# 4. Verify no open positions
curl http://localhost:8001/api/v1/trading/positions
```

### Emergency Stop Workflow

```bash
# 1. Trigger kill switch
curl -X POST http://localhost:8001/api/v1/agent/kill-switch \
  -H "Content-Type: application/json" \
  -d '{"action": "trigger", "reason": "Emergency"}'

# 2. Verify kill switch active
curl http://localhost:8001/api/v1/agent/status | grep '"kill_switch_active": true'

# 3. Review damage
curl "http://localhost:8001/api/v1/agent/metrics?period=all"

# 4. Get reset code (after review and approval)
curl -X POST http://localhost:8001/api/v1/agent/safety/kill-switch/reset-code

# 5. Reset kill switch
curl -X POST http://localhost:8001/api/v1/agent/kill-switch \
  -H "Content-Type: application/json" \
  -d '{"action": "reset"}'
```

### Circuit Breaker Recovery Workflow

```bash
# 1. Check what triggered
curl http://localhost:8001/api/v1/agent/safety/events?limit=5

# 2. Review recent trades
curl "http://localhost:8001/api/v1/trading/history?limit=10"

# 3. Review metrics
curl "http://localhost:8001/api/v1/agent/metrics?period=24h"

# 4. Reset breaker after investigation
curl -X POST http://localhost:8001/api/v1/agent/safety/circuit-breakers/reset \
  -H "Content-Type: application/json" \
  -d '{"breaker_name": "consecutive_loss"}'

# 5. Resume agent
curl -X POST http://localhost:8001/api/v1/agent/resume
```

## Monitoring Checklist

### Every 5 Minutes (Automated)

- [ ] Agent health check (status 200)
- [ ] Cycle count increasing
- [ ] No error status

### Every Hour (Automated)

- [ ] Win rate > 50%
- [ ] Profit factor > 1.5
- [ ] Max drawdown < 8%
- [ ] No circuit breaker triggers

### Daily (Manual)

- [ ] Review metrics for past 24h
- [ ] Check for safety events
- [ ] Verify open positions reasonable
- [ ] Review largest losses
- [ ] Check logs for warnings

### Weekly (Manual)

- [ ] Backup database
- [ ] Review all-time metrics
- [ ] Analyze circuit breaker triggers
- [ ] Check model performance
- [ ] Review agent configuration

## Quick Metrics

| Metric | Green | Yellow | Red |
|--------|-------|--------|-----|
| Win Rate | > 55% | 50-55% | < 50% |
| Profit Factor | > 2.0 | 1.5-2.0 | < 1.5 |
| Drawdown | < 5% | 5-8% | > 8% |
| Circuit Breaker Triggers | 0 | 1-2/week | > 3/week |
| Consecutive Losses | < 3 | 3-4 | 5+ (halt triggered) |

## Support Resources

- **Full Documentation**: [AI-TRADING-AGENT.md](AI-TRADING-AGENT.md)
- **Operations Guide**: [AGENT-OPERATIONS-GUIDE.md](AGENT-OPERATIONS-GUIDE.md)
- **API Reference**: [AGENT-API-REFERENCE.md](AGENT-API-REFERENCE.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **Main README**: [README.md](../README.md)

---

**Version**: 2.0.0
**Last Updated**: 2024-01-22
