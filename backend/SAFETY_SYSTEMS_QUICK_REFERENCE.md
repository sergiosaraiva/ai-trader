# Safety Systems Quick Reference

**Quick guide for monitoring and managing trading agent safety systems**

---

## Safety Status Check

```bash
# Get current safety status
curl http://localhost:8001/api/v1/agent/safety
```

**Response Indicators:**
- `is_safe_to_trade: true` - ✅ All systems OK
- `is_safe_to_trade: false` - ⚠️ Trading restricted

---

## Circuit Breakers

### Types

| Breaker | Trigger | Action |
|---------|---------|--------|
| **Consecutive Loss** | 5 losses in a row | HALT (12h cooldown) |
| **Drawdown** | 10% from peak | HALT (72h cooldown) |
| **Daily Loss** | 5% or $5000 | Kill Switch |
| **Model Degradation** | Win rate < 45% (optional) | HALT |

### States

- **ACTIVE** - Normal trading
- **REDUCED** - Reduced position sizes (50-75%)
- **HALTED** - No trading allowed
- **RECOVERING** - Testing phase after halt

### Reset Circuit Breaker

```bash
# Reset specific breaker
curl -X POST http://localhost:8001/api/v1/agent/safety/circuit-breakers/reset \
  -H "Content-Type: application/json" \
  -d '{"breaker_name": "consecutive_loss"}'
```

**Available breakers:**
- `consecutive_loss`
- `drawdown`
- `model_degradation`

---

## Kill Switch

### When It Triggers

**Automatic:**
- Daily loss ≥ 5% or $5000
- Daily trades ≥ 50
- MT5 disconnected > 60s

**Manual:**
- Emergency stop command

### Reset Kill Switch

**Step 1: Get reset code**
```bash
curl -X POST http://localhost:8001/api/v1/agent/safety/kill-switch/reset-code
```

**Response:**
```json
{
  "reset_code": "A3F7B2C9",
  "expires_at": "2026-01-22T15:35:00Z",
  "message": "Valid for 5 minutes"
}
```

**Step 2: Reset with code**
```bash
curl -X POST http://localhost:8001/api/v1/agent/kill-switch \
  -H "Content-Type: application/json" \
  -d '{
    "action": "reset",
    "authorization": "A3F7B2C9"
  }'
```

---

## Safety Events Audit

```bash
# View recent safety events
curl "http://localhost:8001/api/v1/agent/safety/events?limit=20"

# Filter by severity
curl "http://localhost:8001/api/v1/agent/safety/events?severity=critical"

# Filter by breaker type
curl "http://localhost:8001/api/v1/agent/safety/events?breaker_type=kill_switch"
```

---

## Configuration

### Environment Variables

**Safety Limits:**
```bash
AGENT_MAX_CONSECUTIVE_LOSSES=5
AGENT_MAX_DRAWDOWN_PERCENT=10.0
AGENT_MAX_DAILY_LOSS_PERCENT=5.0
```

**Kill Switch:**
```bash
AGENT_SAFETY_MAX_DAILY_TRADES=50
AGENT_SAFETY_MAX_DAILY_LOSS_AMOUNT=5000.0
AGENT_SAFETY_REQUIRE_TOKEN_FOR_RESET=true
```

---

## Common Scenarios

### Scenario 1: Agent Hit 5 Consecutive Losses

**Status:**
```json
{
  "is_safe_to_trade": false,
  "circuit_breakers": {
    "overall_state": "halted",
    "active_breakers": ["consecutive_loss"],
    "reasons": ["Consecutive loss limit reached: 5"]
  }
}
```

**Action:**
1. Review the 5 losing trades
2. Check if market conditions changed
3. Wait for 12-hour cooldown
4. System enters recovery protocol automatically
5. Or manually reset: `POST /safety/circuit-breakers/reset`

---

### Scenario 2: Drawdown Approaching Limit

**Status:**
```json
{
  "circuit_breakers": {
    "overall_state": "reduced",
    "active_breakers": ["drawdown"],
    "size_multiplier": 0.5,
    "reasons": ["Drawdown at 7.5% (75% of limit)"]
  },
  "account_metrics": {
    "current_equity": 92500,
    "peak_equity": 100000,
    "drawdown_pct": 7.5
  }
}
```

**Action:**
- System automatically reduces position sizes to 50%
- Continues trading with caution
- If drawdown hits 10%, full halt triggered

---

### Scenario 3: Daily Loss Triggered Kill Switch

**Status:**
```json
{
  "is_safe_to_trade": false,
  "kill_switch": {
    "is_active": true,
    "reason": "Daily loss limit exceeded: 5.2% >= 5.0%"
  },
  "daily_metrics": {
    "trades": 28,
    "loss_pct": 5.2,
    "loss_amount": 5200
  }
}
```

**Action:**
1. Review today's trades - what went wrong?
2. Check market conditions - unexpected volatility?
3. Request reset code: `POST /safety/kill-switch/reset-code`
4. Use code to reset if appropriate
5. System auto-resets at next trading day if configured

---

### Scenario 4: MT5 Connection Lost

**Status:**
```json
{
  "kill_switch": {
    "is_active": true,
    "reason": "Broker disconnected for 62s >= 60s"
  }
}
```

**Action:**
1. Check MT5 terminal connection
2. Verify internet connectivity
3. Check MT5 credentials
4. Reconnect broker
5. Reset kill switch once connection stable

---

## Safety Best Practices

### Daily Checks
- [ ] Check safety status at start of day
- [ ] Review yesterday's safety events
- [ ] Verify circuit breaker states
- [ ] Check current equity vs peak

### Weekly Reviews
- [ ] Review all safety event triggers
- [ ] Analyze circuit breaker patterns
- [ ] Adjust thresholds if needed
- [ ] Validate kill switch functionality

### After Safety Event
1. ✅ **Stop** - Don't immediately reset
2. ✅ **Review** - Understand why it triggered
3. ✅ **Analyze** - Check trades, market, model
4. ✅ **Decide** - Is it safe to resume?
5. ✅ **Reset** - Use proper authorization
6. ✅ **Monitor** - Watch closely after reset

---

## Emergency Procedures

### Emergency Stop
```bash
# Trigger kill switch immediately
curl -X POST http://localhost:8001/api/v1/agent/kill-switch \
  -H "Content-Type: application/json" \
  -d '{
    "action": "trigger",
    "reason": "Emergency stop - market conditions"
  }'
```

### Force Stop Agent
```bash
# Stop agent (closes positions)
curl -X POST http://localhost:8001/api/v1/agent/stop \
  -H "Content-Type: application/json" \
  -d '{"force": true, "close_positions": true}'
```

---

## Monitoring Dashboard

**Key Metrics to Watch:**

```python
# Safety Status
is_safe_to_trade: bool
circuit_breaker_state: str (active/reduced/halted)

# Daily Limits
daily_trades: int / 50 max
daily_loss_pct: float / 5.0% max

# Account Health
current_equity: float
drawdown_pct: float / 10.0% max
consecutive_losses: int / 5 max

# Connection
broker_connected: bool
last_connection_check: datetime
```

---

## Troubleshooting

### Kill Switch Won't Reset

**Possible Causes:**
1. Invalid authorization code
2. Code expired (>5 minutes)
3. Agent not running
4. Database connection issue

**Solution:**
1. Request new reset code
2. Use code within 5 minutes
3. Verify agent status
4. Check logs for errors

### Circuit Breaker Keeps Triggering

**Possible Causes:**
1. Model not performing well
2. Market regime changed
3. Thresholds too aggressive
4. Data quality issues

**Solution:**
1. Review model metrics
2. Check current market conditions
3. Adjust safety thresholds
4. Validate data pipeline

---

## Support Contacts

**For Safety System Issues:**
- Check logs: `backend/logs/agent.log`
- Review events: `GET /api/v1/agent/safety/events`
- Agent status: `GET /api/v1/agent/status`

**Emergency Escalation:**
1. Stop agent immediately
2. Close all open positions
3. Review database events
4. Contact system administrator

---

**Last Updated:** January 22, 2026
**Safety Systems Version:** 1.0.0
