# Agent API Reference

Complete REST API reference for controlling and monitoring the AI Trading Agent.

**Base URL**: `http://localhost:8001/api/v1/agent`

**Content-Type**: `application/json`

## Authentication

Currently, the API does not require authentication. In production deployments, implement API key or OAuth2 authentication.

## Command Endpoints

### POST /start

Start the trading agent with specified configuration.

**Request Body:**

```json
{
  "mode": "simulation",
  "confidence_threshold": 0.70,
  "cycle_interval_seconds": 60,
  "max_position_size": 0.1,
  "use_kelly_sizing": true
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `mode` | string | No | `simulation` | Execution mode: `simulation`, `paper`, `live` |
| `confidence_threshold` | float | No | `0.70` | Minimum confidence to execute trades (0.5-0.95) |
| `cycle_interval_seconds` | integer | No | `60` | Seconds between agent cycles (10-3600) |
| `max_position_size` | float | No | `0.1` | Maximum position size as fraction of equity (0-1) |
| `use_kelly_sizing` | boolean | No | `true` | Use Kelly criterion for position sizing |

**Response:**

```json
{
  "status": "queued",
  "command_id": 1,
  "message": "Start command queued successfully in simulation mode. Agent will process shortly."
}
```

**Status Codes:**

- `200 OK`: Command queued successfully
- `400 Bad Request`: Invalid parameters or agent already running
- `500 Internal Server Error`: Server error

**Example:**

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

---

### POST /stop

Stop the trading agent.

**Request Body:**

```json
{
  "force": false,
  "close_positions": false
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `force` | boolean | No | `false` | Force stop even if positions are open |
| `close_positions` | boolean | No | `false` | Close open positions before stopping |

**Response:**

```json
{
  "status": "queued",
  "command_id": 2,
  "message": "Stop command queued successfully. Agent will stop after current cycle."
}
```

**Status Codes:**

- `200 OK`: Command queued successfully
- `400 Bad Request`: Agent already stopped
- `404 Not Found`: Agent not initialized
- `500 Internal Server Error`: Server error

**Example:**

```bash
curl -X POST http://localhost:8001/api/v1/agent/stop \
  -H "Content-Type: application/json" \
  -d '{
    "force": false,
    "close_positions": true
  }'
```

---

### POST /pause

Pause the trading agent. Trading cycles stop but positions remain open.

**Request Body:** None

**Response:**

```json
{
  "status": "queued",
  "command_id": 3,
  "message": "Pause command queued successfully. Agent will pause after current cycle."
}
```

**Status Codes:**

- `200 OK`: Command queued successfully
- `400 Bad Request`: Agent not running
- `404 Not Found`: Agent not initialized
- `500 Internal Server Error`: Server error

**Example:**

```bash
curl -X POST http://localhost:8001/api/v1/agent/pause
```

---

### POST /resume

Resume a paused agent.

**Request Body:** None

**Response:**

```json
{
  "status": "queued",
  "command_id": 4,
  "message": "Resume command queued successfully. Agent will resume trading."
}
```

**Status Codes:**

- `200 OK`: Command queued successfully
- `400 Bad Request`: Agent not paused
- `404 Not Found`: Agent not initialized
- `500 Internal Server Error`: Server error

**Example:**

```bash
curl -X POST http://localhost:8001/api/v1/agent/resume
```

---

### PUT /config

Update agent configuration while running. Only non-null fields will be updated.

**Request Body:**

```json
{
  "confidence_threshold": 0.75,
  "cycle_interval_seconds": 120,
  "max_position_size": 0.05,
  "use_kelly_sizing": false
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `confidence_threshold` | float | No | Update confidence threshold (0.5-0.95) |
| `cycle_interval_seconds` | integer | No | Update cycle interval (10-3600) |
| `max_position_size` | float | No | Update max position size (0-1) |
| `use_kelly_sizing` | boolean | No | Update Kelly sizing setting |

**Response:**

```json
{
  "status": "queued",
  "command_id": 5,
  "message": "Config update queued successfully. 2 fields will be updated."
}
```

**Status Codes:**

- `200 OK`: Command queued successfully
- `400 Bad Request`: No fields provided or agent stopped
- `404 Not Found`: Agent not initialized
- `500 Internal Server Error`: Server error

**Example:**

```bash
curl -X PUT http://localhost:8001/api/v1/agent/config \
  -H "Content-Type: application/json" \
  -d '{
    "confidence_threshold": 0.75,
    "cycle_interval_seconds": 120
  }'
```

---

### POST /kill-switch

Trigger or reset the kill switch.

**Request Body:**

```json
{
  "action": "trigger",
  "reason": "Unexpected market volatility - manual intervention required"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `action` | string | Yes | Action: `trigger` or `reset` |
| `reason` | string | No | Reason for triggering kill switch (max 500 chars) |

**Response (trigger):**

```json
{
  "status": "queued",
  "command_id": 6,
  "message": "Kill switch triggered. Agent will halt immediately and close positions."
}
```

**Response (reset):**

```json
{
  "status": "queued",
  "command_id": 7,
  "message": "Kill switch reset queued. Agent can be restarted after processing."
}
```

**Status Codes:**

- `200 OK`: Command queued successfully
- `400 Bad Request`: Kill switch already active/inactive
- `404 Not Found`: Agent not initialized
- `500 Internal Server Error`: Server error

**Example (trigger):**

```bash
curl -X POST http://localhost:8001/api/v1/agent/kill-switch \
  -H "Content-Type: application/json" \
  -d '{
    "action": "trigger",
    "reason": "Emergency stop - operator initiated"
  }'
```

**Example (reset):**

```bash
curl -X POST http://localhost:8001/api/v1/agent/kill-switch \
  -H "Content-Type: application/json" \
  -d '{
    "action": "reset"
  }'
```

---

## Status & Metrics Endpoints

### GET /status

Get current agent status.

**Response:**

```json
{
  "status": "running",
  "mode": "simulation",
  "cycle_count": 142,
  "last_cycle_at": "2024-01-15T14:30:00Z",
  "account_equity": 103450.00,
  "open_positions": 1,
  "circuit_breaker_state": null,
  "kill_switch_active": false,
  "error_message": null,
  "uptime_seconds": 8520.0,
  "last_prediction": {
    "direction": "long",
    "confidence": 0.72,
    "should_trade": true
  },
  "config": {
    "confidence_threshold": 0.70,
    "cycle_interval_seconds": 60,
    "max_position_size": 0.1,
    "use_kelly_sizing": true
  }
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Agent status: `stopped`, `starting`, `running`, `paused`, `stopping`, `error` |
| `mode` | string | Execution mode: `simulation`, `paper`, `live` |
| `cycle_count` | integer | Number of cycles executed |
| `last_cycle_at` | string | ISO 8601 timestamp of last cycle |
| `account_equity` | float | Current account equity |
| `open_positions` | integer | Number of open positions |
| `circuit_breaker_state` | string | Circuit breaker status (null if none triggered) |
| `kill_switch_active` | boolean | Whether kill switch is triggered |
| `error_message` | string | Error message if status is `error` |
| `uptime_seconds` | float | Seconds since agent started |
| `last_prediction` | object | Last prediction made by agent |
| `config` | object | Current agent configuration |

**Status Codes:**

- `200 OK`: Status retrieved successfully
- `404 Not Found`: Agent not initialized
- `500 Internal Server Error`: Server error

**Example:**

```bash
curl http://localhost:8001/api/v1/agent/status
```

---

### GET /metrics

Get agent performance metrics for specified time period.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `period` | string | `all` | Time period: `all`, `24h`, `7d`, `30d` |

**Response:**

```json
{
  "total_trades": 47,
  "winning_trades": 29,
  "losing_trades": 18,
  "win_rate": 0.617,
  "total_pips": 892.5,
  "profit_factor": 2.45,
  "sharpe_ratio": 3.2,
  "max_drawdown": -125.0,
  "circuit_breaker_triggers": 2,
  "period": "all"
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `total_trades` | integer | Total trades executed |
| `winning_trades` | integer | Number of winning trades |
| `losing_trades` | integer | Number of losing trades |
| `win_rate` | float | Win rate (0-1) |
| `total_pips` | float | Total profit/loss in pips |
| `profit_factor` | float | Gross profit / gross loss |
| `sharpe_ratio` | float | Risk-adjusted return |
| `max_drawdown` | float | Maximum drawdown in pips |
| `circuit_breaker_triggers` | integer | Number of circuit breaker triggers |
| `period` | string | Time period for metrics |

**Status Codes:**

- `200 OK`: Metrics retrieved successfully
- `400 Bad Request`: Invalid period parameter
- `500 Internal Server Error`: Server error

**Example:**

```bash
# Get all-time metrics
curl http://localhost:8001/api/v1/agent/metrics?period=all

# Get 24-hour metrics
curl http://localhost:8001/api/v1/agent/metrics?period=24h
```

---

## Command Status Endpoints

### GET /commands/{command_id}

Get status of a specific command.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `command_id` | integer | Command ID returned from command endpoint |

**Response:**

```json
{
  "command_id": 1,
  "command": "start",
  "status": "completed",
  "created_at": "2024-01-15T14:00:00Z",
  "processed_at": "2024-01-15T14:00:02Z",
  "result": {
    "success": true,
    "status": "running"
  },
  "error_message": null
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `command_id` | integer | Unique command ID |
| `command` | string | Command name: `start`, `stop`, `pause`, `resume`, `kill`, `update_config` |
| `status` | string | Status: `pending`, `processing`, `completed`, `failed` |
| `created_at` | string | ISO 8601 timestamp of command creation |
| `processed_at` | string | ISO 8601 timestamp of command processing |
| `result` | object | Command execution result |
| `error_message` | string | Error message if command failed |

**Status Codes:**

- `200 OK`: Command status retrieved successfully
- `404 Not Found`: Command not found
- `500 Internal Server Error`: Server error

**Example:**

```bash
curl http://localhost:8001/api/v1/agent/commands/1
```

---

### GET /commands

List recent commands with optional filtering.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | `20` | Number of commands to return (max 100) |
| `offset` | integer | `0` | Offset for pagination |
| `status` | string | None | Filter by status: `pending`, `processing`, `completed`, `failed` |

**Response:**

```json
{
  "commands": [
    {
      "command_id": 42,
      "command": "start",
      "status": "completed",
      "created_at": "2024-01-15T14:00:00Z",
      "processed_at": "2024-01-15T14:00:02Z"
    },
    {
      "command_id": 41,
      "command": "stop",
      "status": "completed",
      "created_at": "2024-01-15T13:50:00Z",
      "processed_at": "2024-01-15T13:50:01Z"
    }
  ],
  "count": 2,
  "total": 42
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `commands` | array | List of command items |
| `count` | integer | Number of commands returned |
| `total` | integer | Total commands in database |

**Status Codes:**

- `200 OK`: Commands retrieved successfully
- `400 Bad Request`: Invalid parameters
- `500 Internal Server Error`: Server error

**Example:**

```bash
# Get last 20 commands
curl http://localhost:8001/api/v1/agent/commands

# Get pending commands only
curl "http://localhost:8001/api/v1/agent/commands?status=pending&limit=10"

# Pagination
curl "http://localhost:8001/api/v1/agent/commands?limit=10&offset=20"
```

---

## Safety Endpoints

### GET /safety

Get detailed safety status including circuit breakers and kill switch.

**Response:**

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
    "trades": 15,
    "loss_pct": 2.3,
    "loss_amount": 2300.0
  },
  "account_metrics": {
    "current_equity": 103450.00,
    "peak_equity": 104500.00,
    "drawdown_pct": 1.0
  }
}
```

**Status Codes:**

- `200 OK`: Safety status retrieved successfully
- `404 Not Found`: Agent not initialized
- `500 Internal Server Error`: Server error

**Example:**

```bash
curl http://localhost:8001/api/v1/agent/safety
```

---

### POST /safety/kill-switch/reset-code

Generate authorization code for kill switch reset. Code expires in 5 minutes.

**Response:**

```json
{
  "reset_code": "A3B7C9D2",
  "expires_at": "2024-01-15T14:35:00Z",
  "message": "Use this code to reset the kill switch within 5 minutes."
}
```

**Status Codes:**

- `200 OK`: Reset code generated successfully
- `400 Bad Request`: Kill switch not active
- `404 Not Found`: Agent not initialized
- `500 Internal Server Error`: Server error

**Example:**

```bash
curl -X POST http://localhost:8001/api/v1/agent/safety/kill-switch/reset-code
```

---

### POST /safety/circuit-breakers/reset

Reset a specific circuit breaker after investigation.

**Request Body:**

```json
{
  "breaker_name": "consecutive_loss"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `breaker_name` | string | Yes | Breaker to reset: `consecutive_loss`, `drawdown`, `model_degradation` |

**Response:**

```json
{
  "status": "queued",
  "command_id": 8,
  "message": "Circuit breaker 'consecutive_loss' reset queued successfully."
}
```

**Status Codes:**

- `200 OK`: Reset command queued successfully
- `400 Bad Request`: Invalid breaker name
- `404 Not Found`: Agent not initialized
- `500 Internal Server Error`: Server error

**Example:**

```bash
curl -X POST http://localhost:8001/api/v1/agent/safety/circuit-breakers/reset \
  -H "Content-Type: application/json" \
  -d '{
    "breaker_name": "consecutive_loss"
  }'
```

---

### GET /safety/events

Get recent safety events (circuit breaker triggers, kill switch activations).

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | `50` | Number of events to return (max 200) |
| `breaker_type` | string | None | Filter by breaker type |
| `severity` | string | None | Filter by severity |

**Response:**

```json
{
  "events": [
    {
      "id": 15,
      "breaker_type": "consecutive_loss",
      "severity": "critical",
      "action": "triggered",
      "reason": "5 consecutive losses reached",
      "value": 5,
      "threshold": 5,
      "triggered_at": "2024-01-15T14:25:00Z",
      "recovered_at": null
    },
    {
      "id": 14,
      "breaker_type": "daily_loss",
      "severity": "warning",
      "action": "triggered",
      "reason": "Daily loss: 3.2%",
      "value": 3.2,
      "threshold": 5.0,
      "triggered_at": "2024-01-15T12:00:00Z",
      "recovered_at": "2024-01-15T14:00:00Z"
    }
  ],
  "count": 2,
  "limit": 50
}
```

**Status Codes:**

- `200 OK`: Events retrieved successfully
- `400 Bad Request`: Invalid parameters
- `500 Internal Server Error`: Server error

**Example:**

```bash
# Get last 50 events
curl http://localhost:8001/api/v1/agent/safety/events

# Get critical events only
curl "http://localhost:8001/api/v1/agent/safety/events?severity=critical&limit=20"

# Get consecutive_loss breaker events
curl "http://localhost:8001/api/v1/agent/safety/events?breaker_type=consecutive_loss"
```

---

## Error Responses

All endpoints may return error responses in this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common Error Codes:**

| Code | Meaning | Common Causes |
|------|---------|---------------|
| `400` | Bad Request | Invalid parameters, invalid state transition |
| `404` | Not Found | Agent not initialized, command not found |
| `500` | Internal Server Error | Database error, unexpected exception |
| `503` | Service Unavailable | Agent not responding, database down |

---

## Rate Limiting

Currently, there is no rate limiting. In production deployments, implement rate limiting:

- Command endpoints: 10 requests/minute per IP
- Status endpoints: 60 requests/minute per IP
- Metrics endpoints: 30 requests/minute per IP

---

## Webhooks (Future)

Webhook support is planned for future releases to notify external systems of:

- Agent state changes (started, stopped, paused)
- Circuit breaker triggers
- Kill switch activations
- Trade executions
- Error conditions

---

**Version**: 2.0.0
**Last Updated**: 2024-01-22
**Related Documentation**:
- [AI Trading Agent](AI-TRADING-AGENT.md)
- [Agent Operations Guide](AGENT-OPERATIONS-GUIDE.md)
