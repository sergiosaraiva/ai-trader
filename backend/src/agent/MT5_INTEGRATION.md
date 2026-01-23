# Phase 5: MT5 Integration

## Overview

The agent now supports actual trade execution via MetaTrader 5 in three modes:

- **simulation**: No MT5 connection, logs signals only
- **paper**: Connects to MT5 demo account, executes real orders on demo
- **live**: Connects to MT5 live account, executes real trades

## Architecture

```
AgentRunner
  ├── BrokerManager (manages MT5 connection)
  ├── TradeExecutor (executes trades)
  └── TradingCycle (orchestrates predict → signal → trade)
```

## New Modules

### 1. `broker_manager.py`

Manages MT5 broker connection lifecycle:

- `connect()` - Connect to MT5 broker
- `disconnect()` - Disconnect gracefully
- `reconnect()` - Automatic reconnection on connection loss
- `is_connected()` - Check connection status
- `get_account_info()` - Get account equity, balance, etc.
- `get_open_positions()` - Get list of open positions

### 2. `trade_executor.py`

Handles trade execution:

- `execute_signal()` - Execute a trading signal
  1. Validates signal
  2. Calculates position size (Kelly Criterion or fixed)
  3. Submits order to MT5
  4. Waits for fill
  5. Stores trade in database with `mt5_ticket`

- `check_open_positions()` - Check open positions for exit conditions (TP/SL/timeout)
- `close_position()` - Close a specific position
- `close_all_positions()` - Close all positions (on shutdown)

### 3. `position_tracker.py`

Tracks positions using triple barrier method:

- `track_position()` - Start tracking a position
- `check_exits()` - Check all positions for exit signals
  - Take Profit hit
  - Stop Loss hit
  - Timeout (max holding period)

### 4. Updated `models.py`

New data classes:

- `TradeResult` - Result of trade execution
- `PositionStatus` - Status of an open position
- `ExitSignal` - Signal to exit a position

## Configuration

Set via environment variables:

```bash
# Trading mode
export AGENT_MODE=paper  # or "simulation", "live"

# MT5 credentials (required for paper/live)
export AGENT_MT5_LOGIN=12345678
export AGENT_MT5_PASSWORD="your_password"
export AGENT_MT5_SERVER="YourBroker-Demo"

# Trading parameters
export AGENT_SYMBOL=EURUSD
export AGENT_CONFIDENCE_THRESHOLD=0.70
export AGENT_MAX_POSITION_SIZE=0.1
export AGENT_USE_KELLY_SIZING=true
export AGENT_CYCLE_INTERVAL=60  # seconds
```

## Workflow

### Simulation Mode

```
1. Generate prediction
2. Check confidence threshold
3. [SIMULATION] Log signal (no execution)
4. Wait for next cycle
```

### Paper/Live Mode

```
1. Check open positions for exits
   - Close positions that hit TP/SL/timeout

2. Generate prediction
3. Check confidence threshold
4. Create trading signal
5. Execute trade via MT5
   - Calculate position size
   - Submit order
   - Wait for fill
   - Store in database with mt5_ticket

6. Wait for next cycle
```

## Database Integration

Trades are stored in the `trades` table with:

- `execution_mode` - simulation/paper/live
- `broker` - "mt5"
- `mt5_ticket` - MT5 order ticket number
- `entry_price`, `entry_time`
- `exit_price`, `exit_time`, `exit_reason`
- `lot_size`, `confidence`
- `stop_loss`, `take_profit`
- `pips`, `pnl_usd`, `is_winner`
- `status` - "open" or "closed"

## Connection Management

### Automatic Reconnection

The broker manager automatically reconnects on connection loss:

- Max 5 reconnection attempts
- 5-second delay between attempts
- Health checks every cycle

### Graceful Shutdown

On agent stop:

1. Close all open positions (configurable)
2. Disconnect from MT5 broker
3. Save final state

## Error Handling

### Trade Execution Errors

- **OrderRejectedError**: Order rejected by broker
- **InsufficientFundsError**: Not enough margin
- **BrokerError**: General broker error

All errors are logged and returned in `TradeResult.error`.

### Connection Errors

- **AuthenticationError**: Invalid MT5 credentials
- **ConnectionError**: Failed to connect to MT5

Failed trades do NOT crash the agent - it continues with next cycle.

### Critical Errors

If a trade executes but fails to store in database, a **CRITICAL** log is written:

```
TRADE EXECUTED BUT NOT RECORDED:
MT5 ticket=123456, symbol=EURUSD, side=buy, quantity=0.10, price=1.08450
```

## Position Sizing

### Kelly Criterion (enabled by default)

```python
kelly_fraction = (odds * win_prob - loss_prob) / odds
position_pct = kelly_fraction * kelly_multiplier
```

### Fixed Position Size

If `use_kelly_sizing=false`:

```python
position_pct = signal.position_size_pct
```

Both are capped at `max_position_size` (e.g., 0.1 = 10% of equity).

## Exit Management

### Triple Barrier Method

Positions are closed when:

1. **Take Profit** - Price reaches TP level
2. **Stop Loss** - Price reaches SL level
3. **Timeout** - Max holding period exceeded (default: 24 hours)

### Default Exit Parameters

- Stop Loss: 2% (2x ATR adjusted by confidence)
- Take Profit: 4% (1.5-2.5x SL based on confidence)
- Max Holding: 24 bars (1H timeframe)

## Testing

### Test in Simulation Mode

```bash
export AGENT_MODE=simulation
python -m src.agent.main
```

All signals are logged but not executed.

### Test in Paper Mode

```bash
export AGENT_MODE=paper
export AGENT_MT5_LOGIN=...
export AGENT_MT5_PASSWORD=...
export AGENT_MT5_SERVER="YourBroker-Demo"
python -m src.agent.main
```

Trades executed on demo account.

### Monitor Trades

Query database:

```sql
SELECT * FROM trades
WHERE execution_mode = 'paper'
ORDER BY entry_time DESC;
```

## Monitoring

### Check Agent Status

```bash
curl http://localhost:8001/api/v1/agent/status
```

Returns:

```json
{
  "status": "running",
  "mode": "paper",
  "cycle_count": 42,
  "broker_connected": true,
  "open_trades": 2,
  "broker_stats": {
    "connected": true,
    "broker_status": "connected",
    "reconnect_attempts": 0,
    "last_connection": "2026-01-22T10:30:00Z"
  }
}
```

### Check Open Positions

Via MT5 directly or query database:

```sql
SELECT id, symbol, direction, entry_price, entry_time,
       lot_size, stop_loss, take_profit, mt5_ticket
FROM trades
WHERE status = 'open'
  AND execution_mode = 'paper';
```

## Security

### Never Commit Credentials

- MT5 credentials stored in environment variables only
- `.env` files are gitignored
- Use `.env.example` as template

### Live Mode Protection

- Requires explicit `AGENT_MODE=live`
- Double-check credentials before running
- Start with small `max_position_size` (e.g., 0.01)
- Test thoroughly in paper mode first

## Troubleshooting

### "Broker not connected"

- Check MT5 credentials are correct
- Verify MT5 terminal is running
- Check network connection
- Review logs for connection errors

### "Order rejected"

- Check account margin/balance
- Verify symbol is enabled in MT5
- Check lot size is within broker limits
- Review MT5 terminal for error details

### "Position not found"

- Position may have been closed manually in MT5
- Check MT5 terminal for position status
- Verify mt5_ticket matches broker ticket

### Reconnection failures

- Check broker server status
- Verify credentials haven't changed
- Restart MT5 terminal
- Check logs for specific error codes

## Future Enhancements

Potential improvements for Phase 6+:

1. **Advanced Position Sizing**
   - Volatility-adjusted sizing
   - Drawdown-based scaling

2. **Multi-Symbol Support**
   - Track multiple pairs simultaneously
   - Portfolio-level risk management

3. **Partial Exits**
   - Scale out at multiple TP levels
   - Trail stops dynamically

4. **Enhanced Monitoring**
   - Real-time performance dashboard
   - Alert system for critical events
   - Trade analytics and reporting

## Support

For issues or questions:

1. Check logs in `logs/` directory
2. Review this documentation
3. Consult existing trading infrastructure in `src/trading/`
