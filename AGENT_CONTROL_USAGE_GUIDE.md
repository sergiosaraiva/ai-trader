# Agent Control Panel Usage Guide

## Quick Reference

### Component Import
```jsx
import { AgentControlPanel } from './components/AgentControlPanel';
```

### Basic Usage
```jsx
<AgentControlPanel
  status={agentStatus}
  safety={agentSafety}
  loading={agentLoading}
  onRefresh={refetchAgent}
/>
```

## Props API

### `status` (object | null)
Agent status information from API endpoint `/api/v1/agent/status`

**Shape**:
```javascript
{
  status: 'running' | 'paused' | 'stopped' | 'error',
  mode: 'simulation' | 'paper' | 'live',
  cycle_count: number,
  last_cycle_at: string (ISO timestamp),
  open_positions: number,
  uptime_seconds: number,
  last_prediction: {
    signal: 'BUY' | 'SELL' | 'HOLD',
    confidence: number
  }
}
```

**Example**:
```javascript
{
  status: 'running',
  mode: 'simulation',
  cycle_count: 147,
  last_cycle_at: '2026-01-25T10:30:00Z',
  open_positions: 2,
  uptime_seconds: 3600
}
```

### `safety` (object | null)
Safety systems information from API endpoint `/api/v1/agent/status`

**Shape**:
```javascript
{
  is_safe_to_trade: boolean,
  kill_switch: {
    is_active: boolean,
    reason?: string,
    triggered_at?: string
  },
  circuit_breakers: {
    overall_state: 'active' | 'tripped',
    can_trade: boolean,
    active_breakers?: string[]
  },
  daily_metrics: {
    trades: number,
    loss_pct: number
  },
  account_metrics: {
    current_equity: number,
    peak_equity: number,
    drawdown_pct: number
  }
}
```

**Example**:
```javascript
{
  is_safe_to_trade: true,
  kill_switch: {
    is_active: false
  },
  circuit_breakers: {
    overall_state: 'active',
    can_trade: true
  }
}
```

### `loading` (boolean, required)
Whether data is currently being fetched

### `onRefresh` (function, required)
Callback to trigger data refresh

**Signature**: `() => void`

## Status Card Details

### 1. Agent Status Card
- **Icon**: Activity (pulse when running)
- **Label**: "Status"
- **Values**:
  - `RUNNING` (green, pulsing)
  - `PAUSED` (yellow)
  - `STOPPED` (gray)
  - `ERROR` (red)
  - `KILLED` (red, pulsing) - shown when kill switch active

### 2. Mode Card
- **Icon**: Target
- **Label**: "Mode"
- **Values**:
  - `SIMULATION` (blue) - safe testing mode
  - `PAPER` (yellow) - paper trading with fake money
  - `LIVE` (red) - real money trading

### 3. Cycles Card
- **Icon**: BarChart3
- **Label**: "Cycles"
- **Value**: Number of completed cycles (formatted with commas)
- **Color**: Blue

### 4. Kill Switch Card
- **Icon**: AlertTriangle
- **Label**: "Kill Switch"
- **Values**:
  - `ACTIVE` (red, pulsing) - trading halted
  - `Inactive` (green) - normal operation

### 5. Circuit Breaker Card
- **Icon**: Shield
- **Label**: "Breaker"
- **Values**:
  - `OK` (green) - all breakers normal
  - `TRIPPED` (red) - one or more breakers triggered

### 6. Open Positions Card
- **Icon**: TrendingUp
- **Label**: "Positions"
- **Value**: Number of open positions
- **Colors**:
  - Blue when > 0
  - Gray when 0

## Control Buttons

### Start Button
- **Visible**: When agent is stopped and kill switch inactive
- **Action**: Starts agent with current configuration
- **Confirmation**: Required for live mode
- **Icon**: Play

### Pause Button
- **Visible**: When agent is running
- **Action**: Pauses agent (keeps positions open)
- **Icon**: Pause

### Resume Button
- **Visible**: When agent is paused
- **Action**: Resumes agent operation
- **Icon**: PlayCircle

### Stop Button
- **Visible**: When agent is running or paused
- **Action**: Stops agent gracefully (keeps positions open)
- **Icon**: Square

### Stop & Close Button
- **Visible**: When agent is running or paused
- **Action**: Stops agent and closes all positions
- **Confirmation**: Required
- **Icon**: Square
- **Color**: Orange

### Kill Switch Button
- **Visible**: When agent is running or paused (not stopped)
- **Action**: Immediately halts all trading and closes positions
- **Confirmation**: Requires reason input
- **Icon**: AlertTriangle
- **Color**: Red (with border)

### Config Button
- **Visible**: Always
- **Action**: Toggles configuration panel
- **Icon**: Settings

## Configuration Panel

### Mode Selector
- **Options**: Simulation, Paper Trading, Live Trading
- **Editable**: Only when agent is stopped
- **Default**: Simulation

### Confidence Threshold Slider
- **Range**: 50% - 85%
- **Step**: 5%
- **Default**: 70%
- **Description**: Minimum confidence required for trade execution

### Cycle Interval Input
- **Range**: 60 - 3600 seconds
- **Step**: 60 seconds
- **Default**: 300 seconds (5 minutes)
- **Description**: Time between agent cycles

### Update Config Button
- **Visible**: Only when agent is running
- **Action**: Updates configuration without restarting agent
- **Note**: Some settings require restart to take effect

## State Handling

### Loading State
Shows animated skeleton with 6 placeholder cards

### Agent Not Initialized (404)
Shows empty state with:
- Message: "Agent not initialized"
- Helper: "Start the agent to begin trading"
- Start button

### Error State
Shows error banner at top with error message

### Normal Operation
Shows full status grid with all metrics

## Example Integration

```jsx
import { useAgent } from '../hooks/useAgent';
import { AgentControlPanel } from './components/AgentControlPanel';

function Dashboard() {
  const AGENT_POLL_INTERVAL = 5000; // 5 seconds

  const {
    status: agentStatus,
    safety: agentSafety,
    loading: agentLoading,
    refetch: refetchAgent,
  } = useAgent(AGENT_POLL_INTERVAL);

  return (
    <div>
      <AgentControlPanel
        status={agentStatus}
        safety={agentSafety}
        loading={agentLoading}
        onRefresh={refetchAgent}
      />
    </div>
  );
}
```

## Color Palette Reference

| Status | Color Class | Hex | Use Case |
|--------|------------|-----|----------|
| Success/Running | `text-green-400` | #4ade80 | Running status, active systems |
| Warning/Paused | `text-yellow-400` | #facc15 | Paused status, paper mode |
| Info | `text-blue-400` | #60a5fa | Cycles, simulation mode, positions |
| Error/Danger | `text-red-400` | #f87171 | Stopped, kill switch, live mode |
| Neutral | `text-gray-400` | #9ca3af | Inactive states |

## Accessibility Features

- All buttons have proper `aria-label` attributes
- Icon colors convey status semantically
- Hover states on interactive elements
- Loading states announce changes
- Error messages are visible and descriptive
- Keyboard navigation supported

## Performance Considerations

- Component re-renders only when props change
- Timeout cleanup on unmount prevents memory leaks
- Hover effects use CSS transitions (GPU accelerated)
- Status polling handled by parent (not in component)

## Best Practices

1. **Poll Interval**: Recommended 5 seconds for agent status
2. **Error Handling**: Always handle 404 gracefully (agent not started)
3. **Confirmations**: Always confirm destructive actions (stop with close, kill switch)
4. **Input Sanitization**: User input automatically sanitized (kill switch reason)
5. **Loading States**: Show loading during async operations

## Troubleshooting

### Status Not Updating
- Check polling interval in parent component
- Verify API endpoint `/api/v1/agent/status` is accessible
- Check browser console for fetch errors

### Buttons Not Working
- Check `onRefresh` callback is provided
- Verify API endpoints are configured correctly
- Check for CORS issues in browser console

### Kill Switch Not Activating
- Ensure reason is provided (non-empty)
- Check API endpoint `/api/v1/agent/kill-switch` exists
- Verify authentication/authorization

### Config Updates Not Applied
- Some settings require agent restart
- Check API response for errors
- Verify mode changes only work when stopped

---

**Version**: 1.0.0
**Last Updated**: 2026-01-25
**Compatibility**: React 19, Vite 7, TailwindCSS 4
