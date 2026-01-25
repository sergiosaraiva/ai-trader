# PerformanceChart Integration Guide

## Overview
The `PerformanceChart` component displays 30-day daily trading performance with cumulative P&L visualization using Recharts.

## Component Location
```
frontend/src/components/PerformanceChart.jsx
frontend/src/components/PerformanceChart.test.jsx (27 passing tests)
```

## Features
- **Daily P&L bars**: Green for profitable days, red for losing days
- **Cumulative line**: Blue line showing running total
- **Statistics footer**: Best day, worst day, max drawdown
- **Header summary**: Total P&L, day count, profitable day percentage
- **Responsive design**: Matches existing dark theme
- **Loading/error states**: Consistent with other components

## Props Interface

```javascript
{
  trades: Array,          // Array of trade objects (required)
  loading: Boolean,       // Loading state
  error: String,          // Error message
  assetMetadata: Object   // Asset metadata for formatting
}
```

## Trade Object Structure

```javascript
{
  id: 1,
  symbol: "EURUSD",
  direction: "long",
  entry_price: 1.08500,
  entry_time: "2024-01-15T14:01:00",
  exit_price: 1.08750,
  exit_time: "2024-01-15T16:30:00",  // Required for grouping by date
  exit_reason: "tp",
  lot_size: 0.1,
  pips: 25.0,
  pnl_usd: 250.0,
  is_winner: true,
  status: "closed",  // Only "closed" trades are included
}
```

## Integration Example

### 1. Import the Component

```javascript
import { PerformanceChart } from './components/PerformanceChart';
```

### 2. Add to Dashboard

```javascript
export function Dashboard() {
  const [trades, setTrades] = useState([]);
  const [tradesLoading, setTradesLoading] = useState(true);
  const [tradesError, setTradesError] = useState(null);

  // Fetch trades from API
  useEffect(() => {
    const fetchTrades = async () => {
      try {
        setTradesLoading(true);
        const response = await apiClient.get('/api/v1/trading/history');
        setTrades(response.trades || []);
        setTradesError(null);
      } catch (err) {
        setTradesError(err.message);
      } finally {
        setTradesLoading(false);
      }
    };

    fetchTrades();
    const interval = setInterval(fetchTrades, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Other components */}

        {/* Add Performance Chart */}
        <PerformanceChart
          trades={trades}
          loading={tradesLoading}
          error={tradesError}
          assetMetadata={{ type: 'forex', symbol: 'EURUSD' }}
        />
      </div>
    </div>
  );
}
```

### 3. API Endpoint

The component expects data from the `/api/v1/trading/history` endpoint:

```javascript
// Backend response format
{
  "trades": [
    {
      "id": 1,
      "symbol": "EURUSD",
      "direction": "long",
      "entry_price": 1.08500,
      "entry_time": "2024-01-15T14:01:00",
      "exit_price": 1.08750,
      "exit_time": "2024-01-15T16:30:00",
      "exit_reason": "tp",
      "lot_size": 0.1,
      "pips": 25.0,
      "pnl_usd": 250.0,
      "is_winner": true,
      "status": "closed"
    }
  ]
}
```

## Data Processing Logic

The component automatically:
1. Filters to `status === 'closed'` trades only
2. Groups trades by date (extracted from `exit_time`)
3. Calculates daily totals: pips, trade count, wins/losses
4. Computes cumulative running total
5. Limits display to last 30 days
6. Calculates statistics: best day, worst day, max drawdown

## Styling Details

- **Card background**: `bg-gray-800` with `card-hover` effect
- **Chart height**: `300px`
- **Colors**:
  - Profitable bars: `#22c55e` (green-400)
  - Losing bars: `#ef4444` (red-400)
  - Cumulative line: `#3b82f6` (blue-400)
  - Grid: `#374151` (gray-700)
- **Loading state**: Animated pulse skeleton
- **Error state**: Red error message

## Testing

Run the test suite:

```bash
cd frontend
npm test -- PerformanceChart.test.jsx
```

**Test coverage:**
- Loading states
- Error states
- Empty states
- Chart rendering
- Statistics calculations
- Data processing
- Profit unit labeling
- Day counting
- 27 total tests, all passing

## Dependencies

Already included in project:
- `recharts` - Chart library
- `lucide-react` - Icons
- React, TailwindCSS

## Performance Considerations

- Uses `useMemo` for expensive data calculations
- Only processes closed trades
- Limits to 30 days for performance
- Chart animations disabled for better performance

## Accessibility

- Semantic HTML structure
- Color contrast meets WCAG AA standards
- Tooltip provides detailed information on hover
- Statistics available without interaction

## Example Output

```
┌─────────────────────────────────────────────┐
│ 30-Day Performance                          │
│ +8,693 pips  📈 30 days • 18 profitable (60%)│
├─────────────────────────────────────────────┤
│  [Chart: Bars + Line]                       │
│                                             │
│  █                                          │
│  █         █                                │
│  █    █    █    ─────  Cumulative          │
│  █    █    █   /                           │
├─────────────────────────────────────────────┤
│ Best Day: +125.0 pips                       │
│ Worst Day: -45.0 pips                       │
│ Max Drawdown: -85.0 pips                    │
└─────────────────────────────────────────────┘
```

## Notes

- Component is production-ready
- Follows all existing patterns in the codebase
- No additional dependencies required
- Compatible with current API structure
- Fully tested and linted
