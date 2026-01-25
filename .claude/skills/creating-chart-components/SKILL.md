---
name: creating-chart-components
description: This skill should be used when the user asks to "create a chart", "add a graph", "implement data visualization", "build a performance chart", "add Recharts component". Creates React chart components using Recharts with useMemo optimization, memoized tooltips, and responsive containers.
version: 1.0.0
---

# Creating Chart Components with Recharts

## Quick Reference

- Use `useMemo` for expensive data transformations
- Use `memo` for tooltip and legend components
- Wrap charts in `ResponsiveContainer` for sizing
- Handle loading/error/empty states before rendering
- Add PropTypes for all props including callback types

## When to Use

- Creating performance/trading charts
- Visualizing time-series data
- Building data-heavy dashboard components
- Adding interactive tooltips and legends

## When NOT to Use

- Simple data display (use `frontend` skill)
- Static non-interactive visualizations
- Tables and lists (use basic components)

## Implementation Guide

```
Is data transformation expensive?
├─ Yes → Wrap in useMemo with proper dependencies
│   └─ Include all variables used in transformation
└─ No → Transform inline (small arrays)

Does chart have custom tooltip?
├─ Yes → Use memo() for tooltip component
│   └─ Handle null payload gracefully
└─ No → Use default Recharts tooltip

Chart sizing?
├─ Always → Use ResponsiveContainer
│   └─ Set width="100%" height={number}
└─ Never → Direct width/height props only

Custom components (tooltip, legend)?
├─ Yes → Extract as memoized components
│   └─ Add PropTypes for each
└─ No → Use Recharts defaults
```

## Examples

**Example 1: Memoized Custom Tooltip**

```jsx
// From: frontend/src/components/PerformanceChart.jsx:25-60
import { memo } from 'react';
import PropTypes from 'prop-types';

const CustomTooltip = memo(function CustomTooltip({ active, payload, profitUnit }) {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0]?.payload;
  if (!data) return null;

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-xl">
      <p className="text-gray-400 text-xs mb-2">{data.date}</p>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        <span className="text-gray-500">Daily P&L:</span>
        <span className={data.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
          {data.daily_pnl >= 0 ? '+' : ''}{data.daily_pnl.toFixed(1)} {profitUnit}
        </span>
        <span className="text-gray-500">Cumulative:</span>
        <span className="text-blue-400">
          {data.cumulative_pnl >= 0 ? '+' : ''}{data.cumulative_pnl.toFixed(1)} {profitUnit}
        </span>
        <span className="text-gray-500">Trades:</span>
        <span className="text-gray-300">{data.trades}</span>
      </div>
    </div>
  );
});

CustomTooltip.propTypes = {
  active: PropTypes.bool,
  payload: PropTypes.array,
  profitUnit: PropTypes.string,
};
```

**Explanation**: Use `memo` to prevent re-renders. Handle null/empty payload. Color based on positive/negative values. Grid layout for aligned data.

**Example 2: useMemo for Data Transformation**

```jsx
// From: frontend/src/components/PerformanceChart.jsx:68-141
import { useMemo } from 'react';

export function PerformanceChart({ trades, loading, error, assetMetadata }) {
  const profitUnit = getProfitUnitLabel(assetMetadata);

  const chartData = useMemo(() => {
    if (!trades || !Array.isArray(trades) || trades.length === 0) {
      return [];
    }

    // Filter to closed trades only
    const closedTrades = trades.filter(trade => trade.status === 'closed');

    if (closedTrades.length === 0) {
      return [];
    }

    // Group by date
    const dailyData = {};
    closedTrades.forEach(trade => {
      if (!trade.exit_time) return;

      const exitDate = new Date(trade.exit_time);
      const dateKey = exitDate.toISOString().split('T')[0];

      if (!dailyData[dateKey]) {
        dailyData[dateKey] = {
          date: dateKey,
          trades: 0,
          wins: 0,
          total_pnl: 0,
        };
      }

      dailyData[dateKey].trades += 1;
      dailyData[dateKey].total_pnl += trade.pips || 0;
      if (trade.is_winner) dailyData[dateKey].wins += 1;
    });

    // Sort and calculate cumulative
    const sortedData = Object.values(dailyData).sort(
      (a, b) => new Date(a.date) - new Date(b.date)
    );

    let cumulativePnl = 0;
    return sortedData.slice(-30).map(day => {
      cumulativePnl += day.total_pnl;
      return {
        date: new Date(day.date).toLocaleDateString([], { month: 'short', day: 'numeric' }),
        daily_pnl: day.total_pnl,
        cumulative_pnl: cumulativePnl,
        trades: day.trades,
        win_rate: day.trades > 0 ? ((day.wins / day.trades) * 100).toFixed(0) : 0,
      };
    });
  }, [trades]);  // Only recompute when trades change
```

**Explanation**: Early return for empty data. Group, sort, and calculate cumulative in single memo. Dependency array includes only `trades`.

**Example 3: Stats Calculation with useMemo**

```jsx
// From: frontend/src/components/PerformanceChart.jsx:143-165
const stats = useMemo(() => {
  if (chartData.length === 0) {
    return { totalDays: 0, profitableDays: 0, totalPnl: 0, maxDrawdown: 0 };
  }

  const total = chartData.length;
  const profitable = chartData.filter(d => d.daily_pnl > 0).length;
  const pnl = chartData[chartData.length - 1]?.cumulative_pnl || 0;

  // Calculate max drawdown
  let peak = 0;
  let maxDD = 0;
  chartData.forEach(d => {
    if (d.cumulative_pnl > peak) peak = d.cumulative_pnl;
    const dd = peak - d.cumulative_pnl;
    if (dd > maxDD) maxDD = dd;
  });

  return {
    totalDays: total,
    profitableDays: profitable,
    totalPnl: pnl,
    maxDrawdown: maxDD,
    winRate: total > 0 ? ((profitable / total) * 100).toFixed(0) : 0,
  };
}, [chartData]);  // Depends on transformed chartData
```

**Explanation**: Separate memo for derived calculations. Depends on `chartData` (output of first memo). Calculate drawdown inline.

**Example 4: ResponsiveContainer with ComposedChart**

```jsx
// From: frontend/src/components/PerformanceChart.jsx:200-280
import {
  ComposedChart,
  Bar,
  Cell,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  CartesianGrid,
} from 'recharts';

return (
  <div className="bg-gray-800 rounded-lg p-6">
    <h2 className="text-lg font-semibold text-gray-200 mb-4">
      30-Day Performance
    </h2>

    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
        <XAxis
          dataKey="date"
          tick={{ fill: '#9CA3AF', fontSize: 12 }}
          axisLine={{ stroke: '#4B5563' }}
        />
        <YAxis
          yAxisId="left"
          tick={{ fill: '#9CA3AF', fontSize: 12 }}
          axisLine={{ stroke: '#4B5563' }}
          tickFormatter={(value) => `${value > 0 ? '+' : ''}${value}`}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          tick={{ fill: '#9CA3AF', fontSize: 12 }}
          axisLine={{ stroke: '#4B5563' }}
        />

        <Tooltip content={<CustomTooltip profitUnit={profitUnit} />} />

        <ReferenceLine y={0} yAxisId="left" stroke="#6B7280" strokeDasharray="3 3" />

        <Bar dataKey="daily_pnl" yAxisId="left" radius={[4, 4, 0, 0]}>
          {chartData.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={entry.daily_pnl >= 0 ? '#10B981' : '#EF4444'}
            />
          ))}
        </Bar>

        <Line
          type="monotone"
          dataKey="cumulative_pnl"
          yAxisId="right"
          stroke="#3B82F6"
          strokeWidth={2}
          dot={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  </div>
);
```

**Explanation**: ResponsiveContainer with fixed height. Dual Y-axes (left for bars, right for line). Custom tooltip passed as component. Cell for per-bar coloring.

**Example 5: Complete Component with PropTypes**

```jsx
// From: frontend/src/components/PerformanceChart.jsx:350-366
PerformanceChart.propTypes = {
  trades: PropTypes.arrayOf(PropTypes.shape({
    exit_time: PropTypes.string,
    status: PropTypes.string,
    pips: PropTypes.number,
    is_winner: PropTypes.bool,
  })),
  loading: PropTypes.bool,
  error: PropTypes.string,
  assetMetadata: PropTypes.shape({
    profit_unit: PropTypes.string,
    asset_type: PropTypes.string,
  }),
};

PerformanceChart.defaultProps = {
  trades: [],
  loading: false,
  error: null,
  assetMetadata: null,
};

export default PerformanceChart;
```

**Explanation**: Shape for complex object props. Array of shapes for trade data. Default empty array for trades.

## Quality Checklist

- [ ] `useMemo` for expensive data transformations
- [ ] `memo` for tooltip and custom components
- [ ] Pattern matches `frontend/src/components/PerformanceChart.jsx`
- [ ] ResponsiveContainer wraps all charts
- [ ] Loading/error/empty states handled
- [ ] PropTypes for all props
- [ ] Dependency arrays correct
- [ ] Unique keys in Cell/Bar maps

## Common Mistakes

- **Missing useMemo**: Expensive calculations on every render
  - Wrong: Transform data inline without memo
  - Correct: Wrap in `useMemo` with dependencies

- **Wrong dependency array**: Stale or excessive recalculation
  - Wrong: `useMemo(() => transform(data), [])` (never updates)
  - Correct: `useMemo(() => transform(data), [data])`

- **Unmemoized tooltip**: Re-renders on every hover
  - Wrong: Inline function component
  - Correct: `memo(function CustomTooltip...)`

- **Hardcoded dimensions**: Chart doesn't resize
  - Wrong: `<ComposedChart width={800} height={300}>`
  - Correct: `<ResponsiveContainer width="100%" height={300}>`

## Validation

- [ ] Pattern confirmed in `frontend/src/components/PerformanceChart.jsx`
- [ ] Tests exist in `frontend/src/components/PerformanceChart.test.jsx`
- [ ] Chart renders in Dashboard
- [ ] Tooltip displays correct data

## Related Skills

- `frontend` - Basic React component patterns
- `writing-vitest-tests` - Test chart components

---

<!-- Skill Metadata
Version: 1.0.0
Created: 2026-01-23
Last Verified: 2026-01-23
Last Modified: 2026-01-23
Patterns From: .claude/discovery/codebase-patterns.md v3.0 (Pattern 4.5)
Lines: 280
-->
