---
name: creating-react-components
description: Creates React functional components with loading, error, and data states using TailwindCSS styling. Use when building UI cards, data displays, or dashboard widgets for the React frontend.
---

# Creating React Components

## Quick Reference

- Handle all states: loading, error, empty, data
- Use skeleton loaders (`animate-pulse`) for loading state
- Import icons from `lucide-react`
- Style with TailwindCSS utility classes
- Export as named function for tree-shaking

## When to Use

- Creating dashboard cards or widgets
- Building data display components
- Implementing async data visualization
- Adding new UI elements to the frontend

## When NOT to Use

- Page-level layouts (use Dashboard pattern)
- Utility functions (use hooks)
- API calls (use api/client.js)

## Implementation Guide

```
Does component receive async data?
├─ Yes → Accept loading, error, data props
│   └─ Render all 4 states (loading/error/empty/data)
└─ No → Render data directly

Does component need icons?
├─ Yes → Import from lucide-react
│   └─ Use size prop for consistent sizing
└─ No → Skip icon imports

Does component have interactive elements?
├─ Yes → Add event handlers and state
│   └─ Use useCallback for memoization
└─ No → Keep as pure display component
```

## Examples

**Example 1: Component with All States**

```jsx
// From: frontend/src/components/PredictionCard.jsx:1-35
import { TrendingUp, TrendingDown, Minus, AlertCircle, Clock } from 'lucide-react';

/**
 * PredictionCard - Displays the current trading prediction
 */
export function PredictionCard({ prediction, loading, error }) {
  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="h-16 bg-gray-700 rounded mb-4"></div>
        <div className="h-4 bg-gray-700 rounded w-2/3"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 border border-red-500/30">
        <div className="flex items-center gap-2 text-red-400">
          <AlertCircle size={20} />
          <span>Error loading prediction</span>
        </div>
        <p className="text-gray-500 text-sm mt-2">{error}</p>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <p className="text-gray-500">No prediction available</p>
      </div>
    );
  }

  // ... render data state
}
```

**Explanation**: Loading shows skeleton with `animate-pulse`. Error shows red border and icon. Empty shows message. All share same base styling.

**Example 2: Data State Rendering**

```jsx
// From: frontend/src/components/PredictionCard.jsx:37-95
const { signal, confidence, current_price, symbol, timestamp, timeframe_signals } = prediction;

const getSignalColor = (sig) => {
  if (sig === 'BUY' || sig === 1) return 'text-green-400';
  if (sig === 'SELL' || sig === -1) return 'text-red-400';
  return 'text-gray-400';
};

const getSignalIcon = (sig) => {
  if (sig === 'BUY' || sig === 1) return <TrendingUp size={32} />;
  if (sig === 'SELL' || sig === -1) return <TrendingDown size={32} />;
  return <Minus size={32} />;
};

return (
  <div className="bg-gray-800 rounded-lg p-6 card-hover">
    <div className="flex justify-between items-start mb-4">
      <div>
        <h2 className="text-lg font-semibold text-gray-300">Current Prediction</h2>
        <p className="text-sm text-gray-500">{symbol || 'EUR/USD'}</p>
      </div>
      <div className="flex items-center gap-1 text-gray-500 text-sm">
        <Clock size={14} />
        <span>{formatTime(timestamp)}</span>
      </div>
    </div>

    {/* Main Signal */}
    <div className="flex items-center justify-center gap-4 py-6">
      <div className={`${getSignalColor(signal)}`}>
        {getSignalIcon(signal)}
      </div>
      <div className="text-center">
        <span className={`text-4xl font-bold ${getSignalColor(signal)}`}>
          {getSignalText(signal)}
        </span>
        <p className="text-gray-500 text-sm mt-1">
          @ {current_price?.toFixed(5) || 'N/A'}
        </p>
      </div>
    </div>
  </div>
);
```

**Explanation**: Helper functions for dynamic styling. Destructure props at top. Use optional chaining for nullable values. Responsive grid layout.

**Example 3: Progress Bar Component**

```jsx
// From: frontend/src/components/PredictionCard.jsx:97-111
{/* Confidence Bar */}
<div className="mt-4">
  <div className="flex justify-between items-center mb-2">
    <span className="text-sm text-gray-400">Confidence</span>
    <span className="text-sm font-medium text-gray-300">
      {((confidence || 0) * 100).toFixed(1)}%
    </span>
  </div>
  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
    <div
      className={`h-full ${getConfidenceColor(confidence)} transition-all duration-500`}
      style={{ width: `${(confidence || 0) * 100}%` }}
    />
  </div>
</div>
```

**Explanation**: Label and value in flex row. Background bar with rounded corners. Inner bar with dynamic width and transition. Color varies by value.

**Example 4: Conditional Section Rendering**

```jsx
// From: frontend/src/components/PredictionCard.jsx:113-133
{/* Timeframe Breakdown */}
{timeframe_signals && Object.keys(timeframe_signals).length > 0 && (
  <div className="mt-6 pt-4 border-t border-gray-700">
    <h3 className="text-sm text-gray-400 mb-3">Timeframe Breakdown</h3>
    <div className="grid grid-cols-3 gap-3">
      {Object.entries(timeframe_signals).map(([tf, data]) => (
        <div key={tf} className="bg-gray-700/50 rounded p-3 text-center">
          <span className="text-xs text-gray-500 block mb-1">{tf}</span>
          <span className={`text-sm font-medium ${getSignalColor(data.signal || data)}`}>
            {getSignalText(data.signal || data)}
          </span>
          {data.confidence && (
            <span className="text-xs text-gray-500 block mt-1">
              {(data.confidence * 100).toFixed(0)}%
            </span>
          )}
        </div>
      ))}
    </div>
  </div>
)}
```

**Explanation**: Conditional rendering with `&&`. Border top for section divider. Grid layout for items. Map with unique `key` prop.

**Example 5: Export Pattern**

```jsx
// From: frontend/src/components/PredictionCard.jsx:136-138
export function PredictionCard({ prediction, loading, error }) {
  // ... component body
}

export default PredictionCard;
```

**Explanation**: Named export for tree-shaking. Default export for convenience. Both patterns supported.

## Quality Checklist

- [ ] Handles loading, error, empty, and data states
- [ ] Skeleton loader uses `animate-pulse` class
- [ ] Pattern matches `frontend/src/components/PredictionCard.jsx:6-35`
- [ ] Icons imported from `lucide-react`
- [ ] TailwindCSS classes for all styling
- [ ] Named export function
- [ ] Props destructured with defaults

## Common Mistakes

- **Missing loading state**: UI shows nothing during fetch
  - Wrong: `if (!data) return null;`
  - Correct: `if (loading) return <Skeleton />; if (!data) return <Empty />;`

- **Inline style instead of Tailwind**: Inconsistent styling
  - Wrong: `style={{ color: 'red' }}`
  - Correct: `className="text-red-400"`

- **Missing key in map**: React warning
  - Wrong: `items.map(item => <div>...`
  - Correct: `items.map(item => <div key={item.id}>...`

## Validation

- [ ] Pattern confirmed in `frontend/src/components/PredictionCard.jsx:6-35`
- [ ] Tests exist in `frontend/src/components/PredictionCard.test.jsx`
- [ ] Renders in Dashboard component

## Related Skills

- `creating-api-clients` - Fetch data for components
- `writing-vitest-tests` - Test component states
