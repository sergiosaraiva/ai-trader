---
name: frontend
description: Creates React functional components with loading, error, and data states using TailwindCSS styling and PropTypes.
version: 1.3.0
---

# Creating React Components

## Quick Reference

- Handle all states: loading, error, empty, data
- Use `animate-pulse` skeleton for loading
- Import icons from `lucide-react`
- Style with TailwindCSS classes
- Add PropTypes validation for all props
- Export as named function and default

## Decision Tree

```
Receives async data? → Accept loading, error, data props, render all 4 states
Needs icons? → Import from lucide-react, use size prop
Has interactivity? → Add event handlers, useCallback for memoization
PropTypes? → Always define after component, add defaultProps
```

## Pattern: Component with All States

```jsx
// Reference: frontend/src/components/ModelHighlights.jsx
import { AlertCircle } from 'lucide-react';
import PropTypes from 'prop-types';

export function ModelHighlights({ performance, loading, error }) {
  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-4"></div>
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-gray-700/50 rounded-lg p-4 mb-2">
            <div className="h-3 bg-gray-600 rounded w-2/3"></div>
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 border border-yellow-500/30">
        <div className="flex items-center gap-2 text-yellow-400">
          <AlertCircle size={20} />
          <span>Data unavailable</span>
        </div>
        <p className="text-gray-500 text-sm mt-2">{error}</p>
      </div>
    );
  }

  if (!performance || performance?.highlights?.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-2 text-gray-400">
          <AlertCircle size={20} />
          <span>No data available</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      {/* Render data */}
    </div>
  );
}

ModelHighlights.propTypes = {
  performance: PropTypes.shape({
    highlights: PropTypes.arrayOf(PropTypes.shape({
      title: PropTypes.string,
      value: PropTypes.string,
      status: PropTypes.string,
    })),
  }),
  loading: PropTypes.bool,
  error: PropTypes.string,
};

ModelHighlights.defaultProps = {
  performance: null,
  loading: false,
  error: null,
};

export default ModelHighlights;
```

## Pattern: Status-Based Styling

```jsx
const getStatusColor = (status) => {
  switch (status) {
    case 'excellent': return 'text-green-400';
    case 'good': return 'text-blue-400';
    case 'moderate': return 'text-yellow-400';
    case 'poor': return 'text-red-400';
    default: return 'text-gray-400';
  }
};
```

## Pattern: Safe Data Access

```jsx
<p className="text-xs text-gray-500">
  Metrics based on {performance?.metrics?.total_trades?.toLocaleString() ?? 'N/A'} trades
</p>
```

## Quality Checklist

- [ ] Handles loading, error, empty, data states
- [ ] Skeleton uses `animate-pulse`
- [ ] Icons from `lucide-react`
- [ ] PropTypes defined after component
- [ ] DefaultProps for optional props
- [ ] Unique `key` in all maps
- [ ] Optional chaining for nullable values

## Common Mistakes

| Wrong | Correct |
|-------|---------|
| `if (!data) return null` | `if (loading) return <Skeleton />; if (!data) return <Empty />` |
| `style={{ color: 'red' }}` | `className="text-red-400"` |
| `items.map(item => <div>` | `items.map((item, index) => <div key={item.id || index}>` |
| `performance.metrics.total` | `performance?.metrics?.total ?? 'N/A'` |

## Related Skills

- `creating-api-clients` - Fetch data for components
- `writing-vitest-tests` - Test component states
- `creating-chart-components` - For chart-heavy components

---
<!-- v1.3.0 | 2026-01-24 -->
