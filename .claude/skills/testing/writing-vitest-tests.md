---
name: writing-vitest-tests
description: Writes Vitest tests with Testing Library for React components, testing loading, error, and data states. Use when testing frontend components, hooks, or UI interactions.
---

# Writing Vitest Tests

## Quick Reference

- Import `describe`, `it`, `expect` from `vitest`
- Import `render`, `screen` from `@testing-library/react`
- Test all component states: loading, error, empty, data
- Use `screen.getByText()` for assertions
- Use `document.querySelector()` for class-based checks

## When to Use

- Testing React components
- Verifying UI state rendering
- Testing prop variations
- Validating conditional rendering

## When NOT to Use

- API endpoint tests (use pytest)
- E2E tests (use Playwright/Cypress)
- Backend service tests (use pytest)

## Implementation Guide

```
Is component data-driven?
├─ Yes → Test all 4 states
│   └─ loading=true, error=message, data=null, data=valid
└─ No → Test render and interactions

Does component have conditional sections?
├─ Yes → Test with and without condition
│   └─ e.g., timeframe_signals present/absent
└─ No → Test main render

Does component accept multiple value types?
├─ Yes → Test each type variant
│   └─ e.g., signal='BUY' and signal=1
└─ No → Test primary type
```

## Examples

**Example 1: Test File Structure**

```jsx
// From: frontend/src/components/PredictionCard.test.jsx:1-4
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PredictionCard } from './PredictionCard';
```

**Explanation**: Import Vitest functions. Import Testing Library render and screen. Import component to test.

**Example 2: Loading State Test**

```jsx
// From: frontend/src/components/PredictionCard.test.jsx:5-11
describe('PredictionCard', () => {
  it('renders loading state', () => {
    render(<PredictionCard loading={true} />);
    // Check for skeleton loader (animate-pulse class)
    const skeleton = document.querySelector('.animate-pulse');
    expect(skeleton).toBeInTheDocument();
  });
```

**Explanation**: Use `document.querySelector` for class-based checks. Verify skeleton loader renders during loading.

**Example 3: Error State Test**

```jsx
// From: frontend/src/components/PredictionCard.test.jsx:13-17
  it('renders error state', () => {
    render(<PredictionCard error="Test error message" />);
    expect(screen.getByText('Error loading prediction')).toBeInTheDocument();
    expect(screen.getByText('Test error message')).toBeInTheDocument();
  });
```

**Explanation**: Pass error prop. Verify both error label and specific message render.

**Example 4: Empty/Null Data Test**

```jsx
// From: frontend/src/components/PredictionCard.test.jsx:19-22
  it('renders no prediction state', () => {
    render(<PredictionCard prediction={null} />);
    expect(screen.getByText('No prediction available')).toBeInTheDocument();
  });
```

**Explanation**: Pass null data. Verify empty state message renders.

**Example 5: Data State Tests**

```jsx
// From: frontend/src/components/PredictionCard.test.jsx:24-50
  it('renders BUY prediction correctly', () => {
    const prediction = {
      signal: 'BUY',
      confidence: 0.72,
      current_price: 1.08543,
      symbol: 'EURUSD',
      timestamp: new Date().toISOString(),
    };
    render(<PredictionCard prediction={prediction} />);

    expect(screen.getByText('BUY')).toBeInTheDocument();
    expect(screen.getByText('72.0%')).toBeInTheDocument();
    expect(screen.getByText('@ 1.08543')).toBeInTheDocument();
  });

  it('renders SELL prediction correctly', () => {
    const prediction = {
      signal: 'SELL',
      confidence: 0.65,
      current_price: 1.08123,
      timestamp: new Date().toISOString(),
    };
    render(<PredictionCard prediction={prediction} />);

    expect(screen.getByText('SELL')).toBeInTheDocument();
    expect(screen.getByText('65.0%')).toBeInTheDocument();
  });
```

**Explanation**: Create realistic test data. Test BUY and SELL variations. Verify computed values (72.0% from 0.72).

**Example 6: Numeric Signal Variant Test**

```jsx
// From: frontend/src/components/PredictionCard.test.jsx:52-63
  it('renders numeric signal values correctly', () => {
    const prediction = {
      signal: 1,
      confidence: 0.80,
      current_price: 1.09000,
      timestamp: new Date().toISOString(),
    };
    render(<PredictionCard prediction={prediction} />);

    expect(screen.getByText('BUY')).toBeInTheDocument();
    expect(screen.getByText('80.0%')).toBeInTheDocument();
  });
```

**Explanation**: Test numeric signal (1 = BUY, -1 = SELL). Verify component handles both string and numeric formats.

**Example 7: Conditional Section Test**

```jsx
// From: frontend/src/components/PredictionCard.test.jsx:65-83
  it('renders timeframe signals breakdown', () => {
    const prediction = {
      signal: 'BUY',
      confidence: 0.75,
      current_price: 1.08500,
      timestamp: new Date().toISOString(),
      timeframe_signals: {
        '1H': { signal: 'BUY', confidence: 0.80 },
        '4H': { signal: 'BUY', confidence: 0.70 },
        'D': { signal: 'HOLD', confidence: 0.55 },
      },
    };
    render(<PredictionCard prediction={prediction} />);

    expect(screen.getByText('Timeframe Breakdown')).toBeInTheDocument();
    expect(screen.getByText('1H')).toBeInTheDocument();
    expect(screen.getByText('4H')).toBeInTheDocument();
    expect(screen.getByText('D')).toBeInTheDocument();
  });
});
```

**Explanation**: Test optional section with nested data. Verify section title and all timeframe labels render.

## Quality Checklist

- [ ] Test loading, error, empty, and data states
- [ ] Pattern matches `frontend/src/components/PredictionCard.test.jsx:5-22`
- [ ] Use `screen.getByText()` for text assertions
- [ ] Use `document.querySelector()` for class checks
- [ ] Test prop variations (BUY/SELL, string/number)
- [ ] Test conditional sections present/absent
- [ ] Descriptive test names

## Common Mistakes

- **Missing state tests**: Only test happy path
  - Wrong: Only test with valid data
  - Correct: Test loading, error, empty, and data states

- **Hardcoded test data**: Tests break on format changes
  - Wrong: `expect(screen.getByText('72%'))`
  - Correct: `expect(screen.getByText('72.0%'))` matching exact format

- **Missing imports**: Test fails to run
  - Wrong: Forget `@testing-library/react` imports
  - Correct: Import render, screen at top of file

## Validation

- [ ] Pattern confirmed in `frontend/src/components/PredictionCard.test.jsx:5-22`
- [ ] Tests pass with `cd frontend && npm test`
- [ ] All component states covered

## Related Skills

- `creating-react-components` - Components being tested
- `writing-pytest-tests` - Backend API tests
