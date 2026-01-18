---
name: creating-api-clients
description: This skill should be used when the user asks to "add an API call", "create an API client", "fetch data from backend", "handle API errors". Creates centralized API client modules with custom error handling, typed endpoints, and consistent request patterns.
version: 1.1.0
---

# Creating API Clients

## Quick Reference

- Use `fetch()` with JSON headers
- Create custom `APIError` class with status and data
- Export endpoints as object methods
- Handle network errors separately from HTTP errors
- Use relative URLs with `/api` base

## When to Use

- Adding new API endpoint methods
- Creating error handling for HTTP requests
- Centralizing API configuration
- Implementing retry or caching logic

## When NOT to Use

- Direct fetch in components (use client methods)
- WebSocket connections (different pattern)
- Server-side API calls (use services)

## Implementation Guide

```
Is this a new endpoint?
├─ Yes → Add method to api object
│   └─ Use request() helper for JSON endpoints
│   └─ Use fetch() directly for non-JSON
└─ No → Modify existing method

Does endpoint have query parameters?
├─ Yes → Build URL with template literal
│   └─ Provide sensible defaults
└─ No → Use plain endpoint path

Is endpoint for health check?
├─ Yes → Use fetch() directly (may return non-JSON)
└─ No → Use request() helper
```

## Examples

**Example 1: API Client Module Structure**

```javascript
// From: frontend/src/api/client.js:1-14
/**
 * API Client for AI Trader Backend
 */

const API_BASE = '/api';

class APIError extends Error {
  constructor(message, status, data) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.data = data;
  }
}
```

**Explanation**: Module docstring, base URL constant, custom error class with HTTP status and response data.

**Example 2: Request Helper Function**

```javascript
// From: frontend/src/api/client.js:16-46
async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;

  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, config);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        errorData.detail || `HTTP error ${response.status}`,
        response.status,
        errorData
      );
    }

    return await response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError(error.message || 'Network error', 0, null);
  }
}
```

**Explanation**: Builds full URL from base + endpoint. Merges headers with defaults. Parses error response for FastAPI `detail` field. Re-throws APIError, wraps network errors.

**Example 3: API Endpoint Methods**

```javascript
// From: frontend/src/api/client.js:48-78
/**
 * API endpoints
 */
export const api = {
  // Health check
  health: () => fetch('/health').then(r => r.json()),

  // Predictions
  getPrediction: () => request('/predict'),
  getPredictionHistory: (limit = 50) => request(`/predictions?limit=${limit}`),

  // Price data
  getCandles: (symbol = 'EURUSD', timeframe = '1H', count = 24) =>
    request(`/candles?symbol=${symbol}&timeframe=${timeframe}&count=${count}`),

  // Performance metrics
  getPerformance: () => request('/performance'),

  // Trading signals
  getSignals: (limit = 20) => request(`/signals?limit=${limit}`),

  // Model status
  getModelStatus: () => request('/model/status'),

  // Pipeline status
  getPipelineStatus: () => request('/pipeline/status'),

  // Pipeline data by timeframe
  getPipelineData: (timeframe = '1H', limit = 100) =>
    request(`/pipeline/data/${timeframe}?limit=${limit}`),
};

export { APIError };
export default api;
```

**Explanation**: Health uses fetch directly (may not return JSON). Other endpoints use request helper. Query params with defaults. Named and default exports.

**Example 4: Usage in Components**

```jsx
// Usage pattern in Dashboard component
import { api } from '../api/client';
import { usePolling } from '../hooks/usePolling';

const { data, loading, error } = usePolling(
  useCallback(() => api.getPrediction(), []),
  30000  // Poll every 30 seconds
);
```

**Explanation**: Import api object. Use with custom polling hook. Wrap in useCallback for memoization.

**Example 5: Error Handling in Components**

```jsx
// Error handling pattern
try {
  const data = await api.getPrediction();
  setPrediction(data);
} catch (error) {
  if (error.status === 503) {
    setError('Service unavailable. Please try again later.');
  } else {
    setError(error.message);
  }
}
```

**Explanation**: Catch APIError, check status for specific handling, fall back to message for generic error.

## Quality Checklist

- [ ] Custom APIError class with status and data
- [ ] Request helper handles JSON parsing
- [ ] Pattern matches `frontend/src/api/client.js:16-46`
- [ ] Default values for optional parameters
- [ ] Health endpoint uses fetch directly
- [ ] Both named and default exports

## Common Mistakes

- **Missing error parsing**: Lose FastAPI detail message
  - Wrong: `throw new Error('Request failed');`
  - Correct: `throw new APIError(errorData.detail || 'Request failed', ...)`

- **Hardcoded URLs**: Can't configure for different environments
  - Wrong: `fetch('http://localhost:8001/api/predict')`
  - Correct: `fetch(`${API_BASE}/predict`)`

- **No network error handling**: Crashes on connection failure
  - Wrong: Only handle response errors
  - Correct: Wrap in try/catch, handle both APIError and other errors

## Validation

- [ ] Pattern confirmed in `frontend/src/api/client.js:16-46`
- [ ] Endpoints match backend routes in `src/api/routes/`
- [ ] Used in components via `usePolling` hook

## Related Skills

- `creating-react-components` - Use API client in components
- `creating-fastapi-endpoints` - Backend endpoints that client calls
