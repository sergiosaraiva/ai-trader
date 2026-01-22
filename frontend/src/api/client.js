/**
 * API Client for AI Assets Trader Backend
 */

// Use environment variable if set, otherwise use relative URL (for nginx proxy)
const API_BASE = import.meta.env.VITE_API_URL || '/api/v1';

class APIError extends Error {
  constructor(message, status, data) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.data = data;
  }
}

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

/**
 * API endpoints - mapped to backend /api/v1/* routes
 */
export const api = {
  // Health check
  health: () => fetch('/health').then(r => r.json()),

  // Predictions
  getPrediction: () => request('/predictions/latest'),
  getPredictionHistory: (limit = 50) => request(`/predictions/history?limit=${limit}`),
  generatePrediction: () => request('/predictions/generate', { method: 'POST' }),
  getPredictionStats: () => request('/predictions/stats'),
  getExplanation: (forceRefresh = false) =>
    request(`/predictions/explanation${forceRefresh ? '?force_refresh=true' : ''}`),

  // Market data / Candles
  getCandles: (symbol, timeframe = '1H', limit = 24) =>
    request(`/market/candles?symbol=${symbol}&timeframe=${timeframe}&limit=${limit}`),
  getCurrentPrice: () => request('/market/current'),
  getVix: () => request('/market/vix'),

  // Performance metrics
  getPerformance: () => request('/trading/performance'),
  getModelPerformance: () => request('/model/performance'),
  getTradingPerformance: () => request('/trading/performance'),

  // Trading signals (using prediction history)
  getSignals: (limit = 20) => request(`/predictions/history?limit=${limit}`),

  // Model status
  getModelStatus: () => request('/models/status'),

  // Pipeline status
  getPipelineStatus: () => request('/pipeline/status'),
  runPipeline: () => request('/pipeline/run', { method: 'POST' }),
  runPipelineSync: () => request('/pipeline/run-sync', { method: 'POST' }),

  // Pipeline data by timeframe
  getPipelineData: (timeframe = '1H', limit = 100) =>
    request(`/pipeline/data/${timeframe}?limit=${limit}`),

  // Trading
  getTradingStatus: () => request('/trading/status'),
  getTradingHistory: (limit = 50) => request(`/trading/history?limit=${limit}`),
  getEquityCurve: () => request('/trading/equity-curve'),
  closePosition: (positionId) => request('/trading/close-position', {
    method: 'POST',
    body: JSON.stringify({ position_id: positionId }),
  }),

  // Positions
  getPositions: () => request('/positions'),

  // Risk metrics
  getRiskMetrics: () => request('/risk/metrics'),

  // Backtest data for What If Calculator
  getBacktestPeriods: () => request('/trading/backtest-periods'),
};

export { APIError };
export default api;
