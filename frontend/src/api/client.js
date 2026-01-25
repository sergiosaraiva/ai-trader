/**
 * API Client for AI Trading Agent Backend
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

  // What-If Performance simulation (30-day historical simulation)
  getWhatIfPerformance: (days = 30, confidenceThreshold = 0.70) =>
    request(`/trading/whatif-performance?days=${days}&confidence_threshold=${confidenceThreshold}`),

  // Agent control
  startAgent: (config) => request('/agent/start', {
    method: 'POST',
    body: JSON.stringify(config),
  }),
  stopAgent: (options) => request('/agent/stop', {
    method: 'POST',
    body: JSON.stringify(options),
  }),
  pauseAgent: () => request('/agent/pause', { method: 'POST' }),
  resumeAgent: () => request('/agent/resume', { method: 'POST' }),

  // Agent status
  getAgentHealth: () => request('/agent/health'),
  getAgentStatus: () => request('/agent/status'),
  getAgentMetrics: (period = 'all') => request(`/agent/metrics?period=${period}`),
  getAgentSafety: () => request('/agent/safety'),

  // Agent config
  updateAgentConfig: (config) => request('/agent/config', {
    method: 'PUT',
    body: JSON.stringify(config),
  }),

  // Kill switch
  triggerKillSwitch: (reason) => request('/agent/kill-switch', {
    method: 'POST',
    body: JSON.stringify({ action: 'trigger', reason }),
  }),
  resetKillSwitch: () => request('/agent/kill-switch', {
    method: 'POST',
    body: JSON.stringify({ action: 'reset' }),
  }),
  getKillSwitchResetCode: () => request('/agent/safety/kill-switch/reset-code', { method: 'POST' }),

  // Command status
  getCommandStatus: (commandId) => request(`/agent/commands/${commandId}`),
  listCommands: (limit = 20, offset = 0, status = null) => {
    const params = new URLSearchParams({ limit, offset });
    if (status) params.append('status', status);
    return request(`/agent/commands?${params}`);
  },

  // Safety
  getSafetyEvents: (limit = 50, breakerType = null, severity = null) => {
    const params = new URLSearchParams({ limit });
    if (breakerType) params.append('breaker_type', breakerType);
    if (severity) params.append('severity', severity);
    return request(`/agent/safety/events?${params}`);
  },
  resetCircuitBreaker: (breakerName) => request('/agent/safety/circuit-breakers/reset', {
    method: 'POST',
    body: JSON.stringify({ breaker_name: breakerName }),
  }),
};

export { APIError };
export default api;
