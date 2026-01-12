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
