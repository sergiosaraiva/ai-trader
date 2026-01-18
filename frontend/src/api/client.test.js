import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { api, APIError } from './client';

describe('API Client', () => {
  beforeEach(() => {
    // Mock fetch
    global.fetch = vi.fn();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('health check', () => {
    it('returns health data on success', async () => {
      const mockResponse = { status: 'healthy', version: '1.0.0' };
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await api.health();
      expect(result).toEqual(mockResponse);
      expect(fetch).toHaveBeenCalledWith('/health');
    });
  });

  describe('getPrediction', () => {
    it('fetches prediction successfully', async () => {
      const mockPrediction = {
        signal: 'BUY',
        confidence: 0.72,
        current_price: 1.08543,
      };
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockPrediction),
      });

      const result = await api.getPrediction();
      expect(result).toEqual(mockPrediction);
      expect(fetch).toHaveBeenCalledWith('/api/v1/predictions/latest', expect.any(Object));
    });

    it('throws APIError on failure', async () => {
      global.fetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.resolve({ detail: 'Server error' }),
      });

      await expect(api.getPrediction()).rejects.toThrow(APIError);
    });
  });

  describe('getCandles', () => {
    it('fetches candles with default params', async () => {
      const mockCandles = { candles: [{ open: 1.08, close: 1.09 }] };
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockCandles),
      });

      const result = await api.getCandles('EURUSD');
      expect(result).toEqual(mockCandles);
      expect(fetch).toHaveBeenCalledWith(
        '/api/v1/market/candles?symbol=EURUSD&timeframe=1H&count=24',
        expect.any(Object)
      );
    });

    it('fetches candles with custom params', async () => {
      const mockCandles = { candles: [{ open: 1.08, close: 1.09 }] };
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockCandles),
      });

      await api.getCandles('GBPUSD', '4H', 48);
      expect(fetch).toHaveBeenCalledWith(
        '/api/v1/market/candles?symbol=GBPUSD&timeframe=4H&count=48',
        expect.any(Object)
      );
    });
  });

  describe('getSignals', () => {
    it('fetches signals with default limit', async () => {
      const mockSignals = { signals: [] };
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSignals),
      });

      const result = await api.getSignals();
      expect(result).toEqual(mockSignals);
      expect(fetch).toHaveBeenCalledWith(
        '/api/v1/predictions/history?limit=20',
        expect.any(Object)
      );
    });
  });

  describe('getPipelineStatus', () => {
    it('fetches pipeline status', async () => {
      const mockStatus = { status: 'healthy', last_run: '2025-01-12T10:00:00Z' };
      global.fetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockStatus),
      });

      const result = await api.getPipelineStatus();
      expect(result).toEqual(mockStatus);
      expect(fetch).toHaveBeenCalledWith('/api/v1/pipeline/status', expect.any(Object));
    });
  });

  describe('APIError', () => {
    it('creates error with message, status and data', () => {
      const error = new APIError('Test error', 404, { detail: 'Not found' });
      expect(error.message).toBe('Test error');
      expect(error.status).toBe(404);
      expect(error.data).toEqual({ detail: 'Not found' });
      expect(error.name).toBe('APIError');
    });
  });
});
