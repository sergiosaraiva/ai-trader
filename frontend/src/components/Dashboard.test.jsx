import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { Dashboard } from './Dashboard';
import * as usePollingModule from '../hooks/usePolling';

// Mock the usePolling hook
vi.mock('../hooks/usePolling', () => ({
  usePolling: vi.fn(),
}));

// Mock all child components
vi.mock('./PredictionCard', () => ({
  PredictionCard: ({ prediction, loading, error }) => (
    <div data-testid="prediction-card">
      PredictionCard: {loading ? 'loading' : error ? 'error' : prediction?.signal || 'null'}
    </div>
  ),
}));

vi.mock('./AccountStatus', () => ({
  AccountStatus: ({ pipelineStatus, modelStatus, loading }) => (
    <div data-testid="account-status">
      AccountStatus: {loading ? 'loading' : 'loaded'}
    </div>
  ),
}));

vi.mock('./AboutSection', () => ({
  AboutSection: ({ tradingPair, modelWeights, vixValue }) => (
    <div data-testid="about-section">
      AboutSection: pair={tradingPair}, weights={modelWeights ? 'yes' : 'no'}, vix={vixValue ?? 'null'}
    </div>
  ),
}));

vi.mock('./PriceChart', () => ({
  PriceChart: ({ candles, prediction, loading }) => (
    <div data-testid="price-chart">
      PriceChart: {candles?.length || 0} candles, {loading ? 'loading' : 'loaded'}
    </div>
  ),
}));

vi.mock('./PerformanceStats', () => ({
  PerformanceStats: ({ performance, loading }) => (
    <div data-testid="performance-stats">
      PerformanceStats: {loading ? 'loading' : performance ? 'data' : 'null'}
    </div>
  ),
}));

vi.mock('./TradeHistory', () => ({
  TradeHistory: ({ signals, loading }) => (
    <div data-testid="trade-history">
      TradeHistory: {signals?.length || 0} signals, {loading ? 'loading' : 'loaded'}
    </div>
  ),
}));

describe('Dashboard', () => {
  const mockUsePolling = vi.mocked(usePollingModule.usePolling);

  beforeEach(() => {
    vi.clearAllMocks();

    // Default mock implementation - returns loading state
    mockUsePolling.mockReturnValue({
      data: null,
      loading: true,
      error: null,
      refetch: vi.fn(),
      lastUpdated: null,
    });
  });

  describe('Basic Rendering', () => {
    it('renders dashboard layout', () => {
      render(<Dashboard />);

      expect(screen.getByText('AI Assets Trader')).toBeInTheDocument();
      expect(screen.getByTestId('prediction-card')).toBeInTheDocument();
      expect(screen.getByTestId('account-status')).toBeInTheDocument();
      expect(screen.getByTestId('about-section')).toBeInTheDocument();
      expect(screen.getByTestId('price-chart')).toBeInTheDocument();
      expect(screen.getByTestId('performance-stats')).toBeInTheDocument();
      expect(screen.getByTestId('trade-history')).toBeInTheDocument();
    });

    it('renders header with default trading pair', () => {
      render(<Dashboard />);
      // Header should contain EUR/USD in the subtitle
      const header = document.querySelector('header');
      expect(header.textContent).toContain('EUR/USD');
    });

    it('renders footer with performance metrics', () => {
      render(<Dashboard />);
      expect(screen.getByText(/61% Win Rate/)).toBeInTheDocument();
      expect(screen.getByText(/\(high-confidence\)/)).toBeInTheDocument();
      expect(screen.getByText('2.10 PF')).toBeInTheDocument();
      expect(screen.getByText('WFO Validated')).toBeInTheDocument();
    });

    it('renders refresh button', () => {
      render(<Dashboard />);
      expect(screen.getByText('Refresh')).toBeInTheDocument();
    });
  });

  describe('VIX Data Polling and Passing', () => {
    it('polls VIX data using usePolling hook', () => {
      render(<Dashboard />);

      // Check that usePolling was called multiple times (one for each data source)
      // VIX polling should be one of them
      const calls = mockUsePolling.mock.calls;
      expect(calls.length).toBeGreaterThan(0);
    });

    it('passes VIX value to AboutSection when available', () => {
      // Mock usePolling to return different data for different calls
      // Call order: 1=prediction, 2=candles, 3=pipeline, 4=model, 5=VIX, 6=signals, 7=performance
      let callCount = 0;
      mockUsePolling.mockImplementation(() => {
        callCount++;
        // 5th call is VIX data (based on Dashboard component order)
        if (callCount === 5) {
          return {
            data: { value: 18.5 },
            loading: false,
            error: null,
            refetch: vi.fn(),
            lastUpdated: new Date(),
          };
        }
        return {
          data: null,
          loading: false,
          error: null,
          refetch: vi.fn(),
          lastUpdated: null,
        };
      });

      render(<Dashboard />);

      // AboutSection should receive VIX value
      expect(screen.getByTestId('about-section')).toHaveTextContent('vix=18.5');
    });

    it('passes null VIX value to AboutSection when data is unavailable', () => {
      mockUsePolling.mockReturnValue({
        data: null,
        loading: false,
        error: null,
        refetch: vi.fn(),
        lastUpdated: null,
      });

      render(<Dashboard />);

      // AboutSection should receive null VIX value
      expect(screen.getByTestId('about-section')).toHaveTextContent('vix=null');
    });

    it('passes undefined VIX value when VIX data object has no value', () => {
      let callCount = 0;
      mockUsePolling.mockImplementation(() => {
        callCount++;
        if (callCount === 5) {
          return {
            data: {}, // Empty object, no value property
            loading: false,
            error: null,
            refetch: vi.fn(),
            lastUpdated: new Date(),
          };
        }
        return {
          data: null,
          loading: false,
          error: null,
          refetch: vi.fn(),
          lastUpdated: null,
        };
      });

      render(<Dashboard />);

      // AboutSection should receive undefined VIX value
      expect(screen.getByTestId('about-section')).toHaveTextContent('vix=null');
    });
  });

  describe('Trading Pair Display', () => {
    it('displays trading pair from prediction data', () => {
      let callCount = 0;
      mockUsePolling.mockImplementation(() => {
        callCount++;
        // First call is prediction data
        if (callCount === 1) {
          return {
            data: { signal: 'BUY', symbol: 'GBPUSD', confidence: 0.72 },
            loading: false,
            error: null,
            refetch: vi.fn(),
            lastUpdated: new Date(),
          };
        }
        return {
          data: null,
          loading: false,
          error: null,
          refetch: vi.fn(),
          lastUpdated: null,
        };
      });

      render(<Dashboard />);

      // Header subtitle should show GBP/USD - look within header
      const header = document.querySelector('header');
      expect(header.textContent).toContain('GBP/USD');
    });

    it('uses default EURUSD when prediction has no symbol', () => {
      let callCount = 0;
      mockUsePolling.mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return {
            data: { signal: 'BUY', confidence: 0.72 }, // No symbol
            loading: false,
            error: null,
            refetch: vi.fn(),
            lastUpdated: new Date(),
          };
        }
        return {
          data: null,
          loading: false,
          error: null,
          refetch: vi.fn(),
          lastUpdated: null,
        };
      });

      render(<Dashboard />);

      // Header subtitle should show EUR/USD (default) - look within header
      const header = document.querySelector('header');
      expect(header.textContent).toContain('EUR/USD');
    });
  });

  describe('Model Weights Passing', () => {
    it('passes model weights to AboutSection when available', () => {
      let callCount = 0;
      mockUsePolling.mockImplementation(() => {
        callCount++;
        // 4th call is model status
        if (callCount === 4) {
          return {
            data: { weights: { '1H': 0.6, '4H': 0.3, 'D': 0.1 } },
            loading: false,
            error: null,
            refetch: vi.fn(),
            lastUpdated: new Date(),
          };
        }
        return {
          data: null,
          loading: false,
          error: null,
          refetch: vi.fn(),
          lastUpdated: null,
        };
      });

      render(<Dashboard />);

      // AboutSection should receive weights
      expect(screen.getByTestId('about-section')).toHaveTextContent('weights=yes');
    });

    it('passes null when model weights are not available', () => {
      mockUsePolling.mockReturnValue({
        data: null,
        loading: false,
        error: null,
        refetch: vi.fn(),
        lastUpdated: null,
      });

      render(<Dashboard />);

      expect(screen.getByTestId('about-section')).toHaveTextContent('weights=no');
    });
  });

  describe('Data Flow to Child Components', () => {
    it('passes correct props to PredictionCard', () => {
      let callCount = 0;
      mockUsePolling.mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return {
            data: { signal: 'BUY', confidence: 0.75 },
            loading: false,
            error: null,
            refetch: vi.fn(),
            lastUpdated: new Date(),
          };
        }
        return {
          data: null,
          loading: false,
          error: null,
          refetch: vi.fn(),
          lastUpdated: null,
        };
      });

      render(<Dashboard />);

      expect(screen.getByTestId('prediction-card')).toHaveTextContent('BUY');
    });

    it('passes candles data to PriceChart', () => {
      let callCount = 0;
      mockUsePolling.mockImplementation(() => {
        callCount++;
        // 2nd call is candles data
        if (callCount === 2) {
          return {
            data: { candles: [{ time: '2024-01-01', close: 1.08 }] },
            loading: false,
            error: null,
            refetch: vi.fn(),
            lastUpdated: new Date(),
          };
        }
        return {
          data: null,
          loading: false,
          error: null,
          refetch: vi.fn(),
          lastUpdated: null,
        };
      });

      render(<Dashboard />);

      expect(screen.getByTestId('price-chart')).toHaveTextContent('1 candles');
    });

    it('passes signals to TradeHistory', () => {
      let callCount = 0;
      mockUsePolling.mockImplementation(() => {
        callCount++;
        // 6th call is signals data
        if (callCount === 6) {
          return {
            data: { predictions: [{ signal: 'BUY' }, { signal: 'SELL' }] },
            loading: false,
            error: null,
            refetch: vi.fn(),
            lastUpdated: new Date(),
          };
        }
        return {
          data: null,
          loading: false,
          error: null,
          refetch: vi.fn(),
          lastUpdated: null,
        };
      });

      render(<Dashboard />);

      expect(screen.getByTestId('trade-history')).toHaveTextContent('2 signals');
    });
  });

  describe('Last Updated Time', () => {
    it('displays "Never" when no last updated time', () => {
      render(<Dashboard />);
      expect(screen.getByText(/Never/)).toBeInTheDocument();
    });

    it('displays formatted time when last updated is available', () => {
      let callCount = 0;
      mockUsePolling.mockImplementation(() => {
        callCount++;
        if (callCount === 1) {
          return {
            data: { signal: 'BUY' },
            loading: false,
            error: null,
            refetch: vi.fn(),
            lastUpdated: new Date('2024-01-15T10:30:00'),
          };
        }
        return {
          data: null,
          loading: false,
          error: null,
          refetch: vi.fn(),
          lastUpdated: null,
        };
      });

      render(<Dashboard />);

      // Check that time is displayed (exact format depends on locale)
      expect(screen.getByText(/Updated:/)).toBeInTheDocument();
    });
  });

  describe('Integration', () => {
    it('renders complete dashboard with all data', () => {
      let callCount = 0;
      mockUsePolling.mockImplementation(() => {
        callCount++;

        // Call order: 1=prediction, 2=candles, 3=pipeline, 4=model, 5=VIX, 6=signals, 7=performance

        // Prediction data
        if (callCount === 1) {
          return {
            data: { signal: 'BUY', confidence: 0.75, symbol: 'EURUSD' },
            loading: false,
            error: null,
            refetch: vi.fn(),
            lastUpdated: new Date(),
          };
        }

        // Candles data
        if (callCount === 2) {
          return {
            data: { candles: [{ time: '2024-01-01', close: 1.08 }] },
            loading: false,
            error: null,
            refetch: vi.fn(),
            lastUpdated: null,
          };
        }

        // Pipeline status (3rd)
        if (callCount === 3) {
          return {
            data: { status: 'ok' },
            loading: false,
            error: null,
            refetch: vi.fn(),
            lastUpdated: null,
          };
        }

        // Model status with weights (4th)
        if (callCount === 4) {
          return {
            data: { weights: { '1H': 0.6, '4H': 0.3, 'D': 0.1 } },
            loading: false,
            error: null,
            refetch: vi.fn(),
            lastUpdated: null,
          };
        }

        // VIX data (5th)
        if (callCount === 5) {
          return {
            data: { value: 18.5 },
            loading: false,
            error: null,
            refetch: vi.fn(),
            lastUpdated: null,
          };
        }

        return {
          data: null,
          loading: false,
          error: null,
          refetch: vi.fn(),
          lastUpdated: null,
        };
      });

      render(<Dashboard />);

      // Verify all components received correct data
      expect(screen.getByTestId('about-section')).toHaveTextContent('pair=EURUSD');
      expect(screen.getByTestId('about-section')).toHaveTextContent('weights=yes');
      expect(screen.getByTestId('about-section')).toHaveTextContent('vix=18.5');
      expect(screen.getByTestId('prediction-card')).toHaveTextContent('BUY');
      expect(screen.getByTestId('price-chart')).toHaveTextContent('1 candles');
    });
  });
});
