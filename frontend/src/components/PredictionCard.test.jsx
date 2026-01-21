import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PredictionCard } from './PredictionCard';

describe('PredictionCard', () => {
  it('renders loading state', () => {
    render(<PredictionCard loading={true} />);
    // Check for skeleton loader (animate-pulse class)
    const skeleton = document.querySelector('.animate-pulse');
    expect(skeleton).toBeInTheDocument();
  });

  it('renders error state', () => {
    render(<PredictionCard error="Test error message" />);
    expect(screen.getByText('Error loading recommendation')).toBeInTheDocument();
    expect(screen.getByText('Test error message')).toBeInTheDocument();
  });

  it('renders no prediction state', () => {
    render(<PredictionCard prediction={null} />);
    expect(screen.getByText('No recommendation available')).toBeInTheDocument();
  });

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

  // Dynamic Asset Metadata Tests
  it('displays formatted forex symbol with metadata', () => {
    const prediction = {
      signal: 'BUY',
      confidence: 0.72,
      current_price: 1.08543,
      symbol: 'EURUSD',
      timestamp: new Date().toISOString(),
      asset_metadata: {
        formatted_symbol: 'EUR/USD',
        price_precision: 5,
        asset_type: 'forex',
      },
    };
    render(<PredictionCard prediction={prediction} />);

    expect(screen.getByText(/Forex Currency Pair.*EUR\/USD/)).toBeInTheDocument();
    expect(screen.getByText('@ 1.08543')).toBeInTheDocument();
  });

  it('displays formatted crypto symbol with metadata', () => {
    const prediction = {
      signal: 'BUY',
      confidence: 0.68,
      current_price: 50123.12345678,
      symbol: 'BTC-USD',
      timestamp: new Date().toISOString(),
      asset_metadata: {
        formatted_symbol: 'BTC/USD',
        price_precision: 8,
        asset_type: 'crypto',
      },
    };
    render(<PredictionCard prediction={prediction} />);

    expect(screen.getByText(/Cryptocurrency.*BTC\/USD/)).toBeInTheDocument();
    expect(screen.getByText('@ 50123.12345678')).toBeInTheDocument();
  });

  it('displays formatted stock symbol with metadata', () => {
    const prediction = {
      signal: 'BUY',
      confidence: 0.75,
      current_price: 150.5,
      symbol: 'AAPL',
      timestamp: new Date().toISOString(),
      asset_metadata: {
        formatted_symbol: 'AAPL',
        price_precision: 2,
        asset_type: 'stock',
      },
    };
    render(<PredictionCard prediction={prediction} />);

    expect(screen.getByText(/Stock.*AAPL/)).toBeInTheDocument();
    expect(screen.getByText('@ 150.50')).toBeInTheDocument();
  });

  it('uses formatPrice with asset metadata precision', () => {
    const prediction = {
      signal: 'BUY',
      confidence: 0.72,
      current_price: 1.08543789,
      symbol: 'EURUSD',
      timestamp: new Date().toISOString(),
      asset_metadata: {
        formatted_symbol: 'EUR/USD',
        price_precision: 5,
      },
    };
    render(<PredictionCard prediction={prediction} />);

    // Should be formatted to 5 decimals
    expect(screen.getByText('@ 1.08544')).toBeInTheDocument();
  });

  it('falls back to default formatting without metadata', () => {
    const prediction = {
      signal: 'BUY',
      confidence: 0.72,
      current_price: 1.08543,
      symbol: 'EURUSD',
      timestamp: new Date().toISOString(),
      // No asset_metadata
    };
    render(<PredictionCard prediction={prediction} />);

    // Should still work with defaults - shows generic asset type and symbol
    expect(screen.getByText(/Financial Asset/)).toBeInTheDocument();
    expect(screen.getByText(/EUR\/USD/)).toBeInTheDocument();
    expect(screen.getByText('@ 1.08543')).toBeInTheDocument(); // Default precision
  });

  it('handles missing symbol gracefully', () => {
    const prediction = {
      signal: 'BUY',
      confidence: 0.72,
      current_price: 1.08543,
      timestamp: new Date().toISOString(),
      // No symbol
    };
    render(<PredictionCard prediction={prediction} />);

    // Should show N/A or handle gracefully
    expect(screen.getByText('Current Recommendation')).toBeInTheDocument();
  });

  // Timestamp Display Tests
  describe('Timestamp Display', () => {
    const mockPredictionBase = {
      signal: 'BUY',
      direction: 'long',
      confidence: 0.75,
      current_price: 1.08543,
      symbol: 'EURUSD',
      timestamp: '2024-01-15T14:05:32',
      should_trade: true,
      component_directions: { '1H': 1, '4H': 1, 'D': 1 },
      component_confidences: { '1H': 0.72, '4H': 0.75, 'D': 0.68 },
      component_weights: { '1H': 0.6, '4H': 0.3, 'D': 0.1 },
    };

    it('renders data_timestamp when provided', () => {
      const predictionWithDataTimestamp = {
        ...mockPredictionBase,
        data_timestamp: '2024-01-15T14:00:00',
      };

      render(<PredictionCard prediction={predictionWithDataTimestamp} />);

      // Check that data_timestamp is rendered
      expect(screen.getByText(/Data from:/)).toBeInTheDocument();
    });

    it('renders next_prediction_at when provided', () => {
      const predictionWithNextPrediction = {
        ...mockPredictionBase,
        next_prediction_at: '2024-01-15T15:01:00',
      };

      render(<PredictionCard prediction={predictionWithNextPrediction} />);

      // Check that next_prediction_at is rendered
      expect(screen.getByText(/Next update:/)).toBeInTheDocument();
    });

    it('renders both data_timestamp and next_prediction_at', () => {
      const predictionWithBothTimestamps = {
        ...mockPredictionBase,
        data_timestamp: '2024-01-15T14:00:00',
        next_prediction_at: '2024-01-15T15:01:00',
      };

      render(<PredictionCard prediction={predictionWithBothTimestamps} />);

      // Check both timestamps are rendered
      expect(screen.getByText(/Data from:/)).toBeInTheDocument();
      expect(screen.getByText(/Next update:/)).toBeInTheDocument();
    });

    it('formats data_timestamp correctly', () => {
      const predictionWithDataTimestamp = {
        ...mockPredictionBase,
        data_timestamp: '2024-01-15T14:00:00',
      };

      render(<PredictionCard prediction={predictionWithDataTimestamp} />);

      // The formatTime function should format it as locale string
      const dataFromText = screen.getByText(/Data from:/).textContent;
      expect(dataFromText).toContain('Data from:');
      // Should contain formatted date (exact format depends on locale)
      expect(dataFromText).toMatch(/\d{1,2}\/\d{1,2}\/\d{4}/);
    });

    it('formats next_prediction_at correctly', () => {
      const predictionWithNextPrediction = {
        ...mockPredictionBase,
        next_prediction_at: '2024-01-15T15:01:00',
      };

      render(<PredictionCard prediction={predictionWithNextPrediction} />);

      // The formatTime function should format it as locale string
      const nextUpdateText = screen.getByText(/Next update:/).textContent;
      expect(nextUpdateText).toContain('Next update:');
      // Should contain formatted date
      expect(nextUpdateText).toMatch(/\d{1,2}\/\d{1,2}\/\d{4}/);
    });

    it('does not render data_timestamp when missing', () => {
      render(<PredictionCard prediction={mockPredictionBase} />);

      // Should not render data_timestamp label
      expect(screen.queryByText(/Data from:/)).not.toBeInTheDocument();
    });

    it('does not render next_prediction_at when missing', () => {
      render(<PredictionCard prediction={mockPredictionBase} />);

      // Should not render next_prediction_at label
      expect(screen.queryByText(/Next update:/)).not.toBeInTheDocument();
    });

    it('handles null data_timestamp gracefully', () => {
      const predictionWithNullTimestamp = {
        ...mockPredictionBase,
        data_timestamp: null,
      };

      render(<PredictionCard prediction={predictionWithNullTimestamp} />);

      // Should not crash and should not render the label
      expect(screen.queryByText(/Data from:/)).not.toBeInTheDocument();
    });

    it('handles undefined data_timestamp gracefully', () => {
      const predictionWithUndefinedTimestamp = {
        ...mockPredictionBase,
        data_timestamp: undefined,
      };

      render(<PredictionCard prediction={predictionWithUndefinedTimestamp} />);

      // Should not crash and should not render the label
      expect(screen.queryByText(/Data from:/)).not.toBeInTheDocument();
    });

    it('handles null next_prediction_at gracefully', () => {
      const predictionWithNullNextPrediction = {
        ...mockPredictionBase,
        next_prediction_at: null,
      };

      render(<PredictionCard prediction={predictionWithNullNextPrediction} />);

      // Should not crash and should not render the label
      expect(screen.queryByText(/Next update:/)).not.toBeInTheDocument();
    });

    it('handles invalid timestamp format gracefully', () => {
      const predictionWithInvalidTimestamp = {
        ...mockPredictionBase,
        data_timestamp: 'invalid-date',
      };

      // Should not crash
      expect(() => {
        render(<PredictionCard prediction={predictionWithInvalidTimestamp} />);
      }).not.toThrow();
    });

    it('renders main timestamp regardless of new timestamp fields', () => {
      const predictionWithAllTimestamps = {
        ...mockPredictionBase,
        timestamp: '2024-01-15T14:05:32',
        data_timestamp: '2024-01-15T14:00:00',
        next_prediction_at: '2024-01-15T15:01:00',
      };

      render(<PredictionCard prediction={predictionWithAllTimestamps} />);

      // Main timestamp should be rendered (in the header)
      // The Clock icon and timestamp should be visible
      const clockIcon = document.querySelector('svg');
      expect(clockIcon).toBeInTheDocument();
    });
  });

  describe('Timestamp Integration with Signals', () => {
    const mockPredictionBase = {
      signal: 'BUY',
      direction: 'long',
      confidence: 0.75,
      current_price: 1.08543,
      symbol: 'EURUSD',
      timestamp: '2024-01-15T14:05:32',
      should_trade: true,
      component_directions: { '1H': 1, '4H': 1, 'D': 1 },
      component_confidences: { '1H': 0.72, '4H': 0.75, 'D': 0.68 },
      component_weights: { '1H': 0.6, '4H': 0.3, 'D': 0.1 },
    };

    it('renders timestamps with HOLD signal', () => {
      const holdPredictionWithTimestamps = {
        ...mockPredictionBase,
        confidence: 0.65, // Below 70% threshold
        should_trade: false,
        data_timestamp: '2024-01-15T14:00:00',
        next_prediction_at: '2024-01-15T15:01:00',
      };

      render(<PredictionCard prediction={holdPredictionWithTimestamps} />);

      // Should show HOLD
      expect(screen.getByText('HOLD')).toBeInTheDocument();

      // Timestamps should still be visible
      expect(screen.getByText(/Data from:/)).toBeInTheDocument();
      expect(screen.getByText(/Next update:/)).toBeInTheDocument();
    });

    it('renders timestamps with BUY signal', () => {
      const buyPredictionWithTimestamps = {
        ...mockPredictionBase,
        signal: 'BUY',
        direction: 'long',
        should_trade: true,
        data_timestamp: '2024-01-15T14:00:00',
        next_prediction_at: '2024-01-15T15:01:00',
      };

      render(<PredictionCard prediction={buyPredictionWithTimestamps} />);

      // Should show BUY (check for main recommendation)
      expect(screen.getAllByText('BUY').length).toBeGreaterThan(0);

      // Timestamps should be visible
      expect(screen.getByText(/Data from:/)).toBeInTheDocument();
      expect(screen.getByText(/Next update:/)).toBeInTheDocument();
    });

    it('renders timestamps with SELL signal', () => {
      const sellPredictionWithTimestamps = {
        ...mockPredictionBase,
        signal: 'SELL',
        direction: 'short',
        should_trade: true,
        data_timestamp: '2024-01-15T14:00:00',
        next_prediction_at: '2024-01-15T15:01:00',
      };

      render(<PredictionCard prediction={sellPredictionWithTimestamps} />);

      // Should show SELL
      expect(screen.getByText('SELL')).toBeInTheDocument();

      // Timestamps should be visible
      expect(screen.getByText(/Data from:/)).toBeInTheDocument();
      expect(screen.getByText(/Next update:/)).toBeInTheDocument();
    });

    it('renders timestamps with asset metadata', () => {
      const predictionWithAssetMetadata = {
        ...mockPredictionBase,
        asset_metadata: {
          asset_type: 'forex',
          formatted_symbol: 'EUR/USD',
          price_precision: 5,
          price_unit: 'pips',
        },
        data_timestamp: '2024-01-15T14:00:00',
        next_prediction_at: '2024-01-15T15:01:00',
      };

      render(<PredictionCard prediction={predictionWithAssetMetadata} />);

      // Should render formatted symbol
      expect(screen.getByText(/EUR\/USD/)).toBeInTheDocument();

      // Timestamps should be visible
      expect(screen.getByText(/Data from:/)).toBeInTheDocument();
      expect(screen.getByText(/Next update:/)).toBeInTheDocument();
    });
  });

  describe('Timestamp Edge Cases', () => {
    const mockPredictionBase = {
      signal: 'BUY',
      direction: 'long',
      confidence: 0.75,
      current_price: 1.08543,
      symbol: 'EURUSD',
      timestamp: '2024-01-15T14:05:32',
      should_trade: true,
    };

    it('handles very old data_timestamp', () => {
      const predictionWithOldTimestamp = {
        ...mockPredictionBase,
        data_timestamp: '2020-01-01T00:00:00',
      };

      render(<PredictionCard prediction={predictionWithOldTimestamp} />);

      // Should render without error
      expect(screen.getByText(/Data from:/)).toBeInTheDocument();
    });

    it('handles future next_prediction_at', () => {
      const predictionWithFutureTimestamp = {
        ...mockPredictionBase,
        next_prediction_at: '2030-01-01T00:00:00',
      };

      render(<PredictionCard prediction={predictionWithFutureTimestamp} />);

      // Should render without error
      expect(screen.getByText(/Next update:/)).toBeInTheDocument();
    });

    it('handles empty string timestamps', () => {
      const predictionWithEmptyTimestamps = {
        ...mockPredictionBase,
        data_timestamp: '',
        next_prediction_at: '',
      };

      render(<PredictionCard prediction={predictionWithEmptyTimestamps} />);

      // Should not crash - formatTime handles empty strings
      // Empty strings should result in no display or "N/A"
      expect(screen.queryByText(/Data from:/)).not.toBeInTheDocument();
      expect(screen.queryByText(/Next update:/)).not.toBeInTheDocument();
    });
  });
});
