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
    expect(screen.getByText('Error loading prediction')).toBeInTheDocument();
    expect(screen.getByText('Test error message')).toBeInTheDocument();
  });

  it('renders no prediction state', () => {
    render(<PredictionCard prediction={null} />);
    expect(screen.getByText('No prediction available')).toBeInTheDocument();
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
    expect(screen.getByText('Current Prediction')).toBeInTheDocument();
  });
});
