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
});
