import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { TradeHistory } from './TradeHistory';

describe('TradeHistory', () => {
  it('renders loading state', () => {
    render(<TradeHistory loading={true} />);
    const skeleton = document.querySelector('.animate-pulse');
    expect(skeleton).toBeInTheDocument();
  });

  it('renders error state', () => {
    render(<TradeHistory error="Failed to load signals" />);
    expect(screen.getByText('Failed to load signals')).toBeInTheDocument();
  });

  it('renders empty state', () => {
    render(<TradeHistory signals={[]} />);
    expect(screen.getByText('No signals recorded yet')).toBeInTheDocument();
    expect(screen.getByText('Signals will appear here as they are generated')).toBeInTheDocument();
  });

  it('renders signal history header', () => {
    render(<TradeHistory signals={[]} />);
    expect(screen.getByText('Signal History')).toBeInTheDocument();
  });

  it('renders BUY signals correctly', () => {
    const signals = [
      {
        id: '1',
        signal: 'BUY',
        price: 1.08500,
        confidence: 0.72,
        timestamp: new Date().toISOString(),
      },
    ];
    render(<TradeHistory signals={signals} />);

    expect(screen.getByText('BUY')).toBeInTheDocument();
    expect(screen.getByText('@ 1.08500')).toBeInTheDocument();
    expect(screen.getByText('72%')).toBeInTheDocument();
  });

  it('renders SELL signals correctly', () => {
    const signals = [
      {
        id: '1',
        signal: 'SELL',
        price: 1.08200,
        confidence: 0.65,
        timestamp: new Date().toISOString(),
      },
    ];
    render(<TradeHistory signals={signals} />);

    expect(screen.getByText('SELL')).toBeInTheDocument();
    expect(screen.getByText('@ 1.08200')).toBeInTheDocument();
  });

  it('renders multiple signals', () => {
    const signals = [
      {
        id: '1',
        signal: 'BUY',
        price: 1.08500,
        confidence: 0.72,
        timestamp: new Date().toISOString(),
      },
      {
        id: '2',
        signal: 'SELL',
        price: 1.08300,
        confidence: 0.68,
        timestamp: new Date(Date.now() - 3600000).toISOString(),
      },
    ];
    render(<TradeHistory signals={signals} />);

    expect(screen.getByText('2 signals')).toBeInTheDocument();
  });

  it('handles numeric signal values', () => {
    const signals = [
      {
        id: '1',
        signal: 1,
        current_price: 1.08500,
        confidence: 0.80,
        timestamp: new Date().toISOString(),
      },
    ];
    render(<TradeHistory signals={signals} />);

    expect(screen.getByText('BUY')).toBeInTheDocument();
    expect(screen.getByText('@ 1.08500')).toBeInTheDocument();
  });
});
