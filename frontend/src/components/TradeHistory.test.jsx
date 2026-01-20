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

  // Dynamic Asset Metadata Tests
  it('formats forex prices with 5 decimals', () => {
    const signals = [
      {
        id: '1',
        signal: 'BUY',
        price: 1.08543,
        confidence: 0.72,
        timestamp: new Date().toISOString(),
        asset_metadata: {
          price_precision: 5,
          asset_type: 'forex',
        },
      },
    ];
    render(<TradeHistory signals={signals} />);

    expect(screen.getByText('@ 1.08543')).toBeInTheDocument();
  });

  it('formats crypto prices with 8 decimals', () => {
    const signals = [
      {
        id: '1',
        signal: 'BUY',
        price: 50123.12345678,
        confidence: 0.68,
        timestamp: new Date().toISOString(),
        asset_metadata: {
          price_precision: 8,
          asset_type: 'crypto',
        },
      },
    ];
    render(<TradeHistory signals={signals} />);

    expect(screen.getByText('@ 50123.12345678')).toBeInTheDocument();
  });

  it('formats stock prices with 2 decimals', () => {
    const signals = [
      {
        id: '1',
        signal: 'BUY',
        price: 150.5,
        confidence: 0.75,
        timestamp: new Date().toISOString(),
        asset_metadata: {
          price_precision: 2,
          asset_type: 'stock',
        },
      },
    ];
    render(<TradeHistory signals={signals} />);

    expect(screen.getByText('@ 150.50')).toBeInTheDocument();
  });

  it('uses global assetMetadata when signal metadata missing', () => {
    const signals = [
      {
        id: '1',
        signal: 'BUY',
        price: 1.08543,
        confidence: 0.72,
        timestamp: new Date().toISOString(),
        // No asset_metadata on signal
      },
    ];
    const assetMetadata = {
      price_precision: 5,
      asset_type: 'forex',
    };
    render(<TradeHistory signals={signals} assetMetadata={assetMetadata} />);

    expect(screen.getByText('@ 1.08543')).toBeInTheDocument();
  });

  it('falls back to default formatting without metadata', () => {
    const signals = [
      {
        id: '1',
        signal: 'BUY',
        price: 1.08543,
        confidence: 0.72,
        timestamp: new Date().toISOString(),
        // No metadata
      },
    ];
    render(<TradeHistory signals={signals} />);

    // Should use default precision (5)
    expect(screen.getByText('@ 1.08543')).toBeInTheDocument();
  });

  it('handles mixed asset types in signal list', () => {
    const signals = [
      {
        id: '1',
        signal: 'BUY',
        price: 1.08543,
        confidence: 0.72,
        timestamp: new Date().toISOString(),
        asset_metadata: {
          price_precision: 5,
          asset_type: 'forex',
        },
      },
      {
        id: '2',
        signal: 'BUY',
        price: 50123.12,
        confidence: 0.68,
        timestamp: new Date().toISOString(),
        asset_metadata: {
          price_precision: 2,
          asset_type: 'crypto',
        },
      },
    ];
    render(<TradeHistory signals={signals} />);

    expect(screen.getByText('@ 1.08543')).toBeInTheDocument();
    expect(screen.getByText('@ 50123.12')).toBeInTheDocument();
  });

  // should_trade Tests (HOLD signals)
  it('renders HOLD signal when should_trade is false', () => {
    const signals = [
      {
        id: '1',
        direction: 'long',
        price: 1.08500,
        confidence: 0.68,
        timestamp: new Date().toISOString(),
        should_trade: false,
      },
    ];
    render(<TradeHistory signals={signals} />);

    expect(screen.getByText('HOLD')).toBeInTheDocument();
    expect(screen.getByText('@ 1.08500')).toBeInTheDocument();
    expect(screen.getByText('68%')).toBeInTheDocument();
  });

  it('renders BUY signal when should_trade is true', () => {
    const signals = [
      {
        id: '1',
        direction: 'long',
        price: 1.08500,
        confidence: 0.75,
        timestamp: new Date().toISOString(),
        should_trade: true,
      },
    ];
    render(<TradeHistory signals={signals} />);

    expect(screen.getByText('BUY')).toBeInTheDocument();
    expect(screen.queryByText('HOLD')).not.toBeInTheDocument();
  });

  it('renders SELL signal when should_trade is true and direction is short', () => {
    const signals = [
      {
        id: '1',
        direction: 'short',
        price: 1.08200,
        confidence: 0.72,
        timestamp: new Date().toISOString(),
        should_trade: true,
      },
    ];
    render(<TradeHistory signals={signals} />);

    expect(screen.getByText('SELL')).toBeInTheDocument();
    expect(screen.queryByText('HOLD')).not.toBeInTheDocument();
  });

  it('prioritizes should_trade over direction when determining signal type', () => {
    const signals = [
      {
        id: '1',
        direction: 'long',
        price: 1.08500,
        confidence: 0.68,
        timestamp: new Date().toISOString(),
        should_trade: false, // should_trade=false overrides direction
      },
    ];
    render(<TradeHistory signals={signals} />);

    // Should show HOLD, not BUY, because should_trade=false
    expect(screen.getByText('HOLD')).toBeInTheDocument();
    expect(screen.queryByText('BUY')).not.toBeInTheDocument();
  });

  it('handles mixed signals with should_trade true and false', () => {
    const signals = [
      {
        id: '1',
        direction: 'long',
        price: 1.08500,
        confidence: 0.75,
        timestamp: new Date().toISOString(),
        should_trade: true,
      },
      {
        id: '2',
        direction: 'short',
        price: 1.08300,
        confidence: 0.65,
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        should_trade: false,
      },
      {
        id: '3',
        direction: 'short',
        price: 1.08200,
        confidence: 0.72,
        timestamp: new Date(Date.now() - 7200000).toISOString(),
        should_trade: true,
      },
    ];
    render(<TradeHistory signals={signals} />);

    expect(screen.getByText('BUY')).toBeInTheDocument();
    expect(screen.getByText('HOLD')).toBeInTheDocument();
    expect(screen.getByText('SELL')).toBeInTheDocument();
    expect(screen.getByText('3 signals')).toBeInTheDocument();
  });

  it('renders HOLD signal with gray styling', () => {
    const signals = [
      {
        id: '1',
        direction: 'long',
        price: 1.08500,
        confidence: 0.68,
        timestamp: new Date().toISOString(),
        should_trade: false,
      },
    ];
    const { container } = render(<TradeHistory signals={signals} />);

    // Check HOLD text appears
    expect(screen.getByText('HOLD')).toBeInTheDocument();

    // Check gray color class is applied
    const holdText = screen.getByText('HOLD');
    expect(holdText).toHaveClass('text-gray-400');

    // Check gray icon background
    const iconContainer = container.querySelector('.bg-gray-500\\/20');
    expect(iconContainer).toBeInTheDocument();
  });

  it('handles should_trade undefined (backward compatibility)', () => {
    const signals = [
      {
        id: '1',
        direction: 'long',
        price: 1.08500,
        confidence: 0.72,
        timestamp: new Date().toISOString(),
        // No should_trade field (old data)
      },
    ];
    render(<TradeHistory signals={signals} />);

    // Should fallback to showing BUY based on direction
    expect(screen.getByText('BUY')).toBeInTheDocument();
  });

  it('handles should_trade null (old records)', () => {
    const signals = [
      {
        id: '1',
        direction: 'long',
        price: 1.08500,
        confidence: 0.72,
        timestamp: new Date().toISOString(),
        should_trade: null, // Old record
      },
    ];
    render(<TradeHistory signals={signals} />);

    // Should fallback to showing BUY based on direction
    expect(screen.getByText('BUY')).toBeInTheDocument();
  });

  it('uses ChevronRight icon for HOLD signals', () => {
    const signals = [
      {
        id: '1',
        direction: 'long',
        price: 1.08500,
        confidence: 0.68,
        timestamp: new Date().toISOString(),
        should_trade: false,
      },
    ];
    const { container } = render(<TradeHistory signals={signals} />);

    // Verify ChevronRight icon is used (size 18, gray color)
    const chevronIcon = container.querySelector('.text-gray-400');
    expect(chevronIcon).toBeInTheDocument();
  });
});
