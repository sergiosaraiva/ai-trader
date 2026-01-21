import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PerformanceStats } from './PerformanceStats';

describe('PerformanceStats', () => {
  it('renders loading state', () => {
    render(<PerformanceStats loading={true} />);
    const skeleton = document.querySelector('.animate-pulse');
    expect(skeleton).toBeInTheDocument();
  });

  it('renders error state', () => {
    render(<PerformanceStats error="Failed to load performance" />);
    expect(screen.getByText('Failed to load performance')).toBeInTheDocument();
  });

  it('renders default stats when no performance data', () => {
    render(<PerformanceStats />);

    // Check for default values (75/10/15 weights at 55% confidence)
    expect(screen.getByText('+7,238')).toBeInTheDocument();
    expect(screen.getByText('57.1%')).toBeInTheDocument();
    expect(screen.getByText('2.10')).toBeInTheDocument();
    expect(screen.getByText('1,078')).toBeInTheDocument();
    expect(screen.getByText('5.74')).toBeInTheDocument();
    expect(screen.getByText('+6.7')).toBeInTheDocument();
  });

  it('renders custom performance data', () => {
    const performance = {
      total_pips: 5000,
      win_rate: 0.58,
      profit_factor: 2.1,
      total_trades: 500,
      sharpe_ratio: 5.5,
      avg_pips_per_trade: 10.0,
    };
    render(<PerformanceStats performance={performance} />);

    expect(screen.getByText('+5,000')).toBeInTheDocument();
    expect(screen.getByText('58.0%')).toBeInTheDocument();
    expect(screen.getByText('2.10')).toBeInTheDocument();
    expect(screen.getByText('500')).toBeInTheDocument();
    expect(screen.getByText('5.50')).toBeInTheDocument();
    expect(screen.getByText('+10.0')).toBeInTheDocument();
  });

  it('displays validation info', () => {
    render(<PerformanceStats />);

    expect(screen.getByText('Walk-Forward Optimization (7 windows)')).toBeInTheDocument();
    expect(screen.getByText('100% (7/7 profitable)')).toBeInTheDocument();
  });

  // Dynamic Asset Metadata Tests
  it('displays pips unit for forex by default', () => {
    const performance = {
      total_pips: 5000,
      avg_pips_per_trade: 10.0,
    };
    render(<PerformanceStats performance={performance} />);

    // Should show "Total Pips" and "Avg Pips/Trade"
    expect(screen.getByText('Total Pips')).toBeInTheDocument();
    expect(screen.getByText('Avg Pips/Trade')).toBeInTheDocument();
  });

  it('displays dollars unit for crypto', () => {
    const performance = {
      total_pips: 5000,
      avg_pips_per_trade: 10.0,
    };
    const assetMetadata = {
      profit_unit: 'dollars',
      asset_type: 'crypto',
    };
    render(<PerformanceStats performance={performance} assetMetadata={assetMetadata} />);

    // Should show "Total Dollars" and "Avg Dollars/Trade"
    expect(screen.getByText('Total Dollars')).toBeInTheDocument();
    expect(screen.getByText('Avg Dollars/Trade')).toBeInTheDocument();
  });

  it('displays points unit for stocks', () => {
    const performance = {
      total_pips: 5000,
      avg_pips_per_trade: 10.0,
    };
    const assetMetadata = {
      profit_unit: 'points',
      asset_type: 'stock',
    };
    render(<PerformanceStats performance={performance} assetMetadata={assetMetadata} />);

    // Should show "Total Points" and "Avg Points/Trade"
    expect(screen.getByText('Total Points')).toBeInTheDocument();
    expect(screen.getByText('Avg Points/Trade')).toBeInTheDocument();
  });

  it('capitalizes profit unit in labels', () => {
    const performance = {
      total_pips: 5000,
      avg_pips_per_trade: 10.0,
    };
    const assetMetadata = {
      profit_unit: 'ticks',
    };
    render(<PerformanceStats performance={performance} assetMetadata={assetMetadata} />);

    // Should capitalize first letter: "Ticks"
    expect(screen.getByText('Total Ticks')).toBeInTheDocument();
    expect(screen.getByText('Avg Ticks/Trade')).toBeInTheDocument();
  });

  it('handles missing assetMetadata gracefully', () => {
    const performance = {
      total_pips: 5000,
      avg_pips_per_trade: 10.0,
    };
    render(<PerformanceStats performance={performance} />);

    // Should default to pips
    expect(screen.getByText('Total Pips')).toBeInTheDocument();
  });
});
