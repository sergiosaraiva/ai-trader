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

    // Check for default values
    expect(screen.getByText('+8,693')).toBeInTheDocument();
    expect(screen.getByText('62.1%')).toBeInTheDocument();
    expect(screen.getByText('2.69')).toBeInTheDocument();
    expect(screen.getByText('966')).toBeInTheDocument();
    expect(screen.getByText('7.67')).toBeInTheDocument();
    expect(screen.getByText('+9.0')).toBeInTheDocument();
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
});
