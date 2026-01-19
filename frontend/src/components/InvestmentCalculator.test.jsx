import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { InvestmentCalculator } from './InvestmentCalculator';

// Mock API response
const mockBacktestData = {
  periods: {
    '6m': {
      label: 'Last 6 Months',
      total_pips: 2079,
      win_rate: 0.477,
      profit_factor: 1.48,
      total_trades: 568,
      period_start: '2025-01-01',
      period_end: '2025-06-30',
      period_years: 0.5,
      period_months: 6,
    },
    '1y': {
      label: 'Last Year',
      total_pips: 4317,
      win_rate: 0.517,
      profit_factor: 1.73,
      total_trades: 948,
      period_start: '2024-07-01',
      period_end: '2025-06-30',
      period_years: 1.0,
      period_months: 12,
    },
    '5y': {
      label: 'All Time (5 Years)',
      total_pips: 8693,
      win_rate: 0.621,
      profit_factor: 2.69,
      total_trades: 966,
      period_start: '2020-01-01',
      period_end: '2025-12-31',
      period_years: 5.0,
      period_months: 60,
    },
  },
  leverage_options: [
    { value: 1, label: 'No Leverage (1:1)', risk: 'low' },
    { value: 10, label: '10:1', risk: 'medium' },
    { value: 30, label: '30:1 (EU Retail)', risk: 'high' },
  ],
  forex_constants: {
    standard_lot_size: 100000,
    pip_value_per_lot: 10,
  },
  data_source: 'WFO Validation (70% confidence threshold)',
};

// Mock the API client
vi.mock('../api/client', () => ({
  default: {
    getBacktestPeriods: vi.fn(() => Promise.resolve(mockBacktestData)),
  },
}));

describe('InvestmentCalculator', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows loading state initially', () => {
    render(<InvestmentCalculator />);
    expect(screen.getByText('Loading backtest data...')).toBeInTheDocument();
  });

  it('renders calculator with default investment amount after loading', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    expect(screen.getByText('What If Calculator')).toBeInTheDocument();
    expect(screen.getByLabelText('Investment amount')).toHaveValue('1,000');
  });

  it('displays quick amount buttons after loading', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    expect(screen.getByText('€500')).toBeInTheDocument();
    expect(screen.getByText('€1,000')).toBeInTheDocument();
    expect(screen.getByText('€5,000')).toBeInTheDocument();
    expect(screen.getByText('€10,000')).toBeInTheDocument();
  });

  it('displays time period selector buttons from API', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    expect(screen.getByText('Last 6 Months')).toBeInTheDocument();
    expect(screen.getByText('Last Year')).toBeInTheDocument();
    expect(screen.getByText('All Time (5 Years)')).toBeInTheDocument();
  });

  it('displays leverage selector buttons from API', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    expect(screen.getByText('No Leverage (1:1)')).toBeInTheDocument();
    expect(screen.getByText('10:1')).toBeInTheDocument();
    expect(screen.getByText('30:1 (EU Retail)')).toBeInTheDocument();
  });

  it('calculates returns for default €1,000 investment with no leverage', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    // Default: no leverage (1:1), Last Year period with 4317 pips
    // €1,000 / 100,000 = 0.01 lots, pip value = $0.10/pip
    // 4317 pips × $0.10 = $432 profit
    expect(screen.getByText('+€432')).toBeInTheDocument();
    expect(screen.getByText('€1,432')).toBeInTheDocument();
  });

  it('updates calculations when leverage changes', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    // Click 30:1 leverage
    fireEvent.click(screen.getByText('30:1 (EU Retail)'));

    // With €1,000 at 30:1 leverage = 0.3 lots = $3/pip
    // 4317 pips × $3 = $12,951 profit
    expect(screen.getByText('+€12,951')).toBeInTheDocument();
    expect(screen.getByText('€13,951')).toBeInTheDocument();
  });

  it('updates calculations when investment amount changes', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    const input = screen.getByLabelText('Investment amount');
    fireEvent.change(input, { target: { value: '10000' } });

    // With €10,000 no leverage = 0.1 lots = $1/pip
    // 4317 pips × $1 = $4,317 profit
    expect(screen.getByText('+€4,317')).toBeInTheDocument();
    expect(screen.getByText('€14,317')).toBeInTheDocument();
  });

  it('updates calculations when time period changes', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    // Click "All Time (5 Years)" button - 8693 pips
    fireEvent.click(screen.getByText('All Time (5 Years)'));

    // With €1,000 no leverage = 0.01 lots = $0.10/pip
    // 8693 pips × $0.10 = $869 profit
    expect(screen.getByText('+€869')).toBeInTheDocument();
    expect(screen.getByText('€1,869')).toBeInTheDocument();
  });

  it('shows percentage return', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    // €1,000 → €1,432 = 43% return (Last Year, no leverage)
    expect(screen.getByText(/\+43% total/)).toBeInTheDocument();
  });

  it('shows annualized return', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    // 43% over 1 year = 43% per year
    expect(screen.getByText(/43% per year/)).toBeInTheDocument();
  });

  it('toggles details section when info button clicked', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    // Details should not be visible initially
    expect(screen.queryByText('Calculation Details')).not.toBeInTheDocument();

    // Click info button
    const infoButton = screen.getByLabelText('Show calculation details');
    fireEvent.click(infoButton);

    // Details should now be visible
    expect(screen.getByText('Calculation Details')).toBeInTheDocument();
    expect(screen.getByText('None (1:1)')).toBeInTheDocument();
    expect(screen.getByText('+4,317')).toBeInTheDocument();
    expect(screen.getByText('51.7%')).toBeInTheDocument();
  });

  it('shows details with data source from API', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    // Open details
    fireEvent.click(screen.getByLabelText('Show calculation details'));

    // Check data source is shown
    expect(screen.getByText('WFO Validation (70% confidence threshold)')).toBeInTheDocument();
  });

  it('does not show leverage warning when no leverage selected', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    // Default is no leverage - warning should not be shown
    expect(screen.queryByText(/Leverage Warning/)).not.toBeInTheDocument();
  });

  it('shows leverage warning when leverage is selected', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    // Select leverage
    fireEvent.click(screen.getByText('30:1 (EU Retail)'));

    expect(screen.getByText(/Leverage Warning/)).toBeInTheDocument();
    expect(screen.getByText(/amplifies both gains AND losses/)).toBeInTheDocument();
  });

  it('displays disclaimer note', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    expect(screen.getByText(/simulating trades/)).toBeInTheDocument();
    expect(screen.getByText(/past performance does not guarantee/i)).toBeInTheDocument();
  });

  it('validates input to only accept numbers', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    const input = screen.getByLabelText('Investment amount');
    fireEvent.change(input, { target: { value: 'abc123xyz' } });

    expect(input).toHaveValue('123');
  });

  it('limits investment to maximum of 1,000,000', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    const input = screen.getByLabelText('Investment amount');
    fireEvent.change(input, { target: { value: '2000000' } });

    expect(input).toHaveValue('1,000');
  });

  it('handles empty input gracefully', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    const input = screen.getByLabelText('Investment amount');
    fireEvent.change(input, { target: { value: '' } });

    expect(input).toHaveValue('0');
    expect(screen.getByText('+€0')).toBeInTheDocument();
  });

  it('highlights selected leverage button with appropriate color', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    // Default "No Leverage" should be highlighted green
    const noLeverage = screen.getByText('No Leverage (1:1)');
    expect(noLeverage).toHaveClass('bg-green-500');

    // Click high leverage
    fireEvent.click(screen.getByText('30:1 (EU Retail)'));

    // Should now be highlighted orange
    const leverage30 = screen.getByText('30:1 (EU Retail)');
    expect(leverage30).toHaveClass('bg-orange-500');
    expect(noLeverage).not.toHaveClass('bg-green-500');
  });

  it('has proper accessibility attributes', async () => {
    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    expect(screen.getByRole('region', { name: 'Investment Calculator' })).toBeInTheDocument();
    expect(screen.getByLabelText('Investment amount')).toBeInTheDocument();
  });

  it('shows error message but still renders with fallback data on API error', async () => {
    // Mock API to reject
    const api = await import('../api/client');
    api.default.getBacktestPeriods.mockRejectedValueOnce(new Error('Network error'));

    render(<InvestmentCalculator />);

    await waitFor(() => {
      expect(screen.queryByText('Loading backtest data...')).not.toBeInTheDocument();
    });

    // Should show error message
    expect(screen.getByText(/Using cached data/)).toBeInTheDocument();

    // Should still render calculator with fallback data
    expect(screen.getByText('What If Calculator')).toBeInTheDocument();
    expect(screen.getByLabelText('Investment amount')).toBeInTheDocument();
  });
});
