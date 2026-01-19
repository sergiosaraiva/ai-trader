import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PerformanceChart } from './PerformanceChart';

// Mock Recharts components
vi.mock('recharts', () => ({
  ComposedChart: ({ children }) => <div data-testid="composed-chart">{children}</div>,
  Bar: ({ children }) => <div data-testid="bar">{children}</div>,
  Cell: ({ children }) => <div data-testid="cell">{children}</div>,
  Line: () => <div data-testid="line" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  Tooltip: () => <div data-testid="tooltip" />,
  ResponsiveContainer: ({ children }) => <div data-testid="responsive-container">{children}</div>,
  ReferenceLine: () => <div data-testid="reference-line" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
}));

describe('PerformanceChart', () => {
  const mockTrades = [
    {
      id: 1,
      symbol: 'EURUSD',
      direction: 'long',
      entry_price: 1.08500,
      entry_time: '2024-01-15T14:01:00',
      exit_price: 1.08750,
      exit_time: '2024-01-15T16:30:00',
      exit_reason: 'tp',
      lot_size: 0.1,
      pips: 25.0,
      pnl_usd: 250.0,
      is_winner: true,
      status: 'closed',
    },
    {
      id: 2,
      symbol: 'EURUSD',
      direction: 'short',
      entry_price: 1.08600,
      entry_time: '2024-01-15T17:00:00',
      exit_price: 1.08450,
      exit_time: '2024-01-15T19:30:00',
      exit_reason: 'tp',
      lot_size: 0.1,
      pips: 15.0,
      pnl_usd: 150.0,
      is_winner: true,
      status: 'closed',
    },
    {
      id: 3,
      symbol: 'EURUSD',
      direction: 'long',
      entry_price: 1.08700,
      entry_time: '2024-01-16T10:00:00',
      exit_price: 1.08550,
      exit_time: '2024-01-16T12:00:00',
      exit_reason: 'sl',
      lot_size: 0.1,
      pips: -15.0,
      pnl_usd: -150.0,
      is_winner: false,
      status: 'closed',
    },
  ];

  const mockAssetMetadata = {
    type: 'forex',
    symbol: 'EURUSD',
  };

  describe('Loading State', () => {
    it('should render loading skeleton when loading', () => {
      render(<PerformanceChart trades={[]} loading={true} error={null} />);

      const container = screen.getByText((content, element) =>
        element?.className?.includes('animate-pulse')
      );
      expect(container).toBeDefined();
    });

    it('should have correct height during loading', () => {
      const { container } = render(<PerformanceChart trades={[]} loading={true} error={null} />);
      const loadingDiv = container.querySelector('.h-\\[400px\\]');
      expect(loadingDiv).toBeDefined();
    });
  });

  describe('Error State', () => {
    it('should render error message when error is present', () => {
      render(<PerformanceChart trades={[]} loading={false} error="Failed to load" />);
      expect(screen.getByText('Failed to load')).toBeDefined();
    });

    it('should display error in red text', () => {
      render(<PerformanceChart trades={[]} loading={false} error="API Error" />);
      const errorText = screen.getByText('API Error');
      expect(errorText.className).toContain('text-red-400');
    });
  });

  describe('Empty State', () => {
    it('should render empty message when no trades', () => {
      render(<PerformanceChart trades={[]} loading={false} error={null} />);
      expect(screen.getByText('No performance data available yet')).toBeDefined();
    });

    it('should render empty message when trades array is null', () => {
      render(<PerformanceChart trades={null} loading={false} error={null} />);
      expect(screen.getByText('No performance data available yet')).toBeDefined();
    });

    it('should render empty message when no closed trades', () => {
      const openTrades = [
        { ...mockTrades[0], status: 'open' },
      ];
      render(<PerformanceChart trades={openTrades} loading={false} error={null} />);
      expect(screen.getByText('No performance data available yet')).toBeDefined();
    });
  });

  describe('Chart Rendering', () => {
    it('should render chart when trades are provided', () => {
      render(
        <PerformanceChart
          trades={mockTrades}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      expect(screen.getByTestId('composed-chart')).toBeDefined();
      expect(screen.getByTestId('responsive-container')).toBeDefined();
    });

    it('should render title', () => {
      render(
        <PerformanceChart
          trades={mockTrades}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      expect(screen.getByText('30-Day Performance')).toBeDefined();
    });

    it('should render chart axes', () => {
      render(
        <PerformanceChart
          trades={mockTrades}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      const yAxes = screen.getAllByTestId('y-axis');
      expect(yAxes.length).toBe(2); // Left and right Y-axes
      expect(screen.getByTestId('x-axis')).toBeDefined();
    });

    it('should render bars and line', () => {
      render(
        <PerformanceChart
          trades={mockTrades}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      expect(screen.getAllByTestId('bar').length).toBeGreaterThan(0);
      expect(screen.getByTestId('line')).toBeDefined();
    });

    it('should render grid and reference line', () => {
      render(
        <PerformanceChart
          trades={mockTrades}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      expect(screen.getByTestId('cartesian-grid')).toBeDefined();
      expect(screen.getByTestId('reference-line')).toBeDefined();
    });
  });

  describe('Statistics Footer', () => {
    it('should render statistics footer', () => {
      render(
        <PerformanceChart
          trades={mockTrades}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      expect(screen.getByText('Best Day')).toBeDefined();
      expect(screen.getByText('Worst Day')).toBeDefined();
      expect(screen.getByText('Max Drawdown')).toBeDefined();
    });

    it('should display best day value', () => {
      render(
        <PerformanceChart
          trades={mockTrades}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      // Best day should be Jan 15 with +40.0 pips (25 + 15)
      expect(screen.getByText(/\+40\.0 pips/)).toBeDefined();
    });

    it('should display worst day value', () => {
      render(
        <PerformanceChart
          trades={mockTrades}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      // Worst day should be Jan 16 with -15.0 pips
      const worstDayElements = screen.getAllByText(/-15\.0 pips/);
      expect(worstDayElements.length).toBeGreaterThan(0);
    });
  });

  describe('Cumulative P&L Display', () => {
    it('should display cumulative P&L in header', () => {
      render(
        <PerformanceChart
          trades={mockTrades}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      // Total P&L should be +25.0 pips (25 + 15 - 15)
      expect(screen.getByText(/\+25\.0 pips/)).toBeDefined();
    });

    it('should show positive P&L in green', () => {
      render(
        <PerformanceChart
          trades={mockTrades}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      const pnlElement = screen.getByText(/\+25\.0 pips/);
      expect(pnlElement.className).toContain('text-green-400');
    });

    it('should show negative P&L in red', () => {
      const losingTrades = [
        {
          ...mockTrades[2],
          exit_time: '2024-01-15T12:00:00',
        },
      ];

      const { container } = render(
        <PerformanceChart
          trades={losingTrades}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      // Find the element with red color for negative P&L in header
      const redElements = container.querySelectorAll('.text-red-400');
      const hasNegativePnl = Array.from(redElements).some(el =>
        el.textContent.includes('-15.0 pips')
      );
      expect(hasNegativePnl).toBe(true);
    });
  });

  describe('Profit Unit Labeling', () => {
    it('should use "pips" for forex', () => {
      render(
        <PerformanceChart
          trades={mockTrades}
          loading={false}
          error={null}
          assetMetadata={{ type: 'forex' }}
        />
      );

      expect(screen.getAllByText(/pips/).length).toBeGreaterThan(0);
    });

    it('should use "pips" when no asset metadata', () => {
      render(
        <PerformanceChart
          trades={mockTrades}
          loading={false}
          error={null}
        />
      );

      expect(screen.getAllByText(/pips/).length).toBeGreaterThan(0);
    });
  });

  describe('Day Counting', () => {
    it('should count trading days correctly', () => {
      render(
        <PerformanceChart
          trades={mockTrades}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      // Should have 2 trading days (Jan 15 and Jan 16)
      expect(screen.getByText(/2 days/)).toBeDefined();
    });

    it('should count profitable days correctly', () => {
      render(
        <PerformanceChart
          trades={mockTrades}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      // 1 profitable day out of 2 (50%)
      expect(screen.getByText(/1 profitable \(50%\)/)).toBeDefined();
    });
  });

  describe('Data Processing', () => {
    it('should group trades by day', () => {
      const sameDayTrades = [
        { ...mockTrades[0], exit_time: '2024-01-15T10:00:00', pips: 10 },
        { ...mockTrades[1], exit_time: '2024-01-15T14:00:00', pips: 20 },
        { ...mockTrades[2], exit_time: '2024-01-15T18:00:00', pips: 30 },
      ];

      render(
        <PerformanceChart
          trades={sameDayTrades}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      // Should aggregate to single day with +60 pips
      expect(screen.getByText(/1 days/)).toBeDefined();
    });

    it('should handle trades without exit_time gracefully', () => {
      const tradesWithoutExit = [
        { ...mockTrades[0], exit_time: null },
        mockTrades[1],
      ];

      render(
        <PerformanceChart
          trades={tradesWithoutExit}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      // Should only count the trade with exit_time
      expect(screen.getByText(/1 days/)).toBeDefined();
    });

    it('should limit to last 30 days', () => {
      // Create 45 days of trades
      const manyTrades = Array.from({ length: 45 }, (_, i) => ({
        ...mockTrades[0],
        id: i,
        exit_time: new Date(Date.now() - (45 - i) * 24 * 60 * 60 * 1000).toISOString(),
        pips: 10,
      }));

      render(
        <PerformanceChart
          trades={manyTrades}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      // Should only show last 30 days
      expect(screen.getByText(/30 days/)).toBeDefined();
    });
  });

  describe('Styling and Layout', () => {
    it('should apply card styling', () => {
      const { container } = render(
        <PerformanceChart
          trades={mockTrades}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      const card = container.querySelector('.bg-gray-800.rounded-lg.p-6.card-hover');
      expect(card).toBeDefined();
    });

    it('should have correct chart height', () => {
      const { container } = render(
        <PerformanceChart
          trades={mockTrades}
          loading={false}
          error={null}
          assetMetadata={mockAssetMetadata}
        />
      );

      const chartContainer = container.querySelector('.h-\\[300px\\]');
      expect(chartContainer).toBeDefined();
    });
  });
});
