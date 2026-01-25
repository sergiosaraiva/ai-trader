import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AgentStatus } from './AgentStatus';

describe('AgentStatus', () => {
  it('renders loading state', () => {
    render(<AgentStatus loading={true} />);
    const skeleton = document.querySelector('.animate-pulse');
    expect(skeleton).toBeInTheDocument();
  });

  it('renders "not initialized" when no status', () => {
    render(<AgentStatus status={null} loading={false} />);
    expect(screen.getByText('Agent not initialized')).toBeInTheDocument();
    expect(screen.getByText('Start the agent to see status')).toBeInTheDocument();
  });

  it('shows correct status colors for running', () => {
    const status = {
      status: 'running',
      mode: 'simulation',
      cycle_count: 10,
    };

    render(<AgentStatus status={status} loading={false} />);

    const runningText = screen.getByText('RUNNING');
    expect(runningText).toHaveClass('text-green-400');
  });

  it('shows correct status colors for paused', () => {
    const status = {
      status: 'paused',
      mode: 'simulation',
      cycle_count: 5,
    };

    render(<AgentStatus status={status} loading={false} />);

    const pausedText = screen.getByText('PAUSED');
    expect(pausedText).toHaveClass('text-yellow-400');
  });

  it('shows correct status colors for stopped', () => {
    const status = {
      status: 'stopped',
      mode: 'simulation',
      cycle_count: 0,
    };

    render(<AgentStatus status={status} loading={false} />);

    const stoppedText = screen.getByText('STOPPED');
    expect(stoppedText).toHaveClass('text-gray-400');
  });

  it('shows correct status colors for error', () => {
    const status = {
      status: 'error',
      mode: 'simulation',
      cycle_count: 0,
    };

    render(<AgentStatus status={status} loading={false} />);

    const errorText = screen.getByText('ERROR');
    expect(errorText).toHaveClass('text-red-400');
  });

  it('displays all metrics correctly', () => {
    const status = {
      status: 'running',
      mode: 'paper',
      cycle_count: 42,
      last_cycle_at: new Date().toISOString(),
      open_positions: 3,
      account_equity: 102345.67,
      uptime_seconds: 7265, // 2h 1m 5s
    };

    render(<AgentStatus status={status} loading={false} />);

    expect(screen.getByText('RUNNING')).toBeInTheDocument();
    expect(screen.getByText('PAPER')).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
    expect(screen.getByText('$102,345.67')).toBeInTheDocument();
  });

  it('formats dates correctly', () => {
    // Test with relative times
    const now = new Date();
    const justNow = new Date(now.getTime() - 10000); // 10 seconds ago
    const minutesAgo = new Date(now.getTime() - 5 * 60000); // 5 minutes ago
    const hoursAgo = new Date(now.getTime() - 2 * 3600000 - 30 * 60000); // 2h 30m ago

    const { rerender } = render(
      <AgentStatus
        status={{
          status: 'running',
          mode: 'simulation',
          cycle_count: 1,
          last_cycle_at: justNow.toISOString(),
        }}
        loading={false}
      />
    );
    expect(screen.getByText('Just now')).toBeInTheDocument();

    rerender(
      <AgentStatus
        status={{
          status: 'running',
          mode: 'simulation',
          cycle_count: 1,
          last_cycle_at: minutesAgo.toISOString(),
        }}
        loading={false}
      />
    );
    expect(screen.getByText('5m ago')).toBeInTheDocument();

    rerender(
      <AgentStatus
        status={{
          status: 'running',
          mode: 'simulation',
          cycle_count: 1,
          last_cycle_at: hoursAgo.toISOString(),
        }}
        loading={false}
      />
    );
    expect(screen.getByText('2h 30m ago')).toBeInTheDocument();
  });

  it('shows last prediction when available', () => {
    const status = {
      status: 'running',
      mode: 'simulation',
      cycle_count: 5,
      last_prediction: {
        signal: 'BUY',
        confidence: 0.75,
      },
    };

    render(<AgentStatus status={status} loading={false} />);

    expect(screen.getByText('Last Prediction')).toBeInTheDocument();
    expect(screen.getByText('BUY')).toBeInTheDocument();
    expect(screen.getByText('75.0%')).toBeInTheDocument();
  });

  it('shows SELL signal in last prediction', () => {
    const status = {
      status: 'running',
      mode: 'simulation',
      cycle_count: 5,
      last_prediction: {
        signal: 'SELL',
        confidence: 0.68,
      },
    };

    render(<AgentStatus status={status} loading={false} />);

    const sellText = screen.getByText('SELL');
    expect(sellText).toHaveClass('text-red-400');
    expect(screen.getByText('68.0%')).toBeInTheDocument();
  });

  it('shows HOLD signal in last prediction', () => {
    const status = {
      status: 'running',
      mode: 'simulation',
      cycle_count: 5,
      last_prediction: {
        signal: 'HOLD',
        confidence: 0.55,
      },
    };

    render(<AgentStatus status={status} loading={false} />);

    const holdText = screen.getByText('HOLD');
    expect(holdText).toHaveClass('text-gray-400');
  });

  it('formats uptime correctly', () => {
    const status = {
      status: 'running',
      mode: 'simulation',
      cycle_count: 10,
      uptime_seconds: 3665, // 1h 1m 5s
    };

    render(<AgentStatus status={status} loading={false} />);

    expect(screen.getByText('1h 1m')).toBeInTheDocument();
  });

  it('formats uptime for less than 1 hour', () => {
    const status = {
      status: 'running',
      mode: 'simulation',
      cycle_count: 10,
      uptime_seconds: 1845, // 30m 45s
    };

    render(<AgentStatus status={status} loading={false} />);

    expect(screen.getByText('30m')).toBeInTheDocument();
  });

  it('does not show uptime when not available', () => {
    const status = {
      status: 'running',
      mode: 'simulation',
      cycle_count: 10,
      // No uptime_seconds
    };

    render(<AgentStatus status={status} loading={false} />);

    // Should not have Uptime label
    const uptimeLabels = screen.queryAllByText('Uptime');
    expect(uptimeLabels).toHaveLength(0);
  });

  it('does not show account equity when zero', () => {
    const status = {
      status: 'running',
      mode: 'simulation',
      cycle_count: 10,
      account_equity: 0,
    };

    render(<AgentStatus status={status} loading={false} />);

    expect(screen.queryByText(/Account Equity/)).not.toBeInTheDocument();
  });

  it('highlights open positions when greater than zero', () => {
    const status = {
      status: 'running',
      mode: 'paper',
      cycle_count: 10,
      open_positions: 2,
    };

    render(<AgentStatus status={status} loading={false} />);

    const positionsValue = screen.getByText('2');
    expect(positionsValue).toHaveClass('text-blue-400');
  });

  it('shows gray color for zero open positions', () => {
    const status = {
      status: 'running',
      mode: 'simulation',
      cycle_count: 10,
      open_positions: 0,
    };

    render(<AgentStatus status={status} loading={false} />);

    const positionsValue = screen.getByText('0');
    expect(positionsValue).toHaveClass('text-gray-500');
  });

  it('shows correct mode colors', () => {
    // Test live mode
    const { rerender } = render(
      <AgentStatus
        status={{ status: 'running', mode: 'live', cycle_count: 5 }}
        loading={false}
      />
    );
    expect(screen.getByText('LIVE')).toHaveClass('text-red-400');

    // Test paper mode
    rerender(
      <AgentStatus
        status={{ status: 'running', mode: 'paper', cycle_count: 5 }}
        loading={false}
      />
    );
    expect(screen.getByText('PAPER')).toHaveClass('text-yellow-400');

    // Test simulation mode
    rerender(
      <AgentStatus
        status={{ status: 'running', mode: 'simulation', cycle_count: 5 }}
        loading={false}
      />
    );
    expect(screen.getByText('SIMULATION')).toHaveClass('text-blue-400');
  });

  it('handles missing optional fields gracefully', () => {
    const status = {
      status: 'running',
      // Missing many optional fields
    };

    render(<AgentStatus status={status} loading={false} />);

    expect(screen.getByText('RUNNING')).toBeInTheDocument();
    // Mode defaults to 'N/A'
    const naElements = screen.getAllByText('N/A');
    expect(naElements.length).toBeGreaterThan(0);
    // Cycle count defaults to 0
    expect(screen.getByText('0')).toBeInTheDocument();
  });

  it('formats large cycle counts with commas', () => {
    const status = {
      status: 'running',
      mode: 'simulation',
      cycle_count: 123456,
    };

    render(<AgentStatus status={status} loading={false} />);

    expect(screen.getByText('123,456')).toBeInTheDocument();
  });

  it('formats account equity with 2 decimals', () => {
    const status = {
      status: 'running',
      mode: 'paper',
      cycle_count: 10,
      account_equity: 100000.5,
    };

    render(<AgentStatus status={status} loading={false} />);

    expect(screen.getByText('$100,000.50')).toBeInTheDocument();
  });

  it('shows Never for missing last_cycle_at', () => {
    const status = {
      status: 'stopped',
      mode: 'simulation',
      cycle_count: 0,
      // No last_cycle_at
    };

    render(<AgentStatus status={status} loading={false} />);

    expect(screen.getByText('Never')).toBeInTheDocument();
  });

  it('formats days in time difference', () => {
    const now = new Date();
    const threeDaysAgo = new Date(now.getTime() - 3 * 24 * 3600000 - 5 * 3600000);

    const status = {
      status: 'stopped',
      mode: 'simulation',
      cycle_count: 100,
      last_cycle_at: threeDaysAgo.toISOString(),
    };

    render(<AgentStatus status={status} loading={false} />);

    // Should show "3d Xh ago" where X might vary slightly due to timing
    expect(screen.getByText(/3d \d+h ago/)).toBeInTheDocument();
  });
});
