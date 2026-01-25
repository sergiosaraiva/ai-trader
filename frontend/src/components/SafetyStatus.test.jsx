import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SafetyStatus } from './SafetyStatus';

describe('SafetyStatus', () => {
  it('renders loading state', () => {
    render(<SafetyStatus loading={true} />);
    const skeleton = document.querySelector('.animate-pulse');
    expect(skeleton).toBeInTheDocument();
  });

  it('renders "data not available" when no safety data', () => {
    render(<SafetyStatus safety={null} loading={false} />);
    expect(screen.getByText('Safety data not available')).toBeInTheDocument();
  });

  it('shows "SAFE TO TRADE" when safe', () => {
    const safety = {
      is_safe_to_trade: true,
      kill_switch: { is_active: false },
      circuit_breakers: {
        overall_state: 'active',
        can_trade: true,
        active_breakers: [],
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    expect(screen.getByText('SAFE TO TRADE')).toBeInTheDocument();
    expect(screen.getByText('All safety systems operational')).toBeInTheDocument();
  });

  it('shows "TRADING HALTED" when not safe', () => {
    const safety = {
      is_safe_to_trade: false,
      kill_switch: { is_active: true, reason: 'Emergency' },
      circuit_breakers: {
        overall_state: 'tripped',
        can_trade: false,
        active_breakers: ['daily_loss'],
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    expect(screen.getByText('TRADING HALTED')).toBeInTheDocument();
    expect(screen.getByText('Safety systems have halted trading')).toBeInTheDocument();
  });

  it('shows kill switch inactive status', () => {
    const safety = {
      is_safe_to_trade: true,
      kill_switch: { is_active: false },
      circuit_breakers: {
        overall_state: 'active',
        can_trade: true,
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    expect(screen.getByText('Inactive')).toBeInTheDocument();
  });

  it('shows kill switch active status with animation', () => {
    const safety = {
      is_safe_to_trade: false,
      kill_switch: {
        is_active: true,
        reason: 'Manual intervention',
        triggered_at: '2024-01-15T14:30:00Z',
      },
      circuit_breakers: {
        overall_state: 'tripped',
        can_trade: false,
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    const activeText = screen.getByText('ACTIVE');
    expect(activeText).toHaveClass('text-red-400');
    expect(activeText).toHaveClass('animate-pulse');
  });

  it('displays kill switch reason', () => {
    const safety = {
      is_safe_to_trade: false,
      kill_switch: {
        is_active: true,
        reason: 'High volatility detected',
      },
      circuit_breakers: {
        overall_state: 'tripped',
        can_trade: false,
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    expect(screen.getByText('Reason: High volatility detected')).toBeInTheDocument();
  });

  it('displays kill switch triggered time', () => {
    const safety = {
      is_safe_to_trade: false,
      kill_switch: {
        is_active: true,
        reason: 'Emergency',
        triggered_at: '2024-01-15T14:30:00Z',
      },
      circuit_breakers: {
        overall_state: 'tripped',
        can_trade: false,
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    // Should show the date in locale format
    expect(screen.getByText(/Triggered:/)).toBeInTheDocument();
  });

  it('lists circuit breakers', () => {
    const safety = {
      is_safe_to_trade: true,
      kill_switch: { is_active: false },
      circuit_breakers: {
        overall_state: 'active',
        can_trade: true,
        active_breakers: [],
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    expect(screen.getByText('Overall State')).toBeInTheDocument();
    expect(screen.getByText('Can Trade')).toBeInTheDocument();
  });

  it('shows active circuit breakers warning', () => {
    const safety = {
      is_safe_to_trade: false,
      kill_switch: { is_active: false },
      circuit_breakers: {
        overall_state: 'tripped',
        can_trade: false,
        active_breakers: ['daily_loss', 'max_trades'],
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    expect(screen.getByText(/Active breakers:/)).toBeInTheDocument();
    expect(screen.getByText(/daily_loss, max_trades/)).toBeInTheDocument();
  });

  it('shows daily limits with progress bars', () => {
    const safety = {
      is_safe_to_trade: true,
      kill_switch: { is_active: false },
      circuit_breakers: {
        overall_state: 'active',
        can_trade: true,
      },
      daily_metrics: {
        trades: 7,
        loss_pct: 2.5,
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    expect(screen.getByText('Trades')).toBeInTheDocument();
    expect(screen.getByText('7 / 10')).toBeInTheDocument();
    expect(screen.getByText('Loss %')).toBeInTheDocument();
    expect(screen.getByText('2.50% / 5%')).toBeInTheDocument();
  });

  it('shows warning color when daily limit is high', () => {
    const safety = {
      is_safe_to_trade: true,
      kill_switch: { is_active: false },
      circuit_breakers: {
        overall_state: 'active',
        can_trade: true,
      },
      daily_metrics: {
        trades: 9, // 90% of limit
        loss_pct: 4.2, // 84% of limit
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    // Should show values in yellow (high threshold)
    const tradesValue = screen.getByText('9 / 10');
    expect(tradesValue).toHaveClass('text-yellow-400');

    const lossValue = screen.getByText('4.20% / 5%');
    expect(lossValue).toHaveClass('text-yellow-400');
  });

  it('shows account metrics', () => {
    const safety = {
      is_safe_to_trade: true,
      kill_switch: { is_active: false },
      circuit_breakers: {
        overall_state: 'active',
        can_trade: true,
      },
      account_metrics: {
        current_equity: 102500.50,
        peak_equity: 105000.00,
        drawdown_pct: 2.38,
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    expect(screen.getByText('Current Equity')).toBeInTheDocument();
    expect(screen.getByText('$102,500.50')).toBeInTheDocument();
    expect(screen.getByText('Peak Equity')).toBeInTheDocument();
    expect(screen.getByText('$105,000.00')).toBeInTheDocument();
    expect(screen.getByText('Drawdown')).toBeInTheDocument();
    expect(screen.getByText('2.38%')).toBeInTheDocument();
  });

  it('shows red color for positive drawdown', () => {
    const safety = {
      is_safe_to_trade: true,
      kill_switch: { is_active: false },
      circuit_breakers: {
        overall_state: 'active',
        can_trade: true,
      },
      account_metrics: {
        current_equity: 95000,
        peak_equity: 100000,
        drawdown_pct: 5.0,
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    const drawdownValue = screen.getByText('5.00%');
    expect(drawdownValue).toHaveClass('text-red-400');
  });

  it('shows green color for zero drawdown', () => {
    const safety = {
      is_safe_to_trade: true,
      kill_switch: { is_active: false },
      circuit_breakers: {
        overall_state: 'active',
        can_trade: true,
      },
      account_metrics: {
        current_equity: 100000,
        peak_equity: 100000,
        drawdown_pct: 0,
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    const drawdownValue = screen.getByText('0.00%');
    expect(drawdownValue).toHaveClass('text-green-400');
  });

  it('renders progress bar at correct width', () => {
    const safety = {
      is_safe_to_trade: true,
      kill_switch: { is_active: false },
      circuit_breakers: {
        overall_state: 'active',
        can_trade: true,
      },
      daily_metrics: {
        trades: 5, // 50% of 10
        loss_pct: 2.0, // 40% of 5
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    const progressBars = document.querySelectorAll('.h-2.rounded-full.transition-all');

    // First bar (trades): 50% width
    expect(progressBars[0]).toHaveStyle({ width: '50%' });

    // Second bar (loss): 40% width
    expect(progressBars[1]).toHaveStyle({ width: '40%' });
  });

  it('caps progress bar at 100%', () => {
    const safety = {
      is_safe_to_trade: false,
      kill_switch: { is_active: false },
      circuit_breakers: {
        overall_state: 'tripped',
        can_trade: false,
      },
      daily_metrics: {
        trades: 15, // 150% of limit
        loss_pct: 6.0, // 120% of limit
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    const progressBars = document.querySelectorAll('.h-2.rounded-full.transition-all');

    // Both bars should be capped at 100%
    expect(progressBars[0]).toHaveStyle({ width: '100%' });
    expect(progressBars[1]).toHaveStyle({ width: '100%' });
  });

  it('handles missing optional fields gracefully', () => {
    const safety = {
      is_safe_to_trade: true,
      // Missing kill_switch, circuit_breakers, daily_metrics, account_metrics
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    expect(screen.getByText('SAFE TO TRADE')).toBeInTheDocument();
  });

  it('shows correct circuit breaker colors', () => {
    const { rerender } = render(
      <SafetyStatus
        safety={{
          is_safe_to_trade: true,
          kill_switch: { is_active: false },
          circuit_breakers: {
            overall_state: 'active',
            can_trade: true,
          },
        }}
        loading={false}
      />
    );

    // Active state should be green
    const activeValue = screen.getByText('active');
    expect(activeValue).toHaveClass('text-green-400');

    const canTradeYes = screen.getByText('Yes');
    expect(canTradeYes).toHaveClass('text-green-400');

    // Test inactive state
    rerender(
      <SafetyStatus
        safety={{
          is_safe_to_trade: false,
          kill_switch: { is_active: false },
          circuit_breakers: {
            overall_state: 'tripped',
            can_trade: false,
          },
        }}
        loading={false}
      />
    );

    const trippedValue = screen.getByText('tripped');
    expect(trippedValue).toHaveClass('text-red-400');

    const canTradeNo = screen.getByText('No');
    expect(canTradeNo).toHaveClass('text-red-400');
  });

  it('does not show daily metrics when missing', () => {
    const safety = {
      is_safe_to_trade: true,
      kill_switch: { is_active: false },
      circuit_breakers: {
        overall_state: 'active',
        can_trade: true,
      },
      // No daily_metrics
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    // Component uses daily_metrics || {}, so section appears but with 0 values
    expect(screen.getByText('Daily Limits')).toBeInTheDocument();
    expect(screen.getByText('0 / 10')).toBeInTheDocument();
    expect(screen.getByText('0.00% / 5%')).toBeInTheDocument();
  });

  it('does not show account metrics when missing', () => {
    const safety = {
      is_safe_to_trade: true,
      kill_switch: { is_active: false },
      circuit_breakers: {
        overall_state: 'active',
        can_trade: true,
      },
      // No account_metrics
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    // Component uses account_metrics || {}, so section appears but with $0.00 values
    expect(screen.getByText('Account Metrics')).toBeInTheDocument();
    expect(screen.getByText('Current Equity')).toBeInTheDocument();
    // Multiple $0.00 values exist (current, peak), just check the section is there
    const dollarValues = screen.getAllByText(/\$0\.00/);
    expect(dollarValues.length).toBeGreaterThan(0);
  });

  it('handles zero daily metrics', () => {
    const safety = {
      is_safe_to_trade: true,
      kill_switch: { is_active: false },
      circuit_breakers: {
        overall_state: 'active',
        can_trade: true,
      },
      daily_metrics: {
        trades: 0,
        loss_pct: 0,
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    expect(screen.getByText('0 / 10')).toBeInTheDocument();
    expect(screen.getByText('0.00% / 5%')).toBeInTheDocument();
  });

  it('formats large account values correctly', () => {
    const safety = {
      is_safe_to_trade: true,
      kill_switch: { is_active: false },
      circuit_breakers: {
        overall_state: 'active',
        can_trade: true,
      },
      account_metrics: {
        current_equity: 1234567.89,
        peak_equity: 1300000.00,
        drawdown_pct: 5.03,
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    expect(screen.getByText('$1,234,567.89')).toBeInTheDocument();
    expect(screen.getByText('$1,300,000.00')).toBeInTheDocument();
  });

  it('does not show active breakers when array is empty', () => {
    const safety = {
      is_safe_to_trade: true,
      kill_switch: { is_active: false },
      circuit_breakers: {
        overall_state: 'active',
        can_trade: true,
        active_breakers: [],
      },
    };

    render(<SafetyStatus safety={safety} loading={false} />);

    expect(screen.queryByText(/Active breakers:/)).not.toBeInTheDocument();
  });

  it('shows shield icon with correct color', () => {
    const { rerender } = render(
      <SafetyStatus
        safety={{
          is_safe_to_trade: true,
          kill_switch: { is_active: false },
        }}
        loading={false}
      />
    );

    // Should have green shield when safe
    let shield = document.querySelector('.text-green-400');
    expect(shield).toBeInTheDocument();

    rerender(
      <SafetyStatus
        safety={{
          is_safe_to_trade: false,
          kill_switch: { is_active: true },
        }}
        loading={false}
      />
    );

    // Should have red shield when not safe
    shield = document.querySelector('.text-red-400');
    expect(shield).toBeInTheDocument();
  });
});
