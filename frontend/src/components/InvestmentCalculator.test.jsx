import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { InvestmentCalculator } from './InvestmentCalculator';

describe('InvestmentCalculator', () => {
  it('renders calculator with default investment amount', () => {
    render(<InvestmentCalculator />);

    expect(screen.getByText('What If Calculator')).toBeInTheDocument();
    expect(screen.getByLabelText('Investment amount')).toHaveValue('1,000');
  });

  it('displays quick amount buttons', () => {
    render(<InvestmentCalculator />);

    expect(screen.getByText('€500')).toBeInTheDocument();
    expect(screen.getByText('€1,000')).toBeInTheDocument();
    expect(screen.getByText('€5,000')).toBeInTheDocument();
    expect(screen.getByText('€10,000')).toBeInTheDocument();
  });

  it('calculates returns for default €1,000 investment', () => {
    render(<InvestmentCalculator />);

    // With €1,000 at 30:1 leverage = 0.3 lots = $3/pip
    // 8693 pips × $3 = $26,079 profit
    expect(screen.getByText('+€26,079')).toBeInTheDocument();
    expect(screen.getByText('€27,079')).toBeInTheDocument(); // Final balance
  });

  it('updates calculations when investment amount changes', () => {
    render(<InvestmentCalculator />);

    const input = screen.getByLabelText('Investment amount');

    // Clear and type new value
    fireEvent.change(input, { target: { value: '5000' } });

    // With €5,000 at 30:1 leverage = 1.5 lots = $15/pip
    // 8693 pips × $15 = $130,395 profit
    expect(screen.getByText('+€130,395')).toBeInTheDocument();
    expect(screen.getByText('€135,395')).toBeInTheDocument(); // Final balance
  });

  it('updates calculations when quick amount button clicked', () => {
    render(<InvestmentCalculator />);

    // Click €500 button
    fireEvent.click(screen.getByText('€500'));

    // With €500 at 30:1 leverage = 0.15 lots = $1.5/pip
    // 8693 pips × $1.5 = $13,040 profit (rounded)
    expect(screen.getByText('+€13,040')).toBeInTheDocument();
  });

  it('shows percentage return', () => {
    render(<InvestmentCalculator />);

    // €1,000 → €27,079 = 2607.9% return (rounded to 2608%)
    expect(screen.getByText(/\+2608% total/)).toBeInTheDocument();
  });

  it('shows annualized return', () => {
    render(<InvestmentCalculator />);

    // 2608% over 5 years = ~522% per year
    expect(screen.getByText(/522% per year/)).toBeInTheDocument();
  });

  it('toggles details section when info button clicked', () => {
    render(<InvestmentCalculator />);

    // Details should not be visible initially
    expect(screen.queryByText('Calculation Details')).not.toBeInTheDocument();

    // Click info button
    const infoButton = screen.getByLabelText('Show calculation details');
    fireEvent.click(infoButton);

    // Details should now be visible
    expect(screen.getByText('Calculation Details')).toBeInTheDocument();
    expect(screen.getByText('30:1')).toBeInTheDocument(); // Leverage
    expect(screen.getByText('+8,693')).toBeInTheDocument(); // Total pips
    expect(screen.getByText('62.1%')).toBeInTheDocument(); // Win rate
  });

  it('shows details with correct calculation values', () => {
    render(<InvestmentCalculator />);

    // Open details
    fireEvent.click(screen.getByLabelText('Show calculation details'));

    // Check all detail values
    expect(screen.getByText('0.30 lots')).toBeInTheDocument();
    expect(screen.getByText('$3.00/pip')).toBeInTheDocument();
    expect(screen.getByText('2.69')).toBeInTheDocument(); // Profit factor
    expect(screen.getByText('2020-01-01 to 2025-12-31')).toBeInTheDocument();
  });

  it('displays disclaimer note', () => {
    render(<InvestmentCalculator />);

    expect(screen.getByText(/historical backtest performance/)).toBeInTheDocument();
    expect(screen.getByText(/Past performance does not guarantee/)).toBeInTheDocument();
  });

  it('validates input to only accept numbers', () => {
    render(<InvestmentCalculator />);

    const input = screen.getByLabelText('Investment amount');

    // Try to enter non-numeric characters
    fireEvent.change(input, { target: { value: 'abc123xyz' } });

    // Should only keep the numbers
    expect(input).toHaveValue('123');
  });

  it('limits investment to maximum of 1,000,000', () => {
    render(<InvestmentCalculator />);

    const input = screen.getByLabelText('Investment amount');

    // Try to enter a value over 1 million
    fireEvent.change(input, { target: { value: '2000000' } });

    // Should not update (stays at default)
    expect(input).toHaveValue('1,000');
  });

  it('handles empty input gracefully', () => {
    render(<InvestmentCalculator />);

    const input = screen.getByLabelText('Investment amount');

    // Clear the input
    fireEvent.change(input, { target: { value: '' } });

    // Should set to 0
    expect(input).toHaveValue('0');
    expect(screen.getByText('+€0')).toBeInTheDocument();
  });

  it('highlights selected quick amount button', () => {
    render(<InvestmentCalculator />);

    // Default €1,000 should be highlighted
    const button1000 = screen.getByText('€1,000');
    expect(button1000).toHaveClass('bg-blue-500');

    // Click different amount
    fireEvent.click(screen.getByText('€5,000'));

    // €5,000 should now be highlighted
    const button5000 = screen.getByText('€5,000');
    expect(button5000).toHaveClass('bg-blue-500');
    expect(button1000).not.toHaveClass('bg-blue-500');
  });

  it('has proper accessibility attributes', () => {
    render(<InvestmentCalculator />);

    // Region with label
    expect(screen.getByRole('region', { name: 'Investment Calculator' })).toBeInTheDocument();

    // Input with label
    expect(screen.getByLabelText('Investment amount')).toBeInTheDocument();
  });
});
