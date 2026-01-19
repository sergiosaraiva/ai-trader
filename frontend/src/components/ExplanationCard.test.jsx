import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ExplanationCard } from './ExplanationCard';

describe('ExplanationCard', () => {
  it('renders loading state', () => {
    render(<ExplanationCard loading={true} />);

    // Check for skeleton loader (animate-pulse class)
    const skeleton = document.querySelector('.animate-pulse');
    expect(skeleton).toBeInTheDocument();

    // Should have Sparkles icon placeholder
    const sparklesContainer = skeleton.querySelector('.text-blue-400');
    expect(sparklesContainer).toBeInTheDocument();
  });

  it('renders error state', () => {
    render(<ExplanationCard error="API connection failed" />);

    expect(screen.getByText('AI explanation unavailable')).toBeInTheDocument();

    // Should show AlertCircle icon
    const alertIcon = document.querySelector('svg');
    expect(alertIcon).toBeInTheDocument();
  });

  it('renders error state with refresh button', () => {
    const mockRefresh = vi.fn();
    render(<ExplanationCard error="API error" onRefresh={mockRefresh} />);

    expect(screen.getByText('AI explanation unavailable')).toBeInTheDocument();

    // Should have refresh button
    const refreshButton = screen.getByTitle('Retry');
    expect(refreshButton).toBeInTheDocument();

    // Click refresh button
    fireEvent.click(refreshButton);
    expect(mockRefresh).toHaveBeenCalledTimes(1);
  });

  it('renders nothing when explanation is null', () => {
    const { container } = render(<ExplanationCard explanation={null} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing when explanation object is empty', () => {
    const { container } = render(<ExplanationCard explanation={{}} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders nothing when explanation text is missing', () => {
    const { container } = render(
      <ExplanationCard explanation={{ cached: true, generated_at: '2024-01-01T00:00:00Z' }} />
    );
    expect(container.firstChild).toBeNull();
  });

  it('renders explanation text correctly', () => {
    const explanation = {
      explanation: 'The AI recommends buying EUR/USD because all three timeframes show bullish signals with high confidence. Market volatility is low, suggesting a stable trading environment.',
      cached: false,
      generated_at: '2024-01-15T10:30:00Z',
    };

    render(<ExplanationCard explanation={explanation} />);

    expect(screen.getByText('AI Analysis')).toBeInTheDocument();
    expect(screen.getByText(/The AI recommends buying EUR\/USD/)).toBeInTheDocument();
    expect(screen.getByText(/Market volatility is low/)).toBeInTheDocument();
  });

  it('shows cached indicator when explanation is cached', () => {
    const explanation = {
      explanation: 'This is a cached explanation.',
      cached: true,
      generated_at: '2024-01-15T10:00:00Z',
    };

    render(<ExplanationCard explanation={explanation} />);

    expect(screen.getByText('AI Analysis')).toBeInTheDocument();
    expect(screen.getByText('cached')).toBeInTheDocument();
    expect(screen.getByText('This is a cached explanation.')).toBeInTheDocument();
  });

  it('does not show cached indicator when explanation is fresh', () => {
    const explanation = {
      explanation: 'This is a fresh explanation.',
      cached: false,
      generated_at: '2024-01-15T10:30:00Z',
    };

    render(<ExplanationCard explanation={explanation} />);

    expect(screen.getByText('AI Analysis')).toBeInTheDocument();
    expect(screen.queryByText('cached')).not.toBeInTheDocument();
    expect(screen.getByText('This is a fresh explanation.')).toBeInTheDocument();
  });

  it('renders refresh button when onRefresh callback provided', () => {
    const mockRefresh = vi.fn();
    const explanation = {
      explanation: 'The market shows strong bullish momentum.',
      cached: false,
    };

    render(<ExplanationCard explanation={explanation} onRefresh={mockRefresh} />);

    const refreshButton = screen.getByTitle('Refresh explanation');
    expect(refreshButton).toBeInTheDocument();
    expect(refreshButton).toHaveAttribute('aria-label', 'Refresh explanation');
  });

  it('does not render refresh button when onRefresh not provided', () => {
    const explanation = {
      explanation: 'The market shows strong bullish momentum.',
      cached: false,
    };

    render(<ExplanationCard explanation={explanation} />);

    const refreshButton = screen.queryByTitle('Refresh explanation');
    expect(refreshButton).not.toBeInTheDocument();
  });

  it('calls onRefresh when refresh button clicked', () => {
    const mockRefresh = vi.fn();
    const explanation = {
      explanation: 'Current market analysis.',
      cached: true,
    };

    render(<ExplanationCard explanation={explanation} onRefresh={mockRefresh} />);

    const refreshButton = screen.getByTitle('Refresh explanation');
    fireEvent.click(refreshButton);

    expect(mockRefresh).toHaveBeenCalledTimes(1);
  });

  it('has correct accessibility attributes', () => {
    const explanation = {
      explanation: 'Market analysis explanation.',
      cached: false,
    };

    render(<ExplanationCard explanation={explanation} />);

    const region = screen.getByRole('region', { name: 'AI Analysis Explanation' });
    expect(region).toBeInTheDocument();
  });

  it('renders long explanation text without truncation', () => {
    const longExplanation = 'The AI trading system recommends a BUY position on EUR/USD based on unanimous agreement across all three timeframes. The 1-hour model shows strong bullish momentum with 85% confidence, the 4-hour model confirms the uptrend with 78% confidence, and the daily model supports the move with 72% confidence. Market volatility is currently low with VIX at 14.5, indicating a calm trading environment that favors trend-following strategies.';

    const explanation = {
      explanation: longExplanation,
      cached: false,
    };

    render(<ExplanationCard explanation={explanation} />);

    expect(screen.getByText(longExplanation)).toBeInTheDocument();
  });

  it('applies correct styling classes for gradient background', () => {
    const explanation = {
      explanation: 'Test explanation.',
      cached: false,
    };

    const { container } = render(<ExplanationCard explanation={explanation} />);

    const card = container.querySelector('.bg-gradient-to-r');
    expect(card).toBeInTheDocument();
    expect(card).toHaveClass('from-blue-900/30');
    expect(card).toHaveClass('to-purple-900/30');
    expect(card).toHaveClass('border-blue-500/20');
  });

  it('handles explanation with special characters', () => {
    const explanation = {
      explanation: 'The EUR/USD pair shows a 5% gain. VIX < 15 suggests "low volatility" & calm markets.',
      cached: false,
    };

    render(<ExplanationCard explanation={explanation} />);

    expect(screen.getByText(/EUR\/USD pair shows a 5% gain/)).toBeInTheDocument();
    expect(screen.getByText(/VIX < 15 suggests "low volatility"/)).toBeInTheDocument();
  });

  it('handles cached indicator toggle correctly', () => {
    const explanation = {
      explanation: 'Market analysis.',
      cached: true,
    };

    const { rerender } = render(<ExplanationCard explanation={explanation} />);
    expect(screen.getByText('cached')).toBeInTheDocument();

    // Update to non-cached
    rerender(<ExplanationCard explanation={{ ...explanation, cached: false }} />);
    expect(screen.queryByText('cached')).not.toBeInTheDocument();
  });

  it('renders with all props at default values', () => {
    const { container } = render(<ExplanationCard />);
    expect(container.firstChild).toBeNull();
  });

  it('renders Sparkles icon in normal state', () => {
    const explanation = {
      explanation: 'Test explanation.',
      cached: false,
    };

    render(<ExplanationCard explanation={explanation} />);

    // Check for Sparkles icon (by checking for text-blue-400 class)
    const sparklesIcon = document.querySelector('.text-blue-400');
    expect(sparklesIcon).toBeInTheDocument();
  });
});
