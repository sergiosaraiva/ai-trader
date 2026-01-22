import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ModelHighlights } from './ModelHighlights';

describe('ModelHighlights', () => {
  // Loading State Tests
  it('renders loading state', () => {
    render(<ModelHighlights loading={true} />);
    // Check for skeleton loader (animate-pulse class)
    const skeleton = document.querySelector('.animate-pulse');
    expect(skeleton).toBeInTheDocument();
  });

  it('renders skeleton with 4 placeholder cards during loading', () => {
    render(<ModelHighlights loading={true} />);
    // Check for 4 skeleton cards
    const skeletonCards = document.querySelectorAll('.bg-gray-700\\/50');
    expect(skeletonCards.length).toBeGreaterThanOrEqual(4);
  });

  // Error State Tests
  it('renders error state', () => {
    render(<ModelHighlights error="Test error message" />);
    expect(screen.getByText('Performance data unavailable')).toBeInTheDocument();
  });

  it('renders error state with default message', () => {
    render(<ModelHighlights error="Test error" />);
    expect(screen.getByText('Using default highlights')).toBeInTheDocument();
  });

  it('renders error state with AlertCircle icon', () => {
    const { container } = render(<ModelHighlights error="Test error" />);
    // Check for yellow-400 text color (error indication)
    const errorElement = container.querySelector('.text-yellow-400');
    expect(errorElement).toBeInTheDocument();
  });

  // Default Highlights Tests
  it('renders default highlights when no performance data provided', () => {
    render(<ModelHighlights />);

    // Check for default highlights
    expect(screen.getByText('High-Confidence Trading')).toBeInTheDocument();
    expect(screen.getByText('Model Consensus')).toBeInTheDocument();
    expect(screen.getByText('Walk-Forward Validated')).toBeInTheDocument();
    expect(screen.getByText('All-Regime Profitable')).toBeInTheDocument();
  });

  it('renders default highlight values', () => {
    render(<ModelHighlights />);

    expect(screen.getByText('62.1%')).toBeInTheDocument();
    expect(screen.getByText('82%')).toBeInTheDocument();
    expect(screen.getByText('7/7')).toBeInTheDocument();
    expect(screen.getByText('6/6')).toBeInTheDocument();
  });

  it('renders default summary headline', () => {
    render(<ModelHighlights />);

    expect(screen.getByText('Solid Performance')).toBeInTheDocument();
  });

  it('renders default summary description', () => {
    render(<ModelHighlights />);

    const description = screen.getByText(/The MTF Ensemble model demonstrates solid performance/);
    expect(description).toBeInTheDocument();
    expect(description.textContent).toContain('58.6%');
    expect(description.textContent).toContain('2.26x profit factor');
  });

  // Custom Performance Data Tests
  it('renders custom highlights from performance data', () => {
    const performance = {
      highlights: [
        {
          type: 'confidence',
          title: 'Custom Confidence',
          value: '75.0%',
          description: 'Custom confidence description',
        },
        {
          type: 'agreement',
          title: 'Custom Agreement',
          value: '90%',
          description: 'Custom agreement description',
        },
      ],
      summary: {
        headline: 'Custom Headline',
        description: 'Custom description',
      },
    };

    render(<ModelHighlights performance={performance} />);

    expect(screen.getByText('Custom Confidence')).toBeInTheDocument();
    expect(screen.getByText('75.0%')).toBeInTheDocument();
    expect(screen.getByText('Custom Agreement')).toBeInTheDocument();
    expect(screen.getByText('90%')).toBeInTheDocument();
  });

  it('renders custom summary headline and description', () => {
    const performance = {
      highlights: [],
      summary: {
        headline: 'Excellent Performance',
        description: 'Model is performing exceptionally well with 65% win rate.',
      },
    };

    render(<ModelHighlights performance={performance} />);

    expect(screen.getByText('Excellent Performance')).toBeInTheDocument();
    expect(screen.getByText(/Model is performing exceptionally well/)).toBeInTheDocument();
  });

  // Highlight Card Structure Tests
  it('renders all 4 highlight cards', () => {
    render(<ModelHighlights />);

    // Grid should have 4 cards
    const highlightCards = document.querySelectorAll('.bg-gray-700\\/50.rounded-lg');
    expect(highlightCards.length).toBe(4);
  });

  it('renders highlight descriptions', () => {
    render(<ModelHighlights />);

    expect(screen.getByText('Win rate when model confidence exceeds 70%')).toBeInTheDocument();
    expect(screen.getByText('Accuracy when all 3 timeframes agree')).toBeInTheDocument();
    expect(screen.getByText('Profitable across all test periods')).toBeInTheDocument();
    expect(screen.getByText('Works in trending and ranging markets')).toBeInTheDocument();
  });

  // Icon Color Tests
  it('renders confidence highlight with green color', () => {
    const { container } = render(<ModelHighlights />);

    // Check for green-400 text (TrendingUp icon for confidence)
    const greenIcons = container.querySelectorAll('.text-green-400');
    expect(greenIcons.length).toBeGreaterThan(0);
  });

  it('renders agreement highlight with blue color', () => {
    const { container } = render(<ModelHighlights />);

    // Check for blue-400 text (Target icon for agreement)
    const blueIcons = container.querySelectorAll('.text-blue-400');
    expect(blueIcons.length).toBeGreaterThan(0);
  });

  it('renders validation highlight with purple color', () => {
    const { container } = render(<ModelHighlights />);

    // Check for purple-400 text (CheckCircle icon for validation)
    const purpleIcons = container.querySelectorAll('.text-purple-400');
    expect(purpleIcons.length).toBeGreaterThan(0);
  });

  it('renders robustness highlight with orange color', () => {
    const { container } = render(<ModelHighlights />);

    // Check for orange-400 text (Shield icon for robustness)
    const orangeIcons = container.querySelectorAll('.text-orange-400');
    expect(orangeIcons.length).toBeGreaterThan(0);
  });

  // Footer Tests
  it('renders footer with trade count', () => {
    render(<ModelHighlights />);

    expect(screen.getByText(/Metrics based on/)).toBeInTheDocument();
    expect(screen.getByText(/1,093 trades/)).toBeInTheDocument();
  });

  it('renders footer with custom trade count from performance', () => {
    const performance = {
      metrics: {
        total_trades: 2500,
      },
      highlights: [],
      summary: { headline: 'Test', description: 'Test' },
    };

    render(<ModelHighlights performance={performance} />);

    // The component displays numbers without comma formatting
    expect(screen.getByText(/2500 trades/)).toBeInTheDocument();
  });

  it('renders confidence threshold in footer', () => {
    render(<ModelHighlights />);

    expect(screen.getByText(/70% confidence threshold/)).toBeInTheDocument();
  });

  // Dynamic Summary Tests
  it('displays different summary headlines based on performance', () => {
    const excellentPerformance = {
      highlights: [],
      summary: {
        headline: 'Excellent Performance',
        description: 'Test',
      },
    };

    const { rerender } = render(<ModelHighlights performance={excellentPerformance} />);
    expect(screen.getByText('Excellent Performance')).toBeInTheDocument();

    const moderatePerformance = {
      highlights: [],
      summary: {
        headline: 'Moderate Performance',
        description: 'Test',
      },
    };

    rerender(<ModelHighlights performance={moderatePerformance} />);
    expect(screen.getByText('Moderate Performance')).toBeInTheDocument();
  });

  // Hover Effects Tests
  it('applies hover effect classes to highlight cards', () => {
    const { container } = render(<ModelHighlights />);

    // Check for hover:bg-gray-700/70 transition-colors classes
    const hoverCards = container.querySelectorAll('.hover\\:bg-gray-700\\/70');
    expect(hoverCards.length).toBeGreaterThanOrEqual(4);
  });

  // Edge Cases
  it('handles empty highlights array', () => {
    const performance = {
      highlights: [],
      summary: { headline: 'Test', description: 'Test' },
    };

    // Should not crash and render defaults
    expect(() => {
      render(<ModelHighlights performance={performance} />);
    }).not.toThrow();
  });

  it('handles null performance gracefully', () => {
    render(<ModelHighlights performance={null} />);

    // Should render defaults
    expect(screen.getByText('High-Confidence Trading')).toBeInTheDocument();
  });

  it('handles undefined summary', () => {
    const performance = {
      highlights: [],
      summary: undefined,
    };

    render(<ModelHighlights performance={performance} />);

    // Should render default summary
    expect(screen.getByText('Solid Performance')).toBeInTheDocument();
  });

  it('handles partial performance data', () => {
    const performance = {
      highlights: [
        {
          type: 'confidence',
          title: 'Custom Test Title',
          value: '70%',
          description: 'Custom test description',
        },
      ],
      // Missing summary
    };

    render(<ModelHighlights performance={performance} />);

    // Should render provided highlight
    expect(screen.getByText('Custom Test Title')).toBeInTheDocument();
    expect(screen.getByText('70%')).toBeInTheDocument();

    // Should render default summary
    expect(screen.getByText('Solid Performance')).toBeInTheDocument();
  });

  // Integration Tests
  it('renders complete component with all sections', () => {
    const performance = {
      metrics: {
        total_pips: 8693,
        win_rate: 0.621,
        profit_factor: 2.69,
        total_trades: 966,
      },
      highlights: [
        {
          type: 'confidence',
          title: 'High-Confidence Trading',
          value: '62.1%',
          description: 'Win rate when model confidence exceeds 70%',
        },
        {
          type: 'agreement',
          title: 'Model Consensus',
          value: '82%',
          description: 'Accuracy when all 3 timeframes agree',
        },
        {
          type: 'validation',
          title: 'Walk-Forward Validated',
          value: '7/7',
          description: 'Profitable across all test periods',
        },
        {
          type: 'robustness',
          title: 'All-Regime Profitable',
          value: '6/6',
          description: 'Works in trending and ranging markets',
        },
      ],
      summary: {
        headline: 'Excellent Performance',
        description: 'The MTF Ensemble model demonstrates excellent performance with 62.1% overall win rate and 2.69x profit factor. High-confidence predictions (≥70%) achieve 62.1% accuracy. Walk-forward optimization confirms 100% consistency across all test periods.',
      },
    };

    render(<ModelHighlights performance={performance} />);

    // Verify header section
    expect(screen.getByText('Excellent Performance')).toBeInTheDocument();
    expect(screen.getByText(/62.1% overall win rate/)).toBeInTheDocument();

    // Verify all 4 highlights
    expect(screen.getByText('High-Confidence Trading')).toBeInTheDocument();
    expect(screen.getByText('Model Consensus')).toBeInTheDocument();
    expect(screen.getByText('Walk-Forward Validated')).toBeInTheDocument();
    expect(screen.getByText('All-Regime Profitable')).toBeInTheDocument();

    // Verify footer
    expect(screen.getByText(/966 trades/)).toBeInTheDocument();
  });

  it('renders with card-hover class for animation', () => {
    const { container } = render(<ModelHighlights />);

    // Check for card-hover class on main container
    const mainCard = container.querySelector('.card-hover');
    expect(mainCard).toBeInTheDocument();
  });

  // Accessibility Tests
  it('renders semantic HTML structure', () => {
    const { container } = render(<ModelHighlights />);

    // Check for h2 heading
    const heading = container.querySelector('h2');
    expect(heading).toBeInTheDocument();

    // Check for paragraphs
    const paragraphs = container.querySelectorAll('p');
    expect(paragraphs.length).toBeGreaterThan(0);
  });

  it('renders highlight titles with proper hierarchy', () => {
    const { container } = render(<ModelHighlights />);

    // Check for h3 headings in highlights
    const h3Headings = container.querySelectorAll('h3');
    expect(h3Headings.length).toBe(4); // One per highlight
  });
});
