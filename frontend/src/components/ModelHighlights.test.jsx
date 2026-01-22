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

  it('renders error state with error message', () => {
    render(<ModelHighlights error="Test error" />);
    expect(screen.getByText('Test error')).toBeInTheDocument();
  });

  it('renders error state with AlertCircle icon', () => {
    const { container } = render(<ModelHighlights error="Test error" />);
    // Check for yellow-400 text color (error indication)
    const errorElement = container.querySelector('.text-yellow-400');
    expect(errorElement).toBeInTheDocument();
  });

  // Empty State Tests (no data)
  it('renders empty state when no performance data provided', () => {
    render(<ModelHighlights />);

    // Should show empty state message
    expect(screen.getByText('No performance highlights available')).toBeInTheDocument();
    expect(screen.getByText('Performance metrics are being generated. Please check back shortly.')).toBeInTheDocument();
  });

  it('renders empty state when performance is null', () => {
    render(<ModelHighlights performance={null} />);

    // Should render empty state
    expect(screen.getByText('No performance highlights available')).toBeInTheDocument();
  });

  it('renders empty state when highlights array is empty', () => {
    const performance = {
      highlights: [],
      summary: { headline: 'Test', description: 'Test' },
    };

    render(<ModelHighlights performance={performance} />);

    // Should show empty state
    expect(screen.getByText('No performance highlights available')).toBeInTheDocument();
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
          status: 'excellent',
        },
        {
          type: 'agreement',
          title: 'Custom Agreement',
          value: '90%',
          description: 'Custom agreement description',
          status: 'excellent',
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
      highlights: [
        { type: 'test', title: 'Test', value: '100%', description: 'Test', status: 'excellent' },
      ],
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
  it('renders correct number of highlight cards based on data', () => {
    const performance = {
      highlights: [
        { type: 'agreement', title: 'Model Agreement', value: '82%', description: 'Test', status: 'excellent' },
        { type: 'validation', title: 'WFO Validation', value: '8/8', description: 'Test', status: 'excellent' },
        { type: 'robustness', title: 'All Conditions', value: '6/6', description: 'Test', status: 'excellent' },
        { type: 'returns', title: 'Profit Factor', value: '1.57x', description: 'Test', status: 'moderate' },
      ],
      summary: { headline: 'Test', description: 'Test' },
    };

    render(<ModelHighlights performance={performance} />);

    // Grid should have 4 cards
    const highlightCards = document.querySelectorAll('.bg-gray-700\\/50.rounded-lg');
    expect(highlightCards.length).toBe(4);
  });

  it('renders highlight descriptions', () => {
    const performance = {
      highlights: [
        { type: 'agreement', title: 'Model Agreement', value: '82%', description: 'Accuracy when all 3 timeframes align', status: 'excellent' },
        { type: 'validation', title: 'WFO Validation', value: '8/8', description: 'Profitable across all test periods', status: 'excellent' },
        { type: 'robustness', title: 'All Conditions', value: '6/6', description: 'Works in any market regime', status: 'excellent' },
        { type: 'returns', title: 'Profit Factor', value: '1.57x', description: 'Returns $1.57 for every $1 risked', status: 'moderate' },
      ],
      summary: { headline: 'Test', description: 'Test' },
    };

    render(<ModelHighlights performance={performance} />);

    expect(screen.getByText('Accuracy when all 3 timeframes align')).toBeInTheDocument();
    expect(screen.getByText('Profitable across all test periods')).toBeInTheDocument();
    expect(screen.getByText('Works in any market regime')).toBeInTheDocument();
  });

  // Semantic Color Tests (based on status)
  it('renders excellent status highlights with green color', () => {
    const performance = {
      highlights: [
        { type: 'agreement', title: 'Model Agreement', value: '82%', description: 'Test', status: 'excellent' },
      ],
      summary: { headline: 'Test', description: 'Test' },
    };

    const { container } = render(<ModelHighlights performance={performance} />);

    // Check for green-400 text (excellent status)
    const greenIcons = container.querySelectorAll('.text-green-400');
    expect(greenIcons.length).toBeGreaterThan(0);
  });

  it('renders good status highlights with blue color', () => {
    const performance = {
      highlights: [
        { type: 'agreement', title: 'Model Agreement', value: '65%', description: 'Test', status: 'good' },
      ],
      summary: { headline: 'Test', description: 'Test' },
    };

    const { container } = render(<ModelHighlights performance={performance} />);

    // Check for blue-400 text (good status)
    const blueIcons = container.querySelectorAll('.text-blue-400');
    expect(blueIcons.length).toBeGreaterThan(0);
  });

  it('renders moderate status highlights with yellow color', () => {
    const performance = {
      highlights: [
        { type: 'returns', title: 'Profit Factor', value: '1.57x', description: 'Test', status: 'moderate' },
      ],
      summary: { headline: 'Test', description: 'Test' },
    };

    const { container } = render(<ModelHighlights performance={performance} />);

    // Check for yellow-400 text (moderate status)
    const yellowIcons = container.querySelectorAll('.text-yellow-400');
    expect(yellowIcons.length).toBeGreaterThan(0);
  });

  it('renders poor status highlights with red color', () => {
    const performance = {
      highlights: [
        { type: 'returns', title: 'Profit Factor', value: '0.9x', description: 'Test', status: 'poor' },
      ],
      summary: { headline: 'Test', description: 'Test' },
    };

    const { container } = render(<ModelHighlights performance={performance} />);

    // Check for red-400 text (poor status)
    const redIcons = container.querySelectorAll('.text-red-400');
    expect(redIcons.length).toBeGreaterThan(0);
  });

  // Footer Tests
  it('renders footer with N/A when no metrics available', () => {
    const performance = {
      highlights: [
        { type: 'test', title: 'Test', value: '100%', description: 'Test', status: 'excellent' },
      ],
      summary: { headline: 'Test', description: 'Test' },
    };

    render(<ModelHighlights performance={performance} />);

    expect(screen.getByText(/Metrics based on/)).toBeInTheDocument();
    expect(screen.getByText(/N\/A/)).toBeInTheDocument();
  });

  it('renders footer with formatted trade count from performance', () => {
    const performance = {
      metrics: {
        total_trades: 3821,
      },
      highlights: [
        { type: 'test', title: 'Test', value: '100%', description: 'Test', status: 'excellent' },
      ],
      summary: { headline: 'Test', description: 'Test' },
    };

    render(<ModelHighlights performance={performance} />);

    // The component formats with toLocaleString()
    expect(screen.getByText(/3,821 trades/)).toBeInTheDocument();
  });

  it('renders confidence threshold in footer from metrics', () => {
    const performance = {
      metrics: {
        high_confidence: { threshold: 0.70 },
      },
      highlights: [
        { type: 'test', title: 'Test', value: '100%', description: 'Test', status: 'excellent' },
      ],
      summary: { headline: 'Test', description: 'Test' },
    };

    render(<ModelHighlights performance={performance} />);

    expect(screen.getByText(/70% confidence threshold/)).toBeInTheDocument();
  });

  it('renders "high" when threshold not available', () => {
    const performance = {
      highlights: [
        { type: 'test', title: 'Test', value: '100%', description: 'Test', status: 'excellent' },
      ],
      summary: { headline: 'Test', description: 'Test' },
    };

    render(<ModelHighlights performance={performance} />);

    expect(screen.getByText(/high confidence threshold/)).toBeInTheDocument();
  });

  // Dynamic Summary Tests
  it('displays different summary headlines based on performance', () => {
    const excellentPerformance = {
      highlights: [
        { type: 'test', title: 'Test', value: '100%', description: 'Test', status: 'excellent' },
      ],
      summary: {
        headline: 'Excellent Performance',
        description: 'Test',
      },
    };

    const { rerender } = render(<ModelHighlights performance={excellentPerformance} />);
    expect(screen.getByText('Excellent Performance')).toBeInTheDocument();

    const moderatePerformance = {
      highlights: [
        { type: 'test', title: 'Test', value: '100%', description: 'Test', status: 'excellent' },
      ],
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
    const performance = {
      highlights: [
        { type: 'agreement', title: 'Model Agreement', value: '82%', description: 'Test', status: 'excellent' },
        { type: 'validation', title: 'WFO Validation', value: '8/8', description: 'Test', status: 'excellent' },
        { type: 'robustness', title: 'All Conditions', value: '6/6', description: 'Test', status: 'excellent' },
        { type: 'returns', title: 'Profit Factor', value: '1.57x', description: 'Test', status: 'moderate' },
      ],
      summary: { headline: 'Test', description: 'Test' },
    };

    const { container } = render(<ModelHighlights performance={performance} />);

    // Check for hover:bg-gray-700/70 transition-colors classes
    const hoverCards = container.querySelectorAll('.hover\\:bg-gray-700\\/70');
    expect(hoverCards.length).toBeGreaterThanOrEqual(4);
  });

  // Edge Cases
  it('handles empty highlights array without crash', () => {
    const performance = {
      highlights: [],
      summary: { headline: 'Test', description: 'Test' },
    };

    // Should not crash and render empty state
    expect(() => {
      render(<ModelHighlights performance={performance} />);
    }).not.toThrow();
  });

  it('handles partial performance data', () => {
    const performance = {
      highlights: [
        {
          type: 'confidence',
          title: 'Custom Test Title',
          value: '70%',
          description: 'Custom test description',
          status: 'good',
        },
      ],
      // Missing summary - should use default
    };

    render(<ModelHighlights performance={performance} />);

    // Should render provided highlight
    expect(screen.getByText('Custom Test Title')).toBeInTheDocument();
    expect(screen.getByText('70%')).toBeInTheDocument();

    // Should render default summary
    expect(screen.getByText('Model Performance')).toBeInTheDocument();
  });

  // Integration Tests
  it('renders complete component with all sections', () => {
    const performance = {
      metrics: {
        total_pips: 14837,
        win_rate: 0.5064,
        profit_factor: 1.57,
        total_trades: 3821,
      },
      highlights: [
        {
          type: 'agreement',
          title: 'Model Agreement',
          value: '82%',
          description: 'Accuracy when all 3 timeframes align',
          status: 'excellent',
        },
        {
          type: 'validation',
          title: 'WFO Validation',
          value: '8/8 Windows Profitable',
          description: 'Profitable across all test periods',
          status: 'excellent',
        },
        {
          type: 'robustness',
          title: 'All Conditions',
          value: '6/6 Regimes',
          description: 'Works in any market regime',
          status: 'excellent',
        },
        {
          type: 'returns',
          title: 'Profit Factor',
          value: '1.57x',
          description: 'Returns $1.57 for every $1 risked',
          status: 'moderate',
        },
      ],
      summary: {
        headline: 'Solid Performance',
        description: 'The MTF Ensemble model demonstrates solid performance with 50.6% overall win rate and 1.57x profit factor.',
      },
    };

    render(<ModelHighlights performance={performance} />);

    // Verify header section
    expect(screen.getByText('Solid Performance')).toBeInTheDocument();
    expect(screen.getByText(/50.6% overall win rate/)).toBeInTheDocument();

    // Verify all 4 highlights
    expect(screen.getByText('Model Agreement')).toBeInTheDocument();
    expect(screen.getByText('WFO Validation')).toBeInTheDocument();
    expect(screen.getByText('All Conditions')).toBeInTheDocument();
    expect(screen.getByText('Profit Factor')).toBeInTheDocument();

    // Verify footer
    expect(screen.getByText(/3,821 trades/)).toBeInTheDocument();
  });

  it('renders with card-hover class for animation', () => {
    const performance = {
      highlights: [
        { type: 'test', title: 'Test', value: '100%', description: 'Test', status: 'excellent' },
      ],
      summary: { headline: 'Test', description: 'Test' },
    };

    const { container } = render(<ModelHighlights performance={performance} />);

    // Check for card-hover class on main container
    const mainCard = container.querySelector('.card-hover');
    expect(mainCard).toBeInTheDocument();
  });

  // Accessibility Tests
  it('renders semantic HTML structure', () => {
    const performance = {
      highlights: [
        { type: 'test', title: 'Test', value: '100%', description: 'Test', status: 'excellent' },
      ],
      summary: { headline: 'Test', description: 'Test' },
    };

    const { container } = render(<ModelHighlights performance={performance} />);

    // Check for h2 heading
    const heading = container.querySelector('h2');
    expect(heading).toBeInTheDocument();

    // Check for paragraphs
    const paragraphs = container.querySelectorAll('p');
    expect(paragraphs.length).toBeGreaterThan(0);
  });

  it('renders highlight titles with proper hierarchy', () => {
    const performance = {
      highlights: [
        { type: 'agreement', title: 'Model Agreement', value: '82%', description: 'Test', status: 'excellent' },
        { type: 'validation', title: 'WFO Validation', value: '8/8', description: 'Test', status: 'excellent' },
        { type: 'robustness', title: 'All Conditions', value: '6/6', description: 'Test', status: 'excellent' },
        { type: 'returns', title: 'Profit Factor', value: '1.57x', description: 'Test', status: 'moderate' },
      ],
      summary: { headline: 'Test', description: 'Test' },
    };

    const { container } = render(<ModelHighlights performance={performance} />);

    // Check for h3 headings in highlights
    const h3Headings = container.querySelectorAll('h3');
    expect(h3Headings.length).toBe(4); // One per highlight
  });
});
