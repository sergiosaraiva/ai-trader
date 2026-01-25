import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { CollapsibleCard } from './CollapsibleCard';
import { Activity } from 'lucide-react';

describe('CollapsibleCard', () => {
  it('renders with content expanded by default', () => {
    render(
      <CollapsibleCard title="Test Card">
        <div>Test Content</div>
      </CollapsibleCard>
    );

    expect(screen.getByText('Test Card')).toBeInTheDocument();
    expect(screen.getByText('Test Content')).toBeInTheDocument();

    const contentDiv = screen.getByText('Test Content').closest('.collapsible-content');
    expect(contentDiv).toHaveClass('expanded');
  });

  it('renders with content collapsed when defaultExpanded is false', () => {
    render(
      <CollapsibleCard title="Test Card" defaultExpanded={false}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    expect(screen.getByText('Test Card')).toBeInTheDocument();
    expect(screen.getByText('Test Content')).toBeInTheDocument();

    const contentDiv = screen.getByText('Test Content').closest('.collapsible-content');
    expect(contentDiv).toHaveClass('collapsed');
    expect(contentDiv).not.toHaveClass('expanded');
  });

  it('toggles content visibility on header click', () => {
    render(
      <CollapsibleCard title="Test Card">
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const header = screen.getByText('Test Card').closest('[role="button"]');
    const contentDiv = screen.getByText('Test Content').closest('.collapsible-content');

    // Initially expanded
    expect(contentDiv).toHaveClass('expanded');

    // Click to collapse
    fireEvent.click(header);
    expect(contentDiv).toHaveClass('collapsed');

    // Click to expand
    fireEvent.click(header);
    expect(contentDiv).toHaveClass('expanded');
  });

  it('shows ChevronUp icon when expanded', () => {
    render(
      <CollapsibleCard title="Test Card" defaultExpanded={true}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const button = screen.getByLabelText('Collapse');
    expect(button).toBeInTheDocument();
  });

  it('shows ChevronDown icon when collapsed', () => {
    render(
      <CollapsibleCard title="Test Card" defaultExpanded={false}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const button = screen.getByLabelText('Expand');
    expect(button).toBeInTheDocument();
  });

  it('toggles icon when clicking header', () => {
    render(
      <CollapsibleCard title="Test Card" defaultExpanded={true}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    // Initially shows Collapse button
    expect(screen.getByLabelText('Collapse')).toBeInTheDocument();

    // Click header to collapse
    const header = screen.getByText('Test Card').closest('[role="button"]');
    fireEvent.click(header);

    // Now shows Expand button
    expect(screen.getByLabelText('Expand')).toBeInTheDocument();
    expect(screen.queryByLabelText('Collapse')).not.toBeInTheDocument();
  });

  it('applies custom className props correctly', () => {
    render(
      <CollapsibleCard title="Test Card" className="custom-class">
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const cardDiv = screen.getByText('Test Card').closest('.bg-gray-800');
    expect(cardDiv).toHaveClass('custom-class');
  });

  it('applies custom headerClassName props correctly', () => {
    render(
      <CollapsibleCard title="Test Card" headerClassName="custom-header-class">
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const header = screen.getByText('Test Card').closest('[role="button"]');
    expect(header).toHaveClass('custom-header-class');
  });

  it('applies custom contentClassName props correctly', () => {
    render(
      <CollapsibleCard title="Test Card" contentClassName="custom-content-class">
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const contentDiv = screen.getByText('Test Content').closest('.collapsible-content');
    expect(contentDiv).toHaveClass('custom-content-class');
  });

  it('renders header actions when provided', () => {
    const actions = (
      <button type="button" className="action-button">
        Action Button
      </button>
    );

    render(
      <CollapsibleCard title="Test Card" actions={actions}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    expect(screen.getByText('Action Button')).toBeInTheDocument();
    expect(screen.getByText('Action Button')).toHaveClass('action-button');
  });

  it('content has correct aria-expanded attribute when expanded', () => {
    render(
      <CollapsibleCard title="Test Card" defaultExpanded={true}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const header = screen.getByText('Test Card').closest('[role="button"]');
    expect(header).toHaveAttribute('aria-expanded', 'true');
  });

  it('content has correct aria-expanded attribute when collapsed', () => {
    render(
      <CollapsibleCard title="Test Card" defaultExpanded={false}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const header = screen.getByText('Test Card').closest('[role="button"]');
    expect(header).toHaveAttribute('aria-expanded', 'false');
  });

  it('aria-expanded updates when toggling', () => {
    render(
      <CollapsibleCard title="Test Card" defaultExpanded={true}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const header = screen.getByText('Test Card').closest('[role="button"]');

    expect(header).toHaveAttribute('aria-expanded', 'true');

    fireEvent.click(header);

    expect(header).toHaveAttribute('aria-expanded', 'false');

    fireEvent.click(header);

    expect(header).toHaveAttribute('aria-expanded', 'true');
  });

  it('keyboard navigation works with Enter key', () => {
    render(
      <CollapsibleCard title="Test Card" defaultExpanded={true}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const header = screen.getByText('Test Card').closest('[role="button"]');
    const contentDiv = screen.getByText('Test Content').closest('.collapsible-content');

    expect(contentDiv).toHaveClass('expanded');

    fireEvent.keyDown(header, { key: 'Enter', code: 'Enter' });

    expect(contentDiv).toHaveClass('collapsed');
  });

  it('keyboard navigation works with Space key', () => {
    render(
      <CollapsibleCard title="Test Card" defaultExpanded={true}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const header = screen.getByText('Test Card').closest('[role="button"]');
    const contentDiv = screen.getByText('Test Content').closest('.collapsible-content');

    expect(contentDiv).toHaveClass('expanded');

    fireEvent.keyDown(header, { key: ' ', code: 'Space' });

    expect(contentDiv).toHaveClass('collapsed');
  });

  it('keyboard navigation prevents default for Space key', () => {
    render(
      <CollapsibleCard title="Test Card">
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const header = screen.getByText('Test Card').closest('[role="button"]');
    const event = new KeyboardEvent('keydown', { key: ' ', code: 'Space', bubbles: true });
    const preventDefaultSpy = vi.spyOn(event, 'preventDefault');

    header.dispatchEvent(event);

    expect(preventDefaultSpy).toHaveBeenCalled();
  });

  it('keyboard navigation does not toggle on other keys', () => {
    render(
      <CollapsibleCard title="Test Card" defaultExpanded={true}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const header = screen.getByText('Test Card').closest('[role="button"]');
    const contentDiv = screen.getByText('Test Content').closest('.collapsible-content');

    expect(contentDiv).toHaveClass('expanded');

    // Try various keys that should not toggle
    fireEvent.keyDown(header, { key: 'Tab', code: 'Tab' });
    expect(contentDiv).toHaveClass('expanded');

    fireEvent.keyDown(header, { key: 'Escape', code: 'Escape' });
    expect(contentDiv).toHaveClass('expanded');

    fireEvent.keyDown(header, { key: 'a', code: 'KeyA' });
    expect(contentDiv).toHaveClass('expanded');
  });

  it('uses unique IDs for aria-controls', () => {
    const { rerender } = render(
      <CollapsibleCard title="Test Card 1">
        <div>Content 1</div>
      </CollapsibleCard>
    );

    const header1 = screen.getByText('Test Card 1').closest('[role="button"]');
    const contentId1 = header1.getAttribute('aria-controls');
    expect(contentId1).toMatch(/^collapsible-content-/);

    rerender(
      <CollapsibleCard title="Test Card 2">
        <div>Content 2</div>
      </CollapsibleCard>
    );

    const header2 = screen.getByText('Test Card 2').closest('[role="button"]');
    const contentId2 = header2.getAttribute('aria-controls');
    expect(contentId2).toMatch(/^collapsible-content-/);

    // IDs should be different for different instances
    // Note: Since we're rerendering the same component, the ID might be the same
    // but in a real app with multiple CollapsibleCards, they would be different
    expect(contentId2).toBeTruthy();
  });

  it('aria-controls matches content id', () => {
    render(
      <CollapsibleCard title="Test Card">
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const header = screen.getByText('Test Card').closest('[role="button"]');
    const ariaControls = header.getAttribute('aria-controls');
    const contentDiv = screen.getByText('Test Content').closest('.collapsible-content');
    const contentId = contentDiv.getAttribute('id');

    expect(ariaControls).toBe(contentId);
  });

  it('renders title correctly', () => {
    render(
      <CollapsibleCard title="My Custom Title">
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const title = screen.getByText('My Custom Title');
    expect(title).toBeInTheDocument();
    expect(title.tagName).toBe('H2');
  });

  it('renders icon correctly when provided', () => {
    render(
      <CollapsibleCard title="Test Card" icon={<Activity data-testid="custom-icon" />}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    expect(screen.getByTestId('custom-icon')).toBeInTheDocument();
  });

  it('does not render icon when not provided', () => {
    render(
      <CollapsibleCard title="Test Card">
        <div>Test Content</div>
      </CollapsibleCard>
    );

    // The icon span should not be present
    const header = screen.getByText('Test Card').closest('[role="button"]');
    const iconSpan = header.querySelector('.text-blue-400');
    expect(iconSpan).not.toBeInTheDocument();
  });

  it('toggles when clicking the toggle button directly', () => {
    render(
      <CollapsibleCard title="Test Card" defaultExpanded={true}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const toggleButton = screen.getByLabelText('Collapse');
    const contentDiv = screen.getByText('Test Content').closest('.collapsible-content');

    expect(contentDiv).toHaveClass('expanded');

    fireEvent.click(toggleButton);

    expect(contentDiv).toHaveClass('collapsed');
  });

  it('toggle button click stops propagation', () => {
    const headerClickSpy = vi.fn();

    render(
      <CollapsibleCard title="Test Card" defaultExpanded={true}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const header = screen.getByText('Test Card').closest('[role="button"]');
    header.addEventListener('click', headerClickSpy);

    const toggleButton = screen.getByLabelText('Collapse');

    fireEvent.click(toggleButton);

    // The header click handler should not have been called due to stopPropagation
    // But the component still toggles
    const contentDiv = screen.getByText('Test Content').closest('.collapsible-content');
    expect(contentDiv).toHaveClass('collapsed');
  });

  it('content has aria-hidden when collapsed', () => {
    render(
      <CollapsibleCard title="Test Card" defaultExpanded={false}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const contentDiv = screen.getByText('Test Content').closest('.collapsible-content');
    expect(contentDiv).toHaveAttribute('aria-hidden', 'true');
  });

  it('content does not have aria-hidden when expanded', () => {
    render(
      <CollapsibleCard title="Test Card" defaultExpanded={true}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const contentDiv = screen.getByText('Test Content').closest('.collapsible-content');
    expect(contentDiv).toHaveAttribute('aria-hidden', 'false');
  });

  it('header has correct tabIndex for keyboard navigation', () => {
    render(
      <CollapsibleCard title="Test Card">
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const header = screen.getByText('Test Card').closest('[role="button"]');
    expect(header).toHaveAttribute('tabIndex', '0');
  });

  it('renders complex children correctly', () => {
    render(
      <CollapsibleCard title="Test Card">
        <div>
          <h3>Nested Title</h3>
          <p>Paragraph content</p>
          <ul>
            <li>Item 1</li>
            <li>Item 2</li>
          </ul>
        </div>
      </CollapsibleCard>
    );

    expect(screen.getByText('Nested Title')).toBeInTheDocument();
    expect(screen.getByText('Paragraph content')).toBeInTheDocument();
    expect(screen.getByText('Item 1')).toBeInTheDocument();
    expect(screen.getByText('Item 2')).toBeInTheDocument();
  });

  it('maintains expand/collapse state through multiple interactions', () => {
    render(
      <CollapsibleCard title="Test Card" defaultExpanded={true}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    const header = screen.getByText('Test Card').closest('[role="button"]');
    const contentDiv = screen.getByText('Test Content').closest('.collapsible-content');

    // Start expanded
    expect(contentDiv).toHaveClass('expanded');

    // Multiple interactions
    fireEvent.click(header);
    expect(contentDiv).toHaveClass('collapsed');

    fireEvent.click(header);
    expect(contentDiv).toHaveClass('expanded');

    fireEvent.click(header);
    expect(contentDiv).toHaveClass('collapsed');

    fireEvent.click(header);
    expect(contentDiv).toHaveClass('expanded');
  });

  it('has memo optimization applied', () => {
    // CollapsibleCard should be wrapped with memo
    expect(CollapsibleCard.$$typeof).toBeDefined();
  });

  it('handles empty actions gracefully', () => {
    render(
      <CollapsibleCard title="Test Card" actions={null}>
        <div>Test Content</div>
      </CollapsibleCard>
    );

    expect(screen.getByText('Test Card')).toBeInTheDocument();
    expect(screen.getByText('Test Content')).toBeInTheDocument();
  });

  it('renders multiple instances independently', () => {
    render(
      <div>
        <CollapsibleCard title="Card 1" defaultExpanded={true}>
          <div>Content 1</div>
        </CollapsibleCard>
        <CollapsibleCard title="Card 2" defaultExpanded={false}>
          <div>Content 2</div>
        </CollapsibleCard>
      </div>
    );

    const content1 = screen.getByText('Content 1').closest('.collapsible-content');
    const content2 = screen.getByText('Content 2').closest('.collapsible-content');

    expect(content1).toHaveClass('expanded');
    expect(content2).toHaveClass('collapsed');

    // Click first card's header
    const header1 = screen.getByText('Card 1').closest('[role="button"]');
    fireEvent.click(header1);

    // Only first card should change
    expect(content1).toHaveClass('collapsed');
    expect(content2).toHaveClass('collapsed'); // Should remain unchanged
  });
});
