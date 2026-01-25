import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useCollapsible } from './useCollapsible';

describe('useCollapsible', () => {
  it('initializes with defaultExpanded value true by default', () => {
    const { result } = renderHook(() => useCollapsible());

    expect(result.current.isExpanded).toBe(true);
    expect(typeof result.current.toggle).toBe('function');
    expect(typeof result.current.setExpanded).toBe('function');
  });

  it('initializes with custom defaultExpanded value false', () => {
    const { result } = renderHook(() => useCollapsible(false));

    expect(result.current.isExpanded).toBe(false);
  });

  it('initializes with custom defaultExpanded value true', () => {
    const { result } = renderHook(() => useCollapsible(true));

    expect(result.current.isExpanded).toBe(true);
  });

  it('toggle function switches isExpanded state from true to false', () => {
    const { result } = renderHook(() => useCollapsible(true));

    expect(result.current.isExpanded).toBe(true);

    act(() => {
      result.current.toggle();
    });

    expect(result.current.isExpanded).toBe(false);
  });

  it('toggle function switches isExpanded state from false to true', () => {
    const { result } = renderHook(() => useCollapsible(false));

    expect(result.current.isExpanded).toBe(false);

    act(() => {
      result.current.toggle();
    });

    expect(result.current.isExpanded).toBe(true);
  });

  it('toggle function can be called multiple times', () => {
    const { result } = renderHook(() => useCollapsible(true));

    expect(result.current.isExpanded).toBe(true);

    act(() => {
      result.current.toggle();
    });
    expect(result.current.isExpanded).toBe(false);

    act(() => {
      result.current.toggle();
    });
    expect(result.current.isExpanded).toBe(true);

    act(() => {
      result.current.toggle();
    });
    expect(result.current.isExpanded).toBe(false);
  });

  it('setExpanded directly sets the state to true', () => {
    const { result } = renderHook(() => useCollapsible(false));

    expect(result.current.isExpanded).toBe(false);

    act(() => {
      result.current.setExpanded(true);
    });

    expect(result.current.isExpanded).toBe(true);
  });

  it('setExpanded directly sets the state to false', () => {
    const { result } = renderHook(() => useCollapsible(true));

    expect(result.current.isExpanded).toBe(true);

    act(() => {
      result.current.setExpanded(false);
    });

    expect(result.current.isExpanded).toBe(false);
  });

  it('setExpanded can override toggle', () => {
    const { result } = renderHook(() => useCollapsible(true));

    act(() => {
      result.current.toggle();
    });
    expect(result.current.isExpanded).toBe(false);

    act(() => {
      result.current.setExpanded(true);
    });
    expect(result.current.isExpanded).toBe(true);
  });

  it('multiple hook instances maintain independent state', () => {
    const { result: result1 } = renderHook(() => useCollapsible(true));
    const { result: result2 } = renderHook(() => useCollapsible(false));

    expect(result1.current.isExpanded).toBe(true);
    expect(result2.current.isExpanded).toBe(false);

    act(() => {
      result1.current.toggle();
    });

    expect(result1.current.isExpanded).toBe(false);
    expect(result2.current.isExpanded).toBe(false); // Should not change

    act(() => {
      result2.current.toggle();
    });

    expect(result1.current.isExpanded).toBe(false); // Should not change
    expect(result2.current.isExpanded).toBe(true);
  });

  it('toggle function maintains reference stability', () => {
    const { result, rerender } = renderHook(() => useCollapsible(true));

    const firstToggle = result.current.toggle;

    // Re-render the hook
    rerender();

    const secondToggle = result.current.toggle;

    // Toggle function should maintain the same reference (useCallback)
    expect(firstToggle).toBe(secondToggle);
  });

  it('setExpanded is the same as setIsExpanded', () => {
    const { result } = renderHook(() => useCollapsible(true));

    act(() => {
      result.current.setExpanded(false);
    });

    expect(result.current.isExpanded).toBe(false);
  });

  it('returns object with expected properties', () => {
    const { result } = renderHook(() => useCollapsible());

    expect(result.current).toHaveProperty('isExpanded');
    expect(result.current).toHaveProperty('toggle');
    expect(result.current).toHaveProperty('setExpanded');
    expect(Object.keys(result.current)).toHaveLength(3);
  });

  it('handles rapid toggle calls', () => {
    const { result } = renderHook(() => useCollapsible(true));

    act(() => {
      result.current.toggle();
      result.current.toggle();
      result.current.toggle();
    });

    // After 3 toggles (true -> false -> true -> false)
    expect(result.current.isExpanded).toBe(false);
  });
});
