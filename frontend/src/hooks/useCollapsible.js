import { useState, useCallback } from 'react';

/**
 * Custom hook for managing collapsible panel state
 * @param {boolean} defaultExpanded - Initial expanded state (default: true)
 * @returns {object} - { isExpanded, toggle, setExpanded }
 */
export function useCollapsible(defaultExpanded = true) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded);

  const toggle = useCallback(() => {
    setIsExpanded((prev) => !prev);
  }, []);

  return {
    isExpanded,
    toggle,
    setExpanded: setIsExpanded,
  };
}

export default useCollapsible;
