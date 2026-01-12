import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Custom hook for polling data with automatic refresh
 * @param {Function} fetchFn - Async function to fetch data
 * @param {number} interval - Polling interval in milliseconds
 * @param {boolean} enabled - Whether polling is enabled
 * @returns {Object} - { data, loading, error, refetch, lastUpdated }
 */
export function usePolling(fetchFn, interval = 30000, enabled = true) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const intervalRef = useRef(null);
  const fetchFnRef = useRef(fetchFn);

  // Keep fetchFn reference updated
  useEffect(() => {
    fetchFnRef.current = fetchFn;
  }, [fetchFn]);

  const fetchData = useCallback(async (isInitial = false) => {
    if (isInitial) {
      setLoading(true);
    }

    try {
      const result = await fetchFnRef.current();
      setData(result);
      setError(null);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err.message || 'Failed to fetch data');
      // Don't clear existing data on error for better UX
    } finally {
      if (isInitial) {
        setLoading(false);
      }
    }
  }, []);

  const refetch = useCallback(() => {
    return fetchData(false);
  }, [fetchData]);

  useEffect(() => {
    if (!enabled) {
      return;
    }

    // Initial fetch
    fetchData(true);

    // Set up polling
    if (interval > 0) {
      intervalRef.current = setInterval(() => {
        fetchData(false);
      }, interval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [enabled, interval, fetchData]);

  return { data, loading, error, refetch, lastUpdated };
}

/**
 * Custom hook for fetching data once (no polling)
 * @param {Function} fetchFn - Async function to fetch data
 * @param {Array} deps - Dependencies array
 * @returns {Object} - { data, loading, error, refetch }
 */
export function useFetch(fetchFn, deps = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const result = await fetchFn();
      setData(result);
      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to fetch data');
    } finally {
      setLoading(false);
    }
  }, [fetchFn]);

  useEffect(() => {
    fetchData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { data, loading, error, refetch: fetchData };
}

export default usePolling;
