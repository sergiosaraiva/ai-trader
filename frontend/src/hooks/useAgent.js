import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '../api/client';

/**
 * Custom hook for agent state management with auto-refresh
 * Uses the /agent/health endpoint which always returns (never 404)
 * @param {number} interval - Polling interval in milliseconds (default: 5000)
 * @returns {Object} - { status, safety, health, loading, error, refetch, isInitialized }
 */
export function useAgent(interval = 5000) {
  const [health, setHealth] = useState(null);
  const [status, setStatus] = useState(null);
  const [safety, setSafety] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const intervalRef = useRef(null);

  const fetchData = useCallback(async (isInitial = false) => {
    if (isInitial) {
      setLoading(true);
    }

    try {
      // First fetch health - this always returns (never 404)
      const healthData = await api.getAgentHealth().catch(() => null);
      setHealth(healthData);

      // Only fetch status/safety if agent is initialized
      if (healthData && healthData.status !== 'not_initialized') {
        const [statusData, safetyData] = await Promise.all([
          api.getAgentStatus().catch(() => null),
          api.getAgentSafety().catch(() => null),
        ]);
        setStatus(statusData);
        setSafety(safetyData);
      } else {
        // Agent not initialized - set null without making 404 requests
        setStatus(null);
        setSafety(null);
      }

      setError(null);
    } catch (err) {
      setError(err.message || 'Failed to fetch agent data');
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
  }, [interval, fetchData]);

  // Derive isInitialized from health status
  const isInitialized = health && health.status !== 'not_initialized';

  return { status, safety, health, loading, error, refetch, isInitialized };
}

export default useAgent;
