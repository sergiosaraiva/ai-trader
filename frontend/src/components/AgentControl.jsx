import { useState, useRef, useEffect } from 'react';
import { Play, Square, Pause, PlayCircle, AlertTriangle, Settings, CheckCircle } from 'lucide-react';
import PropTypes from 'prop-types';
import { api } from '../api/client';

/**
 * Sanitize user input by stripping HTML tags and limiting length
 * @param {string} input - User input to sanitize
 * @param {number} maxLength - Maximum allowed length
 * @returns {string} Sanitized input
 */
function sanitizeInput(input, maxLength = 500) {
  if (!input || typeof input !== 'string') return '';

  let sanitized = input;
  // Strip HTML tags
  sanitized = sanitized.replace(/<[^>]*>/g, '');
  // Remove javascript: URIs
  sanitized = sanitized.replace(/javascript:/gi, '');
  // Remove event handlers (onclick=, onerror=, onload=, etc.)
  sanitized = sanitized.replace(/on\w+\s*=/gi, '');
  // Remove data: URIs (can execute code)
  sanitized = sanitized.replace(/data:/gi, '');

  return sanitized.trim().slice(0, maxLength);
}

/**
 * AgentControl - Main agent control panel component
 */
export function AgentControl({ status, safety, loading, onRefresh }) {
  // Track timeout IDs for cleanup on unmount
  const timeoutIdsRef = useRef([]);

  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showConfig, setShowConfig] = useState(false);
  const [config, setConfig] = useState({
    mode: 'simulation',
    confidence_threshold: 0.70,
    cycle_interval_seconds: 300,
    max_position_size: 1.0,
    use_kelly_sizing: false,
  });

  // Cleanup timeouts on unmount
  useEffect(() => {
    return () => {
      // Clear all pending timeouts when component unmounts
      timeoutIdsRef.current.forEach(timeoutId => clearTimeout(timeoutId));
      timeoutIdsRef.current = [];
    };
  }, []);

  // Helper to schedule a timeout and track it for cleanup
  const scheduleTimeout = (callback, delay) => {
    const timeoutId = setTimeout(() => {
      // Remove from tracking array when it executes
      timeoutIdsRef.current = timeoutIdsRef.current.filter(id => id !== timeoutId);
      callback();
    }, delay);
    timeoutIdsRef.current.push(timeoutId);
    return timeoutId;
  };

  const agentStatus = status?.status || 'stopped';
  const isRunning = agentStatus === 'running';
  const isPaused = agentStatus === 'paused';
  const isStopped = agentStatus === 'stopped';
  const killSwitchActive = safety?.kill_switch?.is_active || false;

  // Handle start agent
  const handleStart = async () => {
    if (config.mode === 'live') {
      const confirmed = window.confirm(
        '⚠️ WARNING: Starting in LIVE mode will trade with real money. Are you sure?'
      );
      if (!confirmed) return;
    }

    setActionLoading(true);
    setError(null);

    try {
      const response = await api.startAgent(config);
      console.log('Start command queued:', response);
      scheduleTimeout(() => onRefresh(), 2000);
    } catch (err) {
      setError(err.message || 'Failed to start agent');
    } finally {
      setActionLoading(false);
    }
  };

  // Handle stop agent
  const handleStop = async (closePositions = false) => {
    if (closePositions) {
      const confirmed = window.confirm(
        'This will close all open positions. Are you sure?'
      );
      if (!confirmed) return;
    }

    setActionLoading(true);
    setError(null);

    try {
      const response = await api.stopAgent({
        force: false,
        close_positions: closePositions,
      });
      console.log('Stop command queued:', response);
      scheduleTimeout(() => onRefresh(), 2000);
    } catch (err) {
      setError(err.message || 'Failed to stop agent');
    } finally {
      setActionLoading(false);
    }
  };

  // Handle pause agent
  const handlePause = async () => {
    setActionLoading(true);
    setError(null);

    try {
      const response = await api.pauseAgent();
      console.log('Pause command queued:', response);
      scheduleTimeout(() => onRefresh(), 2000);
    } catch (err) {
      setError(err.message || 'Failed to pause agent');
    } finally {
      setActionLoading(false);
    }
  };

  // Handle resume agent
  const handleResume = async () => {
    setActionLoading(true);
    setError(null);

    try {
      const response = await api.resumeAgent();
      console.log('Resume command queued:', response);
      scheduleTimeout(() => onRefresh(), 2000);
    } catch (err) {
      setError(err.message || 'Failed to resume agent');
    } finally {
      setActionLoading(false);
    }
  };

  // Handle kill switch
  const handleKillSwitch = async () => {
    const rawReason = window.prompt(
      '⚠️ KILL SWITCH\n\nThis will immediately halt all trading and close positions.\n\nEnter reason for activation:'
    );

    if (!rawReason) return;

    // Sanitize user input to prevent XSS
    const reason = sanitizeInput(rawReason, 200);

    if (!reason) {
      setError('Invalid reason provided');
      return;
    }

    setActionLoading(true);
    setError(null);

    try {
      const response = await api.triggerKillSwitch(reason);
      console.log('Kill switch triggered:', response);
      scheduleTimeout(() => onRefresh(), 2000);
    } catch (err) {
      setError(err.message || 'Failed to trigger kill switch');
    } finally {
      setActionLoading(false);
    }
  };

  // Handle config update
  const handleUpdateConfig = async () => {
    setActionLoading(true);
    setError(null);

    try {
      const response = await api.updateAgentConfig(config);
      console.log('Config update queued:', response);
      setShowConfig(false);
      scheduleTimeout(() => onRefresh(), 2000);
    } catch (err) {
      setError(err.message || 'Failed to update config');
    } finally {
      setActionLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-3"></div>
        <div className="space-y-2">
          <div className="h-10 bg-gray-700 rounded"></div>
          <div className="h-10 bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-base font-semibold text-gray-300 mb-3">Controls</h3>
      {/* Error Display */}
      {error && (
        <div className="mb-3 p-2 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-xs">
          {error}
        </div>
      )}

      {/* Status Indicator */}
      <div className="mb-3 p-2 bg-gray-700/50 rounded">
        <div className="flex items-center gap-2">
          <StatusBadge status={agentStatus} killSwitchActive={killSwitchActive} />
          <div>
            <span className="text-xs text-gray-300 block">
              {killSwitchActive ? 'KILL SWITCH ACTIVE' : agentStatus.toUpperCase()}
            </span>
            {status?.mode && (
              <span className="text-xs text-gray-500">
                Mode: {status.mode}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Configuration Button */}
      <div className="mb-3">
        <button
          onClick={() => setShowConfig(!showConfig)}
          className="w-full flex items-center justify-center gap-2 p-1.5 hover:bg-gray-700 rounded transition-colors text-xs text-gray-300"
        >
          <Settings size={14} className="text-gray-400" />
          {showConfig ? 'Hide Configuration' : 'Show Configuration'}
        </button>
      </div>

      {/* Configuration Panel */}
      {showConfig && (
        <div className="mb-3 p-3 bg-gray-700/30 rounded space-y-2">
          <h3 className="text-sm font-medium text-gray-300 mb-2">Configuration</h3>

          {/* Mode Selector */}
          <div>
            <label className="block text-xs text-gray-500 mb-1">Mode</label>
            <select
              value={config.mode}
              onChange={(e) => setConfig({ ...config, mode: e.target.value })}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm text-gray-200"
              disabled={!isStopped}
            >
              <option value="simulation">Simulation</option>
              <option value="paper">Paper Trading</option>
              <option value="live">Live Trading</option>
            </select>
          </div>

          {/* Confidence Threshold */}
          <div>
            <label className="block text-xs text-gray-500 mb-1">
              Confidence Threshold: {(config.confidence_threshold * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.50"
              max="0.85"
              step="0.05"
              value={config.confidence_threshold}
              onChange={(e) => setConfig({ ...config, confidence_threshold: parseFloat(e.target.value) })}
              className="w-full"
            />
          </div>

          {/* Cycle Interval */}
          <div>
            <label className="block text-xs text-gray-500 mb-1">Cycle Interval (seconds)</label>
            <input
              type="number"
              min="60"
              max="3600"
              step="60"
              value={config.cycle_interval_seconds}
              onChange={(e) => setConfig({ ...config, cycle_interval_seconds: parseInt(e.target.value) })}
              className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-sm text-gray-200"
            />
          </div>

          {/* Update Config Button (only when running) */}
          {!isStopped && (
            <button
              onClick={handleUpdateConfig}
              disabled={actionLoading}
              className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded transition-colors text-sm"
            >
              {actionLoading ? 'Updating...' : 'Update Config'}
            </button>
          )}
        </div>
      )}

      {/* Control Buttons */}
      <div className="space-y-2">
        {isStopped && !killSwitchActive && (
          <button
            onClick={handleStart}
            disabled={actionLoading}
            className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded transition-colors font-medium"
          >
            <Play size={18} />
            {actionLoading ? 'Starting...' : 'Start Agent'}
          </button>
        )}

        {isRunning && (
          <>
            <button
              onClick={handlePause}
              disabled={actionLoading}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-600 rounded transition-colors"
            >
              <Pause size={18} />
              {actionLoading ? 'Pausing...' : 'Pause'}
            </button>
            <button
              onClick={() => handleStop(false)}
              disabled={actionLoading}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 disabled:bg-gray-500 rounded transition-colors"
            >
              <Square size={18} />
              {actionLoading ? 'Stopping...' : 'Stop'}
            </button>
            <button
              onClick={() => handleStop(true)}
              disabled={actionLoading}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-orange-600 hover:bg-orange-700 disabled:bg-gray-600 rounded transition-colors text-sm"
            >
              <Square size={16} />
              {actionLoading ? 'Stopping...' : 'Stop & Close Positions'}
            </button>
          </>
        )}

        {isPaused && (
          <>
            <button
              onClick={handleResume}
              disabled={actionLoading}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded transition-colors font-medium"
            >
              <PlayCircle size={18} />
              {actionLoading ? 'Resuming...' : 'Resume'}
            </button>
            <button
              onClick={() => handleStop(false)}
              disabled={actionLoading}
              className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 disabled:bg-gray-500 rounded transition-colors"
            >
              <Square size={18} />
              {actionLoading ? 'Stopping...' : 'Stop'}
            </button>
          </>
        )}

        {/* Kill Switch Button */}
        {!killSwitchActive && !isStopped && (
          <button
            onClick={handleKillSwitch}
            disabled={actionLoading}
            className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 rounded transition-colors text-sm border-2 border-red-500"
          >
            <AlertTriangle size={16} />
            {actionLoading ? 'Activating...' : 'KILL SWITCH'}
          </button>
        )}
      </div>
    </div>
  );
}

/**
 * StatusBadge - Visual indicator of agent status
 */
function StatusBadge({ status, killSwitchActive }) {
  if (killSwitchActive) {
    return (
      <div className="w-3 h-3 rounded-full bg-red-500 animate-pulse" title="Kill Switch Active" />
    );
  }

  const colors = {
    running: 'bg-green-500',
    paused: 'bg-yellow-500',
    stopped: 'bg-gray-500',
    error: 'bg-red-500',
  };

  const color = colors[status] || 'bg-gray-500';

  return (
    <div className={`w-3 h-3 rounded-full ${color} ${status === 'running' ? 'animate-pulse' : ''}`} title={status} />
  );
}

StatusBadge.propTypes = {
  status: PropTypes.string.isRequired,
  killSwitchActive: PropTypes.bool.isRequired,
};

AgentControl.propTypes = {
  status: PropTypes.shape({
    status: PropTypes.string,
    mode: PropTypes.string,
    uptime_seconds: PropTypes.number,
    cycle_count: PropTypes.number,
  }),
  safety: PropTypes.shape({
    kill_switch: PropTypes.shape({
      is_active: PropTypes.bool,
    }),
    circuit_breakers: PropTypes.object,
  }),
  loading: PropTypes.bool.isRequired,
  onRefresh: PropTypes.func.isRequired,
};

export default AgentControl;
