import { useState, useRef, useEffect } from 'react';
import { Play, Square, Pause, PlayCircle, AlertTriangle, Settings, Activity, Target, BarChart3, TrendingUp, Shield, Clock } from 'lucide-react';
import PropTypes from 'prop-types';
import { api } from '../api/client';
import { CollapsibleCard } from './common/CollapsibleCard';

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
 * Format time difference from ISO timestamp
 */
function formatTimeDiff(isoString) {
  if (!isoString) return 'Never';

  const date = new Date(isoString);
  const now = new Date();
  const diff = now - date;

  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) return `${hours}h ${minutes % 60}m ago`;
  if (minutes > 0) return `${minutes}m ago`;
  return 'Just now';
}

/**
 * AgentControlPanel - Unified agent control panel component
 */
export function AgentControlPanel({ status, safety, loading, onRefresh }) {
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
      timeoutIdsRef.current.forEach(timeoutId => clearTimeout(timeoutId));
      timeoutIdsRef.current = [];
    };
  }, []);

  // Helper to schedule a timeout and track it for cleanup
  const scheduleTimeout = (callback, delay) => {
    const timeoutId = setTimeout(() => {
      timeoutIdsRef.current = timeoutIdsRef.current.filter(id => id !== timeoutId);
      callback();
    }, delay);
    timeoutIdsRef.current.push(timeoutId);
    return timeoutId;
  };

  const agentStatus = status?.status || 'stopped';
  const mode = status?.mode || 'simulation';
  const cycleCount = status?.cycle_count || 0;
  const lastCycleAt = status?.last_cycle_at;
  const openPositions = status?.open_positions || 0;
  const isRunning = agentStatus === 'running';
  const isPaused = agentStatus === 'paused';
  const isStopped = agentStatus === 'stopped';
  const killSwitchActive = safety?.kill_switch?.is_active || false;
  const canTrade = safety?.circuit_breakers?.can_trade ?? true;

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
      <div className="bg-gray-800 rounded-lg p-6 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
          {[1, 2, 3, 4, 5, 6].map((i) => (
            <div key={i} className="bg-gray-700/50 rounded-lg p-3">
              <div className="h-3 bg-gray-600 rounded w-2/3 mb-2"></div>
              <div className="h-6 bg-gray-600 rounded w-1/2"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Handle 404 gracefully (agent not initialized)
  if (!status) {
    return (
      <CollapsibleCard
        title="Agent Control"
        icon={<Play size={18} />}
        className="card-hover"
        defaultExpanded={false}
      >
        <div className="text-center py-6">
          <p className="text-gray-400 text-sm mb-2">Agent not initialized</p>
          <p className="text-gray-600 text-xs mb-4">Start the agent to begin trading</p>
          {error && (
            <div className="mb-3 p-2 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-xs">
              {error}
            </div>
          )}
          <button
            onClick={handleStart}
            disabled={actionLoading}
            className="inline-flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded transition-colors font-medium"
          >
            <Play size={18} />
            {actionLoading ? 'Starting...' : 'Start Agent'}
          </button>
        </div>
      </CollapsibleCard>
    );
  }

  // Get status color
  const getStatusColor = (status) => {
    switch (status) {
      case 'running':
        return 'text-green-400';
      case 'paused':
        return 'text-yellow-400';
      case 'stopped':
        return 'text-gray-400';
      case 'error':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  // Get mode color
  const getModeColor = (mode) => {
    switch (mode) {
      case 'live':
        return 'text-red-400';
      case 'paper':
        return 'text-yellow-400';
      case 'simulation':
        return 'text-blue-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <CollapsibleCard
      title="Agent Control"
      icon={<Play size={18} />}
      className="card-hover"
      defaultExpanded={isRunning || isPaused}
    >
      {/* Brief description */}
      <p className="text-xs text-gray-400 mb-3">
        Autonomous trading agent status and controls
      </p>

      {/* Error Display */}
      {error && (
        <div className="mb-3 p-2 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-xs">
          {error}
        </div>
      )}

      {/* Status grid - similar to ModelHighlights */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
        {/* Agent Status */}
        <div className="bg-gray-700/50 rounded-lg p-2 hover:bg-gray-700/70 transition-colors">
          <div className="flex items-center gap-1.5 mb-1">
            <Activity size={14} className={getStatusColor(agentStatus)} />
            <span className="text-xs text-gray-400">Status</span>
          </div>
          <div className={`text-sm font-semibold ${getStatusColor(agentStatus)}`}>
            {killSwitchActive ? 'KILLED' : agentStatus.toUpperCase()}
          </div>
        </div>

        {/* Mode */}
        <div className="bg-gray-700/50 rounded-lg p-2 hover:bg-gray-700/70 transition-colors">
          <div className="flex items-center gap-1.5 mb-1">
            <Target size={14} className="text-gray-400" />
            <span className="text-xs text-gray-400">Mode</span>
          </div>
          <div className={`text-sm font-semibold ${getModeColor(mode)}`}>
            {mode.toUpperCase()}
          </div>
        </div>

        {/* Cycle Count */}
        <div className="bg-gray-700/50 rounded-lg p-2 hover:bg-gray-700/70 transition-colors">
          <div className="flex items-center gap-1.5 mb-1">
            <BarChart3 size={14} className="text-gray-400" />
            <span className="text-xs text-gray-400">Cycles</span>
          </div>
          <div className="text-sm font-semibold text-blue-400">
            {cycleCount.toLocaleString()}
          </div>
        </div>

        {/* Kill Switch */}
        <div className="bg-gray-700/50 rounded-lg p-2 hover:bg-gray-700/70 transition-colors">
          <div className="flex items-center gap-1.5 mb-1">
            <AlertTriangle size={14} className={killSwitchActive ? 'text-red-400' : 'text-gray-400'} />
            <span className="text-xs text-gray-400">Kill Switch</span>
          </div>
          <div className={`text-sm font-semibold ${killSwitchActive ? 'text-red-400 animate-pulse' : 'text-green-400'}`}>
            {killSwitchActive ? 'ACTIVE' : 'Inactive'}
          </div>
        </div>

        {/* Circuit Breaker */}
        <div className="bg-gray-700/50 rounded-lg p-2 hover:bg-gray-700/70 transition-colors">
          <div className="flex items-center gap-1.5 mb-1">
            <Shield size={14} className={canTrade ? 'text-green-400' : 'text-red-400'} />
            <span className="text-xs text-gray-400">Breaker</span>
          </div>
          <div className={`text-sm font-semibold ${canTrade ? 'text-green-400' : 'text-red-400'}`}>
            {canTrade ? 'OK' : 'TRIPPED'}
          </div>
        </div>

        {/* Open Positions */}
        <div className="bg-gray-700/50 rounded-lg p-2 hover:bg-gray-700/70 transition-colors">
          <div className="flex items-center gap-1.5 mb-1">
            <TrendingUp size={14} className="text-gray-400" />
            <span className="text-xs text-gray-400">Positions</span>
          </div>
          <div className={`text-sm font-semibold ${openPositions > 0 ? 'text-blue-400' : 'text-gray-500'}`}>
            {openPositions}
          </div>
        </div>
      </div>

      {/* Last Cycle Time - subtle footer */}
      {lastCycleAt && (
        <div className="mt-2 pt-2 border-t border-gray-700">
          <div className="flex items-center justify-center gap-1.5 text-xs text-gray-500">
            <Clock size={12} />
            <span>Last cycle: {formatTimeDiff(lastCycleAt)}</span>
          </div>
        </div>
      )}

      {/* Configuration Panel - Collapsible */}
      {showConfig && (
        <div className="mt-3 p-3 bg-gray-700/30 rounded space-y-2">
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

      {/* Control buttons - compact row */}
      <div className="flex items-center gap-2 mt-3 pt-3 border-t border-gray-700 flex-wrap">
        {/* Start Button */}
        {isStopped && !killSwitchActive && (
          <button
            onClick={handleStart}
            disabled={actionLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded transition-colors text-sm"
          >
            <Play size={14} />
            {actionLoading ? 'Starting...' : 'Start'}
          </button>
        )}

        {/* Pause Button */}
        {isRunning && (
          <button
            onClick={handlePause}
            disabled={actionLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-600 rounded transition-colors text-sm"
          >
            <Pause size={14} />
            {actionLoading ? 'Pausing...' : 'Pause'}
          </button>
        )}

        {/* Resume Button */}
        {isPaused && (
          <button
            onClick={handleResume}
            disabled={actionLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded transition-colors text-sm"
          >
            <PlayCircle size={14} />
            {actionLoading ? 'Resuming...' : 'Resume'}
          </button>
        )}

        {/* Stop Button */}
        {!isStopped && (
          <button
            onClick={() => handleStop(false)}
            disabled={actionLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-600 hover:bg-gray-700 disabled:bg-gray-500 rounded transition-colors text-sm"
          >
            <Square size={14} />
            {actionLoading ? 'Stopping...' : 'Stop'}
          </button>
        )}

        {/* Stop & Close Button */}
        {!isStopped && (
          <button
            onClick={() => handleStop(true)}
            disabled={actionLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-orange-600 hover:bg-orange-700 disabled:bg-gray-600 rounded transition-colors text-sm"
          >
            <Square size={14} />
            {actionLoading ? 'Stopping...' : 'Stop & Close'}
          </button>
        )}

        {/* Kill Switch Button */}
        {!killSwitchActive && !isStopped && (
          <button
            onClick={handleKillSwitch}
            disabled={actionLoading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 rounded transition-colors text-sm border border-red-500"
          >
            <AlertTriangle size={14} />
            {actionLoading ? 'Activating...' : 'KILL'}
          </button>
        )}

        {/* Config Button */}
        <button
          onClick={() => setShowConfig(!showConfig)}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-gray-600 hover:bg-gray-700 rounded transition-colors text-sm ml-auto"
        >
          <Settings size={14} />
          Config
        </button>
      </div>
    </CollapsibleCard>
  );
}

AgentControlPanel.propTypes = {
  status: PropTypes.shape({
    status: PropTypes.string,
    mode: PropTypes.string,
    uptime_seconds: PropTypes.number,
    cycle_count: PropTypes.number,
    last_cycle_at: PropTypes.string,
    open_positions: PropTypes.number,
  }),
  safety: PropTypes.shape({
    kill_switch: PropTypes.shape({
      is_active: PropTypes.bool,
    }),
    circuit_breakers: PropTypes.shape({
      overall_state: PropTypes.string,
      can_trade: PropTypes.bool,
    }),
  }),
  loading: PropTypes.bool.isRequired,
  onRefresh: PropTypes.func.isRequired,
};

export default AgentControlPanel;
