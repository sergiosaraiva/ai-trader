import { Activity, Clock, TrendingUp, BarChart3, Target } from 'lucide-react';

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
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days}d ${hours % 24}h ago`;
  if (hours > 0) return `${hours}h ${minutes % 60}m ago`;
  if (minutes > 0) return `${minutes}m ago`;
  return 'Just now';
}

/**
 * Format uptime in seconds
 */
function formatUptime(seconds) {
  if (!seconds) return 'N/A';

  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);

  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

/**
 * AgentStatus - Detailed status display component
 */
export function AgentStatus({ status, loading }) {
  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-3"></div>
        <div className="space-y-2">
          <div className="h-8 bg-gray-700 rounded"></div>
          <div className="h-8 bg-gray-700 rounded"></div>
          <div className="h-8 bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <h2 className="text-base font-semibold text-gray-300 mb-3">Agent Status</h2>
        <div className="text-center py-6">
          <p className="text-gray-500 text-sm">Agent not initialized</p>
          <p className="text-gray-600 text-xs mt-1">Start the agent to see status</p>
        </div>
      </div>
    );
  }

  const agentStatus = status.status || 'unknown';
  const mode = status.mode || 'N/A';
  const cycleCount = status.cycle_count || 0;
  const lastCycleAt = status.last_cycle_at;
  const openPositions = status.open_positions || 0;
  const accountEquity = status.account_equity || 0;
  const uptime = status.uptime_seconds;
  const lastPrediction = status.last_prediction;

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-base font-semibold text-gray-300 mb-3">Agent Status</h3>
      {/* Status Grid */}
      <div className="space-y-2">
        {/* Status & Mode */}
        <MetricRow
          icon={<Activity size={16} className="text-gray-400" />}
          label="Status"
          value={
            <span className={getStatusColor(agentStatus)}>
              {agentStatus.toUpperCase()}
            </span>
          }
        />

        <MetricRow
          icon={<Target size={16} className="text-gray-400" />}
          label="Mode"
          value={
            <span className={getModeColor(mode)}>
              {mode.toUpperCase()}
            </span>
          }
        />

        {/* Cycle Count */}
        <MetricRow
          icon={<BarChart3 size={16} className="text-gray-400" />}
          label="Cycles"
          value={cycleCount.toLocaleString()}
        />

        {/* Last Cycle */}
        <MetricRow
          icon={<Clock size={16} className="text-gray-400" />}
          label="Last Cycle"
          value={formatTimeDiff(lastCycleAt)}
        />

        {/* Uptime */}
        {uptime && (
          <MetricRow
            icon={<Clock size={16} className="text-gray-400" />}
            label="Uptime"
            value={formatUptime(uptime)}
          />
        )}

        {/* Open Positions */}
        <MetricRow
          icon={<TrendingUp size={16} className="text-gray-400" />}
          label="Open Positions"
          value={
            <span className={openPositions > 0 ? 'text-blue-400' : 'text-gray-500'}>
              {openPositions}
            </span>
          }
        />

        {/* Account Equity */}
        {accountEquity > 0 && (
          <MetricRow
            icon={<TrendingUp size={16} className="text-gray-400" />}
            label="Account Equity"
            value={
              <span className="text-green-400">
                ${accountEquity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
            }
          />
        )}

        {/* Last Prediction Summary */}
        {lastPrediction && (
          <div className="mt-3 pt-3 border-t border-gray-700">
            <h3 className="text-xs text-gray-500 mb-2">Last Prediction</h3>
            <div className="bg-gray-700/30 rounded p-2">
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Signal:</span>
                <span className={`text-xs font-medium ${
                  lastPrediction.signal === 'BUY' ? 'text-green-400' :
                  lastPrediction.signal === 'SELL' ? 'text-red-400' :
                  'text-gray-400'
                }`}>
                  {lastPrediction.signal}
                </span>
              </div>
              <div className="flex justify-between items-center mt-1">
                <span className="text-xs text-gray-400">Confidence:</span>
                <span className="text-xs text-blue-400">
                  {(lastPrediction.confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * MetricRow - Reusable metric display component
 */
function MetricRow({ icon, label, value }) {
  return (
    <div className="flex items-center justify-between p-2 bg-gray-700/50 rounded">
      <div className="flex items-center gap-2">
        {icon}
        <span className="text-xs text-gray-300">{label}</span>
      </div>
      <span className="text-xs font-medium text-gray-200">{value}</span>
    </div>
  );
}

/**
 * Get status color class
 */
function getStatusColor(status) {
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
}

/**
 * Get mode color class
 */
function getModeColor(mode) {
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
}

export default AgentStatus;
