import { useState, useEffect } from 'react';
import { Activity, Wifi, WifiOff, Server, Database, Clock } from 'lucide-react';

/**
 * Format time difference from a timestamp
 */
function formatTimeDiff(ts, now) {
  if (!ts) return 'N/A';
  const diff = now - new Date(ts).getTime();
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days}d ${hours % 24}h ago`;
  if (hours > 0) return `${hours}h ${minutes % 60}m ago`;
  if (minutes > 0) return `${minutes}m ago`;
  return 'Just now';
}

/**
 * AccountStatus - Displays system status and data pipeline health
 */
export function AccountStatus({ pipelineStatus, modelStatus, loading, error }) {
  // State for current time, updated periodically for relative time display
  const [now, setNow] = useState(Date.now);

  // Update the current time every minute for the relative time display
  useEffect(() => {
    const timer = setInterval(() => {
      setNow(Date.now());
    }, 60000);
    return () => clearInterval(timer);
  }, []);

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="space-y-3">
          <div className="h-8 bg-gray-700 rounded"></div>
          <div className="h-8 bg-gray-700 rounded"></div>
          <div className="h-8 bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  const getStatusColor = (status) => {
    if (status === 'healthy' || status === 'loaded' || status === true) return 'text-green-400';
    if (status === 'warning' || status === 'stale') return 'text-yellow-400';
    return 'text-red-400';
  };

  const getStatusIcon = (status) => {
    if (status === 'healthy' || status === 'loaded' || status === true) {
      return <Wifi size={16} className="text-green-400" />;
    }
    return <WifiOff size={16} className="text-red-400" />;
  };

  const formatTime = (ts) => {
    if (!ts) return 'Never';
    const date = new Date(ts);
    return date.toLocaleString();
  };

  // Extract data from pipeline status
  const pipeline = pipelineStatus || {};
  const model = modelStatus || {};

  return (
    <div className="bg-gray-800 rounded-lg p-6 card-hover">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold text-gray-300">System Status</h2>
        <Activity size={20} className={error ? 'text-red-400' : 'text-green-400'} />
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-sm">
          {error}
        </div>
      )}

      <div className="space-y-4">
        {/* Pipeline Status */}
        <div className="flex items-center justify-between p-3 bg-gray-700/50 rounded">
          <div className="flex items-center gap-3">
            <Database size={18} className="text-gray-400" />
            <div>
              <span className="text-sm text-gray-300">Data Pipeline</span>
              <p className="text-xs text-gray-500">
                {pipeline.last_run ? formatTimeDiff(pipeline.last_run, now) : 'Not initialized'}
              </p>
            </div>
          </div>
          {getStatusIcon(pipeline.status || 'unknown')}
        </div>

        {/* Model Status */}
        <div className="flex items-center justify-between p-3 bg-gray-700/50 rounded">
          <div className="flex items-center gap-3">
            <Server size={18} className="text-gray-400" />
            <div>
              <span className="text-sm text-gray-300">ML Models</span>
              <p className="text-xs text-gray-500">
                {model.models_loaded ? 'All models loaded' : 'Not loaded'}
              </p>
            </div>
          </div>
          {getStatusIcon(model.models_loaded ? 'loaded' : 'error')}
        </div>

        {/* Data Quality */}
        {pipeline.data_quality && (
          <div className="mt-4 pt-4 border-t border-gray-700">
            <h3 className="text-sm text-gray-400 mb-3">Data Quality</h3>
            <div className="grid grid-cols-3 gap-2">
              {Object.entries(pipeline.data_quality).map(([tf, quality]) => (
                <div key={tf} className="text-center p-2 bg-gray-700/30 rounded">
                  <span className="text-xs text-gray-500 block">{tf}</span>
                  <span className={`text-sm font-medium ${getStatusColor(quality?.status)}`}>
                    {quality?.rows || 0} rows
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Last Update */}
        <div className="flex items-center gap-2 text-xs text-gray-500 mt-4">
          <Clock size={12} />
          <span>Last update: {formatTime(pipeline.last_run)}</span>
        </div>
      </div>
    </div>
  );
}

export default AccountStatus;
