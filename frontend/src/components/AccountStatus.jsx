import { useState, useEffect } from 'react';
import { Activity, Wifi, WifiOff, Server, Database, Clock, Cpu } from 'lucide-react';
import { CollapsibleCard } from './common/CollapsibleCard';

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
  // Initialize with a function to avoid calling Date.now() during render
  const [now, setNow] = useState(() => Date.now());

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
    if (status === 'healthy' || status === 'ok' || status === 'loaded' || status === true) return 'text-green-400';
    if (status === 'warning' || status === 'stale') return 'text-yellow-400';
    return 'text-red-400';
  };

  const getStatusIcon = (status) => {
    if (status === 'healthy' || status === 'ok' || status === 'loaded' || status === true) {
      return <Wifi size={16} className="text-green-400" />;
    }
    return <WifiOff size={16} className="text-red-400" />;
  };

  // Extract data from pipeline status - handle nested structure
  const pipelineData = pipelineStatus?.pipeline || pipelineStatus || {};
  const pipelineInitialized = pipelineData.initialized || pipelineStatus?.status === 'ok';
  const lastUpdate = pipelineData.last_update || pipelineData.last_run;
  const dataQuality = pipelineData.data_info?.data_quality || pipelineData.data_quality;

  // Extract data from model status
  const modelsLoaded = modelStatus?.loaded || modelStatus?.models_loaded || false;
  const modelDetails = modelStatus?.models || {};
  const modelCount = Object.keys(modelDetails).length;

  return (
    <CollapsibleCard
      title="System Status"
      icon={<Activity size={20} className={error ? 'text-red-400' : 'text-green-400'} />}
      className="card-hover"
      defaultExpanded={false}
    >

      {error && (
        <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-sm">
          {error}
        </div>
      )}

      <div className="space-y-2">
        {/* Pipeline Status */}
        <div className="flex items-center justify-between p-2 bg-gray-700/50 rounded" role="status" aria-label="Data Pipeline Status">
          <div className="flex items-center gap-2">
            <Database size={16} className="text-gray-400" aria-hidden="true" />
            <div>
              <span className="text-xs text-gray-300">Data Pipeline</span>
              <p className="text-xs text-gray-500">
                {pipelineInitialized
                  ? (lastUpdate ? formatTimeDiff(lastUpdate, now) : 'Initialized')
                  : 'Not initialized'}
              </p>
            </div>
          </div>
          <span aria-label={`Pipeline status: ${pipelineInitialized ? 'ok' : 'error'}`}>
            {getStatusIcon(pipelineInitialized)}
          </span>
        </div>

        {/* Model Status */}
        <div className="flex items-center justify-between p-2 bg-gray-700/50 rounded" role="status" aria-label="AI Models Status">
          <div className="flex items-center gap-2">
            <Cpu size={16} className="text-gray-400" aria-hidden="true" />
            <div>
              <span className="text-xs text-gray-300">AI Models</span>
              <p className="text-xs text-gray-500">
                {modelsLoaded
                  ? `${modelCount} analyzers active`
                  : 'Not loaded'}
              </p>
            </div>
          </div>
          <span aria-label={`Model status: ${modelsLoaded ? 'loaded' : 'not loaded'}`}>
            {getStatusIcon(modelsLoaded)}
          </span>
        </div>

        {/* Model Details - Show individual model accuracy */}
        {modelsLoaded && modelCount > 0 && (
          <div className="mt-2 pt-2 border-t border-gray-700">
            <h3 className="text-xs text-gray-500 mb-2">Analysis Accuracy</h3>
            <div className="grid grid-cols-3 gap-2">
              {Object.entries(modelDetails).map(([tf, data]) => (
                <div key={tf} className="text-center p-2 bg-gray-700/30 rounded">
                  <span className="text-xs text-gray-500 block">{tf === 'D' ? '1D' : tf}</span>
                  <span className="text-sm font-medium text-green-400">
                    {data?.val_accuracy ? `${(data.val_accuracy * 100).toFixed(1)}%` : 'N/A'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Data Quality Indicator */}
        {dataQuality && (
          <div className="flex items-center gap-2 text-xs">
            <span className="text-gray-500">Data:</span>
            <span className={getStatusColor(dataQuality === 'fresh' ? 'ok' : 'warning')}>
              {dataQuality}
            </span>
          </div>
        )}

        {/* Last Update */}
        <div className="flex items-center gap-2 text-xs text-gray-500 mt-4 pt-2 border-t border-gray-700">
          <Clock size={12} />
          <span>Data refreshed: {lastUpdate ? formatTimeDiff(lastUpdate, now) : 'Never'}</span>
        </div>
      </div>
    </CollapsibleCard>
  );
}

export default AccountStatus;
