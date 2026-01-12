import { useCallback } from 'react';
import { RefreshCw, Activity, Clock } from 'lucide-react';

import { api } from '../api/client';
import { usePolling } from '../hooks/usePolling';

import { PredictionCard } from './PredictionCard';
import { AccountStatus } from './AccountStatus';
import { PriceChart } from './PriceChart';
import { PerformanceStats } from './PerformanceStats';
import { TradeHistory } from './TradeHistory';

// Polling intervals (in milliseconds)
const INTERVALS = {
  prediction: 30000,    // 30 seconds
  candles: 60000,       // 1 minute
  pipeline: 60000,      // 1 minute
  signals: 30000,       // 30 seconds
  performance: 300000,  // 5 minutes (rarely changes)
};

/**
 * Dashboard - Main trading dashboard layout
 */
export function Dashboard() {
  // Prediction data
  const {
    data: prediction,
    loading: predictionLoading,
    error: predictionError,
    refetch: refetchPrediction,
    lastUpdated: predictionUpdated,
  } = usePolling(
    useCallback(() => api.getPrediction(), []),
    INTERVALS.prediction
  );

  // Candles/Price data
  const {
    data: candlesData,
    loading: candlesLoading,
    error: candlesError,
    refetch: refetchCandles,
  } = usePolling(
    useCallback(() => api.getCandles('EURUSD', '1H', 48), []),
    INTERVALS.candles
  );

  // Pipeline status
  const {
    data: pipelineStatus,
    loading: pipelineLoading,
    error: pipelineError,
  } = usePolling(
    useCallback(() => api.getPipelineStatus(), []),
    INTERVALS.pipeline
  );

  // Model status
  const {
    data: modelStatus,
    loading: modelLoading,
  } = usePolling(
    useCallback(() => api.getModelStatus(), []),
    INTERVALS.pipeline
  );

  // Signal history
  const {
    data: signalsData,
    loading: signalsLoading,
    error: signalsError,
  } = usePolling(
    useCallback(() => api.getSignals(20), []),
    INTERVALS.signals
  );

  // Performance metrics (using mock data for now as endpoint may not exist)
  const {
    data: performance,
    loading: performanceLoading,
    error: performanceError,
  } = usePolling(
    useCallback(async () => {
      try {
        return await api.getPerformance();
      } catch {
        // Return null if endpoint doesn't exist, component will use defaults
        return null;
      }
    }, []),
    INTERVALS.performance
  );

  // Extract candles array from response
  const candles = candlesData?.candles || candlesData || [];

  // Extract signals array from response
  const signals = signalsData?.signals || signalsData || [];

  // Refresh all data
  const handleRefreshAll = () => {
    refetchPrediction();
    refetchCandles();
  };

  const formatLastUpdated = (date) => {
    if (!date) return 'Never';
    return date.toLocaleTimeString();
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-3">
              <Activity size={24} className="text-blue-400" />
              <div>
                <h1 className="text-xl font-bold">AI Trader</h1>
                <p className="text-xs text-gray-500">MTF Ensemble Trading System</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <Clock size={14} />
                <span>Updated: {formatLastUpdated(predictionUpdated)}</span>
              </div>
              <button
                onClick={handleRefreshAll}
                className="flex items-center gap-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
              >
                <RefreshCw size={16} />
                <span className="text-sm">Refresh</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Prediction & Status */}
          <div className="space-y-6">
            <PredictionCard
              prediction={prediction}
              loading={predictionLoading}
              error={predictionError}
            />
            <AccountStatus
              pipelineStatus={pipelineStatus}
              modelStatus={modelStatus}
              loading={pipelineLoading || modelLoading}
              error={pipelineError}
            />
          </div>

          {/* Middle Column - Chart & Performance */}
          <div className="lg:col-span-2 space-y-6">
            <PriceChart
              candles={candles}
              prediction={prediction}
              loading={candlesLoading}
              error={candlesError}
              onRefresh={refetchCandles}
            />
            <PerformanceStats
              performance={performance}
              loading={performanceLoading}
              error={performanceError}
            />
          </div>
        </div>

        {/* Bottom Row - Trade History */}
        <div className="mt-6">
          <TradeHistory
            signals={signals}
            loading={signalsLoading}
            error={signalsError}
          />
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700 mt-8">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex justify-between items-center text-sm text-gray-500">
            <span>AI Assets Trader - MTF Ensemble System</span>
            <div className="flex items-center gap-4">
              <span>Win Rate: 62.1%</span>
              <span>Profit Factor: 2.69</span>
              <span>WFO Validated: 100%</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default Dashboard;
