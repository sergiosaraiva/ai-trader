import { useCallback } from 'react';
import { RefreshCw, Activity, Clock, Brain, Mail, Github } from 'lucide-react';

import { api } from '../api/client';
import { usePolling } from '../hooks/usePolling';
import { getFormattedSymbol } from '../utils/assetFormatting';

import { PredictionCard } from './PredictionCard';
import { AccountStatus } from './AccountStatus';
import { PriceChart } from './PriceChart';
import { PerformanceStats } from './PerformanceStats';
import { TradeHistory } from './TradeHistory';
import { AboutSection } from './AboutSection';

// Polling intervals (in milliseconds)
const INTERVALS = {
  prediction: 30000,    // 30 seconds
  candles: 60000,       // 1 minute
  pipeline: 60000,      // 1 minute
  signals: 30000,       // 30 seconds
  performance: 300000,  // 5 minutes (rarely changes)
};

/**
 * Check if forex markets are currently open
 * Forex markets are open from Sunday 5pm ET to Friday 5pm ET
 */
const isForexMarketOpen = () => {
  const now = new Date();
  const day = now.getUTCDay(); // 0 = Sunday, 6 = Saturday
  const hour = now.getUTCHours();

  // Closed on Saturday
  if (day === 6) return false;

  // Sunday: opens at 22:00 UTC (5pm ET)
  if (day === 0 && hour < 22) return false;

  // Friday: closes at 22:00 UTC (5pm ET)
  if (day === 5 && hour >= 22) return false;

  return true;
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

  // Get trading pair and asset metadata from prediction or use default
  const tradingPair = prediction?.symbol || 'EURUSD';
  const assetMetadata = prediction?.asset_metadata;

  // Candles/Price data - use dynamic trading pair
  const {
    data: candlesData,
    loading: candlesLoading,
    error: candlesError,
    refetch: refetchCandles,
  } = usePolling(
    useCallback(() => api.getCandles(tradingPair, '1H', 48), [tradingPair]),
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

  // VIX sentiment data
  const {
    data: vixData,
  } = usePolling(
    useCallback(() => api.getVix(), []),
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

  // Extract signals array from response (API returns 'predictions' not 'signals')
  const signals = signalsData?.predictions || signalsData?.signals || (Array.isArray(signalsData) ? signalsData : []);

  // Extract model weights for AboutSection
  const modelWeights = modelStatus?.weights || null;

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
        <div className="max-w-[1600px] mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-3">
              <Brain size={24} className="text-blue-400" />
              <div>
                <h1 className="text-xl font-bold">AI Trader</h1>
                <p className="text-xs text-gray-500">
                  MTF Ensemble • <span className="text-blue-400">{getFormattedSymbol(tradingPair, assetMetadata)}</span>
                </p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {/* Market Status */}
              <div className={`flex items-center gap-2 px-2 py-1 rounded text-xs ${
                isForexMarketOpen()
                  ? 'bg-green-500/20 text-green-400'
                  : 'bg-yellow-500/20 text-yellow-400'
              }`}>
                <span className={`w-2 h-2 rounded-full ${
                  isForexMarketOpen() ? 'bg-green-400 animate-pulse' : 'bg-yellow-400'
                }`}></span>
                {isForexMarketOpen() ? 'Market Open' : 'Market Closed'}
              </div>
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
      <main className="max-w-[1600px] mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Prediction, About (with VIX) & Status */}
          <div className="space-y-6">
            <PredictionCard
              prediction={prediction}
              loading={predictionLoading}
              error={predictionError}
            />
            <AboutSection
              tradingPair={tradingPair}
              modelWeights={modelWeights}
              vixValue={vixData?.value}
              assetMetadata={assetMetadata}
              marketOpen={isForexMarketOpen()}
              performance={performance}
            />
            <AccountStatus
              pipelineStatus={pipelineStatus}
              modelStatus={modelStatus}
              loading={pipelineLoading || modelLoading}
              error={pipelineError}
            />
          </div>

          {/* Middle Column - Chart, Performance & Trade History */}
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
              assetMetadata={assetMetadata}
            />
            <TradeHistory
              signals={signals}
              loading={signalsLoading}
              error={signalsError}
              assetMetadata={assetMetadata}
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700 mt-8">
        <div className="max-w-[1600px] mx-auto px-4 py-4">
          {/* Stats Row */}
          <div className="flex justify-between items-center text-sm text-gray-500 mb-4">
            <span>AI Trader • Multi-Timeframe AI Ensemble</span>
            <div className="flex items-center gap-4">
              <span className="text-green-400">
                {performance?.win_rate ? `${(performance.win_rate * 100).toFixed(0)}%` : '62%'} Win Rate
              </span>
              <span className="text-blue-400">
                {performance?.profit_factor?.toFixed(2) || '2.69'} Profit Factor
              </span>
              <span className="text-yellow-400">WFO Validated</span>
            </div>
          </div>

          {/* Contact Row */}
          <div className="border-t border-gray-700 pt-4">
            <div className="flex flex-col sm:flex-row justify-between items-center gap-3 text-sm">
              <div className="text-gray-400">
                Created by <span className="text-gray-200 font-medium">Sergio Saraiva</span>
              </div>
              <div className="flex items-center gap-4">
                <a
                  href="mailto:sergio.saraiva@gmail.com"
                  className="flex items-center gap-2 text-gray-400 hover:text-blue-400 transition-colors"
                >
                  <Mail size={16} />
                  <span>sergio.saraiva@gmail.com</span>
                </a>
                <a
                  href="https://github.com/sergiosaraiva/ai-trader"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-gray-400 hover:text-blue-400 transition-colors"
                >
                  <Github size={16} />
                  <span>GitHub</span>
                </a>
              </div>
            </div>
            <p className="text-center text-xs text-gray-500 mt-3">
              Feel free to contact me with any questions or clarifications about this project.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default Dashboard;
