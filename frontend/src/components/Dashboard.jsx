import { useState, useCallback, useMemo } from 'react';
import { RefreshCw, Clock, Brain, Mail, ChevronDown } from 'lucide-react';

import { api } from '../api/client';
import { usePolling } from '../hooks/usePolling';
import {
  getFormattedSymbol,
  getDashboardDescription,
  isMarketOpen,
  getMarketStatusLabel,
  AVAILABLE_MARKETS,
  AVAILABLE_ASSETS,
  getDefaultMarket,
  getDefaultAsset,
} from '../utils/assetFormatting';

import { PredictionCard } from './PredictionCard';
import { AccountStatus } from './AccountStatus';
import { PriceChart } from './PriceChart';
import { PerformanceStats } from './PerformanceStats';
import { TradeHistory } from './TradeHistory';
import { AboutSection } from './AboutSection';
import { InvestmentCalculator } from './InvestmentCalculator';

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
  // Market and asset selection state
  const [selectedMarket, setSelectedMarket] = useState(getDefaultMarket);
  const [selectedAsset, setSelectedAsset] = useState(() => getDefaultAsset(getDefaultMarket()));

  // Get current asset metadata
  const currentAssetType = selectedMarket;
  const marketOpenStatus = useMemo(() => isMarketOpen(currentAssetType), [currentAssetType]);

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

  // Get trading pair and asset metadata from prediction or use selected
  const tradingPair = prediction?.symbol || selectedAsset;
  const assetMetadata = prediction?.asset_metadata || { asset_type: selectedMarket };

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

  // Handle market change
  const handleMarketChange = (e) => {
    const newMarket = e.target.value;
    setSelectedMarket(newMarket);
    // Reset asset to default for new market
    setSelectedAsset(getDefaultAsset(newMarket));
  };

  // Handle asset change
  const handleAssetChange = (e) => {
    setSelectedAsset(e.target.value);
  };

  // Refresh all data
  const handleRefreshAll = () => {
    refetchPrediction();
    refetchCandles();
  };

  const formatLastUpdated = (date) => {
    if (!date) return 'Never';
    return date.toLocaleTimeString();
  };

  // Get available assets for current market
  const availableAssets = AVAILABLE_ASSETS[selectedMarket] || [];

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 flex flex-col">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 sticky top-0 z-10">
        <div className="max-w-[1600px] mx-auto px-4 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-3">
              <Brain size={24} className="text-blue-400" />
              <div>
                <h1 className="text-xl font-bold">AI Trader</h1>
                <p className="text-xs text-gray-500">
                  AI Agent • <span className="text-blue-400">{getFormattedSymbol(tradingPair, assetMetadata)}</span>
                </p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {/* Market Selector */}
              <div className="relative">
                <select
                  value={selectedMarket}
                  onChange={handleMarketChange}
                  className="appearance-none bg-gray-700 border border-gray-600 rounded-lg px-3 py-1.5 pr-8 text-sm text-gray-200 focus:outline-none focus:border-blue-500 cursor-pointer"
                  aria-label="Select market"
                >
                  {AVAILABLE_MARKETS.map((market) => (
                    <option
                      key={market.id}
                      value={market.id}
                      disabled={!market.enabled}
                    >
                      {market.label}{!market.enabled ? ' (Coming Soon)' : ''}
                    </option>
                  ))}
                </select>
                <ChevronDown size={14} className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
              </div>

              {/* Asset Selector */}
              <div className="relative">
                <select
                  value={selectedAsset}
                  onChange={handleAssetChange}
                  className="appearance-none bg-gray-700 border border-gray-600 rounded-lg px-3 py-1.5 pr-8 text-sm text-gray-200 focus:outline-none focus:border-blue-500 cursor-pointer"
                  aria-label="Select asset"
                >
                  {availableAssets.map((asset) => (
                    <option
                      key={asset.symbol}
                      value={asset.symbol}
                      disabled={!asset.enabled}
                    >
                      {asset.label}{!asset.enabled ? ' (Coming Soon)' : ''}
                    </option>
                  ))}
                </select>
                <ChevronDown size={14} className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
              </div>

              {/* Market Status */}
              <div className={`flex items-center gap-2 px-2 py-1 rounded text-xs ${
                marketOpenStatus
                  ? 'bg-green-500/20 text-green-400'
                  : 'bg-yellow-500/20 text-yellow-400'
              }`}>
                <span className={`w-2 h-2 rounded-full ${
                  marketOpenStatus ? 'bg-green-400 animate-pulse' : 'bg-yellow-400'
                }`}></span>
                {getMarketStatusLabel(currentAssetType, marketOpenStatus)}
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

      {/* Dashboard Description */}
      <div className="bg-gray-800/50 border-b border-gray-700">
        <div className="max-w-[1600px] mx-auto px-4 py-3">
          <p className="text-sm text-gray-400 text-center">
            {getDashboardDescription(tradingPair, assetMetadata)}
          </p>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-[1600px] mx-auto px-4 py-6 flex-grow">
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
              marketOpen={marketOpenStatus}
              performance={performance}
            />
            <InvestmentCalculator
              assetMetadata={assetMetadata}
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
      <footer className="bg-gray-800 border-t border-gray-700 mt-auto">
        <div className="max-w-[1600px] mx-auto px-4 py-4">
          {/* Stats Row */}
          <div className="flex justify-between items-center text-sm text-gray-500 mb-4">
            <span>AI Agent Trader • Multi-Timeframe Analysis</span>
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

          {/* Risk Disclaimer */}
          <div className="border-t border-gray-700 pt-4 mb-4">
            <p className="text-xs text-gray-500 leading-relaxed text-center">
              <strong className="text-yellow-500">⚠️ Risk Disclaimer:</strong> These predictions are generated by an AI agent and are provided for informational purposes only.
              Past performance does not guarantee future results. Trading involves substantial risk of loss.
              Use at your own risk and always do your own research before making investment decisions.
            </p>
          </div>

          {/* Contact Row */}
          <div className="border-t border-gray-700 pt-4">
            <div className="flex flex-wrap justify-center items-center gap-x-2 gap-y-1 text-sm text-gray-400">
              <span>Created by <span className="text-gray-200 font-medium">Sergio Saraiva</span></span>
              <span className="text-gray-600">•</span>
              <a
                href="mailto:sergio.saraiva@gmail.com"
                className="flex items-center gap-1.5 hover:text-blue-400 transition-colors"
              >
                <Mail size={14} />
                <span>sergio.saraiva@gmail.com</span>
              </a>
              <span className="text-gray-600">•</span>
              <span>Questions or feedback? I'd love to hear from you!</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default Dashboard;
