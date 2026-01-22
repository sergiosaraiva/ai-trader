import { useState, useCallback, useMemo } from 'react';
import { RefreshCw, Clock, Mail, ChevronDown } from 'lucide-react';

import { api } from '../api/client';
import { usePolling } from '../hooks/usePolling';
import {
  getFormattedSymbol,
  getAssetTypeLabel,
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
import { PerformanceChart } from './PerformanceChart';
import { TradeHistory } from './TradeHistory';
import { AboutSection } from './AboutSection';
import { InvestmentCalculator } from './InvestmentCalculator';
import { ExplanationCard } from './ExplanationCard';
import { ModelHighlights } from './ModelHighlights';

// Polling intervals (in milliseconds)
const INTERVALS = {
  prediction: 30000,    // 30 seconds
  candles: 60000,       // 1 minute
  pipeline: 60000,      // 1 minute
  signals: 30000,       // 30 seconds
  performance: 300000,  // 5 minutes (rarely changes)
  tradeHistory: 300000, // 5 minutes (for 30-day chart)
  explanation: 60000,   // 1 minute (cached on backend anyway)
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

  // Trading performance metrics for PerformanceStats
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

  // Model performance highlights
  const {
    data: modelPerformance,
    loading: modelPerformanceLoading,
    error: modelPerformanceError,
  } = usePolling(
    useCallback(async () => {
      try {
        return await api.getModelPerformance();
      } catch {
        // Return null if endpoint doesn't exist, component will use defaults
        return null;
      }
    }, []),
    INTERVALS.performance
  );

  // Trade history for 30-day performance chart
  const {
    data: tradeHistoryData,
    loading: tradeHistoryLoading,
    error: tradeHistoryError,
  } = usePolling(
    useCallback(async () => {
      try {
        return await api.getTradingHistory(100);
      } catch {
        return null;
      }
    }, []),
    INTERVALS.tradeHistory
  );

  // AI Explanation (LLM-generated)
  const {
    data: explanation,
    loading: explanationLoading,
    error: explanationError,
    refetch: refetchExplanation,
  } = usePolling(
    useCallback(async () => {
      try {
        return await api.getExplanation();
      } catch {
        return null;
      }
    }, []),
    INTERVALS.explanation
  );

  // Extract trades array from response
  const trades = tradeHistoryData?.trades || [];

  // Extract candles array from response
  const candles = candlesData?.candles || candlesData || [];

  // Extract signals array from response (API returns 'predictions' not 'signals')
  const signals = signalsData?.predictions || signalsData?.signals || (Array.isArray(signalsData) ? signalsData : []);

  // Extract model weights and stacking info for AboutSection
  const modelWeights = modelStatus?.weights || null;
  const useStacking = modelStatus?.use_stacking || false;

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
        <div className="max-w-[1600px] mx-auto px-4 py-3">
          {/* Mobile: Stack vertically, Desktop: Single row */}
          <div className="flex flex-col gap-3 md:flex-row md:justify-between md:items-center">
            {/* Logo and Title */}
            <div className="flex items-center gap-3">
              <img src="/favicon.svg" alt="AI Trading Agent" className="w-8 h-8" />
              <div>
                <h1 className="text-lg md:text-xl font-bold">AI Trading Agent</h1>
                <p className="text-xs text-gray-500">
                  {getAssetTypeLabel(assetMetadata)} • <span className="text-blue-400">{getFormattedSymbol(tradingPair, assetMetadata)}</span>
                </p>
              </div>
            </div>

            {/* Controls - wrap on mobile */}
            <div className="flex flex-wrap items-center gap-2 md:gap-4">
              {/* Selectors Row */}
              <div className="flex items-center gap-2">
                {/* Market Selector */}
                <div className="relative">
                  <select
                    value={selectedMarket}
                    onChange={handleMarketChange}
                    className="appearance-none bg-gray-700 border border-gray-600 rounded-lg px-2 md:px-3 py-1.5 pr-7 md:pr-8 text-xs md:text-sm text-gray-200 focus:outline-none focus:border-blue-500 cursor-pointer"
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
                    className="appearance-none bg-gray-700 border border-gray-600 rounded-lg px-2 md:px-3 py-1.5 pr-7 md:pr-8 text-xs md:text-sm text-gray-200 focus:outline-none focus:border-blue-500 cursor-pointer"
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
              </div>

              {/* Status Row */}
              <div className="flex items-center gap-2">
                {/* Market Status */}
                <div className={`flex items-center gap-1.5 px-2 py-1 rounded text-xs ${
                  marketOpenStatus
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-yellow-500/20 text-yellow-400'
                }`}>
                  <span className={`w-2 h-2 rounded-full ${
                    marketOpenStatus ? 'bg-green-400 animate-pulse' : 'bg-yellow-400'
                  }`}></span>
                  <span className="hidden sm:inline">{getMarketStatusLabel(currentAssetType, marketOpenStatus)}</span>
                  <span className="sm:hidden">{marketOpenStatus ? 'Open' : 'Closed'}</span>
                </div>

                {/* Updated time - hide on very small screens */}
                <div className="hidden sm:flex items-center gap-1.5 text-xs md:text-sm text-gray-500">
                  <Clock size={14} />
                  <span className="hidden md:inline">Updated: </span>
                  <span>{formatLastUpdated(predictionUpdated)}</span>
                </div>

                {/* Refresh button */}
                <button
                  onClick={handleRefreshAll}
                  className="flex items-center gap-1.5 px-2 md:px-3 py-1.5 md:py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                  aria-label="Refresh data"
                >
                  <RefreshCw size={14} className="md:w-4 md:h-4" />
                  <span className="hidden md:inline text-sm">Refresh</span>
                </button>
              </div>
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
        {/* AI Explanation - Full Width */}
        <div className="mb-6">
          <ExplanationCard
            explanation={explanation}
            loading={explanationLoading}
            error={explanationError}
            onRefresh={() => refetchExplanation()}
          />
        </div>

        {/* Model Highlights - Full Width */}
        <div className="mb-6">
          <ModelHighlights
            performance={modelPerformance}
            loading={modelPerformanceLoading}
            error={modelPerformanceError}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Prediction, About, Status & Signal History */}
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
              useStacking={useStacking}
            />
            <AccountStatus
              pipelineStatus={pipelineStatus}
              modelStatus={modelStatus}
              loading={pipelineLoading || modelLoading}
              error={pipelineError}
            />
            <TradeHistory
              signals={signals}
              loading={signalsLoading}
              error={signalsError}
              assetMetadata={assetMetadata}
            />
          </div>

          {/* Right Column - Chart, Performance & Calculator */}
          <div className="lg:col-span-2 space-y-6">
            <PriceChart
              candles={candles}
              prediction={prediction}
              loading={candlesLoading}
              error={candlesError}
              onRefresh={refetchCandles}
            />
            <PerformanceChart
              trades={trades}
              loading={tradeHistoryLoading}
              error={tradeHistoryError}
              assetMetadata={assetMetadata}
            />
            <PerformanceStats
              performance={performance}
              loading={performanceLoading}
              error={performanceError}
              assetMetadata={assetMetadata}
            />
            <InvestmentCalculator
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
            <span>AI Trading Agent • Multi-Timeframe Analysis</span>
            <div className="flex items-center gap-4">
              <span className="text-green-400 font-medium">
                {performance?.win_rate_high_conf ? `${(performance.win_rate_high_conf * 100).toFixed(0)}%` : '61%'} Win Rate <span className="text-xs text-gray-500">(high-confidence)</span>
              </span>
              <span className="text-blue-400">
                {performance?.profit_factor?.toFixed(2) || '2.10'} PF
              </span>
              <span className="text-yellow-400">WFO Validated</span>
            </div>
          </div>

          {/* Risk Disclaimer */}
          <div className="border-t border-gray-700 pt-4 mb-4">
            <p className="text-xs text-gray-500 leading-relaxed text-center">
              <strong className="text-yellow-500">⚠️ Risk Disclaimer:</strong> These recommendations are generated by an AI agent and are provided for informational purposes only.
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
