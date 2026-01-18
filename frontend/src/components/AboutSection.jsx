import { Brain, TrendingUp, Shield, Database, Zap, BarChart3, Activity, AlertTriangle } from 'lucide-react';
import { getAssetTypeLabel, getFormattedSymbol, inferAssetMetadata, getProfitUnitLabel } from '../utils/assetFormatting';

/**
 * Format timeframe for display (e.g., "D" -> "1D", "1H" -> "1H")
 */
const formatTimeframe = (tf) => {
  if (tf === 'D') return '1D';
  return tf;
};

/**
 * Get market closed message based on asset type and current day
 */
const getMarketClosedMessage = (assetType) => {
  const now = new Date();
  const day = now.getUTCDay(); // 0 = Sunday, 6 = Saturday

  if (assetType === 'crypto') {
    return 'Cryptocurrency markets are 24/7. Data may be delayed.';
  }

  if (assetType === 'stock') {
    if (day === 0) {
      return 'Stock markets reopen Monday. Data shown is from last trading session.';
    } else if (day === 6) {
      return 'Stock markets reopen Monday. Data shown is from last trading session.';
    }
    return 'Stock markets are closed. Data shown is from last trading session.';
  }

  // Forex (default)
  if (day === 0) {
    return 'Forex markets reopen tonight at 10pm UTC (5pm ET). Data shown is from Friday.';
  } else if (day === 6) {
    return 'Forex markets reopen Sunday 10pm UTC (5pm ET). Data shown is from Friday.';
  }
  return 'Forex markets are closed. Data shown is from last trading session.';
};

/**
 * AboutSection - Explains the AI trading system
 *
 * Props:
 * - tradingPair: The asset being traded (e.g., "EURUSD")
 * - modelWeights: Object with timeframe weights (e.g., { "1H": 0.6, "4H": 0.3, "D": 0.1 })
 * - vixValue: Current VIX value for sentiment display
 * - assetMetadata: Asset metadata from API (optional)
 * - marketOpen: Whether the market is currently open
 * - performance: Performance stats from API (optional)
 */
export function AboutSection({ tradingPair = "EURUSD", modelWeights, vixValue, assetMetadata, marketOpen = true, performance }) {
  // Use provided metadata or infer from trading pair
  const metadata = assetMetadata || inferAssetMetadata(tradingPair);
  const profitUnit = getProfitUnitLabel(metadata);

  // Get VIX sentiment level and color
  const getVixLevel = (vix) => {
    if (vix === undefined || vix === null) return { label: 'N/A', color: 'text-gray-400' };
    if (vix < 15) return { label: 'Low Vol', color: 'text-green-400' };
    if (vix < 20) return { label: 'Normal', color: 'text-blue-400' };
    if (vix < 30) return { label: 'Elevated', color: 'text-yellow-400' };
    return { label: 'High Vol', color: 'text-red-400' };
  };

  const vixLevel = getVixLevel(vixValue);

  const features = [
    {
      icon: Brain,
      title: 'Multi-Timeframe AI',
      description: 'Ensemble of 3 XGBoost models analyzing 1H, 4H, and Daily patterns',
    },
    {
      icon: TrendingUp,
      title: 'Walk-Forward Validated',
      description: 'Backtested on 7 rolling time periods (2022-2025) with consistent profitability',
    },
    {
      icon: Shield,
      title: 'Risk-Optimized',
      description: '70% confidence threshold filters low-quality signals',
    },
  ];

  const dataSources = [
    { name: 'Price Data', source: 'MetaTrader 5 / yfinance', detail: '5-min OHLCV' },
    { name: 'VIX Index', source: 'FRED API', detail: 'Market volatility' },
    { name: 'EPU Index', source: 'FRED API', detail: 'Economic uncertainty' },
  ];

  return (
    <div className="bg-gray-800 rounded-lg p-6 card-hover" role="region" aria-label="About This System">
      {/* Header with Trading Pair */}
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className="text-lg font-semibold text-gray-300">AI Trading System</h2>
          <p className="text-xs text-gray-500 mt-1">
            {getAssetTypeLabel(metadata)} • <span className="text-blue-400 font-medium">{getFormattedSymbol(tradingPair, metadata)}</span>
          </p>
        </div>
        <Brain size={24} className="text-blue-400" aria-hidden="true" />
      </div>

      {/* Market Closed Banner */}
      {!marketOpen && (
        <div className="mb-4 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg flex items-center gap-3">
          <AlertTriangle size={18} className="text-yellow-400 flex-shrink-0" />
          <div>
            <span className="text-yellow-400 font-medium text-sm">Markets Closed</span>
            <p className="text-yellow-400/70 text-xs">{getMarketClosedMessage(metadata?.asset_type)}</p>
          </div>
        </div>
      )}

      {/* VIX Sentiment Indicator - Highlighted */}
      <div className={`mb-4 p-4 rounded-lg border ${
        !marketOpen
          ? 'bg-gray-700/30 border-gray-600'
          : vixValue >= 30
            ? 'bg-red-500/10 border-red-500/30'
            : vixValue >= 20
              ? 'bg-yellow-500/10 border-yellow-500/30'
              : 'bg-blue-500/10 border-blue-500/30'
      }`} role="region" aria-label="Market Sentiment Indicator">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Activity size={20} className={vixLevel.color} />
            <div>
              <span className="text-xs text-gray-400 block">Market Sentiment (VIX)</span>
              <div className="flex items-center gap-2 mt-1">
                <span className="text-2xl font-bold text-gray-100">
                  {vixValue !== undefined && vixValue !== null ? vixValue.toFixed(2) : 'N/A'}
                </span>
                <span className={`text-sm font-medium px-2 py-0.5 rounded ${vixLevel.color} bg-gray-800/50`}>
                  {vixLevel.label}
                </span>
              </div>
            </div>
          </div>
          <div className="text-right">
            <span className="text-xs text-gray-500 block">Input to</span>
            <span className="text-xs text-gray-400">Daily Model</span>
          </div>
        </div>
      </div>

      {/* Brief Description */}
      <p className="text-sm text-gray-400 mb-4 leading-relaxed">
        AI ensemble combining technical analysis with sentiment indicators
        to generate high-confidence trading predictions.
      </p>

      {/* Key Features */}
      <div className="space-y-3 mb-4">
        {features.map((feature, idx) => (
          <div key={idx} className="flex items-start gap-3">
            <feature.icon size={16} className="text-blue-400 mt-0.5 flex-shrink-0" aria-hidden="true" />
            <div>
              <span className="text-sm text-gray-300 font-medium">{feature.title}</span>
              <p className="text-xs text-gray-500">{feature.description}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Model Weights if available */}
      {modelWeights && Object.keys(modelWeights).length > 0 && (
        <div className="mb-4 pt-3 border-t border-gray-700">
          <div className="flex items-center gap-2 mb-2">
            <BarChart3 size={14} className="text-gray-500" />
            <span className="text-xs text-gray-500">Model Weights</span>
          </div>
          <div className="flex gap-2">
            {Object.entries(modelWeights).map(([tf, weight]) => (
              <div key={tf} className="flex-1 text-center p-2 bg-gray-700/30 rounded">
                <span className="text-xs text-gray-500 block">{formatTimeframe(tf)}</span>
                <span className="text-sm font-medium text-blue-400">{(weight * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Data Sources */}
      <div className="pt-3 border-t border-gray-700">
        <div className="flex items-center gap-2 mb-2">
          <Database size={14} className="text-gray-500" />
          <span className="text-xs text-gray-500">Data Sources</span>
        </div>
        <div className="grid grid-cols-3 gap-2">
          {dataSources.map((source, idx) => (
            <div key={idx} className="text-center p-2 bg-gray-700/30 rounded">
              <span className="text-xs text-gray-400 block">{source.name}</span>
              <span className="text-xs text-gray-500">{source.source}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Performance Highlight */}
      <div className="mt-4 pt-3 border-t border-gray-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Zap size={14} className="text-yellow-400" />
          <span className="text-xs text-gray-400">Backtested Performance</span>
        </div>
        <div className="flex gap-3 text-xs">
          <span className="text-green-400">
            {performance?.win_rate ? `${(performance.win_rate * 100).toFixed(0)}%` : '62%'} Win
          </span>
          <span className="text-blue-400">
            {performance?.profit_factor?.toFixed(2) || '2.69'} PF
          </span>
          <span className="text-yellow-400">
            {performance?.sharpe_ratio?.toFixed(2) || '7.67'} Sharpe
          </span>
        </div>
      </div>

      {/* Risk Disclaimer */}
      <div className="mt-4 pt-3 border-t border-gray-700">
        <p className="text-xs text-gray-500 leading-relaxed">
          <strong className="text-yellow-500">⚠️ Risk Disclaimer:</strong> These predictions are generated by AI models and are provided for informational purposes only.
          Past performance does not guarantee future results. Trading involves substantial risk of loss.
          Use at your own risk and always do your own research before making investment decisions.
        </p>
      </div>
    </div>
  );
}

export default AboutSection;
