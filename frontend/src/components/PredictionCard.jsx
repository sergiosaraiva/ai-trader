import { TrendingUp, TrendingDown, Minus, AlertCircle, Clock } from 'lucide-react';
import { formatPrice, getFormattedSymbol, getAssetTypeLabel } from '../utils/assetFormatting';

/**
 * PredictionCard - Displays the current trading recommendation
 */
export function PredictionCard({ prediction, loading, error }) {
  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="h-16 bg-gray-700 rounded mb-4"></div>
        <div className="h-4 bg-gray-700 rounded w-2/3"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 border border-red-500/30">
        <div className="flex items-center gap-2 text-red-400">
          <AlertCircle size={20} />
          <span>Error loading recommendation</span>
        </div>
        <p className="text-gray-500 text-sm mt-2">{error}</p>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <p className="text-gray-500">No recommendation available</p>
      </div>
    );
  }

  // Handle both old format (signal) and new format (direction)
  const {
    signal,
    direction,
    confidence,
    current_price,
    market_price,
    symbol,
    timestamp,
    timeframe_signals,
    component_directions,
    component_confidences,
    component_weights,
    should_trade,
    data_timestamp,
    next_prediction_at,
  } = prediction;

  // Normalize signal: API returns "long"/"short" in direction field
  const rawSignal = signal || direction;

  // Determine if this is a BUY signal (else SELL)
  const isBuySignal = rawSignal === 'BUY' || rawSignal === 'long' || rawSignal === 1;

  // HOLD when confidence is below 70% threshold (should_trade = false)
  const isHold = should_trade === false;

  // Final recommendation considering should_trade
  const recommendation = isHold ? 'HOLD' : (isBuySignal ? 'BUY' : 'SELL');

  // Generate explanation for the recommendation
  const getRecommendationReason = () => {
    const confidencePct = ((confidence || 0) * 100).toFixed(0);
    const underlyingSignal = isBuySignal ? 'bullish' : 'bearish';

    if (isHold) {
      return `Model shows ${underlyingSignal} bias but confidence (${confidencePct}%) is below 70% threshold. Wait for stronger signal.`;
    }

    if (isBuySignal) {
      return `Model predicts upward movement with ${confidencePct}% confidence. Consider long position.`;
    }

    return `Model predicts downward movement with ${confidencePct}% confidence. Consider short position.`;
  };

  // Get primary timeframe (highest weight) - defaults to 1H
  const getPrimaryTimeframe = () => {
    if (!component_weights) return '1H';
    const entries = Object.entries(component_weights);
    if (entries.length === 0) return '1H';
    const [timeframe] = entries.reduce((max, curr) => curr[1] > max[1] ? curr : max);
    return timeframe;
  };
  const primaryTimeframe = getPrimaryTimeframe();

  const getSignalColor = (sig, forceHold = false) => {
    if (forceHold || sig === 'HOLD') return 'text-yellow-400';
    if (sig === 'BUY' || sig === 'long' || sig === 1) return 'text-green-400';
    if (sig === 'SELL' || sig === 'short' || sig === -1 || sig === 0) return 'text-red-400';
    return 'text-gray-400';
  };

  const getSignalIcon = (sig, forceHold = false) => {
    if (forceHold || sig === 'HOLD') return <Minus size={32} />;
    if (sig === 'BUY' || sig === 'long' || sig === 1) return <TrendingUp size={32} />;
    if (sig === 'SELL' || sig === 'short' || sig === -1 || sig === 0) return <TrendingDown size={32} />;
    return <Minus size={32} />;
  };

  const getSignalText = (sig) => {
    if (sig === 'HOLD') return 'HOLD';
    if (sig === 'BUY' || sig === 'long' || sig === 1) return 'BUY';
    if (sig === 'SELL' || sig === 'short' || sig === -1 || sig === 0) return 'SELL';
    return 'HOLD';
  };

  // Use market_price if current_price not available
  const displayPrice = current_price || market_price;

  const getConfidenceColor = (conf) => {
    const value = conf * 100;
    if (value >= 70) return 'bg-green-500';
    if (value >= 60) return 'bg-yellow-500';
    return 'bg-gray-500';
  };

  const formatTime = (ts) => {
    if (!ts) return 'N/A';
    try {
      const date = new Date(ts);
      if (isNaN(date.getTime())) return 'Invalid date';
      return date.toLocaleString();
    } catch {
      return 'Invalid date';
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 card-hover">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className="text-lg font-semibold text-gray-300">Current Recommendation <span className="text-blue-400 font-normal">({primaryTimeframe})</span></h2>
          <p className="text-sm text-gray-500">{getAssetTypeLabel(prediction?.asset_metadata)} • {getFormattedSymbol(symbol, prediction?.asset_metadata)}</p>
        </div>
        <div className="flex flex-col items-end gap-1 text-gray-500 text-sm">
          <div className="flex items-center gap-1">
            <Clock size={14} />
            <span>{formatTime(timestamp)}</span>
          </div>
          {data_timestamp && (
            <div className="text-xs text-gray-600">
              Data from: {formatTime(data_timestamp)}
            </div>
          )}
          {next_prediction_at && (
            <div className="text-xs text-gray-600">
              Next update: {formatTime(next_prediction_at)}
            </div>
          )}
        </div>
      </div>

      {/* Main Signal */}
      <div className="flex items-center justify-center gap-4 py-4" role="status" aria-live="polite">
        <div className={`${getSignalColor(recommendation, isHold)}`} aria-hidden="true">
          {getSignalIcon(recommendation, isHold)}
        </div>
        <div className="text-center">
          <span className={`text-4xl font-bold ${getSignalColor(recommendation, isHold)}`} aria-label={`Recommendation: ${recommendation}`}>
            {recommendation}
          </span>
          <p className="text-gray-500 text-sm mt-1">
            @ {formatPrice(displayPrice, prediction?.asset_metadata)}
          </p>
        </div>
      </div>

      {/* Recommendation Explanation */}
      <div className="bg-gray-700/30 rounded-lg p-3 mb-2">
        <p className="text-sm text-gray-400 leading-relaxed">
          <span className={`font-medium ${getSignalColor(recommendation, isHold)}`}>{recommendation}:</span>{' '}
          {getRecommendationReason()}
        </p>
      </div>

      {/* Confidence Bar */}
      <div className="mt-4" aria-label="Recommendation confidence">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-gray-400" id="confidence-label">Confidence</span>
          <span className="text-sm font-medium text-gray-300">
            {((confidence || 0) * 100).toFixed(1)}%
          </span>
        </div>
        <div
          className="h-2 bg-gray-700 rounded-full overflow-hidden"
          role="progressbar"
          aria-labelledby="confidence-label"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round((confidence || 0) * 100)}
        >
          <div
            className={`h-full ${getConfidenceColor(confidence)} transition-all duration-500`}
            style={{ width: `${(confidence || 0) * 100}%` }}
          />
        </div>
      </div>

      {/* Timeframe Breakdown */}
      {(timeframe_signals || component_directions) && (
        <div className="mt-6 pt-4 border-t border-gray-700">
          <h3 className="text-sm text-gray-400 mb-3">Timeframe Breakdown</h3>
          <div className="grid grid-cols-3 gap-3">
            {/* Handle new API format with component_directions */}
            {component_directions && Object.entries(component_directions).map(([tf, dir]) => (
              <div key={tf} className="bg-gray-700/50 rounded p-3 text-center">
                <span className="text-xs text-gray-500 block mb-1">{tf === 'D' ? '1D' : tf}</span>
                <span className={`text-sm font-medium ${getSignalColor(dir)}`}>
                  {getSignalText(dir)}
                </span>
                {component_confidences?.[tf] && (
                  <span className="text-xs text-gray-500 block mt-1">
                    {(component_confidences[tf] * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            ))}
            {/* Handle old API format with timeframe_signals */}
            {!component_directions && timeframe_signals && Object.entries(timeframe_signals).map(([tf, data]) => (
              <div key={tf} className="bg-gray-700/50 rounded p-3 text-center">
                <span className="text-xs text-gray-500 block mb-1">{tf}</span>
                <span className={`text-sm font-medium ${getSignalColor(data.signal || data)}`}>
                  {getSignalText(data.signal || data)}
                </span>
                {data.confidence && (
                  <span className="text-xs text-gray-500 block mt-1">
                    {(data.confidence * 100).toFixed(0)}%
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default PredictionCard;
