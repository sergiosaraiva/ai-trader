import { useMemo } from 'react';
import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { TrendingUp, TrendingDown, RefreshCw } from 'lucide-react';
import { formatPrice, getFormattedSymbol, getAssetTypeLabel } from '../utils/assetFormatting';

/**
 * Custom tooltip component for the price chart
 */
function CustomTooltip({ active, payload, assetMetadata }) {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0]?.payload;
  if (!data) return null;

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-xl">
      <p className="text-gray-400 text-xs mb-2">{data.time}</p>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        <span className="text-gray-500">Open:</span>
        <span className="text-gray-300">{formatPrice(data.open, assetMetadata)}</span>
        <span className="text-gray-500">High:</span>
        <span className="text-green-400">{formatPrice(data.high, assetMetadata)}</span>
        <span className="text-gray-500">Low:</span>
        <span className="text-red-400">{formatPrice(data.low, assetMetadata)}</span>
        <span className="text-gray-500">Close:</span>
        <span className={data.isBullish ? 'text-green-400' : 'text-red-400'}>
          {formatPrice(data.close, assetMetadata)}
        </span>
      </div>
    </div>
  );
}

/**
 * PriceChart - Displays candlestick-style price chart
 */
export function PriceChart({ candles, prediction, loading, error, onRefresh }) {
  const chartData = useMemo(() => {
    if (!candles || !Array.isArray(candles) || candles.length === 0) {
      return [];
    }

    return candles.map((candle, idx) => {
      const open = candle.open || candle.Open;
      const high = candle.high || candle.High;
      const low = candle.low || candle.Low;
      const close = candle.close || candle.Close;
      const timestamp = candle.timestamp || candle.time || candle.datetime;

      const isBullish = close >= open;

      return {
        idx,
        time: timestamp ? new Date(timestamp).toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit'
        }) : `#${idx}`,
        open,
        high,
        low,
        close,
        // For candlestick-like display
        wickTop: high - Math.max(open, close),
        wickBottom: Math.min(open, close) - low,
        body: Math.abs(close - open),
        bodyBase: Math.min(open, close),
        isBullish,
        color: isBullish ? '#22c55e' : '#ef4444',
      };
    });
  }, [candles]);

  const { minPrice, maxPrice, currentPrice, priceChange, tickPrecision } = useMemo(() => {
    if (chartData.length === 0) {
      return { minPrice: 0, maxPrice: 1, currentPrice: null, priceChange: null, tickPrecision: 4 };
    }

    const prices = chartData.flatMap(d => [d.high, d.low]);
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const range = max - min;

    // Add padding (at least 10% of range, but ensure minimum visible range)
    const padding = Math.max(range * 0.1, min * 0.0001);

    const current = chartData[chartData.length - 1]?.close;
    const first = chartData[0]?.open;
    const change = current && first ? ((current - first) / first) * 100 : null;

    // Determine tick precision based on price magnitude and range
    // For forex (prices ~1.0), use 5 decimals; for crypto/stocks, use fewer
    let precision = 4;
    if (min > 0 && min < 10) {
      precision = 5; // Forex-like prices
    } else if (min >= 10 && min < 1000) {
      precision = 2; // Stock-like prices
    } else if (min >= 1000) {
      precision = 0; // Large prices (crypto in USD)
    }

    return {
      minPrice: min - padding,
      maxPrice: max + padding,
      currentPrice: current,
      priceChange: change,
      tickPrecision: precision,
    };
  }, [chartData]);

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 h-[400px] animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/4 mb-4"></div>
        <div className="h-full bg-gray-700/50 rounded"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 h-[400px] flex flex-col items-center justify-center">
        <p className="text-red-400 mb-4">{error}</p>
        {onRefresh && (
          <button
            onClick={onRefresh}
            className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded transition-colors"
          >
            <RefreshCw size={16} />
            Retry
          </button>
        )}
      </div>
    );
  }

  if (chartData.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 h-[400px] flex items-center justify-center">
        <p className="text-gray-500">No price data available</p>
      </div>
    );
  }

  // Get asset metadata from prediction or infer from candles
  const assetMetadata = prediction?.asset_metadata;
  const symbol = prediction?.symbol || candles[0]?.symbol || '';

  return (
    <div className="bg-gray-800 rounded-lg p-6 card-hover">
      {/* Header */}
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className="text-lg font-semibold text-gray-300">{getAssetTypeLabel(assetMetadata)} • {getFormattedSymbol(symbol, assetMetadata)}</h2>
          <div className="flex items-center gap-3 mt-1">
            <span className="text-2xl font-bold text-gray-100">
              {formatPrice(currentPrice, assetMetadata)}
            </span>
            {priceChange !== null && (
              <span className={`flex items-center gap-1 text-sm ${
                priceChange >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {priceChange >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
                {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(3)}%
              </span>
            )}
          </div>
        </div>
        {onRefresh && (
          <button
            onClick={onRefresh}
            className="p-2 text-gray-400 hover:text-gray-200 hover:bg-gray-700 rounded transition-colors"
            title="Refresh"
          >
            <RefreshCw size={18} />
          </button>
        )}
      </div>

      {/* Chart */}
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <XAxis
              dataKey="time"
              tick={{ fill: '#6b7280', fontSize: 11 }}
              axisLine={{ stroke: '#374151' }}
              tickLine={{ stroke: '#374151' }}
              interval="preserveStartEnd"
            />
            <YAxis
              type="number"
              domain={[minPrice, maxPrice]}
              tick={{ fill: '#6b7280', fontSize: 11 }}
              axisLine={{ stroke: '#374151' }}
              tickLine={{ stroke: '#374151' }}
              tickFormatter={(value) => value.toFixed(tickPrecision)}
              width={80}
              tickCount={6}
            />
            <Tooltip content={<CustomTooltip assetMetadata={assetMetadata} />} />

            {/* Current price reference line */}
            {currentPrice && (
              <ReferenceLine
                y={currentPrice}
                stroke="#3b82f6"
                strokeDasharray="3 3"
                strokeWidth={1}
              />
            )}

            {/* Price line */}
            <Line
              type="monotone"
              dataKey="close"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: '#3b82f6' }}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Chart explanation */}
      <p className="text-xs text-gray-500 mt-3">
        Real-time price chart showing {getFormattedSymbol(symbol, assetMetadata)} price movements.
        The blue line tracks the closing price, helping identify trends and entry/exit points for {getAssetTypeLabel(assetMetadata).toLowerCase()} trading.
      </p>

      {/* Recommendation indicator */}
      {prediction && (() => {
        // Normalize signal: API returns "long"/"short" in direction field
        const sig = prediction.signal || prediction.direction;
        const isBuy = sig === 'BUY' || sig === 'long' || sig === 1;
        const isSell = sig === 'SELL' || sig === 'short' || sig === -1 || sig === 0;

        // HOLD when confidence is below 70% threshold (should_trade = false)
        const isHold = prediction.should_trade === false;
        const recommendation = isHold ? 'HOLD' : (isBuy ? 'BUY' : 'SELL');

        const signalColor = isHold ? 'text-yellow-400' : isBuy ? 'text-green-400' : 'text-red-400';

        // Generate short explanation
        const confidencePct = ((prediction.confidence || 0) * 100).toFixed(0);
        const reason = isHold
          ? `Confidence ${confidencePct}% below 70% threshold`
          : isBuy
            ? `Bullish with ${confidencePct}% confidence`
            : `Bearish with ${confidencePct}% confidence`;

        return (
          <div className="mt-4 pt-4 border-t border-gray-700">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Agent Recommendation:</span>
              <span className={`text-sm font-medium ${signalColor}`}>
                {recommendation}
              </span>
            </div>
            <p className="text-xs text-gray-500 mt-1">{reason}</p>
          </div>
        );
      })()}
    </div>
  );
}

export default PriceChart;
