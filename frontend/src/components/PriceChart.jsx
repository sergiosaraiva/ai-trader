import { useMemo } from 'react';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { TrendingUp, TrendingDown, RefreshCw } from 'lucide-react';

/**
 * Custom tooltip component for the price chart
 */
function CustomTooltip({ active, payload }) {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0]?.payload;
  if (!data) return null;

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-xl">
      <p className="text-gray-400 text-xs mb-2">{data.time}</p>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        <span className="text-gray-500">Open:</span>
        <span className="text-gray-300">{data.open?.toFixed(5)}</span>
        <span className="text-gray-500">High:</span>
        <span className="text-green-400">{data.high?.toFixed(5)}</span>
        <span className="text-gray-500">Low:</span>
        <span className="text-red-400">{data.low?.toFixed(5)}</span>
        <span className="text-gray-500">Close:</span>
        <span className={data.isBullish ? 'text-green-400' : 'text-red-400'}>
          {data.close?.toFixed(5)}
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

  const { minPrice, maxPrice, currentPrice, priceChange } = useMemo(() => {
    if (chartData.length === 0) {
      return { minPrice: 0, maxPrice: 1, currentPrice: null, priceChange: null };
    }

    const prices = chartData.flatMap(d => [d.high, d.low]);
    const min = Math.min(...prices);
    const max = Math.max(...prices);
    const padding = (max - min) * 0.1;
    const current = chartData[chartData.length - 1]?.close;
    const first = chartData[0]?.open;
    const change = current && first ? ((current - first) / first) * 100 : null;

    return {
      minPrice: min - padding,
      maxPrice: max + padding,
      currentPrice: current,
      priceChange: change,
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

  return (
    <div className="bg-gray-800 rounded-lg p-6 card-hover">
      {/* Header */}
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className="text-lg font-semibold text-gray-300">EUR/USD Price</h2>
          <div className="flex items-center gap-3 mt-1">
            <span className="text-2xl font-bold text-gray-100">
              {currentPrice?.toFixed(5) || 'N/A'}
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
              domain={[minPrice, maxPrice]}
              tick={{ fill: '#6b7280', fontSize: 11 }}
              axisLine={{ stroke: '#374151' }}
              tickLine={{ stroke: '#374151' }}
              tickFormatter={(value) => value.toFixed(4)}
              width={70}
            />
            <Tooltip content={<CustomTooltip />} />

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
            />

            {/* High-Low range */}
            <Bar
              dataKey={(d) => d.high - d.low}
              fill="transparent"
              stroke="#4b5563"
              barSize={1}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Prediction indicator */}
      {prediction && (
        <div className="mt-4 pt-4 border-t border-gray-700 flex items-center justify-between">
          <span className="text-sm text-gray-400">Model Prediction:</span>
          <span className={`text-sm font-medium ${
            prediction.signal === 'BUY' || prediction.signal === 1
              ? 'text-green-400'
              : prediction.signal === 'SELL' || prediction.signal === -1
                ? 'text-red-400'
                : 'text-gray-400'
          }`}>
            {prediction.signal === 1 ? 'BUY' : prediction.signal === -1 ? 'SELL' : prediction.signal || 'HOLD'}
            {prediction.confidence && ` (${(prediction.confidence * 100).toFixed(0)}%)`}
          </span>
        </div>
      )}
    </div>
  );
}

export default PriceChart;
