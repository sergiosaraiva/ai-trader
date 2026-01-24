import { useMemo, memo, useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import {
  ComposedChart,
  Bar,
  Cell,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  CartesianGrid,
} from 'recharts';
import { TrendingUp, History, Activity } from 'lucide-react';
import { getProfitUnitLabel, getAssetTypeLabel } from '../utils/assetFormatting';
import { api } from '../api/client';

// Constants
const DAYS_TO_DISPLAY = 30;

/**
 * Custom tooltip component for the performance chart
 * Memoized to prevent unnecessary re-renders
 */
const CustomTooltip = memo(function CustomTooltip({ active, payload, profitUnit }) {
  if (!active || !payload || payload.length === 0) return null;

  const data = payload[0]?.payload;
  if (!data) return null;

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg p-3 shadow-xl">
      <p className="text-gray-400 text-xs mb-2">{data.date}</p>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        <span className="text-gray-500">Daily P&L:</span>
        <span className={data.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
          {data.daily_pnl >= 0 ? '+' : ''}{data.daily_pnl.toFixed(1)} {profitUnit}
        </span>
        <span className="text-gray-500">Cumulative:</span>
        <span className="text-blue-400">
          {data.cumulative_pnl >= 0 ? '+' : ''}{data.cumulative_pnl.toFixed(1)} {profitUnit}
        </span>
        <span className="text-gray-500">Trades:</span>
        <span className="text-gray-300">{data.trades}</span>
        {data.wins !== undefined && (
          <>
            <span className="text-gray-500">Win Rate:</span>
            <span className="text-gray-300">{data.win_rate}%</span>
          </>
        )}
      </div>
    </div>
  );
});

CustomTooltip.propTypes = {
  active: PropTypes.bool,
  payload: PropTypes.array,
  profitUnit: PropTypes.string,
};

/**
 * View mode toggle button component
 */
const ViewToggle = memo(function ViewToggle({ mode, onModeChange }) {
  return (
    <div className="flex gap-1 bg-gray-700/50 rounded-lg p-1">
      <button
        onClick={() => onModeChange('live')}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
          mode === 'live'
            ? 'bg-blue-600 text-white'
            : 'text-gray-400 hover:text-white hover:bg-gray-600'
        }`}
        aria-pressed={mode === 'live'}
      >
        <Activity size={14} />
        Live
      </button>
      <button
        onClick={() => onModeChange('whatif')}
        className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
          mode === 'whatif'
            ? 'bg-purple-600 text-white'
            : 'text-gray-400 hover:text-white hover:bg-gray-600'
        }`}
        aria-pressed={mode === 'whatif'}
      >
        <History size={14} />
        What If
      </button>
    </div>
  );
});

ViewToggle.propTypes = {
  mode: PropTypes.oneOf(['live', 'whatif']).isRequired,
  onModeChange: PropTypes.func.isRequired,
};

/**
 * PerformanceChart - Displays 30-day daily P&L with cumulative line
 * Supports both Live trading performance and What-If historical simulation
 */
export function PerformanceChart({ trades, loading, error, assetMetadata }) {
  const profitUnit = getProfitUnitLabel(assetMetadata);

  // View mode state: 'live' or 'whatif'
  const [viewMode, setViewMode] = useState('whatif'); // Default to whatif since live may be empty
  const [whatIfData, setWhatIfData] = useState(null);
  const [whatIfLoading, setWhatIfLoading] = useState(false);
  const [whatIfError, setWhatIfError] = useState(null);

  // Fetch What-If data when mode changes to 'whatif'
  useEffect(() => {
    if (viewMode === 'whatif' && !whatIfData && !whatIfLoading) {
      setWhatIfLoading(true);
      setWhatIfError(null);

      api.getWhatIfPerformance(30, 0.70)
        .then(data => {
          setWhatIfData(data);
          setWhatIfLoading(false);
        })
        .catch(err => {
          console.error('Failed to fetch What-If data:', err);
          setWhatIfError(err.message || 'Failed to load simulation data');
          setWhatIfLoading(false);
        });
    }
  }, [viewMode, whatIfData, whatIfLoading]);

  // Process live trades data
  const liveChartData = useMemo(() => {
    if (!trades || !Array.isArray(trades) || trades.length === 0) {
      return [];
    }

    // Filter to closed trades only
    const closedTrades = trades.filter(trade => trade.status === 'closed');

    if (closedTrades.length === 0) {
      return [];
    }

    // Group by date (extract date from exit_time)
    const dailyData = {};

    closedTrades.forEach(trade => {
      if (!trade.exit_time) return;

      const exitDate = new Date(trade.exit_time);
      const dateKey = exitDate.toISOString().split('T')[0]; // YYYY-MM-DD

      if (!dailyData[dateKey]) {
        dailyData[dateKey] = {
          date: dateKey,
          trades: 0,
          wins: 0,
          losses: 0,
          total_pnl: 0,
        };
      }

      dailyData[dateKey].trades += 1;
      dailyData[dateKey].total_pnl += trade.pips || 0;

      if (trade.is_winner) {
        dailyData[dateKey].wins += 1;
      } else {
        dailyData[dateKey].losses += 1;
      }
    });

    // Convert to array and sort by date
    const sortedData = Object.values(dailyData).sort((a, b) =>
      new Date(a.date) - new Date(b.date)
    );

    // Take last N days
    const last30Days = sortedData.slice(-DAYS_TO_DISPLAY);

    // Calculate cumulative P&L
    let cumulativePnl = 0;
    const result = last30Days.map(day => {
      cumulativePnl += day.total_pnl;

      const winRate = day.trades > 0
        ? ((day.wins / day.trades) * 100).toFixed(0)
        : 0;

      return {
        date: new Date(day.date).toLocaleDateString([], {
          month: 'short',
          day: 'numeric'
        }),
        daily_pnl: day.total_pnl,
        cumulative_pnl: cumulativePnl,
        trades: day.trades,
        wins: day.wins,
        losses: day.losses,
        win_rate: winRate,
      };
    });

    return result;
  }, [trades]);

  // Process What-If data for chart
  const whatIfChartData = useMemo(() => {
    if (!whatIfData?.daily_performance) {
      return [];
    }

    return whatIfData.daily_performance.map(day => ({
      date: new Date(day.date).toLocaleDateString([], {
        month: 'short',
        day: 'numeric'
      }),
      daily_pnl: day.daily_pnl,
      cumulative_pnl: day.cumulative_pnl,
      trades: day.trades,
      wins: day.wins,
      win_rate: day.win_rate,
    }));
  }, [whatIfData]);

  // Select chart data based on view mode
  const chartData = viewMode === 'live' ? liveChartData : whatIfChartData;
  const isLoading = viewMode === 'live' ? loading : whatIfLoading;
  const displayError = viewMode === 'live' ? error : whatIfError;

  const { totalDays, profitableDays, totalPnl, maxDrawdown, bestDay, worstDay } = useMemo(() => {
    if (chartData.length === 0) {
      return { totalDays: 0, profitableDays: 0, totalPnl: 0, maxDrawdown: 0, bestDay: 0, worstDay: 0 };
    }

    const total = chartData.length;
    const profitable = chartData.filter(d => d.daily_pnl > 0).length;
    const pnl = chartData[chartData.length - 1]?.cumulative_pnl || 0;

    // Calculate max drawdown, best and worst days
    let peak = 0;
    let maxDD = 0;
    let best = -Infinity;
    let worst = Infinity;

    chartData.forEach(day => {
      if (day.cumulative_pnl > peak) {
        peak = day.cumulative_pnl;
      }
      const drawdown = peak - day.cumulative_pnl;
      if (drawdown > maxDD) {
        maxDD = drawdown;
      }
      if (day.daily_pnl > best) {
        best = day.daily_pnl;
      }
      if (day.daily_pnl < worst) {
        worst = day.daily_pnl;
      }
    });

    return {
      totalDays: total,
      profitableDays: profitable,
      totalPnl: pnl,
      maxDrawdown: maxDD,
      bestDay: best === -Infinity ? 0 : best,
      worstDay: worst === Infinity ? 0 : worst,
    };
  }, [chartData]);

  // Loading state
  if (isLoading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 h-[400px]">
        <div className="flex justify-between items-center mb-4">
          <div className="h-4 bg-gray-700 rounded w-1/4 animate-pulse"></div>
          <ViewToggle mode={viewMode} onModeChange={setViewMode} />
        </div>
        <div className="h-full bg-gray-700/50 rounded animate-pulse"></div>
      </div>
    );
  }

  // Error state
  if (displayError) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 h-[400px]">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold text-gray-300">30-Day Performance</h2>
          <ViewToggle mode={viewMode} onModeChange={setViewMode} />
        </div>
        <div className="flex items-center justify-center h-[300px]">
          <p className="text-red-400">{displayError}</p>
        </div>
      </div>
    );
  }

  // Empty state
  if (chartData.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 h-[400px]">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold text-gray-300">30-Day Performance</h2>
          <ViewToggle mode={viewMode} onModeChange={setViewMode} />
        </div>
        <div className="flex flex-col items-center justify-center h-[300px] text-center">
          <p className="text-gray-500 mb-2">
            {viewMode === 'live'
              ? 'No live trading data available yet'
              : 'No simulation data available'}
          </p>
          {viewMode === 'live' && (
            <p className="text-gray-600 text-sm">
              Switch to &quot;What If&quot; to see historical simulation
            </p>
          )}
        </div>
      </div>
    );
  }

  // Calculate profitable percentage safely (avoid division by zero)
  const profitablePercentage = totalDays > 0
    ? ((profitableDays / totalDays) * 100).toFixed(0)
    : 0;

  // Summary info for What-If mode
  const whatIfSummary = viewMode === 'whatif' && whatIfData?.summary;

  return (
    <div
      className="bg-gray-800 rounded-lg p-6 card-hover"
      role="region"
      aria-label="30-day trading performance chart"
    >
      {/* Header */}
      <div className="flex justify-between items-start mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold text-gray-300">30-Day Performance</h2>
            {viewMode === 'whatif' && (
              <span className="px-2 py-0.5 bg-purple-600/20 text-purple-400 text-xs rounded-full">
                Simulation
              </span>
            )}
          </div>
          <div className="flex items-center gap-3 mt-1">
            <span className={`text-2xl font-bold ${
              totalPnl >= 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              {totalPnl >= 0 ? '+' : ''}{totalPnl.toFixed(1)} {profitUnit}
            </span>
            <span className="flex items-center gap-1 text-sm text-gray-400">
              <TrendingUp size={16} aria-hidden="true" />
              {totalDays} days • {profitableDays} profitable ({profitablePercentage}%)
            </span>
          </div>
          {whatIfSummary && (
            <p className="text-xs text-gray-500 mt-1">
              {whatIfSummary.total_trades} trades • {whatIfSummary.win_rate}% win rate •
              {whatIfSummary.confidence_threshold * 100}% confidence threshold
            </p>
          )}
        </div>
        <ViewToggle mode={viewMode} onModeChange={setViewMode} />
      </div>
      {/* Screen reader summary */}
      <span className="sr-only">
        Total {profitUnit}: {totalPnl >= 0 ? 'positive' : 'negative'} {Math.abs(totalPnl).toFixed(1)}.
        {totalDays} trading days with {profitableDays} profitable days.
      </span>

      {/* Chart */}
      <div className="h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="date"
              tick={{ fill: '#6b7280', fontSize: 11 }}
              axisLine={{ stroke: '#374151' }}
              tickLine={{ stroke: '#374151' }}
              interval="preserveStartEnd"
            />
            <YAxis
              yAxisId="left"
              tick={{ fill: '#6b7280', fontSize: 11 }}
              axisLine={{ stroke: '#374151' }}
              tickLine={{ stroke: '#374151' }}
              label={{ value: `Daily P&L (${profitUnit})`, angle: -90, position: 'insideLeft', fill: '#6b7280', fontSize: 11 }}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              tick={{ fill: '#6b7280', fontSize: 11 }}
              axisLine={{ stroke: '#374151' }}
              tickLine={{ stroke: '#374151' }}
              label={{ value: `Cumulative (${profitUnit})`, angle: 90, position: 'insideRight', fill: '#6b7280', fontSize: 11 }}
            />
            <Tooltip content={<CustomTooltip profitUnit={profitUnit} />} />

            {/* Zero reference line */}
            <ReferenceLine
              yAxisId="left"
              y={0}
              stroke="#6b7280"
              strokeDasharray="3 3"
              strokeWidth={1}
            />

            {/* Daily P&L bars */}
            <Bar
              yAxisId="left"
              dataKey="daily_pnl"
              radius={[4, 4, 0, 0]}
              isAnimationActive={false}
            >
              {chartData.map((entry, index) => (
                <Cell
                  key={`bar-${index}`}
                  fill={entry.daily_pnl >= 0 ? '#22c55e' : '#ef4444'}
                />
              ))}
            </Bar>

            {/* Cumulative P&L line */}
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="cumulative_pnl"
              stroke={viewMode === 'whatif' ? '#a855f7' : '#3b82f6'}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: viewMode === 'whatif' ? '#a855f7' : '#3b82f6' }}
              isAnimationActive={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Chart explanation */}
      <p className="text-xs text-gray-500 mt-3">
        {viewMode === 'whatif' ? (
          <>
            Historical simulation showing what would have happened following AI predictions over the last 30 days.
            Based on {whatIfSummary?.confidence_threshold * 100 || 70}% confidence threshold.
          </>
        ) : (
          <>
            Live performance of AI recommendations for {getAssetTypeLabel(assetMetadata).toLowerCase()} trading.
            Green/red bars show daily profit or loss in {profitUnit}, while the blue line tracks cumulative results over time.
          </>
        )}
      </p>

      {/* Statistics footer */}
      <div className="mt-4 pt-4 border-t border-gray-700 grid grid-cols-3 gap-4 text-sm" role="list" aria-label="Performance statistics">
        <div role="listitem">
          <p className="text-gray-500">Best Day</p>
          <p className="text-green-400 font-semibold">
            +{bestDay.toFixed(1)} {profitUnit}
          </p>
        </div>
        <div role="listitem">
          <p className="text-gray-500">Worst Day</p>
          <p className="text-red-400 font-semibold">
            {worstDay.toFixed(1)} {profitUnit}
          </p>
        </div>
        <div role="listitem">
          <p className="text-gray-500">Max Drawdown</p>
          <p className="text-orange-400 font-semibold">
            -{maxDrawdown.toFixed(1)} {profitUnit}
          </p>
        </div>
      </div>
    </div>
  );
}

// PropTypes validation
PerformanceChart.propTypes = {
  trades: PropTypes.arrayOf(PropTypes.shape({
    status: PropTypes.string,
    exit_time: PropTypes.string,
    pips: PropTypes.number,
    is_winner: PropTypes.bool,
  })),
  loading: PropTypes.bool,
  error: PropTypes.string,
  assetMetadata: PropTypes.shape({
    asset_type: PropTypes.string,
    price_precision: PropTypes.number,
    profit_unit: PropTypes.string,
  }),
};

PerformanceChart.defaultProps = {
  trades: [],
  loading: false,
  error: null,
  assetMetadata: null,
};

export default PerformanceChart;
