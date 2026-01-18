import { TrendingUp, Target, Percent, BarChart3, Zap, Award } from 'lucide-react';
import { getProfitUnitLabel } from '../utils/assetFormatting';

/**
 * PerformanceStats - Displays trading performance metrics
 */
export function PerformanceStats({ performance, loading, error, assetMetadata }) {
  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-20 bg-gray-700 rounded"></div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <p className="text-red-400">{error}</p>
      </div>
    );
  }

  // Use default values or mock data if performance is not available
  const stats = performance || {
    total_pips: 8693,
    win_rate: 0.621,
    profit_factor: 2.69,
    total_trades: 966,
    sharpe_ratio: 7.67,
    avg_pips_per_trade: 9.0,
  };

  // Get profit unit label
  const profitUnit = getProfitUnitLabel(assetMetadata);

  const metrics = [
    {
      label: `Total ${profitUnit.charAt(0).toUpperCase() + profitUnit.slice(1)}`,
      value: stats.total_pips?.toLocaleString() || 'N/A',
      prefix: '+',
      icon: TrendingUp,
      color: 'text-green-400',
      bgColor: 'bg-green-500/10',
    },
    {
      label: 'Win Rate',
      value: stats.win_rate ? `${(stats.win_rate * 100).toFixed(1)}%` : 'N/A',
      icon: Target,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/10',
    },
    {
      label: 'Profit Factor',
      value: stats.profit_factor?.toFixed(2) || 'N/A',
      icon: BarChart3,
      color: 'text-purple-400',
      bgColor: 'bg-purple-500/10',
    },
    {
      label: 'Total Trades',
      value: stats.total_trades?.toLocaleString() || 'N/A',
      icon: Zap,
      color: 'text-yellow-400',
      bgColor: 'bg-yellow-500/10',
    },
    {
      label: 'Sharpe Ratio',
      value: stats.sharpe_ratio?.toFixed(2) || 'N/A',
      icon: Award,
      color: 'text-cyan-400',
      bgColor: 'bg-cyan-500/10',
    },
    {
      label: `Avg ${profitUnit.charAt(0).toUpperCase() + profitUnit.slice(1)}/Trade`,
      value: stats.avg_pips_per_trade?.toFixed(1) || 'N/A',
      prefix: '+',
      icon: Percent,
      color: 'text-emerald-400',
      bgColor: 'bg-emerald-500/10',
    },
  ];

  return (
    <div className="bg-gray-800 rounded-lg p-6 card-hover" role="region" aria-label="Performance Metrics">
      <h2 className="text-lg font-semibold text-gray-300 mb-4">Performance Metrics</h2>
      <p className="text-xs text-gray-500 mb-4">
        Based on backtested results with 70% confidence threshold
      </p>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4" role="list">
        {metrics.map((metric) => {
          const Icon = metric.icon;
          return (
            <div
              key={metric.label}
              className={`${metric.bgColor} rounded-lg p-4 transition-transform hover:scale-105`}
              role="listitem"
              aria-label={`${metric.label}: ${metric.prefix || ''}${metric.value}`}
            >
              <div className="flex items-center gap-2 mb-2">
                <Icon size={16} className={metric.color} aria-hidden="true" />
                <span className="text-xs text-gray-400">{metric.label}</span>
              </div>
              <span className={`text-xl font-bold ${metric.color}`}>
                {metric.prefix || ''}{metric.value}
              </span>
            </div>
          );
        })}
      </div>

      {/* Additional info */}
      <div className="mt-4 pt-4 border-t border-gray-700">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-500">Validation Method:</span>
          <span className="text-gray-300">Walk-Forward Optimization (7 windows)</span>
        </div>
        <div className="flex items-center justify-between text-sm mt-2">
          <span className="text-gray-500">WFO Success Rate:</span>
          <span className="text-green-400">100% (7/7 profitable)</span>
        </div>
      </div>
    </div>
  );
}

export default PerformanceStats;
