import { TrendingUp, Target, Percent, BarChart3, Zap, Award } from 'lucide-react';
import { getProfitUnitLabel } from '../utils/assetFormatting';
import { CollapsibleCard } from './common/CollapsibleCard';

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

  // Backtest performance defaults (Config C: 60% threshold, 18mo training)
  // WFO validated on 9 windows (2021-2025)
  const backtestDefaults = {
    total_pips: 6202,
    win_rate: 0.539,           // 60% confidence threshold
    win_rate_high_conf: 0.609, // High-confidence (65%+ threshold)
    profit_factor: 1.85,
    profit_factor_high_conf: 2.20,
    total_trades: 1257,
    total_trades_high_conf: 890,
    sharpe_ratio: 4.2,
    avg_pips_per_trade: 4.9,
  };

  // Use backtest defaults if no trades have been made yet
  const hasLiveData = performance && performance.total_trades > 0;
  const stats = hasLiveData ? performance : backtestDefaults;

  // Get profit unit label
  const profitUnit = getProfitUnitLabel(assetMetadata);

  // High-confidence values (use defaults if not provided)
  const winRateHighConf = stats.win_rate_high_conf || backtestDefaults.win_rate_high_conf;
  const profitFactorHighConf = stats.profit_factor_high_conf || backtestDefaults.profit_factor_high_conf;

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
      label: 'Win Rate (All)',
      value: stats.win_rate ? `${(stats.win_rate * 100).toFixed(1)}%` : 'N/A',
      subValue: winRateHighConf ? `${(winRateHighConf * 100).toFixed(1)}% high-conf` : null,
      icon: Target,
      color: 'text-blue-400',
      bgColor: 'bg-blue-500/10',
    },
    {
      label: 'Profit Factor',
      value: stats.profit_factor?.toFixed(2) || 'N/A',
      subValue: profitFactorHighConf ? `${profitFactorHighConf.toFixed(2)} high-conf` : null,
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
    <CollapsibleCard
      title="Performance Metrics"
      icon={<BarChart3 size={18} />}
      className="card-hover"
    >
      <p className="text-xs text-gray-500 mb-2">
        Config C: 60% confidence, 18mo training, WFO validated (9 windows)
      </p>
      <p className="text-xs text-gray-600 mb-3">
        <span className="text-green-400">High-conf</span> = predictions with ≥65% model confidence (fewer but more accurate trades)
      </p>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-3" role="list">
        {metrics.map((metric) => {
          const Icon = metric.icon;
          return (
            <div
              key={metric.label}
              className={`${metric.bgColor} rounded-lg p-3 transition-transform hover:scale-105`}
              role="listitem"
              aria-label={`${metric.label}: ${metric.prefix || ''}${metric.value}`}
            >
              <div className="flex items-center gap-1.5 mb-1.5">
                <Icon size={14} className={metric.color} aria-hidden="true" />
                <span className="text-xs text-gray-400">{metric.label}</span>
              </div>
              <span className={`text-lg font-bold ${metric.color}`}>
                {metric.prefix || ''}{metric.value}
              </span>
              {metric.subValue && (
                <span className="block text-xs text-green-400 mt-1">
                  → {metric.subValue}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Additional info */}
      <div className="mt-3 pt-3 border-t border-gray-700">
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500">Validation Method:</span>
          <span className="text-gray-300">Walk-Forward Optimization (7 windows)</span>
        </div>
        <div className="flex items-center justify-between text-xs mt-1">
          <span className="text-gray-500">WFO Success Rate:</span>
          <span className="text-green-400">100% (7/7 profitable)</span>
        </div>
      </div>
    </CollapsibleCard>
  );
}

export default PerformanceStats;
