import { TrendingUp, Target, CheckCircle, Shield, AlertCircle, DollarSign } from 'lucide-react';
import { CollapsibleCard } from './common/CollapsibleCard';

/**
 * ModelHighlights - Displays key model performance highlights
 */
export function ModelHighlights({ performance, loading, error }) {
  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="bg-gray-700/50 rounded-lg p-4">
              <div className="h-3 bg-gray-600 rounded w-2/3 mb-2"></div>
              <div className="h-8 bg-gray-600 rounded w-1/2 mb-2"></div>
              <div className="h-3 bg-gray-600 rounded w-full"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 border border-yellow-500/30">
        <div className="flex items-center gap-2 text-yellow-400">
          <AlertCircle size={20} />
          <span>Performance data unavailable</span>
        </div>
        <p className="text-gray-500 text-sm mt-2">{error}</p>
      </div>
    );
  }

  // Use performance data directly from backend (no fallbacks)
  const highlights = performance?.highlights || [];
  const summary = performance?.summary || {
    headline: "Model Performance",
    description: "Performance data is loading..."
  };

  // Show empty state if no highlights are available
  if (!performance || highlights.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center gap-2 text-gray-400">
          <AlertCircle size={20} />
          <span>No performance highlights available</span>
        </div>
        <p className="text-gray-500 text-sm mt-2">
          Performance metrics are being generated. Please check back shortly.
        </p>
      </div>
    );
  }

  // Status-based color mapping (semantic colors)
  const getStatusColor = (status) => {
    switch (status) {
      case 'excellent':
        return 'text-green-400';
      case 'good':
        return 'text-blue-400';
      case 'moderate':
        return 'text-yellow-400';
      case 'poor':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  // Icon mapping by type, colored by status
  const getIcon = (type, status) => {
    const colorClass = getStatusColor(status);
    switch (type) {
      case 'agreement':
        return <Target size={24} className={colorClass} />;
      case 'validation':
        return <CheckCircle size={24} className={colorClass} />;
      case 'robustness':
        return <Shield size={24} className={colorClass} />;
      case 'pips':
        return <TrendingUp size={24} className={colorClass} />;
      case 'returns':
        return <DollarSign size={24} className={colorClass} />;
      default:
        return <TrendingUp size={24} className={colorClass} />;
    }
  };

  return (
    <CollapsibleCard
      title={summary.headline}
      icon={<TrendingUp size={20} />}
      className="card-hover"
    >
      {/* Description */}
      <p className="text-sm text-gray-400 leading-relaxed mb-4">{summary.description}</p>

      {/* Highlights Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {highlights.map((highlight, index) => (
          <div
            key={index}
            className="bg-gray-700/50 rounded-lg p-3 hover:bg-gray-700/70 transition-colors"
          >
            {/* Icon and Title */}
            <div className="flex items-start gap-2 mb-2">
              <div className="flex-shrink-0 mt-0.5">
                {getIcon(highlight.type, highlight.status)}
              </div>
              <div className="flex-grow">
                <h3 className="text-xs font-medium text-gray-300 leading-tight">
                  {highlight.title}
                </h3>
              </div>
            </div>

            {/* Value */}
            <div className={`text-2xl font-bold mb-1.5 ${getStatusColor(highlight.status)}`}>
              {highlight.value}
            </div>

            {/* Description */}
            <p className="text-xs text-gray-500 leading-relaxed">
              {highlight.description}
            </p>
          </div>
        ))}
      </div>

      {/* Footer Note */}
      <div className="mt-3 pt-3 border-t border-gray-700">
        <p className="text-xs text-gray-500 text-center">
          Metrics based on {performance?.metrics?.total_trades?.toLocaleString() ?? 'N/A'} trades with {performance?.metrics?.high_confidence?.threshold ? `${(performance.metrics.high_confidence.threshold * 100).toFixed(0)}%` : 'high'} confidence threshold
        </p>
      </div>
    </CollapsibleCard>
  );
}

export default ModelHighlights;
