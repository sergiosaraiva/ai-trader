import { TrendingUp, Target, CheckCircle, Shield, AlertCircle } from 'lucide-react';

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
        <p className="text-gray-500 text-sm mt-2">Using default highlights</p>
      </div>
    );
  }

  // Use performance data or defaults
  const highlights = performance?.highlights || [
    {
      type: "confidence",
      title: "High-Confidence Trading",
      value: "62.1%",
      description: "Win rate when model confidence exceeds 70%"
    },
    {
      type: "agreement",
      title: "Model Consensus",
      value: "82%",
      description: "Accuracy when all 3 timeframes agree"
    },
    {
      type: "validation",
      title: "Walk-Forward Validated",
      value: "7/7",
      description: "Profitable across all test periods"
    },
    {
      type: "robustness",
      title: "All-Regime Profitable",
      value: "6/6",
      description: "Works in trending and ranging markets"
    }
  ];

  const summary = performance?.summary || {
    headline: "Solid Performance",
    description: "The MTF Ensemble model demonstrates solid performance with 58.6% overall win rate and 2.26x profit factor. High-confidence predictions (≥70%) achieve 62.1% accuracy. Walk-forward optimization confirms 100% consistency across all test periods."
  };

  // Icon mapping
  const getIcon = (type) => {
    switch (type) {
      case 'confidence':
        return <TrendingUp size={24} className="text-green-400" />;
      case 'agreement':
        return <Target size={24} className="text-blue-400" />;
      case 'validation':
        return <CheckCircle size={24} className="text-purple-400" />;
      case 'robustness':
        return <Shield size={24} className="text-orange-400" />;
      default:
        return <TrendingUp size={24} className="text-gray-400" />;
    }
  };

  // Color mapping for values
  const getValueColor = (type) => {
    switch (type) {
      case 'confidence':
        return 'text-green-400';
      case 'agreement':
        return 'text-blue-400';
      case 'validation':
        return 'text-purple-400';
      case 'robustness':
        return 'text-orange-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 card-hover">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-xl font-bold text-gray-200 mb-2">{summary.headline}</h2>
        <p className="text-sm text-gray-400 leading-relaxed">{summary.description}</p>
      </div>

      {/* Highlights Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {highlights.map((highlight, index) => (
          <div
            key={index}
            className="bg-gray-700/50 rounded-lg p-4 hover:bg-gray-700/70 transition-colors"
          >
            {/* Icon and Title */}
            <div className="flex items-start gap-3 mb-3">
              <div className="flex-shrink-0 mt-0.5">
                {getIcon(highlight.type)}
              </div>
              <div className="flex-grow">
                <h3 className="text-sm font-medium text-gray-300 leading-tight">
                  {highlight.title}
                </h3>
              </div>
            </div>

            {/* Value */}
            <div className={`text-3xl font-bold mb-2 ${getValueColor(highlight.type)}`}>
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
      <div className="mt-4 pt-4 border-t border-gray-700">
        <p className="text-xs text-gray-500 text-center">
          Metrics based on {performance?.metrics?.total_trades || '1,093'} trades with 70% confidence threshold
        </p>
      </div>
    </div>
  );
}

export default ModelHighlights;
