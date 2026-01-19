import { Sparkles, RefreshCw, AlertCircle } from 'lucide-react';
import PropTypes from 'prop-types';

/**
 * ExplanationCard - Displays AI-generated explanation of the trading recommendation
 *
 * Shows a plain English explanation of why the AI is recommending
 * BUY, SELL, or HOLD based on technical and sentiment analysis.
 */
export function ExplanationCard({ explanation, loading, error, onRefresh }) {
  if (loading) {
    return (
      <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 border border-blue-500/20 rounded-lg p-4 animate-pulse">
        <div className="flex items-center gap-2 mb-2">
          <Sparkles size={18} className="text-blue-400" />
          <div className="h-4 bg-gray-700 rounded w-32"></div>
        </div>
        <div className="space-y-2">
          <div className="h-4 bg-gray-700/50 rounded w-full"></div>
          <div className="h-4 bg-gray-700/50 rounded w-3/4"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-gray-400">
            <AlertCircle size={18} />
            <span className="text-sm">AI explanation unavailable</span>
          </div>
          {onRefresh && (
            <button
              onClick={onRefresh}
              className="p-1.5 text-gray-500 hover:text-gray-300 hover:bg-gray-700 rounded transition-colors"
              title="Retry"
            >
              <RefreshCw size={14} />
            </button>
          )}
        </div>
      </div>
    );
  }

  if (!explanation?.explanation) {
    return null;
  }

  return (
    <div
      className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 border border-blue-500/20 rounded-lg p-4"
      role="region"
      aria-label="AI Analysis Explanation"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          <div className="flex items-center gap-2 mb-2">
            <Sparkles size={18} className="text-blue-400" aria-hidden="true" />
            <h2 className="text-sm font-medium text-blue-300">AI Analysis</h2>
            {explanation.cached && (
              <span className="text-xs text-gray-500 bg-gray-800/50 px-1.5 py-0.5 rounded">
                cached
              </span>
            )}
          </div>
          <p className="text-gray-300 text-sm leading-relaxed">
            {explanation.explanation}
          </p>
        </div>
        {onRefresh && (
          <button
            onClick={onRefresh}
            className="p-1.5 text-gray-500 hover:text-blue-400 hover:bg-gray-800/50 rounded transition-colors flex-shrink-0"
            title="Refresh explanation"
            aria-label="Refresh explanation"
          >
            <RefreshCw size={14} />
          </button>
        )}
      </div>
    </div>
  );
}

ExplanationCard.propTypes = {
  explanation: PropTypes.shape({
    explanation: PropTypes.string,
    cached: PropTypes.bool,
    generated_at: PropTypes.string,
    error: PropTypes.string,
  }),
  loading: PropTypes.bool,
  error: PropTypes.string,
  onRefresh: PropTypes.func,
};

ExplanationCard.defaultProps = {
  explanation: null,
  loading: false,
  error: null,
  onRefresh: null,
};

export default ExplanationCard;
