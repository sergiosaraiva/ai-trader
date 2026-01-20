import { ArrowUpRight, ArrowDownRight, Clock, Minus } from 'lucide-react';
import { formatPrice } from '../utils/assetFormatting';

/**
 * TradeHistory - Displays recent trading signals
 */
export function TradeHistory({ signals, loading, error, assetMetadata }) {
  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-4"></div>
        <div className="space-y-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-16 bg-gray-700 rounded"></div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-lg font-semibold text-gray-300 mb-4">Signal History</h2>
        <p className="text-red-400">{error}</p>
      </div>
    );
  }

  const signalList = signals || [];

  const formatTime = (ts) => {
    if (!ts) return 'N/A';
    const date = new Date(ts);
    return date.toLocaleString([], {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getSignalType = (signal) => {
    // Null safety check
    if (!signal) return 'HOLD';

    // Use should_trade flag if available (explicit false = HOLD)
    if (Object.hasOwn(signal, 'should_trade') && signal.should_trade === false) {
      return 'HOLD';
    }

    // Handle both 'signal' and 'direction' field names from API
    const direction = signal.signal || signal.direction;
    if (direction === 'BUY' || direction === 'long' || direction === 1) return 'BUY';
    if (direction === 'SELL' || direction === 'short' || direction === -1) return 'SELL';
    return 'HOLD';
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 card-hover" role="region" aria-label="Signal History">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-semibold text-gray-300">Signal History</h2>
        <span className="text-xs text-gray-500">{signalList.length} signals</span>
      </div>

      {signalList.length === 0 ? (
        <div className="text-center py-8">
          <Clock size={32} className="mx-auto text-gray-600 mb-2" aria-hidden="true" />
          <p className="text-gray-500">No signals recorded yet</p>
          <p className="text-xs text-gray-600 mt-1">
            Signals will appear here as they are generated
          </p>
        </div>
      ) : (
        <div className="space-y-2 max-h-[400px] overflow-y-auto" role="list" aria-label="Trading signals list">
          {signalList.map((signal, idx) => {
            // Get signal type using full signal object (checks should_trade flag)
            const signalType = getSignalType(signal);
            const isBuy = signalType === 'BUY';
            const isSell = signalType === 'SELL';
            // Handle both 'price', 'current_price', and 'market_price' field names
            const price = formatPrice(
              signal.price || signal.current_price || signal.market_price,
              signal.asset_metadata || assetMetadata
            );
            const confidence = signal.confidence ? `${(signal.confidence * 100).toFixed(0)}%` : 'N/A';

            return (
              <div
                key={signal.id || idx}
                className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg hover:bg-gray-700 transition-colors"
                role="listitem"
                aria-label={`${signalType} signal at ${price}, confidence ${confidence}, ${formatTime(signal.timestamp)}`}
              >
                <div className="flex items-center gap-3">
                  <div
                    className={`p-2 rounded-full ${
                      isBuy
                        ? 'bg-green-500/20'
                        : isSell
                          ? 'bg-red-500/20'
                          : 'bg-yellow-500/20'
                    }`}
                    aria-hidden="true"
                  >
                    {isBuy ? (
                      <ArrowUpRight size={18} className="text-green-400" />
                    ) : isSell ? (
                      <ArrowDownRight size={18} className="text-red-400" />
                    ) : (
                      <Minus size={18} className="text-yellow-400" />
                    )}
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span
                        className={`font-medium ${
                          isBuy
                            ? 'text-green-400'
                            : isSell
                              ? 'text-red-400'
                              : 'text-yellow-400'
                        }`}
                      >
                        {signalType}
                      </span>
                      <span className="text-gray-400 text-sm">
                        @ {price}
                      </span>
                    </div>
                    <span className="text-xs text-gray-500">
                      {formatTime(signal.timestamp)}
                    </span>
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-sm text-gray-300">
                    {confidence}
                  </div>
                  <span className="text-xs text-gray-500">confidence</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default TradeHistory;
