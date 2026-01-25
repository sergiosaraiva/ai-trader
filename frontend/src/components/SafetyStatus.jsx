import { Shield, AlertTriangle, CheckCircle, XCircle, Activity } from 'lucide-react';

/**
 * SafetyStatus - Safety systems monitoring component
 */
export function SafetyStatus({ safety, loading }) {
  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-4 animate-pulse">
        <div className="h-4 bg-gray-700 rounded w-1/3 mb-3"></div>
        <div className="space-y-2">
          <div className="h-8 bg-gray-700 rounded"></div>
          <div className="h-8 bg-gray-700 rounded"></div>
          <div className="h-8 bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  if (!safety) {
    return (
      <div className="bg-gray-800 rounded-lg p-4">
        <h2 className="text-base font-semibold text-gray-300 mb-3">Safety Systems</h2>
        <div className="text-center py-6">
          <p className="text-gray-500 text-sm">Safety data not available</p>
        </div>
      </div>
    );
  }

  const isSafeToTrade = safety.is_safe_to_trade;
  const killSwitch = safety.kill_switch || {};
  const circuitBreakers = safety.circuit_breakers || {};
  const dailyMetrics = safety.daily_metrics || {};
  const accountMetrics = safety.account_metrics || {};

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <h3 className="text-base font-semibold text-gray-300 mb-3 flex items-center gap-2">
        <Shield size={18} className={isSafeToTrade ? 'text-green-400' : 'text-red-400'} />
        Safety Systems
      </h3>
      {/* Overall Safety Status */}
      <div className={`mb-3 p-3 rounded border ${
        isSafeToTrade
          ? 'bg-green-500/10 border-green-500/30'
          : 'bg-red-500/10 border-red-500/30'
      }`}>
        <div className="flex items-center gap-2">
          {isSafeToTrade ? (
            <CheckCircle size={20} className="text-green-400" />
          ) : (
            <XCircle size={20} className="text-red-400" />
          )}
          <div>
            <span className={`text-xs font-medium ${
              isSafeToTrade ? 'text-green-400' : 'text-red-400'
            }`}>
              {isSafeToTrade ? 'SAFE TO TRADE' : 'TRADING HALTED'}
            </span>
            <p className="text-xs text-gray-500">
              {isSafeToTrade
                ? 'All safety systems operational'
                : 'Safety systems have halted trading'}
            </p>
          </div>
        </div>
      </div>

      {/* Kill Switch Status */}
      <div className="mb-3">
        <h3 className="text-xs text-gray-500 mb-2 flex items-center gap-2">
          <AlertTriangle size={14} />
          Kill Switch
        </h3>
        <div className={`p-2 rounded ${
          killSwitch.is_active
            ? 'bg-red-500/10 border border-red-500/30'
            : 'bg-gray-700/30'
        }`}>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-300">Status</span>
            <span className={`text-xs font-medium ${
              killSwitch.is_active ? 'text-red-400 animate-pulse' : 'text-green-400'
            }`}>
              {killSwitch.is_active ? 'ACTIVE' : 'Inactive'}
            </span>
          </div>
          {killSwitch.reason && (
            <p className="text-xs text-red-400 mt-1">
              Reason: {killSwitch.reason}
            </p>
          )}
          {killSwitch.triggered_at && (
            <p className="text-xs text-gray-500 mt-1">
              Triggered: {new Date(killSwitch.triggered_at).toLocaleString()}
            </p>
          )}
        </div>
      </div>

      {/* Circuit Breakers */}
      <div className="mb-3">
        <h3 className="text-xs text-gray-500 mb-2 flex items-center gap-2">
          <Activity size={14} />
          Circuit Breakers
        </h3>
        <div className="space-y-1">
          <CircuitBreakerItem
            label="Overall State"
            value={circuitBreakers.overall_state || 'unknown'}
            isActive={circuitBreakers.overall_state === 'active'}
          />
          <CircuitBreakerItem
            label="Can Trade"
            value={circuitBreakers.can_trade ? 'Yes' : 'No'}
            isActive={circuitBreakers.can_trade}
          />
          {circuitBreakers.active_breakers && circuitBreakers.active_breakers.length > 0 && (
            <div className="mt-2 p-2 bg-yellow-500/10 border border-yellow-500/30 rounded">
              <p className="text-xs text-yellow-400">
                Active breakers: {circuitBreakers.active_breakers.join(', ')}
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Daily Limits */}
      {dailyMetrics && (
        <div className="mb-3">
          <h3 className="text-xs text-gray-500 mb-2">Daily Limits</h3>
          <div className="space-y-1">
            <MetricBar
              label="Trades"
              value={dailyMetrics.trades || 0}
              max={10}
              unit=""
            />
            <MetricBar
              label="Loss %"
              value={dailyMetrics.loss_pct || 0}
              max={5}
              unit="%"
              isNegative
            />
          </div>
        </div>
      )}

      {/* Account Metrics */}
      {accountMetrics && (
        <div>
          <h3 className="text-xs text-gray-500 mb-2">Account Metrics</h3>
          <div className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">Current Equity</span>
              <span className="text-green-400">
                ${accountMetrics.current_equity?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
              </span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">Peak Equity</span>
              <span className="text-gray-300">
                ${accountMetrics.peak_equity?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
              </span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">Drawdown</span>
              <span className={accountMetrics.drawdown_pct > 0 ? 'text-red-400' : 'text-green-400'}>
                {accountMetrics.drawdown_pct?.toFixed(2) || '0.00'}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * CircuitBreakerItem - Individual circuit breaker status
 */
function CircuitBreakerItem({ label, value, isActive }) {
  return (
    <div className="flex items-center justify-between p-2 bg-gray-700/30 rounded">
      <span className="text-xs text-gray-400">{label}</span>
      <span className={`text-xs font-medium ${
        isActive ? 'text-green-400' : 'text-red-400'
      }`}>
        {value}
      </span>
    </div>
  );
}

/**
 * MetricBar - Progress bar for limits
 */
function MetricBar({ label, value, max, unit, isNegative = false }) {
  const percentage = Math.min((value / max) * 100, 100);
  const isHigh = percentage > 80;

  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-400">{label}</span>
        <span className={isHigh ? 'text-yellow-400' : 'text-gray-300'}>
          {value.toFixed(isNegative ? 2 : 0)}{unit} / {max}{unit}
        </span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-2">
        <div
          className={`h-2 rounded-full transition-all ${
            isHigh ? 'bg-yellow-500' : 'bg-green-500'
          }`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

export default SafetyStatus;
