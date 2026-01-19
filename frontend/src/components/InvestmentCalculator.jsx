import { useState, useMemo, useEffect } from 'react';
import { Calculator, TrendingUp, Info, DollarSign, Calendar, Scale, AlertCircle, Loader2 } from 'lucide-react';
import api from '../api/client';

/**
 * Investment return calculation utilities for forex trading
 */

// Default/fallback data (used while loading or on error)
const DEFAULT_BACKTEST_PERIODS = {
  '1y': {
    label: 'Last Year',
    totalPips: 4317,
    winRate: 0.517,
    profitFactor: 1.73,
    totalTrades: 948,
    periodStart: '2024-07-01',
    periodEnd: '2025-06-30',
    periodYears: 1,
    periodMonths: 12,
  },
};

const DEFAULT_LEVERAGE_OPTIONS = [
  { value: 1, label: 'No Leverage (1:1)', risk: 'low' },
  { value: 10, label: '10:1', risk: 'medium' },
  { value: 20, label: '20:1', risk: 'high' },
  { value: 30, label: '30:1 (EU Retail)', risk: 'high' },
  { value: 50, label: '50:1', risk: 'extreme' },
];

const DEFAULT_FOREX_CONSTANTS = {
  standardLotSize: 100000,
  pipValuePerLot: 10,
};

/**
 * Transform API response to frontend format (snake_case to camelCase)
 */
function transformPeriodData(apiPeriods) {
  const transformed = {};
  for (const [key, data] of Object.entries(apiPeriods)) {
    transformed[key] = {
      label: data.label,
      totalPips: data.total_pips,
      winRate: data.win_rate,
      profitFactor: data.profit_factor,
      totalTrades: data.total_trades,
      periodStart: data.period_start,
      periodEnd: data.period_end,
      periodYears: data.period_years,
      periodMonths: data.period_months,
    };
  }
  return transformed;
}

/**
 * Calculate potential returns based on investment amount
 * @param {number} investment - Initial investment in EUR
 * @param {object} periodData - Backtest data for the selected period
 * @param {number} leverage - Trading leverage (1 = no leverage)
 * @param {object} forexConstants - Forex trading constants
 * @returns {object} Calculated returns
 */
function calculateReturns(investment, periodData, leverage = 1, forexConstants = DEFAULT_FOREX_CONSTANTS) {
  // Position size in lots based on investment and leverage
  const leveragedAmount = investment * leverage;
  const lots = leveragedAmount / forexConstants.standardLotSize;

  // Pip value at this position size
  const pipValue = lots * forexConstants.pipValuePerLot;

  // Total profit in USD (assuming EUR/USD ~ 1.0 for simplicity)
  const totalProfit = periodData.totalPips * pipValue;

  // Return on investment percentage
  const returnPercentage = (totalProfit / investment) * 100;

  // Annualized return (use periodYears for accurate annualization)
  const annualizedReturn = returnPercentage / periodData.periodYears;

  return {
    investment,
    leveragedAmount,
    lots: lots.toFixed(2),
    pipValue: pipValue.toFixed(2),
    totalProfit: Math.round(totalProfit),
    finalBalance: Math.round(investment + totalProfit),
    returnPercentage: returnPercentage.toFixed(0),
    annualizedReturn: annualizedReturn.toFixed(0),
  };
}

/**
 * InvestmentCalculator - Shows hypothetical returns based on backtest performance
 */
export function InvestmentCalculator({ assetMetadata }) {
  const [investment, setInvestment] = useState(1000);
  const [selectedPeriod, setSelectedPeriod] = useState('1y');
  const [leverage, setLeverage] = useState(1);
  const [showDetails, setShowDetails] = useState(false);

  // API data state
  const [backtestData, setBacktestData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch backtest data from API
  useEffect(() => {
    let isMounted = true;

    async function fetchBacktestData() {
      try {
        setLoading(true);
        setError(null);
        const response = await api.getBacktestPeriods();

        if (isMounted) {
          setBacktestData({
            periods: transformPeriodData(response.periods),
            leverageOptions: response.leverage_options,
            forexConstants: {
              standardLotSize: response.forex_constants.standard_lot_size,
              pipValuePerLot: response.forex_constants.pip_value_per_lot,
            },
            dataSource: response.data_source,
          });
        }
      } catch (err) {
        if (isMounted) {
          setError(err.message || 'Failed to load backtest data');
          // Use fallback data
          setBacktestData({
            periods: DEFAULT_BACKTEST_PERIODS,
            leverageOptions: DEFAULT_LEVERAGE_OPTIONS,
            forexConstants: DEFAULT_FOREX_CONSTANTS,
            dataSource: 'Fallback (offline)',
          });
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }

    fetchBacktestData();

    return () => {
      isMounted = false;
    };
  }, []);

  // Use API data or fallback
  const periods = backtestData?.periods || DEFAULT_BACKTEST_PERIODS;
  const leverageOptions = backtestData?.leverageOptions || DEFAULT_LEVERAGE_OPTIONS;
  const forexConstants = backtestData?.forexConstants || DEFAULT_FOREX_CONSTANTS;

  // Get the selected period data and leverage option
  const periodData = periods[selectedPeriod] || periods['1y'] || Object.values(periods)[0];
  const leverageOption = leverageOptions.find(l => l.value === leverage) || leverageOptions[0];

  // Calculate returns based on current investment, period, and leverage
  const returns = useMemo(
    () => calculateReturns(investment, periodData, leverage, forexConstants),
    [investment, periodData, leverage, forexConstants]
  );

  // Handle investment input change
  const handleInvestmentChange = (e) => {
    const value = e.target.value.replace(/[^0-9]/g, '');
    const numValue = parseInt(value, 10);
    if (!isNaN(numValue) && numValue >= 0 && numValue <= 1000000) {
      setInvestment(numValue);
    } else if (value === '') {
      setInvestment(0);
    }
  };

  // Quick amount buttons
  const quickAmounts = [500, 1000, 5000, 10000];

  return (
    <div className="bg-gray-800 rounded-lg p-6 card-hover" role="region" aria-label="Investment Calculator">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Calculator size={20} className="text-blue-400" />
          <h2 className="text-lg font-semibold text-gray-300">What If Calculator</h2>
        </div>
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="text-gray-500 hover:text-gray-300 transition-colors"
          aria-label="Show calculation details"
        >
          <Info size={18} />
        </button>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="flex items-center justify-center py-8 text-gray-400">
          <Loader2 size={24} className="animate-spin mr-2" />
          <span>Loading backtest data...</span>
        </div>
      )}

      {/* Error state (but still show calculator with fallback data) */}
      {error && !loading && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3 mb-4">
          <p className="text-xs text-yellow-400 flex items-center gap-2">
            <AlertCircle size={14} />
            Using cached data: {error}
          </p>
        </div>
      )}

      {/* Main content (show when not loading) */}
      {!loading && (
        <>
          {/* Investment Input */}
          <div className="mb-4">
            <label htmlFor="investment-input" className="block text-sm text-gray-400 mb-2">
              If you had invested:
            </label>
            <div className="relative">
              <DollarSign size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
              <input
                id="investment-input"
                type="text"
                inputMode="numeric"
                value={investment.toLocaleString()}
                onChange={handleInvestmentChange}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg pl-8 pr-12 py-2 text-white text-lg font-medium focus:outline-none focus:border-blue-500"
                aria-label="Investment amount"
              />
              <span className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 text-sm">EUR</span>
            </div>

            {/* Quick amount buttons */}
            <div className="flex gap-2 mt-2">
              {quickAmounts.map((amount) => (
                <button
                  key={amount}
                  onClick={() => setInvestment(amount)}
                  className={`px-3 py-1 text-xs rounded transition-colors ${
                    investment === amount
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                  }`}
                >
                  €{amount.toLocaleString()}
                </button>
              ))}
            </div>
          </div>

          {/* Time Period Selector */}
          <div className="mb-4">
            <label className="flex items-center gap-2 text-sm text-gray-400 mb-2">
              <Calendar size={14} />
              Time period:
            </label>
            <div className="flex flex-wrap gap-2">
              {Object.entries(periods).map(([key, data]) => (
                <button
                  key={key}
                  onClick={() => setSelectedPeriod(key)}
                  className={`px-3 py-1.5 text-xs rounded transition-colors ${
                    selectedPeriod === key
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                  }`}
                >
                  {data.label}
                </button>
              ))}
            </div>
          </div>

          {/* Leverage Selector */}
          <div className="mb-4">
            <label className="flex items-center gap-2 text-sm text-gray-400 mb-2">
              <Scale size={14} />
              Leverage:
            </label>
            <div className="flex flex-wrap gap-2">
              {leverageOptions.map((option) => (
                <button
                  key={option.value}
                  onClick={() => setLeverage(option.value)}
                  className={`px-3 py-1.5 text-xs rounded transition-colors ${
                    leverage === option.value
                      ? option.risk === 'low' ? 'bg-green-500 text-white'
                        : option.risk === 'medium' ? 'bg-yellow-500 text-white'
                        : option.risk === 'extreme' ? 'bg-red-600 text-white'
                        : 'bg-orange-500 text-white'
                      : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          {/* Results */}
          <div className="bg-gray-700/50 rounded-lg p-4 mb-4">
            <p className="text-sm text-gray-400 mb-2">
              Based on {periodData.label.toLowerCase()} backtest ({periodData.periodStart} to {periodData.periodEnd}):
            </p>

            <div className="flex items-center justify-between mb-3">
              <span className="text-gray-300">Your potential return:</span>
              <span className="text-2xl font-bold text-green-400">
                +€{returns.totalProfit.toLocaleString()}
              </span>
            </div>

            <div className="flex items-center justify-between mb-3">
              <span className="text-gray-300">Final balance:</span>
              <span className="text-xl font-semibold text-blue-400">
                €{returns.finalBalance.toLocaleString()}
              </span>
            </div>

            <div className="flex items-center gap-2 text-sm">
              <TrendingUp size={14} className="text-green-400" />
              <span className="text-green-400 font-medium">+{returns.returnPercentage}% total</span>
              <span className="text-gray-500">({returns.annualizedReturn}% per year)</span>
            </div>
          </div>

          {/* Expandable Details */}
          {showDetails && (
            <div className="bg-gray-700/30 rounded-lg p-4 mb-4 text-sm">
              <h3 className="text-gray-300 font-medium mb-2">Calculation Details</h3>
              <div className="space-y-1.5 text-gray-400">
                <div className="flex justify-between">
                  <span>Leverage used:</span>
                  <span className="text-gray-300">{leverage === 1 ? 'None (1:1)' : `${leverage}:1`}</span>
                </div>
                <div className="flex justify-between">
                  <span>Position size:</span>
                  <span className="text-gray-300">{returns.lots} lots</span>
                </div>
                <div className="flex justify-between">
                  <span>Pip value:</span>
                  <span className="text-gray-300">${returns.pipValue}/pip</span>
                </div>
                <div className="flex justify-between">
                  <span>Total pips earned:</span>
                  <span className="text-gray-300">+{periodData.totalPips.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Total trades:</span>
                  <span className="text-gray-300">{periodData.totalTrades.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Win rate:</span>
                  <span className="text-gray-300">{(periodData.winRate * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span>Profit factor:</span>
                  <span className="text-gray-300">{periodData.profitFactor}</span>
                </div>
                <div className="flex justify-between">
                  <span>Period:</span>
                  <span className="text-gray-300">{periodData.periodStart} to {periodData.periodEnd}</span>
                </div>
                {backtestData?.dataSource && (
                  <div className="flex justify-between pt-2 border-t border-gray-600">
                    <span>Data source:</span>
                    <span className="text-gray-500 text-xs">{backtestData.dataSource}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Leverage Warning - only show when leverage > 1 */}
          {leverage > 1 && (
            <div className={`border rounded-lg p-3 mb-3 ${
              leverageOption.risk === 'extreme'
                ? 'bg-red-500/10 border-red-500/30'
                : 'bg-yellow-500/10 border-yellow-500/30'
            }`}>
              <p className={`text-xs leading-relaxed ${
                leverageOption.risk === 'extreme' ? 'text-red-400' : 'text-yellow-400'
              }`}>
                <strong>⚠️ Leverage Warning:</strong> Returns shown use {leverage}:1 leverage,
                which amplifies both gains AND losses by {leverage}×. With this leverage,
                a {(100 / leverage).toFixed(1)}% adverse move would lose your entire investment.
                {leverageOption.risk === 'extreme' && ' This is extremely high risk!'}
              </p>
            </div>
          )}

          {/* Disclaimer */}
          <p className="text-xs text-gray-500 leading-relaxed">
            <strong className="text-gray-400">Disclaimer:</strong> These results are calculated by simulating trades
            using our AI models on historical market data — a "what if" scenario showing how the strategy would have
            performed in the past. This is not actual trading and past performance does not guarantee future returns.
            Trading with leverage involves significant risk of loss.
          </p>
        </>
      )}
    </div>
  );
}

export default InvestmentCalculator;
