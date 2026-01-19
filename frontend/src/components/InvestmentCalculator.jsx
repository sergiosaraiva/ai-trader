import { useState, useMemo } from 'react';
import { Calculator, TrendingUp, Info, DollarSign } from 'lucide-react';

/**
 * Investment return calculation utilities for forex trading
 */

// Backtest performance data (from WFO validation with 70% confidence threshold)
const BACKTEST_DATA = {
  totalPips: 8693,
  winRate: 0.621,
  profitFactor: 2.69,
  totalTrades: 966,
  periodStart: '2020-01-01',
  periodEnd: '2025-12-31',
  periodYears: 5,
};

// Forex trading constants
const FOREX_CONSTANTS = {
  standardLotSize: 100000,  // 100,000 units per standard lot
  pipValuePerLot: 10,       // $10 per pip for EUR/USD standard lot
  defaultLeverage: 30,      // 30:1 leverage (EU retail forex)
};

/**
 * Calculate potential returns based on investment amount
 * @param {number} investment - Initial investment in EUR
 * @param {number} totalPips - Total pips from backtest
 * @param {number} leverage - Trading leverage (default 30:1)
 * @returns {object} Calculated returns
 */
function calculateReturns(investment, totalPips, leverage = FOREX_CONSTANTS.defaultLeverage) {
  // Position size in lots based on investment and leverage
  const leveragedAmount = investment * leverage;
  const lots = leveragedAmount / FOREX_CONSTANTS.standardLotSize;

  // Pip value at this position size
  const pipValue = lots * FOREX_CONSTANTS.pipValuePerLot;

  // Total profit in USD (assuming EUR/USD ~ 1.0 for simplicity)
  const totalProfit = totalPips * pipValue;

  // Return on investment percentage
  const returnPercentage = (totalProfit / investment) * 100;

  // Annualized return
  const annualizedReturn = returnPercentage / BACKTEST_DATA.periodYears;

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
  const [showDetails, setShowDetails] = useState(false);

  // Calculate returns based on current investment
  const returns = useMemo(
    () => calculateReturns(investment, BACKTEST_DATA.totalPips),
    [investment]
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

      {/* Results */}
      <div className="bg-gray-700/50 rounded-lg p-4 mb-4">
        <p className="text-sm text-gray-400 mb-2">
          Based on {BACKTEST_DATA.periodYears}-year backtest performance:
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
              <span className="text-gray-300">{FOREX_CONSTANTS.defaultLeverage}:1</span>
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
              <span className="text-gray-300">+{BACKTEST_DATA.totalPips.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span>Win rate:</span>
              <span className="text-gray-300">{(BACKTEST_DATA.winRate * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between">
              <span>Profit factor:</span>
              <span className="text-gray-300">{BACKTEST_DATA.profitFactor}</span>
            </div>
            <div className="flex justify-between">
              <span>Period:</span>
              <span className="text-gray-300">{BACKTEST_DATA.periodStart} to {BACKTEST_DATA.periodEnd}</span>
            </div>
          </div>
        </div>
      )}

      {/* Disclaimer */}
      <p className="text-xs text-gray-500 leading-relaxed">
        <strong className="text-yellow-500/80">Note:</strong> This calculation is based on historical
        backtest performance with {FOREX_CONSTANTS.defaultLeverage}:1 leverage. Actual results may vary.
        Past performance does not guarantee future returns. Trading forex involves significant risk.
      </p>
    </div>
  );
}

export default InvestmentCalculator;
