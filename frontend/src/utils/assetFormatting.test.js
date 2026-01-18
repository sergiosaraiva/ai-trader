/**
 * Tests for asset formatting utilities
 */

import { describe, it, expect } from 'vitest';
import {
  formatPrice,
  getProfitUnitLabel,
  formatProfit,
  getFormattedSymbol,
  getAssetTypeLabel,
  inferAssetMetadata,
} from './assetFormatting';

describe('formatPrice', () => {
  it('formats forex price with 5 decimals', () => {
    const metadata = { price_precision: 5 };
    const result = formatPrice(1.08543, metadata);
    expect(result).toBe('1.08543');
  });

  it('formats crypto price with 8 decimals', () => {
    const metadata = { price_precision: 8 };
    const result = formatPrice(50123.12345678, metadata);
    expect(result).toBe('50123.12345678');
  });

  it('formats stock price with 2 decimals', () => {
    const metadata = { price_precision: 2 };
    const result = formatPrice(150.5, metadata);
    expect(result).toBe('150.50');
  });

  it('rounds price to specified precision', () => {
    const metadata = { price_precision: 2 };
    const result = formatPrice(150.567, metadata);
    expect(result).toBe('150.57');
  });

  it('handles null price', () => {
    const metadata = { price_precision: 5 };
    const result = formatPrice(null, metadata);
    expect(result).toBe('N/A');
  });

  it('handles undefined price', () => {
    const metadata = { price_precision: 5 };
    const result = formatPrice(undefined, metadata);
    expect(result).toBe('N/A');
  });

  it('uses default precision when metadata is missing', () => {
    const result = formatPrice(1.08543, null);
    expect(result).toBe('1.08543'); // Default is 5
  });

  it('uses default precision when price_precision is missing', () => {
    const result = formatPrice(1.08543, {});
    expect(result).toBe('1.08543'); // Default is 5
  });

  it('handles zero price', () => {
    const metadata = { price_precision: 2 };
    const result = formatPrice(0, metadata);
    expect(result).toBe('0.00');
  });

  it('handles negative price', () => {
    const metadata = { price_precision: 2 };
    const result = formatPrice(-150.5, metadata);
    expect(result).toBe('-150.50');
  });
});

describe('getProfitUnitLabel', () => {
  it('returns pips for forex', () => {
    const metadata = { profit_unit: 'pips' };
    const result = getProfitUnitLabel(metadata);
    expect(result).toBe('pips');
  });

  it('returns dollars for crypto', () => {
    const metadata = { profit_unit: 'dollars' };
    const result = getProfitUnitLabel(metadata);
    expect(result).toBe('dollars');
  });

  it('returns points for stocks', () => {
    const metadata = { profit_unit: 'points' };
    const result = getProfitUnitLabel(metadata);
    expect(result).toBe('points');
  });

  it('returns default when metadata is missing', () => {
    const result = getProfitUnitLabel(null);
    expect(result).toBe('pips'); // Default
  });

  it('returns default when profit_unit is missing', () => {
    const result = getProfitUnitLabel({});
    expect(result).toBe('pips'); // Default
  });
});

describe('formatProfit', () => {
  it('formats positive profit with plus sign', () => {
    const metadata = { profit_unit: 'pips' };
    const result = formatProfit(123.4, metadata);
    expect(result).toBe('+123.4 pips');
  });

  it('formats negative profit with minus sign', () => {
    const metadata = { profit_unit: 'pips' };
    const result = formatProfit(-45.6, metadata);
    expect(result).toBe('-45.6 pips');
  });

  it('formats zero profit with plus sign', () => {
    const metadata = { profit_unit: 'pips' };
    const result = formatProfit(0, metadata);
    expect(result).toBe('+0.0 pips');
  });

  it('formats crypto profit in dollars', () => {
    const metadata = { profit_unit: 'dollars' };
    const result = formatProfit(1234.5, metadata);
    expect(result).toBe('+1234.5 dollars');
  });

  it('formats stock profit in points', () => {
    const metadata = { profit_unit: 'points' };
    const result = formatProfit(567.8, metadata);
    expect(result).toBe('+567.8 points');
  });

  it('formats to 1 decimal place', () => {
    const metadata = { profit_unit: 'pips' };
    const result = formatProfit(123.456, metadata);
    expect(result).toBe('+123.5 pips');
  });

  it('handles null value', () => {
    const metadata = { profit_unit: 'pips' };
    const result = formatProfit(null, metadata);
    expect(result).toBe('N/A');
  });

  it('handles undefined value', () => {
    const metadata = { profit_unit: 'pips' };
    const result = formatProfit(undefined, metadata);
    expect(result).toBe('N/A');
  });

  it('uses default unit when metadata is missing', () => {
    const result = formatProfit(100, null);
    expect(result).toBe('+100.0 pips'); // Default unit
  });
});

describe('getFormattedSymbol', () => {
  it('returns formatted_symbol from metadata when available', () => {
    const metadata = { formatted_symbol: 'EUR/USD' };
    const result = getFormattedSymbol('EURUSD', metadata);
    expect(result).toBe('EUR/USD');
  });

  it('formats 6-char symbol with slash when no metadata', () => {
    const result = getFormattedSymbol('EURUSD', null);
    expect(result).toBe('EUR/USD');
  });

  it('returns symbol unchanged for non-6-char symbols', () => {
    const result = getFormattedSymbol('AAPL', null);
    expect(result).toBe('AAPL');
  });

  it('returns N/A for null symbol', () => {
    const result = getFormattedSymbol(null, null);
    expect(result).toBe('N/A');
  });

  it('returns N/A for undefined symbol', () => {
    const result = getFormattedSymbol(undefined, null);
    expect(result).toBe('N/A');
  });

  it('returns N/A for empty symbol', () => {
    const result = getFormattedSymbol('', null);
    expect(result).toBe('N/A');
  });

  it('uses metadata formatted_symbol over fallback logic', () => {
    const metadata = { formatted_symbol: 'BTC/USD' };
    const result = getFormattedSymbol('BTCUSD', metadata);
    expect(result).toBe('BTC/USD');
  });

  it('handles crypto symbols with dashes', () => {
    const metadata = { formatted_symbol: 'ETH/USD' };
    const result = getFormattedSymbol('ETH-USD', metadata);
    expect(result).toBe('ETH/USD');
  });
});

describe('getAssetTypeLabel', () => {
  it('returns label for forex', () => {
    const metadata = { asset_type: 'forex' };
    const result = getAssetTypeLabel(metadata);
    expect(result).toBe('Forex Currency Pair');
  });

  it('returns label for crypto', () => {
    const metadata = { asset_type: 'crypto' };
    const result = getAssetTypeLabel(metadata);
    expect(result).toBe('Cryptocurrency');
  });

  it('returns label for stock', () => {
    const metadata = { asset_type: 'stock' };
    const result = getAssetTypeLabel(metadata);
    expect(result).toBe('Stock');
  });

  it('returns label for commodity', () => {
    const metadata = { asset_type: 'commodity' };
    const result = getAssetTypeLabel(metadata);
    expect(result).toBe('Commodity');
  });

  it('returns label for index', () => {
    const metadata = { asset_type: 'index' };
    const result = getAssetTypeLabel(metadata);
    expect(result).toBe('Index');
  });

  it('returns default for unknown type', () => {
    const metadata = { asset_type: 'unknown' };
    const result = getAssetTypeLabel(metadata);
    expect(result).toBe('Financial Asset');
  });

  it('returns default when metadata is missing', () => {
    const result = getAssetTypeLabel(null);
    expect(result).toBe('Financial Asset');
  });

  it('returns default when asset_type is missing', () => {
    const result = getAssetTypeLabel({});
    expect(result).toBe('Financial Asset');
  });
});

describe('inferAssetMetadata', () => {
  // Forex Detection Tests
  it('infers forex for EURUSD', () => {
    const result = inferAssetMetadata('EURUSD');
    expect(result.asset_type).toBe('forex');
    expect(result.price_precision).toBe(5);
    expect(result.profit_unit).toBe('pips');
    expect(result.profit_multiplier).toBe(10000);
    expect(result.formatted_symbol).toBe('EUR/USD');
  });

  it('infers forex for GBPJPY', () => {
    const result = inferAssetMetadata('GBPJPY');
    expect(result.asset_type).toBe('forex');
    expect(result.formatted_symbol).toBe('GBP/JPY');
  });

  it('infers forex for lowercase eurusd', () => {
    const result = inferAssetMetadata('eurusd');
    expect(result.asset_type).toBe('forex');
    expect(result.formatted_symbol).toBe('EUR/USD');
  });

  it('does not infer forex for non-currency 6-char symbols', () => {
    const result = inferAssetMetadata('ABCDEF');
    expect(result.asset_type).not.toBe('forex');
  });

  // Crypto Detection Tests
  it('infers crypto for BTCUSD', () => {
    const result = inferAssetMetadata('BTCUSD');
    expect(result.asset_type).toBe('crypto');
    expect(result.price_precision).toBe(2);
    expect(result.profit_unit).toBe('dollars');
    expect(result.profit_multiplier).toBe(1);
  });

  it('infers crypto for ETHUSD', () => {
    const result = inferAssetMetadata('ETHUSD');
    expect(result.asset_type).toBe('crypto');
    expect(result.formatted_symbol).toBe('ETH/USD');
  });

  it('infers crypto for BTC with dash', () => {
    const result = inferAssetMetadata('BTC-USD');
    expect(result.asset_type).toBe('crypto');
  });

  it('infers crypto for SOL', () => {
    const result = inferAssetMetadata('SOLUSD');
    expect(result.asset_type).toBe('crypto');
  });

  it('infers crypto for ADA', () => {
    const result = inferAssetMetadata('ADAUSD');
    expect(result.asset_type).toBe('crypto');
  });

  it('infers crypto for DOGE', () => {
    const result = inferAssetMetadata('DOGEUSD');
    expect(result.asset_type).toBe('crypto');
  });

  // Stock Detection Tests
  it('infers stock for AAPL', () => {
    const result = inferAssetMetadata('AAPL');
    expect(result.asset_type).toBe('stock');
    expect(result.price_precision).toBe(2);
    expect(result.profit_unit).toBe('points');
    expect(result.profit_multiplier).toBe(1);
    expect(result.formatted_symbol).toBe('AAPL');
  });

  it('infers stock for TSLA', () => {
    const result = inferAssetMetadata('TSLA');
    expect(result.asset_type).toBe('stock');
  });

  it('infers stock for lowercase aapl', () => {
    const result = inferAssetMetadata('aapl');
    expect(result.asset_type).toBe('stock');
    expect(result.formatted_symbol).toBe('AAPL');
  });

  it('defaults to stock for unknown symbols', () => {
    const result = inferAssetMetadata('UNKNOWN');
    expect(result.asset_type).toBe('stock');
  });

  // Edge Cases
  it('returns default for null symbol', () => {
    const result = inferAssetMetadata(null);
    expect(result.asset_type).toBe('forex'); // Default
  });

  it('returns default for undefined symbol', () => {
    const result = inferAssetMetadata(undefined);
    expect(result.asset_type).toBe('forex'); // Default
  });

  it('returns default for empty symbol', () => {
    const result = inferAssetMetadata('');
    expect(result.asset_type).toBe('forex'); // Default
  });

  it('handles mixed case symbols', () => {
    const result = inferAssetMetadata('EurUsd');
    expect(result.asset_type).toBe('forex');
  });

  it('preserves original symbol in result', () => {
    const result = inferAssetMetadata('EURUSD');
    expect(result.symbol).toBe('EURUSD');
  });

  // Comprehensive Asset Type Coverage
  it('correctly identifies all forex currencies', () => {
    const forexPairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF'];
    forexPairs.forEach(pair => {
      const result = inferAssetMetadata(pair);
      expect(result.asset_type).toBe('forex');
    });
  });

  it('correctly identifies all crypto symbols', () => {
    const cryptoSymbols = ['BTCUSD', 'ETHUSD', 'XRPUSD', 'SOLUSD', 'ADAUSD', 'DOGEUSD'];
    cryptoSymbols.forEach(symbol => {
      const result = inferAssetMetadata(symbol);
      expect(result.asset_type).toBe('crypto');
    });
  });

  it('sets correct symbol field from input', () => {
    const result = inferAssetMetadata('BTCUSD');
    expect(result.symbol).toBe('BTCUSD');
  });
});

describe('Integration Tests', () => {
  it('formats complete forex display correctly', () => {
    const metadata = inferAssetMetadata('EURUSD');
    const price = formatPrice(1.08543, metadata);
    const profit = formatProfit(123.4, metadata);
    const symbol = getFormattedSymbol('EURUSD', metadata);
    const type = getAssetTypeLabel(metadata);

    expect(price).toBe('1.08543');
    expect(profit).toBe('+123.4 pips');
    expect(symbol).toBe('EUR/USD');
    expect(type).toBe('Forex Currency Pair');
  });

  it('formats complete crypto display correctly', () => {
    const metadata = inferAssetMetadata('BTCUSD');
    const price = formatPrice(50123.12, metadata);
    const profit = formatProfit(1234.5, metadata);
    const symbol = getFormattedSymbol('BTCUSD', metadata);
    const type = getAssetTypeLabel(metadata);

    expect(price).toBe('50123.12');
    expect(profit).toBe('+1234.5 dollars');
    expect(symbol).toBe('BTC/USD');
    expect(type).toBe('Cryptocurrency');
  });

  it('formats complete stock display correctly', () => {
    const metadata = inferAssetMetadata('AAPL');
    const price = formatPrice(150.5, metadata);
    const profit = formatProfit(567.8, metadata);
    const symbol = getFormattedSymbol('AAPL', metadata);
    const type = getAssetTypeLabel(metadata);

    expect(price).toBe('150.50');
    expect(profit).toBe('+567.8 points');
    expect(symbol).toBe('AAPL');
    expect(type).toBe('Stock');
  });

  it('handles missing metadata gracefully', () => {
    const price = formatPrice(1.08543, null);
    const profit = formatProfit(123.4, null);
    const symbol = getFormattedSymbol('EURUSD', null);
    const type = getAssetTypeLabel(null);

    // Should use defaults
    expect(price).toBe('1.08543');
    expect(profit).toBe('+123.4 pips');
    expect(symbol).toBe('EUR/USD');
    expect(type).toBe('Financial Asset');
  });
});
