/**
 * Asset formatting utilities for dynamic asset display
 */

// Default metadata for when API doesn't provide it
const DEFAULT_METADATA = {
  asset_type: 'forex',
  price_precision: 5,
  profit_unit: 'pips',
  profit_multiplier: 10000,
};

/**
 * Format price with appropriate precision
 */
export function formatPrice(price, metadata) {
  if (price === null || price === undefined) return 'N/A';
  const precision = metadata?.price_precision ?? DEFAULT_METADATA.price_precision;
  return Number(price).toFixed(precision);
}

/**
 * Get the profit unit label (pips, points, dollars, etc.)
 */
export function getProfitUnitLabel(metadata) {
  return metadata?.profit_unit ?? DEFAULT_METADATA.profit_unit;
}

/**
 * Format a profit value with its unit
 */
export function formatProfit(value, metadata) {
  if (value === null || value === undefined) return 'N/A';
  const unit = getProfitUnitLabel(metadata);
  return `${value >= 0 ? '+' : ''}${Number(value).toFixed(1)} ${unit}`;
}

/**
 * Get formatted symbol for display (EUR/USD, BTC/USD, AAPL)
 */
export function getFormattedSymbol(symbol, metadata) {
  if (metadata?.formatted_symbol) return metadata.formatted_symbol;
  if (!symbol) return 'N/A';
  // Fallback: try to format 6-char pairs
  if (symbol.length === 6) {
    return `${symbol.slice(0, 3)}/${symbol.slice(3, 6)}`;
  }
  return symbol;
}

/**
 * Get asset type display name
 */
export function getAssetTypeLabel(metadata) {
  const types = {
    forex: 'Forex Currency Pair',
    crypto: 'Cryptocurrency',
    stock: 'Stock',
    commodity: 'Commodity',
    index: 'Index',
  };
  return types[metadata?.asset_type] || 'Financial Asset';
}

/**
 * Infer asset metadata from symbol (fallback when API doesn't provide)
 */
export function inferAssetMetadata(symbol) {
  if (!symbol) return DEFAULT_METADATA;

  const s = symbol.toUpperCase();
  const forexCurrencies = ['EUR', 'USD', 'GBP', 'JPY', 'CHF', 'AUD', 'NZD', 'CAD'];
  const cryptoSymbols = ['BTC', 'ETH', 'XRP', 'SOL', 'ADA', 'DOGE'];

  // Forex: 6 chars, both parts are currencies
  if (s.length === 6 && forexCurrencies.includes(s.slice(0, 3))) {
    return {
      symbol: s,
      asset_type: 'forex',
      price_precision: 5,
      profit_unit: 'pips',
      profit_multiplier: 10000,
      formatted_symbol: `${s.slice(0, 3)}/${s.slice(3, 6)}`,
    };
  }

  // Crypto
  if (cryptoSymbols.some(c => s.includes(c))) {
    return {
      symbol: s,
      asset_type: 'crypto',
      price_precision: 2,
      profit_unit: 'dollars',
      profit_multiplier: 1,
      formatted_symbol: s.replace('USD', '/USD'),
    };
  }

  // Default: stock
  return {
    symbol: s,
    asset_type: 'stock',
    price_precision: 2,
    profit_unit: 'points',
    profit_multiplier: 1,
    formatted_symbol: s,
  };
}
