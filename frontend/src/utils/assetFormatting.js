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
 * Available markets configuration
 * This is the source of truth for all market types
 */
export const AVAILABLE_MARKETS = [
  { id: 'forex', label: 'Forex', enabled: true },
  { id: 'crypto', label: 'Crypto', enabled: false },
  { id: 'stock', label: 'Stocks', enabled: false },
  { id: 'commodity', label: 'Commodities', enabled: false },
  { id: 'index', label: 'Indices', enabled: false },
];

/**
 * Available assets by market type
 * This is the source of truth for all tradeable assets
 */
export const AVAILABLE_ASSETS = {
  forex: [
    { symbol: 'EURUSD', label: 'EUR/USD', enabled: true },
    { symbol: 'GBPUSD', label: 'GBP/USD', enabled: false },
    { symbol: 'USDJPY', label: 'USD/JPY', enabled: false },
    { symbol: 'AUDUSD', label: 'AUD/USD', enabled: false },
    { symbol: 'USDCAD', label: 'USD/CAD', enabled: false },
    { symbol: 'USDCHF', label: 'USD/CHF', enabled: false },
  ],
  crypto: [
    { symbol: 'BTCUSD', label: 'BTC/USD', enabled: false },
    { symbol: 'ETHUSD', label: 'ETH/USD', enabled: false },
  ],
  stock: [
    { symbol: 'AAPL', label: 'Apple', enabled: false },
    { symbol: 'TSLA', label: 'Tesla', enabled: false },
  ],
  commodity: [
    { symbol: 'XAUUSD', label: 'Gold', enabled: false },
    { symbol: 'XAGUSD', label: 'Silver', enabled: false },
  ],
  index: [
    { symbol: 'SPX', label: 'S&P 500', enabled: false },
    { symbol: 'NDX', label: 'Nasdaq 100', enabled: false },
  ],
};

/**
 * Get the default market (first enabled market)
 */
export function getDefaultMarket() {
  const enabled = AVAILABLE_MARKETS.find(m => m.enabled);
  return enabled?.id || AVAILABLE_MARKETS[0]?.id;
}

/**
 * Get the default asset for a given market (first enabled asset)
 */
export function getDefaultAsset(marketId) {
  const assets = AVAILABLE_ASSETS[marketId] || [];
  const enabled = assets.find(a => a.enabled);
  return enabled?.symbol || assets[0]?.symbol;
}

/**
 * Get enabled markets
 */
export function getEnabledMarkets() {
  return AVAILABLE_MARKETS.filter(m => m.enabled);
}

/**
 * Get enabled assets for a market
 */
export function getEnabledAssets(marketId) {
  const assets = AVAILABLE_ASSETS[marketId] || [];
  return assets.filter(a => a.enabled);
}

/**
 * Check if market is currently open based on asset type
 */
export function isMarketOpen(assetType) {
  const now = new Date();
  const day = now.getUTCDay(); // 0 = Sunday, 6 = Saturday
  const hour = now.getUTCHours();

  switch (assetType) {
    case 'forex':
      // Forex: Sunday 22:00 UTC to Friday 22:00 UTC
      if (day === 6) return false; // Saturday closed
      if (day === 0 && hour < 22) return false; // Sunday before open
      if (day === 5 && hour >= 22) return false; // Friday after close
      return true;

    case 'crypto':
      // Crypto: 24/7
      return true;

    case 'stock':
    case 'index':
      // US Markets: Mon-Fri 14:30-21:00 UTC (9:30am-4pm ET)
      if (day === 0 || day === 6) return false; // Weekend
      if (hour < 14 || hour >= 21) return false; // Outside hours
      if (hour === 14 && now.getUTCMinutes() < 30) return false; // Before 14:30
      return true;

    case 'commodity':
      // Similar to forex for most commodities
      if (day === 6) return false;
      if (day === 0 && hour < 22) return false;
      if (day === 5 && hour >= 22) return false;
      return true;

    default:
      return true;
  }

}

/**
 * Get market status label
 */
export function getMarketStatusLabel(assetType, isOpen) {
  if (isOpen) return 'Market Open';

  const labels = {
    forex: 'Forex Closed',
    crypto: 'Market Open', // Always open
    stock: 'Market Closed',
    commodity: 'Market Closed',
    index: 'Market Closed',
  };
  return labels[assetType] || 'Market Closed';
}

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
 * Get a dynamic description for the dashboard based on asset type
 * @param {string} symbol - The trading symbol (e.g., "EURUSD", "BTCUSD")
 * @param {object} metadata - Asset metadata from API or inferred
 * @returns {string} A dynamic description sentence
 */
export function getDashboardDescription(symbol, metadata) {
  const meta = metadata || inferAssetMetadata(symbol);
  const formattedSymbol = getFormattedSymbol(symbol, meta);
  const marketLabel = AVAILABLE_MARKETS.find(m => m.id === meta?.asset_type)?.label || 'Financial';

  const descriptions = {
    forex: `AI agent analyzing ${marketLabel} ${formattedSymbol} exchange rate patterns across multiple timeframes to generate high-confidence trading signals.`,
    crypto: `AI agent analyzing ${marketLabel} ${formattedSymbol} price dynamics across multiple timeframes to generate high-confidence trading signals.`,
    stock: `AI agent analyzing ${marketLabel} ${formattedSymbol} market patterns across multiple timeframes to generate high-confidence trading signals.`,
    commodity: `AI agent analyzing ${marketLabel} ${formattedSymbol} price movements across multiple timeframes to generate high-confidence trading signals.`,
    index: `AI agent analyzing ${marketLabel} ${formattedSymbol} market trends across multiple timeframes to generate high-confidence trading signals.`,
  };

  return descriptions[meta?.asset_type] ||
    `AI agent analyzing ${marketLabel} ${formattedSymbol} across multiple timeframes to generate high-confidence trading signals.`;
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
