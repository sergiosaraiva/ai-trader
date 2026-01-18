import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AboutSection } from './AboutSection';

describe('AboutSection', () => {
  describe('Basic Rendering', () => {
    it('renders component with default props', () => {
      render(<AboutSection />);
      expect(screen.getByRole('region', { name: 'About This System' })).toBeInTheDocument();
      expect(screen.getByText('AI Trading System')).toBeInTheDocument();
    });

    it('renders trading pair with default value', () => {
      render(<AboutSection />);
      expect(screen.getByText('EUR/USD')).toBeInTheDocument();
    });

    it('renders custom trading pair', () => {
      render(<AboutSection tradingPair="GBPUSD" />);
      expect(screen.getByText('GBP/USD')).toBeInTheDocument();
    });
  });

  describe('VIX Display', () => {
    it('renders VIX as N/A when value is null', () => {
      render(<AboutSection vixValue={null} />);
      const vixElements = screen.getAllByText('N/A');
      // Should have at least one N/A (for VIX value)
      expect(vixElements.length).toBeGreaterThan(0);
    });

    it('renders VIX as N/A when value is undefined', () => {
      render(<AboutSection vixValue={undefined} />);
      const vixElements = screen.getAllByText('N/A');
      expect(vixElements.length).toBeGreaterThan(0);
    });

    it('renders low volatility VIX value with green color', () => {
      render(<AboutSection vixValue={12.5} />);
      expect(screen.getByText('12.50')).toBeInTheDocument();
      expect(screen.getByText('Low Vol')).toBeInTheDocument();
      expect(screen.getByText('Low Vol')).toHaveClass('text-green-400');
    });

    it('renders normal volatility VIX value with blue color', () => {
      render(<AboutSection vixValue={17.8} />);
      expect(screen.getByText('17.80')).toBeInTheDocument();
      expect(screen.getByText('Normal')).toBeInTheDocument();
      expect(screen.getByText('Normal')).toHaveClass('text-blue-400');
    });

    it('renders elevated volatility VIX value with yellow color', () => {
      render(<AboutSection vixValue={25.3} />);
      expect(screen.getByText('25.30')).toBeInTheDocument();
      expect(screen.getByText('Elevated')).toBeInTheDocument();
      expect(screen.getByText('Elevated')).toHaveClass('text-yellow-400');
    });

    it('renders high volatility VIX value with red color', () => {
      render(<AboutSection vixValue={35.7} />);
      expect(screen.getByText('35.70')).toBeInTheDocument();
      expect(screen.getByText('High Vol')).toBeInTheDocument();
      expect(screen.getByText('High Vol')).toHaveClass('text-red-400');
    });

    it('renders VIX boundary value at 15 as normal (not low)', () => {
      render(<AboutSection vixValue={15.0} />);
      expect(screen.getByText('Normal')).toBeInTheDocument();
      expect(screen.getByText('Normal')).toHaveClass('text-blue-400');
    });

    it('renders VIX boundary value at 20 as elevated (not normal)', () => {
      render(<AboutSection vixValue={20.0} />);
      expect(screen.getByText('Elevated')).toBeInTheDocument();
      expect(screen.getByText('Elevated')).toHaveClass('text-yellow-400');
    });

    it('renders VIX boundary value at 30 as high volatility (not elevated)', () => {
      render(<AboutSection vixValue={30.0} />);
      expect(screen.getByText('High Vol')).toBeInTheDocument();
      expect(screen.getByText('High Vol')).toHaveClass('text-red-400');
    });

    it('displays VIX usage note', () => {
      render(<AboutSection vixValue={18.5} />);
      expect(screen.getByText('Daily Model')).toBeInTheDocument();
    });
  });

  describe('Model Weights', () => {
    it('does not render model weights section when not provided', () => {
      render(<AboutSection />);
      expect(screen.queryByText('Model Weights')).not.toBeInTheDocument();
    });

    it('does not render model weights section when empty object', () => {
      render(<AboutSection modelWeights={{}} />);
      expect(screen.queryByText('Model Weights')).not.toBeInTheDocument();
    });

    it('renders model weights when provided', () => {
      const weights = { '1H': 0.6, '4H': 0.3, 'D': 0.1 };
      render(<AboutSection modelWeights={weights} />);

      expect(screen.getByText('Model Weights')).toBeInTheDocument();
      expect(screen.getByText('1H')).toBeInTheDocument();
      expect(screen.getByText('60%')).toBeInTheDocument();
      expect(screen.getByText('4H')).toBeInTheDocument();
      expect(screen.getByText('30%')).toBeInTheDocument();
      expect(screen.getByText('1D')).toBeInTheDocument(); // formatTimeframe("D") -> "1D"
      expect(screen.getByText('10%')).toBeInTheDocument();
    });

    it('formats daily timeframe as 1D', () => {
      const weights = { 'D': 0.5 };
      render(<AboutSection modelWeights={weights} />);

      expect(screen.getByText('1D')).toBeInTheDocument();
      expect(screen.queryByText('D')).not.toBeInTheDocument();
    });

    it('keeps other timeframes unchanged', () => {
      const weights = { '1H': 0.5, '4H': 0.5 };
      render(<AboutSection modelWeights={weights} />);

      expect(screen.getByText('1H')).toBeInTheDocument();
      expect(screen.getByText('4H')).toBeInTheDocument();
    });
  });

  describe('Trading Pair Formatting', () => {
    it('formats forex pairs correctly', () => {
      render(<AboutSection tradingPair="GBPJPY" />);
      expect(screen.getByText('GBP/JPY')).toBeInTheDocument();
    });

    it('handles short trading pair gracefully', () => {
      render(<AboutSection tradingPair="BTC" />);
      expect(screen.getByText('BTC')).toBeInTheDocument();
    });

    it('handles null trading pair', () => {
      render(<AboutSection tradingPair={null} />);
      const naElements = screen.getAllByText('N/A');
      expect(naElements.length).toBeGreaterThan(0);
    });

    it('handles undefined trading pair', () => {
      render(<AboutSection tradingPair={undefined} />);
      const naElements = screen.getAllByText('N/A');
      expect(naElements.length).toBeGreaterThan(0);
    });
  });

  describe('Asset Type Detection', () => {
    it('detects forex pairs correctly', () => {
      render(<AboutSection tradingPair="EURUSD" />);
      expect(screen.getByText(/Forex Currency Pair/)).toBeInTheDocument();
    });

    it('detects cryptocurrency', () => {
      render(<AboutSection tradingPair="BTCUSD" />);
      expect(screen.getByText(/Cryptocurrency/)).toBeInTheDocument();
    });

    it('handles unknown asset type', () => {
      render(<AboutSection tradingPair="XYZABC" />);
      // Unknown pairs default to "Stock" in inferAssetMetadata
      expect(screen.getByText(/Stock/)).toBeInTheDocument();
    });

    it('handles null trading pair for asset type', () => {
      render(<AboutSection tradingPair={null} />);
      // Null trading pair defaults to "Forex Currency Pair" (default EURUSD)
      expect(screen.getByText(/Forex Currency Pair/)).toBeInTheDocument();
    });
  });

  describe('Key Features', () => {
    it('renders all three key features', () => {
      render(<AboutSection />);

      expect(screen.getByText('Multi-Timeframe AI')).toBeInTheDocument();
      expect(screen.getByText('Walk-Forward Validated')).toBeInTheDocument();
      expect(screen.getByText('Risk-Optimized')).toBeInTheDocument();
    });

    it('renders feature descriptions', () => {
      render(<AboutSection />);

      expect(screen.getByText(/Ensemble of 3 XGBoost models/)).toBeInTheDocument();
      expect(screen.getByText(/Backtested on 7 rolling time periods/)).toBeInTheDocument();
      expect(screen.getByText(/70% confidence threshold/)).toBeInTheDocument();
    });
  });

  describe('Data Sources', () => {
    it('renders all data sources', () => {
      render(<AboutSection />);

      expect(screen.getByText('Data Sources')).toBeInTheDocument();
      expect(screen.getByText('Price Data')).toBeInTheDocument();
      expect(screen.getByText('VIX Index')).toBeInTheDocument();
      expect(screen.getByText('EPU Index')).toBeInTheDocument();
    });

    it('renders data source details', () => {
      render(<AboutSection />);

      // Check that source information is rendered
      expect(screen.getByText(/MetaTrader 5/)).toBeInTheDocument();
      const fredApiElements = screen.getAllByText('FRED API');
      expect(fredApiElements.length).toBe(2);
    });
  });

  describe('Performance Metrics', () => {
    it('renders performance highlights', () => {
      render(<AboutSection />);

      expect(screen.getByText('Backtested Performance')).toBeInTheDocument();
      expect(screen.getByText('62% Win')).toBeInTheDocument();
      expect(screen.getByText('2.69 PF')).toBeInTheDocument();
      expect(screen.getByText('7.67 Sharpe')).toBeInTheDocument();
    });
  });

  describe('Complete Integration', () => {
    it('renders all sections with full props', () => {
      const weights = { '1H': 0.6, '4H': 0.3, 'D': 0.1 };
      render(
        <AboutSection
          tradingPair="EURUSD"
          modelWeights={weights}
          vixValue={18.5}
        />
      );

      // Check all major sections are present
      expect(screen.getByText('AI Trading System')).toBeInTheDocument();
      expect(screen.getByText('EUR/USD')).toBeInTheDocument();
      expect(screen.getByText('Model Weights')).toBeInTheDocument();
      expect(screen.getByText('18.50')).toBeInTheDocument();
      expect(screen.getByText('Normal')).toBeInTheDocument();
      expect(screen.getByText('Data Sources')).toBeInTheDocument();
      expect(screen.getByText('62% Win')).toBeInTheDocument();
    });
  });
});
