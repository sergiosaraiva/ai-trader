#!/usr/bin/env python3
"""
Trading Simulation Script for Sentiment Integration Evaluation.

This script runs a backtesting simulation to evaluate the trading
performance of models with and without sentiment features.

Metrics include:
- Total return
- Sharpe ratio
- Maximum drawdown
- Win rate
- Profit factor

Usage:
    # Run simulation with saved models
    python scripts/trading_simulation.py \
        --baseline-model models/trained/EURUSD_short_term_baseline \
        --sentiment-model models/trained/EURUSD_short_term_sentiment \
        --pair EURUSD

    # Train and simulate
    python scripts/trading_simulation.py --pair EURUSD --train-first
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TradingSimulator:
    """Simple trading simulator for model evaluation."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_size: float = 0.02,  # 2% per trade
        stop_loss: float = 0.01,  # 1% stop loss
        take_profit: float = 0.02,  # 2% take profit
        transaction_cost: float = 0.0001,  # 1 pip
        min_confidence: float = 0.6,  # Minimum confidence to trade
    ):
        """
        Initialize trading simulator.

        Args:
            initial_capital: Starting capital
            position_size: Position size as fraction of capital
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
            transaction_cost: Transaction cost per trade
            min_confidence: Minimum prediction confidence to trade
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.transaction_cost = transaction_cost
        self.min_confidence = min_confidence

        self.reset()

    def reset(self):
        """Reset simulator state."""
        self.capital = self.initial_capital
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [self.initial_capital]
        self.positions: List[Dict] = []

    def run_simulation(
        self,
        predictions: List,
        prices: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run trading simulation.

        Args:
            predictions: List of Prediction objects from model
            prices: Array of actual prices (close prices)
            timestamps: Optional timestamps

        Returns:
            Dictionary with simulation results
        """
        self.reset()

        if len(predictions) != len(prices):
            raise ValueError("Predictions and prices must have same length")

        for i, (pred, price) in enumerate(zip(predictions, prices)):
            # Get prediction direction and confidence
            direction = pred.direction if hasattr(pred, 'direction') else None
            confidence = pred.confidence if hasattr(pred, 'confidence') else 0.5

            # Skip if confidence below threshold
            if confidence < self.min_confidence:
                self.equity_curve.append(self.capital)
                continue

            # Determine trade signal
            if direction == 'bullish':
                signal = 1  # Long
            elif direction == 'bearish':
                signal = -1  # Short
            else:
                signal = 0  # Hold

            # Execute trade if we have a signal
            if signal != 0:
                self._execute_trade(
                    signal=signal,
                    entry_price=price,
                    confidence=confidence,
                    timestamp=timestamps[i] if timestamps is not None else i,
                )

            # Check open positions
            if i < len(prices) - 1:
                self._check_positions(prices[i + 1])

            self.equity_curve.append(self.capital)

        # Close any remaining positions at final price
        self._close_all_positions(prices[-1])

        # Calculate performance metrics
        return self._calculate_metrics()

    def _execute_trade(
        self,
        signal: int,
        entry_price: float,
        confidence: float,
        timestamp: any,
    ):
        """Execute a trade."""
        # Calculate position value
        position_value = self.capital * self.position_size * confidence

        # Apply transaction cost
        cost = position_value * self.transaction_cost
        self.capital -= cost

        # Record position
        position = {
            'signal': signal,  # 1 = long, -1 = short
            'entry_price': entry_price,
            'position_value': position_value,
            'confidence': confidence,
            'timestamp': timestamp,
            'stop_loss': entry_price * (1 - signal * self.stop_loss),
            'take_profit': entry_price * (1 + signal * self.take_profit),
        }
        self.positions.append(position)

    def _check_positions(self, current_price: float):
        """Check and close positions at stop loss or take profit."""
        positions_to_close = []

        for i, pos in enumerate(self.positions):
            signal = pos['signal']
            entry = pos['entry_price']
            sl = pos['stop_loss']
            tp = pos['take_profit']

            # Check stop loss
            if (signal == 1 and current_price <= sl) or (signal == -1 and current_price >= sl):
                positions_to_close.append((i, current_price, 'stop_loss'))
            # Check take profit
            elif (signal == 1 and current_price >= tp) or (signal == -1 and current_price <= tp):
                positions_to_close.append((i, current_price, 'take_profit'))

        # Close positions (in reverse order to maintain indices)
        for idx, exit_price, exit_type in reversed(positions_to_close):
            self._close_position(idx, exit_price, exit_type)

    def _close_position(self, idx: int, exit_price: float, exit_type: str):
        """Close a position and record the trade."""
        pos = self.positions.pop(idx)

        # Calculate P&L
        signal = pos['signal']
        entry = pos['entry_price']
        pnl_pct = (exit_price - entry) / entry * signal
        pnl = pos['position_value'] * pnl_pct

        # Apply transaction cost
        cost = pos['position_value'] * self.transaction_cost
        pnl -= cost

        # Update capital
        self.capital += pos['position_value'] + pnl

        # Record trade
        trade = {
            'entry_price': entry,
            'exit_price': exit_price,
            'signal': signal,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_type': exit_type,
            'confidence': pos['confidence'],
        }
        self.trades.append(trade)

    def _close_all_positions(self, final_price: float):
        """Close all remaining positions."""
        while self.positions:
            self._close_position(0, final_price, 'end_of_simulation')

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trades:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0,
            }

        # Returns
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns)]

        total_return = (self.capital - self.initial_capital) / self.initial_capital

        # Sharpe ratio (annualized, assuming hourly data)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24)
        else:
            sharpe = 0.0

        # Maximum drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0

        # Win rate
        winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        win_rate = winning_trades / len(self.trades) if self.trades else 0.0

        # Profit factor
        gross_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average trade metrics
        avg_win = np.mean([t['pnl'] for t in self.trades if t['pnl'] > 0]) if winning_trades > 0 else 0.0
        avg_loss = np.mean([t['pnl'] for t in self.trades if t['pnl'] < 0]) if len(self.trades) - winning_trades > 0 else 0.0

        return {
            'total_return': float(total_return),
            'total_return_pct': float(total_return * 100),
            'final_capital': float(self.capital),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'max_drawdown_pct': float(max_drawdown * 100),
            'win_rate': float(win_rate),
            'win_rate_pct': float(win_rate * 100),
            'profit_factor': float(profit_factor) if profit_factor != float('inf') else 999.0,
            'total_trades': len(self.trades),
            'winning_trades': winning_trades,
            'losing_trades': len(self.trades) - winning_trades,
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'gross_profit': float(gross_profit),
            'gross_loss': float(gross_loss),
        }


def run_trading_comparison(
    pair: str,
    baseline_model_path: Optional[Path] = None,
    sentiment_model_path: Optional[Path] = None,
    train_first: bool = False,
    epochs: int = 30,
    max_rows: Optional[int] = None,
    resample: str = '1h',
    save_dir: Path = Path('results/trading_simulation'),
) -> Dict:
    """
    Run trading comparison between baseline and sentiment models.

    Args:
        pair: Trading pair
        baseline_model_path: Path to baseline model
        sentiment_model_path: Path to sentiment model
        train_first: Whether to train models first
        epochs: Training epochs (if training)
        max_rows: Maximum data rows
        resample: Resample timeframe
        save_dir: Directory to save results

    Returns:
        Dictionary with comparison results
    """
    from scripts.train_with_sentiment import (
        load_price_data, prepare_training_data, train_model
    )

    print("\n" + "=" * 70)
    print("TRADING SIMULATION: SENTIMENT VS BASELINE")
    print("=" * 70)
    print(f"Pair: {pair}")
    print(f"Timeframe: {resample}")
    print("=" * 70)

    # Prepare test data
    print("\nPreparing test data...")
    data_sentiment = prepare_training_data(
        pair=pair,
        include_sentiment=True,
        max_rows=max_rows,
        resample_timeframe=resample,
    )

    data_baseline = prepare_training_data(
        pair=pair,
        include_sentiment=False,
        max_rows=max_rows,
        resample_timeframe=resample,
    )

    # Train or load models
    if train_first:
        print("\nTraining baseline model...")
        baseline_model, _, _ = train_model(
            model_type='short_term',
            data=data_baseline,
            epochs=epochs,
        )

        print("\nTraining sentiment model...")
        sentiment_model, _, _ = train_model(
            model_type='short_term',
            data=data_sentiment,
            epochs=epochs,
        )
    else:
        # Load pre-trained models
        from src.models.technical import ShortTermModel

        if baseline_model_path and baseline_model_path.exists():
            baseline_model = ShortTermModel.load(baseline_model_path / 'model')
        else:
            print("No baseline model found, training...")
            baseline_model, _, _ = train_model('short_term', data_baseline, epochs=epochs)

        if sentiment_model_path and sentiment_model_path.exists():
            sentiment_model = ShortTermModel.load(sentiment_model_path / 'model')
        else:
            print("No sentiment model found, training...")
            sentiment_model, _, _ = train_model('short_term', data_sentiment, epochs=epochs)

    # Get predictions on test data
    print("\nGenerating predictions...")

    print("  Baseline predictions...")
    baseline_preds = baseline_model.predict_batch(data_baseline['X_test'].astype(np.float32))

    print("  Sentiment predictions...")
    sentiment_preds = sentiment_model.predict_batch(data_sentiment['X_test'].astype(np.float32))

    # Run simulations
    print("\nRunning trading simulations...")

    simulator = TradingSimulator(
        initial_capital=100000.0,
        position_size=0.02,
        stop_loss=0.01,
        take_profit=0.02,
        min_confidence=0.0,  # No confidence filter - trade on all signals
    )

    # Baseline simulation
    print("  Baseline simulation...")
    baseline_metrics = simulator.run_simulation(
        predictions=baseline_preds,
        prices=data_baseline['y_test'],
    )

    # Sentiment simulation
    print("  Sentiment simulation...")
    simulator.reset()
    sentiment_metrics = simulator.run_simulation(
        predictions=sentiment_preds,
        prices=data_sentiment['y_test'],
    )

    # Compare results
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)

    comparison = compare_results(baseline_metrics, sentiment_metrics)

    # Print results
    print_results(baseline_metrics, sentiment_metrics, comparison)

    # Save results
    save_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results = {
        'config': {
            'pair': pair,
            'resample': resample,
            'max_rows': max_rows,
            'timestamp': timestamp,
        },
        'baseline': baseline_metrics,
        'sentiment': sentiment_metrics,
        'comparison': comparison,
    }

    results_file = save_dir / f"trading_sim_{pair}_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Generate recommendation
    generate_trading_recommendation(comparison)

    return results


def compare_results(baseline: Dict, sentiment: Dict) -> Dict:
    """Compare trading results between models."""
    comparison = {}

    metrics_to_compare = [
        ('total_return_pct', True),  # Higher is better
        ('sharpe_ratio', True),
        ('max_drawdown_pct', False),  # Lower is better
        ('win_rate_pct', True),
        ('profit_factor', True),
        ('total_trades', None),  # No preference
    ]

    for metric, higher_is_better in metrics_to_compare:
        baseline_val = baseline.get(metric, 0)
        sentiment_val = sentiment.get(metric, 0)

        if higher_is_better is not None:
            if higher_is_better:
                is_better = sentiment_val > baseline_val
                change = (sentiment_val - baseline_val)
            else:
                is_better = sentiment_val < baseline_val
                change = (baseline_val - sentiment_val)
        else:
            is_better = None
            change = sentiment_val - baseline_val

        comparison[metric] = {
            'baseline': baseline_val,
            'sentiment': sentiment_val,
            'change': change,
            'is_better': is_better,
        }

    return comparison


def print_results(baseline: Dict, sentiment: Dict, comparison: Dict):
    """Print formatted results."""
    print(f"\n{'Metric':<25} {'Baseline':>15} {'Sentiment':>15} {'Change':>15}")
    print("-" * 70)

    for metric, comp in comparison.items():
        baseline_val = comp['baseline']
        sentiment_val = comp['sentiment']
        change = comp['change']
        is_better = comp['is_better']

        # Format values
        if 'pct' in metric or 'rate' in metric:
            b_str = f"{baseline_val:.2f}%"
            s_str = f"{sentiment_val:.2f}%"
            c_str = f"{change:+.2f}%"
        elif 'ratio' in metric or 'factor' in metric:
            b_str = f"{baseline_val:.3f}"
            s_str = f"{sentiment_val:.3f}"
            c_str = f"{change:+.3f}"
        else:
            b_str = f"{baseline_val:.0f}"
            s_str = f"{sentiment_val:.0f}"
            c_str = f"{change:+.0f}"

        # Add indicator
        if is_better is True:
            indicator = " ✓"
        elif is_better is False:
            indicator = " ✗"
        else:
            indicator = ""

        print(f"{metric:<25} {b_str:>15} {s_str:>15} {c_str:>15}{indicator}")


def generate_trading_recommendation(comparison: Dict):
    """Generate trading recommendation based on simulation results."""
    print("\n" + "=" * 70)
    print("TRADING RECOMMENDATION")
    print("=" * 70)

    # Key metrics for trading
    return_better = comparison.get('total_return_pct', {}).get('is_better', False)
    sharpe_better = comparison.get('sharpe_ratio', {}).get('is_better', False)
    drawdown_better = comparison.get('max_drawdown_pct', {}).get('is_better', False)
    winrate_better = comparison.get('win_rate_pct', {}).get('is_better', False)
    pf_better = comparison.get('profit_factor', {}).get('is_better', False)

    improvements = sum([return_better, sharpe_better, drawdown_better, winrate_better, pf_better])

    # Check if return is positive
    sentiment_return = comparison.get('total_return_pct', {}).get('sentiment', 0)
    baseline_return = comparison.get('total_return_pct', {}).get('baseline', 0)

    print(f"\nKey improvements: {improvements}/5")
    print(f"Sentiment return: {sentiment_return:+.2f}%")
    print(f"Baseline return: {baseline_return:+.2f}%")

    print("\n" + "-" * 70)

    if improvements >= 4 and sentiment_return > 0:
        print("✓✓ STRONG RECOMMENDATION: Sentiment integration shows clear trading benefits")
        print("   Consider merging to main branch.")
    elif improvements >= 3 or (return_better and sharpe_better):
        print("✓ RECOMMENDATION: Sentiment shows promise for trading")
        print("   Continue testing with live paper trading.")
    elif improvements >= 2 or sentiment_return > baseline_return:
        print("? NEUTRAL: Mixed trading results")
        print("   More backtesting on different time periods recommended.")
    else:
        print("✗ NOT RECOMMENDED: Sentiment did not improve trading performance")
        print("   Keep on feature branch for further development.")

    print("-" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Trading simulation for sentiment evaluation")

    parser.add_argument('--pair', type=str, default='EURUSD', help='Trading pair')
    parser.add_argument('--baseline-model', type=str, default=None, help='Path to baseline model')
    parser.add_argument('--sentiment-model', type=str, default=None, help='Path to sentiment model')
    parser.add_argument('--train-first', action='store_true', help='Train models before simulation')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs if training')
    parser.add_argument('--max-rows', type=int, default=None, help='Maximum data rows')
    parser.add_argument('--resample', type=str, default='1h', help='Resample timeframe')
    parser.add_argument('--save-dir', type=str, default='results/trading_simulation', help='Results directory')

    args = parser.parse_args()

    run_trading_comparison(
        pair=args.pair,
        baseline_model_path=Path(args.baseline_model) if args.baseline_model else None,
        sentiment_model_path=Path(args.sentiment_model) if args.sentiment_model else None,
        train_first=args.train_first,
        epochs=args.epochs,
        max_rows=args.max_rows,
        resample=args.resample,
        save_dir=Path(args.save_dir),
    )


if __name__ == '__main__':
    main()
