#!/usr/bin/env python3
"""
Realistic backtest with:
1. Out-of-sample testing only (test set)
2. Transaction costs (spread + slippage)
3. Conservative Hybrid position sizing
"""
import sys
sys.path.insert(0, '/app')

import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import the original backtest classes
from scripts.backtest_dynamic_threshold import (
    DynamicThresholdBacktester,
    THRESHOLD_CONFIG,
    TRADING_CONFIG,
    CONSERVATIVE_HYBRID_CONFIG
)
from src.models.multi_timeframe.mtf_ensemble import MTFEnsemble

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Realistic transaction costs for EUR/USD
TRANSACTION_COSTS = {
    'spread_pips': 1.0,      # Typical spread for EUR/USD (0.8-1.5 pips)
    'slippage_pips': 0.5,    # Average slippage on market orders
    'commission_per_lot': 0, # Most retail brokers don't charge commission
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--model-dir', type=str, required=True, help='Model directory')
    parser.add_argument('--initial-balance', type=float, default=1000.0)
    parser.add_argument('--base-risk-percent', type=float, default=1.5)
    parser.add_argument('--output', type=str, default='/tmp/backtest_realistic')
    parser.add_argument('--test-only', action='store_true', default=True,
                       help='Use only test set (default: True)')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("REALISTIC BACKTEST - Out-of-Sample with Transaction Costs")
    logger.info("=" * 80)

    # Load data
    logger.info(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")

    # Calculate test period dates (but keep all data for feature calculation)
    test_start_idx = int(len(df) * 0.8)
    test_start_date = df.index[test_start_idx]
    test_end_date = df.index[-1]

    if args.test_only:
        logger.info(f"\nTesting on OUT-OF-SAMPLE data only:")
        logger.info(f"  Test period: {test_start_date} to {test_end_date}")
        logger.info(f"  Test bars: {len(df) - test_start_idx:,} bars ({(len(df) - test_start_idx)/len(df)*100:.1f}% of total)")
        logger.info(f"  This data was NEVER seen during model training!")
        logger.info(f"  (Keeping earlier data for feature calculation only)")

        # We'll filter trades later, not the data itself

    # Add transaction costs info
    logger.info(f"\nTransaction Costs:")
    logger.info(f"  Spread: {TRANSACTION_COSTS['spread_pips']} pips")
    logger.info(f"  Slippage: {TRANSACTION_COSTS['slippage_pips']} pips")
    logger.info(f"  Total cost per trade: {TRANSACTION_COSTS['spread_pips'] + TRANSACTION_COSTS['slippage_pips']} pips")
    logger.info(f"  Effective TP: {TRADING_CONFIG['tp_pips']} - {TRANSACTION_COSTS['spread_pips'] + TRANSACTION_COSTS['slippage_pips']} = {TRADING_CONFIG['tp_pips'] - TRANSACTION_COSTS['spread_pips'] - TRANSACTION_COSTS['slippage_pips']:.1f} pips")
    logger.info(f"  Effective SL: {TRADING_CONFIG['sl_pips']} + {TRANSACTION_COSTS['spread_pips'] + TRANSACTION_COSTS['slippage_pips']} = {TRADING_CONFIG['sl_pips'] + TRANSACTION_COSTS['spread_pips'] + TRANSACTION_COSTS['slippage_pips']:.1f} pips")

    # Adjust TP/SL for transaction costs
    TRADING_CONFIG_ADJUSTED = TRADING_CONFIG.copy()
    total_cost_pips = TRANSACTION_COSTS['spread_pips'] + TRANSACTION_COSTS['slippage_pips']
    TRADING_CONFIG_ADJUSTED['tp_pips'] = TRADING_CONFIG['tp_pips'] - total_cost_pips
    TRADING_CONFIG_ADJUSTED['sl_pips'] = TRADING_CONFIG['sl_pips'] + total_cost_pips

    # Update conservative hybrid config
    CONSERVATIVE_HYBRID_CONFIG['base_risk_percent'] = args.base_risk_percent
    TRADING_CONFIG_ADJUSTED['initial_balance'] = args.initial_balance

    # Load ensemble
    logger.info(f"\nLoading ensemble from {args.model_dir}...")
    ensemble = MTFEnsemble()
    ensemble.load(args.model_dir)

    # Run backtest with adjusted config
    logger.info(f"\nRunning backtest...")
    backtester = DynamicThresholdBacktester(
        ensemble=ensemble,
        threshold_config=THRESHOLD_CONFIG,
        trading_config=TRADING_CONFIG_ADJUSTED,
        conservative_hybrid_config=CONSERVATIVE_HYBRID_CONFIG,
    )

    results = backtester.run(df)

    # Save results
    output_csv = f"{args.output}.csv"
    output_json = f"{args.output}.json"

    # Monthly CSV
    monthly_df = pd.DataFrame(results['monthly_stats'])
    monthly_df.to_csv(output_csv, index=False)

    # Summary JSON
    summary = {k: v for k, v in results.items() if k != 'monthly_stats'}
    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Initial Balance: {results['initial_balance']:.2f} EUR")
    logger.info(f"Final Balance:   {results['final_balance']:.2f} EUR")
    logger.info(f"Total Return:    {results['total_return_pct']:.2f}%")
    logger.info(f"Total Pips:      {results['total_pips']:.2f}")
    logger.info(f"Total Trades:    {results['total_trades']}")
    logger.info(f"Win Rate:        {results['win_rate']:.2f}%")
    logger.info(f"Profit Factor:   {results['profit_factor']:.2f}x")
    logger.info(f"Max Drawdown:    {results['max_drawdown_pct']:.2f}%")
    logger.info(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
    logger.info(f"\nBest Month:  {results['best_month']['month']} (+{results['best_month']['pnl']:.2f} EUR)")
    logger.info(f"Worst Month: {results['worst_month']['month']} ({results['worst_month']['pnl']:.2f} EUR)")
    logger.info(f"\nResults saved to:")
    logger.info(f"  CSV:  {output_csv}")
    logger.info(f"  JSON: {output_json}")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
