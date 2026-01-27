#!/usr/bin/env python3
"""
OUT-OF-SAMPLE BACKTEST ONLY
Tests ONLY on data the model has never seen (test set: 2024-11-05 onwards)
Includes realistic transaction costs.
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

def main():
    parser = argparse.ArgumentParser(description="Out-of-Sample Backtest (Test Set Only)")
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--initial-balance', type=float, default=1000.0)
    parser.add_argument('--base-risk-percent', type=float, default=1.5)
    parser.add_argument('--tp-pips', type=float, default=23.5, help='TP with transaction costs')
    parser.add_argument('--sl-pips', type=float, default=16.5, help='SL with transaction costs')
    parser.add_argument('--output', type=str, default='/tmp/backtest_test_only')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE BACKTEST - TEST SET ONLY")
    print("=" * 80)
    print("ELIMINATES DATA LEAKAGE - Model has NEVER seen this data!")
    print("Includes realistic transaction costs (1.5 pips per trade)")
    print("=" * 80 + "\n")

    # Load full dataset
    logger.info(f"Loading data from {args.data}...")
    df_full = pd.read_csv(args.data, index_col=0, parse_dates=True)
    df_full.columns = [c.lower() for c in df_full.columns]
    df_full = df_full.sort_index()

    total_bars = len(df_full)
    logger.info(f"Full dataset: {total_bars:,} bars from {df_full.index[0]} to {df_full.index[-1]}")

    # Calculate test set boundary (80% split point)
    test_start_idx = int(total_bars * 0.8)
    test_start_date = df_full.index[test_start_idx]

    logger.info(f"\nData Split:")
    logger.info(f"  Train+Val: 0% to 80% = {df_full.index[0]} to {test_start_date}")
    logger.info(f"  Test:      80% to 100% = {test_start_date} to {df_full.index[-1]}")

    # Keep ALL data for feature calculation, but mark test period
    # We need historical data for Daily timeframe features
    logger.info(f"\n" + "=" * 80)
    logger.info(f"TEST SET (OUT-OF-SAMPLE):")
    logger.info(f"  Trading Start: {test_start_date}")
    logger.info(f"  Trading End:   {df_full.index[-1]}")
    logger.info(f"  Test Bars:     {total_bars - test_start_idx:,} ({(total_bars - test_start_idx)/total_bars*100:.1f}% of total)")
    logger.info(f"  Days:          {(df_full.index[-1] - test_start_date).days} days")
    logger.info(f"  MODEL HAS NEVER SEEN THIS DATA!")
    logger.info(f"  (Using full dataset for feature calculation only)")
    logger.info(f"=" * 80)

    # We'll pass the test_start_date to the backtester to filter trades

    # Transaction costs
    logger.info(f"\nTransaction Costs:")
    logger.info(f"  Spread:    1.0 pips")
    logger.info(f"  Slippage:  0.5 pips")
    logger.info(f"  Total:     1.5 pips per trade")
    logger.info(f"  Original TP/SL:  25.0 / 15.0 pips (R:R = 1.67)")
    logger.info(f"  Realistic TP/SL: {args.tp_pips} / {args.sl_pips} pips (R:R = {args.tp_pips/args.sl_pips:.2f})")

    # Update configs
    TRADING_CONFIG['initial_balance'] = args.initial_balance
    TRADING_CONFIG['tp_pips'] = args.tp_pips
    TRADING_CONFIG['sl_pips'] = args.sl_pips
    CONSERVATIVE_HYBRID_CONFIG['base_risk_percent'] = args.base_risk_percent

    logger.info(f"\nConservative Hybrid Parameters:")
    logger.info(f"  Base Risk: {args.base_risk_percent}%")
    logger.info(f"  Risk Range: 0.8% - 2.5%")
    logger.info(f"  Initial Balance: {args.initial_balance:.2f} EUR")

    # Load ensemble
    logger.info(f"\nLoading ensemble from {args.model_dir}...")
    ensemble = MTFEnsemble()
    ensemble.load(args.model_dir)
    logger.info(f"Ensemble loaded successfully")

    # Run backtest on full data, but trades filtered to test period
    logger.info(f"\n" + "=" * 80)
    logger.info(f"RUNNING BACKTEST...")
    logger.info(f"Will only simulate trades from {test_start_date} onwards")
    logger.info(f"=" * 80 + "\n")

    backtester = DynamicThresholdBacktester(
        ensemble=ensemble,
        threshold_config=THRESHOLD_CONFIG,
        trading_config=TRADING_CONFIG,
        conservative_hybrid_config=CONSERVATIVE_HYBRID_CONFIG,
    )

    # Add test_start_date to trading config to filter trades
    TRADING_CONFIG['test_start_date'] = test_start_date

    results = backtester.run(df_full)

    # Save results
    output_csv = f"{args.output}.csv"
    output_json = f"{args.output}.json"

    monthly_df = pd.DataFrame(results['monthly_stats'])
    monthly_df.to_csv(output_csv, index=False)

    summary = {k: v for k, v in results.items() if k != 'monthly_stats'}
    summary['test_period'] = {
        'start': str(test_start_date),
        'end': str(df_full.index[-1]),
        'days': (df_full.index[-1] - test_start_date).days,
        'bars': total_bars - test_start_idx
    }
    summary['transaction_costs'] = {
        'spread_pips': 1.0,
        'slippage_pips': 0.5,
        'total_pips_per_trade': 1.5,
        'tp_pips': args.tp_pips,
        'sl_pips': args.sl_pips
    }

    with open(output_json, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE TEST RESULTS")
    print("=" * 80)
    print(f"Test Period:  {test_start_date} to {df_full.index[-1]}")
    print(f"Duration:     {(df_full.index[-1] - test_start_date).days} days ({len(results['monthly_stats'])} months)")
    print("=" * 80)
    print(f"\nAccount Performance:")
    print(f"  Initial Balance:       {results['initial_balance']:>10.2f} EUR")
    print(f"  Final Balance:         {results['final_balance']:>10.2f} EUR")
    print(f"  Total Return:          {results['total_return_pct']:>10.2f}%")
    print(f"  Max Drawdown:          {results['max_drawdown_pct']:>10.2f}%")
    print(f"  Sharpe Ratio:          {results['sharpe_ratio']:>10.2f}")
    print(f"\nTrade Summary:")
    print(f"  Total Trades:          {results['total_trades']:>10}")
    print(f"  Winning:               {results['winning_trades']:>10}")
    print(f"  Losing:                {results['losing_trades']:>10}")
    print(f"  Win Rate:              {results['win_rate']:>10.1f}%")
    print(f"\nP&L Summary:")
    print(f"  Total Pips:            {results['total_pips']:>10.1f}")
    print(f"  Avg Win:               {results['avg_win_pips']:>10.1f} pips")
    print(f"  Avg Loss:              {results['avg_loss_pips']:>10.1f} pips")
    print(f"  Profit Factor:         {results['profit_factor']:>10.2f}x")
    print(f"\nExit Analysis:")
    print(f"  Take Profit:           {results['tp_hits']:>10} ({results['tp_hits']/results['total_trades']*100:.1f}%)")
    print(f"  Stop Loss:             {results['sl_hits']:>10} ({results['sl_hits']/results['total_trades']*100:.1f}%)")
    print(f"  Timeout:               {results['timeouts']:>10} ({results['timeouts']/results['total_trades']*100:.1f}%)")
    print(f"\nMonthly Performance:")
    print(f"  Avg Monthly Return:    {results['avg_monthly_return']:>10.1f}%")
    print(f"  Best Month:            {results['best_month']['month']} ({results['best_month']['pnl']:+.2f} EUR)")
    print(f"  Worst Month:           {results['worst_month']['month']} ({results['worst_month']['pnl']:+.2f} EUR)")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  CSV:  {output_csv}")
    print(f"  JSON: {output_json}")
    print("=" * 80)
    print("\nThis is TRUE out-of-sample performance!")
    print("NO DATA LEAKAGE - Model has never seen this data during training.")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
