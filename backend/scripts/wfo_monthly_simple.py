#!/usr/bin/env python3
"""
Simplified Monthly Disaggregation using WFO backtest function.

This version imports and reuses the proven WFO backtest logic,
then extracts trade timestamps for monthly aggregation.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the proven WFO backtest function
from walk_forward_optimization import (
    load_wfo_results,
    load_data,
    run_window_backtest,
)
from src.models.multi_timeframe import MTFEnsemble

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_monthly_from_wfo():
    """
    Extract monthly performance by re-running WFO backtests
    and capturing trade-level data with timestamps.

    This approach:
    1. Loads each WFO window model
    2. Runs the exact same backtest as WFO validation
    3. But captures individual trade timestamps
    4. Aggregates by month
    """
    print("Loading WFO results...")
    wfo_data = load_wfo_results()

    data_path = project_root / "data/forex/EURUSD_20200101_20251231_5min_combined.csv"
    df_5min = load_data(data_path)

    all_trades_with_timestamps = []

    for window in wfo_data["windows"]:
        window_id = window["window_id"]
        test_period = window["test_period"]

        print(f"\nProcessing Window {window_id}: {test_period}")

        # Parse test period
        start_str, end_str = test_period.split(" to ")
        start_date = pd.Timestamp(f"{start_str}-01")
        end_date = pd.Timestamp(f"{end_str}-28") + pd.offsets.MonthEnd(0)

        # Filter test data
        df_test = df_5min[(df_5min.index >= start_date) & (df_5min.index <= end_date)].copy()

        if len(df_test) < 1000:
            print(f"  Skipping - insufficient data")
            continue

        # Load window model
        window_dir = project_root / "models" / "wfo_conf60" / f"window_{window_id}"
        if not window_dir.exists():
            print(f"  Skipping - model not found")
            continue

        ensemble = MTFEnsemble(model_dir=window_dir)
        ensemble.load()

        if not ensemble.is_trained:
            print(f"  Skipping - model not trained")
            continue

        # Run WFO backtest (this returns aggregated results, not trade-level)
        backtest_results = run_window_backtest(
            ensemble=ensemble,
            df_5min=df_test,
            min_confidence=0.60,  # Use 60% threshold
            initial_balance=10000.0,
            risk_per_trade=0.02,
            reduce_risk_on_losses=True,
        )

        print(f"  Window {window_id}: {backtest_results['total_trades']} trades, {backtest_results['win_rate']:.1f}% win rate")

    print("\n⚠️  Note: WFO backtest function doesn't return trade-level timestamps")
    print("    Monthly disaggregation requires modifying the backtest function")
    print("    to capture individual trade data.")


if __name__ == "__main__":
    extract_monthly_from_wfo()
