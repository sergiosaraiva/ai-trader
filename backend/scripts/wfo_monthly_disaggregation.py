#!/usr/bin/env python3
"""
Monthly Disaggregation of WFO Results.

This script takes the 8 WFO validation windows and disaggregates the results
into month-by-month performance across the entire test period (2022-2025).

Uses the validated WFO backtest function to ensure feature parity.

Output:
- CSV with monthly/weekly/quarterly gains/losses
- JSON with period metrics
- Summary statistics by period
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import WFO backtest function to reuse proven logic
from walk_forward_optimization import run_window_backtest
from src.models.multi_timeframe import MTFEnsemble

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_wfo_results(wfo_dir: str = "wfo_validation") -> Dict:
    """Load WFO results JSON."""
    wfo_path = project_root / "models" / wfo_dir / "wfo_results.json"
    if not wfo_path.exists():
        raise FileNotFoundError(f"WFO results not found at {wfo_path}")

    with open(wfo_path) as f:
        return json.load(f)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load 5-minute OHLCV data."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_index()
    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


def run_window_with_trades(
    window_id: int,
    window_dir: Path,
    df_test: pd.DataFrame,
    min_confidence: float = 0.70,
) -> List[Dict]:
    """
    Run backtest on a window using the validated WFO function.

    Returns:
        List of trade dicts with: entry_time, exit_time, direction, pnl_pips, pnl_usd, confidence, exit_reason
    """
    # Load ensemble for this window
    ensemble = MTFEnsemble(model_dir=window_dir)
    ensemble.load()
    if not ensemble.is_trained:
        logger.warning(f"Could not load ensemble for window {window_id}")
        return []

    # Call the proven WFO backtest function with return_trades=True
    result = run_window_backtest(
        ensemble=ensemble,
        df_5min=df_test,
        min_confidence=min_confidence,
        min_agreement=0.5,
        tp_pips=25.0,
        sl_pips=15.0,
        max_holding_bars=12,
        initial_balance=10000.0,
        risk_per_trade=0.02,
        reduce_risk_on_losses=True,
        spread_pips=1.0,
        slippage_pips=0.5,
        use_regime_filter=True,
        use_equity_filter=True,
        use_volatility_sizing=True,
        daily_loss_limit=0.03,
        return_trades=True,  # Request trade-level data
    )

    trades = result.get("trades", [])
    logger.info(f"Window {window_id}: Generated {len(trades)} trades")
    return trades


def aggregate_by_period(trades: List[Dict], period: str = "M") -> pd.DataFrame:
    """
    Aggregate trades by time period.

    Args:
        trades: List of trade dictionaries
        period: Aggregation period - 'W' (weekly), 'M' (monthly), 'Q' (quarterly)

    Returns:
        DataFrame with columns: period_label, trades, win_rate, total_pips, pnl_usd, balance_end
    """
    if not trades:
        return pd.DataFrame()

    # Map period codes to labels
    period_labels = {
        "W": "week",
        "M": "month",
        "Q": "quarter"
    }

    period_label = period_labels.get(period, "period")

    df = pd.DataFrame(trades)
    # Use exit_time for period assignment (when P&L realized)
    df["period"] = df["exit_time"].dt.to_period(period)

    aggregated = df.groupby("period").agg({
        "pnl_pips": ["count", "sum", lambda x: (x > 0).sum()],
        "pnl_usd": "sum",
        "balance_after": "last",
    })

    aggregated.columns = ["trades", "total_pips", "wins", "pnl_usd", "balance_end"]
    aggregated["win_rate"] = (aggregated["wins"] / aggregated["trades"] * 100).round(1)
    aggregated["total_pips"] = aggregated["total_pips"].round(1)
    aggregated["pnl_usd"] = aggregated["pnl_usd"].round(2)
    aggregated["balance_end"] = aggregated["balance_end"].round(2)

    aggregated = aggregated.reset_index()
    aggregated["period"] = aggregated["period"].astype(str)

    # Rename column to reflect period type
    aggregated = aggregated.rename(columns={"period": f"year_{period_label}"})

    return aggregated[[f"year_{period_label}", "trades", "win_rate", "total_pips", "pnl_usd", "balance_end"]]


def main():
    parser = argparse.ArgumentParser(description="Time-based disaggregation of WFO results")
    parser.add_argument(
        "--data",
        type=str,
        default="data/forex/EURUSD_20200101_20251231_5min_combined.csv",
        help="Path to 5-minute data",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/wfo_monthly_results.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="data/wfo_monthly_results.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.70,
        help="Minimum confidence threshold",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="M",
        choices=["W", "M", "Q"],
        help="Aggregation period: W (weekly), M (monthly), Q (quarterly). Default: M",
    )
    parser.add_argument(
        "--wfo-dir",
        type=str,
        default="wfo_validation",
        help="WFO directory name under models/ (e.g., 'wfo_validation', 'wfo_conf60', 'wfo_conf60_18mo')",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("WFO PERIOD DISAGGREGATION")
    print("=" * 80)
    print(f"WFO Directory: models/{args.wfo_dir}")

    # Load WFO results
    wfo_results = load_wfo_results(wfo_dir=args.wfo_dir)
    print(f"\nLoaded WFO results: {wfo_results['summary']['total_windows']} windows")

    # Load data
    data_path = project_root / args.data
    df_5min = load_data(data_path)

    # Process each window
    all_trades = []

    for window in wfo_results["windows"]:
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
            print(f"  Warning: Insufficient test data ({len(df_test)} bars), skipping")
            continue

        # Load window model
        window_dir = project_root / "models" / args.wfo_dir / f"window_{window_id}"
        if not window_dir.exists():
            print(f"  Warning: Window directory not found, skipping")
            continue

        # Run backtest using validated WFO function
        trades = run_window_with_trades(
            window_id=window_id,
            window_dir=window_dir,
            df_test=df_test,
            min_confidence=args.confidence,
        )

        all_trades.extend(trades)

    if not all_trades:
        print("\n❌ No trades generated. Check WFO models and data availability.")
        return

    print(f"\n✅ Total trades across all windows: {len(all_trades)}")

    # Aggregate by period
    period_name = {"W": "week", "M": "month", "Q": "quarter"}[args.period]
    print(f"\nAggregating by {period_name}...")
    period_df = aggregate_by_period(all_trades, period=args.period)

    # Add cumulative balance
    period_df["balance_change"] = period_df["balance_end"].diff()
    period_df.loc[0, "balance_change"] = period_df.loc[0, "balance_end"] - 10000.0

    # Calculate returns (column name reflects period)
    period_col = f"{period_name}_return_pct"
    period_df[period_col] = (
        period_df["pnl_usd"] / period_df["balance_end"].shift(1).fillna(10000.0) * 100
    ).round(2)

    # Save CSV
    output_csv = project_root / args.output_csv
    period_df.to_csv(output_csv, index=False)
    print(f"\n✅ CSV saved to: {output_csv}")

    # Save JSON
    output_json = project_root / args.output_json
    period_json = {
        "metadata": {
            "generated_at": pd.Timestamp.now().isoformat(),
            "confidence_threshold": args.confidence,
            "aggregation_period": args.period,
            "period_name": period_name,
            "total_periods": len(period_df),
            "total_trades": int(period_df["trades"].sum()),
        },
        "results": period_df.to_dict(orient="records"),
        "summary": {
            "total_pips": float(period_df["total_pips"].sum()),
            "total_pnl_usd": float(period_df["pnl_usd"].sum()),
            f"avg_{period_name}_trades": float(period_df["trades"].mean()),
            f"avg_{period_name}_return_pct": float(period_df[period_col].mean()),
            f"best_{period_name}": period_df.loc[period_df["pnl_usd"].idxmax()].to_dict(),
            f"worst_{period_name}": period_df.loc[period_df["pnl_usd"].idxmin()].to_dict(),
        },
    }

    with open(output_json, "w") as f:
        json.dump(period_json, f, indent=2)
    print(f"✅ JSON saved to: {output_json}")

    # Print summary
    period_col_name = list(period_df.columns)[0]  # First column is period identifier
    print("\n" + "=" * 80)
    print(f"{period_name.upper()} SUMMARY")
    print("=" * 80)
    print(f"\nPeriod: {period_df[period_col_name].iloc[0]} to {period_df[period_col_name].iloc[-1]}")
    print(f"Total {period_name.capitalize()}s: {len(period_df)}")
    print(f"Total Trades: {period_df['trades'].sum()}")
    print(f"Avg Trades/{period_name.capitalize()}: {period_df['trades'].mean():.1f}")
    print(f"Total Pips: {period_df['total_pips'].sum():+.1f}")
    print(f"Total PnL: ${period_df['pnl_usd'].sum():,.2f}")
    print(f"Avg {period_name.capitalize()} Return: {period_df[period_col].mean():+.2f}%")

    print(f"\nBest {period_name.capitalize()}: {period_json['summary'][f'best_{period_name}'][period_col_name]}")
    print(f"  PnL: ${period_json['summary'][f'best_{period_name}']['pnl_usd']:,.2f}")
    print(f"  Return: {period_json['summary'][f'best_{period_name}'][period_col]:+.2f}%")

    print(f"\nWorst {period_name.capitalize()}: {period_json['summary'][f'worst_{period_name}'][period_col_name]}")
    print(f"  PnL: ${period_json['summary'][f'worst_{period_name}']['pnl_usd']:,.2f}")
    print(f"  Return: {period_json['summary'][f'worst_{period_name}'][period_col]:+.2f}%")

    # Print period breakdown
    print("\n" + "=" * 80)
    print(f"{period_name.upper()}-BY-{period_name.upper()} BREAKDOWN")
    print("=" * 80)
    print(f"\n{period_name.capitalize():<14} {'Trades':>7} {'Win%':>7} {'Pips':>9} {'PnL ($)':>12} {'Return%':>9} {'Balance':>12}")
    print("-" * 80)

    for _, row in period_df.iterrows():
        print(
            f"{row[period_col_name]:<14} "
            f"{row['trades']:>7.0f} "
            f"{row['win_rate']:>6.1f}% "
            f"{row['total_pips']:>+9.1f} "
            f"{row['pnl_usd']:>+12.2f} "
            f"{row[period_col]:>+8.2f}% "
            f"${row['balance_end']:>11,.2f}"
        )

    print("=" * 80)
    print(f"\n✅ {period_name.capitalize()} disaggregation complete!")


if __name__ == "__main__":
    main()
