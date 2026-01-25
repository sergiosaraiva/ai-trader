#!/usr/bin/env python3
"""Compare gradient boosting frameworks for MTF Ensemble.

Benchmarks XGBoost, LightGBM, and CatBoost to determine the best framework
for the trading model ensemble.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.models.multi_timeframe import MTFEnsemble, MTFEnsembleConfig, StackingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load 5-minute OHLCV data."""
    logger.info(f"Loading data from {data_path}")

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    df.columns = [c.lower() for c in df.columns]

    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = None
        for col in ["timestamp", "time", "date", "datetime"]:
            if col in df.columns:
                time_col = col
                break

        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
        else:
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    return df


def train_and_backtest(
    df_5min: pd.DataFrame,
    model_type: str,
    output_dir: Path,
    confidence_threshold: float = 0.55,
) -> dict:
    """Train model with given framework and run backtest."""

    start_time = time.time()

    # Create config with model_type
    config = MTFEnsembleConfig(
        weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        agreement_bonus=0.05,
        use_regime_adjustment=True,
        include_sentiment=True,
        sentiment_source="epu",
        sentiment_by_timeframe={"1H": False, "4H": False, "D": True},
        trading_pair="EURUSD",
        use_stacking=True,
        stacking_config=StackingConfig(blend_with_weighted_avg=0.0),
        model_type=model_type,
    )

    # Create and train ensemble
    ensemble = MTFEnsemble(config=config, model_dir=output_dir)

    logger.info(f"Training {model_type.upper()} ensemble...")
    train_results = ensemble.train(
        df_5min,
        train_ratio=0.6,
        val_ratio=0.2,
        timeframes=["1H", "4H", "D"],
    )

    train_time = time.time() - start_time
    logger.info(f"Training completed in {train_time:.1f}s")

    # Save the model
    ensemble.save()

    # Run backtest using predict_batch
    logger.info(f"Running backtest for {model_type.upper()}...")
    backtest_start = time.time()

    # Prepare test data for backtesting
    from src.features.technical.calculator import TechnicalIndicatorCalculator
    calc = TechnicalIndicatorCalculator(model_type="short_term")

    timeframes = ["1H", "4H", "D"]
    X_dict = {}
    y_dict = {}

    for tf in timeframes:
        model = ensemble.models[tf]
        config_tf = ensemble.model_configs[tf]

        # Resample
        df_tf = ensemble.resample_data(df_5min, config_tf.base_timeframe)

        # Prepare higher TF data
        higher_tf_data = ensemble.prepare_higher_tf_data(df_5min, config_tf.base_timeframe)

        # Get features and labels
        X, y, _ = model.prepare_data(df_tf, higher_tf_data)

        # Split
        n_total = len(X)
        n_train = int(n_total * 0.6)
        n_val = int(n_total * 0.2)
        test_start = n_train + n_val

        X_dict[tf] = X[test_start:]
        y_dict[tf] = y[test_start:]

    # Get common length
    min_len = min(len(X_dict[tf]) for tf in timeframes)
    for tf in timeframes:
        X_dict[tf] = X_dict[tf][:min_len]
        y_dict[tf] = y_dict[tf][:min_len]

    # Get price data for enhanced meta-features
    df_1h = ensemble.resample_data(df_5min, "1H")
    n_total_1h = len(df_1h)
    n_train_1h = int(n_total_1h * 0.6)
    n_val_1h = int(n_total_1h * 0.2)
    test_start_1h = n_train_1h + n_val_1h
    price_data = df_1h.iloc[test_start_1h:test_start_1h + min_len].reset_index(drop=True)

    # Get predictions
    directions, confidences, agreement_scores = ensemble.predict_batch(X_dict, price_data=price_data)

    # Use 1H labels as ground truth
    y_test = y_dict["1H"][:len(directions)]

    # Simple backtest simulation
    positions = []
    equity_curve = [10000.0]  # Start with $10k

    for i in range(len(directions)):
        conf = confidences[i]
        direction = directions[i]
        actual = y_test[i]

        # Only trade if confidence >= threshold
        if conf >= confidence_threshold:
            # Simulate pip profit/loss
            if direction == actual:
                # Correct prediction - profit
                pips = 20  # Average profit per correct trade
            else:
                # Incorrect prediction - loss
                pips = -10  # Average loss per incorrect trade

            positions.append({
                "index": i,
                "direction": direction,
                "confidence": conf,
                "actual": actual,
                "pips": pips,
                "correct": direction == actual,
            })

            # Update equity (simplified: $10 per pip)
            equity_curve.append(equity_curve[-1] + pips * 10)
        else:
            equity_curve.append(equity_curve[-1])

    backtest_time = time.time() - backtest_start

    # Calculate metrics
    total_trades = len(positions)
    if total_trades > 0:
        winning_trades = sum(1 for p in positions if p["correct"])
        win_rate = winning_trades / total_trades
        total_pips = sum(p["pips"] for p in positions)
        avg_pips_per_trade = total_pips / total_trades

        winning_pips = sum(p["pips"] for p in positions if p["correct"])
        losing_pips = abs(sum(p["pips"] for p in positions if not p["correct"]))
        profit_factor = winning_pips / losing_pips if losing_pips > 0 else 0

        # Sharpe ratio calculation
        equity_returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe_ratio = (equity_returns.mean() / equity_returns.std()) * np.sqrt(252) if len(equity_returns) > 0 else 0

        # Max drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
    else:
        win_rate = 0
        total_pips = 0
        avg_pips_per_trade = 0
        profit_factor = 0
        sharpe_ratio = 0
        max_drawdown = 0

    # Extract key metrics
    results = {
        "model_type": model_type,
        "train_time_seconds": round(train_time, 1),
        "backtest_time_seconds": round(backtest_time, 1),
        "total_pips": int(total_pips),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 2),
        "total_trades": total_trades,
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown": round(max_drawdown, 4),
        "avg_pips_per_trade": round(avg_pips_per_trade, 2),
        "individual_results": {
            tf: {
                "val_accuracy": train_results.get(tf, {}).get("val_accuracy", 0),
                "train_accuracy": train_results.get(tf, {}).get("train_accuracy", 0),
            }
            for tf in ["1H", "4H", "D"]
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare Gradient Boosting Frameworks")
    parser.add_argument(
        "--data",
        type=str,
        default="data/forex/EURUSD_20200101_20251231_5min_combined.csv",
        help="Path to 5-minute data",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.55,
        help="Confidence threshold for backtest (default: 0.55)",
    )
    parser.add_argument(
        "--frameworks",
        type=str,
        default="xgboost,lightgbm,catboost",
        help="Comma-separated frameworks to compare",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("GRADIENT BOOSTING FRAMEWORK COMPARISON")
    print("=" * 70)
    print(f"Data:       {args.data}")
    print(f"Confidence: {args.confidence}")
    print(f"Frameworks: {args.frameworks}")
    print("=" * 70 + "\n")

    # Load data
    data_path = project_root / args.data
    df_5min = load_data(data_path)

    # Parse frameworks
    frameworks = [f.strip() for f in args.frameworks.split(",")]

    # Run comparison
    all_results = []

    for framework in frameworks:
        print(f"\n{'=' * 70}")
        print(f"TRAINING & TESTING: {framework.upper()}")
        print("=" * 70)

        output_dir = project_root / f"models/gradient_boosting/{framework}"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            results = train_and_backtest(
                df_5min=df_5min,
                model_type=framework,
                output_dir=output_dir,
                confidence_threshold=args.confidence,
            )
            all_results.append(results)

            print(f"\n{framework.upper()} Results:")
            print(f"  Total Pips:    {results['total_pips']:+,}")
            print(f"  Win Rate:      {results['win_rate']:.1%}")
            print(f"  Profit Factor: {results['profit_factor']:.2f}")
            print(f"  Total Trades:  {results['total_trades']}")
            print(f"  Train Time:    {results['train_time_seconds']:.1f}s")

        except Exception as e:
            logger.error(f"Error with {framework}: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "model_type": framework,
                "error": str(e),
            })

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print(f"\n{'Framework':<12} {'Pips':>10} {'Win Rate':>10} {'PF':>8} {'Trades':>8} {'Train(s)':>10}")
    print("-" * 70)

    baseline_pips = 7987  # XGBoost baseline from docs (55% threshold)
    best_framework = None
    best_pips = -float("inf")

    for result in all_results:
        if "error" in result:
            print(f"{result['model_type']:<12} ERROR: {result['error']}")
            continue

        pips = result["total_pips"]
        wr = result["win_rate"]
        pf = result["profit_factor"]
        trades = result["total_trades"]
        train_time = result["train_time_seconds"]

        print(f"{result['model_type']:<12} {pips:>+10,} {wr:>9.1%} {pf:>8.2f} {trades:>8} {train_time:>10.1f}")

        if pips > best_pips:
            best_pips = pips
            best_framework = result["model_type"]

    print("-" * 70)
    print(f"{'Baseline:':<12} {baseline_pips:>+10,} (XGBoost from docs at 55%)")

    if best_framework:
        improvement = ((best_pips - baseline_pips) / baseline_pips) * 100
        print(f"\nWinner: {best_framework.upper()} with {best_pips:+,} pips")
        print(f"Improvement vs baseline: {improvement:+.1f}%")

    # Save results
    output_path = project_root / "data/gradient_boosting_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparison_data = {
        "comparison_date": datetime.now().isoformat(),
        "confidence_threshold": args.confidence,
        "baseline_pips": baseline_pips,
        "results": all_results,
        "winner": best_framework,
        "best_pips": int(best_pips) if best_framework else None,
    }

    with open(output_path, "w") as f:
        json.dump(comparison_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
