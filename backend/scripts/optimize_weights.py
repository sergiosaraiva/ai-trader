#!/usr/bin/env python3
"""Optimize MTF Ensemble weights through grid search.

This script finds the optimal weight combination for 1H, 4H, and Daily models
by running backtests across a grid of weight values.

OPTIMIZED: Pre-computes all predictions once, then only varies weight combinations
during trading simulation for 100x+ speedup.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import product

import warnings
warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.models.multi_timeframe import MTFEnsemble, MTFEnsembleConfig

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a single backtest."""
    weights: Dict[str, float]
    total_trades: int
    win_rate: float
    total_pips: float
    profit_factor: float
    avg_pips: float
    high_conf_win_rate: float  # >= 60%
    sharpe_estimate: float


@dataclass
class CachedPredictions:
    """Pre-computed predictions for all timestamps."""
    timestamps: np.ndarray
    closes: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    prob_up_1h: np.ndarray  # Probability of UP for 1H model
    prob_up_4h: np.ndarray  # Probability of UP for 4H model
    prob_up_d: np.ndarray   # Probability of UP for Daily model


def prepare_cached_predictions(
    ensemble: MTFEnsemble,
    df_5min: pd.DataFrame,
) -> CachedPredictions:
    """Pre-compute all predictions once for efficient weight optimization."""
    from src.features.technical.calculator import TechnicalIndicatorCalculator

    print("  Pre-computing predictions (this happens once)...")
    calc = TechnicalIndicatorCalculator(model_type="short_term")

    # Prepare 1H data
    print("    Processing 1H model...")
    model_1h = ensemble.models["1H"]
    df_1h = ensemble.resample_data(df_5min, "1H")
    higher_tf_data_1h = ensemble.prepare_higher_tf_data(df_5min, "1H")
    df_1h_features = calc.calculate(df_1h)
    df_1h_features = model_1h.feature_engine.add_all_features(df_1h_features, higher_tf_data_1h)
    df_1h_features = df_1h_features.dropna()

    feature_cols_1h = model_1h.feature_names
    available_cols_1h = [c for c in feature_cols_1h if c in df_1h_features.columns]
    X_1h = df_1h_features[available_cols_1h].values

    # Split
    n_total = len(X_1h)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.2)
    test_start = n_train + n_val

    X_1h_test = X_1h[test_start:]
    df_test = df_1h_features.iloc[test_start:]

    # Get 1H predictions
    preds_1h, confs_1h = model_1h.predict_batch(X_1h_test)
    # Convert to prob_up: if pred=1, prob_up=conf; if pred=0, prob_up=1-conf
    prob_up_1h = np.where(preds_1h == 1, confs_1h, 1 - confs_1h)

    # Prepare 4H predictions
    print("    Processing 4H model...")
    model_4h = ensemble.models["4H"]
    df_4h = ensemble.resample_data(df_5min, "4H")
    higher_tf_data_4h = ensemble.prepare_higher_tf_data(df_5min, "4H")
    df_4h_features = calc.calculate(df_4h)
    df_4h_features = model_4h.feature_engine.add_all_features(df_4h_features, higher_tf_data_4h)
    df_4h_features = df_4h_features.dropna()

    feature_cols_4h = model_4h.feature_names
    available_cols_4h = [c for c in feature_cols_4h if c in df_4h_features.columns]
    X_4h = df_4h_features[available_cols_4h].values
    preds_4h_all, confs_4h_all = model_4h.predict_batch(X_4h)
    prob_up_4h_all = np.where(preds_4h_all == 1, confs_4h_all, 1 - confs_4h_all)
    pred_4h_map = dict(zip(df_4h_features.index, prob_up_4h_all))

    # Prepare Daily predictions
    print("    Processing Daily model...")
    model_d = ensemble.models["D"]
    df_d = ensemble.resample_data(df_5min, "D")
    df_d_features = calc.calculate(df_d)
    df_d_features = model_d.feature_engine.add_all_features(df_d_features, {})
    df_d_features = df_d_features.dropna()

    feature_cols_d = model_d.feature_names
    available_cols_d = [c for c in feature_cols_d if c in df_d_features.columns]
    X_d = df_d_features[available_cols_d].values
    preds_d_all, confs_d_all = model_d.predict_batch(X_d)
    prob_up_d_all = np.where(preds_d_all == 1, confs_d_all, 1 - confs_d_all)
    pred_d_map = dict(zip(df_d_features.index.date, prob_up_d_all))

    # Get aligned data
    timestamps = df_test.index.values
    closes = df_test["close"].values
    highs = df_test["high"].values
    lows = df_test["low"].values

    # Build aligned prob_up arrays for 4H and D
    print("    Aligning predictions across timeframes...")
    n_test = len(timestamps)
    prob_up_4h = np.zeros(n_test)
    prob_up_d = np.zeros(n_test)

    for i, ts in enumerate(df_test.index):
        # 4H alignment
        ts_4h = ts.floor("4H")
        if ts_4h in pred_4h_map:
            prob_up_4h[i] = pred_4h_map[ts_4h]
        else:
            prev_4h_times = [t for t in pred_4h_map.keys() if t <= ts]
            if prev_4h_times:
                prob_up_4h[i] = pred_4h_map[max(prev_4h_times)]
            else:
                prob_up_4h[i] = prob_up_1h[i]  # Fallback

        # Daily alignment
        day = ts.date()
        if day in pred_d_map:
            prob_up_d[i] = pred_d_map[day]
        else:
            prev_days = [d for d in pred_d_map.keys() if d <= day]
            if prev_days:
                prob_up_d[i] = pred_d_map[max(prev_days)]
            else:
                prob_up_d[i] = prob_up_1h[i]  # Fallback

    print(f"    Cached {n_test} prediction rows")

    return CachedPredictions(
        timestamps=timestamps,
        closes=closes,
        highs=highs,
        lows=lows,
        prob_up_1h=prob_up_1h,
        prob_up_4h=prob_up_4h,
        prob_up_d=prob_up_d,
    )


def run_backtest_with_weights(
    cache: CachedPredictions,
    weights: Dict[str, float],
    min_confidence: float = 0.55,
    min_agreement: float = 0.5,
    tp_pips: float = 25.0,
    sl_pips: float = 15.0,
    max_holding_bars: int = 12,
) -> BacktestResult:
    """Run backtest using cached predictions with specified weights."""
    w_1h = weights["1H"]
    w_4h = weights["4H"]
    w_d = weights["D"]

    n = len(cache.timestamps)
    closes = cache.closes
    highs = cache.highs
    lows = cache.lows

    # Pre-compute ensemble predictions with these weights
    weighted_prob_up = (
        w_1h * cache.prob_up_1h +
        w_4h * cache.prob_up_4h +
        w_d * cache.prob_up_d
    )

    # Direction: 1 if prob_up > 0.5, else 0
    directions = (weighted_prob_up > 0.5).astype(int)

    # Confidence: how far from 0.5
    base_conf = np.abs(weighted_prob_up - 0.5) * 2 + 0.5

    # Agreement: count how many models agree with final direction
    dir_1h = (cache.prob_up_1h > 0.5).astype(int)
    dir_4h = (cache.prob_up_4h > 0.5).astype(int)
    dir_d = (cache.prob_up_d > 0.5).astype(int)

    agreement_count = (
        (dir_1h == directions).astype(int) +
        (dir_4h == directions).astype(int) +
        (dir_d == directions).astype(int)
    )
    agreement_score = agreement_count / 3.0

    # Apply agreement bonus
    confidences = np.where(agreement_count == 3, np.minimum(base_conf + 0.05, 1.0), base_conf)

    # Simulate trading
    pip_value = 0.0001
    trades = []
    i = 0

    while i < n - max_holding_bars:
        conf = confidences[i]
        agreement = agreement_score[i]
        pred = directions[i]

        if conf >= min_confidence and agreement >= min_agreement:
            entry_price = closes[i]
            is_long = pred == 1

            if is_long:
                tp_price = entry_price + tp_pips * pip_value
                sl_price = entry_price - sl_pips * pip_value
            else:
                tp_price = entry_price - tp_pips * pip_value
                sl_price = entry_price + sl_pips * pip_value

            exit_price = None
            exit_idx = i

            for j in range(i + 1, min(i + max_holding_bars + 1, n)):
                if is_long:
                    if highs[j] >= tp_price:
                        exit_price = tp_price
                        exit_idx = j
                        break
                    if lows[j] <= sl_price:
                        exit_price = sl_price
                        exit_idx = j
                        break
                else:
                    if lows[j] <= tp_price:
                        exit_price = tp_price
                        exit_idx = j
                        break
                    if highs[j] >= sl_price:
                        exit_price = sl_price
                        exit_idx = j
                        break

            if exit_price is None:
                exit_idx = min(i + max_holding_bars, n - 1)
                exit_price = closes[exit_idx]

            if is_long:
                pnl_pips = (exit_price - entry_price) / pip_value
            else:
                pnl_pips = (entry_price - exit_price) / pip_value

            trades.append({"pnl": pnl_pips, "conf": conf})
            i = exit_idx

        i += 1

    # Calculate results
    if not trades:
        return BacktestResult(
            weights=weights,
            total_trades=0,
            win_rate=0,
            total_pips=0,
            profit_factor=0,
            avg_pips=0,
            high_conf_win_rate=0,
            sharpe_estimate=0,
        )

    pnls = [t["pnl"] for t in trades]
    confs = [t["conf"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_profit = sum(wins) if wins else 0
    total_loss = abs(sum(losses)) if losses else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    # High confidence trades
    high_conf_trades = [(p, c) for p, c in zip(pnls, confs) if c >= 0.60]
    high_conf_wins = [p for p, c in high_conf_trades if p > 0]
    high_conf_win_rate = len(high_conf_wins) / len(high_conf_trades) * 100 if high_conf_trades else 0

    # Sharpe estimate (simplified)
    pnl_array = np.array(pnls)
    sharpe = np.mean(pnl_array) / np.std(pnl_array) * np.sqrt(252) if np.std(pnl_array) > 0 else 0

    return BacktestResult(
        weights=weights,
        total_trades=len(trades),
        win_rate=len(wins) / len(trades) * 100,
        total_pips=sum(pnls),
        profit_factor=profit_factor,
        avg_pips=np.mean(pnls),
        high_conf_win_rate=high_conf_win_rate,
        sharpe_estimate=sharpe,
    )


def generate_weight_grid(step: float = 0.05, min_weight: float = 0.05) -> List[Dict[str, float]]:
    """Generate all valid weight combinations."""
    combinations = []

    # Generate weights from min_weight to 1-2*min_weight
    w1_range = np.arange(min_weight, 1 - 2 * min_weight + step, step)

    for w1 in w1_range:
        for w2 in np.arange(min_weight, 1 - w1 - min_weight + step, step):
            w3 = 1 - w1 - w2
            if w3 >= min_weight - 0.001:  # Small tolerance for floating point
                combinations.append({
                    "1H": round(w1, 2),
                    "4H": round(w2, 2),
                    "D": round(max(0, w3), 2),
                })

    return combinations


def main():
    parser = argparse.ArgumentParser(description="Optimize MTF Ensemble weights")
    parser.add_argument("--data", type=str, default="data/forex/EURUSD_20200101_20251231_5min_combined.csv")
    parser.add_argument("--model-dir", type=str, default="models/mtf_ensemble")
    parser.add_argument("--step", type=float, default=0.05, help="Weight increment step (default: 0.05)")
    parser.add_argument("--min-weight", type=float, default=0.05, help="Minimum weight for any model (default: 0.05)")
    parser.add_argument("--confidence", type=float, default=0.55, help="Minimum confidence threshold")
    parser.add_argument("--metric", type=str, default="pips", choices=["pips", "pf", "winrate", "sharpe"],
                        help="Metric to optimize (default: pips)")
    parser.add_argument("--top", type=int, default=20, help="Show top N results")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("MTF ENSEMBLE WEIGHT OPTIMIZATION")
    print("=" * 70)

    # Load data
    data_path = project_root / args.data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = next((c for c in ["timestamp", "time", "date", "datetime"] if c in df.columns), None)
        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
        else:
            df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    print(f"Loaded {len(df)} bars")

    # Load ensemble
    model_dir = project_root / args.model_dir
    metadata_path = model_dir / "training_metadata.json"

    include_sentiment = False
    trading_pair = "EURUSD"
    sentiment_by_timeframe = {"1H": False, "4H": False, "D": False}
    sentiment_source = "epu"
    use_stacking = False

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            include_sentiment = metadata.get("include_sentiment", False)
            trading_pair = metadata.get("trading_pair", "EURUSD")
            sentiment_by_timeframe = metadata.get("sentiment_by_timeframe", sentiment_by_timeframe)
            sentiment_source = metadata.get("sentiment_source", "epu")
            use_stacking = metadata.get("use_stacking", False)

    config = MTFEnsembleConfig(
        weights={"1H": 0.6, "4H": 0.3, "D": 0.1},  # Will be overridden
        include_sentiment=include_sentiment,
        trading_pair=trading_pair,
        sentiment_source=sentiment_source,
        sentiment_by_timeframe=sentiment_by_timeframe,
        use_stacking=use_stacking,
    )

    ensemble = MTFEnsemble(config=config, model_dir=model_dir)
    ensemble.load()
    print(f"Loaded ensemble from {model_dir}")

    # Pre-compute all predictions ONCE
    print("\nCaching predictions...")
    cache = prepare_cached_predictions(ensemble, df)

    # Generate weight combinations
    weight_grid = generate_weight_grid(step=args.step, min_weight=args.min_weight)
    print(f"\nTesting {len(weight_grid)} weight combinations (step={args.step}, min={args.min_weight})")
    print(f"Optimization metric: {args.metric}")
    print(f"Confidence threshold: {args.confidence}")

    # Run backtests (now FAST since predictions are cached)
    results: List[BacktestResult] = []

    print("\nRunning backtests (using cached predictions)...")
    for i, weights in enumerate(weight_grid):
        if (i + 1) % 50 == 0 or i == 0 or (i + 1) == len(weight_grid):
            print(f"  Progress: {i + 1}/{len(weight_grid)} ({(i + 1) / len(weight_grid) * 100:.1f}%)")

        result = run_backtest_with_weights(
            cache=cache,
            weights=weights,
            min_confidence=args.confidence,
        )
        results.append(result)

    # Sort by chosen metric
    if args.metric == "pips":
        results.sort(key=lambda x: x.total_pips, reverse=True)
    elif args.metric == "pf":
        results.sort(key=lambda x: x.profit_factor if x.profit_factor != float("inf") else 0, reverse=True)
    elif args.metric == "winrate":
        results.sort(key=lambda x: x.win_rate, reverse=True)
    elif args.metric == "sharpe":
        results.sort(key=lambda x: x.sharpe_estimate, reverse=True)

    # Print top results
    print("\n" + "=" * 70)
    print(f"TOP {args.top} WEIGHT COMBINATIONS (by {args.metric})")
    print("=" * 70)
    print(f"{'Rank':<5} {'1H':>6} {'4H':>6} {'D':>6} {'Trades':>7} {'WinRate':>8} {'Pips':>10} {'PF':>7} {'HCWin%':>8}")
    print("-" * 70)

    for i, r in enumerate(results[:args.top]):
        print(f"{i + 1:<5} {r.weights['1H']:>6.2f} {r.weights['4H']:>6.2f} {r.weights['D']:>6.2f} "
              f"{r.total_trades:>7} {r.win_rate:>7.1f}% {r.total_pips:>+9.1f} {r.profit_factor:>6.2f} {r.high_conf_win_rate:>7.1f}%")

    # Show current weights comparison
    current = next((r for r in results if r.weights == {"1H": 0.60, "4H": 0.30, "D": 0.10}), None)
    if current:
        current_rank = results.index(current) + 1
        print("-" * 70)
        print(f"Current (60/30/10) rank: #{current_rank}")
        print(f"{'*':<5} {current.weights['1H']:>6.2f} {current.weights['4H']:>6.2f} {current.weights['D']:>6.2f} "
              f"{current.total_trades:>7} {current.win_rate:>7.1f}% {current.total_pips:>+9.1f} {current.profit_factor:>6.2f} {current.high_conf_win_rate:>7.1f}%")

    # Best result
    best = results[0]
    print("\n" + "=" * 70)
    print("OPTIMAL WEIGHTS")
    print("=" * 70)
    print(f"  1H Weight: {best.weights['1H']:.2f}")
    print(f"  4H Weight: {best.weights['4H']:.2f}")
    print(f"  D Weight:  {best.weights['D']:.2f}")
    print(f"\nPerformance:")
    print(f"  Total Trades:     {best.total_trades}")
    print(f"  Win Rate:         {best.win_rate:.1f}%")
    print(f"  Total Pips:       {best.total_pips:+.1f}")
    print(f"  Profit Factor:    {best.profit_factor:.2f}")
    print(f"  High-Conf WR:     {best.high_conf_win_rate:.1f}%")
    print(f"  Sharpe Estimate:  {best.sharpe_estimate:.2f}")

    # Improvement over current
    if current:
        print(f"\nImprovement over 60/30/10:")
        pips_diff = best.total_pips - current.total_pips
        pips_pct = (pips_diff / current.total_pips * 100) if current.total_pips != 0 else 0
        print(f"  Pips:        {pips_diff:+.1f} ({pips_pct:+.1f}%)")
        print(f"  Win Rate:    {best.win_rate - current.win_rate:+.1f}%")
        print(f"  PF:          {best.profit_factor - current.profit_factor:+.2f}")

    # Save results to JSON
    output_path = project_root / "data" / "weight_optimization_results.json"
    output_data = {
        "optimization_metric": args.metric,
        "confidence_threshold": args.confidence,
        "step": args.step,
        "min_weight": args.min_weight,
        "total_combinations": len(weight_grid),
        "optimal_weights": best.weights,
        "optimal_performance": {
            "total_trades": best.total_trades,
            "win_rate": best.win_rate,
            "total_pips": best.total_pips,
            "profit_factor": best.profit_factor,
            "high_conf_win_rate": best.high_conf_win_rate,
            "sharpe_estimate": best.sharpe_estimate,
        },
        "current_weights": {"1H": 0.60, "4H": 0.30, "D": 0.10},
        "current_performance": {
            "total_trades": current.total_trades if current else 0,
            "win_rate": current.win_rate if current else 0,
            "total_pips": current.total_pips if current else 0,
            "profit_factor": current.profit_factor if current else 0,
        } if current else None,
        "top_20": [
            {
                "rank": i + 1,
                "weights": r.weights,
                "total_trades": r.total_trades,
                "win_rate": r.win_rate,
                "total_pips": r.total_pips,
                "profit_factor": r.profit_factor,
            }
            for i, r in enumerate(results[:20])
        ],
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
