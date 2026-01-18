#!/usr/bin/env python3
"""Confidence Threshold Optimization for MTF Ensemble.

This script systematically tests different confidence thresholds to find
the optimal balance between:
- Win rate (higher threshold = higher accuracy)
- Trade frequency (higher threshold = fewer trades)
- Total profit (balance of the above)
- Risk-adjusted returns

The goal is to find the threshold that maximizes risk-adjusted profit,
not just win rate or total pips.
"""

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.models.multi_timeframe import MTFEnsemble, MTFEnsembleConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    """Results for a single confidence threshold."""
    threshold: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pips: float
    avg_pips_per_trade: float
    profit_factor: float
    max_drawdown_pips: float
    avg_win_pips: float
    avg_loss_pips: float
    trade_frequency: float  # Trades per day
    sharpe_ratio: float
    profit_per_day: float
    risk_adjusted_return: float  # Pips / Max DD


def run_threshold_backtest(
    ensemble: MTFEnsemble,
    df_5min: pd.DataFrame,
    min_confidence: float,
    min_agreement: float = 0.5,
    tp_pips: float = 25.0,
    sl_pips: float = 15.0,
    max_holding_bars: int = 12,
) -> Tuple[ThresholdResult, List[Dict]]:
    """Run backtest with specific confidence threshold.

    Args:
        ensemble: Trained MTF Ensemble
        df_5min: 5-minute price data
        min_confidence: Minimum confidence threshold
        min_agreement: Minimum agreement score
        tp_pips: Take profit in pips
        sl_pips: Stop loss in pips
        max_holding_bars: Maximum bars to hold

    Returns:
        Tuple of (ThresholdResult, list of trades)
    """
    from src.features.technical.calculator import TechnicalIndicatorCalculator

    calc = TechnicalIndicatorCalculator(model_type="short_term")

    # Prepare 1H data
    model_1h = ensemble.models["1H"]
    df_1h = ensemble.resample_data(df_5min, "1H")
    higher_tf_data_1h = ensemble.prepare_higher_tf_data(df_5min, "1H")
    df_1h_features = calc.calculate(df_1h)
    df_1h_features = model_1h.feature_engine.add_all_features(df_1h_features, higher_tf_data_1h)
    df_1h_features = df_1h_features.dropna()

    feature_cols_1h = model_1h.feature_names
    available_cols_1h = [c for c in feature_cols_1h if c in df_1h_features.columns]
    X_1h = df_1h_features[available_cols_1h].values

    # Split - test portion only
    n_total = len(X_1h)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.2)
    test_start = n_train + n_val

    X_1h_test = X_1h[test_start:]
    df_test = df_1h_features.iloc[test_start:]

    preds_1h, confs_1h = model_1h.predict_batch(X_1h_test)

    # 4H predictions
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
    pred_4h_map = dict(zip(df_4h_features.index, zip(preds_4h_all, confs_4h_all)))

    # Daily predictions
    model_d = ensemble.models["D"]
    df_d = ensemble.resample_data(df_5min, "D")
    df_d_features = calc.calculate(df_d)
    df_d_features = model_d.feature_engine.add_all_features(df_d_features, {})
    df_d_features = df_d_features.dropna()

    feature_cols_d = model_d.feature_names
    available_cols_d = [c for c in feature_cols_d if c in df_d_features.columns]
    X_d = df_d_features[available_cols_d].values
    preds_d_all, confs_d_all = model_d.predict_batch(X_d)
    pred_d_map = dict(zip(df_d_features.index.date, zip(preds_d_all, confs_d_all)))

    # Ensemble weights
    weights = ensemble._normalize_weights(ensemble.config.weights)
    w_1h, w_4h, w_d = weights.get("1H", 0.6), weights.get("4H", 0.3), weights.get("D", 0.1)

    closes = df_test["close"].values
    highs = df_test["high"].values
    lows = df_test["low"].values
    timestamps = df_test.index

    # Combine predictions
    test_directions = []
    test_confidences = []
    test_agreements = []

    for i, ts in enumerate(timestamps):
        p_1h, c_1h = preds_1h[i], confs_1h[i]

        ts_4h = ts.floor("4H")
        if ts_4h in pred_4h_map:
            p_4h, c_4h = pred_4h_map[ts_4h]
        else:
            prev_4h = [t for t in pred_4h_map.keys() if t <= ts]
            p_4h, c_4h = pred_4h_map[max(prev_4h)] if prev_4h else (p_1h, c_1h)

        day = ts.date()
        if day in pred_d_map:
            p_d, c_d = pred_d_map[day]
        else:
            prev_days = [d for d in pred_d_map.keys() if d <= day]
            p_d, c_d = pred_d_map[max(prev_days)] if prev_days else (p_1h, c_1h)

        prob_up_1h = c_1h if p_1h == 1 else 1 - c_1h
        prob_up_4h = c_4h if p_4h == 1 else 1 - c_4h
        prob_up_d = c_d if p_d == 1 else 1 - c_d

        weighted_prob_up = w_1h * prob_up_1h + w_4h * prob_up_4h + w_d * prob_up_d

        direction = 1 if weighted_prob_up > 0.5 else 0
        base_conf = abs(weighted_prob_up - 0.5) * 2 + 0.5

        agreement_count = sum([1 for p in [p_1h, p_4h, p_d] if p == direction])
        agreement_score = agreement_count / 3.0

        if agreement_count == 3:
            conf = min(base_conf + ensemble.config.agreement_bonus, 1.0)
        else:
            conf = base_conf

        test_directions.append(direction)
        test_confidences.append(conf)
        test_agreements.append(agreement_score)

    test_directions = np.array(test_directions)
    test_confidences = np.array(test_confidences)
    test_agreements = np.array(test_agreements)

    # Simulate trading
    pip_multiplier = 0.0001
    trades = []
    cumulative_pips = 0.0
    peak_pips = 0.0
    max_drawdown = 0.0
    pips_history = [0.0]

    i = 0
    n = len(test_directions)

    while i < n - max_holding_bars:
        conf = test_confidences[i]
        agreement = test_agreements[i]
        pred = test_directions[i]

        if conf >= min_confidence and agreement >= min_agreement:
            entry_price = closes[i]
            entry_time = timestamps[i]
            direction = "long" if pred == 1 else "short"

            if direction == "long":
                tp_price = entry_price + tp_pips * pip_multiplier
                sl_price = entry_price - sl_pips * pip_multiplier
            else:
                tp_price = entry_price - tp_pips * pip_multiplier
                sl_price = entry_price + sl_pips * pip_multiplier

            exit_price = None
            exit_reason = None
            exit_idx = i

            for j in range(i + 1, min(i + max_holding_bars + 1, n)):
                if direction == "long":
                    if highs[j] >= tp_price:
                        exit_price, exit_reason = tp_price, "take_profit"
                        exit_idx = j
                        break
                    if lows[j] <= sl_price:
                        exit_price, exit_reason = sl_price, "stop_loss"
                        exit_idx = j
                        break
                else:
                    if lows[j] <= tp_price:
                        exit_price, exit_reason = tp_price, "take_profit"
                        exit_idx = j
                        break
                    if highs[j] >= sl_price:
                        exit_price, exit_reason = sl_price, "stop_loss"
                        exit_idx = j
                        break

            if exit_price is None:
                exit_idx = min(i + max_holding_bars, n - 1)
                exit_price = closes[exit_idx]
                exit_reason = "timeout"

            if direction == "long":
                pnl_pips = (exit_price - entry_price) / pip_multiplier
            else:
                pnl_pips = (entry_price - exit_price) / pip_multiplier

            cumulative_pips += pnl_pips
            pips_history.append(cumulative_pips)

            # Track drawdown
            if cumulative_pips > peak_pips:
                peak_pips = cumulative_pips
            drawdown = peak_pips - cumulative_pips
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            trades.append({
                "entry_time": entry_time,
                "exit_time": timestamps[exit_idx],
                "direction": direction,
                "confidence": conf,
                "pnl_pips": pnl_pips,
                "exit_reason": exit_reason,
                "cumulative_pips": cumulative_pips,
            })

            i = exit_idx

        i += 1

    # Calculate results
    if not trades:
        return ThresholdResult(
            threshold=min_confidence,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            total_pips=0,
            avg_pips_per_trade=0,
            profit_factor=0,
            max_drawdown_pips=0,
            avg_win_pips=0,
            avg_loss_pips=0,
            trade_frequency=0,
            sharpe_ratio=0,
            profit_per_day=0,
            risk_adjusted_return=0,
        ), []

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["pnl_pips"] > 0]
    losses = trades_df[trades_df["pnl_pips"] <= 0]

    total_profit = wins["pnl_pips"].sum() if len(wins) > 0 else 0
    total_loss = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    avg_win = wins["pnl_pips"].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses["pnl_pips"].mean()) if len(losses) > 0 else 0

    # Calculate trading days
    first_trade = trades_df["entry_time"].min()
    last_trade = trades_df["exit_time"].max()
    trading_days = (last_trade - first_trade).days or 1

    trade_frequency = len(trades_df) / trading_days
    profit_per_day = trades_df["pnl_pips"].sum() / trading_days

    # Sharpe ratio (simplified - using pips)
    returns = trades_df["pnl_pips"].values
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

    # Risk-adjusted return
    risk_adjusted = trades_df["pnl_pips"].sum() / max_drawdown if max_drawdown > 0 else float("inf")

    result = ThresholdResult(
        threshold=min_confidence,
        total_trades=len(trades_df),
        winning_trades=len(wins),
        losing_trades=len(losses),
        win_rate=len(wins) / len(trades_df) * 100,
        total_pips=trades_df["pnl_pips"].sum(),
        avg_pips_per_trade=trades_df["pnl_pips"].mean(),
        profit_factor=profit_factor if profit_factor != float("inf") else 999.99,
        max_drawdown_pips=max_drawdown,
        avg_win_pips=avg_win,
        avg_loss_pips=avg_loss,
        trade_frequency=trade_frequency,
        sharpe_ratio=sharpe,
        profit_per_day=profit_per_day,
        risk_adjusted_return=risk_adjusted if risk_adjusted != float("inf") else 999.99,
    )

    return result, trades


def find_optimal_threshold(results: List[ThresholdResult]) -> Tuple[ThresholdResult, str]:
    """Find the optimal threshold based on multiple criteria.

    Returns:
        Tuple of (optimal_result, recommendation_reason)
    """
    if not results:
        return None, "No results"

    # Filter out results with 0 trades
    valid_results = [r for r in results if r.total_trades > 0]
    if not valid_results:
        return None, "No valid results"

    # Score each threshold
    scores = []
    for r in valid_results:
        # Normalize metrics (0-1 scale)
        max_pips = max(x.total_pips for x in valid_results)
        max_rar = max(x.risk_adjusted_return for x in valid_results if x.risk_adjusted_return < 999)
        max_pf = max(x.profit_factor for x in valid_results if x.profit_factor < 999)
        max_wr = max(x.win_rate for x in valid_results)

        # Scoring weights
        pips_score = r.total_pips / max_pips if max_pips > 0 else 0
        rar_score = min(r.risk_adjusted_return, max_rar) / max_rar if max_rar > 0 else 0
        pf_score = min(r.profit_factor, max_pf) / max_pf if max_pf > 0 else 0
        wr_score = r.win_rate / max_wr if max_wr > 0 else 0

        # Composite score (weighted)
        # Prioritize: RAR (35%), Profit Factor (25%), Win Rate (25%), Total Pips (15%)
        composite = (
            0.35 * rar_score +
            0.25 * pf_score +
            0.25 * wr_score +
            0.15 * pips_score
        )

        scores.append((r, composite))

    # Find best score
    best_result, best_score = max(scores, key=lambda x: x[1])

    # Generate recommendation reason
    reason = (
        f"Optimal threshold {best_result.threshold:.0%} provides best balance: "
        f"{best_result.win_rate:.1f}% win rate, "
        f"{best_result.profit_factor:.2f} PF, "
        f"{best_result.total_pips:+.0f} pips, "
        f"{best_result.risk_adjusted_return:.2f} RAR"
    )

    return best_result, reason


def main():
    parser = argparse.ArgumentParser(description="Optimize Confidence Threshold")
    parser.add_argument("--data", type=str, default="data/forex/EURUSD_20200101_20251231_5min_combined.csv")
    parser.add_argument("--model-dir", type=str, default="models/mtf_ensemble")
    parser.add_argument("--output", type=str, default="results/confidence_optimization")
    parser.add_argument("--thresholds", type=str, default="0.55,0.58,0.60,0.62,0.65,0.68,0.70,0.75",
                        help="Comma-separated confidence thresholds to test")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("CONFIDENCE THRESHOLD OPTIMIZATION")
    print("=" * 80)
    print(f"Data:       {args.data}")
    print(f"Model:      {args.model_dir}")
    print(f"Thresholds: {args.thresholds}")
    print("=" * 80)

    # Parse thresholds
    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]

    # Load data
    data_path = project_root / args.data
    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]
    time_col = next((c for c in ["timestamp", "time", "date", "datetime"] if c in df.columns), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    df = df.sort_index()
    logger.info(f"Loaded {len(df)} bars")

    # Load model
    model_dir = project_root / args.model_dir
    metadata_path = model_dir / "training_metadata.json"

    weights = {"1H": 0.6, "4H": 0.3, "D": 0.1}
    include_sentiment = False
    sentiment_by_timeframe = {"1H": False, "4H": False, "D": False}
    sentiment_source = "epu"

    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            include_sentiment = metadata.get("include_sentiment", False)
            weights = metadata.get("weights", weights)
            sentiment_by_timeframe = metadata.get("sentiment_by_timeframe", sentiment_by_timeframe)
            sentiment_source = metadata.get("sentiment_source", "epu")

    config = MTFEnsembleConfig(
        weights=weights,
        include_sentiment=include_sentiment,
        sentiment_source=sentiment_source,
        sentiment_by_timeframe=sentiment_by_timeframe,
    )

    ensemble = MTFEnsemble(config=config, model_dir=model_dir)
    ensemble.load()

    # Run backtests for each threshold
    results = []
    all_trades = {}

    for threshold in thresholds:
        print(f"\nTesting threshold: {threshold:.0%}...")
        result, trades = run_threshold_backtest(
            ensemble=ensemble,
            df_5min=df,
            min_confidence=threshold,
        )
        results.append(result)
        all_trades[threshold] = trades
        print(f"  Trades: {result.total_trades}, Win Rate: {result.win_rate:.1f}%, "
              f"Pips: {result.total_pips:+.0f}, PF: {result.profit_factor:.2f}")

    # Print comparison table
    print("\n" + "=" * 100)
    print("RESULTS COMPARISON")
    print("=" * 100)

    print(f"\n{'Threshold':<12} {'Trades':>8} {'Win%':>8} {'Pips':>10} {'PF':>8} "
          f"{'MaxDD':>8} {'Avg/Trade':>10} {'RAR':>8} {'Sharpe':>8}")
    print("-" * 100)

    baseline = results[0]  # 55% threshold as baseline

    for r in results:
        pips_diff = r.total_pips - baseline.total_pips
        indicator = "  " if r.threshold == 0.55 else f"({pips_diff:+.0f})"
        print(f"{r.threshold:<12.0%} {r.total_trades:>8} {r.win_rate:>7.1f}% "
              f"{r.total_pips:>+9.0f} {r.profit_factor:>7.2f} "
              f"{r.max_drawdown_pips:>7.0f} {r.avg_pips_per_trade:>+9.1f} "
              f"{r.risk_adjusted_return:>7.2f} {r.sharpe_ratio:>7.2f}")

    # Find optimal threshold
    optimal, reason = find_optimal_threshold(results)

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Trade-off analysis
    print("\nTrade-off Analysis (vs 55% baseline):")
    print("-" * 80)

    for r in results[1:]:  # Skip baseline
        trade_reduction = (baseline.total_trades - r.total_trades) / baseline.total_trades * 100
        win_rate_gain = r.win_rate - baseline.win_rate
        pips_diff = r.total_pips - baseline.total_pips
        pf_diff = r.profit_factor - baseline.profit_factor

        print(f"  {r.threshold:.0%}: {trade_reduction:+.1f}% trades, {win_rate_gain:+.1f}% win rate, "
              f"{pips_diff:+.0f} pips, {pf_diff:+.2f} PF")

    # Efficiency analysis
    print("\nEfficiency Analysis (Pips per Trade):")
    print("-" * 80)

    for r in results:
        efficiency = r.total_pips / r.total_trades if r.total_trades > 0 else 0
        print(f"  {r.threshold:.0%}: {efficiency:+.2f} pips/trade")

    # Find best by different criteria
    print("\nBest by Criteria:")
    print("-" * 80)

    best_pips = max(results, key=lambda x: x.total_pips)
    best_wr = max(results, key=lambda x: x.win_rate)
    best_pf = max(results, key=lambda x: x.profit_factor if x.profit_factor < 999 else 0)
    best_rar = max(results, key=lambda x: x.risk_adjusted_return if x.risk_adjusted_return < 999 else 0)
    best_sharpe = max(results, key=lambda x: x.sharpe_ratio)

    print(f"  Most Pips:           {best_pips.threshold:.0%} ({best_pips.total_pips:+.0f} pips)")
    print(f"  Best Win Rate:       {best_wr.threshold:.0%} ({best_wr.win_rate:.1f}%)")
    print(f"  Best Profit Factor:  {best_pf.threshold:.0%} ({best_pf.profit_factor:.2f})")
    print(f"  Best Risk-Adjusted:  {best_rar.threshold:.0%} ({best_rar.risk_adjusted_return:.2f})")
    print(f"  Best Sharpe:         {best_sharpe.threshold:.0%} ({best_sharpe.sharpe_ratio:.2f})")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if optimal:
        print(f"\n  OPTIMAL THRESHOLD: {optimal.threshold:.0%}")
        print(f"\n  {reason}")
        print(f"\n  Comparison to Baseline (55%):")
        print(f"    Trades:      {baseline.total_trades} → {optimal.total_trades} "
              f"({optimal.total_trades - baseline.total_trades:+d})")
        print(f"    Win Rate:    {baseline.win_rate:.1f}% → {optimal.win_rate:.1f}% "
              f"({optimal.win_rate - baseline.win_rate:+.1f}%)")
        print(f"    Total Pips:  {baseline.total_pips:+.0f} → {optimal.total_pips:+.0f} "
              f"({optimal.total_pips - baseline.total_pips:+.0f})")
        print(f"    Profit Factor: {baseline.profit_factor:.2f} → {optimal.profit_factor:.2f} "
              f"({optimal.profit_factor - baseline.profit_factor:+.2f})")
        print(f"    Max Drawdown: {baseline.max_drawdown_pips:.0f} → {optimal.max_drawdown_pips:.0f} pips")

    # Save results
    output_dir = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "thresholds": thresholds,
            "data_file": args.data,
            "model_dir": args.model_dir,
        },
        "results": [
            {
                "threshold": r.threshold,
                "total_trades": r.total_trades,
                "winning_trades": r.winning_trades,
                "losing_trades": r.losing_trades,
                "win_rate": r.win_rate,
                "total_pips": r.total_pips,
                "avg_pips_per_trade": r.avg_pips_per_trade,
                "profit_factor": r.profit_factor,
                "max_drawdown_pips": r.max_drawdown_pips,
                "risk_adjusted_return": r.risk_adjusted_return,
                "sharpe_ratio": r.sharpe_ratio,
            }
            for r in results
        ],
        "optimal": {
            "threshold": optimal.threshold if optimal else None,
            "reason": reason,
        },
        "baseline_comparison": {
            "baseline_threshold": 0.55,
            "baseline_pips": baseline.total_pips,
            "optimal_threshold": optimal.threshold if optimal else None,
            "optimal_pips": optimal.total_pips if optimal else None,
            "pips_difference": optimal.total_pips - baseline.total_pips if optimal else None,
        }
    }

    results_path = output_dir / "confidence_optimization.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
