#!/usr/bin/env python3
"""Optimize multi-timeframe backtest parameters."""

import argparse
import logging
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.models.multi_timeframe import (
    MultiTimeframeModel,
    TimeframeConfig,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_all_signals(model, df_5min, min_confidence=0.65):
    """Compute signals for all bars at once."""
    from src.features.technical.calculator import TechnicalIndicatorCalculator

    feature_calc = TechnicalIndicatorCalculator(model_type="short_term")
    signals_dfs = []

    for tf_name, config in model.configs.items():
        if config.resample_rule != "5min":
            resampled = df_5min.resample(config.resample_rule).agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()
        else:
            resampled = df_5min.copy()

        features = feature_calc.calculate(resampled).dropna()

        tf_model = model.models.get(tf_name)
        if tf_model is None or not tf_model.is_trained:
            continue

        available_features = [f for f in tf_model.feature_names if f in features.columns]
        X = features[available_features].values

        X_scaled = tf_model.scaler.transform(X)
        probs = tf_model.model.predict_proba(X_scaled)
        preds = tf_model.model.predict(X_scaled)
        confidence = np.max(probs, axis=1)

        signals = pd.DataFrame({
            f"{tf_name}_direction": preds,
            f"{tf_name}_confidence": confidence,
        }, index=features.index)

        signals_dfs.append(signals)

    combined = df_5min[["close"]].copy()
    for signals_df in signals_dfs:
        combined = combined.join(signals_df, how="left")
        for col in signals_df.columns:
            combined[col] = combined[col].ffill()

    return combined.dropna()


def backtest_signals(signals_df, min_confidence, sl_pips, tp_pips, require_confirmation, require_trend):
    """Fast backtest."""
    primary_conf = signals_df["5min_confidence"]
    primary_dir = signals_df["5min_direction"]
    high_conf_mask = primary_conf >= min_confidence

    if require_confirmation and "15min_direction" in signals_df.columns:
        confirm_aligned = primary_dir == signals_df["15min_direction"]
    else:
        confirm_aligned = pd.Series(True, index=signals_df.index)

    if require_trend and "30min_direction" in signals_df.columns:
        trend_aligned = primary_dir == signals_df["30min_direction"]
    else:
        trend_aligned = pd.Series(True, index=signals_df.index)

    signal_mask = high_conf_mask & confirm_aligned & trend_aligned

    closes = signals_df["close"].values
    directions = primary_dir.values
    confidences = primary_conf.values
    signals = signal_mask.values

    trades = []
    i = 0
    n = len(closes)
    pip_value = 0.0001

    while i < n:
        if signals[i]:
            entry_price = closes[i]
            direction = int(directions[i])
            confidence = confidences[i]

            if direction == 1:
                sl = entry_price - sl_pips * pip_value
                tp = entry_price + tp_pips * pip_value
            else:
                sl = entry_price + sl_pips * pip_value
                tp = entry_price - tp_pips * pip_value

            exit_price = None
            exit_reason = None
            max_bars = 24

            for j in range(i + 1, min(i + max_bars + 1, n)):
                price = closes[j]
                if direction == 1:
                    if price <= sl:
                        exit_price = sl
                        exit_reason = "sl"
                        break
                    elif price >= tp:
                        exit_price = tp
                        exit_reason = "tp"
                        break
                else:
                    if price >= sl:
                        exit_price = sl
                        exit_reason = "sl"
                        break
                    elif price <= tp:
                        exit_price = tp
                        exit_reason = "tp"
                        break

            if exit_price is None:
                exit_price = closes[min(i + max_bars, n - 1)]
                exit_reason = "timeout"

            if direction == 1:
                pnl_pips = (exit_price - entry_price) / pip_value
            else:
                pnl_pips = (entry_price - exit_price) / pip_value

            trades.append({
                "direction": direction,
                "confidence": confidence,
                "pnl_pips": pnl_pips,
                "exit_reason": exit_reason,
            })

            i = min(i + max_bars, n - 1)
        i += 1

    if not trades:
        return {"total_trades": 0, "win_rate": 0, "total_pips": 0, "profit_factor": 0}

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["pnl_pips"] > 0]
    losses = trades_df[trades_df["pnl_pips"] <= 0]

    total_profit = wins["pnl_pips"].sum() if len(wins) > 0 else 0
    total_loss = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

    return {
        "total_trades": len(trades_df),
        "win_rate": len(wins) / len(trades_df) * 100,
        "total_pips": trades_df["pnl_pips"].sum(),
        "profit_factor": profit_factor,
        "avg_pips": trades_df["pnl_pips"].mean(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/forex/EURUSD_20200101_20251231_5min_combined.csv")
    parser.add_argument("--model-dir", type=str, default="models/mtf_scalper")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("MTF BACKTEST OPTIMIZATION")
    print("=" * 70)

    # Load data
    data_path = project_root / args.data
    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]

    time_col = next((c for c in ["timestamp", "time", "date", "datetime"] if c in df.columns), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

    df = df.sort_index()
    test_start_idx = int(len(df) * (1 - args.test_ratio))
    test_df = df.iloc[test_start_idx:]

    print(f"Test period: {test_df.index[0]} to {test_df.index[-1]}")
    print(f"Test bars: {len(test_df)}")

    # Load models
    model_dir = project_root / args.model_dir
    configs = [
        TimeframeConfig.scalper_5min(),
        TimeframeConfig.scalper_15min(),
        TimeframeConfig.scalper_30min(),
    ]
    mtf_model = MultiTimeframeModel(configs=configs, model_dir=model_dir)
    mtf_model.load_all()

    print("\nComputing signals...")
    signals_df = compute_all_signals(mtf_model, test_df, 0.60)
    print(f"Signal rows: {len(signals_df)}")

    # Optimization grid
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)
    print(f"{'Config':<35} {'Trades':>7} {'Win%':>7} {'Pips':>9} {'PF':>6}")
    print("-" * 70)

    results = []

    # Test configurations
    for conf in [0.65, 0.68, 0.70, 0.72, 0.75]:
        for req_conf in [True, False]:
            for req_trend in [True, False]:
                for sl in [8, 10, 12, 15]:
                    for tp in [15, 20, 25, 30]:
                        if tp < sl * 1.5:  # Skip unreasonable R:R
                            continue

                        r = backtest_signals(
                            signals_df,
                            min_confidence=conf,
                            sl_pips=sl,
                            tp_pips=tp,
                            require_confirmation=req_conf,
                            require_trend=req_trend,
                        )

                        if r["total_trades"] > 100:  # Minimum trades
                            results.append({
                                "conf": conf,
                                "req_conf": req_conf,
                                "req_trend": req_trend,
                                "sl": sl,
                                "tp": tp,
                                **r,
                            })

    # Sort by profit factor
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values("profit_factor", ascending=False)

        print("\nTop 20 configurations by Profit Factor:")
        print("-" * 70)
        for i, row in results_df.head(20).iterrows():
            conf_str = f"conf={row['conf']:.0%}"
            mtf_str = f"{'MTF' if row['req_conf'] else 'NO-MTF'}"
            trend_str = f"{'TREND' if row['req_trend'] else 'NO-TREND'}"
            sltp_str = f"SL{row['sl']}/TP{row['tp']}"

            config_name = f"{conf_str} {mtf_str} {trend_str} {sltp_str}"
            print(f"{config_name:<35} {row['total_trades']:>7} {row['win_rate']:>6.1f}% "
                  f"{row['total_pips']:>+8.0f} {row['profit_factor']:>5.2f}")

        print("\n" + "-" * 70)
        print("\nTop 10 by Win Rate (min PF 1.0):")
        profitable = results_df[results_df["profit_factor"] >= 1.0]
        if len(profitable) > 0:
            profitable = profitable.sort_values("win_rate", ascending=False)
            for i, row in profitable.head(10).iterrows():
                conf_str = f"conf={row['conf']:.0%}"
                mtf_str = f"{'MTF' if row['req_conf'] else 'NO-MTF'}"
                trend_str = f"{'TREND' if row['req_trend'] else 'NO-TREND'}"
                sltp_str = f"SL{row['sl']}/TP{row['tp']}"

                config_name = f"{conf_str} {mtf_str} {trend_str} {sltp_str}"
                print(f"{config_name:<35} {row['total_trades']:>7} {row['win_rate']:>6.1f}% "
                      f"{row['total_pips']:>+8.0f} {row['profit_factor']:>5.2f}")

        # Find best balanced config
        print("\n" + "-" * 70)
        print("\nBest balanced configs (PF >= 1.1, Win >= 45%, Trades >= 500):")
        balanced = results_df[(results_df["profit_factor"] >= 1.1) &
                              (results_df["win_rate"] >= 45) &
                              (results_df["total_trades"] >= 500)]
        if len(balanced) > 0:
            balanced = balanced.sort_values("total_pips", ascending=False)
            for i, row in balanced.head(5).iterrows():
                conf_str = f"conf={row['conf']:.0%}"
                mtf_str = f"{'MTF' if row['req_conf'] else 'NO-MTF'}"
                trend_str = f"{'TREND' if row['req_trend'] else 'NO-TREND'}"
                sltp_str = f"SL{row['sl']}/TP{row['tp']}"

                config_name = f"{conf_str} {mtf_str} {trend_str} {sltp_str}"
                print(f"{config_name:<35} {row['total_trades']:>7} {row['win_rate']:>6.1f}% "
                      f"{row['total_pips']:>+8.0f} {row['profit_factor']:>5.2f}")
        else:
            print("No configs meet balanced criteria")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
