#!/usr/bin/env python3
"""Vectorized multi-timeframe backtest for fast iteration.

This version computes all signals upfront then vectorizes the P&L calculation.
"""

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
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_all_signals(
    model: MultiTimeframeModel,
    df_5min: pd.DataFrame,
    min_confidence: float = 0.65,
) -> pd.DataFrame:
    """Compute signals for all bars at once."""

    feature_calc = model.feature_calculator
    signals_dfs = []

    for tf_name, config in model.configs.items():
        logger.info(f"Computing signals for {tf_name}...")

        # Resample data
        if config.resample_rule != "5min":
            resampled = df_5min.resample(config.resample_rule).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()
        else:
            resampled = df_5min.copy()

        # Calculate features
        features = feature_calc.calculate(resampled)
        features = features.dropna()

        # Get model
        tf_model = model.models.get(tf_name)
        if tf_model is None or not tf_model.is_trained:
            continue

        # Only keep features the model expects
        available_features = [f for f in tf_model.feature_names if f in features.columns]
        X = features[available_features].values

        # Get predictions for all rows at once
        X_scaled = tf_model.scaler.transform(X)
        probs = tf_model.model.predict_proba(X_scaled)
        preds = tf_model.model.predict(X_scaled)
        confidence = np.max(probs, axis=1)

        # Create signals dataframe
        signals = pd.DataFrame({
            f"{tf_name}_direction": preds,
            f"{tf_name}_confidence": confidence,
            f"{tf_name}_prob_up": probs[:, 1] if probs.shape[1] == 2 else probs[:, 0],
        }, index=features.index)

        signals_dfs.append(signals)

    # Forward fill signals to 5min timeframe
    combined = df_5min[["close"]].copy()
    for signals_df in signals_dfs:
        combined = combined.join(signals_df, how="left")
        # Forward fill from higher timeframes
        for col in signals_df.columns:
            combined[col] = combined[col].ffill()

    combined = combined.dropna()
    logger.info(f"Combined signals: {len(combined)} rows")

    return combined


def backtest_signals(
    signals_df: pd.DataFrame,
    min_confidence: float = 0.65,
    sl_pips: float = 10,
    tp_pips: float = 20,
    require_confirmation: bool = True,
    require_trend: bool = True,
) -> dict:
    """Fast vectorized backtest."""

    logger.info("Running vectorized backtest...")

    # Get primary signals that meet confidence
    primary_conf = signals_df["5min_confidence"]
    primary_dir = signals_df["5min_direction"]
    high_conf_mask = primary_conf >= min_confidence

    # Check confirmation alignment
    if require_confirmation and "15min_direction" in signals_df.columns:
        confirm_dir = signals_df["15min_direction"]
        confirm_aligned = primary_dir == confirm_dir
    else:
        confirm_aligned = pd.Series(True, index=signals_df.index)

    # Check trend alignment
    if require_trend and "30min_direction" in signals_df.columns:
        trend_dir = signals_df["30min_direction"]
        trend_aligned = primary_dir == trend_dir
    else:
        trend_aligned = pd.Series(True, index=signals_df.index)

    # Combined signal mask
    signal_mask = high_conf_mask & confirm_aligned & trend_aligned

    # Get close prices
    closes = signals_df["close"].values
    directions = primary_dir.values
    confidences = primary_conf.values
    signals = signal_mask.values

    # Simulate trades
    trades = []
    i = 0
    n = len(closes)
    pip_value = 0.0001  # For EUR/USD

    while i < n:
        if signals[i]:
            entry_price = closes[i]
            entry_idx = i
            direction = int(directions[i])
            confidence = confidences[i]

            # Set SL and TP
            if direction == 1:  # Long
                sl = entry_price - sl_pips * pip_value
                tp = entry_price + tp_pips * pip_value
            else:  # Short
                sl = entry_price + sl_pips * pip_value
                tp = entry_price - tp_pips * pip_value

            # Simulate holding the position
            exit_price = None
            exit_reason = None
            max_bars = 24  # 2 hours timeout

            for j in range(i + 1, min(i + max_bars + 1, n)):
                price = closes[j]

                if direction == 1:  # Long
                    if price <= sl:
                        exit_price = sl
                        exit_reason = "stop_loss"
                        break
                    elif price >= tp:
                        exit_price = tp
                        exit_reason = "take_profit"
                        break
                else:  # Short
                    if price >= sl:
                        exit_price = sl
                        exit_reason = "stop_loss"
                        break
                    elif price <= tp:
                        exit_price = tp
                        exit_reason = "take_profit"
                        break

            if exit_price is None:
                # Timeout - exit at current price
                exit_price = closes[min(i + max_bars, n - 1)]
                exit_reason = "timeout"

            # Calculate PnL
            if direction == 1:
                pnl_pips = (exit_price - entry_price) / pip_value
            else:
                pnl_pips = (entry_price - exit_price) / pip_value

            trades.append({
                "entry_idx": entry_idx,
                "direction": "long" if direction == 1 else "short",
                "confidence": confidence,
                "pnl_pips": pnl_pips,
                "exit_reason": exit_reason,
            })

            # Skip to after exit
            i = min(i + max_bars, n - 1)

        i += 1

    # Calculate metrics
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "avg_pips": 0,
            "total_pips": 0,
            "profit_factor": 0,
            "high_conf_signals": high_conf_mask.sum(),
            "confirmed_signals": (high_conf_mask & confirm_aligned).sum(),
            "final_signals": signal_mask.sum(),
        }

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["pnl_pips"] > 0]
    losses = trades_df[trades_df["pnl_pips"] <= 0]

    total_profit_pips = wins["pnl_pips"].sum() if len(wins) > 0 else 0
    total_loss_pips = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0

    profit_factor = total_profit_pips / total_loss_pips if total_loss_pips > 0 else float("inf")

    # Breakdown by exit reason
    exit_breakdown = trades_df.groupby("exit_reason").size().to_dict()

    # Breakdown by confidence level
    high_conf_trades = trades_df[trades_df["confidence"] >= 0.70]
    med_conf_trades = trades_df[(trades_df["confidence"] >= 0.65) & (trades_df["confidence"] < 0.70)]

    return {
        "total_trades": len(trades_df),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(trades_df) * 100,
        "avg_pips": trades_df["pnl_pips"].mean(),
        "total_pips": trades_df["pnl_pips"].sum(),
        "profit_factor": profit_factor,
        "exit_breakdown": exit_breakdown,
        "high_conf_signals": int(high_conf_mask.sum()),
        "confirmed_signals": int((high_conf_mask & confirm_aligned).sum()),
        "final_signals": int(signal_mask.sum()),
        "high_conf_70_trades": len(high_conf_trades),
        "high_conf_70_win_rate": len(high_conf_trades[high_conf_trades["pnl_pips"] > 0]) / len(high_conf_trades) * 100 if len(high_conf_trades) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Vectorized MTF Backtest")
    parser.add_argument(
        "--data",
        type=str,
        default="data/forex/EURUSD_20200101_20251231_5min_combined.csv",
        help="Path to 5-minute data",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/mtf_scalper",
        help="Model directory",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.65,
        help="Minimum confidence threshold",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Portion of data to use for testing",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("VECTORIZED MULTI-TIMEFRAME BACKTEST")
    print("=" * 70)

    # Load data
    data_path = project_root / args.data
    logger.info(f"Loading data from {data_path}")

    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]

    time_col = None
    for col in ["timestamp", "time", "date", "datetime"]:
        if col in df.columns:
            time_col = col
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

    df = df.sort_index()
    logger.info(f"Loaded {len(df)} bars")

    # Split data
    test_start_idx = int(len(df) * (1 - args.test_ratio))
    test_df = df.iloc[test_start_idx:]
    logger.info(f"Test period: {test_df.index[0]} to {test_df.index[-1]}")

    # Load models
    model_dir = project_root / args.model_dir
    configs = [
        TimeframeConfig.scalper_5min(),
        TimeframeConfig.scalper_15min(),
        TimeframeConfig.scalper_30min(),
    ]

    mtf_model = MultiTimeframeModel(configs=configs, model_dir=model_dir)
    mtf_model.load_all()

    # Compute signals for test data
    signals_df = compute_all_signals(mtf_model, test_df, args.confidence)

    # Run different configurations
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    configs = [
        {"name": "Full MTF (conf + confirm + trend)", "require_confirmation": True, "require_trend": True},
        {"name": "No trend filter (conf + confirm)", "require_confirmation": True, "require_trend": False},
        {"name": "Primary only (high conf only)", "require_confirmation": False, "require_trend": False},
    ]

    for cfg in configs:
        print(f"\n{cfg['name']}:")
        print("-" * 50)

        results = backtest_signals(
            signals_df,
            min_confidence=args.confidence,
            sl_pips=10,
            tp_pips=20,
            require_confirmation=cfg["require_confirmation"],
            require_trend=cfg["require_trend"],
        )

        print(f"  High conf signals:  {results['high_conf_signals']:,}")
        if cfg["require_confirmation"]:
            print(f"  Confirmed signals:  {results['confirmed_signals']:,}")
        print(f"  Final signals:      {results['final_signals']:,}")
        print(f"  Total trades:       {results['total_trades']}")
        print(f"  Win rate:           {results['win_rate']:.1f}%")
        print(f"  Avg pips:           {results['avg_pips']:.1f}")
        print(f"  Total pips:         {results['total_pips']:.1f}")
        print(f"  Profit factor:      {results['profit_factor']:.2f}")

        if results.get("exit_breakdown"):
            print(f"  Exit breakdown:")
            for reason, count in results["exit_breakdown"].items():
                pct = count / results["total_trades"] * 100
                print(f"    {reason}: {count} ({pct:.1f}%)")

    # Test different confidence thresholds
    print("\n" + "=" * 70)
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("=" * 70)

    for conf in [0.65, 0.68, 0.70, 0.72, 0.75]:
        results = backtest_signals(
            signals_df,
            min_confidence=conf,
            sl_pips=10,
            tp_pips=20,
            require_confirmation=True,
            require_trend=True,
        )
        print(f"Conf >= {conf:.0%}: {results['total_trades']:4d} trades, "
              f"Win: {results['win_rate']:5.1f}%, "
              f"Pips: {results['total_pips']:+7.1f}, "
              f"PF: {results['profit_factor']:.2f}")

    # Test different SL/TP
    print("\n" + "=" * 70)
    print("STOP LOSS / TAKE PROFIT ANALYSIS")
    print("=" * 70)

    for sl, tp in [(5, 10), (10, 15), (10, 20), (10, 30), (15, 30)]:
        results = backtest_signals(
            signals_df,
            min_confidence=args.confidence,
            sl_pips=sl,
            tp_pips=tp,
            require_confirmation=True,
            require_trend=True,
        )
        print(f"SL={sl:2d} TP={tp:2d}: {results['total_trades']:4d} trades, "
              f"Win: {results['win_rate']:5.1f}%, "
              f"Pips: {results['total_pips']:+7.1f}, "
              f"PF: {results['profit_factor']:.2f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
