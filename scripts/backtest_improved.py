#!/usr/bin/env python3
"""Backtest improved multi-timeframe models.

This script evaluates the improved models using realistic
trading simulation with proper P&L calculation.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict
import warnings

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.models.multi_timeframe import (
    ImprovedModelConfig,
    ImprovedMultiTimeframeModel,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a completed trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    confidence: float
    pnl_pips: float
    exit_reason: str


class ImprovedBacktester:
    """Backtester for improved models."""

    def __init__(
        self,
        model: ImprovedMultiTimeframeModel,
        timeframe: str = "1H",
        min_confidence: float = 0.55,
        tp_pips: float = 25.0,
        sl_pips: float = 15.0,
        max_holding_bars: int = 12,
    ):
        self.model = model
        self.timeframe = timeframe
        self.min_confidence = min_confidence
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        self.max_holding_bars = max_holding_bars

        self.trades: List[Trade] = []

    def run(
        self,
        df_5min: pd.DataFrame,
        test_start_idx: int,
    ) -> Dict:
        """Run backtest on test portion of data."""
        # Resample to model timeframe
        rule = self.timeframe
        df_tf = df_5min.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in df_5min.columns else "first",
        }).dropna()

        # Calculate split point for resampled data
        test_start_time = df_5min.index[test_start_idx]
        df_test = df_tf[df_tf.index >= test_start_time].copy()

        logger.info(f"Test data: {len(df_test)} bars from {df_test.index[0]} to {df_test.index[-1]}")

        # Get model
        tf_model = self.model.models.get(self.timeframe)
        if not tf_model or not tf_model.is_trained:
            raise RuntimeError(f"Model {self.timeframe} not trained")

        # Prepare features for all test data at once
        from src.features.technical.calculator import TechnicalIndicatorCalculator
        calc = TechnicalIndicatorCalculator(model_type="short_term")

        # Need historical data for features
        lookback = 200  # Need historical bars for indicators
        df_with_history = df_tf.iloc[max(0, df_tf.index.get_loc(df_test.index[0]) - lookback):]

        df_features = calc.calculate(df_with_history)

        # Generate higher timeframe data for cross-TF features
        higher_tf_data = {}
        for htf in ["4H", "D"]:
            df_htf = df_5min.resample(htf).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum" if "volume" in df_5min.columns else "first",
            }).dropna()
            df_htf_features = calc.calculate(df_htf)
            higher_tf_data[htf] = df_htf_features

        df_features = tf_model.feature_engine.add_all_features(df_features, higher_tf_data)
        df_features = df_features.dropna()

        # Get feature columns
        feature_cols = tf_model.feature_names
        available_cols = [c for c in feature_cols if c in df_features.columns]

        if len(available_cols) < len(feature_cols) * 0.9:
            missing = set(feature_cols) - set(available_cols)
            logger.warning(f"Many missing features: {len(missing)}")
            logger.warning(f"Missing: {list(missing)[:10]}...")

        # Filter to test period
        df_test_features = df_features[df_features.index >= test_start_time]

        logger.info(f"Test features: {len(df_test_features)} rows, {len(available_cols)} features")

        # Get predictions for all test bars
        X = df_test_features[available_cols].values
        predictions, confidences = tf_model.predict_batch(X)

        # Simulate trading
        pip_value = 0.0001
        closes = df_test_features["close"].values
        highs = df_test_features["high"].values
        lows = df_test_features["low"].values
        timestamps = df_test_features.index

        n = len(X)
        i = 0

        while i < n - self.max_holding_bars:
            conf = confidences[i]
            pred = predictions[i]

            if conf >= self.min_confidence:
                entry_price = closes[i]
                entry_time = timestamps[i]
                direction = "long" if pred == 1 else "short"

                # Set barriers
                if direction == "long":
                    tp_price = entry_price + self.tp_pips * pip_value
                    sl_price = entry_price - self.sl_pips * pip_value
                else:
                    tp_price = entry_price - self.tp_pips * pip_value
                    sl_price = entry_price + self.sl_pips * pip_value

                # Simulate holding
                exit_price = None
                exit_reason = None

                for j in range(i + 1, min(i + self.max_holding_bars + 1, n)):
                    if direction == "long":
                        if highs[j] >= tp_price:
                            exit_price = tp_price
                            exit_reason = "take_profit"
                            break
                        if lows[j] <= sl_price:
                            exit_price = sl_price
                            exit_reason = "stop_loss"
                            break
                    else:  # short
                        if lows[j] <= tp_price:
                            exit_price = tp_price
                            exit_reason = "take_profit"
                            break
                        if highs[j] >= sl_price:
                            exit_price = sl_price
                            exit_reason = "stop_loss"
                            break

                # Time barrier
                if exit_price is None:
                    exit_idx = min(i + self.max_holding_bars, n - 1)
                    exit_price = closes[exit_idx]
                    exit_reason = "timeout"
                    j = exit_idx

                # Calculate PnL
                if direction == "long":
                    pnl_pips = (exit_price - entry_price) / pip_value
                else:
                    pnl_pips = (entry_price - exit_price) / pip_value

                trade = Trade(
                    entry_time=entry_time,
                    exit_time=timestamps[j],
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    confidence=conf,
                    pnl_pips=pnl_pips,
                    exit_reason=exit_reason,
                )
                self.trades.append(trade)

                # Skip to after exit
                i = j

            i += 1

        # Calculate results
        return self._calculate_results()

    def _calculate_results(self) -> Dict:
        """Calculate backtest metrics."""
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "total_pips": 0,
                "profit_factor": 0,
            }

        trades_df = pd.DataFrame([
            {
                "direction": t.direction,
                "confidence": t.confidence,
                "pnl_pips": t.pnl_pips,
                "exit_reason": t.exit_reason,
            }
            for t in self.trades
        ])

        wins = trades_df[trades_df["pnl_pips"] > 0]
        losses = trades_df[trades_df["pnl_pips"] <= 0]

        total_profit = wins["pnl_pips"].sum() if len(wins) > 0 else 0
        total_loss = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0

        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        # Breakdown by exit reason
        exit_breakdown = trades_df.groupby("exit_reason").agg({
            "pnl_pips": ["count", "sum", "mean"],
        }).to_dict()

        # Breakdown by confidence level
        high_conf = trades_df[trades_df["confidence"] >= 0.60]
        very_high_conf = trades_df[trades_df["confidence"] >= 0.65]

        # Win rate by direction
        longs = trades_df[trades_df["direction"] == "long"]
        shorts = trades_df[trades_df["direction"] == "short"]

        return {
            "total_trades": len(trades_df),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(trades_df) * 100,
            "total_pips": trades_df["pnl_pips"].sum(),
            "avg_pips": trades_df["pnl_pips"].mean(),
            "profit_factor": profit_factor,
            "avg_win": total_profit / len(wins) if len(wins) > 0 else 0,
            "avg_loss": total_loss / len(losses) if len(losses) > 0 else 0,
            # Breakdown
            "tp_hits": len(trades_df[trades_df["exit_reason"] == "take_profit"]),
            "sl_hits": len(trades_df[trades_df["exit_reason"] == "stop_loss"]),
            "timeouts": len(trades_df[trades_df["exit_reason"] == "timeout"]),
            # By confidence
            "high_conf_trades": len(high_conf),
            "high_conf_win_rate": len(high_conf[high_conf["pnl_pips"] > 0]) / len(high_conf) * 100 if len(high_conf) > 0 else 0,
            "very_high_conf_trades": len(very_high_conf),
            "very_high_conf_win_rate": len(very_high_conf[very_high_conf["pnl_pips"] > 0]) / len(very_high_conf) * 100 if len(very_high_conf) > 0 else 0,
            # By direction
            "long_trades": len(longs),
            "long_win_rate": len(longs[longs["pnl_pips"] > 0]) / len(longs) * 100 if len(longs) > 0 else 0,
            "short_trades": len(shorts),
            "short_win_rate": len(shorts[shorts["pnl_pips"] > 0]) / len(shorts) * 100 if len(shorts) > 0 else 0,
        }


def main():
    parser = argparse.ArgumentParser(description="Backtest Improved Models")
    parser.add_argument(
        "--data",
        type=str,
        default="data/forex/EURUSD_20200101_20251231_5min_combined.csv",
        help="Path to 5-minute data",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/improved_mtf",
        help="Model directory",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1H",
        help="Timeframe to backtest",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.55,
        help="Minimum confidence threshold",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Portion of data for testing",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("IMPROVED MODEL BACKTEST")
    print("=" * 70)
    print(f"Data:       {args.data}")
    print(f"Model dir:  {args.model_dir}")
    print(f"Timeframe:  {args.timeframe}")
    print(f"Confidence: {args.confidence:.0%}")
    print("=" * 70)

    # Load data
    data_path = project_root / args.data
    logger.info(f"Loading data from {data_path}")

    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]

    time_col = next((c for c in ["timestamp", "time", "date", "datetime"] if c in df.columns), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

    df = df.sort_index()
    logger.info(f"Loaded {len(df)} bars")

    # Load models
    model_dir = project_root / args.model_dir

    # Determine configs based on what's available
    configs = []
    if args.timeframe == "1H":
        configs.append(ImprovedModelConfig.hourly_model())
    elif args.timeframe == "4H":
        configs.append(ImprovedModelConfig.four_hour_model())
    elif args.timeframe == "D":
        configs.append(ImprovedModelConfig.daily_model())

    model = ImprovedMultiTimeframeModel(configs=configs, model_dir=model_dir)
    model.load_all()

    # Get model config for TP/SL settings
    tf_config = model.configs.get(args.timeframe)
    if tf_config:
        tp_pips = tf_config.tp_pips
        sl_pips = tf_config.sl_pips
        max_holding = tf_config.max_holding_bars
    else:
        tp_pips = 25.0
        sl_pips = 15.0
        max_holding = 12

    # Calculate test start
    test_start_idx = int(len(df) * (1 - args.test_ratio))

    # Run backtest
    backtester = ImprovedBacktester(
        model=model,
        timeframe=args.timeframe,
        min_confidence=args.confidence,
        tp_pips=tp_pips,
        sl_pips=sl_pips,
        max_holding_bars=max_holding,
    )

    results = backtester.run(df, test_start_idx)

    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    print(f"\nTrade Summary:")
    print(f"  Total Trades:     {results['total_trades']}")
    print(f"  Winning:          {results['winning_trades']}")
    print(f"  Losing:           {results['losing_trades']}")
    print(f"  Win Rate:         {results['win_rate']:.1f}%")

    print(f"\nP&L Summary:")
    print(f"  Total Pips:       {results['total_pips']:+.1f}")
    print(f"  Avg Pips/Trade:   {results['avg_pips']:+.1f}")
    print(f"  Profit Factor:    {results['profit_factor']:.2f}")
    print(f"  Avg Win:          {results['avg_win']:.1f} pips")
    print(f"  Avg Loss:         {results['avg_loss']:.1f} pips")

    print(f"\nExit Analysis:")
    print(f"  Take Profit:      {results['tp_hits']} ({results['tp_hits']/results['total_trades']*100:.1f}%)" if results['total_trades'] > 0 else "  Take Profit:      0")
    print(f"  Stop Loss:        {results['sl_hits']} ({results['sl_hits']/results['total_trades']*100:.1f}%)" if results['total_trades'] > 0 else "  Stop Loss:        0")
    print(f"  Timeout:          {results['timeouts']} ({results['timeouts']/results['total_trades']*100:.1f}%)" if results['total_trades'] > 0 else "  Timeout:          0")

    print(f"\nConfidence Analysis:")
    print(f"  Conf >= 60%:      {results['high_conf_trades']} trades, {results['high_conf_win_rate']:.1f}% win rate")
    print(f"  Conf >= 65%:      {results['very_high_conf_trades']} trades, {results['very_high_conf_win_rate']:.1f}% win rate")

    print(f"\nDirection Analysis:")
    print(f"  Long:             {results['long_trades']} trades, {results['long_win_rate']:.1f}% win rate")
    print(f"  Short:            {results['short_trades']} trades, {results['short_win_rate']:.1f}% win rate")

    print("\n" + "=" * 70)

    # Target analysis
    print("TARGET ANALYSIS:")
    if results['win_rate'] >= 55:
        print(f"  [OK] Win rate {results['win_rate']:.1f}% meets 55% target")
    else:
        print(f"  [!!] Win rate {results['win_rate']:.1f}% below 55% target")

    if results['profit_factor'] >= 1.5:
        print(f"  [OK] Profit factor {results['profit_factor']:.2f} meets 1.5 target")
    elif results['profit_factor'] >= 1.0:
        print(f"  [OK] Profit factor {results['profit_factor']:.2f} is profitable (target 1.5)")
    else:
        print(f"  [!!] Profit factor {results['profit_factor']:.2f} below 1.0 (losing)")

    print("=" * 70)

    # Test different confidence thresholds
    print("\nCONFIDENCE THRESHOLD SWEEP:")
    print("-" * 50)

    for conf in [0.55, 0.57, 0.60, 0.63, 0.65]:
        bt = ImprovedBacktester(
            model=model,
            timeframe=args.timeframe,
            min_confidence=conf,
            tp_pips=tp_pips,
            sl_pips=sl_pips,
            max_holding_bars=max_holding,
        )
        r = bt.run(df, test_start_idx)

        if r['total_trades'] > 0:
            print(f"Conf >= {conf:.0%}: {r['total_trades']:4d} trades, "
                  f"Win: {r['win_rate']:5.1f}%, "
                  f"Pips: {r['total_pips']:+7.1f}, "
                  f"PF: {r['profit_factor']:.2f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
