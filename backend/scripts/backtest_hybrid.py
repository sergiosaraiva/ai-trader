#!/usr/bin/env python3
"""Backtest hybrid ensemble model.

Compares hybrid ensemble performance against XGBoost baseline.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import warnings
warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.models.multi_timeframe import (
    HybridConfig,
    HybridEnsemble,
    SequenceDataset,
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
    direction: str
    entry_price: float
    exit_price: float
    confidence: float
    pnl_pips: float
    exit_reason: str


class HybridBacktester:
    """Backtester for hybrid ensemble model."""

    def __init__(
        self,
        ensemble: HybridEnsemble,
        min_confidence: float = 0.55,
    ):
        self.ensemble = ensemble
        self.min_confidence = min_confidence
        self.trades: List[Trade] = []

        config = ensemble.config
        self.timeframe = config.base_timeframe
        self.tp_pips = config.xgboost_config.tp_pips
        self.sl_pips = config.xgboost_config.sl_pips
        self.max_holding_bars = config.xgboost_config.max_holding_bars
        self.seq_length = config.sequence_config.sequence_length

    def run(
        self,
        df_5min: pd.DataFrame,
        test_start_idx: int,
    ) -> Dict:
        """Run backtest."""
        # Resample
        df_tf = df_5min.resample(self.timeframe).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in df_5min.columns else "first",
        }).dropna()

        test_start_time = df_5min.index[test_start_idx]
        df_test = df_tf[df_tf.index >= test_start_time].copy()

        logger.info(f"Test: {len(df_test)} bars from {df_test.index[0]} to {df_test.index[-1]}")

        # Prepare XGBoost features
        from src.features.technical.calculator import TechnicalIndicatorCalculator
        calc = TechnicalIndicatorCalculator(model_type="short_term")

        lookback = 200
        start_loc = max(0, df_tf.index.get_loc(df_test.index[0]) - lookback)
        df_with_history = df_tf.iloc[start_loc:]

        # Technical features
        df_features = calc.calculate(df_with_history)

        # Higher TF data
        higher_tf_data = {}
        for htf in ["4H", "D"]:
            df_htf = df_5min.resample(htf).agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum" if "volume" in df_5min.columns else "first",
            }).dropna()
            higher_tf_data[htf] = calc.calculate(df_htf)

        # Add enhanced features
        xgb_model = self.ensemble.xgboost_model
        df_features = xgb_model.feature_engine.add_all_features(df_features, higher_tf_data)
        df_features = df_features.dropna()

        # Filter to test period
        df_test_features = df_features[df_features.index >= test_start_time]

        # Get XGBoost feature matrix
        feature_cols = xgb_model.feature_names
        available_cols = [c for c in feature_cols if c in df_test_features.columns]
        X_xgb = df_test_features[available_cols].values

        # Prepare sequence data
        seq_dataset = SequenceDataset(self.ensemble.config.sequence_config)
        df_norm = seq_dataset._normalize_ohlcv(df_with_history)

        # Get predictions
        logger.info("Getting predictions...")

        xgb_preds, xgb_confs = xgb_model.predict_batch(X_xgb)

        # Prepare sequences for all test bars
        sequences = []
        for i in range(len(df_test_features)):
            # Get corresponding index in df_norm
            time = df_test_features.index[i]
            if time in df_norm.index:
                loc = df_norm.index.get_loc(time)
                if loc >= self.seq_length:
                    seq = df_norm.iloc[loc - self.seq_length:loc][['open', 'high', 'low', 'close', 'volume']].values
                    if not np.isnan(seq).any():
                        sequences.append(seq)
                    else:
                        sequences.append(np.zeros((self.seq_length, 5)))
                else:
                    sequences.append(np.zeros((self.seq_length, 5)))
            else:
                sequences.append(np.zeros((self.seq_length, 5)))

        X_seq = np.array(sequences, dtype=np.float32)
        seq_preds, seq_confs = self.ensemble.sequence_model.predict_batch(X_seq)

        # Ensemble predictions
        ensemble_preds, ensemble_confs = self.ensemble.predict_batch(X_xgb, X_seq)

        # Simulate trading
        pip_value = 0.0001
        closes = df_test_features["close"].values
        highs = df_test_features["high"].values
        lows = df_test_features["low"].values
        timestamps = df_test_features.index

        n = len(ensemble_preds)
        i = 0

        while i < n - self.max_holding_bars:
            conf = ensemble_confs[i]
            pred = ensemble_preds[i]

            if conf >= self.min_confidence:
                entry_price = closes[i]
                entry_time = timestamps[i]
                direction = "long" if pred == 1 else "short"

                if direction == "long":
                    tp_price = entry_price + self.tp_pips * pip_value
                    sl_price = entry_price - self.sl_pips * pip_value
                else:
                    tp_price = entry_price - self.tp_pips * pip_value
                    sl_price = entry_price + self.sl_pips * pip_value

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
                    else:
                        if lows[j] <= tp_price:
                            exit_price = tp_price
                            exit_reason = "take_profit"
                            break
                        if highs[j] >= sl_price:
                            exit_price = sl_price
                            exit_reason = "stop_loss"
                            break

                if exit_price is None:
                    exit_idx = min(i + self.max_holding_bars, n - 1)
                    exit_price = closes[exit_idx]
                    exit_reason = "timeout"
                    j = exit_idx

                if direction == "long":
                    pnl_pips = (exit_price - entry_price) / pip_value
                else:
                    pnl_pips = (entry_price - exit_price) / pip_value

                self.trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=timestamps[j],
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    confidence=conf,
                    pnl_pips=pnl_pips,
                    exit_reason=exit_reason,
                ))

                i = j

            i += 1

        return self._calculate_results()

    def _calculate_results(self) -> Dict:
        """Calculate backtest metrics."""
        if not self.trades:
            return {"total_trades": 0, "win_rate": 0, "total_pips": 0, "profit_factor": 0}

        trades_df = pd.DataFrame([{
            "direction": t.direction,
            "confidence": t.confidence,
            "pnl_pips": t.pnl_pips,
            "exit_reason": t.exit_reason,
        } for t in self.trades])

        wins = trades_df[trades_df["pnl_pips"] > 0]
        losses = trades_df[trades_df["pnl_pips"] <= 0]

        total_profit = wins["pnl_pips"].sum() if len(wins) > 0 else 0
        total_loss = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float("inf")

        high_conf = trades_df[trades_df["confidence"] >= 0.60]
        very_high_conf = trades_df[trades_df["confidence"] >= 0.65]

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
            "tp_hits": len(trades_df[trades_df["exit_reason"] == "take_profit"]),
            "sl_hits": len(trades_df[trades_df["exit_reason"] == "stop_loss"]),
            "timeouts": len(trades_df[trades_df["exit_reason"] == "timeout"]),
            "high_conf_trades": len(high_conf),
            "high_conf_win_rate": len(high_conf[high_conf["pnl_pips"] > 0]) / len(high_conf) * 100 if len(high_conf) > 0 else 0,
            "very_high_conf_trades": len(very_high_conf),
            "very_high_conf_win_rate": len(very_high_conf[very_high_conf["pnl_pips"] > 0]) / len(very_high_conf) * 100 if len(very_high_conf) > 0 else 0,
        }


def main():
    parser = argparse.ArgumentParser(description="Backtest Hybrid Ensemble")
    parser.add_argument("--data", type=str, default="data/forex/EURUSD_20200101_20251231_5min_combined.csv")
    parser.add_argument("--model-dir", type=str, default="models/hybrid")
    parser.add_argument("--timeframe", type=str, default="1H")
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("HYBRID ENSEMBLE BACKTEST")
    print("=" * 70)
    print(f"Data:       {args.data}")
    print(f"Model dir:  {args.model_dir}")
    print(f"Timeframe:  {args.timeframe}")
    print(f"Confidence: {args.confidence:.0%}")
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

    logger.info(f"Loaded {len(df)} bars")

    # Load ensemble
    model_dir = project_root / args.model_dir

    if args.timeframe == "1H":
        config = HybridConfig.hourly()
    elif args.timeframe == "4H":
        config = HybridConfig.four_hour()
    else:
        raise ValueError(f"Unknown timeframe: {args.timeframe}")

    ensemble = HybridEnsemble(config=config, model_dir=model_dir)
    ensemble.load()

    # Run backtest
    test_start_idx = int(len(df) * (1 - args.test_ratio))

    backtester = HybridBacktester(ensemble=ensemble, min_confidence=args.confidence)
    results = backtester.run(df, test_start_idx)

    # Print results
    print("\n" + "=" * 70)
    print("HYBRID ENSEMBLE BACKTEST RESULTS")
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
    if results['total_trades'] > 0:
        print(f"  Take Profit:      {results['tp_hits']} ({results['tp_hits']/results['total_trades']*100:.1f}%)")
        print(f"  Stop Loss:        {results['sl_hits']} ({results['sl_hits']/results['total_trades']*100:.1f}%)")
        print(f"  Timeout:          {results['timeouts']} ({results['timeouts']/results['total_trades']*100:.1f}%)")

    print(f"\nConfidence Analysis:")
    print(f"  Conf >= 60%:      {results['high_conf_trades']} trades, {results['high_conf_win_rate']:.1f}% win rate")
    print(f"  Conf >= 65%:      {results['very_high_conf_trades']} trades, {results['very_high_conf_win_rate']:.1f}% win rate")

    print("\n" + "=" * 70)
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
        print(f"  [!!] Profit factor {results['profit_factor']:.2f} below 1.0")

    print("=" * 70)


if __name__ == "__main__":
    main()
