#!/usr/bin/env python3
"""Backtest enhanced hybrid model."""

import argparse
import json
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

from src.models.multi_timeframe import ImprovedModelConfig, ImprovedTimeframeModel
from src.models.multi_timeframe.enhanced_sequence_model import (
    EnhancedSequenceConfig,
    EnhancedSequencePredictor,
    DEFAULT_SEQUENCE_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    confidence: float
    pnl_pips: float
    exit_reason: str


class EnhancedHybridBacktester:
    def __init__(
        self,
        xgb_model: ImprovedTimeframeModel,
        seq_predictor: EnhancedSequencePredictor,
        config: dict,
        min_confidence: float = 0.55,
    ):
        self.xgb_model = xgb_model
        self.seq_predictor = seq_predictor
        self.config = config
        self.min_confidence = min_confidence
        self.trades: List[Trade] = []

        self.timeframe = config.get('timeframe', '1H')
        self.tp_pips = 25.0 if self.timeframe == '1H' else 50.0
        self.sl_pips = 15.0 if self.timeframe == '1H' else 25.0
        self.max_holding_bars = 12 if self.timeframe == '1H' else 18
        self.seq_length = 30

    def run(self, df_5min: pd.DataFrame, test_start_idx: int) -> Dict:
        from src.features.technical.calculator import TechnicalIndicatorCalculator
        from src.models.multi_timeframe import EnhancedFeatureEngine

        # Resample
        df_tf = df_5min.resample(self.timeframe).agg({
            "open": "first", "high": "max", "low": "min", "close": "last",
            "volume": "sum" if "volume" in df_5min.columns else "first",
        }).dropna()

        test_start_time = df_5min.index[test_start_idx]
        df_test = df_tf[df_tf.index >= test_start_time].copy()
        logger.info(f"Test: {len(df_test)} bars from {df_test.index[0]} to {df_test.index[-1]}")

        # Prepare features
        calc = TechnicalIndicatorCalculator(model_type="short_term")

        lookback = 200
        start_loc = max(0, df_tf.index.get_loc(df_test.index[0]) - lookback)
        df_with_history = df_tf.iloc[start_loc:]

        df_features = calc.calculate(df_with_history)

        # Higher TF data
        higher_tf_data = {}
        for htf in ["4H", "D"]:
            df_htf = df_5min.resample(htf).agg({
                "open": "first", "high": "max", "low": "min", "close": "last",
                "volume": "sum" if "volume" in df_5min.columns else "first",
            }).dropna()
            higher_tf_data[htf] = calc.calculate(df_htf)

        feature_engine = EnhancedFeatureEngine(base_timeframe=self.timeframe)
        df_features = feature_engine.add_all_features(df_features, higher_tf_data)
        df_features = df_features.dropna()

        df_test_features = df_features[df_features.index >= test_start_time]

        # XGBoost features
        xgb_feature_cols = self.xgb_model.feature_names
        available_xgb_cols = [c for c in xgb_feature_cols if c in df_test_features.columns]
        X_xgb = df_test_features[available_xgb_cols].values

        # Sequence features
        seq_feature_names = self.seq_predictor.feature_names
        available_seq_cols = [c for c in seq_feature_names if c in df_features.columns]

        # Prepare sequences
        sequences = []
        for i in range(len(df_test_features)):
            time = df_test_features.index[i]
            if time in df_features.index:
                loc = df_features.index.get_loc(time)
                if loc >= self.seq_length:
                    seq_data = df_features.iloc[loc - self.seq_length:loc][available_seq_cols].values
                    seq_data = pd.DataFrame(seq_data).ffill().bfill().values
                    seq_scaled = self.seq_predictor.scaler.transform(seq_data)
                    seq_scaled = np.clip(seq_scaled, -5, 5)
                    sequences.append(seq_scaled)
                else:
                    sequences.append(np.zeros((self.seq_length, len(available_seq_cols))))
            else:
                sequences.append(np.zeros((self.seq_length, len(available_seq_cols))))

        X_seq = np.array(sequences, dtype=np.float32)

        logger.info("Getting predictions...")

        # Get predictions
        xgb_preds, xgb_confs = self.xgb_model.predict_batch(X_xgb)
        seq_preds, seq_confs = self.seq_predictor.predict_batch(X_seq)

        # Ensemble
        n = min(len(xgb_preds), len(seq_preds))
        xgb_preds, xgb_confs = xgb_preds[:n], xgb_confs[:n]
        seq_preds, seq_confs = seq_preds[:n], seq_confs[:n]

        total_conf = xgb_confs + seq_confs
        xgb_w = xgb_confs / total_conf
        seq_w = seq_confs / total_conf

        xgb_prob = xgb_preds * xgb_confs + (1 - xgb_preds) * (1 - xgb_confs)
        seq_prob = seq_preds * seq_confs + (1 - seq_preds) * (1 - seq_confs)

        ensemble_prob = xgb_w * xgb_prob + seq_w * seq_prob
        ensemble_preds = (ensemble_prob > 0.5).astype(int)
        ensemble_confs = np.abs(ensemble_prob - 0.5) * 2 + 0.5

        agreement = (xgb_preds == seq_preds).astype(float)
        ensemble_confs = np.minimum(ensemble_confs + agreement * 0.05, 1.0)

        # Simulate trading
        pip_value = 0.0001
        closes = df_test_features["close"].values[:n]
        highs = df_test_features["high"].values[:n]
        lows = df_test_features["low"].values[:n]
        timestamps = df_test_features.index[:n]

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
                            exit_price, exit_reason = tp_price, "take_profit"
                            break
                        if lows[j] <= sl_price:
                            exit_price, exit_reason = sl_price, "stop_loss"
                            break
                    else:
                        if lows[j] <= tp_price:
                            exit_price, exit_reason = tp_price, "take_profit"
                            break
                        if highs[j] >= sl_price:
                            exit_price, exit_reason = sl_price, "stop_loss"
                            break

                if exit_price is None:
                    exit_idx = min(i + self.max_holding_bars, n - 1)
                    exit_price, exit_reason = closes[exit_idx], "timeout"
                    j = exit_idx

                pnl_pips = ((exit_price - entry_price) / pip_value if direction == "long"
                           else (entry_price - exit_price) / pip_value)

                self.trades.append(Trade(
                    entry_time=entry_time, exit_time=timestamps[j], direction=direction,
                    entry_price=entry_price, exit_price=exit_price, confidence=conf,
                    pnl_pips=pnl_pips, exit_reason=exit_reason,
                ))
                i = j

            i += 1

        return self._calculate_results()

    def _calculate_results(self) -> Dict:
        if not self.trades:
            return {"total_trades": 0, "win_rate": 0, "total_pips": 0, "profit_factor": 0}

        trades_df = pd.DataFrame([{
            "direction": t.direction, "confidence": t.confidence,
            "pnl_pips": t.pnl_pips, "exit_reason": t.exit_reason,
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
    parser = argparse.ArgumentParser(description="Backtest Enhanced Hybrid Model")
    parser.add_argument("--data", type=str, default="data/forex/EURUSD_20200101_20251231_5min_combined.csv")
    parser.add_argument("--model-dir", type=str, default="models/enhanced_hybrid")
    parser.add_argument("--timeframe", type=str, default="1H")
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("ENHANCED HYBRID MODEL BACKTEST")
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

    # Load models
    model_dir = project_root / args.model_dir

    # Load config
    with open(model_dir / f"{args.timeframe}_config.json") as f:
        config = json.load(f)

    # Load XGBoost
    if args.timeframe == "1H":
        xgb_config = ImprovedModelConfig.hourly_model()
    else:
        xgb_config = ImprovedModelConfig.four_hour_model()

    xgb_model = ImprovedTimeframeModel(xgb_config)
    xgb_model.load(model_dir / f"{args.timeframe}_xgboost.pkl")

    # Load sequence model
    seq_predictor = EnhancedSequencePredictor.load(model_dir / f"{args.timeframe}_enhanced_sequence.pt")

    # Run backtest
    test_start_idx = int(len(df) * (1 - args.test_ratio))

    backtester = EnhancedHybridBacktester(
        xgb_model=xgb_model,
        seq_predictor=seq_predictor,
        config=config,
        min_confidence=args.confidence,
    )
    results = backtester.run(df, test_start_idx)

    # Print results
    print(f"\nTrade Summary:")
    print(f"  Total Trades:     {results['total_trades']}")
    print(f"  Winning:          {results['winning_trades']}")
    print(f"  Losing:           {results['losing_trades']}")
    print(f"  Win Rate:         {results['win_rate']:.1f}%")

    print(f"\nP&L Summary:")
    print(f"  Total Pips:       {results['total_pips']:+.1f}")
    print(f"  Avg Pips/Trade:   {results['avg_pips']:+.1f}")
    print(f"  Profit Factor:    {results['profit_factor']:.2f}")

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
        print(f"  [OK] Profit factor {results['profit_factor']:.2f} is profitable")
    else:
        print(f"  [!!] Profit factor {results['profit_factor']:.2f} below 1.0")

    print("=" * 70)


if __name__ == "__main__":
    main()
