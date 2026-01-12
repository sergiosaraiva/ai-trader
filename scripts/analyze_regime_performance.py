#!/usr/bin/env python3
"""Analyze regime performance from MTF Ensemble backtest.

This script:
1. Runs the full MTF Ensemble backtest
2. Tracks the market regime at each trade entry
3. Analyzes performance by regime
4. Identifies profitable vs unprofitable regimes
5. Tests regime-filtered strategies

Usage:
    python scripts/analyze_regime_performance.py
    python scripts/analyze_regime_performance.py --confidence 0.70
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.models.multi_timeframe import MTFEnsemble, MTFEnsembleConfig
from src.features.regime import RegimeDetector, MarketRegime, TrendRegime, VolatilityRegime, RegimeConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade with regime info."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    confidence: float
    agreement_score: float
    pnl_pips: float
    exit_reason: str
    trend_regime: str
    volatility_regime: str
    market_regime: str


class RegimeAwareBacktester:
    """Backtester that tracks regime at each trade."""

    def __init__(
        self,
        ensemble: MTFEnsemble,
        min_confidence: float = 0.70,
        min_agreement: float = 0.5,
        tp_pips: float = 25.0,
        sl_pips: float = 15.0,
        max_holding_bars: int = 12,
    ):
        self.ensemble = ensemble
        self.min_confidence = min_confidence
        self.min_agreement = min_agreement
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        self.max_holding_bars = max_holding_bars
        self.trades: List[Trade] = []
        self.regime_detector = RegimeDetector()

    def run(self, df_5min: pd.DataFrame) -> Tuple[List[Trade], pd.DataFrame]:
        """Run full ensemble backtest with regime tracking."""
        from src.features.technical.calculator import TechnicalIndicatorCalculator

        logger.info("Preparing data for ensemble backtest with regime tracking...")

        calc = TechnicalIndicatorCalculator(model_type="short_term")

        # Prepare 1H data (primary trading timeframe)
        model_1h = self.ensemble.models["1H"]
        df_1h = self.ensemble.resample_data(df_5min, "1H")
        higher_tf_data_1h = self.ensemble.prepare_higher_tf_data(df_5min, "1H")
        df_1h_features = calc.calculate(df_1h)
        df_1h_features = model_1h.feature_engine.add_all_features(df_1h_features, higher_tf_data_1h)
        df_1h_features = df_1h_features.dropna()

        feature_cols_1h = model_1h.feature_names
        available_cols_1h = [c for c in feature_cols_1h if c in df_1h_features.columns]
        X_1h = df_1h_features[available_cols_1h].values

        # Split 1H data (60/20/20)
        n_total = len(X_1h)
        n_train = int(n_total * 0.6)
        n_val = int(n_total * 0.2)
        test_start = n_train + n_val

        X_1h_test = X_1h[test_start:]
        df_test = df_1h_features.iloc[test_start:].copy()

        logger.info(f"Test period: {df_test.index[0]} to {df_test.index[-1]}")
        logger.info(f"Test bars: {len(X_1h_test)}")

        # Get 1H predictions
        preds_1h, confs_1h = model_1h.predict_batch(X_1h_test)

        # Prepare 4H data
        model_4h = self.ensemble.models["4H"]
        df_4h = self.ensemble.resample_data(df_5min, "4H")
        higher_tf_data_4h = self.ensemble.prepare_higher_tf_data(df_5min, "4H")
        df_4h_features = calc.calculate(df_4h)
        df_4h_features = model_4h.feature_engine.add_all_features(df_4h_features, higher_tf_data_4h)
        df_4h_features = df_4h_features.dropna()

        feature_cols_4h = model_4h.feature_names
        available_cols_4h = [c for c in feature_cols_4h if c in df_4h_features.columns]
        X_4h = df_4h_features[available_cols_4h].values
        preds_4h_all, confs_4h_all = model_4h.predict_batch(X_4h)
        pred_4h_map = dict(zip(df_4h_features.index, zip(preds_4h_all, confs_4h_all)))

        # Prepare Daily data
        model_d = self.ensemble.models["D"]
        df_d = self.ensemble.resample_data(df_5min, "D")
        df_d_features = calc.calculate(df_d)
        df_d_features = model_d.feature_engine.add_all_features(df_d_features, {})
        df_d_features = df_d_features.dropna()

        feature_cols_d = model_d.feature_names
        available_cols_d = [c for c in feature_cols_d if c in df_d_features.columns]
        X_d = df_d_features[available_cols_d].values
        preds_d_all, confs_d_all = model_d.predict_batch(X_d)
        pred_d_map = dict(zip(df_d_features.index.date, zip(preds_d_all, confs_d_all)))

        logger.info(f"4H: {len(pred_4h_map)} predictions, D: {len(pred_d_map)} predictions")

        # Add regime detection to test data
        logger.info("Detecting market regimes...")
        df_test = self.regime_detector.detect_regime(df_test)

        # Get weights
        weights = self.ensemble._normalize_weights(self.ensemble.config.weights)
        w_1h = weights.get("1H", 0.6)
        w_4h = weights.get("4H", 0.3)
        w_d = weights.get("D", 0.1)

        # Pre-compute ensemble predictions
        closes = df_test["close"].values
        highs = df_test["high"].values
        lows = df_test["low"].values
        timestamps = df_test.index

        test_directions = []
        test_confidences = []
        test_agreements = []

        for i, ts in enumerate(timestamps):
            p_1h, c_1h = preds_1h[i], confs_1h[i]

            # Find 4H prediction
            ts_4h = ts.floor("4H")
            if ts_4h in pred_4h_map:
                p_4h, c_4h = pred_4h_map[ts_4h]
            else:
                prev_4h_times = [t for t in pred_4h_map.keys() if t <= ts]
                if prev_4h_times:
                    p_4h, c_4h = pred_4h_map[max(prev_4h_times)]
                else:
                    p_4h, c_4h = p_1h, c_1h

            # Find Daily prediction
            day = ts.date()
            if day in pred_d_map:
                p_d, c_d = pred_d_map[day]
            else:
                prev_days = [d for d in pred_d_map.keys() if d <= day]
                if prev_days:
                    p_d, c_d = pred_d_map[max(prev_days)]
                else:
                    p_d, c_d = p_1h, c_1h

            # Weighted ensemble
            prob_1h = c_1h if p_1h == 1 else (1 - c_1h)
            prob_4h = c_4h if p_4h == 1 else (1 - c_4h)
            prob_d = c_d if p_d == 1 else (1 - c_d)

            combined_prob = w_1h * prob_1h + w_4h * prob_4h + w_d * prob_d

            # Agreement
            agreement_count = sum([1 for p in [p_1h, p_4h, p_d] if p == p_1h])
            agreement_score = agreement_count / 3.0

            if agreement_count == 3:
                combined_prob = min(combined_prob + self.ensemble.config.agreement_bonus, 1.0)

            direction = 1 if combined_prob > 0.5 else 0
            confidence = combined_prob if direction == 1 else (1 - combined_prob)

            test_directions.append(direction)
            test_confidences.append(confidence)
            test_agreements.append(agreement_score)

        logger.info("Running backtest simulation...")

        # Run backtest
        self.trades = []
        in_position = False
        entry_idx = None
        entry_price = None
        entry_direction = None
        entry_confidence = None
        entry_agreement = None
        entry_regime = None

        for i in range(len(timestamps) - self.max_holding_bars):
            if in_position:
                bars_held = i - entry_idx
                high = highs[i]
                low = lows[i]

                pnl_pips = 0
                exit_reason = None

                if entry_direction == "long":
                    if high >= entry_price + self.tp_pips * 0.0001:
                        pnl_pips = self.tp_pips
                        exit_reason = "take_profit"
                    elif low <= entry_price - self.sl_pips * 0.0001:
                        pnl_pips = -self.sl_pips
                        exit_reason = "stop_loss"
                    elif bars_held >= self.max_holding_bars:
                        pnl_pips = (closes[i] - entry_price) / 0.0001
                        exit_reason = "timeout"
                else:
                    if low <= entry_price - self.tp_pips * 0.0001:
                        pnl_pips = self.tp_pips
                        exit_reason = "take_profit"
                    elif high >= entry_price + self.sl_pips * 0.0001:
                        pnl_pips = -self.sl_pips
                        exit_reason = "stop_loss"
                    elif bars_held >= self.max_holding_bars:
                        pnl_pips = (entry_price - closes[i]) / 0.0001
                        exit_reason = "timeout"

                if exit_reason:
                    trade = Trade(
                        entry_time=timestamps[entry_idx],
                        exit_time=timestamps[i],
                        direction=entry_direction,
                        entry_price=entry_price,
                        exit_price=closes[i],
                        confidence=entry_confidence,
                        agreement_score=entry_agreement,
                        pnl_pips=pnl_pips,
                        exit_reason=exit_reason,
                        trend_regime=entry_regime["trend"],
                        volatility_regime=entry_regime["volatility"],
                        market_regime=entry_regime["market"],
                    )
                    self.trades.append(trade)
                    in_position = False

            else:
                conf = test_confidences[i]
                agree = test_agreements[i]

                if conf >= self.min_confidence and agree >= self.min_agreement:
                    in_position = True
                    entry_idx = i
                    entry_price = closes[i]
                    entry_direction = "long" if test_directions[i] == 1 else "short"
                    entry_confidence = conf
                    entry_agreement = agree
                    entry_regime = {
                        "trend": df_test.iloc[i]["trend_regime"],
                        "volatility": df_test.iloc[i]["volatility_regime"],
                        "market": df_test.iloc[i]["market_regime"],
                    }

        logger.info(f"Total trades: {len(self.trades)}")

        return self.trades, df_test


def analyze_by_regime(trades: List[Trade]) -> Dict:
    """Analyze performance by market regime."""
    if not trades:
        return {}

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    stats = {}

    for regime in MarketRegime:
        regime_trades = trades_df[trades_df["market_regime"] == regime.value]

        if len(regime_trades) == 0:
            stats[regime.value] = {
                "trades": 0,
                "win_rate": 0,
                "total_pips": 0,
                "avg_pips": 0,
                "profit_factor": 0,
                "recommendation": "N/A",
            }
            continue

        wins = regime_trades[regime_trades["pnl_pips"] > 0]
        losses = regime_trades[regime_trades["pnl_pips"] <= 0]

        total_wins = wins["pnl_pips"].sum() if len(wins) > 0 else 0
        total_losses = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0
        pf = total_wins / total_losses if total_losses > 0 else float("inf")

        win_rate = len(wins) / len(regime_trades) * 100
        total_pips = regime_trades["pnl_pips"].sum()
        avg_pips = regime_trades["pnl_pips"].mean()

        # Recommendation
        if win_rate >= 60 and pf >= 2.0:
            recommendation = "TRADE"
        elif win_rate >= 55 and pf >= 1.5:
            recommendation = "CAUTION"
        else:
            recommendation = "AVOID"

        stats[regime.value] = {
            "trades": len(regime_trades),
            "win_rate": win_rate,
            "total_pips": total_pips,
            "avg_pips": avg_pips,
            "profit_factor": pf,
            "recommendation": recommendation,
        }

    return stats


def analyze_by_trend(trades: List[Trade]) -> Dict:
    """Analyze performance by trend regime."""
    if not trades:
        return {}

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    stats = {}

    for regime in TrendRegime:
        regime_trades = trades_df[trades_df["trend_regime"] == regime.value]

        if len(regime_trades) == 0:
            stats[regime.value] = {"trades": 0, "win_rate": 0, "total_pips": 0, "avg_pips": 0}
            continue

        wins = regime_trades[regime_trades["pnl_pips"] > 0]
        total_pips = regime_trades["pnl_pips"].sum()

        stats[regime.value] = {
            "trades": len(regime_trades),
            "win_rate": len(wins) / len(regime_trades) * 100,
            "total_pips": total_pips,
            "avg_pips": regime_trades["pnl_pips"].mean(),
        }

    return stats


def analyze_by_volatility(trades: List[Trade]) -> Dict:
    """Analyze performance by volatility regime."""
    if not trades:
        return {}

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    stats = {}

    for regime in VolatilityRegime:
        regime_trades = trades_df[trades_df["volatility_regime"] == regime.value]

        if len(regime_trades) == 0:
            stats[regime.value] = {"trades": 0, "win_rate": 0, "total_pips": 0, "avg_pips": 0}
            continue

        wins = regime_trades[regime_trades["pnl_pips"] > 0]

        stats[regime.value] = {
            "trades": len(regime_trades),
            "win_rate": len(wins) / len(regime_trades) * 100,
            "total_pips": regime_trades["pnl_pips"].sum(),
            "avg_pips": regime_trades["pnl_pips"].mean(),
        }

    return stats


def filter_trades_by_regime(trades: List[Trade], allowed_regimes: List[str]) -> List[Trade]:
    """Filter trades to only those in allowed regimes."""
    return [t for t in trades if t.market_regime in allowed_regimes]


def calculate_metrics(trades: List[Trade]) -> Dict:
    """Calculate trading metrics."""
    if not trades:
        return {"total_trades": 0, "win_rate": 0, "total_pips": 0, "profit_factor": 0, "avg_pips": 0}

    trades_df = pd.DataFrame([asdict(t) for t in trades])

    wins = trades_df[trades_df["pnl_pips"] > 0]
    losses = trades_df[trades_df["pnl_pips"] <= 0]

    total_wins = wins["pnl_pips"].sum() if len(wins) > 0 else 0
    total_losses = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0
    pf = total_wins / total_losses if total_losses > 0 else float("inf")

    return {
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate": len(wins) / len(trades) * 100,
        "total_pips": trades_df["pnl_pips"].sum(),
        "avg_pips": trades_df["pnl_pips"].mean(),
        "profit_factor": pf,
    }


def main():
    parser = argparse.ArgumentParser(description="Regime performance analysis")
    parser.add_argument("--data", default="data/forex/EURUSD_20200101_20251231_5min_combined.csv")
    parser.add_argument("--model-dir", default="models/mtf_ensemble")
    parser.add_argument("--confidence", type=float, default=0.70)
    args = parser.parse_args()

    print("=" * 80)
    print("REGIME PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"Data:       {args.data}")
    print(f"Model:      {args.model_dir}")
    print(f"Confidence: {args.confidence:.0%}")
    print("=" * 80)

    # Load data
    df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df)} bars")

    # Load ensemble
    config = MTFEnsembleConfig(
        weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        include_sentiment=True,
        sentiment_source="epu",
        sentiment_by_timeframe={"1H": False, "4H": False, "D": True},
    )
    ensemble = MTFEnsemble(config=config, model_dir=args.model_dir)
    ensemble.load()
    logger.info("Loaded MTF Ensemble")

    # Run backtest with regime tracking
    print("\n" + "=" * 80)
    print("PHASE 1: BASELINE WITH REGIME TRACKING")
    print("=" * 80)

    backtester = RegimeAwareBacktester(
        ensemble=ensemble,
        min_confidence=args.confidence,
    )

    trades, df_regime = backtester.run(df)
    baseline_metrics = calculate_metrics(trades)

    print(f"\nBaseline Results (Confidence >= {args.confidence:.0%}):")
    print(f"  Total Trades:   {baseline_metrics['total_trades']}")
    print(f"  Win Rate:       {baseline_metrics['win_rate']:.1f}%")
    print(f"  Total Pips:     {baseline_metrics['total_pips']:+.1f}")
    print(f"  Profit Factor:  {baseline_metrics['profit_factor']:.2f}")
    print(f"  Avg Pips/Trade: {baseline_metrics['avg_pips']:+.1f}")

    # Analyze by regime
    print("\n" + "=" * 80)
    print("PHASE 2: PERFORMANCE BY MARKET REGIME")
    print("=" * 80)

    regime_stats = analyze_by_regime(trades)

    print("\n{:<25} {:>8} {:>8} {:>10} {:>8} {:>12}".format(
        "Regime", "Trades", "Win%", "Pips", "PF", "Recommend"
    ))
    print("-" * 80)

    for regime, stats in sorted(regime_stats.items(), key=lambda x: x[1]["total_pips"], reverse=True):
        if stats["trades"] > 0:
            print("{:<25} {:>8} {:>7.1f}% {:>+10.1f} {:>8.2f} {:>12}".format(
                regime,
                stats["trades"],
                stats["win_rate"],
                stats["total_pips"],
                stats["profit_factor"] if stats["profit_factor"] != float("inf") else 999.99,
                stats["recommendation"],
            ))

    # Trend regime analysis
    print("\n" + "-" * 80)
    print("Performance by TREND Regime:")
    print("-" * 80)

    trend_stats = analyze_by_trend(trades)

    print("\n{:<25} {:>8} {:>8} {:>10} {:>10}".format("Trend", "Trades", "Win%", "Pips", "Avg"))
    print("-" * 60)
    for regime, stats in sorted(trend_stats.items(), key=lambda x: x[1]["total_pips"], reverse=True):
        if stats["trades"] > 0:
            print("{:<25} {:>8} {:>7.1f}% {:>+10.1f} {:>+10.1f}".format(
                regime, stats["trades"], stats["win_rate"], stats["total_pips"], stats["avg_pips"]
            ))

    # Volatility regime analysis
    print("\n" + "-" * 80)
    print("Performance by VOLATILITY Regime:")
    print("-" * 80)

    vol_stats = analyze_by_volatility(trades)

    print("\n{:<25} {:>8} {:>8} {:>10} {:>10}".format("Volatility", "Trades", "Win%", "Pips", "Avg"))
    print("-" * 60)
    for regime, stats in sorted(vol_stats.items(), key=lambda x: x[1]["total_pips"], reverse=True):
        if stats["trades"] > 0:
            print("{:<25} {:>8} {:>7.1f}% {:>+10.1f} {:>+10.1f}".format(
                regime, stats["trades"], stats["win_rate"], stats["total_pips"], stats["avg_pips"]
            ))

    # Find optimal regimes
    optimal_regimes = [r for r, s in regime_stats.items() if s["recommendation"] == "TRADE" and s["trades"] >= 10]
    if not optimal_regimes:
        optimal_regimes = [r for r, s in regime_stats.items()
                         if s["recommendation"] in ["TRADE", "CAUTION"] and s["trades"] >= 10]
    if not optimal_regimes:
        optimal_regimes = [r for r, s in regime_stats.items() if s["total_pips"] > 0 and s["trades"] >= 10]

    # Find regimes to avoid
    avoid_regimes = [r for r, s in regime_stats.items()
                    if s["total_pips"] < 0 and s["trades"] >= 10]

    print("\n" + "=" * 80)
    print("PHASE 3: REGIME-FILTERED STRATEGY")
    print("=" * 80)

    print(f"\nOptimal regimes: {optimal_regimes}")
    print(f"Avoid regimes:   {avoid_regimes}")

    # Test filtering by avoiding bad regimes
    if avoid_regimes:
        allowed = [r for r in MarketRegime if r.value not in avoid_regimes]
        allowed_values = [r.value for r in allowed]
        filtered_trades = filter_trades_by_regime(trades, allowed_values)
        filtered_metrics = calculate_metrics(filtered_trades)

        print(f"\nRegime-Filtered Results (avoiding {avoid_regimes}):")
        print(f"  Total Trades:   {filtered_metrics['total_trades']}")
        print(f"  Win Rate:       {filtered_metrics['win_rate']:.1f}%")
        print(f"  Total Pips:     {filtered_metrics['total_pips']:+.1f}")
        print(f"  Profit Factor:  {filtered_metrics['profit_factor']:.2f}")
        print(f"  Avg Pips/Trade: {filtered_metrics['avg_pips']:+.1f}")

        # Comparison
        print("\n" + "=" * 80)
        print("COMPARISON: BASELINE vs REGIME-FILTERED")
        print("=" * 80)

        print("\n{:<20} {:>15} {:>15} {:>15}".format("Metric", "Baseline", "Filtered", "Change"))
        print("-" * 65)

        comparisons = [
            ("Total Trades", baseline_metrics["total_trades"], filtered_metrics["total_trades"]),
            ("Win Rate (%)", baseline_metrics["win_rate"], filtered_metrics["win_rate"]),
            ("Total Pips", baseline_metrics["total_pips"], filtered_metrics["total_pips"]),
            ("Profit Factor", baseline_metrics["profit_factor"], filtered_metrics["profit_factor"]),
            ("Avg Pips/Trade", baseline_metrics["avg_pips"], filtered_metrics["avg_pips"]),
        ]

        for name, base, filt in comparisons:
            diff = filt - base
            if name == "Total Trades":
                print("{:<20} {:>15} {:>15} {:>+15}".format(name, base, filt, int(diff)))
            else:
                print("{:<20} {:>15.2f} {:>15.2f} {:>+15.2f}".format(name, base, filt, diff))
    else:
        filtered_metrics = baseline_metrics
        print("\nNo regimes to avoid - baseline is optimal.")

    # Save results
    output_dir = Path("results/regime_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "confidence_threshold": args.confidence,
            "data_file": args.data,
            "model_dir": args.model_dir,
        },
        "baseline_metrics": baseline_metrics,
        "regime_stats": regime_stats,
        "trend_stats": trend_stats,
        "volatility_stats": vol_stats,
        "optimal_regimes": optimal_regimes,
        "avoid_regimes": avoid_regimes,
        "filtered_metrics": filtered_metrics,
    }

    output_file = output_dir / "regime_performance.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
