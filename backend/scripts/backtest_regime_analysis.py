#!/usr/bin/env python3
"""Regime-aware backtesting for MTF Ensemble.

This script:
1. Detects market regimes (trend + volatility)
2. Analyzes trading performance by regime
3. Identifies profitable vs unprofitable regimes
4. Tests regime-filtered strategies against baseline

Usage:
    python scripts/backtest_regime_analysis.py
    python scripts/backtest_regime_analysis.py --confidence 0.70
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
from src.features.regime import RegimeDetector, MarketRegime, TrendRegime, VolatilityRegime

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
    pnl_pips: float
    exit_reason: str
    trend_regime: str
    volatility_regime: str
    market_regime: str


@dataclass
class RegimeStats:
    """Statistics for a specific regime."""
    regime: str
    bar_count: int
    bar_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pips: float
    avg_pips: float
    profit_factor: float
    recommendation: str  # "TRADE", "AVOID", "CAUTION"


class RegimeBacktester:
    """Backtester with regime analysis."""

    def __init__(
        self,
        ensemble: MTFEnsemble,
        regime_detector: RegimeDetector,
        min_confidence: float = 0.70,
        tp_pips: float = 25.0,
        sl_pips: float = 15.0,
        max_holding_bars: int = 12,
    ):
        self.ensemble = ensemble
        self.regime_detector = regime_detector
        self.min_confidence = min_confidence
        self.tp_pips = tp_pips
        self.sl_pips = sl_pips
        self.max_holding_bars = max_holding_bars
        self.trades: List[Trade] = []

    def run(self, df_5min: pd.DataFrame, test_start_idx: int) -> Tuple[List[Trade], pd.DataFrame]:
        """Run backtest with regime tracking.

        Returns:
            Tuple of (trades list, dataframe with regime columns)
        """
        from src.features.technical.calculator import TechnicalIndicatorCalculator

        logger.info("Preparing data for regime-aware backtest...")

        calc = TechnicalIndicatorCalculator(model_type="short_term")

        # Prepare 1H data
        model_1h = self.ensemble.models["1H"]
        df_1h = self.ensemble.resample_data(df_5min, "1H")
        higher_tf_data_1h = self.ensemble.prepare_higher_tf_data(df_5min, "1H")
        df_1h_features = calc.calculate(df_1h)
        df_1h_features = model_1h.feature_engine.add_all_features(df_1h_features, higher_tf_data_1h)
        df_1h_features = df_1h_features.dropna()

        # Add regime detection
        logger.info("Detecting market regimes...")
        df_1h_regime = self.regime_detector.detect_regime(df_1h_features)

        # Prepare features for prediction
        feature_cols_1h = model_1h.feature_names
        available_cols_1h = [c for c in feature_cols_1h if c in df_1h_regime.columns]

        # Get test period
        test_start_5min = df_5min.index[test_start_idx]
        test_mask = df_1h_regime.index >= test_start_5min
        df_test = df_1h_regime[test_mask].copy()

        logger.info(f"Test period: {df_test.index[0]} to {df_test.index[-1]}")
        logger.info(f"Test bars: {len(df_test)}")

        # Run backtest
        self.trades = []
        in_position = False
        entry_bar = None
        entry_price = None
        entry_direction = None
        entry_confidence = None
        entry_regime = None

        for i in range(len(df_test) - self.max_holding_bars):
            current_bar = df_test.iloc[i]
            current_time = df_test.index[i]

            if in_position:
                # Check exit conditions
                bars_held = i - entry_bar
                high = current_bar["high"]
                low = current_bar["low"]

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
                        pnl_pips = (current_bar["close"] - entry_price) / 0.0001
                        exit_reason = "timeout"
                else:  # short
                    if low <= entry_price - self.tp_pips * 0.0001:
                        pnl_pips = self.tp_pips
                        exit_reason = "take_profit"
                    elif high >= entry_price + self.sl_pips * 0.0001:
                        pnl_pips = -self.sl_pips
                        exit_reason = "stop_loss"
                    elif bars_held >= self.max_holding_bars:
                        pnl_pips = (entry_price - current_bar["close"]) / 0.0001
                        exit_reason = "timeout"

                if exit_reason:
                    trade = Trade(
                        entry_time=df_test.index[entry_bar],
                        exit_time=current_time,
                        direction=entry_direction,
                        entry_price=entry_price,
                        exit_price=current_bar["close"],
                        confidence=entry_confidence,
                        pnl_pips=pnl_pips,
                        exit_reason=exit_reason,
                        trend_regime=entry_regime["trend"],
                        volatility_regime=entry_regime["volatility"],
                        market_regime=entry_regime["market"],
                    )
                    self.trades.append(trade)
                    in_position = False

            else:
                # Check entry conditions
                X = current_bar[available_cols_1h].values.reshape(1, -1)

                try:
                    pred = self.ensemble.models["1H"].model.predict_proba(X)[0]
                    pred_class = 1 if pred[1] > 0.5 else 0
                    confidence = max(pred[0], pred[1])

                    if confidence >= self.min_confidence:
                        in_position = True
                        entry_bar = i
                        entry_price = current_bar["close"]
                        entry_direction = "long" if pred_class == 1 else "short"
                        entry_confidence = confidence
                        entry_regime = {
                            "trend": current_bar["trend_regime"],
                            "volatility": current_bar["volatility_regime"],
                            "market": current_bar["market_regime"],
                        }
                except Exception:
                    continue

        logger.info(f"Total trades: {len(self.trades)}")

        return self.trades, df_1h_regime

    def analyze_by_regime(self) -> Dict[str, RegimeStats]:
        """Analyze performance by market regime."""
        if not self.trades:
            return {}

        trades_df = pd.DataFrame([asdict(t) for t in self.trades])
        stats = {}

        # Analyze by market regime
        for regime in MarketRegime:
            regime_trades = trades_df[trades_df["market_regime"] == regime.value]

            if len(regime_trades) == 0:
                stats[regime.value] = RegimeStats(
                    regime=regime.value,
                    bar_count=0,
                    bar_pct=0,
                    total_trades=0,
                    winning_trades=0,
                    losing_trades=0,
                    win_rate=0,
                    total_pips=0,
                    avg_pips=0,
                    profit_factor=0,
                    recommendation="N/A",
                )
                continue

            wins = regime_trades[regime_trades["pnl_pips"] > 0]
            losses = regime_trades[regime_trades["pnl_pips"] <= 0]

            total_wins = wins["pnl_pips"].sum() if len(wins) > 0 else 0
            total_losses = abs(losses["pnl_pips"].sum()) if len(losses) > 0 else 0
            pf = total_wins / total_losses if total_losses > 0 else float("inf")

            win_rate = len(wins) / len(regime_trades) * 100
            total_pips = regime_trades["pnl_pips"].sum()
            avg_pips = regime_trades["pnl_pips"].mean()

            # Recommendation based on performance
            if win_rate >= 60 and pf >= 2.0:
                recommendation = "TRADE"
            elif win_rate >= 55 and pf >= 1.5:
                recommendation = "CAUTION"
            else:
                recommendation = "AVOID"

            stats[regime.value] = RegimeStats(
                regime=regime.value,
                bar_count=0,  # Will be filled from df
                bar_pct=0,
                total_trades=len(regime_trades),
                winning_trades=len(wins),
                losing_trades=len(losses),
                win_rate=win_rate,
                total_pips=total_pips,
                avg_pips=avg_pips,
                profit_factor=pf,
                recommendation=recommendation,
            )

        return stats

    def analyze_by_trend(self) -> Dict[str, Dict]:
        """Analyze performance by trend regime."""
        if not self.trades:
            return {}

        trades_df = pd.DataFrame([asdict(t) for t in self.trades])
        stats = {}

        for regime in TrendRegime:
            regime_trades = trades_df[trades_df["trend_regime"] == regime.value]

            if len(regime_trades) == 0:
                stats[regime.value] = {
                    "trades": 0,
                    "win_rate": 0,
                    "total_pips": 0,
                    "avg_pips": 0,
                }
                continue

            wins = regime_trades[regime_trades["pnl_pips"] > 0]
            win_rate = len(wins) / len(regime_trades) * 100

            stats[regime.value] = {
                "trades": len(regime_trades),
                "win_rate": win_rate,
                "total_pips": regime_trades["pnl_pips"].sum(),
                "avg_pips": regime_trades["pnl_pips"].mean(),
            }

        return stats

    def analyze_by_volatility(self) -> Dict[str, Dict]:
        """Analyze performance by volatility regime."""
        if not self.trades:
            return {}

        trades_df = pd.DataFrame([asdict(t) for t in self.trades])
        stats = {}

        for regime in VolatilityRegime:
            regime_trades = trades_df[trades_df["volatility_regime"] == regime.value]

            if len(regime_trades) == 0:
                stats[regime.value] = {
                    "trades": 0,
                    "win_rate": 0,
                    "total_pips": 0,
                    "avg_pips": 0,
                }
                continue

            wins = regime_trades[regime_trades["pnl_pips"] > 0]
            win_rate = len(wins) / len(regime_trades) * 100

            stats[regime.value] = {
                "trades": len(regime_trades),
                "win_rate": win_rate,
                "total_pips": regime_trades["pnl_pips"].sum(),
                "avg_pips": regime_trades["pnl_pips"].mean(),
            }

        return stats


def run_regime_filtered_backtest(
    ensemble: MTFEnsemble,
    regime_detector: RegimeDetector,
    df_5min: pd.DataFrame,
    test_start_idx: int,
    min_confidence: float,
    allowed_regimes: List[str],
) -> List[Trade]:
    """Run backtest only trading in allowed regimes."""
    from src.features.technical.calculator import TechnicalIndicatorCalculator

    calc = TechnicalIndicatorCalculator(model_type="short_term")

    # Prepare data
    model_1h = ensemble.models["1H"]
    df_1h = ensemble.resample_data(df_5min, "1H")
    higher_tf_data_1h = ensemble.prepare_higher_tf_data(df_5min, "1H")
    df_1h_features = calc.calculate(df_1h)
    df_1h_features = model_1h.feature_engine.add_all_features(df_1h_features, higher_tf_data_1h)
    df_1h_features = df_1h_features.dropna()

    # Add regime detection
    df_1h_regime = regime_detector.detect_regime(df_1h_features)

    feature_cols_1h = model_1h.feature_names
    available_cols_1h = [c for c in feature_cols_1h if c in df_1h_regime.columns]

    # Get test period
    test_start_5min = df_5min.index[test_start_idx]
    test_mask = df_1h_regime.index >= test_start_5min
    df_test = df_1h_regime[test_mask].copy()

    # Run backtest with regime filter
    trades = []
    in_position = False
    entry_bar = None
    entry_price = None
    entry_direction = None
    entry_confidence = None
    entry_regime = None

    tp_pips = 25.0
    sl_pips = 15.0
    max_holding_bars = 12

    for i in range(len(df_test) - max_holding_bars):
        current_bar = df_test.iloc[i]
        current_time = df_test.index[i]

        if in_position:
            bars_held = i - entry_bar
            high = current_bar["high"]
            low = current_bar["low"]

            pnl_pips = 0
            exit_reason = None

            if entry_direction == "long":
                if high >= entry_price + tp_pips * 0.0001:
                    pnl_pips = tp_pips
                    exit_reason = "take_profit"
                elif low <= entry_price - sl_pips * 0.0001:
                    pnl_pips = -sl_pips
                    exit_reason = "stop_loss"
                elif bars_held >= max_holding_bars:
                    pnl_pips = (current_bar["close"] - entry_price) / 0.0001
                    exit_reason = "timeout"
            else:
                if low <= entry_price - tp_pips * 0.0001:
                    pnl_pips = tp_pips
                    exit_reason = "take_profit"
                elif high >= entry_price + sl_pips * 0.0001:
                    pnl_pips = -sl_pips
                    exit_reason = "stop_loss"
                elif bars_held >= max_holding_bars:
                    pnl_pips = (entry_price - current_bar["close"]) / 0.0001
                    exit_reason = "timeout"

            if exit_reason:
                trade = Trade(
                    entry_time=df_test.index[entry_bar],
                    exit_time=current_time,
                    direction=entry_direction,
                    entry_price=entry_price,
                    exit_price=current_bar["close"],
                    confidence=entry_confidence,
                    pnl_pips=pnl_pips,
                    exit_reason=exit_reason,
                    trend_regime=entry_regime["trend"],
                    volatility_regime=entry_regime["volatility"],
                    market_regime=entry_regime["market"],
                )
                trades.append(trade)
                in_position = False

        else:
            # Check if current regime is allowed
            current_regime = current_bar["market_regime"]
            if current_regime not in allowed_regimes:
                continue

            X = current_bar[available_cols_1h].values.reshape(1, -1)

            try:
                pred = ensemble.models["1H"].model.predict_proba(X)[0]
                pred_class = 1 if pred[1] > 0.5 else 0
                confidence = max(pred[0], pred[1])

                if confidence >= min_confidence:
                    in_position = True
                    entry_bar = i
                    entry_price = current_bar["close"]
                    entry_direction = "long" if pred_class == 1 else "short"
                    entry_confidence = confidence
                    entry_regime = {
                        "trend": current_bar["trend_regime"],
                        "volatility": current_bar["volatility_regime"],
                        "market": current_bar["market_regime"],
                    }
            except Exception:
                continue

    return trades


def calculate_metrics(trades: List[Trade]) -> Dict:
    """Calculate trading metrics from trades."""
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "total_pips": 0,
            "profit_factor": 0,
            "avg_pips": 0,
        }

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
    parser = argparse.ArgumentParser(description="Regime-aware backtesting")
    parser.add_argument(
        "--data",
        default="data/forex/EURUSD_20200101_20251231_5min_combined.csv",
        help="Path to price data"
    )
    parser.add_argument(
        "--model-dir",
        default="models/mtf_ensemble",
        help="Model directory"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.70,
        help="Minimum confidence threshold"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test data ratio"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("REGIME-AWARE BACKTESTING")
    print("=" * 80)
    print(f"Data:       {args.data}")
    print(f"Model:      {args.model_dir}")
    print(f"Confidence: {args.confidence:.0%}")
    print("=" * 80)

    # Load data
    df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    logger.info(f"Loaded {len(df)} bars")

    # Calculate test start
    test_start_idx = int(len(df) * (1 - args.test_ratio))

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

    # Create regime detector
    regime_detector = RegimeDetector()

    # Run baseline backtest with regime tracking
    print("\n" + "=" * 80)
    print("PHASE 1: BASELINE ANALYSIS WITH REGIME TRACKING")
    print("=" * 80)

    backtester = RegimeBacktester(
        ensemble=ensemble,
        regime_detector=regime_detector,
        min_confidence=args.confidence,
    )

    trades, df_regime = backtester.run(df, test_start_idx)

    # Calculate baseline metrics
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

    regime_stats = backtester.analyze_by_regime()

    print("\n{:<25} {:>8} {:>8} {:>10} {:>8} {:>12}".format(
        "Regime", "Trades", "Win%", "Pips", "PF", "Recommend"
    ))
    print("-" * 80)

    for regime, stats in sorted(regime_stats.items(), key=lambda x: x[1].total_pips, reverse=True):
        if stats.total_trades > 0:
            print("{:<25} {:>8} {:>7.1f}% {:>+10.1f} {:>8.2f} {:>12}".format(
                regime,
                stats.total_trades,
                stats.win_rate,
                stats.total_pips,
                stats.profit_factor if stats.profit_factor != float("inf") else 999.99,
                stats.recommendation,
            ))

    # Analyze by trend regime
    print("\n" + "-" * 80)
    print("Performance by TREND Regime:")
    print("-" * 80)

    trend_stats = backtester.analyze_by_trend()

    print("\n{:<25} {:>8} {:>8} {:>10} {:>10}".format(
        "Trend Regime", "Trades", "Win%", "Pips", "Avg/Trade"
    ))
    print("-" * 60)

    for regime, stats in sorted(trend_stats.items(), key=lambda x: x[1]["total_pips"], reverse=True):
        if stats["trades"] > 0:
            print("{:<25} {:>8} {:>7.1f}% {:>+10.1f} {:>+10.1f}".format(
                regime,
                stats["trades"],
                stats["win_rate"],
                stats["total_pips"],
                stats["avg_pips"],
            ))

    # Analyze by volatility regime
    print("\n" + "-" * 80)
    print("Performance by VOLATILITY Regime:")
    print("-" * 80)

    vol_stats = backtester.analyze_by_volatility()

    print("\n{:<25} {:>8} {:>8} {:>10} {:>10}".format(
        "Volatility Regime", "Trades", "Win%", "Pips", "Avg/Trade"
    ))
    print("-" * 60)

    for regime, stats in sorted(vol_stats.items(), key=lambda x: x[1]["total_pips"], reverse=True):
        if stats["trades"] > 0:
            print("{:<25} {:>8} {:>7.1f}% {:>+10.1f} {:>+10.1f}".format(
                regime,
                stats["trades"],
                stats["win_rate"],
                stats["total_pips"],
                stats["avg_pips"],
            ))

    # Determine optimal regimes (TRADE recommendations)
    optimal_regimes = [
        regime for regime, stats in regime_stats.items()
        if stats.recommendation == "TRADE" and stats.total_trades >= 10
    ]

    # If no regimes qualify as TRADE, use CAUTION regimes
    if not optimal_regimes:
        optimal_regimes = [
            regime for regime, stats in regime_stats.items()
            if stats.recommendation in ["TRADE", "CAUTION"] and stats.total_trades >= 10
        ]

    # If still none, use all profitable regimes
    if not optimal_regimes:
        optimal_regimes = [
            regime for regime, stats in regime_stats.items()
            if stats.total_pips > 0 and stats.total_trades >= 10
        ]

    print("\n" + "=" * 80)
    print("PHASE 3: REGIME-FILTERED STRATEGY")
    print("=" * 80)

    print(f"\nOptimal regimes to trade: {optimal_regimes}")

    # Run regime-filtered backtest
    if optimal_regimes:
        filtered_trades = run_regime_filtered_backtest(
            ensemble=ensemble,
            regime_detector=regime_detector,
            df_5min=df,
            test_start_idx=test_start_idx,
            min_confidence=args.confidence,
            allowed_regimes=optimal_regimes,
        )

        filtered_metrics = calculate_metrics(filtered_trades)

        print(f"\nRegime-Filtered Results:")
        print(f"  Total Trades:   {filtered_metrics['total_trades']}")
        print(f"  Win Rate:       {filtered_metrics['win_rate']:.1f}%")
        print(f"  Total Pips:     {filtered_metrics['total_pips']:+.1f}")
        print(f"  Profit Factor:  {filtered_metrics['profit_factor']:.2f}")
        print(f"  Avg Pips/Trade: {filtered_metrics['avg_pips']:+.1f}")

        # Compare to baseline
        print("\n" + "=" * 80)
        print("COMPARISON: BASELINE vs REGIME-FILTERED")
        print("=" * 80)

        print("\n{:<20} {:>15} {:>15} {:>15}".format(
            "Metric", "Baseline", "Filtered", "Improvement"
        ))
        print("-" * 65)

        metrics_compare = [
            ("Total Trades", baseline_metrics["total_trades"], filtered_metrics["total_trades"]),
            ("Win Rate (%)", baseline_metrics["win_rate"], filtered_metrics["win_rate"]),
            ("Total Pips", baseline_metrics["total_pips"], filtered_metrics["total_pips"]),
            ("Profit Factor", baseline_metrics["profit_factor"], filtered_metrics["profit_factor"]),
            ("Avg Pips/Trade", baseline_metrics["avg_pips"], filtered_metrics["avg_pips"]),
        ]

        for name, baseline, filtered in metrics_compare:
            if name == "Total Trades":
                diff = filtered - baseline
                diff_str = f"{diff:+d}"
            else:
                diff = filtered - baseline
                diff_str = f"{diff:+.1f}" if abs(diff) < 100 else f"{diff:+.0f}"

            print("{:<20} {:>15.1f} {:>15.1f} {:>15}".format(
                name,
                baseline if isinstance(baseline, float) else float(baseline),
                filtered if isinstance(filtered, float) else float(filtered),
                diff_str,
            ))

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
        "regime_stats": {k: asdict(v) for k, v in regime_stats.items()},
        "trend_stats": trend_stats,
        "volatility_stats": vol_stats,
        "optimal_regimes": optimal_regimes,
        "filtered_metrics": filtered_metrics if optimal_regimes else None,
    }

    output_file = output_dir / "regime_analysis.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
