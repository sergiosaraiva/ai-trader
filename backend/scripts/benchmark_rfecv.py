#!/usr/bin/env python3
"""Benchmark RFECV feature selection vs baseline performance.

This script trains and backtests both configurations:
1. Baseline: All features (115-134 per model)
2. RFECV: Selected features (reduced set)

Compares:
- Validation accuracy
- Win rate
- Total pips
- Profit factor
- Feature counts
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
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
from src.models.feature_selection import RFECVConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    # Training metrics
    train_time_seconds: float = 0.0
    model_1h_accuracy: float = 0.0
    model_4h_accuracy: float = 0.0
    model_d_accuracy: float = 0.0
    # Feature counts
    model_1h_features: int = 0
    model_4h_features: int = 0
    model_d_features: int = 0
    # Backtest metrics (at 55% threshold)
    total_trades: int = 0
    win_rate: float = 0.0
    total_pips: float = 0.0
    profit_factor: float = 0.0
    avg_pips_per_trade: float = 0.0
    # High confidence metrics (at 70% threshold)
    hc_total_trades: int = 0
    hc_win_rate: float = 0.0
    hc_total_pips: float = 0.0
    hc_profit_factor: float = 0.0
    # Selected features (RFECV only)
    selected_features_1h: List[str] = field(default_factory=list)
    selected_features_4h: List[str] = field(default_factory=list)
    selected_features_d: List[str] = field(default_factory=list)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and prepare 5-minute OHLCV data."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Rename columns to standard format
    df.columns = df.columns.str.lower()

    # Ensure index has a name
    df.index.name = "time"

    logger.info(f"Loaded {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
    return df


def train_baseline(
    df: pd.DataFrame,
    model_dir: Path,
    sentiment: bool = True,
    stacking: bool = True,
) -> Tuple[MTFEnsemble, BenchmarkResult]:
    """Train baseline model (all features, no RFECV)."""
    logger.info("=" * 60)
    logger.info("Training BASELINE model (all features)")
    logger.info("=" * 60)

    result = BenchmarkResult(name="baseline")

    config = MTFEnsembleConfig(
        weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        agreement_bonus=0.05,
        use_regime_adjustment=True,
        include_sentiment=sentiment,
        sentiment_source="epu",
        sentiment_by_timeframe={"1H": False, "4H": False, "D": True},
        trading_pair="EURUSD",
        use_stacking=stacking,
        use_rfecv=False,  # Baseline: no RFECV
    )

    ensemble = MTFEnsemble(config)

    start_time = time.time()
    training_results = ensemble.train(
        df_5min=df,
        train_ratio=0.6,
        val_ratio=0.2,
    )
    result.train_time_seconds = time.time() - start_time

    # Extract metrics
    result.model_1h_accuracy = training_results.get("1H", {}).get("val_accuracy", 0.0)
    result.model_4h_accuracy = training_results.get("4H", {}).get("val_accuracy", 0.0)
    result.model_d_accuracy = training_results.get("D", {}).get("val_accuracy", 0.0)

    # Feature counts
    result.model_1h_features = len(ensemble.models["1H"].feature_names)
    result.model_4h_features = len(ensemble.models["4H"].feature_names)
    result.model_d_features = len(ensemble.models["D"].feature_names)

    # Save model
    ensemble.save(model_dir / "baseline")

    logger.info(f"Baseline training completed in {result.train_time_seconds:.1f}s")
    logger.info(f"Feature counts: 1H={result.model_1h_features}, 4H={result.model_4h_features}, D={result.model_d_features}")

    return ensemble, result


def train_rfecv(
    df: pd.DataFrame,
    model_dir: Path,
    sentiment: bool = True,
    stacking: bool = True,
    rfecv_min_features: int = 20,
    rfecv_step: float = 0.1,
    rfecv_cv_folds: int = 5,
) -> Tuple[MTFEnsemble, BenchmarkResult]:
    """Train model with RFECV feature selection."""
    logger.info("=" * 60)
    logger.info("Training RFECV model (feature selection enabled)")
    logger.info("=" * 60)

    result = BenchmarkResult(name="rfecv")

    rfecv_config = RFECVConfig(
        min_features_to_select=rfecv_min_features,
        step=rfecv_step,
        cv=rfecv_cv_folds,
        verbose=1,
    )

    config = MTFEnsembleConfig(
        weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
        agreement_bonus=0.05,
        use_regime_adjustment=True,
        include_sentiment=sentiment,
        sentiment_source="epu",
        sentiment_by_timeframe={"1H": False, "4H": False, "D": True},
        trading_pair="EURUSD",
        use_stacking=stacking,
        use_rfecv=True,  # Enable RFECV
        rfecv_config=rfecv_config,
    )

    ensemble = MTFEnsemble(config)

    start_time = time.time()
    training_results = ensemble.train(
        df_5min=df,
        train_ratio=0.6,
        val_ratio=0.2,
    )
    result.train_time_seconds = time.time() - start_time

    # Extract metrics
    result.model_1h_accuracy = training_results.get("1H", {}).get("val_accuracy", 0.0)
    result.model_4h_accuracy = training_results.get("4H", {}).get("val_accuracy", 0.0)
    result.model_d_accuracy = training_results.get("D", {}).get("val_accuracy", 0.0)

    # Feature counts (after RFECV selection)
    result.model_1h_features = len(ensemble.models["1H"].selected_features or [])
    result.model_4h_features = len(ensemble.models["4H"].selected_features or [])
    result.model_d_features = len(ensemble.models["D"].selected_features or [])

    # Store selected features
    result.selected_features_1h = ensemble.models["1H"].selected_features or []
    result.selected_features_4h = ensemble.models["4H"].selected_features or []
    result.selected_features_d = ensemble.models["D"].selected_features or []

    # Save model
    ensemble.save(model_dir / "rfecv")

    logger.info(f"RFECV training completed in {result.train_time_seconds:.1f}s")
    logger.info(f"Feature counts: 1H={result.model_1h_features}, 4H={result.model_4h_features}, D={result.model_d_features}")

    return ensemble, result


def run_backtest(
    ensemble: MTFEnsemble,
    df: pd.DataFrame,
    result: BenchmarkResult,
    confidence: float = 0.55,
) -> BenchmarkResult:
    """Run backtest and populate result metrics."""
    from scripts.backtest_mtf_ensemble import MTFEnsembleBacktester

    logger.info(f"Running backtest for {result.name} at {confidence:.0%} confidence...")

    # Calculate test start index (60% train + 20% val = 80% used for training)
    test_start_idx = int(len(df) * 0.8)

    backtester = MTFEnsembleBacktester(
        ensemble=ensemble,
        min_confidence=confidence,
        tp_pips=25.0,
        sl_pips=15.0,
        max_holding_bars=12,
    )

    metrics = backtester.run(df, test_start_idx)

    if confidence == 0.55:
        result.total_trades = metrics.get("total_trades", 0)
        result.win_rate = metrics.get("win_rate", 0.0)
        result.total_pips = metrics.get("total_pips", 0.0)
        result.profit_factor = metrics.get("profit_factor", 0.0)
        result.avg_pips_per_trade = metrics.get("avg_pips_per_trade", 0.0)
    elif confidence == 0.70:
        result.hc_total_trades = metrics.get("total_trades", 0)
        result.hc_win_rate = metrics.get("win_rate", 0.0)
        result.hc_total_pips = metrics.get("total_pips", 0.0)
        result.hc_profit_factor = metrics.get("profit_factor", 0.0)

    return result


def print_comparison(baseline: BenchmarkResult, rfecv: BenchmarkResult) -> None:
    """Print comparison table between baseline and RFECV."""
    print("\n" + "=" * 80)
    print(" BENCHMARK RESULTS: BASELINE vs RFECV")
    print("=" * 80)

    # Training metrics
    print("\n## Training Metrics")
    print("-" * 60)
    print(f"{'Metric':<30} {'Baseline':>15} {'RFECV':>15} {'Diff':>15}")
    print("-" * 60)
    print(f"{'Training Time (s)':<30} {baseline.train_time_seconds:>15.1f} {rfecv.train_time_seconds:>15.1f} {rfecv.train_time_seconds - baseline.train_time_seconds:>+15.1f}")
    print(f"{'1H Val Accuracy':<30} {baseline.model_1h_accuracy:>15.2%} {rfecv.model_1h_accuracy:>15.2%} {(rfecv.model_1h_accuracy - baseline.model_1h_accuracy)*100:>+15.2f}%")
    print(f"{'4H Val Accuracy':<30} {baseline.model_4h_accuracy:>15.2%} {rfecv.model_4h_accuracy:>15.2%} {(rfecv.model_4h_accuracy - baseline.model_4h_accuracy)*100:>+15.2f}%")
    print(f"{'D Val Accuracy':<30} {baseline.model_d_accuracy:>15.2%} {rfecv.model_d_accuracy:>15.2%} {(rfecv.model_d_accuracy - baseline.model_d_accuracy)*100:>+15.2f}%")

    # Feature counts
    print("\n## Feature Counts")
    print("-" * 60)
    print(f"{'Model':<30} {'Baseline':>15} {'RFECV':>15} {'Reduction':>15}")
    print("-" * 60)
    print(f"{'1H Model':<30} {baseline.model_1h_features:>15} {rfecv.model_1h_features:>15} {(baseline.model_1h_features - rfecv.model_1h_features) / baseline.model_1h_features * 100 if baseline.model_1h_features > 0 else 0:>14.1f}%")
    print(f"{'4H Model':<30} {baseline.model_4h_features:>15} {rfecv.model_4h_features:>15} {(baseline.model_4h_features - rfecv.model_4h_features) / baseline.model_4h_features * 100 if baseline.model_4h_features > 0 else 0:>14.1f}%")
    print(f"{'D Model':<30} {baseline.model_d_features:>15} {rfecv.model_d_features:>15} {(baseline.model_d_features - rfecv.model_d_features) / baseline.model_d_features * 100 if baseline.model_d_features > 0 else 0:>14.1f}%")

    # Backtest metrics at 55%
    print("\n## Backtest Performance (55% Confidence)")
    print("-" * 60)
    print(f"{'Metric':<30} {'Baseline':>15} {'RFECV':>15} {'Diff':>15}")
    print("-" * 60)
    print(f"{'Total Trades':<30} {baseline.total_trades:>15} {rfecv.total_trades:>15} {rfecv.total_trades - baseline.total_trades:>+15}")
    print(f"{'Win Rate':<30} {baseline.win_rate:>15.2%} {rfecv.win_rate:>15.2%} {(rfecv.win_rate - baseline.win_rate)*100:>+15.2f}%")
    print(f"{'Total Pips':<30} {baseline.total_pips:>+15.0f} {rfecv.total_pips:>+15.0f} {rfecv.total_pips - baseline.total_pips:>+15.0f}")
    print(f"{'Profit Factor':<30} {baseline.profit_factor:>15.2f} {rfecv.profit_factor:>15.2f} {rfecv.profit_factor - baseline.profit_factor:>+15.2f}")
    print(f"{'Avg Pips/Trade':<30} {baseline.avg_pips_per_trade:>+15.1f} {rfecv.avg_pips_per_trade:>+15.1f} {rfecv.avg_pips_per_trade - baseline.avg_pips_per_trade:>+15.1f}")

    # Backtest metrics at 70%
    print("\n## Backtest Performance (70% Confidence)")
    print("-" * 60)
    print(f"{'Metric':<30} {'Baseline':>15} {'RFECV':>15} {'Diff':>15}")
    print("-" * 60)
    print(f"{'Total Trades':<30} {baseline.hc_total_trades:>15} {rfecv.hc_total_trades:>15} {rfecv.hc_total_trades - baseline.hc_total_trades:>+15}")
    print(f"{'Win Rate':<30} {baseline.hc_win_rate:>15.2%} {rfecv.hc_win_rate:>15.2%} {(rfecv.hc_win_rate - baseline.hc_win_rate)*100:>+15.2f}%")
    print(f"{'Total Pips':<30} {baseline.hc_total_pips:>+15.0f} {rfecv.hc_total_pips:>+15.0f} {rfecv.hc_total_pips - baseline.hc_total_pips:>+15.0f}")
    print(f"{'Profit Factor':<30} {baseline.hc_profit_factor:>15.2f} {rfecv.hc_profit_factor:>15.2f} {rfecv.hc_profit_factor - baseline.hc_profit_factor:>+15.2f}")

    # Overall recommendation
    print("\n" + "=" * 80)
    print(" RECOMMENDATION")
    print("=" * 80)

    # Calculate improvement score
    improvements = 0
    if rfecv.model_1h_accuracy > baseline.model_1h_accuracy:
        improvements += 1
    if rfecv.model_4h_accuracy > baseline.model_4h_accuracy:
        improvements += 1
    if rfecv.model_d_accuracy > baseline.model_d_accuracy:
        improvements += 1
    if rfecv.hc_win_rate > baseline.hc_win_rate:
        improvements += 1
    if rfecv.hc_total_pips > baseline.hc_total_pips:
        improvements += 1
    if rfecv.hc_profit_factor > baseline.hc_profit_factor:
        improvements += 1

    if improvements >= 4:
        print("\n✅ RFECV shows CONSISTENT improvement across most metrics.")
        print("   Recommend: ADOPT RFECV for production")
    elif improvements >= 2:
        print("\n⚠️ RFECV shows MIXED results.")
        print("   Recommend: Further analysis needed before adoption")
    else:
        print("\n❌ RFECV does NOT improve performance.")
        print("   Recommend: KEEP baseline (do not use RFECV)")

    print()


def save_results(
    baseline: BenchmarkResult,
    rfecv: BenchmarkResult,
    output_path: Path,
) -> None:
    """Save benchmark results to JSON."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "baseline": {
            "name": baseline.name,
            "train_time_seconds": baseline.train_time_seconds,
            "model_1h_accuracy": baseline.model_1h_accuracy,
            "model_4h_accuracy": baseline.model_4h_accuracy,
            "model_d_accuracy": baseline.model_d_accuracy,
            "model_1h_features": baseline.model_1h_features,
            "model_4h_features": baseline.model_4h_features,
            "model_d_features": baseline.model_d_features,
            "total_trades": baseline.total_trades,
            "win_rate": baseline.win_rate,
            "total_pips": baseline.total_pips,
            "profit_factor": baseline.profit_factor,
            "hc_total_trades": baseline.hc_total_trades,
            "hc_win_rate": baseline.hc_win_rate,
            "hc_total_pips": baseline.hc_total_pips,
            "hc_profit_factor": baseline.hc_profit_factor,
        },
        "rfecv": {
            "name": rfecv.name,
            "train_time_seconds": rfecv.train_time_seconds,
            "model_1h_accuracy": rfecv.model_1h_accuracy,
            "model_4h_accuracy": rfecv.model_4h_accuracy,
            "model_d_accuracy": rfecv.model_d_accuracy,
            "model_1h_features": rfecv.model_1h_features,
            "model_4h_features": rfecv.model_4h_features,
            "model_d_features": rfecv.model_d_features,
            "total_trades": rfecv.total_trades,
            "win_rate": rfecv.win_rate,
            "total_pips": rfecv.total_pips,
            "profit_factor": rfecv.profit_factor,
            "hc_total_trades": rfecv.hc_total_trades,
            "hc_win_rate": rfecv.hc_win_rate,
            "hc_total_pips": rfecv.hc_total_pips,
            "hc_profit_factor": rfecv.hc_profit_factor,
            "selected_features": {
                "1H": rfecv.selected_features_1h,
                "4H": rfecv.selected_features_4h,
                "D": rfecv.selected_features_d,
            },
        },
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark RFECV vs baseline")
    parser.add_argument(
        "--data",
        type=str,
        default="data/forex/EURUSD_20200101_20251231_5min_combined.csv",
        help="Path to 5-minute OHLCV data",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/benchmark_rfecv",
        help="Directory for model output",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/rfecv_benchmark_results.json",
        help="Path for benchmark results JSON",
    )
    parser.add_argument(
        "--no-sentiment",
        action="store_true",
        help="Disable sentiment features",
    )
    parser.add_argument(
        "--no-stacking",
        action="store_true",
        help="Disable stacking meta-learner",
    )
    parser.add_argument(
        "--rfecv-min-features",
        type=int,
        default=20,
        help="Minimum features for RFECV (default: 20)",
    )
    parser.add_argument(
        "--rfecv-step",
        type=float,
        default=0.1,
        help="RFECV step size (default: 0.1)",
    )
    parser.add_argument(
        "--rfecv-cv-folds",
        type=int,
        default=5,
        help="RFECV cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip backtest (training only)",
    )

    args = parser.parse_args()

    # Setup paths
    data_path = Path(args.data)
    model_dir = Path(args.model_dir)
    output_path = Path(args.output)

    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(data_path)

    # Train baseline
    baseline_ensemble, baseline_result = train_baseline(
        df=df,
        model_dir=model_dir,
        sentiment=not args.no_sentiment,
        stacking=not args.no_stacking,
    )

    # Train RFECV
    rfecv_ensemble, rfecv_result = train_rfecv(
        df=df,
        model_dir=model_dir,
        sentiment=not args.no_sentiment,
        stacking=not args.no_stacking,
        rfecv_min_features=args.rfecv_min_features,
        rfecv_step=args.rfecv_step,
        rfecv_cv_folds=args.rfecv_cv_folds,
    )

    # Run backtests
    if not args.skip_backtest:
        # Backtest at 55% confidence
        baseline_result = run_backtest(baseline_ensemble, df, baseline_result, confidence=0.55)
        rfecv_result = run_backtest(rfecv_ensemble, df, rfecv_result, confidence=0.55)

        # Backtest at 70% confidence
        baseline_result = run_backtest(baseline_ensemble, df, baseline_result, confidence=0.70)
        rfecv_result = run_backtest(rfecv_ensemble, df, rfecv_result, confidence=0.70)

    # Print comparison
    print_comparison(baseline_result, rfecv_result)

    # Save results
    save_results(baseline_result, rfecv_result, output_path)

    logger.info("Benchmark complete!")


if __name__ == "__main__":
    main()
