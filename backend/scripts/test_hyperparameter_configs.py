#!/usr/bin/env python3
"""Test hyperparameter configurations using Walk-Forward Optimization.

This script tests hyperparameter configurations across multiple time windows
to find settings that generalize well across different market conditions,
avoiding overfitting to a single test period.

Key improvement over single-period testing:
- Uses 4 rolling WFO windows instead of one fixed test period
- Evaluates total pips AND consistency across windows
- Selects configs that perform well across ALL market conditions
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.multi_timeframe.mtf_ensemble import MTFEnsemble, MTFEnsembleConfig

# Import backtest function from WFO script
from walk_forward_optimization import run_window_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define configurations to test
CONFIGS = {
    "baseline": {
        "description": "Original production configuration (XGBoost defaults)",
        "1H": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
        "4H": {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.05},
        "D": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05},
    },
    "conservative": {
        "description": "More regularized (less depth, fewer trees, lower LR)",
        "1H": {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.03},
        "4H": {"n_estimators": 120, "max_depth": 4, "learning_rate": 0.03},
        "D": {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.03},
    },
    "deeper": {
        "description": "More capacity (depth +1)",
        "1H": {"n_estimators": 200, "max_depth": 7, "learning_rate": 0.05},
        "4H": {"n_estimators": 150, "max_depth": 6, "learning_rate": 0.05},
        "D": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.05},
    },
    "more_trees": {
        "description": "More trees with lower learning rate",
        "1H": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.03},
        "4H": {"n_estimators": 225, "max_depth": 5, "learning_rate": 0.03},
        "D": {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.03},
    },
    "shallow_fast": {
        "description": "Shallow trees with higher learning rate",
        "1H": {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.08},
        "4H": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.08},
        "D": {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.08},
    },
}


@dataclass
class WFOWindow:
    """Definition of a walk-forward window."""

    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass
class WindowResult:
    """Results from one WFO window."""

    window_id: int
    test_period: str
    total_trades: int
    win_rate: float
    profit_factor: float
    total_pips: float


@dataclass
class ConfigResult:
    """Aggregated results for one hyperparameter configuration."""

    config_name: str
    description: str
    hyperparams: Dict
    window_results: List[WindowResult]
    total_pips: float
    avg_pips_per_window: float
    pips_std: float
    avg_win_rate: float
    avg_profit_factor: float
    profitable_windows: int
    total_windows: int
    consistency_score: float  # % of windows profitable


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and prepare data."""
    logger.info(f"Loading data from {data_path}")
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        # Read CSV with unnamed index column
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    logger.info(f"Loaded {len(df)} bars from {df.index.min()} to {df.index.max()}")
    return df


def create_wfo_windows(
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
    start_year: int = 2020,
    end_year: int = 2025,
) -> List[WFOWindow]:
    """Create walk-forward optimization windows.

    Uses fewer windows than full WFO for faster HPO testing while still
    ensuring robustness across multiple time periods.
    """
    windows = []

    # Create 4 windows covering different market conditions
    # Window 1: Train 2020-2021, Test 2022-H1
    # Window 2: Train 2021-2022, Test 2023-H1
    # Window 3: Train 2022-2023, Test 2024-H1
    # Window 4: Train 2023-2024, Test 2025-H1

    window_defs = [
        ("2020-01", "2021-12", "2022-01", "2022-06"),
        ("2021-01", "2022-12", "2023-01", "2023-06"),
        ("2022-01", "2023-12", "2024-01", "2024-06"),
        ("2023-01", "2024-12", "2025-01", "2025-06"),
    ]

    for train_start, train_end, test_start, test_end in window_defs:
        windows.append(
            WFOWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )

    return windows


def filter_data_by_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Filter dataframe to a specific time period."""
    start_dt = pd.Timestamp(start)
    # Add last day of month for end
    end_dt = pd.Timestamp(end) + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
    return df[(df.index >= start_dt) & (df.index < end_dt)]


def run_single_window(
    window: WFOWindow,
    window_id: int,
    config_name: str,
    hyperparams: Dict,
    df_5min: pd.DataFrame,
    output_dir: Path,
) -> Optional[WindowResult]:
    """Train and test a configuration on a single WFO window."""
    try:
        # Filter data for this window
        train_data = filter_data_by_period(df_5min, window.train_start, window.train_end)
        test_data = filter_data_by_period(df_5min, window.test_start, window.test_end)

        if len(train_data) < 1000 or len(test_data) < 100:
            logger.warning(f"Insufficient data for window {window_id}")
            return None

        # Create ensemble config
        # Disable stacking for HPO to avoid OOF issues with limited data per window
        # Stacking can be re-enabled for final production training
        ensemble_config = MTFEnsembleConfig(
            weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
            agreement_bonus=0.05,
            use_regime_adjustment=True,
            include_sentiment=True,
            trading_pair="EURUSD",
            sentiment_source="epu",
            sentiment_by_timeframe={"1H": False, "4H": False, "D": True},
            use_stacking=False,  # Disable for HPO (faster, avoids OOF issues)
            use_rfecv=False,  # Disable for speed during HPO
            optimized_hyperparams=hyperparams,
        )

        # Create and train ensemble
        model_dir = output_dir / f"hpo_{config_name}_w{window_id}"
        ensemble = MTFEnsemble(config=ensemble_config, model_dir=model_dir)

        # Train on train period only
        ensemble.train(train_data, train_ratio=0.8, val_ratio=0.2)

        # Run backtest on test period using the WFO backtest function
        backtest_results = run_window_backtest(
            ensemble=ensemble,
            df_5min=test_data,
            min_confidence=0.55,
            tp_pips=25.0,
            sl_pips=15.0,
            max_holding_bars=12,
            initial_balance=10000.0,
            risk_per_trade=0.02,
            reduce_risk_on_losses=True,
        )

        return WindowResult(
            window_id=window_id,
            test_period=f"{window.test_start} to {window.test_end}",
            total_trades=backtest_results["total_trades"],
            win_rate=backtest_results["win_rate"] / 100.0,  # Convert from percentage to fraction
            profit_factor=backtest_results["profit_factor"],
            total_pips=backtest_results["total_pips"],
        )

    except Exception as e:
        logger.error(f"Failed window {window_id} for {config_name}: {e}")
        return None


def run_config_wfo(
    config_name: str,
    config: Dict,
    df_5min: pd.DataFrame,
    windows: List[WFOWindow],
    output_dir: Path,
) -> Optional[ConfigResult]:
    """Run a configuration through all WFO windows."""
    logger.info(f"\n{'='*70}")
    logger.info(f"TESTING: {config_name.upper()}")
    logger.info(f"Description: {config['description']}")
    logger.info(f"Windows: {len(windows)}")
    logger.info(f"{'='*70}")

    # Extract hyperparams (excluding description)
    hyperparams = {k: v for k, v in config.items() if k != "description"}

    window_results = []

    for i, window in enumerate(windows):
        logger.info(f"\n--- Window {i+1}/{len(windows)}: Test {window.test_start} to {window.test_end} ---")

        result = run_single_window(
            window=window,
            window_id=i + 1,
            config_name=config_name,
            hyperparams=hyperparams,
            df_5min=df_5min,
            output_dir=output_dir,
        )

        if result:
            window_results.append(result)
            logger.info(f"  Pips: {result.total_pips:+.1f}, WR: {result.win_rate:.1%}, PF: {result.profit_factor:.2f}")

    if not window_results:
        logger.error(f"No valid results for {config_name}")
        return None

    # Aggregate results
    total_pips = sum(r.total_pips for r in window_results)
    pips_values = [r.total_pips for r in window_results]
    profitable_windows = sum(1 for r in window_results if r.total_pips > 0)

    return ConfigResult(
        config_name=config_name,
        description=config["description"],
        hyperparams=hyperparams,
        window_results=window_results,
        total_pips=total_pips,
        avg_pips_per_window=np.mean(pips_values),
        pips_std=np.std(pips_values),
        avg_win_rate=np.mean([r.win_rate for r in window_results]),
        avg_profit_factor=np.mean([r.profit_factor for r in window_results if r.profit_factor < float("inf")]),
        profitable_windows=profitable_windows,
        total_windows=len(window_results),
        consistency_score=profitable_windows / len(window_results),
    )


def main():
    """Run WFO-based hyperparameter comparison."""
    data_path = project_root / "data" / "forex" / "EURUSD_20200101_20251231_5min_combined.csv"
    output_dir = project_root / "models" / "hpo_wfo_test"
    results_file = project_root / "data" / "hyperparameter_wfo_comparison.json"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df_5min = load_data(data_path)

    # Create WFO windows
    windows = create_wfo_windows()

    print("\n" + "=" * 80)
    print("WALK-FORWARD HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Testing {len(CONFIGS)} configurations across {len(windows)} WFO windows")
    print("\nWFO Windows:")
    for i, w in enumerate(windows):
        print(f"  {i+1}. Train: {w.train_start} to {w.train_end} | Test: {w.test_start} to {w.test_end}")
    print("=" * 80 + "\n")

    all_results = []

    for config_name, config in CONFIGS.items():
        result = run_config_wfo(
            config_name=config_name,
            config=config,
            df_5min=df_5min,
            windows=windows,
            output_dir=output_dir,
        )
        if result:
            all_results.append(result)

    # Save results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "methodology": "walk-forward-optimization",
        "num_windows": len(windows),
        "windows": [
            {
                "train": f"{w.train_start} to {w.train_end}",
                "test": f"{w.test_start} to {w.test_end}",
            }
            for w in windows
        ],
        "configs_tested": len(all_results),
        "results": [
            {
                "config_name": r.config_name,
                "description": r.description,
                "hyperparams": r.hyperparams,
                "total_pips": r.total_pips,
                "avg_pips_per_window": r.avg_pips_per_window,
                "pips_std": r.pips_std,
                "avg_win_rate": r.avg_win_rate,
                "avg_profit_factor": r.avg_profit_factor,
                "profitable_windows": r.profitable_windows,
                "total_windows": r.total_windows,
                "consistency_score": r.consistency_score,
                "window_details": [
                    {
                        "window_id": wr.window_id,
                        "test_period": wr.test_period,
                        "total_trades": wr.total_trades,
                        "win_rate": wr.win_rate,
                        "profit_factor": wr.profit_factor,
                        "total_pips": wr.total_pips,
                    }
                    for wr in r.window_results
                ],
            }
            for r in all_results
        ],
    }

    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")

    # Print comparison table
    print("\n" + "=" * 100)
    print("WFO COMPARISON SUMMARY")
    print("=" * 100)
    print(f"{'Config':<15} {'Total Pips':>12} {'Avg/Window':>12} {'Std':>10} {'WR':>8} {'PF':>8} {'Consist':>10}")
    print("-" * 100)

    # Sort by total pips (primary) and consistency (secondary)
    sorted_results = sorted(
        all_results,
        key=lambda x: (x.consistency_score, x.total_pips),
        reverse=True,
    )

    for r in sorted_results:
        consist_str = f"{r.profitable_windows}/{r.total_windows}"
        print(
            f"{r.config_name:<15} "
            f"{r.total_pips:>+12.1f} "
            f"{r.avg_pips_per_window:>+12.1f} "
            f"{r.pips_std:>10.1f} "
            f"{r.avg_win_rate:>7.1%} "
            f"{r.avg_profit_factor:>8.2f} "
            f"{consist_str:>10}"
        )

    print("=" * 100)

    # Print detailed window breakdown for top 3
    print("\n" + "=" * 100)
    print("WINDOW-BY-WINDOW BREAKDOWN (Top 3 Configs)")
    print("=" * 100)

    for r in sorted_results[:3]:
        print(f"\n{r.config_name.upper()} - {r.description}")
        print("-" * 80)
        print(f"{'Window':<12} {'Test Period':<25} {'Trades':>8} {'WR':>8} {'PF':>8} {'Pips':>10}")
        print("-" * 80)

        for wr in r.window_results:
            print(
                f"{'Window '+str(wr.window_id):<12} "
                f"{wr.test_period:<25} "
                f"{wr.total_trades:>8} "
                f"{wr.win_rate:>7.1%} "
                f"{wr.profit_factor:>8.2f} "
                f"{wr.total_pips:>+10.1f}"
            )

        print(f"{'TOTAL':<12} {'':<25} {'':<8} {r.avg_win_rate:>7.1%} {r.avg_profit_factor:>8.2f} {r.total_pips:>+10.1f}")

    # Check if we have results
    if not sorted_results:
        print("\nERROR: No valid results from any configuration!")
        print("Check the logs above for failures.")
        return []

    # Identify best config
    best = sorted_results[0]
    baseline = next((r for r in all_results if r.config_name == "baseline"), None)

    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    print(f"Best Config: {best.config_name}")
    print(f"  Total Pips (across {best.total_windows} windows): {best.total_pips:+.1f}")
    print(f"  Consistency: {best.profitable_windows}/{best.total_windows} windows profitable")
    print(f"  Avg Win Rate: {best.avg_win_rate:.1%}")
    print(f"  Avg Profit Factor: {best.avg_profit_factor:.2f}")

    if baseline and best.config_name != "baseline":
        pips_change = best.total_pips - baseline.total_pips
        pips_change_pct = (pips_change / abs(baseline.total_pips)) * 100 if baseline.total_pips != 0 else 0
        print(f"\n  vs Baseline:")
        print(f"    Pips Change: {pips_change:+.1f} ({pips_change_pct:+.1f}%)")
        print(f"    Baseline Total: {baseline.total_pips:+.1f}")

    print("=" * 100)

    return sorted_results


if __name__ == "__main__":
    main()
