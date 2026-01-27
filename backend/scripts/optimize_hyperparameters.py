#!/usr/bin/env python3
"""Optimize XGBoost hyperparameters using Optuna for MTF Ensemble.

This script uses Bayesian optimization (Optuna) to find optimal XGBoost
hyperparameters for each timeframe model (1H, 4H, Daily). It uses
TimeSeriesSplit for cross-validation to prevent data leakage.

The best hyperparameters are saved to backend/data/optimized_hyperparams.json
and can be loaded by train_mtf_ensemble.py with --use-optimized-params flag.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src.models.multi_timeframe.improved_model import (
    ImprovedTimeframeModel,
    ImprovedModelConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load 5-minute OHLCV data."""
    logger.info(f"Loading data from {data_path}")

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    df.columns = [c.lower() for c in df.columns]

    if not isinstance(df.index, pd.DatetimeIndex):
        time_col = None
        for col in ["timestamp", "time", "date", "datetime"]:
            if col in df.columns:
                time_col = col
                break

        if time_col:
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.set_index(time_col)
        else:
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    return df


def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample 5-minute data to target timeframe."""
    tf_map = {
        "1H": "1H",
        "4H": "4H",
        "D": "1D",
    }

    resampled = df.resample(tf_map[timeframe]).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    return resampled


def create_objective(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    timeframe: str = "1H",
) -> callable:
    """Create Optuna objective function for hyperparameter optimization.

    Args:
        X: Feature matrix
        y: Target labels
        n_splits: Number of TimeSeriesSplit folds
        timeframe: Timeframe identifier for logging

    Returns:
        Objective function for Optuna
    """

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: maximize validation accuracy."""

        # Define hyperparameter search space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "verbosity": 0,
        }

        # TimeSeriesSplit for cross-validation (CRITICAL: no random splits)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]

            # Train model with suggested hyperparameters
            # early_stopping_rounds in constructor for XGBoost >= 2.0
            model = XGBClassifier(**params, early_stopping_rounds=20)
            model.fit(
                X_train_fold,
                y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False,
            )

            # Evaluate on validation fold
            val_accuracy = model.score(X_val_fold, y_val_fold)
            cv_scores.append(val_accuracy)

            # Report intermediate score for pruning
            trial.report(val_accuracy, fold_idx)

            # Prune unpromising trials
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Return mean CV score
        mean_cv_score = np.mean(cv_scores)
        logger.info(f"Trial {trial.number}: CV accuracy = {mean_cv_score:.4f}")

        return mean_cv_score

    return objective


def optimize_timeframe(
    df_5min: pd.DataFrame,
    timeframe: str,
    n_trials: int,
    n_splits: int,
    sentiment: bool = False,
) -> dict:
    """Optimize hyperparameters for a single timeframe.

    Args:
        df_5min: 5-minute OHLCV data
        timeframe: "1H", "4H", or "D"
        n_trials: Number of Optuna trials
        n_splits: Number of TimeSeriesSplit folds
        sentiment: Whether to include sentiment features

    Returns:
        Dict with best params and optimization history
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"OPTIMIZING {timeframe} MODEL")
    logger.info(f"{'='*70}")

    # Create model config
    if timeframe == "1H":
        config = ImprovedModelConfig.hourly_model()
    elif timeframe == "4H":
        config = ImprovedModelConfig.four_hour_model()
    elif timeframe == "D":
        config = ImprovedModelConfig.daily_model()
        if sentiment:
            config.include_sentiment_features = True
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")

    # Resample data
    df_tf = resample_data(df_5min, timeframe)
    logger.info(f"Resampled to {len(df_tf)} {timeframe} bars")

    # Prepare features and labels
    model = ImprovedTimeframeModel(config)

    # Prepare higher TF data for cross-TF features
    higher_tf_data = {}
    if timeframe == "1H":
        higher_tf_data["4H"] = resample_data(df_5min, "4H")
        higher_tf_data["D"] = resample_data(df_5min, "D")
    elif timeframe == "4H":
        higher_tf_data["D"] = resample_data(df_5min, "D")

    X, y, feature_names = model.prepare_data(df_tf, higher_tf_data)

    # Use only training data for optimization (60% of data)
    n_train = int(len(X) * 0.6)
    X_train = X[:n_train]
    y_train = y[:n_train]

    logger.info(f"Using {len(X_train)} samples for optimization")
    logger.info(f"Running {n_trials} trials with {n_splits}-fold TimeSeriesSplit")

    # Create Optuna study with MedianPruner
    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        study_name=f"{timeframe}_optimization",
    )

    # Run optimization
    objective = create_objective(X_train, y_train, n_splits, timeframe)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value

    logger.info(f"\n{'-'*70}")
    logger.info(f"OPTIMIZATION RESULTS FOR {timeframe}")
    logger.info(f"{'-'*70}")
    logger.info(f"Best CV Accuracy: {best_score:.4f}")
    logger.info(f"Best Parameters:")
    for param, value in sorted(best_params.items()):
        logger.info(f"  {param}: {value}")
    logger.info(f"{'-'*70}\n")

    # Return results
    return {
        "timeframe": timeframe,
        "best_params": best_params,
        "best_cv_accuracy": float(best_score),
        "n_trials": len(study.trials),
        "optimization_history": [
            {
                "trial": t.number,
                "value": t.value,
                "params": t.params,
            }
            for t in study.trials
            if t.value is not None
        ],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Optimize XGBoost hyperparameters for MTF Ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default="data/forex/EURUSD_20200101_20251231_5min_combined.csv",
        help="Path to 5-minute OHLCV data",
    )

    # Timeframe selection
    parser.add_argument(
        "--timeframe",
        type=str,
        choices=["1H", "4H", "D", "all"],
        default="all",
        help="Timeframe to optimize (default: all)",
    )

    # Optimization arguments
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials per timeframe (default: 100)",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of TimeSeriesSplit folds for CV (default: 5)",
    )

    # Feature flags
    parser.add_argument(
        "--sentiment",
        action="store_true",
        help="Include sentiment features (Daily model only)",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="configs",
        help="Output directory for optimized hyperparameters (default: configs)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("BAYESIAN HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    print("=" * 70)
    print(f"Data:       {args.data}")
    print(f"Timeframe:  {args.timeframe}")
    print(f"Trials:     {args.n_trials} per timeframe")
    print(f"CV Folds:   {args.n_splits}")
    print(f"Sentiment:  {'Enabled (Daily only)' if args.sentiment else 'Disabled'}")
    print(f"Output:     {args.output_dir}")
    print("=" * 70)

    # Load data
    df_5min = load_data(Path(args.data))

    # Determine which timeframes to optimize
    if args.timeframe == "all":
        timeframes = ["1H", "4H", "D"]
    else:
        timeframes = [args.timeframe]

    # Optimize each timeframe
    results = {}
    for tf in timeframes:
        result = optimize_timeframe(
            df_5min,
            tf,
            args.n_trials,
            args.n_splits,
            sentiment=args.sentiment,
        )
        results[tf] = result

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    output_path = output_dir / "optimized_hyperparams.json"

    # Create output with metadata
    output_data = {
        "optimization_date": datetime.now().isoformat(),
        "data_path": args.data,
        "n_trials_per_timeframe": args.n_trials,
        "n_cv_folds": args.n_splits,
        "sentiment_enabled": args.sentiment,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nOptimization complete!")
    logger.info(f"Results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION SUMMARY")
    print("=" * 70)
    for tf, result in results.items():
        print(f"\n{tf} Model:")
        print(f"  Best CV Accuracy: {result['best_cv_accuracy']:.4f}")
        print(f"  Trials Completed: {result['n_trials']}")
    print("\n" + "=" * 70)
    print(f"\nTo use optimized parameters:")
    print(f"  python scripts/train_mtf_ensemble.py --use-optimized-params")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
