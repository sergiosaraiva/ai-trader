#!/usr/bin/env python3
"""Train hybrid ensemble model (XGBoost + CNN-Transformer).

This script trains the hybrid model that combines:
- XGBoost: Feature-based patterns
- CNN-Transformer: Sequence-based patterns
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

from src.models.multi_timeframe import (
    HybridConfig,
    HybridEnsemble,
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
        df = pd.read_csv(data_path)

    df.columns = [c.lower() for c in df.columns]

    # Handle timestamp column
    time_col = None
    for col in ["timestamp", "time", "date", "datetime"]:
        if col in df.columns:
            time_col = col
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

    df = df.sort_index()
    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    return df


def prepare_higher_tf_data(df_5min: pd.DataFrame) -> dict:
    """Prepare higher timeframe data for XGBoost cross-TF features."""
    from src.features.technical.calculator import TechnicalIndicatorCalculator
    calc = TechnicalIndicatorCalculator(model_type="short_term")

    higher_tf_data = {}
    for tf in ["4H", "D"]:
        df_tf = df_5min.resample(tf).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in df_5min.columns else "first",
        }).dropna()
        higher_tf_data[tf] = calc.calculate(df_tf)

    return higher_tf_data


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Ensemble Model")
    parser.add_argument(
        "--data",
        type=str,
        default="data/forex/EURUSD_20200101_20251231_5min_combined.csv",
        help="Path to 5-minute data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/hybrid",
        help="Output directory for models",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1H",
        help="Timeframe to train (1H or 4H)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Training data ratio",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation data ratio",
    )
    parser.add_argument(
        "--xgboost-weight",
        type=float,
        default=0.6,
        help="Weight for XGBoost model in ensemble",
    )
    parser.add_argument(
        "--sequence-weight",
        type=float,
        default=0.4,
        help="Weight for sequence model in ensemble",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("HYBRID ENSEMBLE MODEL TRAINING")
    print("=" * 70)
    print(f"Data:            {args.data}")
    print(f"Output:          {args.output}")
    print(f"Timeframe:       {args.timeframe}")
    print(f"Train ratio:     {args.train_ratio:.0%}")
    print(f"Val ratio:       {args.val_ratio:.0%}")
    print(f"XGBoost weight:  {args.xgboost_weight}")
    print(f"Sequence weight: {args.sequence_weight}")
    print("=" * 70)

    # Load data
    data_path = project_root / args.data
    df_5min = load_data(data_path)

    # Prepare higher TF data for cross-TF features
    logger.info("Preparing higher timeframe data...")
    higher_tf_data = prepare_higher_tf_data(df_5min)

    # Create config
    if args.timeframe == "1H":
        config = HybridConfig.hourly()
    elif args.timeframe == "4H":
        config = HybridConfig.four_hour()
    else:
        raise ValueError(f"Unknown timeframe: {args.timeframe}")

    config.xgboost_weight = args.xgboost_weight
    config.sequence_weight = args.sequence_weight

    # Create model
    model_dir = project_root / args.output
    ensemble = HybridEnsemble(config=config, model_dir=model_dir)

    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    results = ensemble.train(
        df_5min,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        higher_tf_data=higher_tf_data,
    )

    # Save models
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)
    ensemble.save()

    # Save training metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "data_file": str(args.data),
        "data_bars": len(df_5min),
        "data_start": str(df_5min.index[0]),
        "data_end": str(df_5min.index[-1]),
        "timeframe": args.timeframe,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "xgboost_weight": args.xgboost_weight,
        "sequence_weight": args.sequence_weight,
        "results": results,
    }

    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj

    metadata_path = model_dir / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(convert_numpy(metadata), f, indent=2)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Val Accuracy':>15}")
    print("-" * 40)
    print(f"{'XGBoost':<20} {results['xgboost_val_acc']:>14.2%}")
    print(f"{'Sequence (CNN-TF)':<20} {results['sequence_val_acc']:>14.2%}")
    print(f"{'Hybrid Ensemble':<20} {results['ensemble_val_acc']:>14.2%}")

    # Improvement
    xgb_acc = results['xgboost_val_acc']
    ens_acc = results['ensemble_val_acc']
    improvement = ens_acc - xgb_acc

    print("\n" + "-" * 40)
    if improvement > 0:
        print(f"Ensemble improvement over XGBoost: +{improvement:.2%}")
    else:
        print(f"Ensemble vs XGBoost: {improvement:+.2%}")

    # Accuracy at confidence levels
    print("\n" + "=" * 70)
    print("ENSEMBLE ACCURACY BY CONFIDENCE")
    print("=" * 70)

    for thresh in [55, 60, 65, 70]:
        key = f"ensemble_val_acc_conf_{thresh}"
        samples_key = f"ensemble_val_samples_conf_{thresh}"
        if key in results:
            print(f"Conf >= {thresh}%: {results[key]:.2%} ({results.get(samples_key, 0)} samples)")

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Models saved to: {model_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
