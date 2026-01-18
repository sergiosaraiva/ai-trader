#!/usr/bin/env python3
"""Train improved multi-timeframe models.

This script trains models with:
- Triple barrier labeling (realistic trade outcomes)
- Longer prediction horizons (1H, 4H)
- Enhanced features (time, patterns, cross-TF)
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
    ImprovedModelConfig,
    ImprovedMultiTimeframeModel,
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


def main():
    parser = argparse.ArgumentParser(description="Train Improved MTF Models")
    parser.add_argument(
        "--data",
        type=str,
        default="data/forex/EURUSD_20200101_20251231_5min_combined.csv",
        help="Path to 5-minute data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/improved_mtf",
        help="Output directory for models",
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
        "--timeframes",
        type=str,
        default="1H,4H",
        help="Comma-separated list of timeframes to train",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("IMPROVED MULTI-TIMEFRAME MODEL TRAINING")
    print("=" * 70)
    print(f"Data:        {args.data}")
    print(f"Output:      {args.output}")
    print(f"Timeframes:  {args.timeframes}")
    print(f"Train ratio: {args.train_ratio:.0%}")
    print(f"Val ratio:   {args.val_ratio:.0%}")
    print("=" * 70)

    # Load data
    data_path = project_root / args.data
    df_5min = load_data(data_path)

    # Create model configs based on requested timeframes
    configs = []
    for tf in args.timeframes.split(","):
        tf = tf.strip()
        if tf == "1H":
            configs.append(ImprovedModelConfig.hourly_model())
        elif tf == "4H":
            configs.append(ImprovedModelConfig.four_hour_model())
        elif tf == "D":
            configs.append(ImprovedModelConfig.daily_model())
        else:
            logger.warning(f"Unknown timeframe: {tf}")

    if not configs:
        logger.error("No valid timeframes specified")
        return

    # Create model
    model_dir = project_root / args.output
    model = ImprovedMultiTimeframeModel(configs=configs, model_dir=model_dir)

    # Train all models
    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    results = model.train_all(
        df_5min,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    # Print results
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)

    for timeframe, metrics in results.items():
        print(f"\n{timeframe}:")
        print(f"  Train Samples:  {metrics.get('train_samples', 'N/A')}")
        print(f"  Val Samples:    {metrics.get('val_samples', 'N/A')}")
        print(f"  Train Accuracy: {metrics['train_accuracy']:.2%}")
        print(f"  Val Accuracy:   {metrics['val_accuracy']:.2%}")

        # Print accuracy at different confidence levels
        for thresh in [55, 60, 65, 70]:
            key = f"val_acc_conf_{thresh}"
            samples_key = f"val_samples_conf_{thresh}"
            if key in metrics:
                print(f"  Val Acc @ {thresh}%: {metrics[key]:.2%} ({metrics.get(samples_key, 0)} samples)")

    # Save models
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)
    model.save_all()

    # Save training metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "data_file": str(args.data),
        "data_bars": len(df_5min),
        "data_start": str(df_5min.index[0]),
        "data_end": str(df_5min.index[-1]),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "timeframes": args.timeframes.split(","),
        "results": results,
        "improvements": [
            "Triple barrier labeling",
            "Longer prediction horizons",
            "Enhanced features (time, patterns, ROC)",
            "Cross-timeframe alignment",
            "XGBoost with regularization",
        ],
    }

    # Convert numpy types to Python types
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

    print(f"Saved metadata to {metadata_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Timeframe':<12} {'Val Acc':>10} {'Acc@60%':>10} {'Samples@60%':>12}")
    print("-" * 50)

    for timeframe, metrics in results.items():
        val_acc = metrics["val_accuracy"]
        acc_60 = metrics.get("val_acc_conf_60", 0)
        samples_60 = metrics.get("val_samples_conf_60", 0)
        print(f"{timeframe:<12} {val_acc:>9.2%} {acc_60:>9.2%} {samples_60:>12}")

    print("=" * 70)

    # Analysis
    print("\nANALYSIS:")
    for timeframe, metrics in results.items():
        val_acc = metrics["val_accuracy"]
        acc_60 = metrics.get("val_acc_conf_60", 0)
        acc_65 = metrics.get("val_acc_conf_65", 0)

        if val_acc > 0.55:
            print(f"  [{timeframe}] Base accuracy {val_acc:.1%} exceeds 55% target!")
        else:
            print(f"  [{timeframe}] Base accuracy {val_acc:.1%} - below 55% target")

        if acc_60 > 0.55:
            print(f"  [{timeframe}] At 60% confidence: {acc_60:.1%} accuracy")
        if acc_65 > 0.55:
            print(f"  [{timeframe}] At 65% confidence: {acc_65:.1%} accuracy")

    print("=" * 70)
    print("Training complete!")


if __name__ == "__main__":
    main()
