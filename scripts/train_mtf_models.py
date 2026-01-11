#!/usr/bin/env python3
"""Train multi-timeframe models for the scalper system.

This script trains models for 5min, 15min, and 30min timeframes
and saves them for use in the trading system.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.models.multi_timeframe import (
    MultiTimeframeModel,
    TimeframeConfig,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_5min_data(data_path: Path) -> pd.DataFrame:
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
    parser = argparse.ArgumentParser(description="Train MTF models")
    parser.add_argument(
        "--data",
        type=str,
        default="data/forex/EURUSD_20200101_20251231_5min_combined.csv",
        help="Path to 5-minute data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/mtf_scalper",
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
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("MULTI-TIMEFRAME MODEL TRAINING")
    print("=" * 70)
    print(f"Data:        {args.data}")
    print(f"Output:      {args.output}")
    print(f"Train ratio: {args.train_ratio:.0%}")
    print(f"Val ratio:   {args.val_ratio:.0%}")
    print("=" * 70)

    # Load data
    data_path = project_root / args.data
    df_5min = load_5min_data(data_path)

    # Create model with scalper configs
    configs = [
        TimeframeConfig.scalper_5min(),
        TimeframeConfig.scalper_15min(),
        TimeframeConfig.scalper_30min(),
    ]

    model_dir = project_root / args.output
    mtf_model = MultiTimeframeModel(configs=configs, model_dir=model_dir)

    # Train all models
    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    results = mtf_model.train_all(
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
        print(f"  Train Accuracy: {metrics['train_accuracy']:.2%}")
        print(f"  Val Accuracy:   {metrics['val_accuracy']:.2%}")

        # Print accuracy at different confidence levels
        for thresh in [55, 60, 65, 70]:
            key = f"val_acc_conf_{thresh}"
            samples_key = f"val_samples_conf_{thresh}"
            if key in metrics:
                print(f"  Val Acc @ {thresh}%: {metrics[key]:.2%} ({metrics[samples_key]} samples)")

    # Save models
    print("\n" + "=" * 70)
    print("SAVING MODELS")
    print("=" * 70)
    mtf_model.save_all()

    # Save training metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "data_file": str(args.data),
        "data_bars": len(df_5min),
        "data_start": str(df_5min.index[0]),
        "data_end": str(df_5min.index[-1]),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "results": results,
    }

    import json
    metadata_path = model_dir / "training_metadata.json"

    # Convert numpy types to Python types
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj

    with open(metadata_path, "w") as f:
        json.dump(convert_numpy(metadata), f, indent=2)

    print(f"Saved metadata to {metadata_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Timeframe':<12} {'Val Acc':>10} {'Acc@65%':>10} {'Samples@65%':>12}")
    print("-" * 50)

    for timeframe, metrics in results.items():
        val_acc = metrics["val_accuracy"]
        acc_65 = metrics.get("val_acc_conf_65", 0)
        samples_65 = metrics.get("val_samples_conf_65", 0)
        print(f"{timeframe:<12} {val_acc:>9.2%} {acc_65:>9.2%} {samples_65:>12}")

    print("=" * 70)
    print("Training complete!")


if __name__ == "__main__":
    main()
