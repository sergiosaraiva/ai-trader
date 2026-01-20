#!/usr/bin/env python3
"""Train Multi-Timeframe Ensemble model.

This script trains 3 XGBoost models at different timeframes (1H, 4H, Daily)
and combines them into a weighted ensemble:
- 1H (Short-term): 60% weight - entry timing
- 4H (Medium-term): 30% weight - trend confirmation
- Daily (Long-term): 10% weight - regime context

The ensemble reduces noise through higher timeframe filtering.
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
    MTFEnsemble,
    MTFEnsembleConfig,
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
        # Try to load with unnamed timestamp column (index_col=0)
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    df.columns = [c.lower() for c in df.columns]

    # If index is not datetime, look for a timestamp column
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
            # Try to convert index to datetime
            df.index = pd.to_datetime(df.index)

    df = df.sort_index()
    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    return df


def validate_ensemble(
    ensemble: MTFEnsemble,
    df_5min: pd.DataFrame,
    test_start_idx: int,
) -> dict:
    """Validate ensemble on held-out data."""
    from src.features.technical.calculator import TechnicalIndicatorCalculator

    logger.info("Validating ensemble on test data...")

    # Prepare test data for each timeframe
    timeframes = ["1H", "4H", "D"]
    X_dict = {}
    y_dict = {}

    for tf in timeframes:
        model = ensemble.models[tf]
        config = ensemble.model_configs[tf]

        # Resample
        df_tf = ensemble.resample_data(df_5min, config.base_timeframe)

        # Prepare higher TF data
        higher_tf_data = ensemble.prepare_higher_tf_data(df_5min, config.base_timeframe)

        # Get features and labels
        X, y, _ = model.prepare_data(df_tf, higher_tf_data)

        # Split
        n_total = len(X)
        n_train = int(n_total * 0.6)
        n_val = int(n_total * 0.2)
        test_start = n_train + n_val

        X_dict[tf] = X[test_start:]
        y_dict[tf] = y[test_start:]

        logger.info(f"{tf}: {len(X_dict[tf])} test samples")

    # Get common length (use minimum)
    min_len = min(len(X_dict[tf]) for tf in timeframes)

    # Truncate to common length
    for tf in timeframes:
        X_dict[tf] = X_dict[tf][:min_len]
        y_dict[tf] = y_dict[tf][:min_len]

    # Get ensemble predictions
    directions, confidences, agreement_scores = ensemble.predict_batch(X_dict)

    # Use 1H labels as ground truth (since we're predicting short-term)
    y_test = y_dict["1H"][:len(directions)]

    # Calculate metrics
    accuracy = (directions == y_test).mean()

    results = {
        "test_samples": len(directions),
        "accuracy": accuracy,
        "mean_confidence": confidences.mean(),
        "mean_agreement": agreement_scores.mean(),
    }

    # Accuracy at confidence levels
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        mask = confidences >= thresh
        if mask.sum() > 0:
            acc = (directions[mask] == y_test[mask]).mean()
            results[f"acc_conf_{int(thresh*100)}"] = acc
            results[f"samples_conf_{int(thresh*100)}"] = int(mask.sum())

    # Accuracy by agreement
    full_agree_mask = agreement_scores == 1.0
    if full_agree_mask.sum() > 0:
        acc = (directions[full_agree_mask] == y_test[full_agree_mask]).mean()
        results["acc_full_agreement"] = acc
        results["samples_full_agreement"] = int(full_agree_mask.sum())

    logger.info(f"Ensemble test accuracy: {accuracy:.2%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train MTF Ensemble")
    parser.add_argument(
        "--data",
        type=str,
        default="data/forex/EURUSD_20200101_20251231_5min_combined.csv",
        help="Path to 5-minute data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/mtf_ensemble",
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
        default="1H,4H,D",
        help="Comma-separated timeframes to train",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="0.6,0.3,0.1",
        help="Comma-separated weights for 1H,4H,D",
    )
    parser.add_argument(
        "--sentiment",
        action="store_true",
        help="Include sentiment features (Daily model only, research-based)",
    )
    parser.add_argument(
        "--sentiment-all",
        action="store_true",
        help="Include sentiment for ALL timeframes (not recommended for monthly EPU data)",
    )
    parser.add_argument(
        "--sentiment-tf",
        type=str,
        default=None,
        help="Custom sentiment timeframes (e.g., '4H,D' or '1H,4H,D'). Overrides --sentiment/--sentiment-all",
    )
    parser.add_argument(
        "--pair",
        type=str,
        default="EURUSD",
        help="Trading pair for sentiment data",
    )
    parser.add_argument(
        "--sentiment-source",
        type=str,
        default="epu",
        choices=["epu", "gdelt", "both"],
        help="Sentiment data source: 'epu' (daily VIX/EPU), 'gdelt' (hourly news), or 'both'",
    )
    args = parser.parse_args()

    # Parse weights and timeframes first
    weight_values = [float(w) for w in args.weights.split(",")]
    timeframe_list = [tf.strip() for tf in args.timeframes.split(",")]

    if len(weight_values) != len(timeframe_list):
        logger.error("Number of weights must match number of timeframes")
        return

    weights = dict(zip(timeframe_list, weight_values))

    # Determine sentiment configuration (priority: --sentiment-tf > --sentiment-all > --sentiment)
    if args.sentiment_tf:
        # Custom sentiment timeframes specified
        sentiment_tfs = [tf.strip().upper() for tf in args.sentiment_tf.split(",")]
        sentiment_by_tf = {tf: (tf in sentiment_tfs) for tf in timeframe_list}
        sentiment_mode = f"custom ({args.sentiment_tf})"
        include_sentiment = any(sentiment_by_tf.values())
    elif args.sentiment_all:
        # Full sentiment for all timeframes (for testing)
        sentiment_by_tf = {tf: True for tf in timeframe_list}
        sentiment_mode = "ALL timeframes"
        include_sentiment = True
    elif args.sentiment:
        # Research-based: sentiment only for Daily model
        sentiment_by_tf = {tf: (tf == "D") for tf in timeframe_list}
        sentiment_mode = "Daily only (research-based)"
        include_sentiment = True
    else:
        sentiment_by_tf = {tf: False for tf in timeframe_list}
        sentiment_mode = "disabled"
        include_sentiment = False

    # Determine sentiment source description
    sentiment_source = args.sentiment_source
    source_desc = {
        "epu": "EPU/VIX (daily)",
        "gdelt": "GDELT (hourly news)",
        "both": "EPU/VIX + GDELT (combined)",
    }.get(sentiment_source, sentiment_source)

    print("\n" + "=" * 70)
    print("MTF ENSEMBLE TRAINING")
    if args.sentiment_tf:
        print(f"(3-Timeframe Ensemble + CUSTOM SENTIMENT: {args.sentiment_tf})")
    elif args.sentiment_all:
        print("(3-Timeframe Ensemble + FULL SENTIMENT - all timeframes)")
    elif args.sentiment:
        print("(3-Timeframe Ensemble + SENTIMENT - Daily only, research-based)")
    else:
        print("(3-Timeframe Weighted Ensemble)")
    print("=" * 70)
    print(f"Data:        {args.data}")
    print(f"Output:      {args.output}")
    print(f"Timeframes:  {args.timeframes}")
    print(f"Weights:     {args.weights}")
    print(f"Train ratio: {args.train_ratio:.0%}")
    print(f"Val ratio:   {args.val_ratio:.0%}")
    print(f"Sentiment:   {sentiment_mode}")
    if include_sentiment:
        print(f"  Source:    {source_desc}")
    for tf in timeframe_list:
        status = "ON" if sentiment_by_tf.get(tf, False) else "OFF"
        print(f"  - {tf}: {status}")
    if include_sentiment:
        print(f"Pair:        {args.pair}")
    print("=" * 70)

    config = MTFEnsembleConfig(
        weights=weights,
        agreement_bonus=0.05,
        use_regime_adjustment=True,
        include_sentiment=include_sentiment,
        trading_pair=args.pair,
        sentiment_source=sentiment_source,
        sentiment_by_timeframe=sentiment_by_tf,
    )

    # Load data
    data_path = project_root / args.data
    df_5min = load_data(data_path)

    # Create ensemble
    model_dir = project_root / args.output
    ensemble = MTFEnsemble(config=config, model_dir=model_dir)

    # Train all models
    print("\n" + "=" * 70)
    print("TRAINING INDIVIDUAL MODELS")
    print("=" * 70)

    results = ensemble.train(
        df_5min,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        timeframes=timeframe_list,
    )

    # Print individual model results
    print("\n" + "=" * 70)
    print("INDIVIDUAL MODEL RESULTS")
    print("=" * 70)

    for tf, metrics in results.items():
        print(f"\n{tf} Model:")
        print(f"  Train Samples:  {metrics.get('train_samples', 'N/A')}")
        print(f"  Val Samples:    {metrics.get('val_samples', 'N/A')}")
        print(f"  Train Accuracy: {metrics['train_accuracy']:.2%}")
        print(f"  Val Accuracy:   {metrics['val_accuracy']:.2%}")

        for thresh in [55, 60, 65, 70]:
            key = f"val_acc_conf_{thresh}"
            samples_key = f"val_samples_conf_{thresh}"
            if key in metrics:
                print(f"  Val Acc @ {thresh}%: {metrics[key]:.2%} ({metrics.get(samples_key, 0)} samples)")

    # Validate ensemble
    print("\n" + "=" * 70)
    print("ENSEMBLE VALIDATION")
    print("=" * 70)

    test_start_idx = int(len(df_5min) * (args.train_ratio + args.val_ratio))
    ensemble_results = validate_ensemble(ensemble, df_5min, test_start_idx)

    print(f"\nEnsemble Test Results:")
    print(f"  Test Samples:    {ensemble_results['test_samples']}")
    print(f"  Accuracy:        {ensemble_results['accuracy']:.2%}")
    print(f"  Mean Confidence: {ensemble_results['mean_confidence']:.2%}")
    print(f"  Mean Agreement:  {ensemble_results['mean_agreement']:.2%}")

    print("\nAccuracy by Confidence Level:")
    for thresh in [55, 60, 65, 70]:
        key = f"acc_conf_{thresh}"
        samples_key = f"samples_conf_{thresh}"
        if key in ensemble_results:
            print(f"  Conf >= {thresh}%: {ensemble_results[key]:.2%} ({ensemble_results.get(samples_key, 0)} samples)")

    if "acc_full_agreement" in ensemble_results:
        print(f"\nFull Agreement (all 3 models agree):")
        print(f"  Accuracy: {ensemble_results['acc_full_agreement']:.2%}")
        print(f"  Samples:  {ensemble_results['samples_full_agreement']}")

    # Save ensemble
    print("\n" + "=" * 70)
    print("SAVING ENSEMBLE")
    print("=" * 70)

    ensemble.save()

    # Save training metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "data_file": str(args.data),
        "data_bars": len(df_5min),
        "data_start": str(df_5min.index[0]),
        "data_end": str(df_5min.index[-1]),
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "timeframes": timeframe_list,
        "weights": weights,
        "include_sentiment": include_sentiment,
        "sentiment_mode": sentiment_mode,
        "sentiment_source": sentiment_source if include_sentiment else None,
        "sentiment_by_timeframe": sentiment_by_tf,
        "trading_pair": args.pair if include_sentiment else None,
        "individual_results": {
            k: {kk: float(vv) if isinstance(vv, (np.integer, np.floating)) else vv
                for kk, vv in v.items()}
            for k, v in results.items()
        },
        "ensemble_results": {
            k: float(v) if isinstance(v, (np.integer, np.floating)) else v
            for k, v in ensemble_results.items()
        },
    }

    metadata_path = model_dir / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {metadata_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<12} {'Weight':>8} {'Val Acc':>10} {'Acc@60%':>10}")
    print("-" * 45)

    for tf in timeframe_list:
        if tf in results:
            weight = weights.get(tf, 0)
            val_acc = results[tf]["val_accuracy"]
            acc_60 = results[tf].get("val_acc_conf_60", 0)
            print(f"{tf:<12} {weight:>7.0%} {val_acc:>9.2%} {acc_60:>9.2%}")

    print("-" * 45)
    print(f"{'ENSEMBLE':<12} {'100%':>8} {ensemble_results['accuracy']:>9.2%} {ensemble_results.get('acc_conf_60', 0):>9.2%}")

    # Compare to 1H alone
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    h1_val_acc = results.get("1H", {}).get("val_accuracy", 0)
    ensemble_acc = ensemble_results["accuracy"]
    improvement = ensemble_acc - h1_val_acc

    if improvement > 0:
        print(f"  Ensemble improvement over 1H alone: +{improvement:.2%}")
    else:
        print(f"  Ensemble vs 1H alone: {improvement:+.2%}")

    # Check targets
    print("\nTarget Check:")
    if ensemble_acc >= 0.59:
        print(f"  [OK] Accuracy {ensemble_acc:.1%} >= 59% target")
    else:
        print(f"  [!!] Accuracy {ensemble_acc:.1%} < 59% target")

    acc_60 = ensemble_results.get("acc_conf_60", 0)
    if acc_60 >= 0.65:
        print(f"  [OK] High-conf accuracy {acc_60:.1%} >= 65% target")
    elif acc_60 >= 0.60:
        print(f"  [OK] High-conf accuracy {acc_60:.1%} >= 60%")
    else:
        print(f"  [!!] High-conf accuracy {acc_60:.1%} < 60%")

    print("\n" + "=" * 70)
    print(ensemble.summary())
    print("\nTraining complete!")
    print(f"Models saved to: {model_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
