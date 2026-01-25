#!/usr/bin/env python3
"""Analyze MTF Ensemble features using SHAP.

This script loads trained MTF Ensemble models and performs SHAP analysis
to understand feature importance and identify features for potential removal.

SHAP (SHapley Additive exPlanations) provides model-agnostic explanations
by computing the contribution of each feature to predictions.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.models.multi_timeframe import MTFEnsemble, SHAPAnalyzer
from src.features.technical.calculator import TechnicalIndicatorCalculator

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


def analyze_timeframe(
    ensemble: MTFEnsemble,
    timeframe: str,
    df_5min: pd.DataFrame,
    n_samples: int,
    output_dir: Path,
) -> dict:
    """Run SHAP analysis for a single timeframe model.

    Args:
        ensemble: Trained MTF Ensemble
        timeframe: Timeframe to analyze (1H, 4H, D)
        df_5min: 5-minute OHLCV data
        n_samples: Number of samples for SHAP analysis
        output_dir: Directory to save results

    Returns:
        Analysis results dictionary
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"SHAP ANALYSIS: {timeframe} MODEL")
    logger.info(f"{'='*70}")

    model = ensemble.models[timeframe]
    config = ensemble.model_configs[timeframe]

    # Check if model is XGBoost
    if config.model_type != "xgboost":
        logger.warning(f"SHAP analysis only supports XGBoost (model is {config.model_type})")
        return {}

    # Resample data
    df_tf = ensemble.resample_data(df_5min, config.base_timeframe)

    # Prepare higher timeframe data
    higher_tf_data = ensemble.prepare_higher_tf_data(df_5min, config.base_timeframe)

    # Get features and labels
    X, y, feature_names = model.prepare_data(df_tf, higher_tf_data)

    # Use test set (last 20% chronologically)
    n_total = len(X)
    test_start = int(n_total * 0.8)
    X_test = X[test_start:]
    y_test = y[test_start:]

    logger.info(f"Test samples: {len(X_test)}")

    # Limit samples for SHAP (computational cost)
    if n_samples > 0 and len(X_test) > n_samples:
        logger.info(f"Limiting to {n_samples} samples for SHAP analysis")
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_shap = X_test[indices]
        y_shap = y_test[indices]
    else:
        X_shap = X_test
        y_shap = y_test

    # Filter features if RFECV was used
    if model.config.use_rfecv and model.selected_indices is not None:
        logger.info(f"Using RFECV selected features: {len(model.selected_features)}")
        X_shap = X_shap[:, model.selected_indices]
        feature_names_shap = model.selected_features
    else:
        feature_names_shap = feature_names

    # Scale features
    X_shap_scaled = model.scaler.transform(X_shap)

    # Create SHAP analyzer
    logger.info("Creating SHAP analyzer...")
    analyzer = SHAPAnalyzer(model.model, feature_names_shap)

    # Compute SHAP values
    shap_values = analyzer.compute_shap_values(X_shap_scaled)

    # Get feature importance
    importance_df = analyzer.get_feature_importance(method="mean_abs")

    # Save analysis
    tf_output_dir = output_dir / timeframe
    analyzer.save_analysis(tf_output_dir, X_shap_scaled, n_top_features=50)

    # Print top features
    logger.info(f"\nTop 20 features by SHAP importance:")
    for idx, row in importance_df.head(20).iterrows():
        logger.info(f"  {idx+1:2d}. {row['feature']:30s} {row['importance']:8.4f}")

    # Print bottom features (candidates for removal)
    logger.info(f"\nBottom 20 features (removal candidates):")
    for idx, row in importance_df.tail(20).iloc[::-1].iterrows():
        logger.info(f"  {row['feature']:30s} {row['importance']:8.4f}")

    # Analyze feature interactions
    logger.info("\nAnalyzing feature interactions...")
    try:
        interactions = analyzer.analyze_feature_interactions(X_shap_scaled, max_display=10)
        logger.info("Top 10 feature interactions:")
        for (feat1, feat2), strength in interactions.items():
            logger.info(f"  {feat1:20s} <-> {feat2:20s}: {strength:.4f}")

        # Save interactions
        interactions_df = pd.DataFrame([
            {"feature1": f1, "feature2": f2, "strength": s}
            for (f1, f2), s in interactions.items()
        ])
        interactions_df.to_csv(tf_output_dir / "feature_interactions.csv", index=False)
    except Exception as e:
        logger.warning(f"Failed to compute interactions: {e}")

    # Summary statistics
    results = {
        "timeframe": timeframe,
        "n_features": len(feature_names_shap),
        "n_samples_analyzed": len(X_shap),
        "top_10_features": importance_df.head(10)["feature"].tolist(),
        "bottom_10_features": importance_df.tail(10)["feature"].tolist(),
        "mean_shap_importance": float(importance_df["importance"].mean()),
        "std_shap_importance": float(importance_df["importance"].std()),
    }

    # Save summary
    with open(tf_output_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nSHAP analysis for {timeframe} complete")
    logger.info(f"Results saved to: {tf_output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MTF Ensemble features using SHAP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/mtf_ensemble",
        help="Directory containing trained MTF Ensemble models",
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default="data/forex/EURUSD_20200101_20251231_5min_combined.csv",
        help="Path to 5-minute OHLCV data",
    )

    # Analysis arguments
    parser.add_argument(
        "--timeframes",
        type=str,
        nargs="+",
        default=["1H", "4H", "D"],
        help="Timeframes to analyze (default: all)",
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of samples for SHAP analysis (0 = all, default: 1000)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default="models/shap_analysis",
        help="Output directory for SHAP analysis",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("MTF ENSEMBLE SHAP ANALYSIS")
    print("=" * 70)
    print(f"Model dir:   {args.model_dir}")
    print(f"Data:        {args.data}")
    print(f"Timeframes:  {', '.join(args.timeframes)}")
    print(f"N samples:   {args.n_samples if args.n_samples > 0 else 'all'}")
    print(f"Output:      {args.output}")
    print("=" * 70)

    # Load data
    df_5min = load_data(Path(args.data))

    # Load ensemble
    logger.info(f"\nLoading MTF Ensemble from {args.model_dir}...")
    ensemble = MTFEnsemble.load(Path(args.model_dir))
    logger.info("Ensemble loaded successfully")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run analysis for each timeframe
    all_results = {}

    for timeframe in args.timeframes:
        if timeframe not in ensemble.models:
            logger.warning(f"Timeframe {timeframe} not found in ensemble, skipping")
            continue

        try:
            results = analyze_timeframe(
                ensemble=ensemble,
                timeframe=timeframe,
                df_5min=df_5min,
                n_samples=args.n_samples,
                output_dir=output_dir,
            )
            all_results[timeframe] = results
        except Exception as e:
            logger.error(f"Failed to analyze {timeframe}: {e}", exc_info=True)

    # Save combined summary
    if all_results:
        with open(output_dir / "all_timeframes_summary.json", "w") as f:
            json.dump(all_results, f, indent=2)

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"Results saved to: {output_dir}")
        print("\nSummary by timeframe:")
        for tf, results in all_results.items():
            print(f"\n{tf}:")
            print(f"  Features:  {results['n_features']}")
            print(f"  Samples:   {results['n_samples_analyzed']}")
            print(f"  Top 3:     {', '.join(results['top_10_features'][:3])}")
        print()
    else:
        logger.error("No timeframes analyzed successfully")


if __name__ == "__main__":
    main()
