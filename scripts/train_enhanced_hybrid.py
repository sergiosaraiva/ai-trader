#!/usr/bin/env python3
"""Train enhanced hybrid model with technical indicators in sequence model.

This version feeds technical indicators (not just OHLCV) to the sequence model,
allowing it to learn temporal patterns from RSI, MACD, moving averages, etc.
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
    ImprovedTimeframeModel,
)
from src.models.multi_timeframe.enhanced_sequence_model import (
    EnhancedSequenceConfig,
    EnhancedSequenceDataset,
    EnhancedSequenceTrainer,
    EnhancedSequencePredictor,
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


def prepare_features(df_5min: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample and calculate all features."""
    from src.features.technical.calculator import TechnicalIndicatorCalculator
    from src.models.multi_timeframe import EnhancedFeatureEngine

    # Resample
    df_tf = df_5min.resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum" if "volume" in df_5min.columns else "first",
    }).dropna()

    logger.info(f"Resampled to {timeframe}: {len(df_tf)} bars")

    # Calculate technical indicators
    calc = TechnicalIndicatorCalculator(model_type="short_term")
    df_features = calc.calculate(df_tf)

    # Prepare higher TF data for cross-TF features
    higher_tf_data = {}
    for htf in ["4H", "D"]:
        df_htf = df_5min.resample(htf).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in df_5min.columns else "first",
        }).dropna()
        higher_tf_data[htf] = calc.calculate(df_htf)

    # Add enhanced features
    feature_engine = EnhancedFeatureEngine(base_timeframe=timeframe)
    df_features = feature_engine.add_all_features(df_features, higher_tf_data)

    return df_features


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced Hybrid Model")
    parser.add_argument("--data", type=str, default="data/forex/EURUSD_20200101_20251231_5min_combined.csv")
    parser.add_argument("--output", type=str, default="models/enhanced_hybrid")
    parser.add_argument("--timeframe", type=str, default="1H")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--xgboost-weight", type=float, default=0.5)
    parser.add_argument("--sequence-weight", type=float, default=0.5)
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("ENHANCED HYBRID MODEL TRAINING")
    print("(Sequence model with technical indicators)")
    print("=" * 70)
    print(f"Data:            {args.data}")
    print(f"Output:          {args.output}")
    print(f"Timeframe:       {args.timeframe}")
    print(f"XGBoost weight:  {args.xgboost_weight}")
    print(f"Sequence weight: {args.sequence_weight}")
    print("=" * 70)

    # Load and prepare data
    data_path = project_root / args.data
    df_5min = load_data(data_path)

    # Prepare features
    logger.info("Calculating technical indicators and features...")
    df_features = prepare_features(df_5min, args.timeframe)
    df_features = df_features.dropna()

    logger.info(f"Feature matrix: {len(df_features)} rows, {len(df_features.columns)} columns")

    # Split data
    n_total = len(df_features)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)

    # ==================== Train XGBoost ====================
    print("\n" + "=" * 50)
    print("TRAINING XGBOOST MODEL")
    print("=" * 50)

    if args.timeframe == "1H":
        xgb_config = ImprovedModelConfig.hourly_model()
    elif args.timeframe == "4H":
        xgb_config = ImprovedModelConfig.four_hour_model()
    else:
        raise ValueError(f"Unknown timeframe: {args.timeframe}")

    xgb_model = ImprovedTimeframeModel(xgb_config)

    # Prepare XGBoost data (uses same features but as flat vectors)
    from src.features.technical.calculator import TechnicalIndicatorCalculator
    calc = TechnicalIndicatorCalculator(model_type="short_term")

    # Get higher TF data
    higher_tf_data = {}
    for htf in ["4H", "D"]:
        df_htf = df_5min.resample(htf).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in df_5min.columns else "first",
        }).dropna()
        higher_tf_data[htf] = calc.calculate(df_htf)

    # Resample to timeframe
    df_tf = df_5min.resample(args.timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum" if "volume" in df_5min.columns else "first",
    }).dropna()

    X_xgb, y_xgb, xgb_feature_cols = xgb_model.prepare_data(df_tf, higher_tf_data)

    X_train_xgb = X_xgb[:n_train]
    y_train_xgb = y_xgb[:n_train]
    X_val_xgb = X_xgb[n_train:n_train + n_val]
    y_val_xgb = y_xgb[n_train:n_train + n_val]

    xgb_metrics = xgb_model.train(
        X_train_xgb, y_train_xgb,
        X_val_xgb, y_val_xgb,
        xgb_feature_cols,
    )

    # ==================== Train Enhanced Sequence Model ====================
    print("\n" + "=" * 50)
    print("TRAINING ENHANCED SEQUENCE MODEL")
    print("(With technical indicators)")
    print("=" * 50)

    if args.timeframe == "1H":
        seq_config = EnhancedSequenceConfig.hourly_model()
    else:
        seq_config = EnhancedSequenceConfig.four_hour_model()

    seq_dataset = EnhancedSequenceDataset(seq_config)
    X_seq, y_seq, seq_feature_names = seq_dataset.prepare(df_features, fit_scaler=True)

    # Split sequence data (accounting for sequence length)
    seq_len = seq_config.sequence_length
    seq_train_end = n_train - seq_len
    seq_val_end = n_train + n_val - seq_len

    X_train_seq = X_seq[:seq_train_end]
    y_train_seq = y_seq[:seq_train_end]
    X_val_seq = X_seq[seq_train_end:seq_val_end]
    y_val_seq = y_seq[seq_train_end:seq_val_end]

    logger.info(f"Sequence train: {len(X_train_seq)}, val: {len(X_val_seq)}")
    logger.info(f"Sequence shape: {X_train_seq.shape}")

    trainer = EnhancedSequenceTrainer(seq_config)
    seq_model, seq_metrics = trainer.train(
        X_train_seq, y_train_seq,
        X_val_seq, y_val_seq,
    )

    seq_predictor = EnhancedSequencePredictor(
        seq_model,
        seq_config,
        seq_dataset.scaler,
        seq_feature_names,
    )

    # ==================== Evaluate Ensemble ====================
    print("\n" + "=" * 50)
    print("EVALUATING ENSEMBLE")
    print("=" * 50)

    # Get predictions
    xgb_preds, xgb_confs = xgb_model.predict_batch(X_val_xgb)
    seq_preds, seq_confs = seq_predictor.predict_batch(X_val_seq)

    # Align
    n_common = min(len(xgb_preds), len(seq_preds))
    xgb_preds = xgb_preds[:n_common]
    xgb_confs = xgb_confs[:n_common]
    seq_preds = seq_preds[:n_common]
    seq_confs = seq_confs[:n_common]
    y_val = y_val_xgb[:n_common]

    # Confidence-weighted ensemble
    total_conf = xgb_confs + seq_confs
    xgb_w = xgb_confs / total_conf
    seq_w = seq_confs / total_conf

    xgb_prob = xgb_preds * xgb_confs + (1 - xgb_preds) * (1 - xgb_confs)
    seq_prob = seq_preds * seq_confs + (1 - seq_preds) * (1 - seq_confs)

    ensemble_prob = xgb_w * xgb_prob + seq_w * seq_prob
    ensemble_preds = (ensemble_prob > 0.5).astype(int)
    ensemble_confs = np.abs(ensemble_prob - 0.5) * 2 + 0.5

    # Boost confidence on agreement
    agreement = (xgb_preds == seq_preds).astype(float)
    ensemble_confs = np.minimum(ensemble_confs + agreement * 0.05, 1.0)

    ensemble_acc = (ensemble_preds == y_val).mean()
    agreement_rate = agreement.mean()

    logger.info(f"Model agreement rate: {agreement_rate:.2%}")
    logger.info(f"XGBoost accuracy: {xgb_metrics['val_accuracy']:.2%}")
    logger.info(f"Sequence accuracy: {seq_metrics['val_accuracy']:.2%}")
    logger.info(f"Ensemble accuracy: {ensemble_acc:.2%}")

    # Accuracy at confidence levels
    ensemble_metrics = {'val_accuracy': ensemble_acc, 'agreement_rate': agreement_rate}
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        mask = ensemble_confs >= thresh
        if mask.sum() > 0:
            acc = (ensemble_preds[mask] == y_val[mask]).mean()
            ensemble_metrics[f'val_acc_conf_{int(thresh*100)}'] = acc
            ensemble_metrics[f'val_samples_conf_{int(thresh*100)}'] = int(mask.sum())

    # ==================== Save Models ====================
    print("\n" + "=" * 50)
    print("SAVING MODELS")
    print("=" * 50)

    model_dir = project_root / args.output
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save XGBoost
    xgb_path = model_dir / f"{args.timeframe}_xgboost.pkl"
    xgb_model.save(xgb_path)

    # Save sequence model
    seq_path = model_dir / f"{args.timeframe}_enhanced_sequence.pt"
    seq_predictor.save(seq_path)

    # Save config
    config_data = {
        'timeframe': args.timeframe,
        'xgboost_weight': args.xgboost_weight,
        'sequence_weight': args.sequence_weight,
        'xgboost_val_acc': float(xgb_metrics['val_accuracy']),
        'sequence_val_acc': float(seq_metrics['val_accuracy']),
        'ensemble_val_acc': float(ensemble_acc),
        'agreement_rate': float(agreement_rate),
    }

    with open(model_dir / f"{args.timeframe}_config.json", 'w') as f:
        json.dump(config_data, f, indent=2)

    # Save training metadata
    metadata = {
        "training_date": datetime.now().isoformat(),
        "data_file": str(args.data),
        "timeframe": args.timeframe,
        "xgboost_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v
                           for k, v in xgb_metrics.items()},
        "sequence_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v
                            for k, v in seq_metrics.items()},
        "ensemble_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v
                            for k, v in ensemble_metrics.items()},
    }

    with open(model_dir / "training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # ==================== Results ====================
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Val Accuracy':>15}")
    print("-" * 45)
    print(f"{'XGBoost':<25} {xgb_metrics['val_accuracy']:>14.2%}")
    print(f"{'Enhanced Sequence':<25} {seq_metrics['val_accuracy']:>14.2%}")
    print(f"{'Ensemble':<25} {ensemble_acc:>14.2%}")

    improvement = ensemble_acc - xgb_metrics['val_accuracy']
    print("\n" + "-" * 45)
    if improvement > 0:
        print(f"Ensemble improvement: +{improvement:.2%}")
    else:
        print(f"Ensemble vs XGBoost: {improvement:+.2%}")

    print(f"\nModel agreement rate: {agreement_rate:.2%}")

    print("\n" + "=" * 70)
    print("ENSEMBLE ACCURACY BY CONFIDENCE")
    print("=" * 70)
    for thresh in [55, 60, 65, 70]:
        key = f"val_acc_conf_{thresh}"
        samples_key = f"val_samples_conf_{thresh}"
        if key in ensemble_metrics:
            print(f"Conf >= {thresh}%: {ensemble_metrics[key]:.2%} ({ensemble_metrics.get(samples_key, 0)} samples)")

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Models saved to: {model_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
