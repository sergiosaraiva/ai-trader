#!/usr/bin/env python3
"""Train gradient boosting model for direction prediction.

Gradient boosting often outperforms deep learning on tabular data,
especially with limited samples.
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(timeframe: str, threshold: float = 0.0005, lookback: int = 5):
    """Load data and create features with lookback."""
    from src.features.technical.calculator import TechnicalIndicatorCalculator

    # Load data
    data_path = project_root / f"data/forex/derived_proper/{timeframe}/EURUSD_{timeframe}.parquet"
    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    logger.info(f"Loaded {len(df)} rows from {data_path}")

    # Calculate features
    calc = TechnicalIndicatorCalculator(model_type="short_term")
    df_features = calc.calculate(df)

    # Create binary labels
    returns = df_features["close"].pct_change(1).shift(-1)

    # Create mask for significant moves
    up_mask = returns > threshold
    down_mask = returns < -threshold
    significant_mask = up_mask | down_mask

    # Get feature columns (exclude OHLCV)
    feature_cols = [c for c in df_features.columns if c not in ['open', 'high', 'low', 'close', 'volume']]

    # Create lagged features for additional context
    X_base = df_features[feature_cols].copy()

    # Add lagged features
    for lag in range(1, lookback + 1):
        for col in feature_cols[:10]:  # Only lag first 10 features to avoid explosion
            X_base[f"{col}_lag{lag}"] = X_base[col].shift(lag)

    # Add returns at different horizons
    for h in [1, 2, 3, 5]:
        X_base[f"return_{h}"] = df_features["close"].pct_change(h)

    # Add volatility features
    X_base["volatility_5"] = df_features["close"].pct_change().rolling(5).std()
    X_base["volatility_10"] = df_features["close"].pct_change().rolling(10).std()

    # Add trend features
    X_base["trend_5"] = (df_features["close"] - df_features["close"].rolling(5).mean()) / df_features["close"].rolling(5).std()
    X_base["trend_10"] = (df_features["close"] - df_features["close"].rolling(10).mean()) / df_features["close"].rolling(10).std()

    # Filter to significant moves and drop NaNs
    X_filtered = X_base[significant_mask].copy()
    y_filtered = up_mask[significant_mask].astype(int)

    # Drop rows with NaN
    valid_mask = ~X_filtered.isnull().any(axis=1) & ~y_filtered.isnull()
    X_filtered = X_filtered[valid_mask]
    y_filtered = y_filtered[valid_mask]

    logger.info(f"After filtering: {len(X_filtered)} samples with {len(X_filtered.columns)} features")
    logger.info(f"Label distribution: UP {y_filtered.mean():.1%}, DOWN {(1-y_filtered).mean():.1%}")

    return X_filtered, y_filtered


def train_and_evaluate(X, y, model_type="gradient_boosting"):
    """Train model with time-series cross-validation."""
    # Chronological split
    n_train = int(len(X) * 0.7)
    n_val = int(len(X) * 0.15)

    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]
    X_val = X.iloc[n_train:n_train+n_val]
    y_val = y.iloc[n_train:n_train+n_val]
    X_test = X.iloc[n_train+n_val:]
    y_test = y.iloc[n_train+n_val:]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Train UP ratio: {y_train.mean():.2%}")
    logger.info(f"Test UP ratio: {y_test.mean():.2%}")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    if model_type == "gradient_boosting":
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=20,
            min_samples_leaf=10,
            subsample=0.8,
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=1
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

    logger.info(f"Training {model_type} model...")
    model.fit(X_train_scaled, y_train)

    # Evaluate
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    logger.info(f"\nTrain Accuracy: {train_acc:.2%}")
    logger.info(f"Val Accuracy:   {val_acc:.2%}")
    logger.info(f"Test Accuracy:  {test_acc:.2%}")

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nTop 15 features:")
        for _, row in importance.head(15).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Detailed test results
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"Accuracy: {test_acc:.2%}")
    print(f"\nPrediction distribution: UP {test_pred.mean():.1%}, DOWN {1-test_pred.mean():.1%}")
    print(f"Actual distribution:     UP {y_test.mean():.1%}, DOWN {1-y_test.mean():.1%}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=['DOWN', 'UP']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_pred))
    print("=" * 60)

    return model, scaler, test_acc


def main():
    parser = argparse.ArgumentParser(description="Train gradient boosting classifier")
    parser.add_argument("--timeframe", type=str, default="1H", help="Timeframe")
    parser.add_argument("--threshold", type=float, default=0.0005, help="Threshold for labeling")
    parser.add_argument("--lookback", type=int, default=5, help="Lookback for lagged features")
    parser.add_argument("--model", type=str, default="gradient_boosting", choices=["gradient_boosting", "random_forest"])
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("GRADIENT BOOSTING DIRECTION CLASSIFIER")
    print("=" * 60)
    print(f"Timeframe: {args.timeframe}")
    print(f"Threshold: {args.threshold:.4%}")
    print(f"Model: {args.model}")
    print("=" * 60)

    # Load data
    X, y = load_and_prepare_data(args.timeframe, args.threshold, args.lookback)

    # Train and evaluate
    model, scaler, test_acc = train_and_evaluate(X, y, args.model)

    # Save model
    import pickle
    output_dir = project_root / "models" / "gradient_boosting"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / f"gb_{args.timeframe}.pkl", 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'feature_names': list(X.columns),
            'threshold': args.threshold,
            'test_accuracy': test_acc,
        }, f)

    logger.info(f"Model saved to {output_dir / f'gb_{args.timeframe}.pkl'}")


if __name__ == "__main__":
    main()
