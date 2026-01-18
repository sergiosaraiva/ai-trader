#!/usr/bin/env python3
"""Test model performance at different time resolutions.

This script:
1. Creates 15m and 30m data from 5-minute source
2. Tests model accuracy at each resolution
3. Compares performance across timeframes
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from src.trading.filters import RegimeFilter


def load_5min_data() -> pd.DataFrame:
    """Load 5-minute EURUSD data."""
    data_path = project_root / "data/forex/EURUSD_20200101_20251231_5min_combined.csv"
    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]

    # Handle different possible column names for timestamp
    time_col = None
    for col in ['timestamp', 'time', 'date', 'datetime']:
        if col in df.columns:
            time_col = col
            break

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)

    return df.sort_index()


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLCV data to target timeframe."""
    resampled = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum' if 'volume' in df.columns else 'first',
    }).dropna()
    return resampled


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical features."""
    from src.features.technical.calculator import TechnicalIndicatorCalculator
    calc = TechnicalIndicatorCalculator(model_type="short_term")
    return calc.calculate(df)


def test_timeframe(df: pd.DataFrame, timeframe_name: str, threshold: float = 0.0003):
    """Test model on a specific timeframe."""
    print(f"\n{'=' * 60}")
    print(f"TIMEFRAME: {timeframe_name}")
    print(f"{'=' * 60}")

    print(f"Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Calculate features
    df_features = calculate_features(df)

    # Create labels with appropriate threshold
    returns = df_features["close"].pct_change(1).shift(-1)
    labels = pd.Series(index=df_features.index, dtype=float)
    labels[returns > threshold] = 1
    labels[returns < -threshold] = 0

    feature_cols = [c for c in df_features.columns if c not in ['open', 'high', 'low', 'close', 'volume']]

    X = df_features[feature_cols].values
    y = labels.values
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]
    df_valid = df_features[valid_mask].copy()

    print(f"Valid samples: {len(X)}")
    print(f"Label distribution: UP {y.mean():.1%}, DOWN {(1-y).mean():.1%}")

    # Train/test split (60/40)
    n_train = int(len(X) * 0.6)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    df_test = df_valid.iloc[n_train:].copy()

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Get predictions
    probs = model.predict_proba(X_test_scaled)
    preds = model.predict(X_test_scaled)
    confidences = np.max(probs, axis=1)

    # Overall accuracy
    overall_acc = (preds == y_test).mean()
    print(f"\nOverall Accuracy: {overall_acc:.2%}")

    # Accuracy by confidence
    print("\nAccuracy by Confidence Level:")
    print("-" * 50)

    results = []
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = confidences >= thresh
        if mask.sum() >= 20:
            acc = (preds[mask] == y_test[mask]).mean()
            count = mask.sum()
            results.append({
                'threshold': thresh,
                'accuracy': acc,
                'samples': count,
            })
            print(f"Conf >= {thresh:.0%}: {acc:.2%} ({count} samples)")

    # Best accuracy
    if results:
        best = max(results, key=lambda x: x['accuracy'])
        return {
            'timeframe': timeframe_name,
            'total_samples': len(X_test),
            'overall_accuracy': overall_acc,
            'best_confidence_threshold': best['threshold'],
            'best_accuracy': best['accuracy'],
            'best_samples': best['samples'],
        }
    return None


def main():
    print("=" * 70)
    print("MULTI-RESOLUTION MODEL TESTING")
    print("=" * 70)

    # Load 5-minute data
    print("\nLoading 5-minute data...")
    df_5m = load_5min_data()
    print(f"Loaded {len(df_5m)} 5-minute bars")

    # Create different timeframes
    timeframes = {
        '5min': (df_5m, 0.0001),  # 1 pip threshold
        '15min': (resample_ohlcv(df_5m, '15min'), 0.0002),  # 2 pip
        '30min': (resample_ohlcv(df_5m, '30min'), 0.0003),  # 3 pip
        '1H': (resample_ohlcv(df_5m, '1h'), 0.0005),  # 5 pip
        '4H': (resample_ohlcv(df_5m, '4h'), 0.001),  # 10 pip
    }

    results = []
    for name, (df, threshold) in timeframes.items():
        try:
            result = test_timeframe(df, name, threshold)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error testing {name}: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: ACCURACY BY TIME RESOLUTION")
    print("=" * 70)
    print(f"{'Timeframe':<12} {'Samples':>10} {'Overall':>12} {'Best Conf':>12} {'Best Acc':>12}")
    print("-" * 60)

    for r in results:
        print(f"{r['timeframe']:<12} {r['total_samples']:>10} {r['overall_accuracy']:>11.2%} "
              f">={r['best_confidence_threshold']:.0%}       {r['best_accuracy']:>11.2%}")

    # Find best resolution
    if results:
        best_overall = max(results, key=lambda x: x['overall_accuracy'])
        best_filtered = max(results, key=lambda x: x['best_accuracy'])

        print("\n" + "=" * 70)
        print("CONCLUSIONS")
        print("=" * 70)
        print(f"Best overall accuracy:  {best_overall['timeframe']} ({best_overall['overall_accuracy']:.2%})")
        print(f"Best filtered accuracy: {best_filtered['timeframe']} ({best_filtered['best_accuracy']:.2%} at conf >= {best_filtered['best_confidence_threshold']:.0%})")


if __name__ == "__main__":
    main()
