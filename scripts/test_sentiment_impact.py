#!/usr/bin/env python3
"""
Quick test script to compare MTF ensemble with and without sentiment features.

This uses XGBoost models which train much faster than PyTorch.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from src.features.technical.calculator import TechnicalIndicatorCalculator
from src.features.sentiment.sentiment_loader import SentimentLoader
from src.features.sentiment.sentiment_features import SentimentFeatureCalculator


def load_data_with_sentiment(
    data_path: str,
    include_sentiment: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """Load price data and optionally add sentiment features."""

    # Load price data
    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]

    time_col = next((c for c in ['timestamp', 'time', 'date', 'datetime'] if c in df.columns), None)
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    df = df.sort_index()

    # Resample to 1H
    df_1h = df.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Calculate technical indicators
    calc = TechnicalIndicatorCalculator(model_type='short_term')
    df_features = calc.calculate(df_1h)

    n_sentiment_features = 0

    if include_sentiment:
        # Load sentiment data
        try:
            sentiment_path = PROJECT_ROOT / 'data' / 'sentiment' / 'sentiment_scores_20200101_20251231_daily.csv'
            loader = SentimentLoader(sentiment_path)
            sentiment_df = loader.load()

            if sentiment_df is not None and len(sentiment_df) > 0:
                # Align sentiment to price data using the loader's method
                df_with_sent = loader.align_to_price_data(
                    df_features,
                    pair='EURUSD',
                    include_country_sentiments=True,
                    include_epu=False,
                )

                # Get raw sentiment column for derived features
                sent_col = 'sentiment_raw'
                if sent_col not in df_with_sent.columns:
                    # Find the sentiment column
                    sent_cols = [c for c in df_with_sent.columns if 'sentiment' in c.lower()]
                    if sent_cols:
                        sent_col = sent_cols[0]

                if sent_col in df_with_sent.columns:
                    # Calculate derived sentiment features
                    sent_calc = SentimentFeatureCalculator()
                    df_with_sent = sent_calc.calculate_all(
                        df_with_sent,
                        sentiment_col=sent_col,
                        prefix='sentiment'
                    )

                # Get sentiment feature columns
                sent_feature_cols = [c for c in df_with_sent.columns if 'sentiment' in c.lower() or 'sent_' in c.lower()]
                n_sentiment_features = len(sent_feature_cols)

                df_features = df_with_sent
                print(f"  Added {n_sentiment_features} sentiment features")
            else:
                print("  Warning: No sentiment data available")
        except Exception as e:
            print(f"  Warning: Could not load sentiment data: {e}")
            import traceback
            traceback.print_exc()

    # Handle NaN
    df_features = df_features.ffill().bfill().dropna()

    return df_features, n_sentiment_features


def create_labels(df: pd.DataFrame, tp_pips: float = 25.0, sl_pips: float = 15.0, max_bars: int = 12) -> pd.Series:
    """Create triple barrier labels."""
    pip_value = 0.0001
    labels = []

    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)

    for i in range(n - max_bars):
        entry = closes[i]
        tp_long = entry + tp_pips * pip_value
        sl_long = entry - sl_pips * pip_value

        label = 0  # Default: short/down
        for j in range(i + 1, min(i + max_bars + 1, n)):
            if highs[j] >= tp_long:
                label = 1  # Long won
                break
            if lows[j] <= sl_long:
                label = 0  # Long lost (short)
                break

        labels.append(label)

    # Pad the end
    labels.extend([0] * max_bars)

    return pd.Series(labels, index=df.index)


def train_and_evaluate(
    df: pd.DataFrame,
    include_sentiment: bool,
    n_sentiment_features: int,
) -> Dict:
    """Train XGBoost model and evaluate."""
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score

    # Create labels
    labels = create_labels(df)

    # Get feature columns (exclude OHLCV)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if not include_sentiment:
        feature_cols = [c for c in feature_cols if 'sentiment' not in c.lower()]

    X = df[feature_cols].values
    y = labels.values

    # Time-based split (60/20/20)
    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    # Get confidence scores for test set
    probs = model.predict_proba(X_test)
    confidences = np.max(probs, axis=1)

    # High confidence accuracy
    high_conf_mask = confidences >= 0.60
    high_conf_acc = accuracy_score(
        y_test[high_conf_mask],
        model.predict(X_test[high_conf_mask])
    ) if high_conf_mask.sum() > 0 else 0

    return {
        'n_features': len(feature_cols),
        'n_sentiment_features': n_sentiment_features if include_sentiment else 0,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'test_accuracy': test_acc,
        'high_conf_accuracy': high_conf_acc,
        'high_conf_samples': int(high_conf_mask.sum()),
    }


def run_single_trial(data_path: Path, seed: int = 42) -> Tuple[Dict, Dict]:
    """Run a single trial with given seed."""
    np.random.seed(seed)

    # Test WITHOUT sentiment
    df_no_sent, _ = load_data_with_sentiment(str(data_path), include_sentiment=False)
    results_baseline = train_and_evaluate(df_no_sent, include_sentiment=False, n_sentiment_features=0)

    # Test WITH sentiment
    np.random.seed(seed)  # Reset for fair comparison
    df_with_sent, n_sent = load_data_with_sentiment(str(data_path), include_sentiment=True)
    results_sentiment = train_and_evaluate(df_with_sent, include_sentiment=True, n_sentiment_features=n_sent)

    return results_baseline, results_sentiment


def main():
    print("\n" + "=" * 70)
    print("SENTIMENT IMPACT TEST (XGBoost) - MULTI-TRIAL")
    print("=" * 70)

    data_path = PROJECT_ROOT / 'data' / 'forex' / 'EURUSD_20200101_20251231_5min_combined.csv'

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return

    n_trials = 5
    baseline_results = []
    sentiment_results = []

    print(f"\nRunning {n_trials} trials...")

    for trial in range(n_trials):
        seed = 42 + trial * 100
        print(f"\n--- Trial {trial + 1}/{n_trials} (seed={seed}) ---")

        results_baseline, results_sentiment = run_single_trial(data_path, seed)

        baseline_results.append(results_baseline)
        sentiment_results.append(results_sentiment)

        print(f"  Baseline Test Acc: {results_baseline['test_accuracy']:.4f}")
        print(f"  Sentiment Test Acc: {results_sentiment['test_accuracy']:.4f}")

    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)

    metrics = ['test_accuracy', 'val_accuracy', 'high_conf_accuracy']

    for metric in metrics:
        base_vals = [r[metric] for r in baseline_results]
        sent_vals = [r[metric] for r in sentiment_results]

        base_mean = np.mean(base_vals)
        base_std = np.std(base_vals)
        sent_mean = np.mean(sent_vals)
        sent_std = np.std(sent_vals)

        diff = (sent_mean - base_mean) * 100

        print(f"\n{metric}:")
        print(f"  Baseline:  {base_mean*100:.2f}% (+/- {base_std*100:.2f}%)")
        print(f"  Sentiment: {sent_mean*100:.2f}% (+/- {sent_std*100:.2f}%)")
        print(f"  Diff:      {diff:+.2f}%")

        # Count wins
        wins = sum(1 for b, s in zip(base_vals, sent_vals) if s > b)
        print(f"  Sentiment wins: {wins}/{n_trials}")

    # Use last trial for detailed comparison display
    results_baseline = baseline_results[-1]
    results_sentiment = sentiment_results[-1]

    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Baseline':>12} {'Sentiment':>12} {'Diff':>10}")
    print("-" * 60)

    metrics = [
        ('Features', 'n_features', ''),
        ('Sentiment Features', 'n_sentiment_features', ''),
        ('Train Accuracy', 'train_accuracy', '%'),
        ('Val Accuracy', 'val_accuracy', '%'),
        ('Test Accuracy', 'test_accuracy', '%'),
        ('High-Conf Accuracy', 'high_conf_accuracy', '%'),
    ]

    for name, key, fmt in metrics:
        base_val = results_baseline[key]
        sent_val = results_sentiment[key]

        if fmt == '%':
            diff = (sent_val - base_val) * 100
            print(f"{name:<25} {base_val*100:>11.2f}% {sent_val*100:>11.2f}% {diff:>+9.2f}%")
        else:
            diff = sent_val - base_val
            print(f"{name:<25} {base_val:>12} {sent_val:>12} {diff:>+10}")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    test_diff = (results_sentiment['test_accuracy'] - results_baseline['test_accuracy']) * 100
    val_diff = (results_sentiment['val_accuracy'] - results_baseline['val_accuracy']) * 100
    high_conf_diff = (results_sentiment['high_conf_accuracy'] - results_baseline['high_conf_accuracy']) * 100

    if test_diff > 1.0 and val_diff > 0.5:
        print("\n[OK] KEEP SENTIMENT: Test accuracy improved by {:.2f}%".format(test_diff))
    elif test_diff > 0 and high_conf_diff > 0:
        print("\n[?] NEUTRAL: Small improvement, more testing recommended")
        print(f"    Test: {test_diff:+.2f}%, High-Conf: {high_conf_diff:+.2f}%")
    elif test_diff < -1.0:
        print("\n[!!] ROLLBACK RECOMMENDED: Test accuracy decreased by {:.2f}%".format(abs(test_diff)))
    else:
        print("\n[?] INCONCLUSIVE: No significant difference")
        print(f"    Test: {test_diff:+.2f}%, Val: {val_diff:+.2f}%")

    print("=" * 70)

    return results_baseline, results_sentiment


if __name__ == '__main__':
    main()
