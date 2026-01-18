#!/usr/bin/env python3
"""Analyze trading performance by confidence level.

This script examines if higher confidence predictions have better win rates.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import List

from src.trading.filters import RegimeFilter


def load_data(timeframe: str) -> pd.DataFrame:
    data_path = project_root / f"data/forex/derived_proper/{timeframe}/EURUSD_{timeframe}.parquet"
    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df.sort_index()


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    from src.features.technical.calculator import TechnicalIndicatorCalculator
    calc = TechnicalIndicatorCalculator(model_type="short_term")
    return calc.calculate(df)


def main():
    print("=" * 70)
    print("CONFIDENCE-BASED PERFORMANCE ANALYSIS")
    print("=" * 70)

    for timeframe in ["1H", "4H"]:
        print(f"\n{'=' * 70}")
        print(f"TIMEFRAME: {timeframe}")
        print("=" * 70)

        # Load and prepare data
        df = load_data(timeframe)
        df_features = calculate_features(df)

        # Create labels
        returns = df_features["close"].pct_change(1).shift(-1)
        labels = pd.Series(index=df_features.index, dtype=float)
        labels[returns > 0.0005] = 1
        labels[returns < -0.0005] = 0

        feature_cols = [c for c in df_features.columns if c not in ['open', 'high', 'low', 'close', 'volume']]

        X = df_features[feature_cols].values
        y = labels.values
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        df_valid = df_features[valid_mask].copy()

        # Train/test split
        n_train = int(len(X) * 0.6)
        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]
        df_test = df_valid.iloc[n_train:].copy()

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

        # Get predictions with probabilities
        probs = model.predict_proba(X_test_scaled)
        preds = model.predict(X_test_scaled)
        confidences = np.max(probs, axis=1)

        # Regime analysis
        regime_filter = RegimeFilter(timeframe=timeframe)
        regimes = []
        lookback = 50
        for i in range(len(df_test)):
            if i < lookback:
                regimes.append("unknown")
            else:
                start_idx = df_valid.index.get_loc(df_test.index[i]) - lookback
                end_idx = df_valid.index.get_loc(df_test.index[i]) + 1
                market_data = df_valid.iloc[start_idx:end_idx]
                analysis = regime_filter.analyze(market_data)
                regimes.append(analysis.regime.value)

        regimes = np.array(regimes)

        # Create results dataframe
        results = pd.DataFrame({
            'prediction': preds,
            'actual': y_test,
            'confidence': confidences,
            'regime': regimes,
            'correct': preds == y_test,
        })

        # Overall accuracy
        print(f"\nOverall Test Accuracy: {results['correct'].mean():.2%}")
        print(f"Total samples: {len(results)}")

        # Analyze by confidence threshold
        print("\n" + "-" * 70)
        print("PERFORMANCE BY CONFIDENCE THRESHOLD")
        print("-" * 70)
        print(f"{'Confidence':<15} {'Samples':>10} {'Accuracy':>12} {'Win Rate':>12}")
        print("-" * 70)

        thresholds = [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75, 0.80]
        for thresh in thresholds:
            mask = results['confidence'] >= thresh
            subset = results[mask]
            if len(subset) > 0:
                acc = subset['correct'].mean()
                # Win rate: when we predict UP, how often is actual UP?
                up_preds = subset[subset['prediction'] == 1]
                win_rate = up_preds['actual'].mean() if len(up_preds) > 0 else 0
                print(f">= {thresh:.0%}          {len(subset):>10} {acc:>11.2%} {win_rate:>11.2%}")

        # Analyze by confidence AND regime (for optimal regime)
        print("\n" + "-" * 70)
        print("PERFORMANCE BY CONFIDENCE + OPTIMAL REGIME")
        print("-" * 70)

        optimal_regimes = regime_filter.optimal_regimes
        optimal_regime_names = [r.value for r in optimal_regimes]

        print(f"Optimal regimes for {timeframe}: {optimal_regime_names}")
        print()
        print(f"{'Confidence':<15} {'Samples':>10} {'Accuracy':>12} {'Win Rate':>12}")
        print("-" * 70)

        for thresh in thresholds:
            mask = (results['confidence'] >= thresh) & (results['regime'].isin(optimal_regime_names))
            subset = results[mask]
            if len(subset) > 0:
                acc = subset['correct'].mean()
                up_preds = subset[subset['prediction'] == 1]
                win_rate = up_preds['actual'].mean() if len(up_preds) > 0 else 0
                print(f">= {thresh:.0%}          {len(subset):>10} {acc:>11.2%} {win_rate:>11.2%}")

        # Best combination analysis
        print("\n" + "-" * 70)
        print("BEST PERFORMING COMBINATIONS")
        print("-" * 70)

        best_combos = []
        for thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
            for regime in ['trending_down', 'trending_up', 'ranging']:
                mask = (results['confidence'] >= thresh) & (results['regime'] == regime)
                subset = results[mask]
                if len(subset) >= 20:  # Minimum sample size
                    acc = subset['correct'].mean()
                    best_combos.append({
                        'threshold': thresh,
                        'regime': regime,
                        'samples': len(subset),
                        'accuracy': acc,
                    })

        best_combos = sorted(best_combos, key=lambda x: x['accuracy'], reverse=True)

        print(f"{'Confidence':<12} {'Regime':<15} {'Samples':>10} {'Accuracy':>12}")
        print("-" * 50)
        for combo in best_combos[:10]:
            print(f">= {combo['threshold']:.0%}        {combo['regime']:<15} {combo['samples']:>10} {combo['accuracy']:>11.2%}")

        # Simulated trading with confidence filter
        print("\n" + "-" * 70)
        print("SIMULATED TRADING RESULTS (2:1 R:R)")
        print("-" * 70)

        for thresh in [0.55, 0.60, 0.65, 0.70]:
            for use_regime in [False, True]:
                if use_regime:
                    mask = (results['confidence'] >= thresh) & (results['regime'].isin(optimal_regime_names))
                    label = f"Conf >= {thresh:.0%} + Regime"
                else:
                    mask = results['confidence'] >= thresh
                    label = f"Conf >= {thresh:.0%}"

                subset = results[mask]
                if len(subset) < 10:
                    continue

                # Simulate trades with 2:1 R:R
                # Win = +2%, Loss = -1%
                wins = subset['correct'].sum()
                losses = len(subset) - wins
                pnl = wins * 0.02 - losses * 0.01
                win_rate = wins / len(subset)

                print(f"{label:<25} | Trades: {len(subset):4d} | Win: {win_rate:5.1%} | PnL: {pnl:+6.2%}")


if __name__ == "__main__":
    main()
