#!/usr/bin/env python3
"""Full trading test across multiple periods and timeframes.

This script tests the trading system across different market periods
to validate the regime filtering approach.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from typing import List
from datetime import datetime

from src.trading.filters import RegimeFilter


@dataclass
class Trade:
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    size: float
    pnl_pct: float
    regime: str
    confidence: float

    @property
    def is_win(self) -> bool:
        return self.pnl_pct > 0


def load_data(timeframe: str) -> pd.DataFrame:
    data_path = project_root / f"data/forex/derived_proper/{timeframe}/EURUSD_{timeframe}.parquet"
    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    from src.features.technical.calculator import TechnicalIndicatorCalculator
    calc = TechnicalIndicatorCalculator(model_type="short_term")
    return calc.calculate(df)


def run_test_period(
    df: pd.DataFrame,
    model,
    scaler: StandardScaler,
    feature_cols: List[str],
    regime_filter: RegimeFilter,
    use_regime_filter: bool,
    min_confidence: float = 0.52,
    position_size: float = 0.02,
    stop_loss: float = 0.01,
    take_profit: float = 0.02,
) -> List[Trade]:
    """Run backtest on a data period."""
    trades = []
    lookback = 50
    position = None
    entry_price = entry_time = entry_confidence = entry_regime = None
    pos_size = 0

    for i in range(lookback, len(df) - 1):
        current_time = df.index[i]
        current_price = df['close'].iloc[i]

        # Get features
        X = df[feature_cols].iloc[i:i+1].values
        if np.isnan(X).any():
            continue
        X_scaled = scaler.transform(X)

        # Prediction
        prob = model.predict_proba(X_scaled)[0]
        pred_class = model.predict(X_scaled)[0]
        confidence = max(prob)

        # Regime analysis
        market_data = df.iloc[i-lookback:i+1].copy()
        regime_analysis = regime_filter.analyze(market_data)
        regime = regime_analysis.regime.value

        # Should trade?
        should_trade = confidence >= min_confidence
        if use_regime_filter and not regime_analysis.should_trade:
            should_trade = False

        # Position management
        if position is not None:
            if position == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            if pnl_pct >= take_profit or pnl_pct <= -stop_loss:
                trades.append(Trade(
                    entry_time=entry_time,
                    exit_time=current_time,
                    direction=position,
                    entry_price=entry_price,
                    exit_price=current_price,
                    size=pos_size,
                    pnl_pct=pnl_pct,
                    regime=entry_regime,
                    confidence=entry_confidence,
                ))
                position = None

        # Enter new position
        if position is None and should_trade:
            adjusted_size = position_size * regime_analysis.confidence_modifier
            position = 'long' if pred_class == 1 else 'short'
            entry_price = current_price
            entry_time = current_time
            entry_confidence = confidence
            entry_regime = regime
            pos_size = adjusted_size

    return trades


def main():
    print("=" * 70)
    print("COMPREHENSIVE TRADING TEST")
    print("=" * 70)

    results = []

    for timeframe in ["1H", "4H"]:
        print(f"\n{'=' * 70}")
        print(f"TIMEFRAME: {timeframe}")
        print("=" * 70)

        # Load data
        df = load_data(timeframe)
        print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

        # Calculate features
        df_features = calculate_features(df)

        # Create labels
        returns = df_features["close"].pct_change(1).shift(-1)
        labels = pd.Series(index=df_features.index, dtype=float)
        labels[returns > 0.0005] = 1
        labels[returns < -0.0005] = 0

        # Feature columns
        feature_cols = [c for c in df_features.columns if c not in ['open', 'high', 'low', 'close', 'volume']]

        # Prepare data
        X = df_features[feature_cols].values
        y = labels.values
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        df_valid = df_features[valid_mask].copy()

        # Use walk-forward: train on first 50%, test on remaining 50%
        n_train = int(len(X) * 0.5)

        X_train, y_train = X[:n_train], y[:n_train]
        X_test, y_test = X[n_train:], y[n_train:]
        df_test = df_valid.iloc[n_train:].copy()

        print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Normalize and train
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

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

        # Validation accuracy
        val_acc = (model.predict(scaler.transform(X_test)) == y_test).mean()
        print(f"Model accuracy: {val_acc:.2%}")

        regime_filter = RegimeFilter(timeframe=timeframe)

        # Analyze regimes in test set
        print("\nRegime Distribution in Test Set:")
        regimes = []
        for i in range(50, len(df_test)):
            market_data = df_test.iloc[i-50:i+1].copy()
            analysis = regime_filter.analyze(market_data)
            regimes.append(analysis.regime.value)

        regime_counts = pd.Series(regimes).value_counts()
        for regime, count in regime_counts.items():
            pct = count / len(regimes) * 100
            print(f"  {regime:15s}: {count:5d} ({pct:5.1f}%)")

        # Run backtests
        for use_filter in [False, True]:
            filter_name = "WITH" if use_filter else "WITHOUT"

            trades = run_test_period(
                df=df_test,
                model=model,
                scaler=scaler,
                feature_cols=feature_cols,
                regime_filter=regime_filter,
                use_regime_filter=use_filter,
                min_confidence=0.52,
                position_size=0.02,
                stop_loss=0.01,
                take_profit=0.02,
            )

            n_trades = len(trades)
            wins = sum(1 for t in trades if t.is_win)
            win_rate = wins / n_trades if n_trades > 0 else 0
            total_pnl = sum(t.pnl_pct * t.size for t in trades)

            gross_profit = sum(t.pnl_pct for t in trades if t.is_win)
            gross_loss = abs(sum(t.pnl_pct for t in trades if not t.is_win))
            pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            print(f"\n{filter_name} Regime Filter:")
            print(f"  Trades: {n_trades}, Win Rate: {win_rate:.2%}, Total PnL: {total_pnl:.4%}, PF: {pf:.2f}")

            # Breakdown by regime
            if trades:
                print("  By Regime:")
                for regime in set(t.regime for t in trades):
                    rt = [t for t in trades if t.regime == regime]
                    rw = sum(1 for t in rt if t.is_win)
                    rwr = rw / len(rt) if rt else 0
                    print(f"    {regime:15s}: {len(rt):3d} trades, {rwr:.2%} win rate")

            results.append({
                'timeframe': timeframe,
                'filter': use_filter,
                'trades': n_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'profit_factor': pf,
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'TF':<5} {'Filter':<10} {'Trades':>8} {'Win Rate':>10} {'Total PnL':>12} {'PF':>8}")
    print("-" * 60)
    for r in results:
        filter_str = "Yes" if r['filter'] else "No"
        print(f"{r['timeframe']:<5} {filter_str:<10} {r['trades']:>8} {r['win_rate']:>9.2%} {r['total_pnl']:>11.4%} {r['profit_factor']:>8.2f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Compare filtered vs unfiltered
    for tf in ["1H", "4H"]:
        no_filter = next(r for r in results if r['timeframe'] == tf and not r['filter'])
        with_filter = next(r for r in results if r['timeframe'] == tf and r['filter'])

        print(f"\n{tf}:")
        if with_filter['trades'] < no_filter['trades']:
            blocked = no_filter['trades'] - with_filter['trades']
            print(f"  Regime filter blocked {blocked} trades")

        if with_filter['win_rate'] > no_filter['win_rate']:
            diff = (with_filter['win_rate'] - no_filter['win_rate']) * 100
            print(f"  Win rate improved by {diff:.2f}pp")
        elif with_filter['win_rate'] == no_filter['win_rate']:
            print(f"  Win rate unchanged (same regime allowed)")

        if with_filter['total_pnl'] > no_filter['total_pnl']:
            print(f"  Total PnL improved from {no_filter['total_pnl']:.4%} to {with_filter['total_pnl']:.4%}")


if __name__ == "__main__":
    main()
