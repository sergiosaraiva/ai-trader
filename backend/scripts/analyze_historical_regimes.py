#!/usr/bin/env python3
"""Analyze historical regime distribution across all data."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from src.trading.filters import RegimeFilter


def load_data(timeframe: str) -> pd.DataFrame:
    data_path = project_root / f"data/forex/derived_proper/{timeframe}/EURUSD_{timeframe}.parquet"
    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def main():
    print("=" * 70)
    print("HISTORICAL REGIME ANALYSIS")
    print("=" * 70)

    for timeframe in ["1H", "4H"]:
        print(f"\n{'=' * 70}")
        print(f"TIMEFRAME: {timeframe}")
        print("=" * 70)

        df = load_data(timeframe)
        regime_filter = RegimeFilter(timeframe=timeframe)

        # Analyze regimes for entire dataset
        regimes = []
        dates = []
        lookback = 50

        for i in range(lookback, len(df)):
            market_data = df.iloc[i-lookback:i+1].copy()
            analysis = regime_filter.analyze(market_data)
            regimes.append(analysis.regime.value)
            dates.append(df.index[i])

        regime_df = pd.DataFrame({'date': dates, 'regime': regimes})
        regime_df['date'] = pd.to_datetime(regime_df['date'])
        regime_df['year'] = regime_df['date'].dt.year
        regime_df['month'] = regime_df['date'].dt.month

        # Overall distribution
        print(f"\nOverall Regime Distribution ({len(regime_df)} bars):")
        print("-" * 40)
        for regime, count in regime_df['regime'].value_counts().items():
            pct = count / len(regime_df) * 100
            print(f"  {regime:15s}: {count:6d} ({pct:5.1f}%)")

        # By year
        print("\nBy Year:")
        print("-" * 60)
        yearly = regime_df.groupby(['year', 'regime']).size().unstack(fill_value=0)
        print(yearly.to_string())

        # Find periods with trending regimes
        print("\nPeriods with Trending Regimes:")
        print("-" * 40)
        regime_df['is_trending'] = regime_df['regime'].isin(['trending_up', 'trending_down'])

        # Group consecutive trending periods
        regime_df['regime_change'] = regime_df['regime'] != regime_df['regime'].shift()
        regime_df['regime_group'] = regime_df['regime_change'].cumsum()

        trending_periods = regime_df[regime_df['is_trending']].groupby('regime_group').agg({
            'date': ['min', 'max', 'count'],
            'regime': 'first'
        })

        if len(trending_periods) > 0:
            trending_periods.columns = ['start', 'end', 'bars', 'regime']
            trending_periods = trending_periods[trending_periods['bars'] >= 10]  # At least 10 bars
            trending_periods = trending_periods.sort_values('bars', ascending=False)

            for _, row in trending_periods.head(10).iterrows():
                print(f"  {row['regime']:15s}: {row['start'].strftime('%Y-%m-%d')} to {row['end'].strftime('%Y-%m-%d')} ({row['bars']:4d} bars)")
        else:
            print("  No significant trending periods found")


if __name__ == "__main__":
    main()
