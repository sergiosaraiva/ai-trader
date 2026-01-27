#!/usr/bin/env python3
"""Analyze typical market regime durations to inform lookback period."""

import pandas as pd
import numpy as np

# Load 5-min data
df = pd.read_csv('data/forex/EURUSD_20200101_20251231_5min_combined.csv', index_col=0, parse_dates=True)

# Resample to daily for regime analysis
df_daily = df.resample('1D').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

# Calculate ATR (14-day) for volatility regimes
df_daily['tr'] = np.maximum(
    df_daily['high'] - df_daily['low'],
    np.maximum(
        abs(df_daily['high'] - df_daily['close'].shift(1)),
        abs(df_daily['low'] - df_daily['close'].shift(1))
    )
)
df_daily['atr_14'] = df_daily['tr'].rolling(14).mean()

# Calculate rolling volatility (21-day)
df_daily['returns'] = df_daily['close'].pct_change()
df_daily['volatility_21'] = df_daily['returns'].rolling(21).std() * np.sqrt(252)

# Define volatility regimes (terciles)
df_daily = df_daily.dropna()
vol_low = df_daily['volatility_21'].quantile(0.33)
vol_high = df_daily['volatility_21'].quantile(0.67)

df_daily['regime'] = 'medium'
df_daily.loc[df_daily['volatility_21'] < vol_low, 'regime'] = 'low_vol'
df_daily.loc[df_daily['volatility_21'] > vol_high, 'regime'] = 'high_vol'

# Calculate regime durations
regime_changes = df_daily['regime'] != df_daily['regime'].shift(1)
df_daily['regime_id'] = regime_changes.cumsum()

regime_durations = df_daily.groupby('regime_id').agg({
    'regime': 'first',
    'close': 'count'
}).rename(columns={'close': 'duration_days'})

print("=" * 80)
print("MARKET REGIME DURATION ANALYSIS")
print("=" * 80)
print()
print(f"Data Period: {df_daily.index.min().date()} to {df_daily.index.max().date()}")
print(f"Total Days: {len(df_daily)}")
print(f"Total Regimes: {len(regime_durations)}")
print()

print("VOLATILITY REGIME STATISTICS")
print("-" * 80)
print()

for regime in ['low_vol', 'medium', 'high_vol']:
    regime_data = regime_durations[regime_durations['regime'] == regime]['duration_days']

    if len(regime_data) > 0:
        print(f"{regime.upper().replace('_', ' ')}:")
        print(f"  Count: {len(regime_data)} regimes")
        print(f"  Mean duration: {regime_data.mean():.1f} days")
        print(f"  Median duration: {regime_data.median():.1f} days")
        print(f"  Min: {regime_data.min()} days")
        print(f"  Max: {regime_data.max()} days")
        print(f"  25th percentile: {regime_data.quantile(0.25):.1f} days")
        print(f"  75th percentile: {regime_data.quantile(0.75):.1f} days")
        print()

# Overall statistics
all_durations = regime_durations['duration_days']
print("OVERALL REGIME PERSISTENCE")
print("-" * 80)
print(f"  Mean regime duration: {all_durations.mean():.1f} days")
print(f"  Median regime duration: {all_durations.median():.1f} days")
print(f"  25th percentile: {all_durations.quantile(0.25):.1f} days")
print(f"  75th percentile: {all_durations.quantile(0.75):.1f} days")
print()

# What percentage of regimes last at least X days?
print("REGIME DURATION PERCENTILES")
print("-" * 80)
for days in [7, 14, 21, 30, 45, 60]:
    pct = (all_durations >= days).sum() / len(all_durations) * 100
    print(f"  {pct:.1f}% of regimes last >= {days} days")
print()

# Recent 90 days analysis
recent = df_daily.tail(90)
recent_regime_changes = recent['regime'] != recent['regime'].shift(1)
num_regime_changes = recent_regime_changes.sum()

print("RECENT 90-DAY ANALYSIS")
print("-" * 80)
print(f"  Regime changes: {num_regime_changes}")
print(f"  Average regime duration: {90 / (num_regime_changes + 1):.1f} days")
print(f"  Current regime: {recent['regime'].iloc[-1]}")
print()

print("=" * 80)
print("LOOKBACK PERIOD RECOMMENDATIONS")
print("=" * 80)
print()

median_duration = all_durations.median()
p75_duration = all_durations.quantile(0.75)

print(f"Based on regime analysis:")
print(f"  • Median regime lasts: {median_duration:.0f} days")
print(f"  • 75% of regimes last: {p75_duration:.0f} days or less")
print()
print("Recommended lookback periods:")
print()
print("SHORT-TERM (Fast Response):")
print(f"  → 7 days")
print(f"     • Catches regime changes within 1 week")
print(f"     • {(all_durations >= 7).sum() / len(all_durations) * 100:.0f}% of regimes last this long")
print(f"     • Good for rapid adaptation")
print()
print("MEDIUM-TERM (Balanced):")
print(f"  → 14-21 days")
print(f"     • Captures typical regime duration")
print(f"     • {(all_durations >= 14).sum() / len(all_durations) * 100:.0f}% of regimes last >= 14 days")
print(f"     • {(all_durations >= 21).sum() / len(all_durations) * 100:.0f}% of regimes last >= 21 days")
print(f"     • Balances responsiveness and stability")
print()
print("LONG-TERM (Stability Anchor):")
print(f"  → 30-60 days")
print(f"     • Includes multiple regime changes")
print(f"     • {(all_durations >= 30).sum() / len(all_durations) * 100:.0f}% of regimes last >= 30 days")
print(f"     • Useful for context and bounds")
print()

print("=" * 80)
print("OPTIMAL CONFIGURATION")
print("=" * 80)
print()
print("For Confidence Distribution (predictions every hour):")
print("  Primary lookback: 14 days (336 predictions)")
print("    • Captures 1-2 typical regime cycles")
print("    • Large enough for stable percentiles")
print("    • Responsive to regime changes")
print()
print("For Performance Feedback (trade outcomes):")
print("  Primary lookback: 25 trades (~15-20 days at 0.60 threshold)")
print("    • Statistically meaningful win rate")
print("    • Aligns with regime duration")
print("    • Fast enough to adapt, slow enough to avoid noise")
print()
print("Update Frequency:")
print("  • Confidence distribution: Every prediction (hourly)")
print("  • Performance feedback: After each trade")
print("  • Threshold calculation: Every prediction")
print("  • Logging/review: Daily summary")
print()
