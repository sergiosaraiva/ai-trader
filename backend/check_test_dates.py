#!/usr/bin/env python3
"""Check the exact test period dates."""
import pandas as pd

# Load data
df = pd.read_csv('data/forex/EURUSD_20200101_20251231_5min_combined.csv', index_col=0, parse_dates=True)

# Calculate split points
total_bars = len(df)
train_end_idx = int(total_bars * 0.6)
val_end_idx = int(total_bars * 0.8)

print(f"Total bars: {total_bars:,}")
print()
print(f"Train: {df.index[0]} to {df.index[train_end_idx]}")
print(f"Val:   {df.index[train_end_idx+1]} to {df.index[val_end_idx]}")
print(f"Test:  {df.index[val_end_idx+1]} to {df.index[-1]}")
print()
print(f"Test start date: {df.index[val_end_idx+1]}")
