# WFO Monthly Disaggregation Feature Mismatch Fix

## Problem

The `wfo_monthly_disaggregation.py` script was generating features manually, which resulted in a feature count mismatch:
- **Generated**: 110 features
- **Expected by Daily model**: 141 features (31 features missing)
- **Missing features**: Sentiment-derived features (VIX, EPU, and their transformations)

## Root Cause

The trained WFO Daily models included sentiment features, but:
1. The feature engine configuration saved with the models had `include_sentiment_features: False`
2. Manual feature generation in the disaggregation script didn't load sentiment data
3. This caused a mismatch between training (141 features) and inference (110 features)

## Solution

### 1. Modified `walk_forward_optimization.py`

**Added trade-level timestamps** (lines 892-893):
```python
trades.append({
    ...
    "entry_time": entry_time,
    "exit_time": timestamps[exit_idx],  # Track exit timestamp
})
```

**Added sentiment support** (lines 567-586):
- Import `SentimentLoader`
- Auto-detect if Daily model was trained with sentiment (feature count > 120)
- Enable sentiment features if needed
- Update sentiment data path to current files

**Added `return_trades` parameter** (line 500):
- Allows function to return raw trade list in addition to aggregated metrics
- Enables reuse of proven WFO logic for disaggregation

### 2. Rewrote `wfo_monthly_disaggregation.py`

**Key changes:**
- **Removed manual feature generation** (eliminated 200+ lines of duplicated logic)
- **Calls `run_window_backtest()`** with `return_trades=True` to reuse validated WFO logic
- **Aggregates by period** using pandas Period functionality
- **Supports multiple periods**: Weekly (W), Monthly (M), Quarterly (Q)

**Benefits:**
- ✅ **Feature parity**: Uses exact same feature generation as WFO training
- ✅ **No code duplication**: Reuses proven backtest function
- ✅ **Maintainability**: Single source of truth for trading logic
- ✅ **Flexibility**: Easy to switch between aggregation periods

## Results

Successfully generated monthly disaggregation for conf=0.60:

```
Total Months: 22 (2022-01 to 2025-09)
Total Trades: 1,087
Avg Trades/Month: 49.4
Total Pips: +5,979.1
Total PnL: $74,365.92
Avg Month Return: +22.96%

Best Month: 2022-03 (+$12,114, +78.37%)
Worst Month: 2024-09 (-$276, -1.94%)
```

## Files Modified

1. `/home/sergio/ai-trader/backend/scripts/walk_forward_optimization.py`
   - Added `SentimentLoader` import
   - Added trade timestamp tracking
   - Added sentiment auto-detection for Daily model
   - Added `return_trades` parameter

2. `/home/sergio/ai-trader/backend/scripts/wfo_monthly_disaggregation.py`
   - Complete rewrite (527 lines → 330 lines)
   - Removed manual feature generation
   - Now calls `run_window_backtest()` with `return_trades=True`

## Usage

```bash
# Monthly disaggregation (default)
.venv/bin/python scripts/wfo_monthly_disaggregation.py \
    --period M \
    --output-csv data/wfo_conf60_monthly.csv \
    --output-json data/wfo_conf60_monthly.json \
    --confidence 0.60

# Weekly disaggregation
.venv/bin/python scripts/wfo_monthly_disaggregation.py \
    --period W \
    --output-csv data/wfo_conf60_weekly.csv \
    --output-json data/wfo_conf60_weekly.json \
    --confidence 0.60

# Quarterly disaggregation
.venv/bin/python scripts/wfo_monthly_disaggregation.py \
    --period Q \
    --output-csv data/wfo_conf60_quarterly.csv \
    --output-json data/wfo_conf60_quarterly.json \
    --confidence 0.60
```

## Technical Details

### Sentiment Feature Auto-Detection

The fix automatically detects if a model needs sentiment by comparing expected feature count:
```python
expected_features = len(model_d.feature_names)
if expected_features > 120 and not model_d.feature_engine.include_sentiment_features:
    model_d.feature_engine.include_sentiment_features = True
    # Update sentiment path to current data
```

This works because:
- Models without sentiment: ~110 features
- Models with sentiment: ~141 features (31 sentiment-derived features)

### Sentiment Features Added

When enabled, the feature engine adds 31 sentiment features:
- `sent_us`, `sent_vix`, `sent_us_combined`
- `sentiment_raw`, `vix_raw`
- Moving averages: `sentiment_ma_3`, `sentiment_ma_7`, `sentiment_ma_14`, `sentiment_ma_30`
- Standard deviations: `sentiment_std_7`, `sentiment_std_14`, `sentiment_std_30`
- Momentum: `sentiment_momentum_3`, `sentiment_momentum_7`, `sentiment_momentum_14`
- ROC: `sentiment_roc_7`, `sentiment_roc_14`
- Regime: `sentiment_regime`, `sentiment_zscore`
- Lags: `sentiment_lag_1`, `sentiment_lag_2`, `sentiment_lag_3`
- VIX features: `vix_roc_3`, `vix_roc_5`, `vix_ma_5`, `vix_ma_20`, `vix_regime`, `vix_zscore`
- Cross features: `cross_sent_mean`, `cross_sent_std`, `cross_sent_range`

## Validation

The fix has been tested and validated:
- ✅ All 8 WFO windows processed successfully
- ✅ 1,087 trades generated across 22 months
- ✅ Feature count matches (141 features)
- ✅ CSV and JSON outputs generated
- ✅ Results match expected WFO performance patterns

## Impact on Other Scripts

The `walk_forward_optimization.py` changes are **backward compatible**:
- `return_trades` parameter defaults to `False` (existing behavior)
- Timestamp fields added to trades don't affect aggregated metrics
- Sentiment auto-detection only affects models with feature count > 120

No other scripts need modification.
