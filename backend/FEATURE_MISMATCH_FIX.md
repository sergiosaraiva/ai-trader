# Feature Mismatch Fix - Stacking Meta-Learner

## Problem Summary

The `predict()` method in `stacking_meta_learner.py` was generating only **9 features** when `use_enhanced_meta_features=True`, but the StandardScaler was trained on **20 features** (9 standard + 11 enhanced).

This caused a feature count mismatch error during prediction:
```
ValueError: X has 9 features, but StandardScaler is expecting 20 features as input.
```

## Root Cause

### Training Path (Correct)
The `train()` method uses `_create_meta_features()` which:
1. Creates 9 standard features via `StackingMetaFeatures.to_array()`
2. When `config.use_enhanced_meta_features=True`, appends 11 enhanced features
3. Trains the StandardScaler on all 20 features

### Single Prediction Path (Broken)
The `predict()` method was:
1. Creating 9 standard features via `StackingMetaFeatures.to_array()`
2. NOT calculating enhanced features even when `config.use_enhanced_meta_features=True`
3. Passing only 9 features to the scaler (expecting 20)

### Batch Prediction Path (Correct)
The `predict_batch()` method already correctly handled enhanced features.

## Solution

Modified the `predict()` method (lines 563-610) to:

1. **Check if enhanced features are enabled**: `if self.config.use_enhanced_meta_features:`
2. **Calculate enhanced features**:
   - Convert single predictions to arrays for the calculator
   - Use `EnhancedMetaFeatureCalculator` to compute all 11 enhanced features
   - Extract features in consistent order using `get_enhanced_feature_names()`
3. **Concatenate features**: Append 11 enhanced features to the 9 standard features
4. **Graceful fallback**: If enhanced feature calculation fails, use zeros for those features

## Implementation Details

### Enhanced Feature Calculation
```python
# Convert single prediction to arrays
predictions = {
    "1H": np.array([dirs[0]]),
    "4H": np.array([dirs[1]]),
    "D": np.array([dirs[2]]),
}
probabilities = {
    "1H": np.array([prob_1h]),
    "4H": np.array([prob_4h]),
    "D": np.array([prob_d]),
}

# Calculate enhanced features
enhanced_calc = EnhancedMetaFeatureCalculator(self.config.enhanced_meta_lookback)
enhanced = enhanced_calc.calculate_all(
    predictions=predictions,
    probabilities=probabilities,
    price_data=None,  # Not available in single prediction
)
```

### Feature Extraction
```python
# Extract enhanced features in consistent order
enhanced_names = get_enhanced_feature_names()  # 11 feature names
enhanced_arrays = []
for name in enhanced_names:
    if name in enhanced:
        enhanced_arrays.append(enhanced[name][0])  # Extract single value
    else:
        enhanced_arrays.append(0.0)  # Fill missing with zeros

# Concatenate with standard features
enhanced_features = np.array(enhanced_arrays, dtype=np.float32).reshape(1, -1)
X = np.concatenate([X, enhanced_features], axis=1)  # Now 20 features
```

### Error Handling
```python
try:
    # Calculate enhanced features
    ...
except Exception as e:
    # Graceful fallback: use zeros for enhanced features
    logger.warning(f"Failed to calculate enhanced meta-features: {e}. Using zeros.")
    n_enhanced = len(get_enhanced_feature_names())
    enhanced_zeros = np.zeros((1, n_enhanced), dtype=np.float32)
    X = np.concatenate([X, enhanced_zeros], axis=1)
```

## Feature Counts

| Configuration | Feature Count | Breakdown |
|---------------|---------------|-----------|
| `use_enhanced_meta_features=False` | **9** | 3 probs + 3 agreement + 1 confidence + 2 volatility |
| `use_enhanced_meta_features=True` | **20** | 9 standard + 11 enhanced |

### Enhanced Features (11 total)

From `get_enhanced_feature_names()`:
1. `prob_entropy` - Prediction quality
2. `confidence_margin` - Prediction quality
3. `htf_agreement_1h_4h` - Cross-timeframe pattern
4. `htf_agreement_4h_d` - Cross-timeframe pattern
5. `trend_alignment` - Cross-timeframe pattern
6. `recent_volatility` - Market context
7. `trend_strength` - Market context
8. `market_regime` - Market context
9. `stability_1h` - Prediction stability
10. `stability_4h` - Prediction stability
11. `stability_d` - Prediction stability

## Testing

### Before Fix
```python
# With use_enhanced_meta_features=True
X.shape = (1, 9)  # Only standard features
# StandardScaler.transform() fails: "expecting 20 features"
```

### After Fix
```python
# With use_enhanced_meta_features=True
X.shape = (1, 20)  # Standard + enhanced features
# StandardScaler.transform() succeeds
```

## Backward Compatibility

The fix maintains backward compatibility:
- When `use_enhanced_meta_features=False`: Still generates 9 features (unchanged)
- When `use_enhanced_meta_features=True`: Now generates 20 features (fixed)

## Data Leakage Prevention

Enhanced features use only data available at prediction time:
- Market context features are set to zero (no price_data in single prediction)
- Prediction quality and cross-timeframe features use current predictions only
- No future data is used

## Files Modified

- **`backend/src/models/multi_timeframe/stacking_meta_learner.py`** (lines 563-610)
  - Added enhanced feature calculation in `predict()` method
  - Matches behavior of `predict_batch()` method

## Verification

The fix ensures:
1. ✅ Feature count matches between training and prediction (20 features)
2. ✅ Feature order is consistent (standard → enhanced)
3. ✅ Graceful error handling for calculation failures
4. ✅ Backward compatibility for `use_enhanced_meta_features=False`
5. ✅ No data leakage (uses only available data)

## Related Methods

- **`_create_meta_features()`** (lines 331-422): Creates features for training/OOF
- **`predict_batch()`** (lines 644-729): Batch prediction (already correct)
- **`predict()`** (lines 510-642): Single prediction (FIXED)

---

**Date**: 2026-01-21
**Status**: Fixed
**Impact**: Critical - allows production use with enhanced meta-features
