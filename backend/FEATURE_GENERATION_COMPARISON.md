# Feature Generation Comparison - Stacking Meta-Learner

This document compares how features are generated across the three prediction paths to verify consistency.

## Feature Generation Paths

### 1. Training/OOF Generation (`_create_meta_features()`)

**Location**: Lines 331-422

**Process**:
```python
def _create_meta_features(predictions, probabilities, confidences, price_data, volatility):
    # Step 1: Create standard features (9 features)
    for i in range(n_samples):
        meta_feat = StackingMetaFeatures(...)
        meta_features_list.append(meta_feat.to_array(self.config))

    meta_features = np.array(meta_features_list)  # Shape: (n, 9)

    # Step 2: Add enhanced features if enabled (11 features)
    if self.config.use_enhanced_meta_features:
        enhanced_calc = EnhancedMetaFeatureCalculator(...)
        enhanced = enhanced_calc.calculate_all(predictions, probabilities, price_data)
        enhanced_features = np.column_stack([enhanced[name] for name in enhanced_names])
        meta_features = np.concatenate([meta_features, enhanced_features], axis=1)  # Shape: (n, 20)

    return meta_features
```

**Result**:
- Standard only: (n, 9)
- With enhanced: (n, 20)

---

### 2. Single Prediction (`predict()`) - FIXED

**Location**: Lines 510-642

**Process**:
```python
def predict(prob_1h, prob_4h, prob_d, conf_1h, conf_4h, conf_d, volatility, ...):
    # Step 1: Create standard features (9 features)
    meta_feat = StackingMetaFeatures(...)
    X = meta_feat.to_array(self.config).reshape(1, -1)  # Shape: (1, 9)

    # Step 2: Add enhanced features if enabled (11 features) - FIXED
    if self.config.use_enhanced_meta_features:
        try:
            # Convert single values to arrays
            predictions = {"1H": np.array([dir_1h]), "4H": ..., "D": ...}
            probabilities = {"1H": np.array([prob_1h]), "4H": ..., "D": ...}

            enhanced_calc = EnhancedMetaFeatureCalculator(...)
            enhanced = enhanced_calc.calculate_all(predictions, probabilities, price_data=None)

            # Extract single values from enhanced feature arrays
            enhanced_arrays = [enhanced[name][0] for name in enhanced_names]
            enhanced_features = np.array(enhanced_arrays).reshape(1, -1)

            X = np.concatenate([X, enhanced_features], axis=1)  # Shape: (1, 20)

        except Exception as e:
            # Fallback: use zeros for enhanced features
            enhanced_zeros = np.zeros((1, n_enhanced))
            X = np.concatenate([X, enhanced_zeros], axis=1)  # Shape: (1, 20)

    X_scaled = self.meta_scaler.transform(X)
    return ...
```

**Result**:
- Standard only: (1, 9)
- With enhanced: (1, 20) ✅ FIXED

---

### 3. Batch Prediction (`predict_batch()`)

**Location**: Lines 644-729

**Process**:
```python
def predict_batch(probs_1h, probs_4h, probs_d, confs_1h, confs_4h, confs_d, volatility, price_data, ...):
    # Step 1: Create standard features (9 features)
    meta_features = []
    for i in range(n):
        meta_feat = StackingMetaFeatures(...)
        meta_features.append(meta_feat.to_array(self.config))

    X = np.array(meta_features)  # Shape: (n, 9)

    # Step 2: Add enhanced features if enabled (11 features)
    if self.config.use_enhanced_meta_features:
        predictions = {
            "1H": (probs_1h > 0.5).astype(int),
            "4H": (probs_4h > 0.5).astype(int),
            "D": (probs_d > 0.5).astype(int),
        }
        probabilities = {"1H": probs_1h, "4H": probs_4h, "D": probs_d}

        enhanced_calc = EnhancedMetaFeatureCalculator(...)
        enhanced = enhanced_calc.calculate_all(predictions, probabilities, price_data)

        enhanced_arrays = [enhanced[name] for name in enhanced_names]
        enhanced_features = np.column_stack(enhanced_arrays)

        X = np.concatenate([X, enhanced_features], axis=1)  # Shape: (n, 20)

    X_scaled = self.meta_scaler.transform(X)
    return ...
```

**Result**:
- Standard only: (n, 9)
- With enhanced: (n, 20) ✅ Already correct

---

## Feature Consistency Matrix

| Method | Standard Only | With Enhanced | Status |
|--------|---------------|---------------|--------|
| `_create_meta_features()` | (n, 9) | (n, 20) | ✅ Correct |
| `predict()` | (1, 9) | (1, 20) | ✅ FIXED |
| `predict_batch()` | (n, 9) | (n, 20) | ✅ Correct |

## Key Differences in Implementation

### Data Conversion

| Method | Input Format | Conversion |
|--------|-------------|------------|
| `_create_meta_features()` | Dict of arrays | Already arrays |
| `predict()` | Single floats | Convert to single-element arrays |
| `predict_batch()` | Arrays | Already arrays |

### Enhanced Feature Extraction

| Method | Extraction |
|--------|-----------|
| `_create_meta_features()` | `np.column_stack([enhanced[name] for name in names])` |
| `predict()` | `[enhanced[name][0] for name in names]` (extract single value) |
| `predict_batch()` | `np.column_stack([enhanced[name] for name in names])` |

### Price Data

| Method | price_data Value | Impact |
|--------|------------------|--------|
| `_create_meta_features()` | None (OOF) | Market context features = zeros |
| `predict()` | None | Market context features = zeros |
| `predict_batch()` | Optional DataFrame | Market context features calculated if provided |

## Error Handling

### `predict()` - Enhanced with Try/Except
```python
try:
    # Calculate enhanced features
    ...
except Exception as e:
    # Fallback: use zeros
    logger.warning(f"Failed to calculate enhanced meta-features: {e}. Using zeros.")
    enhanced_zeros = np.zeros((1, n_enhanced))
    X = np.concatenate([X, enhanced_zeros], axis=1)
```

### `predict_batch()` - No Error Handling
```python
if self.config.use_enhanced_meta_features:
    # Calculate enhanced features (no try/except)
    ...
```

**Note**: `predict()` has more robust error handling for production use.

## Verification Checklist

- [x] All three methods generate 9 standard features consistently
- [x] All three methods append 11 enhanced features when enabled
- [x] Feature concatenation order is consistent (standard → enhanced)
- [x] Feature names match across all methods (via `get_enhanced_feature_names()`)
- [x] StandardScaler receives correct feature count (9 or 20)
- [x] No data leakage (price_data=None in single prediction)
- [x] Graceful error handling in `predict()`

## StandardScaler Feature Expectations

| Config | Expected Features | Training | Prediction | Status |
|--------|------------------|----------|------------|--------|
| `use_enhanced_meta_features=False` | 9 | ✅ 9 | ✅ 9 | Match |
| `use_enhanced_meta_features=True` | 20 | ✅ 20 | ✅ 20 | Match (FIXED) |

---

**Date**: 2026-01-21
**Status**: All paths verified consistent
**Fix Applied**: `predict()` method lines 563-610
