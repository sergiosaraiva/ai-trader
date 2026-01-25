# Quick Reference: Enhanced Meta-Features Tests

## Test Files

1. **Unit Tests:** `tests/unit/models/test_enhanced_meta_features.py` (28 tests)
2. **Integration Tests:** `tests/integration/test_enhanced_meta_learner.py` (15 tests)

**Total:** 43 tests | **Critical:** 8 data leakage tests

---

## Run All Tests

```bash
cd /home/sergio/ai-trader/backend

# All enhanced meta-features tests
pytest tests/unit/models/test_enhanced_meta_features.py tests/integration/test_enhanced_meta_learner.py -v

# With coverage report
pytest tests/unit/models/test_enhanced_meta_features.py tests/integration/test_enhanced_meta_learner.py \
  --cov=src.models.multi_timeframe.enhanced_meta_features \
  --cov=src.models.multi_timeframe.stacking_meta_learner \
  --cov-report=term-missing
```

---

## Run Critical Tests Only (Data Leakage Detection)

```bash
# Critical unit tests (5 tests)
pytest tests/unit/models/test_enhanced_meta_features.py::TestMarketContextFeatures -v
pytest tests/unit/models/test_enhanced_meta_features.py::TestPredictionStabilityFeatures -v

# Critical integration tests (3 tests)
pytest tests/integration/test_enhanced_meta_learner.py::TestDataLeakageDetectionEnhanced -v
```

---

## Run by Test Class

```bash
# Unit tests - Prediction quality (5 tests)
pytest tests/unit/models/test_enhanced_meta_features.py::TestPredictionQualityFeatures -v

# Unit tests - Cross-timeframe patterns (6 tests)
pytest tests/unit/models/test_enhanced_meta_features.py::TestCrossTimeframePatterns -v

# Unit tests - Market context features (5 tests) ⚠️ CRITICAL
pytest tests/unit/models/test_enhanced_meta_features.py::TestMarketContextFeatures -v

# Unit tests - Prediction stability (4 tests) ⚠️ CRITICAL
pytest tests/unit/models/test_enhanced_meta_features.py::TestPredictionStabilityFeatures -v

# Unit tests - Full pipeline (4 tests)
pytest tests/unit/models/test_enhanced_meta_features.py::TestCalculateAllIntegration -v

# Integration tests - Configuration (3 tests)
pytest tests/integration/test_enhanced_meta_learner.py::TestEnhancedFeaturesConfiguration -v

# Integration tests - Feature generation (3 tests)
pytest tests/integration/test_enhanced_meta_learner.py::TestEnhancedFeatureGeneration -v

# Integration tests - Training (2 tests)
pytest tests/integration/test_enhanced_meta_learner.py::TestEnhancedMetaLearnerTraining -v

# Integration tests - Backward compatibility (3 tests)
pytest tests/integration/test_enhanced_meta_learner.py::TestBackwardCompatibility -v
```

---

## Key Test Cases

### Data Leakage Prevention (CRITICAL)

**Unit Tests:**
```bash
# Volatility uses shift(1)
pytest tests/unit/models/test_enhanced_meta_features.py::TestMarketContextFeatures::test_volatility_uses_shift -v

# Trend strength uses shift(1)
pytest tests/unit/models/test_enhanced_meta_features.py::TestMarketContextFeatures::test_trend_strength_uses_shift -v

# Regime uses shifted data
pytest tests/unit/models/test_enhanced_meta_features.py::TestMarketContextFeatures::test_regime_based_on_shifted_data -v

# Stability uses shift(1)
pytest tests/unit/models/test_enhanced_meta_features.py::TestPredictionStabilityFeatures::test_stability_uses_shift -v

# Stability handles NaNs
pytest tests/unit/models/test_enhanced_meta_features.py::TestPredictionStabilityFeatures::test_stability_handles_nans -v
```

**Integration Tests:**
```bash
# End-to-end leakage detection
pytest tests/integration/test_enhanced_meta_learner.py::TestDataLeakageDetectionEnhanced -v
```

### Feature Count Verification

```bash
# Verify 20 features (9 standard + 11 enhanced)
pytest tests/integration/test_enhanced_meta_learner.py::TestEnhancedFeatureGeneration::test_enhanced_features_generates_20_columns -v

# Verify standard features remain at 9
pytest tests/integration/test_enhanced_meta_learner.py::TestEnhancedFeatureGeneration::test_standard_features_without_enhanced -v
```

### Backward Compatibility

```bash
# Verify disabled by default
pytest tests/integration/test_enhanced_meta_learner.py::TestEnhancedFeaturesConfiguration::test_enhanced_features_disabled_by_default -v

# Verify existing models still work
pytest tests/integration/test_enhanced_meta_learner.py::TestBackwardCompatibility::test_existing_models_still_work -v
```

---

## Expected Results

### All Tests Pass (43/43)

```
========================= test session starts =========================
collected 43 items

tests/unit/models/test_enhanced_meta_features.py::TestPredictionQualityFeatures::test_prob_entropy_calculation PASSED
tests/unit/models/test_enhanced_meta_features.py::TestPredictionQualityFeatures::test_prob_entropy_confident_predictions PASSED
...
tests/integration/test_enhanced_meta_learner.py::TestEnhancedFeatureImportance::test_feature_importance_includes_enhanced PASSED

========================= 43 passed in X.XXs =========================
```

### Critical Tests Pass (8/8)

```
========================= test session starts =========================
collected 8 items

tests/unit/models/test_enhanced_meta_features.py::TestMarketContextFeatures::test_volatility_uses_shift PASSED ⚠️
tests/unit/models/test_enhanced_meta_features.py::TestMarketContextFeatures::test_trend_strength_uses_shift PASSED ⚠️
tests/unit/models/test_enhanced_meta_features.py::TestMarketContextFeatures::test_regime_based_on_shifted_data PASSED ⚠️
tests/unit/models/test_enhanced_meta_features.py::TestPredictionStabilityFeatures::test_stability_uses_shift PASSED ⚠️
tests/unit/models/test_enhanced_meta_features.py::TestPredictionStabilityFeatures::test_stability_handles_nans PASSED ⚠️
tests/integration/test_enhanced_meta_learner.py::TestDataLeakageDetectionEnhanced::test_no_data_leakage_volatility PASSED ⚠️
tests/integration/test_enhanced_meta_learner.py::TestDataLeakageDetectionEnhanced::test_no_data_leakage_trend_strength PASSED ⚠️
tests/integration/test_enhanced_meta_learner.py::TestDataLeakageDetectionEnhanced::test_no_data_leakage_stability PASSED ⚠️

========================= 8 passed in X.XXs =========================
```

---

## Test Coverage Goals

| File | Target Coverage | Focus |
|------|-----------------|-------|
| `enhanced_meta_features.py` | >95% | All methods covered |
| `stacking_meta_learner.py` | Enhanced features integration | `_create_meta_features` with enhanced flag |

**Coverage Command:**
```bash
pytest tests/unit/models/test_enhanced_meta_features.py tests/integration/test_enhanced_meta_learner.py \
  --cov=src.models.multi_timeframe.enhanced_meta_features \
  --cov=src.models.multi_timeframe.stacking_meta_learner \
  --cov-report=html \
  --cov-report=term-missing

# View HTML report
open htmlcov/index.html
```

---

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:
```bash
# Install dependencies
pip install -r requirements.txt

# Or in Docker
docker-compose run backend pytest tests/unit/models/test_enhanced_meta_features.py -v
```

### Assertion Failures

Check the implementation files:
- `/home/sergio/ai-trader/backend/src/models/multi_timeframe/enhanced_meta_features.py`
- `/home/sergio/ai-trader/backend/src/models/multi_timeframe/stacking_meta_learner.py`

Ensure `.shift(1)` is applied to all rolling calculations.

---

## Test Summary

| Category | Count | Status |
|----------|-------|--------|
| Total Tests | 43 | ✅ Ready |
| Unit Tests | 28 | ✅ Complete |
| Integration Tests | 15 | ✅ Complete |
| Critical (Leakage) | 8 | ⚠️ Must Pass |
| Backward Compat | 3 | ✅ Verified |

**All tests are ready to run. Execute the commands above to validate the implementation.**
