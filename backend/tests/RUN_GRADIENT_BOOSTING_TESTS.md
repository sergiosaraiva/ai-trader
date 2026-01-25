# Quick Reference: Running Gradient Boosting Tests

## Test File Location

```
backend/tests/unit/models/test_gradient_boosting_frameworks.py
```

## Run All Tests

```bash
cd backend
source ../.venv/bin/activate
pytest tests/unit/models/test_gradient_boosting_frameworks.py -v
```

**Expected Results** (without LightGBM/CatBoost installed):
- 18 passed
- 8 skipped
- 0 failed

## Run Specific Test Categories

### Configuration Tests
```bash
pytest tests/unit/models/test_gradient_boosting_frameworks.py::TestGradientBoostingFrameworks -v
```

### Backward Compatibility Tests
```bash
pytest tests/unit/models/test_gradient_boosting_frameworks.py::TestXGBoostBackwardCompatibility -v
```

### MTF Ensemble Config Tests
```bash
pytest tests/unit/models/test_gradient_boosting_frameworks.py::TestMTFEnsembleModelType -v
```

### Parameter Mapping Tests (requires frameworks installed)
```bash
pytest tests/unit/models/test_gradient_boosting_frameworks.py::TestLightGBMParameterMapping -v
pytest tests/unit/models/test_gradient_boosting_frameworks.py::TestCatBoostParameterMapping -v
```

### Training Compatibility Tests
```bash
pytest tests/unit/models/test_gradient_boosting_frameworks.py::TestFrameworkTrainingCompatibility -v
```

### Framework Availability Tests
```bash
pytest tests/unit/models/test_gradient_boosting_frameworks.py::TestFrameworkAvailabilityFlags -v
```

## Install Frameworks to Run All Tests

To enable all 26 tests (currently 8 are skipped):

```bash
# Install LightGBM
pip install lightgbm>=4.0.0

# Install CatBoost
pip install catboost>=1.2.0

# Run all tests again - should be 26 passed, 0 skipped
pytest tests/unit/models/test_gradient_boosting_frameworks.py -v
```

## Run with Coverage

```bash
pytest tests/unit/models/test_gradient_boosting_frameworks.py --cov=src.models.multi_timeframe.improved_model --cov-report=term-missing -v
```

## Test Output Examples

### Current Output (Frameworks Not Installed)
```
======================== 18 passed, 8 skipped in 2.19s =========================
```

### Expected Output (After Installing Frameworks)
```
======================== 26 passed in 3.5s ======================================
```

## Verify Implementation

### Check Framework Availability
```bash
python3 -c "from src.models.multi_timeframe.improved_model import HAS_LIGHTGBM, HAS_CATBOOST; print(f'LightGBM: {HAS_LIGHTGBM}, CatBoost: {HAS_CATBOOST}')"
```

### Test XGBoost (Default)
```bash
python3 -c "
from src.models.multi_timeframe.improved_model import ImprovedModelConfig, ImprovedTimeframeModel

config = ImprovedModelConfig(name='test', base_timeframe='1H')
model_wrapper = ImprovedTimeframeModel(config)
model = model_wrapper._create_model()
print(f'Model type: {type(model).__name__}')
print('XGBoost works correctly!')
"
```

### Test Error Messages (Without Frameworks)
```bash
python3 -c "
from src.models.multi_timeframe.improved_model import ImprovedModelConfig, ImprovedTimeframeModel

# Test LightGBM error
config = ImprovedModelConfig(name='test', base_timeframe='1H', model_type='lightgbm')
model_wrapper = ImprovedTimeframeModel(config)
try:
    model_wrapper._create_model()
except ImportError as e:
    print(f'LightGBM error: {e}')

# Test CatBoost error
config = ImprovedModelConfig(name='test', base_timeframe='1H', model_type='catboost')
model_wrapper = ImprovedTimeframeModel(config)
try:
    model_wrapper._create_model()
except ImportError as e:
    print(f'CatBoost error: {e}')
"
```

## Test Summary

| Test Category | Tests | Status |
|--------------|-------|--------|
| Configuration | 3 | ✅ All Pass |
| Model Creation | 6 | ✅ 4 Pass, 2 Skip |
| Backward Compatibility | 4 | ✅ All Pass |
| MTF Ensemble Config | 3 | ✅ All Pass |
| Parameter Mapping | 4 | ⏭️ All Skip |
| Training Compatibility | 3 | ✅ 1 Pass, 2 Skip |
| Availability Flags | 3 | ✅ All Pass |
| **Total** | **26** | **18 Pass, 8 Skip** |

## Next Steps

1. Install LightGBM and CatBoost to enable all tests
2. Run comparison script: `python scripts/compare_gradient_boosting.py`
3. Run WFO validation with each framework
4. Compare performance metrics

## Related Files

- Implementation: `src/models/multi_timeframe/improved_model.py`
- Ensemble Config: `src/models/multi_timeframe/mtf_ensemble.py`
- Training Script: `scripts/train_mtf_ensemble.py`
- Comparison Script: `scripts/compare_gradient_boosting.py`
- Test Report: `tests/GRADIENT_BOOSTING_TEST_REPORT.md`

---

**Last Updated**: 2026-01-22
