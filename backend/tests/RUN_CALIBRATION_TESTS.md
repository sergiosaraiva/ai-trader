# Running Calibration Tests

## Quick Start

```bash
cd backend
source ../venv/bin/activate

# Run all calibration tests
pytest tests/unit/models/test_calibration.py tests/integration/test_calibrated_ensemble.py -v
```

## Current Status

⚠️ **API Compatibility Issue**: sklearn 1.8.0 changed `CalibratedClassifierCV` API

**Fix required**: Update `src/models/multi_timeframe/improved_model.py` lines 386-395

## Test Suites

### Unit Tests (21 tests)

```bash
# All unit tests
pytest tests/unit/models/test_calibration.py -v

# Specific test class
pytest tests/unit/models/test_calibration.py::TestCalibrationConfig -v

# Single test
pytest tests/unit/models/test_calibration.py::TestCalibrationConfig::test_calibration_disabled_by_default -v
```

### Integration Tests (18 tests)

```bash
# All integration tests
pytest tests/integration/test_calibrated_ensemble.py -v

# Specific test class
pytest tests/integration/test_calibrated_ensemble.py::TestCalibrationConfiguration -v
```

## Test Categories

### 1. Configuration Tests (Fast)

Tests that verify config settings without training models.

```bash
pytest tests/unit/models/test_calibration.py::TestCalibrationConfig -v
pytest tests/integration/test_calibrated_ensemble.py::TestCalibrationConfiguration -v
```

**Expected**: All pass (✅ 6/6)

### 2. Data Leakage Tests (CRITICAL)

Tests that verify no future data leakage in calibration.

```bash
pytest tests/unit/models/test_calibration.py::TestDataLeakageDetection -v
```

**Expected**: 2/3 pass (1 fails due to API bug, but logic is correct)

### 3. Training & Prediction Tests (Slow)

Tests that train models and make predictions.

```bash
# Unit level
pytest tests/unit/models/test_calibration.py::TestFitCalibrator -v
pytest tests/unit/models/test_calibration.py::TestPredictWithCalibration -v

# Integration level
pytest tests/integration/test_calibrated_ensemble.py::TestEnsembleTrainingWithCalibration -v
pytest tests/integration/test_calibrated_ensemble.py::TestCalibratedPredictions -v
```

**Expected**: Most fail due to API bug

### 4. Serialization Tests

Tests for save/load of calibrated models.

```bash
pytest tests/unit/models/test_calibration.py::TestCalibratorSerializaton -v
pytest tests/integration/test_calibrated_ensemble.py::TestCalibrationPersistence -v
```

**Expected**: Fail due to API bug

## Coverage Report

```bash
# With coverage
pytest tests/unit/models/test_calibration.py tests/integration/test_calibrated_ensemble.py \
    --cov=src.models.multi_timeframe.improved_model \
    --cov=src.models.multi_timeframe.mtf_ensemble \
    --cov-report=html

# View HTML report
open htmlcov/index.html  # or xdg-open on Linux
```

## Test Output

### Successful Test

```
tests/unit/models/test_calibration.py::TestCalibrationConfig::test_calibration_disabled_by_default PASSED [100%]
```

### Failed Test (API Issue)

```
FAILED tests/unit/models/test_calibration.py::TestFitCalibrator::test_fit_calibrator_creates_calibrator
E   sklearn.utils._param_validation.InvalidParameterError: The 'cv' parameter of CalibratedClassifierCV must be...
```

## Current Results

```
===================== test summary =====================
11 passed    ✅ Configuration and conceptual tests
5 failed     ❌ sklearn API compatibility issue
5 errors     ⚠️ Cascading failures from API issue
1 warning    ℹ️ RFECV feature count warning
========================================================
```

## After API Fix

Once `improved_model.py` is fixed, expected results:

```
===================== test summary =====================
39 passed    ✅ All tests pass
0 failed
0 errors
========================================================
```

## Debugging Failed Tests

### View full traceback

```bash
pytest tests/unit/models/test_calibration.py -v --tb=long
```

### Run specific failing test

```bash
pytest tests/unit/models/test_calibration.py::TestFitCalibrator::test_fit_calibrator_creates_calibrator -v -s
```

### Use pdb debugger

```bash
pytest tests/unit/models/test_calibration.py::TestFitCalibrator::test_fit_calibrator_creates_calibrator --pdb
```

## Test Documentation

- **Detailed Report**: `tests/CALIBRATION_TEST_REPORT.md`
- **Test Guide**: `tests/unit/models/README_CALIBRATION_TESTS.md`
- **This File**: Quick reference for running tests

## CI/CD Integration

Add to GitHub Actions workflow:

```yaml
- name: Run Calibration Tests
  run: |
    cd backend
    pytest tests/unit/models/test_calibration.py \
           tests/integration/test_calibrated_ensemble.py \
           -v --tb=short \
           --cov=src.models.multi_timeframe \
           --cov-report=xml

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./backend/coverage.xml
```

## Test Maintenance

### Adding New Tests

1. Follow existing test patterns (AAA: Arrange-Act-Assert)
2. Use descriptive test names
3. Add to appropriate test class
4. Update this document if adding new category

### Updating Tests

1. Run affected tests after code changes
2. Update fixtures if model interface changes
3. Maintain data leakage tests (CRITICAL)

## Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"

**Solution**: Activate virtual environment
```bash
source ../venv/bin/activate
```

### "No tests found"

**Solution**: Check you're in the `backend/` directory
```bash
cd backend
pytest tests/unit/models/test_calibration.py -v
```

### "sklearn API error"

**Solution**: This is a known issue. Fix required in `improved_model.py`

## Contact

For questions about these tests, refer to:
- `tests/CALIBRATION_TEST_REPORT.md` - Full test report
- `.claude/agents/test-automator.md` - Agent documentation
