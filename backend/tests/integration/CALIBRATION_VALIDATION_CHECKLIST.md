# Calibration Validation Checklist

Use this checklist when validating the Prediction Calibration feature for production deployment.

## Pre-Validation Checklist

- [ ] **Data Available**: Verify EUR/USD 5-minute data exists
  ```bash
  ls -lh backend/data/forex/EURUSD_20200101_20251231_5min_combined.csv
  ```

- [ ] **Models Trained**: Verify MTF Ensemble models exist
  ```bash
  ls backend/models/mtf_ensemble/*.pkl
  ```

- [ ] **Environment Ready**: Install test dependencies
  ```bash
  cd backend
  pip install -r requirements-test.txt
  ```

## Run Validation Tests

### Step 1: Quick Smoke Test (30 seconds)

```bash
cd backend
pytest tests/unit/models/test_prediction_calibrator.py -v
```

**Expected**: All unit tests pass ✓

---

### Step 2: Integration Test (2 minutes)

```bash
pytest tests/integration/test_calibration_production_validation.py::TestCalibrationMTFEnsembleIntegration -v
```

**Expected**: All 4 integration tests pass ✓

---

### Step 3: Calibration Quality Test (2 minutes)

```bash
pytest tests/integration/test_calibration_production_validation.py::TestCalibrationQuality -v -s
```

**Expected**:
- ✓ ECE improves after calibration
- ✓ Confidence matches empirical accuracy

**Key Metrics to Check**:
- ECE improvement: Should be positive
- Confidence vs Win Rate: Should align (error < 0.15)

---

### Step 4: Trading Performance Test (2 minutes)

```bash
pytest tests/integration/test_calibration_production_validation.py::TestCalibrationTradingPerformance -v -s
```

**Expected**:
- ✓ Profit factor maintained or improved
- ✓ Total pips maintained (within 5%)
- ✓ Win rate matches confidence buckets

**Key Metrics to Check**:
- Profit Factor: Should not degrade by >10%
- Total Pips: Should not degrade by >5%
- Win Rate: Should be consistent across buckets

---

### Step 5: **DECISION TEST** (3 minutes) ⭐

```bash
pytest tests/integration/test_calibration_production_validation.py::TestCalibrationDecisionCriteria::test_deployment_decision -v -s
```

**This is the main test that determines deployment.**

**Expected Output**:

```
==================== CALIBRATION DEPLOYMENT DECISION ====================

--- BEFORE CALIBRATION ---
ECE:          [value]
Total Pips:   [value]
Win Rate:     [value]%
Profit Factor: [value]
Trades:       [value]

--- AFTER CALIBRATION ---
ECE:          [value] ([change]%)
Total Pips:   [value] ([change]%)
Win Rate:     [value]%
Profit Factor: [value]
Trades:       [value]

--- DECISION CRITERIA ---
ECE improvement > 20%:     [True/False] ([value]%)
Pips not degraded > 5%:    [True/False] ([value]%)

🎯 DECISION: [DEPLOY/ROLLBACK/MARGINAL]
=========================================================================
```

**Decision Matrix**:

| ECE Improvement | Pips Change | Decision | Action |
|----------------|-------------|----------|--------|
| >20% | >-5% | **DEPLOY** | ✅ Enable in production |
| >20% | <-5% | **ROLLBACK** | ❌ Keep disabled |
| <20% | >-5% | **MARGINAL** | ⚠️ Optional |
| Any | <-5% | **ROLLBACK** | ❌ Keep disabled |
| Negative | Any | **ROLLBACK** | ❌ Keep disabled |

---

### Step 6: Full Test Suite (Optional, 10 minutes)

```bash
pytest tests/integration/test_calibration_production_validation.py -v -s
```

**Run this for comprehensive validation before major releases.**

---

## Post-Validation Checklist

### If Decision = DEPLOY ✅

- [ ] **Review Metrics**: Confirm ECE improvement and pips maintained
- [ ] **Update Config**: Enable calibration in production config
  ```python
  config = MTFEnsembleConfig(
      use_calibration=True,
      calibration_method="isotonic",
      # ... other settings
  )
  ```
- [ ] **Update Training**: Ensure training script fits calibrator
  ```bash
  python scripts/train_mtf_ensemble.py --sentiment --calibration isotonic
  ```
- [ ] **Update Backtest**: Enable calibration in backtest script
  ```bash
  python scripts/backtest_mtf_ensemble.py --calibration isotonic
  ```
- [ ] **Documentation**: Update README with calibration feature
- [ ] **Monitoring**: Add calibration ECE to production metrics
- [ ] **Merge PR**: Merge calibration feature branch

### If Decision = ROLLBACK ❌

- [ ] **Review Failure Reason**: Check why calibration degraded performance
  - ECE increased? → Calibration method may not fit data distribution
  - Pips degraded? → Confidence threshold filtering may be too aggressive
- [ ] **Try Alternative Methods**:
  ```bash
  # Try Platt scaling
  pytest tests/integration/test_calibration_production_validation.py -v -s -k "platt"

  # Try Temperature scaling
  pytest tests/integration/test_calibration_production_validation.py -v -s -k "temperature"
  ```
- [ ] **Collect More Data**: Increase calibration set size
- [ ] **Keep Feature Disabled**: Set `use_calibration=False` (default)
- [ ] **Document Findings**: Add to project docs why calibration was not deployed

### If Decision = MARGINAL ⚠️

- [ ] **Evaluate Trade-offs**:
  - Does modest ECE improvement justify added complexity?
  - Would it help specific use cases (e.g., high-confidence trades only)?
- [ ] **Keep Disabled by Default**: Prefer simplicity
- [ ] **Make Optional**: Allow users to enable if desired
- [ ] **Re-evaluate Later**: With more data or better calibration methods

---

## Quick Reference Commands

### Development
```bash
# Run unit tests only
pytest tests/unit/models/test_prediction_calibrator.py -v

# Run integration tests only
pytest tests/integration/test_calibration_production_validation.py -v

# Run decision test only
pytest tests/integration/test_calibration_production_validation.py::TestCalibrationDecisionCriteria -v -s
```

### Training
```bash
# Train with calibration
python scripts/train_mtf_ensemble.py --sentiment --calibration isotonic

# Train with different calibration methods
python scripts/train_mtf_ensemble.py --sentiment --calibration platt
python scripts/train_mtf_ensemble.py --sentiment --calibration temperature
```

### Backtesting
```bash
# Backtest with calibration
python scripts/backtest_mtf_ensemble.py --calibration isotonic

# Backtest without calibration (baseline)
python scripts/backtest_mtf_ensemble.py
```

---

## Troubleshooting

### Common Issues

**Issue**: `Data file not found`
```bash
# Solution: Download data
python scripts/download_market_data.py --pair EURUSD --start 2020-01-01 --end 2025-12-31
```

**Issue**: `Trained models not found`
```bash
# Solution: Train models first
python scripts/train_mtf_ensemble.py --sentiment
```

**Issue**: `Insufficient calibration data`
```bash
# Solution: Check data file has enough bars
wc -l backend/data/forex/EURUSD_20200101_20251231_5min_combined.csv
# Should be > 100,000 lines
```

**Issue**: Tests are slow
```bash
# Solution: Run decision test only for quick validation
pytest tests/integration/test_calibration_production_validation.py::TestCalibrationDecisionCriteria::test_deployment_decision -v -s
```

---

## Success Criteria Summary

| Metric | Target | Critical |
|--------|--------|----------|
| **ECE Improvement** | >20% | ✅ YES |
| **Pips Change** | >-5% | ✅ YES |
| Profit Factor | Maintained | ⚠️ Important |
| Win Rate | Maintained | ⚠️ Important |
| Latency | <1ms/pred | ⚠️ Important |
| Test Pass Rate | 100% | ✅ YES |

---

## Sign-Off

Before deploying calibration to production, confirm:

- [ ] All validation tests pass
- [ ] Decision test recommends DEPLOY
- [ ] ECE improves by >20%
- [ ] Total pips maintained (within 5%)
- [ ] Code reviewed and approved
- [ ] Documentation updated
- [ ] Production config updated

**Validated by**: ___________________
**Date**: ___________________
**Decision**: [ ] DEPLOY  [ ] ROLLBACK  [ ] MARGINAL
**Notes**: ___________________

---

**Remember**: The goal is to improve probability calibration (ECE) without sacrificing trading performance (pips). When in doubt, keep calibration disabled for simplicity.
