# Calibration Validation - Quick Reference Card

## 🚀 Run The Decision Test

```bash
cd backend
pytest tests/integration/test_calibration_production_validation.py::TestCalibrationDecisionCriteria::test_deployment_decision -v -s
```

**Runtime**: ~3 minutes

---

## 📊 Decision Matrix

| ECE Improvement | Pips Change | → Decision |
|----------------|-------------|------------|
| **>20%** | **>-5%** | ✅ **DEPLOY** |
| >20% | <-5% | ❌ ROLLBACK |
| <20% | >-5% | ⚠️ MARGINAL |
| <0% (worse) | Any | ❌ ROLLBACK |

---

## 🎯 Key Metrics

### ECE (Expected Calibration Error)
- **Lower is better**
- Measures calibration quality
- **Target**: >20% improvement

### Total Pips
- **Higher is better**
- Measures trading profit
- **Target**: No degradation >5%

### Profit Factor
- **Higher is better**
- Ratio of profit/loss
- **Target**: Maintained or improved

### Win Rate
- **Higher is better**
- % of winning trades
- **Target**: Maintained or improved

---

## ✅ If DEPLOY

1. **Enable calibration**:
   ```python
   config = MTFEnsembleConfig(
       use_calibration=True,
       calibration_method="isotonic"
   )
   ```

2. **Train with calibration**:
   ```bash
   python scripts/train_mtf_ensemble.py --sentiment --calibration isotonic
   ```

3. **Backtest with calibration**:
   ```bash
   python scripts/backtest_mtf_ensemble.py --calibration isotonic
   ```

4. **Monitor**: Add ECE to production metrics

---

## ❌ If ROLLBACK

1. **Keep disabled** (default):
   ```python
   config = MTFEnsembleConfig(
       use_calibration=False  # Default
   )
   ```

2. **Document reason**: Why calibration didn't help

3. **Try alternatives**:
   - Different calibration method (platt, temperature)
   - More calibration data
   - Different confidence thresholds

---

## ⚠️ If MARGINAL

1. **Keep disabled** for simplicity

2. **Make optional** for advanced users

3. **Re-evaluate** with more data later

---

## 🔧 Troubleshooting

### Data not found
```bash
ls -lh backend/data/forex/EURUSD_20200101_20251231_5min_combined.csv
```

### Models not found
```bash
ls backend/models/mtf_ensemble/*.pkl
python scripts/train_mtf_ensemble.py --sentiment
```

### Tests too slow
```bash
# Run decision test only
pytest tests/integration/test_calibration_production_validation.py::TestCalibrationDecisionCriteria -v -s
```

---

## 📚 Documentation

- **Full Guide**: `tests/integration/README_CALIBRATION_VALIDATION.md`
- **Checklist**: `tests/integration/CALIBRATION_VALIDATION_CHECKLIST.md`
- **Summary**: `CALIBRATION_VALIDATION_SUMMARY.md`

---

## 🧪 All Test Commands

```bash
# Full suite (~10 min)
pytest tests/integration/test_calibration_production_validation.py -v -s

# Decision only (~3 min) ⭐
pytest tests/integration/test_calibration_production_validation.py::TestCalibrationDecisionCriteria -v -s

# Integration only (~2 min)
pytest tests/integration/test_calibration_production_validation.py::TestCalibrationMTFEnsembleIntegration -v

# Quality only (~2 min)
pytest tests/integration/test_calibration_production_validation.py::TestCalibrationQuality -v -s

# Trading performance (~2 min)
pytest tests/integration/test_calibration_production_validation.py::TestCalibrationTradingPerformance -v -s

# Unit tests only (~30 sec)
pytest tests/unit/models/test_prediction_calibrator.py -v
```

---

## 📝 Expected Output Format

```
==================== CALIBRATION DEPLOYMENT DECISION ====================

--- BEFORE CALIBRATION ---
ECE:          0.0542
Total Pips:   +1234.5
Win Rate:     58.2%
Profit Factor: 2.15

--- AFTER CALIBRATION ---
ECE:          0.0321 (-40.8%)  ← Should improve
Total Pips:   +1298.3 (+5.2%)  ← Should maintain
Win Rate:     59.1%
Profit Factor: 2.28

--- DECISION CRITERIA ---
ECE improvement > 20%:     True (+40.8%)   ← Key criterion
Pips not degraded > 5%:    True (+5.2%)    ← Key criterion

🎯 DECISION: DEPLOY ✓
=========================================================================
```

---

## ⚡ Quick Actions

| Action | Command |
|--------|---------|
| **Run decision test** | `pytest tests/integration/test_calibration_production_validation.py::TestCalibrationDecisionCriteria::test_deployment_decision -v -s` |
| **Train with calibration** | `python scripts/train_mtf_ensemble.py --sentiment --calibration isotonic` |
| **Backtest with calibration** | `python scripts/backtest_mtf_ensemble.py --calibration isotonic` |
| **Check models** | `ls backend/models/mtf_ensemble/*.pkl` |
| **Check data** | `ls backend/data/forex/*.csv` |

---

## 🎓 Remember

1. **Objective criteria**: ECE >20% improvement + Pips maintained
2. **Real data**: Uses actual EUR/USD data and trained models
3. **No leakage**: Chronologically split data (train → val → calibration → test)
4. **Production-ready**: Tests latency, persistence, efficiency
5. **Clear decision**: DEPLOY, ROLLBACK, or MARGINAL

---

**When in doubt**: Keep calibration disabled. Only deploy if tests clearly show improvement.

---

Print this card and keep it handy during validation! 📄
