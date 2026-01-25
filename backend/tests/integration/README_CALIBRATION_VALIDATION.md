# Calibration Production Validation Tests

## Overview

This test suite validates the Prediction Calibration feature to determine if it should be:
1. **DEPLOYED** to production (improves results)
2. **ROLLED BACK** (doesn't improve or degrades results)

## Test File

`test_calibration_production_validation.py` - Comprehensive production validation tests

## Test Categories

### 1. Integration Tests with MTFEnsemble
- `test_ensemble_with_calibration_enabled` - Verifies ensemble applies calibration
- `test_ensemble_preserves_raw_confidence` - Checks raw confidence preservation
- `test_calibration_doesnt_change_signal_direction` - Ensures signals unchanged
- `test_ensemble_works_without_calibration` - Backward compatibility

### 2. Calibration Quality Tests
- `test_ece_improves_after_calibration` - ECE should decrease
- `test_confidence_matches_empirical_accuracy` - Calibrated probs match outcomes

### 3. Trading Performance Tests
- `test_profit_factor_comparison` - Compare PF with/without calibration
- `test_total_pips_comparison` - Compare total pips
- `test_win_rate_by_confidence_bucket` - Win rate matches confidence

### 4. Robustness Tests
- `test_calibration_persistence_after_save_load` - Save/load cycle

### 5. Production Performance Tests
- `test_calibration_latency_acceptable` - Latency < 1ms per prediction
- `test_batch_calibration_efficient` - Batch faster than N single calls

### 6. Decision Criteria Test
- `test_deployment_decision` - **MAIN DECISION TEST**

## Running the Tests

### Prerequisites

1. **Trained MTF Ensemble models** in `models/mtf_ensemble/`:
   - `1H_model.pkl`
   - `4H_model.pkl`
   - `D_model.pkl`
   - `training_metadata.json`

2. **EUR/USD data** at:
   - `data/forex/EURUSD_20200101_20251231_5min_combined.csv`

### Run Full Test Suite

```bash
cd backend

# Run all calibration validation tests
pytest tests/integration/test_calibration_production_validation.py -v

# Run with detailed output
pytest tests/integration/test_calibration_production_validation.py -v -s

# Run only the decision criteria test
pytest tests/integration/test_calibration_production_validation.py::TestCalibrationDecisionCriteria::test_deployment_decision -v -s
```

### Expected Runtime

- Full suite: ~5-10 minutes (uses real data and models)
- Decision test only: ~2-3 minutes

## Interpreting Results

### Decision Criteria

The `test_deployment_decision` test outputs a clear recommendation:

```
==================== CALIBRATION DEPLOYMENT DECISION ====================

--- BEFORE CALIBRATION ---
ECE:          0.0542
Total Pips:   +1234.5
Win Rate:     58.2%
Profit Factor: 2.15
Trades:       156

--- AFTER CALIBRATION ---
ECE:          0.0321 (-40.8%)
Total Pips:   +1298.3 (+5.2%)
Win Rate:     59.1%
Profit Factor: 2.28
Trades:       148

--- DECISION CRITERIA ---
ECE improvement > 20%:     True (+40.8%)
Pips not degraded > 5%:    True (+5.2%)

🎯 DECISION: DEPLOY ✓
Calibration improves ECE significantly without degrading trading performance.
=========================================================================
```

### Decision Outcomes

| Outcome | Meaning | Action |
|---------|---------|--------|
| **DEPLOY** | ECE improves >20% AND pips don't degrade >5% | ✅ Enable calibration in production |
| **ROLLBACK** | ECE increases OR pips degrade >5% | ❌ Disable calibration |
| **MARGINAL** | Modest improvement but doesn't meet thresholds | ⚠️ Optional - keep disabled for simplicity |

### Key Metrics

#### ECE (Expected Calibration Error)
- **Lower is better**
- Measures how well predicted probabilities match actual outcomes
- Target: >20% improvement (e.g., 0.05 → 0.04)

#### Total Pips
- **Higher is better**
- Measures actual trading profit
- Target: No degradation >5%

#### Profit Factor
- **Higher is better**
- Ratio of total profit to total loss
- Target: Maintain or improve (>1.5)

#### Win Rate
- **Higher is better**
- Percentage of winning trades
- Target: Maintain or improve (>55%)

## Test Output Examples

### ✅ DEPLOY Example

```
ECE improvement: 35.2%
Pips change: +8.3%
Decision: DEPLOY ✓
```

### ❌ ROLLBACK Example

```
ECE improvement: 5.1%
Pips change: -12.4%
Decision: ROLLBACK (pips degraded by 12.4%)
```

### ⚠️ MARGINAL Example

```
ECE improvement: 15.8%
Pips change: +2.1%
Decision: MARGINAL (ECE improvement < 20%)
```

## Troubleshooting

### Test Failures

#### "Data file not found"
```bash
# Download data if missing
python scripts/download_market_data.py --pair EURUSD --start 2020-01-01 --end 2025-12-31
```

#### "Trained models not found"
```bash
# Train models first
python scripts/train_mtf_ensemble.py --sentiment
```

#### "Insufficient calibration data"
- Ensure you have at least 100 bars in the calibration set
- Check that data file contains enough historical data

### Performance Issues

If tests are slow:
- Tests use real data and models (expected)
- Run decision test only for quick validation
- Use pytest markers to skip slow tests in CI

## Integration with Training

### Training with Calibration

```bash
# Train with calibration enabled
python scripts/train_mtf_ensemble.py --sentiment --calibration isotonic

# This will:
# 1. Train base models on train set
# 2. Fit calibrator on validation set
# 3. Save calibrator to models/mtf_ensemble/prediction_calibrator.pkl
```

### Backtesting with Calibration

```bash
# Backtest with calibration
python scripts/backtest_mtf_ensemble.py --calibration isotonic

# This will:
# 1. Load trained ensemble and calibrator
# 2. Apply calibration to all predictions
# 3. Show calibration analysis in results
```

## Next Steps After Testing

### If DEPLOY Recommended

1. ✅ Merge calibration feature
2. ✅ Update production config to enable calibration
3. ✅ Monitor calibration metrics in production
4. ✅ Update documentation

### If ROLLBACK Recommended

1. ❌ Keep calibration disabled by default
2. ❌ Investigate why calibration degrades performance
3. ❌ Try different calibration methods (platt, temperature)
4. ❌ Collect more calibration data

### If MARGINAL Result

1. ⚠️ Keep calibration disabled for simplicity
2. ⚠️ Consider enabling for specific use cases (high-confidence trades)
3. ⚠️ Re-evaluate with more data

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Calibration Validation

on:
  pull_request:
    paths:
      - 'backend/src/models/multi_timeframe/prediction_calibrator.py'
      - 'backend/tests/integration/test_calibration_production_validation.py'

jobs:
  validate-calibration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Install dependencies
        run: |
          pip install -r backend/requirements-api.txt
          pip install -r backend/requirements-test.txt
      - name: Run calibration validation
        run: |
          cd backend
          pytest tests/integration/test_calibration_production_validation.py::TestCalibrationDecisionCriteria::test_deployment_decision -v -s
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: calibration-results
          path: backend/test-results/
```

## References

- **Calibration Implementation**: `backend/src/models/multi_timeframe/prediction_calibrator.py`
- **Unit Tests**: `backend/tests/unit/models/test_prediction_calibrator.py`
- **MTF Ensemble**: `backend/src/models/multi_timeframe/mtf_ensemble.py`
- **Training Script**: `backend/scripts/train_mtf_ensemble.py`
- **Backtest Script**: `backend/scripts/backtest_mtf_ensemble.py`

## Contact

For questions about calibration validation:
- Review test output carefully
- Check logs for detailed metrics
- Compare before/after results
- Make data-driven decisions based on test criteria
