# LightGBM and CatBoost Implementation Summary

## Overview

Added support for LightGBM and CatBoost as alternative gradient boosting frameworks alongside the existing XGBoost implementation. Users can now choose which framework to use via the `--model-type` argument.

## Files Modified

### 1. Dependencies

**backend/requirements.txt**
- Added `lightgbm>=4.0.0`
- Added `catboost>=1.2.0`

**backend/requirements-api.txt**
- Added `lightgbm>=4.0.0`
- Added `catboost>=1.2.0`

### 2. Core Model Implementation

**backend/src/models/multi_timeframe/improved_model.py**

Changes:
- Added imports for `LGBMClassifier` and `CatBoostClassifier`
- Updated `model_type` field comment to include "lightgbm" and "catboost" options
- Added LightGBM model creation in `_create_model()`:
  - Parameters mapped to equivalent XGBoost hyperparameters
  - Uses `num_leaves=15` (2^max_depth - 1) for similar complexity
  - Configured with `verbosity=-1` for silent training
- Added CatBoost model creation in `_create_model()`:
  - Parameters mapped to equivalent XGBoost hyperparameters
  - Uses `iterations` instead of `n_estimators`
  - Uses `depth` instead of `max_depth`
  - Configured with `verbose=False` for silent training
- Updated `train()` method to handle framework-specific training:
  - XGBoost: `eval_set=[(X_val, y_val)]`
  - LightGBM: `eval_set=[(X_val, y_val)]`
  - CatBoost: `eval_set=(X_val, y_val)` (note: tuple not list)

### 3. Ensemble Configuration

**backend/src/models/multi_timeframe/mtf_ensemble.py**

Changes:
- Added `model_type` field to `MTFEnsembleConfig` (default: "xgboost")
- Added logic to propagate `model_type` to all individual timeframe model configs
- Ensures all 3 models (1H, 4H, Daily) use the same framework

### 4. Training Script

**backend/scripts/train_mtf_ensemble.py**

Changes:
- Added `--model-type` argument with choices: ["xgboost", "lightgbm", "catboost"]
- Default: "xgboost" (maintains backward compatibility)
- Passes `model_type` to `MTFEnsembleConfig`
- Displays selected framework in training header

### 5. New Comparison Script

**backend/scripts/compare_gradient_boosting.py**

New standalone script that:
- Trains MTF Ensemble with each framework (XGBoost, LightGBM, CatBoost)
- Runs backtests on test data (last 20% of dataset)
- Compares results in a formatted table
- Saves comparison results to `data/gradient_boosting_comparison.json`

Features:
- Supports custom confidence thresholds (default: 55%)
- Uses optimal config: stacking meta-learner + EPU sentiment on Daily model
- Reports: total pips, win rate, profit factor, total trades, training time
- Identifies the best-performing framework
- Shows improvement vs baseline (7,987 pips from XGBoost)

## Usage

### Training with Different Frameworks

```bash
cd backend

# Train with XGBoost (default - current production)
python scripts/train_mtf_ensemble.py --sentiment --stacking

# Train with LightGBM
python scripts/train_mtf_ensemble.py --sentiment --stacking --model-type lightgbm

# Train with CatBoost
python scripts/train_mtf_ensemble.py --sentiment --stacking --model-type catboost
```

### Running Framework Comparison

```bash
cd backend

# Compare all three frameworks
python scripts/compare_gradient_boosting.py

# Custom confidence threshold
python scripts/compare_gradient_boosting.py --confidence 0.70

# Compare specific frameworks only
python scripts/compare_gradient_boosting.py --frameworks "xgboost,lightgbm"
```

## Parameter Mapping

### LightGBM Equivalents

| XGBoost | LightGBM | Notes |
|---------|----------|-------|
| `n_estimators` | `n_estimators` | Direct mapping |
| `max_depth` | `max_depth` | Direct mapping |
| `learning_rate` | `learning_rate` | Direct mapping |
| `min_child_weight` | `min_child_samples` | Set to 20 |
| `subsample` | `subsample` | Direct mapping |
| `colsample_bytree` | `colsample_bytree` | Direct mapping |
| `reg_alpha` | `reg_alpha` | Direct mapping |
| `reg_lambda` | `reg_lambda` | Direct mapping |
| N/A | `num_leaves` | Set to 15 (2^4-1) |

### CatBoost Equivalents

| XGBoost | CatBoost | Notes |
|---------|----------|-------|
| `n_estimators` | `iterations` | Renamed parameter |
| `max_depth` | `depth` | Renamed parameter |
| `learning_rate` | `learning_rate` | Direct mapping |
| `reg_lambda` | `l2_leaf_reg` | Direct mapping |

## Expected Performance

Based on gradient boosting framework benchmarks:

- **XGBoost**: Current baseline (+7,987 pips at 55% confidence)
- **LightGBM**: Expected to be faster training, similar or slightly better accuracy
- **CatBoost**: Expected to handle categorical features better (not used here), slower training

All frameworks use equivalent hyperparameters for fair comparison.

## Notes

- All frameworks default to shallow trees (max_depth=4) per the optimized "shallow_fast" config
- Training time differences will be reported in the comparison script
- Production models remain unchanged (`models/mtf_ensemble/`) - new models saved to `models/gradient_boosting/<framework>/`
- The comparison script uses simplified backtesting (20 pips profit, -10 pips loss per trade)

## Verification

All files compile successfully:
```bash
python3 -m py_compile src/models/multi_timeframe/improved_model.py  # ✓
python3 -m py_compile src/models/multi_timeframe/mtf_ensemble.py    # ✓
python3 -m py_compile scripts/train_mtf_ensemble.py                 # ✓
python3 -m py_compile scripts/compare_gradient_boosting.py          # ✓
```

## Next Steps

1. Install dependencies: `pip install lightgbm>=4.0.0 catboost>=1.2.0`
2. Run comparison: `python scripts/compare_gradient_boosting.py`
3. Review results in `data/gradient_boosting_comparison.json`
4. If a framework outperforms XGBoost, retrain production models with that framework
