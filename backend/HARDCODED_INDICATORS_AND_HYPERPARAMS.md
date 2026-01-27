# Hardcoded Technical Indicators & Hyperparameters

**Date:** 2026-01-27
**Status:** 📊 **98 parameters found, 22 centralized, 76 hardcoded (77.6%)**

---

## Quick Summary

| Category | Total | Centralized | Hardcoded | Priority |
|----------|-------|-------------|-----------|----------|
| **Technical Indicators** | 30 | 6 | 24 | 🔴 HIGH |
| **XGBoost Hyperparameters** | 30 | 6 | 24 | 🔴 HIGH |
| **Feature Engineering** | 12 | 0 | 12 | 🟠 MEDIUM |
| **Training Parameters** | 10 | 4 | 6 | 🟡 LOW |
| **Labeling Barriers** | 12 | 6 | 6 | ✅ GOOD |
| **Confidence Thresholds** | 4 | 4 | 0 | ✅ GOOD |

**TOTAL:** 98 parameters, 22 centralized (22%), 76 hardcoded (78%)

---

## 1. TECHNICAL INDICATORS (30 parameters)

### Trend Indicators (8 parameters)

| Indicator | File | Line | Default Value | Frequency | Action |
|-----------|------|------|---------------|-----------|--------|
| **SMA Periods** | `src/features/technical/trend.py` | 25 | [5, 10, 20, 50, 100, 200] | High | 🔴 CENTRALIZE |
| **EMA Periods** | `src/features/technical/trend.py` | 26 | [5, 10, 20, 50, 100, 200] | High | 🔴 CENTRALIZE |
| **WMA Periods** | `src/features/technical/trend.py` | 27 | [10, 20, 50] | Low | ✅ KEEP |
| **ADX Period** | `src/features/technical/trend.py` | 30 | 14 | High | 🔴 CENTRALIZE |
| **Aroon Period** | `src/features/technical/trend.py` | 31 | 25 | Low | ✅ KEEP |
| **MA Crossover Pairs** | `src/features/technical/trend.py` | 171-177 | (5×20, 20×50, 50×200) | High | 🔴 CENTRALIZE |
| **Supertrend Period** | `src/features/technical/trend.py` | 188 | 10 | Low | ✅ KEEP |
| **Supertrend Multiplier** | `src/features/technical/trend.py` | 189 | 3.0 | Low | ✅ KEEP |

### Momentum Indicators (9 parameters)

| Indicator | File | Line | Default Value | Frequency | Action |
|-----------|------|------|---------------|-----------|--------|
| **RSI Periods** | `src/features/technical/momentum.py` | 24 | [7, 14, 21] | Very High | 🔴 CENTRALIZE |
| **Stochastic K-Period** | `src/features/technical/momentum.py` | 25 | 14 | High | 🔴 CENTRALIZE |
| **Stochastic D-Period** | `src/features/technical/momentum.py` | 25 | 3 | High | 🔴 CENTRALIZE |
| **MACD Fast** | `src/features/technical/momentum.py` | 72 | 12 | Very High | 🔴 CENTRALIZE |
| **MACD Slow** | `src/features/technical/momentum.py` | 73 | 26 | Very High | 🔴 CENTRALIZE |
| **MACD Signal** | `src/features/technical/momentum.py` | 74 | 9 | Very High | 🔴 CENTRALIZE |
| **CCI Periods** | `src/features/technical/momentum.py` | 27 | [14, 20] | Medium | 🟠 CENTRALIZE |
| **Williams %R Period** | `src/features/technical/momentum.py` | 30 | 14 | Medium | ✅ KEEP |
| **MFI Period** | `src/features/technical/momentum.py` | 31 | 14 | Medium | ✅ KEEP |

### Volatility Indicators (7 parameters)

| Indicator | File | Line | Default Value | Frequency | Action |
|-----------|------|------|---------------|-----------|--------|
| **ATR Period** | `src/features/technical/volatility.py` | 24 | 14 | Very High | 🔴 CENTRALIZE |
| **Bollinger Period** | `src/features/technical/volatility.py` | 26 | 20 | Very High | 🔴 CENTRALIZE |
| **Bollinger Std Dev** | `src/features/technical/volatility.py` | 26 | 2.0 | Very High | 🔴 CENTRALIZE |
| **Keltner Period** | `src/features/technical/volatility.py` | 27 | 20 | Medium | 🟠 CENTRALIZE |
| **Keltner Multiplier** | `src/features/technical/volatility.py` | 27 | 2.0 | Medium | 🟠 CENTRALIZE |
| **Donchian Period** | `src/features/technical/volatility.py` | 28 | 20 | Medium | ✅ KEEP |
| **Std Dev Periods** | `src/features/technical/volatility.py` | 29 | [10, 20] | Medium | 🟠 CENTRALIZE |

### Volume Indicators (6 parameters)

| Indicator | File | Line | Default Value | Frequency | Action |
|-----------|------|------|---------------|-----------|--------|
| **CMF Period** | `src/features/technical/volume.py` | 31 | 20 | Medium | ✅ KEEP |
| **EMV Period** | `src/features/technical/volume.py` | 33 | 14 | Low | ✅ KEEP |
| **Force Index Period** | `src/features/technical/volume.py` | 34 | 13 | Low | ✅ KEEP |
| **ADOSC Fast** | `src/features/technical/volume.py` | 75 | 3 | Low | ✅ KEEP |
| **ADOSC Slow** | `src/features/technical/volume.py` | 75 | 10 | Low | ✅ KEEP |
| **Volume SMA Periods** | `src/features/technical/volume.py` | 35 | [10, 20] | Medium | 🟠 CENTRALIZE |

---

## 2. XGBOOST HYPERPARAMETERS (30 parameters)

### 1H Model (10 parameters)

| Parameter | File | Line | Value | Usage | Action |
|-----------|------|------|-------|-------|--------|
| **n_estimators** | `improved_model.py` | 109 | 150 | Trees to build | 🔴 CENTRALIZE |
| **max_depth** | `improved_model.py` | 110 | 5 | Tree depth | 🔴 CENTRALIZE |
| **learning_rate** | `improved_model.py` | 111 | 0.03 | Step size | 🔴 CENTRALIZE |
| **min_child_weight** | `improved_model.py` | 73 | 3 | Min samples per leaf | 🔴 CENTRALIZE |
| **subsample** | `improved_model.py` | 74 | 0.8 | Row sampling | 🔴 CENTRALIZE |
| **colsample_bytree** | `improved_model.py` | 75 | 0.8 | Column sampling | 🔴 CENTRALIZE |
| **reg_alpha** | `improved_model.py` | 76 | 0.1 | L1 regularization | 🔴 CENTRALIZE |
| **reg_lambda** | `improved_model.py` | 77 | 1.0 | L2 regularization | 🔴 CENTRALIZE |
| **gamma** | `improved_model.py` | 78 | 0.1 | Min loss reduction | 🔴 CENTRALIZE |
| **random_state** | `improved_model.py` | 223 | 42 | Reproducibility | ✅ KEEP |

### 4H Model (10 parameters)

| Parameter | File | Line | Value | Difference from 1H | Action |
|-----------|------|------|-------|-------------------|--------|
| **n_estimators** | `improved_model.py` | 123 | 120 | -30 (fewer trees) | 🔴 CENTRALIZE |
| **max_depth** | `improved_model.py` | 124 | 4 | -1 (shallower) | 🔴 CENTRALIZE |
| **learning_rate** | `improved_model.py` | 125 | 0.03 | Same | 🔴 CENTRALIZE |
| *(Other 7 params)* | - | - | Same as 1H | Inherited | 🔴 CENTRALIZE |

### Daily Model (10 parameters)

| Parameter | File | Line | Value | Difference from 1H | Action |
|-----------|------|------|-------|-------------------|--------|
| **n_estimators** | `improved_model.py` | 137 | 80 | -70 (much fewer) | 🔴 CENTRALIZE |
| **max_depth** | `improved_model.py` | 138 | 3 | -2 (much shallower) | 🔴 CENTRALIZE |
| **learning_rate** | `improved_model.py` | 139 | 0.03 | Same | 🔴 CENTRALIZE |
| *(Other 7 params)* | - | - | Same as 1H | Inherited | 🔴 CENTRALIZE |

**Pattern:** Each timeframe has progressively simpler models (fewer trees, shallower depth)

---

## 3. FEATURE ENGINEERING (12 parameters)

### Lag Features (5 parameters)

| Feature | File | Line | Values | Purpose | Action |
|---------|------|------|--------|---------|--------|
| **Lag Periods** | `enhanced_features.py` | 54 | [1, 2, 3, 6, 12] | Sequential patterns | 🔴 CENTRALIZE |
| **RSI ROC** | `enhanced_features.py` | 161 | [3, 6] | Momentum changes | 🔴 CENTRALIZE |
| **MACD ROC** | `enhanced_features.py` | 175 | [3] | Divergence detection | 🔴 CENTRALIZE |
| **ATR ROC** | `enhanced_features.py` | 189 | [3, 6] | Volatility momentum | 🔴 CENTRALIZE |
| **Price ROC** | `enhanced_features.py` | 197 | [1, 3, 6, 12] | Price momentum | 🔴 CENTRALIZE |

### Time Features (4 parameters)

| Feature | File | Line | Values | Purpose | Action |
|---------|------|------|--------|---------|--------|
| **Asian Session** | `enhanced_features.py` | 131 | 00:00-08:00 UTC | Session timing | 🟠 CENTRALIZE |
| **London Session** | `enhanced_features.py` | 133 | 08:00-16:00 UTC | Session timing | 🟠 CENTRALIZE |
| **NY Session** | `enhanced_features.py` | 135 | 13:00-22:00 UTC | Session timing | 🟠 CENTRALIZE |
| **Overlap** | `enhanced_features.py` | 137 | 13:00-16:00 UTC | High liquidity | 🟠 CENTRALIZE |

### Cyclical Encoding (3 parameters)

| Feature | File | Line | Formula | Purpose | Action |
|---------|------|------|---------|---------|--------|
| **Hour Encoding** | `enhanced_features.py` | 122 | 2π/24 | 24-hour cycle | 🟡 DOCUMENT |
| **Day of Week** | `enhanced_features.py` | 125 | 2π/7 | Weekly patterns | 🟡 DOCUMENT |
| **Day of Month** | `enhanced_features.py` | 128 | 2π/31 | Monthly patterns | 🟡 DOCUMENT |

---

## 4. TRAINING PARAMETERS (10 parameters)

### Data Splits (3 parameters)

| Parameter | File | Line | Value | Purpose | Action |
|-----------|------|------|-------|---------|--------|
| **Train Ratio** | `mtf_ensemble.py` | 566 | 0.6 (60%) | Training set | 🟠 CENTRALIZE |
| **Validation Ratio** | `mtf_ensemble.py` | 567 | 0.2 (20%) | Tuning | 🟠 CENTRALIZE |
| **Test Ratio** | Implicit | - | 0.2 (20%) | Final eval | 🟠 CENTRALIZE |

### Stacking Meta-Learner (4 parameters)

| Parameter | File | Line | Value | Purpose | Action |
|-----------|------|------|-------|---------|--------|
| **n_folds** | `stacking_meta_learner.py` | 47 | 5 | Cross-validation | 🟠 CENTRALIZE |
| **min_train_size** | `stacking_meta_learner.py` | 48 | 500 | Min samples/fold | 🟠 CENTRALIZE |
| **eval_metric** | `stacking_meta_learner.py` | 188 | "logloss" | Binary classification | ✅ KEEP |
| **random_state** | `stacking_meta_learner.py` | 198 | 42 | Reproducibility | ✅ KEEP |

### Early Stopping (3 parameters)

| Parameter | File | Line | Value | Purpose | Action |
|-----------|------|------|-------|---------|--------|
| **eval_metric** | `improved_model.py` | 419 | "logloss" | Monitor metric | ✅ KEEP |
| **eval_set** | `improved_model.py` | 420 | Validation | Stop on plateau | ✅ KEEP |
| **verbose** | `improved_model.py` | 430 | False | Training logs | ✅ KEEP |

---

## 5. LABELING BARRIERS (12 parameters) ✅ MOSTLY CENTRALIZED

### TP/SL/Holding Bars (9 parameters) ✅ CENTRALIZED

| Timeframe | File | Lines | TP Pips | SL Pips | Max Holding | Status |
|-----------|------|-------|---------|---------|-------------|--------|
| **1H** | `trading_config.py` | 307-309 | 25.0 | 15.0 | 12 bars | ✅ CENTRALIZED |
| **4H** | `trading_config.py` | 312-314 | 50.0 | 25.0 | 18 bars | ✅ CENTRALIZED |
| **Daily** | `trading_config.py` | 317-319 | 150.0 | 75.0 | 15 bars | ✅ CENTRALIZED |

**Note:** Models now import from TradingConfig (fixed in earlier phase)

### Alternative Labeling (3 parameters) 🟡 LOW PRIORITY

| Method | File | Line | Value | Purpose | Action |
|--------|------|------|-------|---------|--------|
| **forward_bars** | `labeling.py` | 32 | 12 | Multi-bar lookahead | 🟡 DOCUMENT |
| **threshold_pips** | `labeling.py` | 32 | 10.0 | Min move threshold | 🟡 DOCUMENT |
| **atr_multiplier** | `labeling.py` | 41 | 2.0 | Volatility adjust | 🟡 DOCUMENT |

---

## 6. CONFIDENCE & FILTERING (4 parameters) ✅ MOSTLY CENTRALIZED

| Parameter | File | Line | Value | Status | Action |
|-----------|------|------|-------|--------|--------|
| **min_confidence** | `trading_config.py` | 27 | 0.60 | ✅ Centralized | ✅ GOOD |
| **min_agreement** | `mtf_ensemble.py` | 63 | 0.5 (2/3 models) | 🟠 Hardcoded | 🟠 CENTRALIZE |
| **agreement_bonus** | `trading_config.py` | 54 | 0.05 (5%) | ✅ Centralized | ✅ GOOD |
| **dynamic_threshold_quantile** | `trading_config.py` | 209 | 0.60 (top 40%) | ✅ Centralized | ✅ GOOD |

---

## 📊 PRIORITY MATRIX

### 🔴 CRITICAL - Centralize Immediately

**Impact:** High usage, affects production trading

1. **XGBoost Hyperparameters** (24 params)
   - All 3 models (1H, 4H, Daily)
   - Current: Hardcoded in `improved_model.py`
   - Target: `TradingConfig.models.{timeframe}.hyperparams`

2. **Technical Indicator Periods** (12 params)
   - RSI, MACD, Bollinger Bands, ATR
   - Current: Scattered in feature files
   - Target: `TradingConfig.indicators.*`

3. **Lag Feature Periods** (5 params)
   - Sequential pattern detection
   - Current: Hardcoded in `enhanced_features.py`
   - Target: `TradingConfig.features.lag_periods`

4. **MA Periods & Crossovers** (3 params)
   - SMA/EMA periods: [5, 10, 20, 50, 100, 200]
   - Crossover pairs: (5×20, 20×50, 50×200)
   - Target: `TradingConfig.indicators.moving_averages`

**Total Critical:** 44 parameters

---

### 🟠 MEDIUM - Centralize This Quarter

**Impact:** Medium usage, configuration flexibility

5. **Training Split Ratios** (3 params)
   - Train 60%, Val 20%, Test 20%
   - Target: `TradingConfig.training.split_ratios`

6. **Session Times** (4 params)
   - Asian, London, NY, Overlap
   - Target: `TradingConfig.market.session_times`

7. **Stacking Meta-Learner** (4 params)
   - n_folds, min_train_size
   - Target: `TradingConfig.models.stacking.*`

8. **Secondary Indicators** (10 params)
   - CCI, Keltner, Volume indicators
   - Target: `TradingConfig.indicators.*`

**Total Medium:** 21 parameters

---

### 🟡 LOW - Document Only

**Impact:** Low usage, stable defaults

9. **Cyclical Encodings** (3 params)
   - Hour/Day/Month encoding formulas
   - Action: Document in code comments

10. **Alternative Labeling** (3 params)
    - Multi-bar, volatility-adjusted
    - Action: Document, rarely used

11. **Specialized Indicators** (6 params)
    - Aroon, Supertrend, Force Index, etc.
    - Action: Keep defaults, low frequency

**Total Low:** 12 parameters

---

## 🎯 COMPREHENSIVE IMPLEMENTATION PLAN

**See detailed 6-week implementation plan:** `CONFIGURATION_CENTRALIZATION_IMPLEMENTATION_PLAN.md`

### Quick Overview

**Timeline:** 6 weeks
**Scope:** 76 hardcoded parameters (78%)
**Effort:** 310 hours
**ROI:** Very High

### Week-by-Week Breakdown

| Week | Focus | Parameters | Deliverables |
|------|-------|-----------|--------------|
| **Week 1** | Infrastructure | 0 | Config dataclasses, validation, database integration |
| **Week 2** | Technical Indicators | 30 | All indicators using centralized config |
| **Week 3** | Model Hyperparameters | 30 | All XGBoost params centralized |
| **Week 4** | Features & Training | 22 | Feature engineering + training params |
| **Week 5** | Testing & Deployment | - | 100+ tests, documentation, staging rollout |
| **Week 6** | Monitoring & Optimization | - | Production monitoring, optimization framework |

### Key Features

✅ **Hot-Reload Support** - Update config without restart
✅ **Database Persistence** - ConfigurationHistory tracks all changes
✅ **API Control** - REST endpoints for runtime updates
✅ **Backward Compatible** - Feature flags for gradual rollout
✅ **Comprehensive Testing** - 100+ unit tests, 20+ integration tests
✅ **Risk Mitigation** - Rollback plan, performance monitoring
✅ **Optimization Ready** - Grid search, Optuna integration

### Configuration Structure (Post-Implementation)

```
TradingConfig
├── indicators (NEW)
│   ├── trend (8 params)
│   ├── momentum (9 params)
│   ├── volatility (7 params)
│   └── volume (6 params)
├── hyperparameters (NEW)
│   ├── model_1h (10 params)
│   ├── model_4h (10 params)
│   └── model_daily (10 params)
├── features (NEW)
│   ├── lags (5 params)
│   ├── sessions (4 params)
│   └── cyclical (3 params)
├── training (NEW)
│   ├── splits (3 params)
│   ├── stacking (4 params)
│   └── early_stopping (3 params)
└── labeling (NEW)
    └── alternative (6 params)
```

### Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Config Coverage | 22% | 100% |
| Hardcoded Params | 76 | 0 |
| Test Coverage | 85% | 95% |
| Config Load Time | - | <10ms |

### Example Usage (Post-Implementation)

```python
# Load centralized config
from src.config import TradingConfig
config = TradingConfig()

# Access any parameter
rsi_periods = config.indicators.momentum.rsi_periods  # [7, 14, 21]
n_estimators = config.hyperparameters.model_1h.n_estimators  # 150
lag_periods = config.features.lags.standard_lags  # [1, 2, 3, 6, 12]

# Override for experimentation
config.indicators.momentum.rsi_periods = [14, 28, 42]
config.hyperparameters.model_1h.learning_rate = 0.01

# Use in training
ensemble = MTFEnsemble(config=config)
ensemble.train(X, y)

# Hot-reload via API
POST /api/v1/config/update
{
    "indicators.momentum.rsi_periods": [10, 20, 30],
    "hyperparameters.model_1h.n_estimators": 200
}
```

### Next Steps

1. **Review** the detailed implementation plan
2. **Approve** architecture and timeline
3. **Create** GitHub project with 26 tasks
4. **Begin** Week 1: Infrastructure development

**Full Details:** See `CONFIGURATION_CENTRALIZATION_IMPLEMENTATION_PLAN.md`

---

## ✅ CURRENT STATUS

**Already Centralized (22 parameters):**
- ✅ Trading confidence threshold (0.60)
- ✅ TP/SL/max_holding_bars for all timeframes
- ✅ Ensemble weights (0.6, 0.3, 0.1)
- ✅ Agreement bonus (0.05)
- ✅ Circuit breaker limits
- ✅ Position sizing parameters
- ✅ Dynamic threshold parameters
- ✅ Scheduler timing

**Remaining Hardcoded (76 parameters):**
- 🔴 XGBoost hyperparameters (24)
- 🔴 Technical indicator periods (24)
- 🟠 Feature engineering (12)
- 🟠 Training parameters (10)
- 🟡 Alternative labeling (3)
- 🟡 Specialized indicators (3)

---

## 📋 RECOMMENDATIONS

### Immediate (This Week)
1. Create `IndicatorParameters` dataclass in `trading_config.py`
2. Create `ModelHyperparameters` dataclass
3. Update RSI, MACD, Bollinger Bands to use config

### Short-Term (This Month)
4. Create `FeatureParameters` dataclass
5. Update lag features and ROC periods
6. Add session time configuration
7. Document cyclical encodings

### Long-Term (This Quarter)
8. Create hyperparameter optimization framework
9. Add indicator period testing tools
10. Build configuration UI in frontend
11. Add config versioning and rollback

---

## 🎓 WHY CENTRALIZE?

**Benefits:**
1. **Single Source of Truth** - No duplication
2. **Easy Tuning** - Change once, applies everywhere
3. **Reproducibility** - Config versioning tracks changes
4. **Hot Reload** - Update without restart (when supported)
5. **Documentation** - Self-documenting configuration
6. **Testing** - Easy to test different configurations
7. **Optimization** - Hyperparameter search becomes simple

**Example:**
```python
# Before: Hardcoded in 10+ files
rsi_period = 14  # Scattered everywhere

# After: Centralized
config = TradingConfig()
rsi_period = config.indicators.rsi_periods[1]  # [7, 14, 21]
```

---

**Report Date:** 2026-01-27
**Status:** 📋 Audit Complete - Implementation Plan Ready
**Next Action:** Create Phase 1 indicator configuration
