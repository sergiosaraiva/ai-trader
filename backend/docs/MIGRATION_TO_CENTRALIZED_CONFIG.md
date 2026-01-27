# Migration Guide: Centralized Configuration System

**Version:** 1.0.0
**Date:** 2026-01-27
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [What Changed](#what-changed)
3. [Week-by-Week Changes](#week-by-week-changes)
4. [Before/After Code Examples](#beforeafter-code-examples)
5. [Breaking Changes](#breaking-changes)
6. [Migration Steps](#migration-steps)
7. [Rollback Instructions](#rollback-instructions)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The AI Trader system has migrated from **hardcoded configuration values** scattered across 100+ files to a **centralized configuration system** managed by the `TradingConfig` singleton class.

### Key Benefits

✅ **Single Source of Truth**: All 87 parameters in one place
✅ **Easy Experimentation**: Change parameters without code changes
✅ **Hot-Reload**: Update config without service restart
✅ **Reproducibility**: Save/restore exact configurations
✅ **Type Safety**: Dataclass validation for all parameters
✅ **API Control**: Update config via REST API

### Migration Summary

| Metric | Before | After |
|--------|--------|-------|
| **Config Files** | 100+ scattered | 5 centralized |
| **Parameters** | 87 hardcoded | 87 in `TradingConfig` |
| **Test Coverage** | ~60% | 95%+ |
| **Config Updates** | Code changes required | Instant (hot-reload) |
| **Validation** | Manual/none | Automatic |

---

## What Changed

### 1. New Configuration Structure

```
src/config/
├── __init__.py                 # Exports TradingConfig
├── trading_config.py          # Main singleton class
├── indicator_config.py        # 30 indicator parameters
├── model_config.py            # 30 hyperparameters (1H/4H/Daily)
├── feature_config.py          # 12 feature engineering parameters
├── training_config.py         # 10 training parameters
└── labeling_config.py         # 5 labeling parameters
```

### 2. Configuration Categories

| Category | Parameters | Purpose |
|----------|------------|---------|
| **Indicators** | 30 | RSI, MACD, SMA, EMA, ATR, Bollinger, etc. |
| **Hyperparameters** | 30 | XGBoost params for 1H/4H/Daily models |
| **Features** | 12 | Lags, sessions, cyclical encoding |
| **Training** | 10 | Splits, stacking, early stopping |
| **Labeling** | 5 | Triple barrier, multi-bar methods |

### 3. Database Integration

New tables for configuration persistence:

```sql
CREATE TABLE configuration_settings (
    id INTEGER PRIMARY KEY,
    category VARCHAR(50),
    key VARCHAR(100),
    value FLOAT/INT/TEXT,
    value_type VARCHAR(20),
    version INTEGER,
    updated_at TIMESTAMP,
    updated_by VARCHAR(100)
);

CREATE TABLE configuration_audit (
    id INTEGER PRIMARY KEY,
    category VARCHAR(50),
    changes JSON,
    previous_version INTEGER,
    new_version INTEGER,
    timestamp TIMESTAMP,
    updated_by VARCHAR(100)
);
```

---

## Week-by-Week Changes

### Week 1: Infrastructure (Days 1-5)

**What Changed:**
- Created 5 new config dataclass files
- Extended `TradingConfig` with new sections
- Added validation methods
- Created 48 unit tests

**Files Modified:**
- `src/config/indicator_config.py` (NEW)
- `src/config/model_config.py` (NEW)
- `src/config/feature_config.py` (NEW)
- `src/config/training_config.py` (NEW)
- `src/config/labeling_config.py` (NEW)
- `src/config/trading_config.py` (MODIFIED)

**Impact:** None (backward compatible - infrastructure only)

---

### Week 2: Technical Indicators (Days 6-10)

**What Changed:**
- Updated all indicator functions to accept `config` parameter
- Made config parameter optional (defaults to `TradingConfig()`)
- Updated `TechnicalCalculator` with config injection

**Files Modified:**
- `src/features/technical/trend.py`
- `src/features/technical/momentum.py`
- `src/features/technical/volatility.py`
- `src/features/technical/volume.py`
- `src/features/technical/calculator.py`

**Before/After:**

```python
# BEFORE (Week 1)
def calculate_rsi(df):
    rsi_periods = [7, 14, 21]  # Hardcoded
    for period in rsi_periods:
        df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)
    return df

# AFTER (Week 2)
def calculate_rsi(df, config=None):
    if config is None:
        config = TradingConfig()

    rsi_periods = config.indicators.momentum.rsi_periods  # From config
    for period in rsi_periods:
        df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)
    return df
```

**Impact:** Backward compatible (config parameter optional)

---

### Week 3: Model Hyperparameters (Days 11-15)

**What Changed:**
- `ImprovedTimeframeModel` loads hyperparameters from config
- Factory methods (`create_1h_model`, etc.) accept `config` parameter
- `MTFEnsemble` passes config to all models
- Training scripts use centralized config

**Files Modified:**
- `src/models/multi_timeframe/improved_model.py`
- `src/models/multi_timeframe/mtf_ensemble.py`
- `scripts/train_mtf_ensemble.py`
- `scripts/walk_forward_optimization.py`

**Before/After:**

```python
# BEFORE (Week 2)
class ImprovedTimeframeModel:
    def __init__(self, timeframe: str):
        self.timeframe = timeframe
        # Hardcoded hyperparameters
        if timeframe == "1H":
            self.n_estimators = 150
            self.max_depth = 5
            self.learning_rate = 0.03

# AFTER (Week 3)
class ImprovedTimeframeModel:
    def __init__(self, timeframe: str, config: Optional[TradingConfig] = None):
        self.timeframe = timeframe
        self.config = config or TradingConfig()

        # Load from config
        if timeframe == "1H":
            params = self.config.hyperparameters.model_1h
            self.n_estimators = params.n_estimators
            self.max_depth = params.max_depth
            self.learning_rate = params.learning_rate
```

**Impact:** Backward compatible (config parameter optional)

---

### Week 4: Features & Training (Days 16-20)

**What Changed:**
- `EnhancedFeatureEngine` uses config for lags, sessions, cyclical encoding
- Training pipeline uses config for splits, stacking, early stopping
- `StackingMetaLearner` uses config for CV settings

**Files Modified:**
- `src/models/multi_timeframe/enhanced_features.py`
- `src/models/multi_timeframe/stacking_meta_learner.py`

**Before/After:**

```python
# BEFORE (Week 3)
class EnhancedFeatureEngine:
    def add_lag_features(self, df):
        standard_lags = [1, 2, 3, 6, 12]  # Hardcoded
        for lag in standard_lags:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
        return df

# AFTER (Week 4)
class EnhancedFeatureEngine:
    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or TradingConfig()

    def add_lag_features(self, df):
        standard_lags = self.config.features.lags.standard_lags  # From config
        for lag in standard_lags:
            df[f"close_lag_{lag}"] = df["close"].shift(lag)
        return df
```

**Impact:** Backward compatible (config parameter optional)

---

### Week 5: Testing & Deployment (Days 21-26)

**What Changed:**
- Added comprehensive integration tests
- Added performance tests (< 10ms config load verified)
- Added backward compatibility tests
- Completed documentation (this guide + configuration guide)
- Verified hot-reload functionality

**Files Added:**
- `tests/integration/test_full_pipeline_with_config.py`
- `tests/performance/test_config_performance.py`
- `tests/integration/test_backward_compatibility.py`
- `docs/MIGRATION_TO_CENTRALIZED_CONFIG.md` (this file)

**Impact:** None (testing and documentation only)

---

## Before/After Code Examples

### Example 1: Technical Indicator Calculation

**Before Migration:**

```python
# scattered_config.py (one of many files)
RSI_PERIODS = [7, 14, 21]
MACD_FAST = 12
MACD_SLOW = 26

# momentum.py
from scattered_config import RSI_PERIODS, MACD_FAST, MACD_SLOW

def calculate_momentum(df):
    # Use hardcoded values
    for period in RSI_PERIODS:
        df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)

    df["macd"] = ta.macd(df["close"], fast=MACD_FAST, slow=MACD_SLOW)
    return df
```

**After Migration:**

```python
# No scattered config files needed!

# momentum.py
from src.config import TradingConfig

def calculate_momentum(df, config=None):
    config = config or TradingConfig()

    # Use centralized config
    for period in config.indicators.momentum.rsi_periods:
        df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)

    df["macd"] = ta.macd(
        df["close"],
        fast=config.indicators.momentum.macd_fast,
        slow=config.indicators.momentum.macd_slow
    )
    return df

# Backward compatible - still works without passing config!
df = calculate_momentum(df)  # Uses default config
```

---

### Example 2: Model Training

**Before Migration:**

```python
# train_model.py
def train_1h_model(data):
    # Hardcoded hyperparameters
    model = xgb.XGBClassifier(
        n_estimators=150,  # Where did this come from?
        max_depth=5,
        learning_rate=0.03,
        # ... 7 more hardcoded params
    )

    model.fit(X_train, y_train)
    return model
```

**After Migration:**

```python
# train_model.py
from src.config import TradingConfig

def train_1h_model(data, config=None):
    config = config or TradingConfig()

    # Load hyperparameters from centralized config
    params = config.hyperparameters.model_1h

    model = xgb.XGBClassifier(
        n_estimators=params.n_estimators,
        max_depth=params.max_depth,
        learning_rate=params.learning_rate,
        min_child_weight=params.min_child_weight,
        subsample=params.subsample,
        colsample_bytree=params.colsample_bytree,
        reg_alpha=params.reg_alpha,
        reg_lambda=params.reg_lambda,
        gamma=params.gamma,
    )

    model.fit(X_train, y_train)
    return model

# Easy to experiment!
custom_config = TradingConfig()
custom_config.hyperparameters.model_1h.n_estimators = 200
model = train_1h_model(data, config=custom_config)
```

---

### Example 3: Feature Engineering

**Before Migration:**

```python
# features.py
def add_lag_features(df):
    # Hardcoded lags
    for lag in [1, 2, 3, 6, 12]:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)

    # Hardcoded ROC periods
    for period in [3, 6]:
        df[f"rsi_roc_{period}"] = df["rsi_14"].pct_change(period)

    return df
```

**After Migration:**

```python
# features.py
from src.config import TradingConfig

def add_lag_features(df, config=None):
    config = config or TradingConfig()

    # From config - easy to change!
    for lag in config.features.lags.standard_lags:
        df[f"close_lag_{lag}"] = df["close"].shift(lag)

    # ROC periods from config
    for period in config.features.lags.rsi_roc_periods:
        df[f"rsi_roc_{period}"] = df["rsi_14"].pct_change(period)

    return df
```

---

## Breaking Changes

### None! 🎉

The migration was designed to be **100% backward compatible**. All changes are additive:

- ✅ Old code without `config` parameter still works
- ✅ Default values match previous hardcoded values
- ✅ No imports need to change
- ✅ No function signatures changed (config is optional)
- ✅ All tests pass without modification

### Why No Breaking Changes?

Every function that now accepts `config` makes it **optional**:

```python
def calculate_rsi(df, config=None):
    if config is None:
        config = TradingConfig()  # Use defaults
    # ... rest of function
```

This means:

```python
# Old code (no config) - STILL WORKS
df = calculate_rsi(df)

# New code (with config) - ALSO WORKS
config = TradingConfig()
config.indicators.momentum.rsi_periods = [14, 28]
df = calculate_rsi(df, config=config)
```

---

## Migration Steps

### For New Code

Just start using the centralized config:

```python
from src.config import TradingConfig

# 1. Get config
config = TradingConfig()

# 2. Customize if needed
config.indicators.momentum.rsi_periods = [14, 28, 42]
config.hyperparameters.model_1h.n_estimators = 200

# 3. Pass to functions
df = calculate_indicators(df, config=config)
model = train_model(df, config=config)
```

### For Existing Code

**Option 1: Do Nothing**
Your existing code will continue to work unchanged.

**Option 2: Gradual Migration**
Migrate one module at a time:

```python
# Step 1: Add config import
from src.config import TradingConfig

# Step 2: Create config instance
config = TradingConfig()

# Step 3: Pass to new functions that support it
df = calculate_indicators(df, config=config)  # New
model = old_train_function(df)  # Old - still works!
```

**Option 3: Full Migration**
Update all calls to use config:

```python
from src.config import TradingConfig

config = TradingConfig()

# Customize configuration
config.indicators.momentum.rsi_periods = [14, 28]
config.hyperparameters.model_1h.n_estimators = 200

# Use everywhere
df = calculate_indicators(df, config=config)
model = train_model(df, config=config)
ensemble = MTFEnsemble(config=config)
```

---

## Rollback Instructions

### If You Need to Rollback

The system is backward compatible, so rollback is simple:

#### Option 1: Just Stop Using Config

```python
# Remove config parameter from calls
df = calculate_indicators(df)  # Works with defaults
model = train_model(df)  # Works with defaults
```

#### Option 2: Git Revert

```bash
# Find the config centralization commits
git log --oneline | grep "config"

# Revert to before centralization
git checkout <commit-before-week-1>

# Or revert specific weeks
git revert <week-5-commit>
git revert <week-4-commit>
# etc.
```

#### Option 3: Use Old Hardcoded Values

Create a config that matches old behavior:

```python
from src.config import TradingConfig

def get_legacy_config():
    """Returns config with old hardcoded values."""
    config = TradingConfig()

    # Set to old values (defaults already match, but explicit here)
    config.indicators.momentum.rsi_periods = [7, 14, 21]
    config.indicators.momentum.macd_fast = 12
    config.indicators.momentum.macd_slow = 26
    config.hyperparameters.model_1h.n_estimators = 150
    # ... etc.

    return config

# Use legacy config
config = get_legacy_config()
df = calculate_indicators(df, config=config)
```

---

## Troubleshooting

### Problem: "Config not updating"

**Symptom:** Changed config values but results unchanged

**Solution:**
```python
# Make sure you're using singleton correctly
config = TradingConfig()  # ✅ Correct

# NOT
config = TradingConfig.__init__()  # ❌ Wrong
```

---

### Problem: "Import errors"

**Symptom:** `ImportError: cannot import name 'TradingConfig'`

**Solution:**
```python
# Correct import
from src.config import TradingConfig

# NOT
from src.config.trading_config import TradingConfig  # Works but not recommended
```

---

### Problem: "Validation errors"

**Symptom:** Config validation fails

**Solution:**
```python
config = TradingConfig()

# Check what's wrong
errors = config.validate()
print(errors)

# Fix the issues
config.trading.confidence_threshold = 0.70  # Must be 0.0-1.0
config.model.weight_1h = 0.6
config.model.weight_4h = 0.3
config.model.weight_daily = 0.1  # Must sum to 1.0

# Verify
errors = config.validate()
assert len(errors) == 0
```

---

### Problem: "Performance issues"

**Symptom:** System slower after using config

**Solution:**

1. **Verify config access is fast:**
```python
import time

start = time.time()
_ = config.indicators.momentum.rsi_periods
duration = time.time() - start

assert duration < 0.001  # Should be < 1ms
```

2. **Check callback performance:**
```python
def slow_callback(params):
    start = time.time()
    # ... your code ...
    duration = time.time() - start
    if duration > 0.01:
        logger.warning(f"Slow callback: {duration*1000:.2f}ms")

config.register_callback("model", slow_callback)
```

3. **Profile your code:**
```bash
python -m cProfile -o profile.stats train_model.py
python -m pstats profile.stats
```

---

### Problem: "Tests failing"

**Symptom:** Tests fail after adding config

**Solution:**

1. **Update tests to pass config:**
```python
def test_calculate_rsi():
    config = TradingConfig()
    config.indicators.momentum.rsi_periods = [14]  # Control for testing

    df = calculate_rsi(sample_data, config=config)
    assert "rsi_14" in df.columns
```

2. **Use fixtures:**
```python
@pytest.fixture
def test_config():
    config = TradingConfig()
    config.indicators.momentum.rsi_periods = [14]
    return config

def test_with_config(test_config):
    df = calculate_rsi(sample_data, config=test_config)
    assert "rsi_14" in df.columns
```

---

## Additional Resources

- **Configuration Guide**: `docs/CONFIGURATION_GUIDE.md`
  Complete reference for all 87 parameters

- **Implementation Checklist**: `backend/CONFIGURATION_CENTRALIZATION_CHECKLIST.md`
  Step-by-step implementation tracking

- **Test Examples**:
  - `tests/unit/config/` - Unit test examples
  - `tests/integration/test_config_hot_reload.py` - Hot-reload examples
  - `tests/performance/test_config_performance.py` - Performance benchmarks
  - `tests/integration/test_backward_compatibility.py` - Compatibility tests

- **Source Code**:
  - `src/config/trading_config.py` - Main config class
  - `src/config/indicator_config.py` - Indicator parameters
  - `src/config/model_config.py` - Hyperparameters
  - `src/config/feature_config.py` - Feature engineering
  - `src/config/training_config.py` - Training parameters

---

## Summary

### What You Need to Know

1. **Backward Compatible**: Your existing code still works
2. **Optional Migration**: Migrate at your own pace
3. **Easy to Use**: Just `config = TradingConfig()`
4. **Well Tested**: 150+ tests, 95%+ coverage
5. **Fully Documented**: Complete guides available

### Quick Start

```python
from src.config import TradingConfig

# Get config
config = TradingConfig()

# Customize (optional)
config.indicators.momentum.rsi_periods = [14, 28, 42]

# Use it
df = calculate_indicators(df, config=config)
model = train_model(df, config=config)
```

### Get Help

- Check `docs/CONFIGURATION_GUIDE.md` for usage examples
- Review tests in `tests/unit/config/` for patterns
- See this migration guide for before/after comparisons

---

**Migration Status**: Complete ✅
**Version**: 1.0.0
**Date**: 2026-01-27
**Contact**: See project documentation for support
