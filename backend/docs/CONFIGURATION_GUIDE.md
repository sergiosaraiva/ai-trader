# Configuration Guide

**Date:** 2026-01-27
**Version:** 5.0.0 (Week 5 - Testing & Deployment Complete)

---

## Overview

All configuration parameters for the AI Trader system are centralized in the `TradingConfig` singleton class. This guide documents the new configuration sections added in Week 1 of the configuration centralization project.

### New Configuration Sections

| Section | Parameters | Purpose |
|---------|------------|---------|
| **Indicators** | 30 | Technical indicator parameters (SMA, EMA, RSI, MACD, ATR, etc.) |
| **Hyperparameters** | 30 | XGBoost model hyperparameters for 1H, 4H, and Daily models |
| **Feature Engineering** | 12 | Lag periods, session times, cyclical encoding |
| **Training** | 10 | Data splits, stacking CV, early stopping |
| **Labeling** | 6 | Triple barrier, multi-bar, volatility-adjusted labeling |

**Total:** 87 parameters centralized (from hardcoded values across 100+ files)

---

## Configuration Sections

### 1. Indicators (`config.indicators`)

Technical indicator parameters used in feature engineering.

#### 1.1 Trend Indicators (`config.indicators.trend`)

```python
from src.config import TradingConfig

config = TradingConfig()

# Moving averages
config.indicators.trend.sma_periods  # [5, 10, 20, 50, 100, 200]
config.indicators.trend.ema_periods  # [5, 10, 20, 50, 100, 200]
config.indicators.trend.wma_periods  # [10, 20, 50]

# Directional indicators
config.indicators.trend.adx_period  # 14
config.indicators.trend.aroon_period  # 25

# Supertrend
config.indicators.trend.supertrend_period  # 10
config.indicators.trend.supertrend_multiplier  # 3.0

# MA Crossovers
config.indicators.trend.sma_crossover_pairs  # [(5, 20), (20, 50), (50, 200)]
config.indicators.trend.ema_crossover_pairs  # [(5, 20), (12, 26)]
```

#### 1.2 Momentum Indicators (`config.indicators.momentum`)

```python
# RSI
config.indicators.momentum.rsi_periods  # [7, 14, 21]

# MACD
config.indicators.momentum.macd_fast  # 12
config.indicators.momentum.macd_slow  # 26
config.indicators.momentum.macd_signal  # 9

# Stochastic
config.indicators.momentum.stochastic_k_period  # 14
config.indicators.momentum.stochastic_d_period  # 3

# CCI
config.indicators.momentum.cci_periods  # [14, 20]
config.indicators.momentum.cci_constant  # 0.015

# Others
config.indicators.momentum.williams_period  # 14
config.indicators.momentum.mfi_period  # 14
config.indicators.momentum.tsi_long  # 25
config.indicators.momentum.tsi_short  # 13
```

#### 1.3 Volatility Indicators (`config.indicators.volatility`)

```python
# ATR
config.indicators.volatility.atr_period  # 14
config.indicators.volatility.natr_period  # 14

# Bollinger Bands
config.indicators.volatility.bollinger_period  # 20
config.indicators.volatility.bollinger_std  # 2.0

# Keltner Channel
config.indicators.volatility.keltner_period  # 20
config.indicators.volatility.keltner_multiplier  # 2.0

# Donchian Channel
config.indicators.volatility.donchian_period  # 20

# Standard Deviation
config.indicators.volatility.std_periods  # [10, 20]

# Historical Volatility
config.indicators.volatility.hvol_periods  # [10, 20, 30]
config.indicators.volatility.hvol_annualization_factor  # 252
```

#### 1.4 Volume Indicators (`config.indicators.volume`)

```python
# Chaikin Money Flow
config.indicators.volume.cmf_period  # 20

# Ease of Movement
config.indicators.volume.emv_period  # 14
config.indicators.volume.emv_scaling_factor  # 1e8

# Force Index
config.indicators.volume.force_index_period  # 13

# A/D Oscillator
config.indicators.volume.adosc_fast  # 3
config.indicators.volume.adosc_slow  # 10

# Volume SMA
config.indicators.volume.volume_sma_periods  # [10, 20]
```

---

### 2. Hyperparameters (`config.hyperparameters`)

XGBoost model hyperparameters for each timeframe.

#### 2.1 1H Model (`config.hyperparameters.model_1h`)

```python
# 1H Model: Short-term (highest weight 60%)
config.hyperparameters.model_1h.n_estimators  # 150
config.hyperparameters.model_1h.max_depth  # 5
config.hyperparameters.model_1h.learning_rate  # 0.03
config.hyperparameters.model_1h.min_child_weight  # 3
config.hyperparameters.model_1h.subsample  # 0.8
config.hyperparameters.model_1h.colsample_bytree  # 0.8
config.hyperparameters.model_1h.reg_alpha  # 0.1
config.hyperparameters.model_1h.reg_lambda  # 1.0
config.hyperparameters.model_1h.gamma  # 0.1
config.hyperparameters.model_1h.eval_metric  # "logloss"
config.hyperparameters.model_1h.random_state  # 42
```

#### 2.2 4H Model (`config.hyperparameters.model_4h`)

```python
# 4H Model: Medium-term (30% weight)
config.hyperparameters.model_4h.n_estimators  # 120
config.hyperparameters.model_4h.max_depth  # 4
config.hyperparameters.model_4h.learning_rate  # 0.03
# ... (same structure as 1H)
```

#### 2.3 Daily Model (`config.hyperparameters.model_daily`)

```python
# Daily Model: Long-term (10% weight + sentiment)
config.hyperparameters.model_daily.n_estimators  # 80
config.hyperparameters.model_daily.max_depth  # 3
config.hyperparameters.model_daily.learning_rate  # 0.03
# ... (same structure as 1H)
```

---

### 3. Feature Engineering (`config.feature_engineering`)

Parameters for feature engineering processes.

#### 3.1 Lag Parameters (`config.feature_engineering.lags`)

```python
# Standard lags for all features
config.feature_engineering.lags.standard_lags  # [1, 2, 3, 6, 12]

# Rate of change (ROC) periods
config.feature_engineering.lags.rsi_roc_periods  # [3, 6]
config.feature_engineering.lags.macd_roc_periods  # [3]
config.feature_engineering.lags.adx_roc_periods  # [3]
config.feature_engineering.lags.atr_roc_periods  # [3, 6]
config.feature_engineering.lags.price_roc_periods  # [1, 3, 6, 12]
config.feature_engineering.lags.volume_roc_periods  # [3, 6]
```

#### 3.2 Session Parameters (`config.feature_engineering.sessions`)

```python
# Trading session times (UTC hours)
config.feature_engineering.sessions.asian_session  # (0, 8)
config.feature_engineering.sessions.london_session  # (8, 16)
config.feature_engineering.sessions.ny_session  # (13, 22)
config.feature_engineering.sessions.overlap_session  # (13, 16)
config.feature_engineering.sessions.timezone_offset_hours  # 0
```

#### 3.3 Cyclical Encoding (`config.feature_engineering.cyclical`)

```python
# Cyclical time feature encoding
config.feature_engineering.cyclical.hour_encoding_cycles  # 24
config.feature_engineering.cyclical.day_of_week_cycles  # 7
config.feature_engineering.cyclical.day_of_month_cycles  # 31
```

#### 3.4 Normalization Windows

```python
config.feature_engineering.percentile_window  # 50
config.feature_engineering.zscore_window  # 50
```

---

### 4. Training (`config.training`)

Training pipeline parameters.

#### 4.1 Data Splits (`config.training.splits`)

```python
# Train/Val/Test ratios
config.training.splits.train_ratio  # 0.6 (60%)
config.training.splits.validation_ratio  # 0.2 (20%)
config.training.splits.test_ratio  # 0.2 (20%, implicit)
config.training.splits.enforce_chronological  # True
```

#### 4.2 Stacking (`config.training.stacking`)

```python
# Cross-validation for stacking meta-learner
config.training.stacking.n_folds  # 5
config.training.stacking.min_train_size  # 500
config.training.stacking.shuffle  # False (time series)
config.training.stacking.stratified  # True
config.training.stacking.use_base_hyperparams  # True
config.training.stacking.custom_hyperparams  # {}
```

#### 4.3 Early Stopping (`config.training.early_stopping`)

```python
# XGBoost early stopping
config.training.early_stopping.enabled  # True
config.training.early_stopping.stopping_rounds  # 10
config.training.early_stopping.eval_metric  # "logloss"
config.training.early_stopping.verbose  # False
```

---

### 5. Labeling (`config.labeling`)

Labeling method parameters.

```python
# Primary labeling method
config.labeling.primary_method  # "triple_barrier"

# Triple barrier (uses TradingConfig.timeframes for TP/SL/holding)
config.labeling.triple_barrier.pip_value  # 0.0001

# Multi-bar lookahead (alternative)
config.labeling.multi_bar.forward_bars  # 12
config.labeling.multi_bar.threshold_pips  # 10.0
config.labeling.multi_bar.pip_value  # 0.0001

# Volatility-adjusted (alternative)
config.labeling.volatility.atr_multiplier  # 2.0
config.labeling.volatility.use_dynamic_barriers  # False
```

---

## Usage Examples

### Basic Usage

```python
from src.config import TradingConfig

# Load configuration with defaults
config = TradingConfig()

# Access indicator parameters
rsi_periods = config.indicators.momentum.rsi_periods  # [7, 14, 21]
macd_fast = config.indicators.momentum.macd_fast  # 12

# Access hyperparameters
n_estimators_1h = config.hyperparameters.model_1h.n_estimators  # 150
max_depth_1h = config.hyperparameters.model_1h.max_depth  # 5

# Access feature engineering params
standard_lags = config.feature_engineering.lags.standard_lags  # [1, 2, 3, 6, 12]
asian_session = config.feature_engineering.sessions.asian_session  # (0, 8)

# Access training params
train_ratio = config.training.splits.train_ratio  # 0.6
n_folds = config.training.stacking.n_folds  # 5
```

### Custom Configuration

```python
from src.config import TradingConfig

config = TradingConfig()

# Override indicator parameters
config.indicators.momentum.rsi_periods = [14, 28, 42]
config.indicators.volatility.bollinger_period = 30

# Override hyperparameters
config.hyperparameters.model_1h.n_estimators = 200
config.hyperparameters.model_1h.learning_rate = 0.01

# Override feature engineering params
config.feature_engineering.lags.standard_lags = [1, 5, 10]
config.feature_engineering.sessions.asian_session = (1, 9)

# Override training params
config.training.splits.train_ratio = 0.7
config.training.stacking.n_folds = 10
```

### Using in Model Training

```python
from src.config import TradingConfig
from src.models.multi_timeframe import MTFEnsemble

# Create custom configuration
config = TradingConfig()
config.hyperparameters.model_1h.n_estimators = 200
config.indicators.momentum.rsi_periods = [14]

# Pass config to ensemble (Week 3 implementation)
# ensemble = MTFEnsemble(config=config)
# ensemble.train(X, y)
```

---

## API Integration (Future)

Once the API integration is complete (Week 2+), you'll be able to update configuration via API:

```bash
# Update indicator configuration
POST /api/v1/config/update
{
    "category": "indicators",
    "updates": {
        "momentum.rsi_periods": [10, 20, 30],
        "volatility.bollinger_period": 25
    }
}

# Update hyperparameters
POST /api/v1/config/update
{
    "category": "hyperparameters",
    "updates": {
        "model_1h.n_estimators": 180,
        "model_1h.learning_rate": 0.02
    }
}

# Changes take effect immediately for new calculations
```

---

## Before/After Comparison

### Before (Hardcoded)

```python
# src/features/technical/momentum.py
def calculate_rsi(df):
    rsi_periods = [7, 14, 21]  # ❌ HARDCODED
    for period in rsi_periods:
        df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)
    return df
```

### After (Centralized)

```python
# src/features/technical/momentum.py
from src.config import TradingConfig

def calculate_rsi(df, config=None):
    if config is None:
        config = TradingConfig()

    rsi_periods = config.indicators.momentum.rsi_periods  # ✅ FROM CONFIG
    for period in rsi_periods:
        df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)
    return df
```

---

## Implementation Status

### Week 1 (✅ COMPLETE)

- ✅ Infrastructure: 5 new config dataclass files created
- ✅ Integration: TradingConfig extended with new sections
- ✅ Testing: 48 unit tests for all config sections
- ✅ Documentation: Configuration guide started

### Week 2 (✅ COMPLETE)

- ✅ Technical indicator functions updated to use config (30 params)
- ✅ TechnicalCalculator updated with config injection
- ✅ All indicator modules centralized
- ✅ 81% test pass rate (momentum indicators)

### Week 3 (✅ COMPLETE)

- ✅ Model hyperparameters integrated (30 params)
- ✅ ImprovedTimeframeModel uses config
- ✅ MTFEnsemble uses config
- ✅ Training scripts updated
- ✅ 100% test pass rate

### Week 4 (✅ COMPLETE)

- ✅ Feature engineering updated (12 params)
- ✅ Training parameters integrated (10 params)
- ✅ Stacking meta-learner uses config
- ✅ Enhanced features use config
- ✅ 100% test pass rate

### Week 5 (✅ COMPLETE)

- ✅ Comprehensive integration tests (full pipeline)
- ✅ Performance tests (< 10ms config load verified)
- ✅ Backward compatibility tests
- ✅ Hot-reload implementation tested
- ✅ Documentation complete (this guide + migration guide)
- ✅ 150+ tests total, 95%+ coverage

**Total Progress**: 87/87 parameters centralized (100%)
**Test Coverage**: 95%+ across all config sections
**Status**: Production Ready

---

## Validation

All configuration parameters are validated on initialization and update:

```python
config = TradingConfig()
errors = config.validate()

if errors:
    print(f"Configuration errors: {errors}")
else:
    print("Configuration valid")
```

---

## Benefits

### 1. Single Source of Truth

All configuration parameters are centralized in one place, eliminating duplication across 100+ files.

### 2. Easy Experimentation

Change parameters in seconds without modifying code:

```python
# Test different RSI periods
for rsi_config in [[7, 14], [14, 21], [7, 14, 21]]:
    config = TradingConfig()
    config.indicators.momentum.rsi_periods = rsi_config
    results = run_backtest(config)
    print(f"RSI {rsi_config}: Win Rate = {results.win_rate:.2%}")
```

### 3. Reproducibility

Save and restore exact configurations for reproducible experiments:

```python
# Save configuration
config_dict = config.get_all()
with open("experiment_config.json", "w") as f:
    json.dump(config_dict, f)

# Load configuration
with open("experiment_config.json", "r") as f:
    config_dict = json.load(f)
# Apply loaded config...
```

### 4. Hot Reload

Update configuration without restarting services - IMPLEMENTED!

---

## Hot-Reload Guide

The configuration system supports hot-reload, allowing you to update configuration without restarting services.

### How It Works

1. **Configuration Version Tracking**: Each reload increments a version number
2. **Callback System**: Services register callbacks to be notified of changes
3. **Database Integration**: Load configuration from database on demand
4. **Thread-Safe**: All operations are thread-safe using locks

### Basic Hot-Reload

```python
from src.config import TradingConfig
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Get config instance
config = TradingConfig()

# Create database session
engine = create_engine("sqlite:///./data/trading.db")
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

# Reload configuration from database
result = config.reload(db_session=db)

print(f"Status: {result['status']}")
print(f"Changes: {result['changes']}")
print(f"Version: {result['version']}")
print(f"Timestamp: {result['timestamp']}")

db.close()
```

### Register Callbacks

Services can register callbacks to react to configuration changes:

```python
def on_config_change(trading_params):
    """Called when trading config changes."""
    print(f"New confidence threshold: {trading_params.confidence_threshold}")
    # Update service state, invalidate caches, etc.

# Register callback
config.register_callback("trading", on_config_change)

# Now reload will trigger the callback
config.reload(db_session=db)
```

### Available Callback Categories

- `trading` - Trading execution parameters
- `model` - MTF ensemble weights and settings
- `risk` - Risk management parameters
- `system` - System-wide settings
- `hyperparameters` - Model hyperparameters
- `indicators` - Technical indicator parameters
- `features` - Feature engineering parameters
- `training` - Training pipeline parameters

### Cache Invalidation

Use config version to invalidate caches:

```python
class ModelService:
    def __init__(self):
        self.config = TradingConfig()
        self.cache = {}
        self.cache_version = self.config.get_config_version()

        # Register callback for config changes
        self.config.register_callback("model", self._on_config_change)

    def _on_config_change(self, model_params):
        """Invalidate cache on config change."""
        new_version = self.config.get_config_version()
        if new_version != self.cache_version:
            self.cache.clear()
            self.cache_version = new_version
            logger.info(f"Cache invalidated (v{new_version})")

    def get_prediction(self, data):
        cache_key = hash(data)
        version_key = (cache_key, self.cache_version)

        if version_key in self.cache:
            return self.cache[version_key]

        # Compute prediction
        result = self._compute(data)
        self.cache[version_key] = result
        return result
```

### Performance

- **Initialization**: < 10ms (tested)
- **Singleton access**: < 0.1ms avg (tested)
- **Hot reload**: < 100ms for typical changes (tested)
- **Validation**: < 5ms (tested)

See `tests/performance/test_config_performance.py` for benchmarks.

---

## Troubleshooting

### Common Issues

#### Issue: Config changes not taking effect

**Symptom**: Updated config values but model still uses old values

**Solution**:
1. Check if you're using singleton correctly:
   ```python
   config = TradingConfig()  # ✅ Correct
   # NOT: config = TradingConfig.__init__()  # ❌ Wrong
   ```

2. Verify config version incremented:
   ```python
   before = config.get_config_version()
   config.update(...)
   after = config.get_config_version()
   assert after > before
   ```

3. Check if service registered callback:
   ```python
   config.register_callback("model", my_callback)
   ```

#### Issue: Validation errors on reload

**Symptom**: `reload()` returns `status="error"` with validation message

**Solution**:
1. Check database values are valid:
   ```python
   # Confidence must be 0.0-1.0
   # Weights must sum to 1.0
   # Periods must be positive integers
   ```

2. Test configuration locally first:
   ```python
   config = TradingConfig()
   config.trading.confidence_threshold = 1.5  # Invalid!
   errors = config.validate()
   print(errors)  # Shows what's wrong
   ```

#### Issue: Performance degradation after hot-reload

**Symptom**: System slower after config reload

**Solution**:
1. Check callback performance:
   ```python
   import time

   def slow_callback(params):
       start = time.time()
       # ... callback code ...
       duration = time.time() - start
       if duration > 0.01:  # 10ms threshold
           logger.warning(f"Slow callback: {duration*1000:.1f}ms")
   ```

2. Ensure callbacks don't do heavy computation:
   ```python
   # ❌ Bad: Heavy computation in callback
   def bad_callback(params):
       model.retrain()  # Too slow!

   # ✅ Good: Set flag, process later
   def good_callback(params):
       self.needs_retrain = True
   ```

#### Issue: Database session errors

**Symptom**: `reload()` fails with database errors

**Solution**:
1. Always pass valid session:
   ```python
   # ✅ Correct
   with SessionLocal() as db:
       config.reload(db_session=db)

   # ❌ Wrong
   config.reload(db_session=None)  # Returns error
   ```

2. Check database schema is up to date:
   ```bash
   # Run migrations if needed
   alembic upgrade head
   ```

### Debugging

Enable debug logging to troubleshoot config issues:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("src.config")
logger.setLevel(logging.DEBUG)

# Now you'll see detailed logs
config = TradingConfig()
config.reload(db_session=db)
```

### Validation Details

Run validation to see what's wrong:

```python
config = TradingConfig()

# Get all validation errors
errors = config.validate()

if errors:
    print("Configuration errors found:")
    for error in errors:
        print(f"  - {error}")
else:
    print("✓ Configuration valid")
```

### Getting Help

1. Check test files for examples:
   - `tests/unit/config/` - Unit tests
   - `tests/integration/test_config_hot_reload.py` - Hot-reload tests
   - `tests/performance/test_config_performance.py` - Performance tests

2. Review documentation:
   - `CONFIGURATION_GUIDE.md` - This file
   - `MIGRATION_TO_CENTRALIZED_CONFIG.md` - Migration guide
   - `CONFIGURATION_CENTRALIZATION_CHECKLIST.md` - Implementation checklist

3. Check implementation:
   - `src/config/trading_config.py` - Main config class
   - `src/config/*.py` - Individual config sections

---

## Support

For questions or issues:
- See `CONFIGURATION_CENTRALIZATION_IMPLEMENTATION_PLAN.md` for full details
- See `CONFIGURATION_CENTRALIZATION_CHECKLIST.md` for progress tracking
- Check unit tests in `tests/unit/config/` for usage examples

---

**Status:** Week 5 Complete (Production Ready)
**Progress:** 87/87 parameters centralized (100%)
**Version:** 5.0.0
**Date:** 2026-01-27
**Test Coverage:** 95%+ (150+ tests passing)
