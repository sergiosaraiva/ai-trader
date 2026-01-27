# Configuration Guide

**Date:** 2026-01-27
**Version:** 1.0.0 (Week 1 - Infrastructure Complete)

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

**Total:** 88 parameters centralized (from 76 hardcoded)

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

### Week 1 (COMPLETE)

- ✅ Infrastructure: 5 new config dataclass files created
- ✅ Integration: TradingConfig extended with new sections
- ✅ Testing: 20+ unit tests for all config sections
- ✅ Documentation: This guide

### Week 2 (Planned)

- Technical indicator functions updated to use config (30 params)
- TechnicalCalculator updated with config injection

### Week 3 (Planned)

- Model hyperparameters integrated (30 params)
- Training scripts updated

### Week 4 (Planned)

- Feature engineering updated (12 params)
- Training parameters integrated (10 params)

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

### 4. Hot Reload (Future)

Update configuration without restarting services (Week 5+).

---

## Support

For questions or issues:
- See `CONFIGURATION_CENTRALIZATION_IMPLEMENTATION_PLAN.md` for full details
- See `CONFIGURATION_CENTRALIZATION_CHECKLIST.md` for progress tracking
- Check unit tests in `tests/unit/config/` for usage examples

---

**Status:** Week 1 Complete (Infrastructure Ready)
**Next:** Week 2 - Technical Indicators Migration
**Version:** 1.0.0
**Date:** 2026-01-27
