# Configuration Centralization - Step-by-Step Checklist

**Total Steps:** 89
**Estimated Time:** 6 weeks (240 hours)
**Current Progress:** 0/89 steps complete

---

## How to Use This Checklist

1. **Complete steps in order** - Dependencies are marked
2. **Check off each step** when done
3. **Run verification** after each section
4. **Don't skip testing steps**
5. **Commit after each major section**

---

## WEEK 1: INFRASTRUCTURE (Days 1-5)

### Phase 1.1: Create Config Files (Day 1)

#### Step 1: Create Indicator Config File
- [ ] **1.1** Create file `src/config/indicator_config.py`
- [ ] **1.2** Add imports:
```python
from dataclasses import dataclass, field
from typing import List, Tuple
```
- [ ] **1.3** Copy `TrendIndicators` dataclass from implementation plan
- [ ] **1.4** Copy `MomentumIndicators` dataclass
- [ ] **1.5** Copy `VolatilityIndicators` dataclass
- [ ] **1.6** Copy `VolumeIndicators` dataclass
- [ ] **1.7** Copy `IndicatorParameters` wrapper dataclass
- [ ] **1.8** Save file

**Verify:** Run `python -m py_compile src/config/indicator_config.py`
**Estimated time:** 2 hours

#### Step 2: Create Model Hyperparameters Config File
- [ ] **2.1** Create file `src/config/model_config.py`
- [ ] **2.2** Add imports
- [ ] **2.3** Copy `XGBoostHyperparameters` dataclass from plan
- [ ] **2.4** Copy `ModelHyperparameters` dataclass
- [ ] **2.5** Verify default values:
  - 1H: n_estimators=150, max_depth=5, lr=0.03
  - 4H: n_estimators=120, max_depth=4, lr=0.03
  - Daily: n_estimators=80, max_depth=3, lr=0.03
- [ ] **2.6** Save file

**Verify:** Run `python -m py_compile src/config/model_config.py`
**Estimated time:** 1.5 hours

#### Step 3: Create Feature Engineering Config File
- [ ] **3.1** Create file `src/config/feature_config.py`
- [ ] **3.2** Copy `LagParameters` dataclass
- [ ] **3.3** Copy `SessionParameters` dataclass
- [ ] **3.4** Copy `CyclicalEncoding` dataclass
- [ ] **3.5** Copy `FeatureParameters` wrapper
- [ ] **3.6** Save file

**Verify:** Run `python -m py_compile src/config/feature_config.py`
**Estimated time:** 1 hour

#### Step 4: Create Training Config File
- [ ] **4.1** Create file `src/config/training_config.py`
- [ ] **4.2** Copy `DataSplitParameters` dataclass
- [ ] **4.3** Copy `StackingParameters` dataclass
- [ ] **4.4** Copy `EarlyStoppingParameters` dataclass
- [ ] **4.5** Copy `TrainingParameters` wrapper
- [ ] **4.6** Save file

**Verify:** Run `python -m py_compile src/config/training_config.py`
**Estimated time:** 1 hour

#### Step 5: Create Labeling Config File
- [ ] **5.1** Create file `src/config/labeling_config.py`
- [ ] **5.2** Copy `TripleBarrierParameters` dataclass
- [ ] **5.3** Copy `MultiBarParameters` dataclass
- [ ] **5.4** Copy `VolatilityAdjustedParameters` dataclass
- [ ] **5.5** Copy `LabelingParameters` wrapper
- [ ] **5.6** Save file

**Verify:** Run `python -m py_compile src/config/labeling_config.py`
**Estimated time:** 1 hour

**✅ CHECKPOINT 1:** All 5 config files created and compile successfully

---

### Phase 1.2: Integrate into TradingConfig (Day 2)

#### Step 6: Update TradingConfig Main File
- [ ] **6.1** Open `src/config/trading_config.py`
- [ ] **6.2** Add imports at top:
```python
from .indicator_config import IndicatorParameters
from .model_config import ModelHyperparameters
from .feature_config import FeatureParameters
from .training_config import TrainingParameters
from .labeling_config import LabelingParameters
```
- [ ] **6.3** Add new fields to `TradingConfig` class:
```python
indicators: IndicatorParameters = field(default_factory=IndicatorParameters)
hyperparameters: ModelHyperparameters = field(default_factory=ModelHyperparameters)
features: FeatureParameters = field(default_factory=FeatureParameters)
training: TrainingParameters = field(default_factory=TrainingParameters)
labeling: LabelingParameters = field(default_factory=LabelingParameters)
```
- [ ] **6.4** Save file

**Verify:**
```bash
python -c "from src.config import TradingConfig; c=TradingConfig(); print('✓ Indicators:', c.indicators); print('✓ Hyperparameters:', c.hyperparameters)"
```
**Estimated time:** 30 minutes

#### Step 7: Create Config Validation
- [ ] **7.1** Add validation method to TradingConfig:
```python
def validate_new_sections(self) -> List[str]:
    """Validate new config sections."""
    errors = []

    # Validate indicator periods
    if not self.indicators.momentum.rsi_periods:
        errors.append("RSI periods cannot be empty")

    # Validate hyperparameters
    if self.hyperparameters.model_1h.n_estimators < 1:
        errors.append("n_estimators must be positive")

    # Add more validations...
    return errors
```
- [ ] **7.2** Call validation in `__post_init__`
- [ ] **7.3** Test validation works

**Verify:**
```bash
python -c "from src.config import TradingConfig; c=TradingConfig(); errs=c.validate_new_sections(); print('Validation errors:', len(errs))"
```
**Estimated time:** 2 hours

**✅ CHECKPOINT 2:** TradingConfig extended with new sections

---

### Phase 1.3: Unit Tests (Day 3)

#### Step 8: Create Indicator Config Tests
- [ ] **8.1** Create file `tests/unit/config/test_indicator_config.py`
- [ ] **8.2** Write test `test_trend_indicators_defaults()`
- [ ] **8.3** Write test `test_momentum_indicators_defaults()`
- [ ] **8.4** Write test `test_volatility_indicators_defaults()`
- [ ] **8.5** Write test `test_volume_indicators_defaults()`
- [ ] **8.6** Write test `test_indicator_override()`
- [ ] **8.7** Run tests: `pytest tests/unit/config/test_indicator_config.py -v`

**Verify:** All tests pass
**Estimated time:** 2 hours

#### Step 9: Create Hyperparameter Config Tests
- [ ] **9.1** Create file `tests/unit/config/test_model_config.py`
- [ ] **9.2** Write test `test_model_1h_defaults()`
- [ ] **9.3** Write test `test_model_4h_defaults()`
- [ ] **9.4** Write test `test_model_daily_defaults()`
- [ ] **9.5** Write test `test_hyperparameter_override()`
- [ ] **9.6** Run tests: `pytest tests/unit/config/test_model_config.py -v`

**Verify:** All tests pass
**Estimated time:** 1.5 hours

#### Step 10: Create Feature Config Tests
- [ ] **10.1** Create file `tests/unit/config/test_feature_config.py`
- [ ] **10.2** Write test `test_lag_parameters_defaults()`
- [ ] **10.3** Write test `test_session_parameters_defaults()`
- [ ] **10.4** Write test `test_cyclical_encoding_defaults()`
- [ ] **10.5** Run tests: `pytest tests/unit/config/test_feature_config.py -v`

**Verify:** All tests pass
**Estimated time:** 1 hour

#### Step 11: Create Training Config Tests
- [ ] **11.1** Create file `tests/unit/config/test_training_config.py`
- [ ] **11.2** Write test `test_split_parameters_defaults()`
- [ ] **11.3** Write test `test_stacking_parameters_defaults()`
- [ ] **11.4** Write test `test_early_stopping_defaults()`
- [ ] **11.5** Run tests: `pytest tests/unit/config/test_training_config.py -v`

**Verify:** All tests pass
**Estimated time:** 1 hour

**✅ CHECKPOINT 3:** All infrastructure tests pass

---

### Phase 1.4: Documentation (Day 4-5)

#### Step 12: Document New Config Sections
- [ ] **12.1** Create file `docs/CONFIGURATION_GUIDE.md`
- [ ] **12.2** Document `indicators` section with all parameters
- [ ] **12.3** Document `hyperparameters` section with all parameters
- [ ] **12.4** Document `features` section with all parameters
- [ ] **12.5** Document `training` section with all parameters
- [ ] **12.6** Document `labeling` section with all parameters
- [ ] **12.7** Add usage examples for each section
- [ ] **12.8** Add "before/after" comparison examples

**Verify:** Documentation complete and readable
**Estimated time:** 4 hours

#### Step 13: Commit Infrastructure
- [ ] **13.1** Stage all changes: `git add src/config/ tests/unit/config/ docs/`
- [ ] **13.2** Commit:
```bash
git commit -m "feat: Add infrastructure for centralized config

- Add 5 new config dataclass files (indicators, hyperparameters, features, training, labeling)
- Extend TradingConfig with new sections
- Add validation for new sections
- Add comprehensive unit tests (20+ tests)
- Add configuration documentation

Part of configuration centralization project (Week 1/6)"
```
- [ ] **13.3** Push: `git push origin feature/centralized-config`

**✅ CHECKPOINT 4:** Week 1 complete - Infrastructure ready

---

## WEEK 2: TECHNICAL INDICATORS (Days 6-10)

### Phase 2.1: Trend Indicators (Day 6)

#### Step 14: Backup Existing Trend File
- [ ] **14.1** Copy `src/features/technical/trend.py` to `src/features/technical/trend.py.backup`

#### Step 15: Update Trend Indicators
- [ ] **15.1** Open `src/features/technical/trend.py`
- [ ] **15.2** Add import: `from ...config import TradingConfig`
- [ ] **15.3** Update `calculate_all()` signature:
```python
def calculate_all(df: pd.DataFrame, config: Optional[TradingConfig] = None) -> pd.DataFrame:
    if config is None:
        config = TradingConfig()
```
- [ ] **15.4** Replace hardcoded `sma_periods = [5, 10, 20, 50, 100, 200]` with:
```python
sma_periods = config.indicators.trend.sma_periods
```
- [ ] **15.5** Replace hardcoded `ema_periods` with config value
- [ ] **15.6** Replace hardcoded `adx_period` with config value
- [ ] **15.7** Replace hardcoded `aroon_period` with config value
- [ ] **15.8** Replace hardcoded crossover pairs with config values
- [ ] **15.9** Save file

**Verify:**
```bash
python -c "from src.features.technical.trend import calculate_all; print('✓ Trend indicators updated')"
```
**Estimated time:** 2 hours

#### Step 16: Test Trend Indicators
- [ ] **16.1** Create test file `tests/unit/features/test_trend_with_config.py`
- [ ] **16.2** Write test that verifies config is used
- [ ] **16.3** Write test that verifies custom config works
- [ ] **16.4** Run: `pytest tests/unit/features/test_trend_with_config.py -v`

**Verify:** All tests pass
**Estimated time:** 1 hour

---

### Phase 2.2: Momentum Indicators (Day 7)

#### Step 17: Update Momentum Indicators
- [ ] **17.1** Backup `src/features/technical/momentum.py`
- [ ] **17.2** Add TradingConfig import
- [ ] **17.3** Update `calculate_all()` signature with config parameter
- [ ] **17.4** Replace hardcoded `rsi_periods` with `config.indicators.momentum.rsi_periods`
- [ ] **17.5** Replace hardcoded `macd_fast` with config value
- [ ] **17.6** Replace hardcoded `macd_slow` with config value
- [ ] **17.7** Replace hardcoded `macd_signal` with config value
- [ ] **17.8** Replace hardcoded `stochastic_k_period` with config value
- [ ] **17.9** Replace hardcoded `stochastic_d_period` with config value
- [ ] **17.10** Replace hardcoded `cci_periods` with config value
- [ ] **17.11** Replace hardcoded `mfi_period` with config value
- [ ] **17.12** Replace hardcoded `williams_period` with config value
- [ ] **17.13** Save file

**Verify:**
```bash
python -c "from src.features.technical.momentum import calculate_all; print('✓ Momentum indicators updated')"
```
**Estimated time:** 2 hours

#### Step 18: Test Momentum Indicators
- [ ] **18.1** Create test file `tests/unit/features/test_momentum_with_config.py`
- [ ] **18.2** Write tests for RSI with config
- [ ] **18.3** Write tests for MACD with config
- [ ] **18.4** Write tests for Stochastic with config
- [ ] **18.5** Run: `pytest tests/unit/features/test_momentum_with_config.py -v`

**Verify:** All tests pass
**Estimated time:** 1 hour

---

### Phase 2.3: Volatility Indicators (Day 8)

#### Step 19: Update Volatility Indicators
- [ ] **19.1** Backup `src/features/technical/volatility.py`
- [ ] **19.2** Add TradingConfig import
- [ ] **19.3** Update `calculate_all()` signature with config parameter
- [ ] **19.4** Replace hardcoded `atr_period` with `config.indicators.volatility.atr_period`
- [ ] **19.5** Replace hardcoded `bollinger_period` with config value
- [ ] **19.6** Replace hardcoded `bollinger_std` with config value
- [ ] **19.7** Replace hardcoded `keltner_period` with config value
- [ ] **19.8** Replace hardcoded `keltner_multiplier` with config value
- [ ] **19.9** Replace hardcoded `donchian_period` with config value
- [ ] **19.10** Replace hardcoded `std_periods` with config value
- [ ] **19.11** Save file

**Verify:**
```bash
python -c "from src.features.technical.volatility import calculate_all; print('✓ Volatility indicators updated')"
```
**Estimated time:** 2 hours

#### Step 20: Test Volatility Indicators
- [ ] **20.1** Create test file `tests/unit/features/test_volatility_with_config.py`
- [ ] **20.2** Write tests for ATR with config
- [ ] **20.3** Write tests for Bollinger Bands with config
- [ ] **20.4** Write tests for Keltner Channel with config
- [ ] **20.5** Run: `pytest tests/unit/features/test_volatility_with_config.py -v`

**Verify:** All tests pass
**Estimated time:** 1 hour

---

### Phase 2.4: Volume Indicators (Day 9)

#### Step 21: Update Volume Indicators
- [ ] **21.1** Backup `src/features/technical/volume.py`
- [ ] **21.2** Add TradingConfig import
- [ ] **21.3** Update `calculate_all()` signature with config parameter
- [ ] **21.4** Replace hardcoded `cmf_period` with `config.indicators.volume.cmf_period`
- [ ] **21.5** Replace hardcoded `volume_sma_periods` with config value
- [ ] **21.6** Replace hardcoded `emv_period` with config value
- [ ] **21.7** Replace hardcoded `force_index_period` with config value
- [ ] **21.8** Replace hardcoded `adosc_fast` with config value
- [ ] **21.9** Replace hardcoded `adosc_slow` with config value
- [ ] **21.10** Save file

**Verify:**
```bash
python -c "from src.features.technical.volume import calculate_all; print('✓ Volume indicators updated')"
```
**Estimated time:** 1.5 hours

#### Step 22: Test Volume Indicators
- [ ] **22.1** Create test file `tests/unit/features/test_volume_with_config.py`
- [ ] **22.2** Write tests for CMF with config
- [ ] **22.3** Write tests for Volume SMA with config
- [ ] **22.4** Run: `pytest tests/unit/features/test_volume_with_config.py -v`

**Verify:** All tests pass
**Estimated time:** 1 hour

---

### Phase 2.5: Technical Calculator (Day 10)

#### Step 23: Update Technical Calculator
- [ ] **23.1** Open `src/features/technical/calculator.py`
- [ ] **23.2** Add config parameter to `__init__()`:
```python
def __init__(self, config: Optional[TradingConfig] = None):
    self.config = config or TradingConfig()
```
- [ ] **23.3** Update `calculate_all_indicators()` to pass config to all functions:
```python
df = calc_trend(df, config=self.config)
df = calc_momentum(df, config=self.config)
df = calc_volatility(df, config=self.config)
df = calc_volume(df, config=self.config)
```
- [ ] **23.4** Save file

**Verify:**
```bash
python -c "from src.features.technical import TechnicalCalculator; calc=TechnicalCalculator(); print('✓ Calculator updated')"
```
**Estimated time:** 1 hour

#### Step 24: Integration Test for Indicators
- [ ] **24.1** Create test `tests/integration/test_indicators_integration.py`
- [ ] **24.2** Write test that calculates all indicators with custom config
- [ ] **24.3** Verify correct indicator periods are generated
- [ ] **24.4** Run: `pytest tests/integration/test_indicators_integration.py -v`

**Verify:** All tests pass
**Estimated time:** 1.5 hours

#### Step 25: Commit Week 2
- [ ] **25.1** Run all tests: `pytest tests/unit/features/ tests/integration/ -v`
- [ ] **25.2** Stage changes: `git add src/features/technical/ tests/`
- [ ] **25.3** Commit:
```bash
git commit -m "feat: Centralize all technical indicator parameters

- Update trend indicators to use TradingConfig (8 params)
- Update momentum indicators to use TradingConfig (9 params)
- Update volatility indicators to use TradingConfig (7 params)
- Update volume indicators to use TradingConfig (6 params)
- Update TechnicalCalculator with config injection
- Add comprehensive tests for all indicators

Total: 30 indicator parameters centralized
Part of configuration centralization project (Week 2/6)"
```
- [ ] **25.4** Push: `git push origin feature/centralized-config`

**✅ CHECKPOINT 5:** Week 2 complete - All indicators centralized

---

## WEEK 3: MODEL HYPERPARAMETERS (Days 11-15)

### Phase 3.1: Update ImprovedTimeframeModel (Day 11-12)

#### Step 26: Update Model Class
- [ ] **26.1** Backup `src/models/multi_timeframe/improved_model.py`
- [ ] **26.2** Add import: `from ...config import TradingConfig`
- [ ] **26.3** Update `__init__()` to accept config:
```python
def __init__(self, timeframe: str, config: Optional[TradingConfig] = None, **kwargs):
    self.timeframe = timeframe
    self.config = config or TradingConfig()

    # Load hyperparameters from config
    if timeframe == "1H":
        self.hyperparams = self.config.hyperparameters.model_1h
    elif timeframe == "4H":
        self.hyperparams = self.config.hyperparameters.model_4h
    elif timeframe == "D":
        self.hyperparams = self.config.hyperparameters.model_daily
    else:
        raise ValueError(f"Unknown timeframe: {timeframe}")
```
- [ ] **26.4** Save file

**Verify:**
```bash
python -c "from src.models.multi_timeframe.improved_model import ImprovedTimeframeModel; print('✓ Model class updated')"
```
**Estimated time:** 1 hour

#### Step 27: Update _build_model Method
- [ ] **27.1** Find `_build_model()` method
- [ ] **27.2** Replace hardcoded hyperparameters with:
```python
self.model = xgb.XGBClassifier(
    n_estimators=self.hyperparams.n_estimators,
    max_depth=self.hyperparams.max_depth,
    learning_rate=self.hyperparams.learning_rate,
    min_child_weight=self.hyperparams.min_child_weight,
    subsample=self.hyperparams.subsample,
    colsample_bytree=self.hyperparams.colsample_bytree,
    reg_alpha=self.hyperparams.reg_alpha,
    reg_lambda=self.hyperparams.reg_lambda,
    gamma=self.hyperparams.gamma,
    eval_metric=self.hyperparams.eval_metric,
    random_state=self.hyperparams.random_state,
    use_label_encoder=False
)
```
- [ ] **27.3** Save file

**Verify:** Model builds successfully
**Estimated time:** 1 hour

#### Step 28: Update Factory Methods
- [ ] **28.1** Find `create_1h_model()` classmethod
- [ ] **28.2** Update signature:
```python
@classmethod
def create_1h_model(cls, trading_config: Optional[TradingConfig] = None, **kwargs):
    trading_config = trading_config or TradingConfig()
    return cls(timeframe="1H", config=trading_config, **kwargs)
```
- [ ] **28.3** Update `create_4h_model()` similarly
- [ ] **28.4** Update `create_daily_model()` similarly
- [ ] **28.5** Remove hardcoded TimeframeConfig from factory methods
- [ ] **28.6** Save file

**Verify:** Factory methods work
**Estimated time:** 1 hour

#### Step 29: Test Model with Config
- [ ] **29.1** Create test `tests/unit/models/test_model_with_config.py`
- [ ] **29.2** Write test that creates 1H model with custom config
- [ ] **29.3** Verify hyperparameters are loaded from config
- [ ] **29.4** Write test for 4H model
- [ ] **29.5** Write test for Daily model
- [ ] **29.6** Run: `pytest tests/unit/models/test_model_with_config.py -v`

**Verify:** All tests pass
**Estimated time:** 2 hours

---

### Phase 3.2: Update MTF Ensemble (Day 13)

#### Step 30: Update MTFEnsemble Class
- [ ] **30.1** Backup `src/models/multi_timeframe/mtf_ensemble.py`
- [ ] **30.2** Update `__init__()` to accept config:
```python
def __init__(self, config: Optional[TradingConfig] = None, **kwargs):
    self.config = config or TradingConfig()

    # Create models with centralized config
    self.models = {
        "1H": ImprovedTimeframeModel.create_1h_model(trading_config=self.config),
        "4H": ImprovedTimeframeModel.create_4h_model(trading_config=self.config),
        "D": ImprovedTimeframeModel.create_daily_model(trading_config=self.config)
    }
```
- [ ] **30.3** Verify weights already use config (from earlier phase)
- [ ] **30.4** Save file

**Verify:**
```bash
python -c "from src.models.multi_timeframe.mtf_ensemble import MTFEnsemble; e=MTFEnsemble(); print('✓ Ensemble updated')"
```
**Estimated time:** 1 hour

#### Step 31: Test MTFEnsemble with Config
- [ ] **31.1** Create test `tests/unit/models/test_ensemble_with_config.py`
- [ ] **31.2** Write test that creates ensemble with custom config
- [ ] **31.3** Verify all 3 models use config hyperparameters
- [ ] **31.4** Run: `pytest tests/unit/models/test_ensemble_with_config.py -v`

**Verify:** All tests pass
**Estimated time:** 1.5 hours

---

### Phase 3.3: Update Training Scripts (Day 14-15)

#### Step 32: Update train_mtf_ensemble.py
- [ ] **32.1** Backup `scripts/train_mtf_ensemble.py`
- [ ] **32.2** Add import: `from src.config import TradingConfig`
- [ ] **32.3** Load config at start of `main()`:
```python
def main(args):
    config = TradingConfig()
    # ... rest of training
```
- [ ] **32.4** Pass config to ensemble:
```python
ensemble = MTFEnsemble(config=config)
```
- [ ] **32.5** Add option to override hyperparameters:
```python
if args.custom_hyperparams:
    config.hyperparameters.model_1h.n_estimators = args.n_estimators_1h
    # ... etc
```
- [ ] **32.6** Save file

**Verify:** Script runs successfully
**Estimated time:** 2 hours

#### Step 33: Update walk_forward_optimization.py
- [ ] **33.1** Backup `scripts/walk_forward_optimization.py`
- [ ] **33.2** Add TradingConfig import
- [ ] **33.3** Load config in WFO function
- [ ] **33.4** Pass config to ensemble creation
- [ ] **33.5** Save file

**Verify:** WFO script runs
**Estimated time:** 1 hour

#### Step 34: Test Training Pipeline
- [ ] **34.1** Create small test dataset
- [ ] **34.2** Run training with default config
- [ ] **34.3** Run training with custom config
- [ ] **34.4** Verify models trained with correct hyperparameters

**Verify:** Training completes successfully
**Estimated time:** 2 hours

#### Step 35: Commit Week 3
- [ ] **35.1** Run all tests: `pytest tests/unit/models/ -v`
- [ ] **35.2** Stage changes: `git add src/models/ scripts/ tests/`
- [ ] **35.3** Commit:
```bash
git commit -m "feat: Centralize all XGBoost hyperparameters

- Update ImprovedTimeframeModel to load hyperparams from config
- Update factory methods (create_1h_model, create_4h_model, create_daily_model)
- Update MTFEnsemble to pass config to all models
- Update training scripts to use centralized config
- Add comprehensive tests for model configuration

Total: 30 hyperparameters centralized (1H, 4H, Daily)
Part of configuration centralization project (Week 3/6)"
```
- [ ] **35.4** Push: `git push origin feature/centralized-config`

**✅ CHECKPOINT 6:** Week 3 complete - All hyperparameters centralized

---

## WEEK 4: FEATURES & TRAINING (Days 16-20)

### Phase 4.1: Feature Engineering (Day 16-18)

#### Step 36: Update Enhanced Features - Lag Features
- [ ] **36.1** Backup `src/models/multi_timeframe/enhanced_features.py`
- [ ] **36.2** Add TradingConfig import
- [ ] **36.3** Update `EnhancedFeatureEngine.__init__()`:
```python
def __init__(self, config: Optional[TradingConfig] = None):
    self.config = config or TradingConfig()
```
- [ ] **36.4** Update `add_lag_features()`:
```python
for lag in self.config.features.lags.standard_lags:
    # ... create lag features
```
- [ ] **36.5** Save file

**Verify:** Lag features use config
**Estimated time:** 1 hour

#### Step 37: Update ROC Features
- [ ] **37.1** Update `add_roc_features()` to use config:
```python
for period in self.config.features.lags.rsi_roc_periods:
    df[f"rsi_roc_{period}"] = df["rsi_14"].pct_change(period)
```
- [ ] **37.2** Update MACD ROC with config
- [ ] **37.3** Update ATR ROC with config
- [ ] **37.4** Update Price ROC with config
- [ ] **37.5** Save file

**Verify:** ROC features use config
**Estimated time:** 1 hour

#### Step 38: Update Session Features
- [ ] **38.1** Update `add_session_features()` to use config:
```python
asian_start, asian_end = self.config.features.sessions.asian_session
london_start, london_end = self.config.features.sessions.london_session
# ... etc
```
- [ ] **38.2** Save file

**Verify:** Session features use config
**Estimated time:** 30 minutes

#### Step 39: Update Cyclical Features
- [ ] **39.1** Update `add_cyclical_features()` to use config:
```python
hour_cycles = self.config.features.cyclical.hour_encoding_cycles
dow_cycles = self.config.features.cyclical.day_of_week_cycles
dom_cycles = self.config.features.cyclical.day_of_month_cycles
```
- [ ] **39.2** Save file

**Verify:** Cyclical features use config
**Estimated time:** 30 minutes

#### Step 40: Test Feature Engineering
- [ ] **40.1** Create test `tests/unit/features/test_enhanced_features_config.py`
- [ ] **40.2** Write test for lag features with config
- [ ] **40.3** Write test for ROC features with config
- [ ] **40.4** Write test for session features with config
- [ ] **40.5** Write test for cyclical features with config
- [ ] **40.6** Run: `pytest tests/unit/features/test_enhanced_features_config.py -v`

**Verify:** All tests pass
**Estimated time:** 2 hours

---

### Phase 4.2: Training Parameters (Day 19-20)

#### Step 41: Update Training Splits
- [ ] **41.1** Open `src/models/multi_timeframe/mtf_ensemble.py`
- [ ] **41.2** Find `train()` method
- [ ] **41.3** Update data splits to use config:
```python
train_ratio = self.config.training.splits.train_ratio
val_ratio = self.config.training.splits.validation_ratio

train_end = int(n * train_ratio)
val_end = int(n * (train_ratio + val_ratio))
```
- [ ] **41.4** Save file

**Verify:** Training splits use config
**Estimated time:** 1 hour

#### Step 42: Update Early Stopping
- [ ] **42.1** Update model training to use early stopping config:
```python
if self.config.training.early_stopping.enabled:
    eval_set = [(X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        early_stopping_rounds=self.config.training.early_stopping.stopping_rounds,
        verbose=self.config.training.early_stopping.verbose
    )
```
- [ ] **42.2** Save file

**Verify:** Early stopping uses config
**Estimated time:** 1 hour

#### Step 43: Update Stacking Meta-Learner
- [ ] **43.1** Backup `src/models/multi_timeframe/stacking_meta_learner.py`
- [ ] **43.2** Add TradingConfig import
- [ ] **43.3** Update `__init__()` to accept config
- [ ] **43.4** Update `generate_oof_predictions()` to use config CV settings:
```python
n_folds = self.config.training.stacking.n_folds
min_train_size = self.config.training.stacking.min_train_size
```
- [ ] **43.5** Save file

**Verify:** Stacking uses config
**Estimated time:** 1 hour

#### Step 44: Test Training Parameters
- [ ] **44.1** Create test `tests/unit/training/test_training_config.py`
- [ ] **44.2** Write test for data splits with config
- [ ] **44.3** Write test for early stopping with config
- [ ] **44.4** Write test for stacking with config
- [ ] **44.5** Run: `pytest tests/unit/training/test_training_config.py -v`

**Verify:** All tests pass
**Estimated time:** 2 hours

#### Step 45: Commit Week 4
- [ ] **45.1** Run all tests: `pytest tests/ -v`
- [ ] **45.2** Stage changes: `git add src/ tests/`
- [ ] **45.3** Commit:
```bash
git commit -m "feat: Centralize feature engineering and training parameters

- Update EnhancedFeatureEngine to use config for lags (5 params)
- Update session features to use config (4 params)
- Update cyclical encoding to use config (3 params)
- Update training splits to use config (3 params)
- Update early stopping to use config (3 params)
- Update stacking meta-learner to use config (4 params)
- Add comprehensive tests

Total: 22 parameters centralized
Part of configuration centralization project (Week 4/6)"
```
- [ ] **45.4** Push: `git push origin feature/centralized-config`

**✅ CHECKPOINT 7:** Week 4 complete - All features & training params centralized

---

## WEEK 5: TESTING & DEPLOYMENT (Days 21-26)

### Phase 5.1: Comprehensive Testing (Day 21-23)

#### Step 46: Integration Tests
- [ ] **46.1** Create test `tests/integration/test_full_pipeline_with_config.py`
- [ ] **46.2** Write test: Load data → Calculate indicators → Train model → Predict
- [ ] **46.3** Verify all steps use centralized config
- [ ] **46.4** Write test with custom config overrides
- [ ] **46.5** Run: `pytest tests/integration/test_full_pipeline_with_config.py -v`

**Verify:** End-to-end pipeline works with config
**Estimated time:** 3 hours

#### Step 47: Performance Tests
- [ ] **47.1** Create test `tests/performance/test_config_performance.py`
- [ ] **47.2** Test config loading time (<10ms)
- [ ] **47.3** Test config singleton performance
- [ ] **47.4** Test memory usage with config
- [ ] **47.5** Run: `pytest tests/performance/ -v`

**Verify:** Performance acceptable
**Estimated time:** 2 hours

#### Step 48: Backward Compatibility Tests
- [ ] **48.1** Create test `tests/integration/test_backward_compatibility.py`
- [ ] **48.2** Test that old code patterns still work
- [ ] **48.3** Test migration path from hardcoded to config
- [ ] **48.4** Run: `pytest tests/integration/test_backward_compatibility.py -v`

**Verify:** Backward compatibility maintained
**Estimated time:** 2 hours

#### Step 49: Run Full Test Suite
- [ ] **49.1** Run all unit tests: `pytest tests/unit/ -v --cov`
- [ ] **49.2** Run all integration tests: `pytest tests/integration/ -v`
- [ ] **49.3** Generate coverage report: `pytest --cov=src --cov-report=html`
- [ ] **49.4** Review coverage (target: 95%+)

**Verify:** All tests pass, coverage >95%
**Estimated time:** 2 hours

---

### Phase 5.2: Documentation (Day 24)

#### Step 50: Update Configuration Guide
- [ ] **50.1** Open `docs/CONFIGURATION_GUIDE.md`
- [ ] **50.2** Add complete parameter reference
- [ ] **50.3** Add usage examples for each section
- [ ] **50.4** Add migration guide from hardcoded values
- [ ] **50.5** Add troubleshooting section
- [ ] **50.6** Save file

**Verify:** Documentation complete
**Estimated time:** 4 hours

#### Step 51: Update README
- [ ] **51.1** Open `README.md` or main documentation
- [ ] **51.2** Add section on centralized configuration
- [ ] **51.3** Link to CONFIGURATION_GUIDE.md
- [ ] **51.4** Update examples to show config usage

**Verify:** README updated
**Estimated time:** 1 hour

#### Step 52: Create Migration Guide
- [ ] **52.1** Create `docs/MIGRATION_TO_CENTRALIZED_CONFIG.md`
- [ ] **52.2** Document what changed
- [ ] **52.3** Provide before/after code examples
- [ ] **52.4** List breaking changes (if any)
- [ ] **52.5** Provide rollback instructions

**Verify:** Migration guide complete
**Estimated time:** 2 hours

---

### Phase 5.3: Staging Deployment (Day 25-26)

#### Step 53: Prepare Staging Environment
- [ ] **53.1** Merge feature branch to staging: `git checkout staging && git merge feature/centralized-config`
- [ ] **53.2** Run tests in staging: `pytest tests/ -v`
- [ ] **53.3** Build Docker image: `docker build -t ai-trader:staging .`

**Verify:** Staging build successful
**Estimated time:** 1 hour

#### Step 54: Deploy to Staging
- [ ] **54.1** Stop staging services: `docker-compose -f docker-compose.staging.yml down`
- [ ] **54.2** Deploy new version: `docker-compose -f docker-compose.staging.yml up -d`
- [ ] **54.3** Check logs: `docker-compose -f docker-compose.staging.yml logs -f`
- [ ] **54.4** Verify config loads: Check logs for "TradingConfig loaded"

**Verify:** Staging deployment successful
**Estimated time:** 1 hour

#### Step 55: Test in Staging
- [ ] **55.1** Test API health: `curl http://staging:8001/health`
- [ ] **55.2** Test indicator calculation with default config
- [ ] **55.3** Test indicator calculation with custom config
- [ ] **55.4** Test model prediction with default config
- [ ] **55.5** Test model prediction with custom hyperparameters
- [ ] **55.6** Monitor for 24 hours

**Verify:** All staging tests pass
**Estimated time:** 4 hours + monitoring

#### Step 56: Final Pre-Production Checklist
- [ ] **56.1** All 89 steps completed
- [ ] **56.2** All tests passing (100+ unit, 20+ integration)
- [ ] **56.3** Test coverage >95%
- [ ] **56.4** Documentation complete
- [ ] **56.5** Staging stable for 24 hours
- [ ] **56.6** Performance acceptable (<10ms config load)
- [ ] **56.7** No config-related errors in logs

**Verify:** Ready for production
**Estimated time:** Review

#### Step 57: Commit Week 5
- [ ] **57.1** Stage all final changes: `git add .`
- [ ] **57.2** Commit:
```bash
git commit -m "feat: Complete configuration centralization testing and deployment

- Add 100+ unit tests for all config sections
- Add 20+ integration tests for full pipeline
- Add performance tests (<10ms config load verified)
- Add backward compatibility tests
- Complete documentation (Configuration Guide, Migration Guide)
- Deploy and verify in staging environment

Configuration centralization project: COMPLETE
- Week 1: Infrastructure ✅
- Week 2: Indicators (30 params) ✅
- Week 3: Hyperparameters (30 params) ✅
- Week 4: Features & Training (22 params) ✅
- Week 5: Testing & Deployment ✅

Total: 82 parameters centralized (from 76 hardcoded)
Test coverage: 95%+
All systems operational in staging"
```
- [ ] **57.3** Push: `git push origin feature/centralized-config`
- [ ] **57.4** Create PR to main: "Configuration Centralization - 82 params centralized"

**✅ CHECKPOINT 8:** Week 5 complete - Ready for production deployment

---

## WEEK 6: PRODUCTION & MONITORING (Days 27-30)

### Phase 6.1: Production Deployment (Day 27-28)

#### Step 58: Production Deployment
- [ ] **58.1** Get approval for production deployment
- [ ] **58.2** Schedule maintenance window
- [ ] **58.3** Backup current production database
- [ ] **58.4** Merge to main: `git checkout main && git merge feature/centralized-config`
- [ ] **58.5** Tag release: `git tag -a v2.0.0 -m "Configuration centralization complete"`
- [ ] **58.6** Deploy to production: `./deploy-production.sh`
- [ ] **58.7** Monitor logs for 2 hours

**Verify:** Production deployment successful
**Estimated time:** 4 hours

#### Step 59: Post-Deployment Verification
- [ ] **59.1** Verify API health: `curl https://api.prod/health`
- [ ] **59.2** Verify config loaded correctly (check logs)
- [ ] **59.3** Verify predictions working
- [ ] **59.4** Compare predictions with previous version (should be identical)
- [ ] **59.5** Monitor for any errors

**Verify:** Production stable
**Estimated time:** 2 hours

---

### Phase 6.2: Monitoring Setup (Day 29)

#### Step 60: Add Config Change Logging
- [ ] **60.1** Implement config change tracking in TradingConfig
- [ ] **60.2** Log all config updates to database
- [ ] **60.3** Add alerting for critical config changes
- [ ] **60.4** Test logging works

**Verify:** Config changes are logged
**Estimated time:** 2 hours

#### Step 61: Setup Monitoring Dashboard
- [ ] **61.1** Add config metrics to monitoring dashboard
- [ ] **61.2** Track config load time
- [ ] **61.3** Track config change frequency
- [ ] **61.4** Setup alerts for config errors

**Verify:** Monitoring active
**Estimated time:** 3 hours

---

### Phase 6.3: Optimization Framework (Day 30)

#### Step 62: Create Optimization Scripts
- [ ] **62.1** Create `scripts/optimize_indicators.py`
- [ ] **62.2** Create `scripts/optimize_hyperparameters.py`
- [ ] **62.3** Add grid search capability
- [ ] **62.4** Add Optuna integration
- [ ] **62.5** Test optimization scripts

**Verify:** Optimization framework works
**Estimated time:** 4 hours

#### Step 63: Final Documentation Update
- [ ] **63.1** Document optimization framework
- [ ] **63.2** Update CONFIGURATION_GUIDE.md with optimization examples
- [ ] **63.3** Create HOW_TO_OPTIMIZE.md guide

**Verify:** Optimization documented
**Estimated time:** 2 hours

---

## FINAL CHECKLIST

### Success Criteria
- [ ] **✅ All 76 hardcoded parameters centralized**
- [ ] **✅ Test coverage >95%**
- [ ] **✅ Config load time <10ms**
- [ ] **✅ Hot-reload working**
- [ ] **✅ Production stable for 1 week**
- [ ] **✅ Zero config-related bugs**
- [ ] **✅ Documentation complete**
- [ ] **✅ Team trained on new system**

### Final Commits
- [ ] **64.1** Create final summary report
- [ ] **64.2** Update CLAUDE.md with new config system info
- [ ] **64.3** Tag final release: `git tag -a v2.0.0-stable`
- [ ] **64.4** Announce completion to team

**✅ PROJECT COMPLETE:** Configuration centralization successful!

---

## Progress Tracking

**Track your progress:**
```bash
# Count completed steps
grep -c "\[x\]" CONFIGURATION_CENTRALIZATION_CHECKLIST.md

# Calculate percentage
echo "scale=2; $(grep -c '\[x\]' CONFIGURATION_CENTRALIZATION_CHECKLIST.md) * 100 / 64" | bc
```

**Backup strategy:**
- Commit after each phase
- Keep `.backup` files until phase complete
- Test before deleting backups

**Need help?** Reference the detailed implementation plan:
`CONFIGURATION_CENTRALIZATION_IMPLEMENTATION_PLAN.md`

---

**Status:** 0/89 steps complete (0%)
**Estimated remaining:** 240 hours (6 weeks)
**Next step:** Step 1.1 - Create indicator config file
