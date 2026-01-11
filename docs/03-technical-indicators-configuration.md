# Technical Indicators Configuration Specification

## 1. Overview

This document specifies the technical indicators configuration for each model (Short-Term, Medium-Term, Long-Term). Technical indicators are **configurable per model** to allow flexibility in testing different indicator combinations and optimizing performance for each time horizon.

## 2. Configuration Philosophy

### 2.1 Timeframe-Appropriate Indicators

Different timeframes require different indicator settings:

| Timeframe | Characteristic | Indicator Focus |
|-----------|----------------|-----------------|
| **Short-Term** (1H-4H) | High noise, quick reversals | Fast oscillators, short MAs, volume-based |
| **Medium-Term** (Daily) | Balanced signals, swing moves | Standard periods, trend confirmation |
| **Long-Term** (Weekly) | Major trends, regime changes | Slow indicators, trend strength, breakouts |

### 2.2 Why Configurable Indicators?

- **Experimentation**: Test impact of different indicator combinations
- **Optimization**: Find optimal indicators for specific market conditions
- **Regime Adaptation**: Different regimes may favor different indicators
- **Model Specialization**: Each model can use indicators suited to its timeframe

## 3. Indicator Configuration Structure

### 3.1 YAML Configuration Format

```yaml
indicators:
  # Enable/disable entire categories
  enabled_categories:
    - trend
    - momentum
    - volatility
    - volume

  # Trend indicators
  trend:
    sma:
      enabled: true
      periods: [10, 20, 50]
    ema:
      enabled: true
      periods: [8, 21]
    # ... more indicators

  # Momentum indicators
  momentum:
    rsi:
      enabled: true
      periods: [7, 14]
    # ... more indicators

  # Volatility indicators
  volatility:
    atr:
      enabled: true
      period: 14
    # ... more indicators

  # Volume indicators
  volume:
    obv:
      enabled: true
    # ... more indicators

  # Derived/engineered features
  derived:
    price_to_ma: true
    ma_crossovers: true
    divergences: false
```

## 4. Short-Term Model Indicators (Intraday)

### 4.1 Rationale

Short-term (1H-4H) trading requires:
- **Fast response**: Quick signals to catch intraday moves
- **Noise filtering**: Short-term data is noisier
- **Overbought/oversold**: Quick reversals are common
- **Volume importance**: Intraday volume patterns matter

### 4.2 Recommended Configuration

```yaml
# configs/indicators/short_term_indicators.yaml
indicators:
  enabled_categories:
    - trend
    - momentum
    - volatility
    - volume

  trend:
    # Fast EMAs for quick trend detection
    ema:
      enabled: true
      periods: [8, 13, 21, 55]
      use_crossovers: true
    sma:
      enabled: true
      periods: [20, 50]
    # Supertrend for trend direction
    supertrend:
      enabled: true
      period: 10
      multiplier: 3

  momentum:
    # RSI with multiple fast periods
    rsi:
      enabled: true
      periods: [7, 14]
      overbought: 70
      oversold: 30
    # Fast Stochastic for quick reversals
    stochastic:
      enabled: true
      k_period: 5
      d_period: 3
      smooth_k: 3
    # Standard MACD
    macd:
      enabled: true
      fast: 12
      slow: 26
      signal: 9
    # Williams %R for overbought/oversold
    williams_r:
      enabled: true
      period: 14
    # Momentum for quick changes
    momentum:
      enabled: true
      period: 10
    # MFI combines price and volume
    mfi:
      enabled: true
      period: 14

  volatility:
    # ATR for position sizing
    atr:
      enabled: true
      period: 14
      normalized: true
    # Bollinger for volatility squeeze
    bollinger:
      enabled: true
      period: 20
      std_dev: 2
      include_width: true
      include_percent_b: true
    # Keltner for alternative bands
    keltner:
      enabled: false  # Optional, may add noise

  volume:
    # OBV for volume trend
    obv:
      enabled: true
    # VWAP critical for intraday
    vwap:
      enabled: true
    # Force Index for volume-weighted momentum
    force_index:
      enabled: true
      period: 13

  derived:
    price_to_ma:
      enabled: true
      ma_types: ['ema']
      periods: [21, 55]
    ma_crossovers:
      enabled: true
      pairs:
        - [8, 21]
        - [13, 55]
    rsi_divergence:
      enabled: true
      period: 14
```

### 4.3 Short-Term Feature Count

| Category | Indicators | Approximate Features |
|----------|-----------|---------------------|
| Trend | EMA(4), SMA(2), Supertrend | ~10 |
| Momentum | RSI(2), Stoch(2), MACD(3), WillR, Mom, MFI | ~12 |
| Volatility | ATR(2), BB(5) | ~7 |
| Volume | OBV, VWAP, FI | ~3 |
| Derived | Price-to-MA(2), Crossovers(2), RSI Div | ~5 |
| **Total** | | **~37 features** |

## 5. Medium-Term Model Indicators (Swing)

### 5.1 Rationale

Medium-term (Daily) trading requires:
- **Balanced signals**: Not too fast, not too slow
- **Trend confirmation**: Multiple indicators confirming direction
- **Support/resistance**: Key levels matter more
- **Volume confirmation**: Daily volume validates moves

### 5.2 Recommended Configuration

```yaml
# configs/indicators/medium_term_indicators.yaml
indicators:
  enabled_categories:
    - trend
    - momentum
    - volatility
    - volume

  trend:
    # Standard MA periods for swing trading
    sma:
      enabled: true
      periods: [20, 50, 100, 200]
      use_crossovers: true
    ema:
      enabled: true
      periods: [12, 26, 50]
    # ADX for trend strength (critical for swing)
    adx:
      enabled: true
      period: 14
      include_di: true  # +DI and -DI
    # Aroon for trend identification
    aroon:
      enabled: true
      period: 25
      include_oscillator: true
    # PSAR for trend direction
    parabolic_sar:
      enabled: true

  momentum:
    # Standard RSI
    rsi:
      enabled: true
      periods: [14]
      overbought: 70
      oversold: 30
    # Standard Stochastic
    stochastic:
      enabled: true
      k_period: 14
      d_period: 3
      smooth_k: 3
    # MACD critical for swing
    macd:
      enabled: true
      fast: 12
      slow: 26
      signal: 9
    # CCI for cyclical patterns
    cci:
      enabled: true
      period: 20
    # ROC for rate of change
    roc:
      enabled: true
      period: 14
    # MFI
    mfi:
      enabled: true
      period: 14
    # TSI for smoothed momentum
    tsi:
      enabled: true
      fast: 13
      slow: 25

  volatility:
    # ATR for stops and position sizing
    atr:
      enabled: true
      period: 14
      normalized: true
    # Bollinger for volatility regime
    bollinger:
      enabled: true
      period: 20
      std_dev: 2
      include_width: true
      include_percent_b: true
    # Keltner for squeeze detection
    keltner:
      enabled: true
      period: 20
      multiplier: 2
    # Historical volatility
    historical_volatility:
      enabled: true
      period: 20

  volume:
    # OBV for accumulation/distribution
    obv:
      enabled: true
    # Chaikin Money Flow
    cmf:
      enabled: true
      period: 20
    # A/D line
    ad_line:
      enabled: true
    # Volume Price Trend
    vpt:
      enabled: true

  derived:
    price_to_ma:
      enabled: true
      ma_types: ['sma', 'ema']
      periods: [20, 50, 200]
    ma_crossovers:
      enabled: true
      pairs:
        - [20, 50]
        - [50, 200]
        - [12, 26]  # EMA
    macd_divergence:
      enabled: true
    rsi_divergence:
      enabled: true
      period: 14
    support_resistance:
      enabled: true
      lookback: 20
```

### 5.3 Medium-Term Feature Count

| Category | Indicators | Approximate Features |
|----------|-----------|---------------------|
| Trend | SMA(4), EMA(3), ADX(3), Aroon(3), PSAR | ~15 |
| Momentum | RSI, Stoch(2), MACD(3), CCI, ROC, MFI, TSI | ~11 |
| Volatility | ATR(2), BB(5), KC(3), HV | ~11 |
| Volume | OBV, CMF, AD, VPT | ~4 |
| Derived | Price-to-MA(6), Crossovers(3), Divs(2), S/R | ~13 |
| **Total** | | **~54 features** |

## 6. Long-Term Model Indicators (Position)

### 6.1 Rationale

Long-term (Weekly) trading requires:
- **Major trends**: Focus on significant moves only
- **Regime detection**: Identify market phases
- **Breakouts**: Weekly breakouts are significant
- **Low noise**: Filter out short-term fluctuations

### 6.2 Recommended Configuration

```yaml
# configs/indicators/long_term_indicators.yaml
indicators:
  enabled_categories:
    - trend
    - momentum
    - volatility

  trend:
    # Major MAs only
    sma:
      enabled: true
      periods: [10, 20, 50]  # In weekly = 10w, 20w, 50w (~1yr)
      use_crossovers: true
    ema:
      enabled: true
      periods: [12, 26]
    # ADX critical for position trading
    adx:
      enabled: true
      period: 14
      include_di: true
    # Aroon for long-term trend changes
    aroon:
      enabled: true
      period: 25
      include_oscillator: true
    # Ichimoku for long-term support/resistance
    ichimoku:
      enabled: true
      tenkan: 9
      kijun: 26
      senkou_b: 52
    # PSAR
    parabolic_sar:
      enabled: true

  momentum:
    # RSI for weekly divergences
    rsi:
      enabled: true
      periods: [14]
      overbought: 70
      oversold: 30
    # Weekly MACD is highly reliable
    macd:
      enabled: true
      fast: 12
      slow: 26
      signal: 9
    # ROC for long-term momentum
    roc:
      enabled: true
      periods: [10, 20]
    # Ultimate Oscillator combines timeframes
    ultimate_oscillator:
      enabled: true
      period1: 7
      period2: 14
      period3: 28

  volatility:
    # ATR for major moves
    atr:
      enabled: true
      period: 14
      normalized: true
    # Bollinger for regime
    bollinger:
      enabled: true
      period: 20
      std_dev: 2
      include_width: true
    # Donchian for breakouts
    donchian:
      enabled: true
      period: 20
    # Historical volatility for regime
    historical_volatility:
      enabled: true
      periods: [10, 20]

  volume:
    # OBV only for weekly
    obv:
      enabled: true

  derived:
    price_to_ma:
      enabled: true
      ma_types: ['sma']
      periods: [20, 50]
    ma_crossovers:
      enabled: true
      pairs:
        - [10, 20]
        - [20, 50]
    regime_detection:
      enabled: true
      volatility_percentile: true
      trend_strength: true
```

### 6.3 Long-Term Feature Count

| Category | Indicators | Approximate Features |
|----------|-----------|---------------------|
| Trend | SMA(3), EMA(2), ADX(3), Aroon(3), Ichimoku(5), PSAR | ~18 |
| Momentum | RSI, MACD(3), ROC(2), UO | ~7 |
| Volatility | ATR(2), BB(4), DC(2), HV(2) | ~10 |
| Volume | OBV | ~1 |
| Derived | Price-to-MA(2), Crossovers(2), Regime(2) | ~6 |
| **Total** | | **~42 features** |

## 7. Indicator Priority Matrix

Based on trading research and quantitative analysis, indicators are ranked by importance for each timeframe:

### 7.1 Priority Levels

| Priority | Description | Action if Resource Constrained |
|----------|-------------|-------------------------------|
| **P0 - Critical** | Must have, core functionality | Never disable |
| **P1 - Important** | Significantly improves accuracy | Disable last |
| **P2 - Useful** | Adds value, can be omitted | Disable if needed |
| **P3 - Optional** | Nice to have | Disable first |

### 7.2 Short-Term Priority

| Indicator | Priority | Reason |
|-----------|----------|--------|
| EMA (8, 21) | P0 | Fast trend detection |
| RSI (7, 14) | P0 | Overbought/oversold critical |
| MACD | P0 | Momentum crossovers |
| ATR | P0 | Position sizing, stops |
| Bollinger | P1 | Volatility squeeze |
| Stochastic | P1 | Quick reversals |
| VWAP | P1 | Intraday value |
| Williams %R | P2 | Additional oversold/overbought |
| MFI | P2 | Volume-price relationship |
| Force Index | P3 | Volume confirmation |

### 7.3 Medium-Term Priority

| Indicator | Priority | Reason |
|-----------|----------|--------|
| SMA (20, 50, 200) | P0 | Major trend levels |
| MACD | P0 | Swing trade signals |
| RSI | P0 | Divergences, extremes |
| ADX | P0 | Trend strength |
| ATR | P0 | Risk management |
| Bollinger | P1 | Volatility regime |
| CCI | P1 | Cyclical patterns |
| OBV | P1 | Volume confirmation |
| Aroon | P2 | Trend identification |
| TSI | P2 | Smoothed momentum |
| CMF | P3 | Money flow |

### 7.4 Long-Term Priority

| Indicator | Priority | Reason |
|-----------|----------|--------|
| SMA (20, 50) | P0 | Major weekly trends |
| MACD | P0 | Weekly signals reliable |
| ADX | P0 | Trend strength |
| ATR | P0 | Major moves |
| RSI | P1 | Weekly divergences |
| Aroon | P1 | Trend changes |
| Donchian | P1 | Breakout detection |
| Ichimoku | P2 | Long-term S/R |
| ROC | P2 | Long-term momentum |
| Ultimate Oscillator | P3 | Multi-timeframe |

## 8. Configuration API

### 8.1 Loading Configuration

```python
from src.config import IndicatorConfig

# Load model-specific configuration
short_config = IndicatorConfig.load('configs/indicators/short_term_indicators.yaml')
medium_config = IndicatorConfig.load('configs/indicators/medium_term_indicators.yaml')
long_config = IndicatorConfig.load('configs/indicators/long_term_indicators.yaml')

# Use with TechnicalIndicators
from src.features.technical import TechnicalIndicators

indicators = TechnicalIndicators(config=short_config)
df_features = indicators.calculate_all(df)
```

### 8.2 Overriding Configuration

```python
# Override specific indicators
config = IndicatorConfig.load('configs/indicators/short_term_indicators.yaml')
config.indicators.momentum.rsi.periods = [5, 10, 14]  # Custom RSI periods
config.indicators.trend.ema.enabled = False  # Disable EMA

# Or via YAML override
config.merge_from_file('configs/experiments/no_ema_experiment.yaml')
```

### 8.3 Feature Selection

```python
# Get list of enabled features
features = config.get_enabled_features()
# ['ema_8', 'ema_21', 'rsi_7', 'rsi_14', ...]

# Get feature count
total_features = config.get_feature_count()
# 37

# Validate configuration
config.validate()  # Raises if invalid
```

## 9. Experiment Tracking

### 9.1 MLflow Integration

```python
import mlflow

with mlflow.start_run():
    # Log indicator configuration
    mlflow.log_artifact('configs/indicators/short_term_indicators.yaml')
    mlflow.log_params({
        'indicator_config': 'short_term_v1',
        'total_features': config.get_feature_count(),
        'enabled_categories': config.enabled_categories,
    })

    # Train model
    model.train(X, y)

    # Log metrics
    mlflow.log_metrics(metrics)
```

### 9.2 A/B Testing Indicators

```yaml
# configs/experiments/rsi_period_test.yaml
experiment:
  name: "RSI Period Comparison"
  variants:
    - name: "rsi_7_14"
      indicators:
        momentum:
          rsi:
            periods: [7, 14]
    - name: "rsi_9_21"
      indicators:
        momentum:
          rsi:
            periods: [9, 21]
    - name: "rsi_14_only"
      indicators:
        momentum:
          rsi:
            periods: [14]
```

## 10. Best Practices

### 10.1 Avoiding Redundancy

- Don't use both RSI and Stochastic RSI (highly correlated)
- Don't use all MA types (SMA, EMA, WMA) for same periods
- Pick either Bollinger or Keltner, not both (unless for squeeze)

### 10.2 Feature Correlation

Monitor correlation between features:
```python
# Check correlation matrix
corr_matrix = df_features.corr()
high_corr = (corr_matrix.abs() > 0.95) & (corr_matrix != 1.0)
redundant_features = high_corr.any()
```

### 10.3 Lookback Considerations

Ensure sufficient history for all indicators:
```python
# Minimum history required
short_term_min_history = 200  # For SMA(50) + warmup
medium_term_min_history = 250  # For SMA(200) + warmup
long_term_min_history = 100   # For SMA(50) weekly + warmup
```

## 11. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-07 | Initial specification |

---

*Document Version: 1.0*
*Last Updated: 2025-01-07*
*Author: AI Trader Development Team*
