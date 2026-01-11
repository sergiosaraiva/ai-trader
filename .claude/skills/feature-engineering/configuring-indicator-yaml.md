---
name: configuring-indicator-yaml
description: Configures technical indicator YAML files for model-specific feature sets with priority levels. Use when customizing which indicators to include for short/medium/long-term models. YAML configuration format.
---

# Configuring Indicator YAML

## Quick Reference

- Config files in `configs/indicators/[model_type]_indicators.yaml`
- Use priority levels: P0 (critical), P1 (important), P2 (useful), P3 (optional)
- Set `enabled: true/false` to include/exclude indicators
- Group by category: trend, momentum, volatility, volume, derived
- Reference from model config: `indicators_config: configs/indicators/xxx.yaml`

## When to Use

- Customizing indicator sets for different timeframes
- Enabling/disabling indicators for feature selection experiments
- Adjusting indicator parameters (periods, multipliers)
- Creating new model-specific indicator configurations
- Documenting which indicators are critical vs optional

## When NOT to Use

- Adding new indicator calculations (modify Python code)
- One-off indicator experiments (use code directly)
- Real-time indicator parameter tuning (use model config)

## Implementation Guide with Decision Tree

```
Which model type?
├─ Short-term (1H-4H) → short_term_indicators.yaml
│   └─ Focus: Fast EMAs, quick oscillators, VWAP
├─ Medium-term (Daily) → medium_term_indicators.yaml
│   └─ Focus: Balanced MAs, standard RSI/MACD, volume confirmation
└─ Long-term (Weekly) → long_term_indicators.yaml
    └─ Focus: Major MAs, regime indicators, Ichimoku

What priority level?
├─ P0 (Critical) → Must have: RSI, MACD, ATR, key MAs
├─ P1 (Important) → Significantly improves accuracy
├─ P2 (Useful) → Adds value, can disable if needed
└─ P3 (Optional) → Nice to have, disable first when reducing features
```

## Examples

**Example 1: Full Configuration File Structure**

```yaml
# From: configs/indicators/short_term_indicators.yaml:1-30
# Short-Term Model (Intraday) - Technical Indicators Configuration
# Timeframe: 1H-4H candles
# Focus: Fast signals, quick reversals, intraday patterns
# Approximate feature count: ~37

version: "1.0"
model_type: short_term

indicators:
  enabled_categories:
    - trend
    - momentum
    - volatility
    - volume
```

**Explanation**: Header comments document purpose, timeframe, focus, and approximate feature count. `enabled_categories` controls which groups are calculated.

**Example 2: Trend Indicators Configuration**

```yaml
# From: configs/indicators/short_term_indicators.yaml:16-53
  # =============================================================================
  # TREND INDICATORS
  # Focus on fast EMAs for quick trend detection
  # =============================================================================
  trend:
    ema:
      enabled: true
      periods: [8, 13, 21, 55]
      priority: P0  # Critical for short-term trend

    sma:
      enabled: true
      periods: [20, 50]
      priority: P1

    supertrend:
      enabled: true
      period: 10
      multiplier: 3.0
      priority: P1

    # ADX less useful for very short-term
    adx:
      enabled: false
      period: 14
      include_di: true
      priority: P2

    aroon:
      enabled: false
      period: 25
      include_oscillator: true
      priority: P3
```

**Explanation**: Each indicator has `enabled`, parameters, and `priority`. Comments explain why indicators are enabled/disabled for this timeframe.

**Example 3: Momentum Indicators Configuration**

```yaml
# From: configs/indicators/short_term_indicators.yaml:54-117
  # =============================================================================
  # MOMENTUM INDICATORS
  # Focus on fast oscillators for quick reversal detection
  # =============================================================================
  momentum:
    rsi:
      enabled: true
      periods: [7, 14]
      overbought: 70
      oversold: 30
      priority: P0  # Critical

    stochastic:
      enabled: true
      k_period: 5
      d_period: 3
      smooth_k: 3
      priority: P1

    macd:
      enabled: true
      fast_period: 12
      slow_period: 26
      signal_period: 9
      priority: P0  # Critical

    williams_r:
      enabled: true
      period: 14
      priority: P2

    momentum:
      enabled: true
      period: 10
      priority: P2

    mfi:
      enabled: true
      period: 14
      priority: P2

    # Optional indicators - disable first if reducing features
    roc:
      enabled: false
      period: 10
      priority: P3

    cci:
      enabled: false
      period: 20
      priority: P3
```

**Explanation**: P0 indicators (RSI, MACD) always enabled. P3 indicators disabled by default but available.

**Example 4: Volatility Indicators Configuration**

```yaml
# From: configs/indicators/short_term_indicators.yaml:118-152
  # =============================================================================
  # VOLATILITY INDICATORS
  # Focus on ATR for sizing and Bollinger for squeeze detection
  # =============================================================================
  volatility:
    atr:
      enabled: true
      period: 14
      normalized: true  # Include NATR
      priority: P0  # Critical for position sizing

    bollinger:
      enabled: true
      period: 20
      std_dev: 2.0
      include_width: true
      include_percent_b: true
      priority: P1

    keltner:
      enabled: false
      period: 20
      multiplier: 2.0
      priority: P3

    donchian:
      enabled: false
      period: 20
      priority: P3
```

**Explanation**: ATR is P0 because it's used for position sizing. Bollinger at P1 for volatility squeeze detection. Keltner/Donchian at P3 - optional.

**Example 5: Volume Indicators Configuration**

```yaml
# From: configs/indicators/short_term_indicators.yaml:153-183
  # =============================================================================
  # VOLUME INDICATORS
  # Focus on VWAP (critical for intraday) and OBV
  # =============================================================================
  volume:
    obv:
      enabled: true
      priority: P1

    vwap:
      enabled: true
      priority: P1  # Critical for intraday

    force_index:
      enabled: true
      period: 13
      priority: P2

    cmf:
      enabled: false
      period: 20
      priority: P3

    ad_line:
      enabled: false
      priority: P3
```

**Explanation**: VWAP critical for intraday trading (support/resistance). OBV for trend confirmation. CMF/AD at P3.

**Example 6: Derived Features Configuration**

```yaml
# From: configs/indicators/short_term_indicators.yaml:184-223
  # =============================================================================
  # DERIVED/ENGINEERED FEATURES
  # =============================================================================
  derived:
    price_to_ma:
      enabled: true
      ma_type: ema
      periods: [21, 55]

    ma_crossovers:
      enabled: true
      pairs:
        - fast: 8
          slow: 21
          type: ema
        - fast: 13
          slow: 55
          type: ema

    rsi_divergence:
      enabled: true
      rsi_period: 14
      lookback: 20

    macd_divergence:
      enabled: false

    support_resistance:
      enabled: false

# =============================================================================
# FEATURE SUMMARY
# =============================================================================
# Trend:      EMA(4) + SMA(2) + Supertrend    = ~10 features
# Momentum:   RSI(2) + Stoch(2) + MACD(3) + WillR + Mom + MFI = ~12 features
# Volatility: ATR(2) + BB(5)                   = ~7 features
# Volume:     OBV + VWAP + FI                  = ~3 features
# Derived:    Price-to-MA(2) + Crossovers(2) + RSI Div = ~5 features
# TOTAL:      ~37 features
```

**Explanation**: Derived features built from base indicators. Feature summary helps track total count.

**Example 7: Reference from Model Config**

```yaml
# From: configs/short_term.yaml:50-63
features:
  # Reference to indicator configuration file
  # See configs/indicators/short_term_indicators.yaml for full specification
  indicators_config: configs/indicators/short_term_indicators.yaml

  # Override specific indicators if needed (optional)
  # indicators_override:
  #   momentum:
  #     rsi:
  #       periods: [5, 10]

  temporal_features: true
  session_features: true
```

**Explanation**: Model config references indicator config file. Optional `indicators_override` for experiment-specific changes without modifying base config.

## Quality Checklist

- [ ] File has header comments: purpose, timeframe, focus, feature count
- [ ] `version` and `model_type` fields present
- [ ] `enabled_categories` lists active groups
- [ ] Every indicator has `enabled`, parameters, and `priority`
- [ ] P0 indicators (RSI, MACD, ATR) are enabled
- [ ] Feature summary at end with approximate counts
- [ ] Referenced correctly in model config file

## Common Mistakes

- **Missing priority**: Can't identify what to disable → Add priority to every indicator
- **No feature count**: Unknown total features → Add summary section at end
- **Enabling everything**: Too many features, overfitting → Start with P0/P1 only
- **Wrong timeframe params**: Periods too long for short-term → Match periods to timeframe

## Validation

- [ ] Pattern confirmed in `configs/indicators/short_term_indicators.yaml`
- [ ] Pattern confirmed in `configs/indicators/medium_term_indicators.yaml`
- [ ] Reference pattern in `configs/short_term.yaml:50-63`

## Related Skills

- [creating-technical-indicators](./creating-technical-indicators.md) - For indicator implementation
- [implementing-prediction-models](../backend/implementing-prediction-models.md) - Consumes indicator config
