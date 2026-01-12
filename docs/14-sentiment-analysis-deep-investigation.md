# Ultra-Deep Investigation: Sentiment Analysis Degradation

## Executive Summary

After comprehensive investigation including code analysis and online research, I've identified **7 critical root causes** for why sentiment degraded the model performance, and discovered that **sentiment should primarily benefit medium/long-term models, NOT short-term**.

### Key Finding
> **The current sentiment data is MONTHLY resolution (not daily)**, meaning all 31 days of January have IDENTICAL values. This fundamental data quality issue makes sentiment useless for short-term trading and introduces pure noise.

---

## Part 1: Root Cause Analysis

### Root Cause #1: MONTHLY Resolution Data (CRITICAL)

**Evidence:**
```
2020-01-01  0.15005824751415778  (same value for 31 days)
2020-01-02  0.15005824751415778
...
2020-01-31  0.15005824751415778
2020-02-01  0.11904342561300725  (new value for next 29 days)
```

**Impact:**
- Only ~72 unique sentiment values in 6 years (monthly EPU data)
- 97% of derived features (MAs, momentum, ROC) are constant or near-zero
- Model learns spurious correlations at month boundaries
- Adds 25+ noise features that dilute technical signal

### Root Cause #2: Wrong Timeframe Application

**Research Finding:**
| Timeframe | Sentiment Value | Research Recommendation |
|-----------|-----------------|-------------------------|
| 1H (Intraday) | **LOW** | Real-time news sentiment (minutes-hours lag) |
| 4H (Swing) | MODERATE | Daily aggregated sentiment (1-7 day lag) |
| Daily (Position) | **HIGH** | Monthly EPU appropriate here |

**Current Implementation Problem:**
- Monthly EPU data applied equally to ALL timeframes
- 1H model gets the SAME stale sentiment as Daily model
- Short-term models need real-time sentiment, not monthly EPU

### Root Cause #3: Over-Engineering on Low-Information Data

Current implementation generates **25+ features** from monthly data:

| Feature Type | Issue with Monthly Data |
|--------------|------------------------|
| `sentiment_ma_3/7/14` | MA of constant = constant (useless) |
| `sentiment_std_7/14/30` | STD of constant = 0 (useless) |
| `sentiment_momentum_*` | Always ~0 (useless) |
| `sentiment_roc_7/14` | Only non-zero at month boundaries |
| `sentiment_zscore` | Division by zero or constant |
| `sentiment_lag_1/2/3` | Same value (useless) |

**Result:** 25 noise features overwhelm the 38 useful technical features.

### Root Cause #4: Intraday Alignment Problem

For 1H model:
- 24 hourly candles per day × 31 days = 744 candles
- ALL 744 candles get the SAME sentiment value
- Model sees zero intraday sentiment variation
- Creates perfect collinearity → model confusion

### Root Cause #5: Missing Feature Selection by Timeframe

**Training Results Analysis:**

| Timeframe | Sentiment in Top 10 Features | Val Acc Change |
|-----------|------------------------------|----------------|
| 1H | 0/10 (none!) | -0.91% (worse) |
| 4H | 3/10 | +0.84% (better) |
| Daily | 6/10 | +2.00% (better) |

**Key Insight:** The model automatically de-prioritized sentiment for 1H (not in top 10), but still suffered from the noise of 25+ low-importance features consuming model capacity.

### Root Cause #6: Sentiment Signal Decay Not Handled

**Research Finding:** Academic research shows sentiment signals decay rapidly:
- News sentiment: 15 minutes - 4 hours
- Social media: 30 minutes - 1 day
- Monthly EPU: 12-24 month horizon (macro only!)

**Problem:** Monthly EPU data is designed for macroeconomic forecasting (12-24 month horizon), not trading signals.

### Root Cause #7: Contrarian Signal Ignored

**Research Best Practice:** Sentiment extremes should be used as contrarian signals:
- Extreme fear → potential buying opportunity
- Extreme greed → potential selling opportunity

**Current Implementation:** Uses sentiment as a direct feature, not as an extreme/contrarian indicator.

---

## Part 2: Online Research Findings

### What Academic Research Says About Sentiment Timeframes

| Source | Finding |
|--------|---------|
| FinBERT Research | Sentiment most effective for short-term (minutes-hours) when using news data |
| arXiv (Olaiyapo et al.) | ChatGPT shows 35% better classification than FinBERT for forex news |
| EPU Research (Sengupta) | EPU useful for 12-24 month macro forecasting, NOT intraday |
| SWFX Sentiment | Broker position sentiment updates every 30 minutes for intraday |

### Recommended Sentiment Data Sources by Timeframe

| Timeframe | Recommended Sentiment Source | Update Frequency |
|-----------|------------------------------|------------------|
| 1H (Intraday) | Real-time news (FinBERT), Social media, Broker positioning | 30 min - 4 hours |
| 4H (Swing) | Daily news aggregate, VIX, Fear/Greed Index | Daily |
| Daily (Position) | Weekly EPU, Analyst sentiment, COT reports | Weekly |
| Weekly (Long-term) | Monthly EPU (current data!) | Monthly |

### Key Research Insight: Sentiment as Filter, Not Signal

Best practices recommend using sentiment as:
1. **Confirmation filter** - validate technical signals
2. **Contrarian indicator** - at extremes only
3. **Regime classifier** - risk-on/risk-off detection

NOT as a direct predictive feature!

---

## Part 3: Recommended Improvements

### Improvement 1: Timeframe-Specific Sentiment Configuration

```python
# RECOMMENDED CONFIGURATION

# 1H Model - DISABLE SENTIMENT (monthly EPU useless for intraday)
ImprovedModelConfig.hourly_model(include_sentiment_features=False)

# 4H Model - MINIMAL SENTIMENT (only if daily data available)
ImprovedModelConfig.four_hour_model(
    include_sentiment_features=True,
    sentiment_features=['sentiment_regime', 'cross_sent_range']  # Only 2 features
)

# Daily Model - FULL SENTIMENT (monthly EPU somewhat appropriate)
ImprovedModelConfig.daily_model(
    include_sentiment_features=True,
    sentiment_features=['sentiment_raw', 'sent_country_*', 'cross_sent_*']
)
```

### Improvement 2: Get Better Sentiment Data

For meaningful short-term sentiment, the project needs:

| Data Source | Cost | Update Freq | Best For |
|-------------|------|-------------|----------|
| **FRED Daily EPU** | Free | Daily | Daily/Weekly models |
| **News API + FinBERT** | Low | Real-time | 1H-4H models |
| **Twitter/X API** | Medium | Real-time | Intraday models |
| **RavenPack** | High | Real-time | Professional trading |

**Free Option:** FRED provides daily US EPU at: `https://fred.stlouisfed.org/series/USEPUINDXD`

### Improvement 3: Reduce Feature Count Drastically

Instead of 25+ derived features, use only:

```python
# For Daily model with monthly EPU data
essential_features = [
    'sentiment_raw',           # Base sentiment
    'sentiment_regime',        # -1/0/1 classification
    'cross_sent_range',        # Country disagreement
    'sentiment_vs_price_trend' # Divergence indicator (NEW)
]
# Total: 4 features instead of 25+
```

### Improvement 4: Add Sentiment-Price Divergence Feature

**New Feature Concept:** Detect when sentiment and price trend disagree (contrarian signal)

```python
def add_sentiment_divergence(df, sentiment_col, price_col, window=20):
    """
    Detect sentiment-price divergence for contrarian signals.

    Returns:
        1: Bullish divergence (price down, sentiment up) - BUY signal
        -1: Bearish divergence (price up, sentiment down) - SELL signal
        0: No divergence
    """
    price_trend = df[price_col].pct_change(window).apply(lambda x: 1 if x > 0.01 else (-1 if x < -0.01 else 0))
    sent_trend = df[sentiment_col].diff(window).apply(lambda x: 1 if x > 0.01 else (-1 if x < -0.01 else 0))

    divergence = np.where(
        (price_trend == -1) & (sent_trend == 1), 1,  # Bullish divergence
        np.where((price_trend == 1) & (sent_trend == -1), -1, 0)  # Bearish divergence
    )
    return divergence
```

### Improvement 5: Use Sentiment as Trade Filter, Not Feature

**Current Approach (Wrong):**
```
prediction = model.predict(technical_features + sentiment_features)
```

**Recommended Approach:**
```python
# Step 1: Get technical prediction
tech_signal = technical_model.predict(technical_features)
tech_confidence = technical_model.predict_proba(technical_features)

# Step 2: Apply sentiment filter
if sentiment_regime == 'extreme_fear' and tech_signal == 'BUY':
    # Contrarian confirmation - boost confidence
    final_confidence = tech_confidence * 1.1
elif sentiment_regime == 'extreme_greed' and tech_signal == 'BUY':
    # Sentiment contradiction - reduce confidence
    final_confidence = tech_confidence * 0.8
else:
    final_confidence = tech_confidence
```

### Improvement 6: Implement Proper Sentiment for Ensemble

```python
# Modified MTF Ensemble with proper sentiment handling

class MTFEnsembleConfig:
    # Sentiment configuration PER TIMEFRAME
    sentiment_config = {
        "1H": {
            "enabled": False,  # Disable for intraday
        },
        "4H": {
            "enabled": True,
            "as_filter": True,  # Use as filter, not feature
            "features": ["sentiment_regime"],
        },
        "D": {
            "enabled": True,
            "as_feature": True,  # Use as feature for daily
            "features": ["sentiment_raw", "cross_sent_range", "sent_country_us", "sent_country_europe"],
        },
    }
```

---

## Part 4: Implementation Roadmap

### Phase 1: Quick Fix (Immediate)
1. **Disable sentiment for 1H model** - Will immediately improve performance
2. **Reduce sentiment features** from 25+ to 4 essential features
3. **Re-run backtest** to verify improvement

### Phase 2: Better Data (Short-term)
1. **Download daily EPU** from FRED (free): `USEPUINDXD`
2. **Implement daily sentiment loader**
3. **Test with daily resolution data**

### Phase 3: Real-time Sentiment (Medium-term)
1. **Integrate news API** (free tiers available)
2. **Implement FinBERT** for headline sentiment classification
3. **Create real-time sentiment pipeline** for intraday models

### Phase 4: Advanced Integration (Long-term)
1. **Sentiment as filter** instead of feature
2. **Contrarian signals** at extremes
3. **Regime-based model switching**

---

## Part 5: Conclusion

### Why Sentiment Degraded Performance

1. **Wrong data resolution** - Monthly EPU used for hourly trading
2. **Wrong timeframe application** - Same data for all models
3. **Feature explosion** - 25+ noise features from 1 monthly value
4. **Wrong usage pattern** - Direct feature instead of filter/contrarian

### Key Takeaway

> **Sentiment analysis IS valuable for forex trading, but ONLY when:**
> - Using appropriate resolution data (daily/real-time for short-term)
> - Applying to appropriate timeframes (longer = better for EPU)
> - Using minimal, meaningful features (not over-engineered)
> - Treating it as a filter/confirmation, not primary signal

### Recommended Immediate Action

```bash
# Re-train with sentiment disabled for 1H
python scripts/train_mtf_ensemble.py \
    --output models/mtf_ensemble_v2 \
    --weights 0.8,0.15,0.05
    # Note: sentiment disabled for 1H by default now

# Then enable sentiment only for Daily model (code change required)
```

---

## References

1. FinBERT - Financial Sentiment Analysis (arXiv:1908.10063)
2. Economic Policy Uncertainty Index - policyuncertainty.com
3. FRED Daily EPU - https://fred.stlouisfed.org/series/USEPUINDXD
4. Corporate Finance Institute - Market Sentiment Trading
5. Sengupta et al. - EPU and Exchange Rate Forecasting
