# Market Regime Detection Analysis

**Date:** January 12, 2026
**Status:** COMPLETED

---

## Executive Summary

Market regime detection has been implemented and analyzed to understand how the MTF Ensemble model performs across different market conditions. The key finding is that **the model is already robust across all market regimes** - no regime filtering is needed.

### Key Findings

1. **All 6 market regimes are profitable** - No need to avoid any regime
2. **Best win rate**: Ranging + High Volatility (84.0% WR, 8.38 PF)
3. **Most trades**: Trending Normal (211 trades, +2,303 pips)
4. **The model adapts well** to both trending and ranging conditions

### Performance by Regime

| Regime | Trades | Win Rate | Total Pips | Profit Factor | Recommendation |
|--------|--------|----------|------------|---------------|----------------|
| Ranging High Vol | 25 | **84.0%** | +443 | 8.38 | TRADE |
| Ranging Low Vol | 67 | 74.6% | +939 | 5.01 | TRADE |
| Ranging Normal | 84 | 73.8% | +1,142 | 4.89 | TRADE |
| Trending High Vol | 96 | 71.9% | +1,241 | 4.15 | TRADE |
| Trending Low Vol | 96 | 69.8% | +1,239 | 4.09 | TRADE |
| Trending Normal | 211 | 65.9% | +2,303 | 3.31 | TRADE |

---

## 1. Regime Detection Implementation

### Module Location

```
src/features/regime/
├── __init__.py
└── regime_detector.py
```

### Regime Classifications

#### Trend Regimes
| Regime | Condition | ADX Threshold |
|--------|-----------|---------------|
| Strong Uptrend | ADX > 25, bullish alignment | > 25 |
| Strong Downtrend | ADX > 25, bearish alignment | > 25 |
| Weak Uptrend | 15 < ADX < 25, bullish | 15-25 |
| Weak Downtrend | 15 < ADX < 25, bearish | 15-25 |
| Ranging | ADX < 15 | < 15 |

#### Volatility Regimes
| Regime | Condition |
|--------|-----------|
| High | ATR percentile > 75% or VIX > 25 |
| Normal | ATR percentile 25-75% |
| Low | ATR percentile < 25% or VIX < 15 |

#### Combined Market Regimes (6 total)
- Trending + High Volatility
- Trending + Normal Volatility
- Trending + Low Volatility
- Ranging + High Volatility
- Ranging + Normal Volatility
- Ranging + Low Volatility

---

## 2. Performance Analysis

### By Trend Regime

| Trend Regime | Trades | Win Rate | Total Pips | Avg Pips |
|--------------|--------|----------|------------|----------|
| Ranging | 176 | **75.6%** | +2,524 | +14.3 |
| Strong Downtrend | 164 | 69.5% | +2,063 | +12.6 |
| Strong Uptrend | 141 | 69.5% | +1,748 | +12.4 |
| Weak Downtrend | 51 | 64.7% | +507 | +9.9 |
| Weak Uptrend | 47 | 63.8% | +467 | +9.9 |

**Insight**: Ranging markets have the highest win rate (75.6%), while strong trends produce consistent profits. Weak trends are less profitable but still positive.

### By Volatility Regime

| Volatility | Trades | Win Rate | Total Pips | Avg Pips |
|------------|--------|----------|------------|----------|
| High | 121 | **74.4%** | +1,684 | +13.9 |
| Low | 163 | 71.8% | +2,178 | +13.4 |
| Normal | 295 | 68.1% | +3,445 | +11.7 |

**Insight**: High volatility has the best win rate (74.4%), but normal volatility produces the most total pips due to higher trade frequency.

---

## 3. Why No Regime Filtering is Needed

The analysis reveals that the MTF Ensemble model is inherently robust:

1. **All regimes profitable**: Every regime has positive expected value
2. **Consistent win rates**: 65-84% across all regimes
3. **High profit factors**: 3.31 to 8.38 across all regimes
4. **No weak links**: Even the "worst" regime (Trending Normal) has 65.9% WR and 3.31 PF

### Model Robustness Factors

1. **Multi-timeframe consensus** reduces noise from short-term regime changes
2. **70% confidence threshold** filters out uncertain predictions
3. **Weighted ensemble** balances short-term and long-term views
4. **Technical features** already capture trend/volatility information

---

## 4. Practical Applications

### Regime-Aware Position Sizing

While filtering is unnecessary, regime information can be used for position sizing:

```python
from src.features.regime import RegimeDetector, MarketRegime

detector = RegimeDetector()

def get_position_multiplier(regime: MarketRegime) -> float:
    """Adjust position size based on regime."""
    if regime in [MarketRegime.RANGING_HIGH_VOL]:
        return 1.2  # Higher win rate, slight increase
    elif regime in [MarketRegime.TRENDING_NORMAL]:
        return 0.9  # Lower win rate, slight decrease
    return 1.0  # Normal sizing
```

### Regime Monitoring Dashboard

Track regime distribution in live trading:

```python
# Count regime occurrences
regime_counts = df["market_regime"].value_counts()

# Calculate time in each regime
regime_pct = regime_counts / len(df) * 100
```

---

## 5. Usage Examples

### Basic Regime Detection

```python
from src.features.regime import RegimeDetector

# Create detector
detector = RegimeDetector()

# Detect regimes for all bars
df_with_regime = detector.detect_regime(df_ohlcv)

# Get current regime
current_regime = detector.get_current_regime(df_ohlcv)
print(f"Current: {current_regime.market_regime}")
```

### Running Regime Analysis

```bash
# Full analysis with 70% threshold
python scripts/analyze_regime_performance.py --confidence 0.70

# Different threshold
python scripts/analyze_regime_performance.py --confidence 0.65
```

---

## 6. Conclusions

### Summary

1. **Regime detection module implemented** - Classifies trend and volatility regimes
2. **All regimes profitable** - Model handles all market conditions well
3. **No filtering required** - Continue trading in all regimes
4. **Potential for enhancement** - Use for position sizing or monitoring

### Recommendations

| Use Case | Recommendation |
|----------|----------------|
| Trade Filtering | **Not Recommended** - All regimes profitable |
| Position Sizing | Consider - Scale up in high WR regimes |
| Risk Management | Consider - Reduce size in high volatility |
| Monitoring | Recommended - Track regime distribution |

### Future Enhancements

1. **Dynamic confidence thresholds** - Adjust threshold by regime
2. **Regime-specific TP/SL** - Optimize parameters per regime
3. **Regime transition alerts** - Notify when regime changes
4. **Regime-based model selection** - Use different weights per regime

---

## Appendix: Results Data

### Full Results Location
`results/regime_analysis/regime_performance.json`

### Scripts Created
- `src/features/regime/regime_detector.py` - Core detection module
- `scripts/analyze_regime_performance.py` - Analysis script

### Test Configuration
- Data: EUR/USD 5-min (2020-2025)
- Model: MTF Ensemble (1H: 60%, 4H: 30%, D: 10%)
- Confidence: 70%
- Test Period: ~14 months out-of-sample
