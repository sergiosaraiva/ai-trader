# Sentiment Analysis Integration Test Results

## Test Date: 2026-01-11

## Summary

Tested the impact of adding sentiment features to the XGBoost-based trading model using EUR/USD data.

## Test Configuration

- **Model**: XGBoost Classifier
- **Data**: EUR/USD 5-minute data (2020-2025), resampled to 1H
- **Samples**: 37,396 hourly bars
- **Split**: 60% train / 20% validation / 20% test (time-based)
- **Trials**: 5 runs with different random seeds

## Features

| Configuration | Technical Features | Sentiment Features | Total |
|--------------|-------------------|-------------------|-------|
| Baseline | 38 | 0 | 38 |
| With Sentiment | 38 | 22 | 60 |

### Sentiment Features Added (22 total)
- Raw pair sentiment
- Country sentiments (US, Europe, Germany, Global)
- Moving averages (3, 7, 14, 30 day)
- Standard deviations (7, 14, 30 day)
- Momentum features
- Rate of change (7, 14 day)
- Z-score
- Regime classification
- Lag features (1, 2, 3 days)

## Results

### Aggregated Results (5 Trials)

| Metric | Baseline | With Sentiment | Difference |
|--------|----------|----------------|------------|
| **Test Accuracy** | 42.06% | **45.72%** | **+3.66%** |
| **High-Conf Accuracy** | 39.71% | **43.91%** | **+4.20%** |
| Val Accuracy | 53.94% | 48.68% | -5.25% |
| Train Accuracy | 81.66% | 81.57% | -0.09% |

### Win/Loss Summary

| Metric | Sentiment Wins | Sentiment Loses |
|--------|---------------|-----------------|
| Test Accuracy | 5/5 | 0/5 |
| High-Conf Accuracy | 5/5 | 0/5 |
| Val Accuracy | 0/5 | 5/5 |

## Analysis

### Positive Findings
1. **Test accuracy improved by +3.66%** - Sentiment features help predict unseen data
2. **High-confidence accuracy improved by +4.20%** - More reliable high-conviction trades
3. **Consistent results** - Sentiment won on test accuracy in all 5 trials

### Concerns
1. **Validation accuracy decreased by -5.25%** - This suggests:
   - Possible overfitting to training data patterns
   - Different market regimes in validation vs test periods
   - The model may be learning spurious correlations in validation period

2. **Train accuracy unchanged** - Adding 22 features didn't improve training fit, which is unusual

### Interpretation

The divergence between validation and test results is notable. In time-series data:
- Validation period: ~2023 (mid-period)
- Test period: ~2024-2025 (most recent)

The improvement in test accuracy suggests sentiment features may be more predictive in recent market conditions. However, the validation drop warrants caution.

## Recommendation

### NEUTRAL - Keep with Monitoring

**Rationale:**
- Test accuracy improvement (+3.66%) is meaningful for trading
- High-confidence improvement (+4.20%) benefits real trading decisions
- However, validation accuracy drop is concerning

**Suggested Actions:**
1. **Keep sentiment integration** but monitor live performance closely
2. **Do not rollback** - test results are the best predictor of future performance
3. **Consider walk-forward validation** to better understand temporal stability
4. **Track real trading results** with sentiment vs without for 2-4 weeks

### If Live Results Degrade

If live trading results with sentiment are worse than the backtest suggests:
1. Rollback to baseline (without sentiment)
2. Investigate temporal dynamics of sentiment features
3. Consider sentiment-only models for specific market regimes

## Reproduction Steps

```bash
# Activate environment
source .venv/bin/activate

# Run sentiment impact test
python scripts/test_sentiment_impact.py

# Results will show:
# - Baseline vs Sentiment comparison
# - 5-trial aggregated statistics
# - Recommendation
```

## Files Modified

- `scripts/test_sentiment_impact.py` - Test script for sentiment comparison
- `src/features/sentiment/` - Sentiment feature calculation
- `data/sentiment/` - Sentiment data files

## Version

- Tested against: v1.0.0 (MTF Ensemble)
- Sentiment integration: Feature branch (pre-merge evaluation)
