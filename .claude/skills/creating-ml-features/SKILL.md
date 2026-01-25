---
name: creating-ml-features
description: This skill should be used when the user asks to "create ML features", "add feature engineering", "calculate technical features", "prevent data leakage", "add rolling calculations". Creates machine learning features with proper data leakage prevention using shift(1) for all time-series calculations.
version: 1.0.0
---

# Creating ML Features with Data Leakage Prevention

## Quick Reference

- **CRITICAL**: Use `.shift(1)` on ALL rolling calculations to prevent look-ahead bias
- Document leakage prevention at module and method level
- Use `min_periods` in rolling calculations to handle edge cases
- Handle NaN values from shifting with appropriate fillna strategies
- Calculate features once and cache results

## When to Use

- Creating features for ML models from time-series data
- Adding rolling calculations (volatility, moving averages, etc.)
- Implementing cross-timeframe or meta-features
- Calculating technical indicators for predictions

## When NOT to Use

- Simple non-time-series feature calculations
- Features that don't involve historical lookback
- Ground truth labels (these need different handling)

## Implementation Guide

```
Is this a rolling/window calculation?
├─ Yes → MUST use .shift(1) after calculation
│   └─ Document with "CRITICAL" comment
│   └─ Handle resulting NaN with fillna()
└─ No → Check if it uses future data anyway
    └─ If yes → Add appropriate lag

Does feature use price data?
├─ Yes → Use only close/open/high/low from t-1 or earlier
│   └─ Never use current bar's data for predictions
└─ No → Verify no other sources of future leakage

Handling NaN from shifting:
├─ For volatility → Fill with median
├─ For boolean → Fill with False or 0
├─ For ratios → Fill with 1.0
└─ For other → Fill with column median/mean
```

## Examples

**Example 1: Module-Level Documentation**

```python
# From: backend/src/models/multi_timeframe/enhanced_meta_features.py:1-21
"""Enhanced Meta-Features for Stacking Meta-Learner.

CRITICAL: Data Leakage Prevention
----------------------------------
All features MUST use .shift(1) or appropriate lag to prevent look-ahead bias:
- Rolling calculations are shifted to use only past data
- Historical accuracy uses lagged predictions vs lagged actuals
- Market context uses only past price data

These features are OPTIONAL and disabled by default for backward compatibility.
Enable with `StackingConfig.use_enhanced_meta_features = True`.
"""
```

**Explanation**: Module docstring explicitly states data leakage prevention policy. Makes it clear that ALL rolling calculations must be shifted.

**Example 2: Class with Leakage Prevention**

```python
# From: backend/src/models/multi_timeframe/enhanced_meta_features.py:32-46
class EnhancedMetaFeatureCalculator:
    """Calculator for enhanced meta-features with data leakage prevention.

    All rolling calculations use .shift(1) to ensure no future data is used.
    NaN values from shifting are handled appropriately.
    """

    def __init__(self, lookback_window: int = 50):
        """Initialize calculator.

        Args:
            lookback_window: Number of past samples for rolling calculations
        """
        self.lookback_window = lookback_window
```

**Explanation**: Class docstring emphasizes leakage prevention. Lookback window parameterized for flexibility.

**Example 3: Volatility Feature with Shift**

```python
# Pattern from: backend/src/models/multi_timeframe/enhanced_meta_features.py
def calculate_market_context(
    self,
    price_data: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """Calculate volatility, trend strength, and regime features.

    CRITICAL: All calculations use .shift(1) to prevent look-ahead bias.
    """
    df = price_data.copy()

    # Calculate returns
    returns = df["close"].pct_change()

    # Volatility: rolling std of returns, SHIFTED to prevent leakage
    vol_raw = returns.rolling(window=20, min_periods=5).std()
    vol_shifted = vol_raw.shift(1)  # CRITICAL: Use past volatility only

    # Trend strength: SHIFTED to prevent leakage
    ma14 = df["close"].rolling(window=14, min_periods=5).mean()
    deviation = abs(df["close"] - ma14) / ma14
    trend_strength_raw = deviation.rolling(window=14, min_periods=5).mean()
    trend_strength_shifted = trend_strength_raw.shift(1)  # CRITICAL

    # Handle NaN from shifting
    vol_shifted = vol_shifted.fillna(vol_shifted.median())
    trend_strength_shifted = trend_strength_shifted.fillna(0.0)

    return {
        "recent_volatility": vol_shifted.values,
        "trend_strength": trend_strength_shifted.values,
    }
```

**Explanation**: Each rolling calculation is immediately followed by `.shift(1)`. Comments mark CRITICAL shifts. NaN handled with appropriate defaults.

**Example 4: Cross-Timeframe Features**

```python
# Pattern from: backend/src/models/multi_timeframe/enhanced_meta_features.py:140-180
def calculate_cross_timeframe_patterns(
    self,
    predictions: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Calculate HTF agreement features.

    These features capture cross-timeframe dynamics:
    - htf_agreement_1h_4h: Do 1H and 4H models agree?
    - htf_agreement_4h_d: Do 4H and Daily models agree?
    - trend_alignment: What fraction of models agree? (0-1)

    Note: These are point-in-time features (no rolling), so no shift needed.
    """
    n_samples = len(predictions["1H"])

    # Agreement between 1H and 4H
    agreement_1h_4h = (predictions["1H"] == predictions["4H"]).astype(int)

    # Agreement between 4H and Daily
    agreement_4h_d = (predictions["4H"] == predictions["D"]).astype(int)

    # Overall trend alignment: fraction of models agreeing
    all_preds = np.stack([predictions["1H"], predictions["4H"], predictions["D"]])
    mode = np.apply_along_axis(
        lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=all_preds
    )
    agreement_count = (all_preds == mode).sum(axis=0)
    trend_alignment = agreement_count / 3.0

    return {
        "htf_agreement_1h_4h": agreement_1h_4h,
        "htf_agreement_4h_d": agreement_4h_d,
        "trend_alignment": trend_alignment,
    }
```

**Explanation**: Point-in-time features don't need shifting. Document why shift is not needed when it's intentional.

**Example 5: Prediction Stability with Shift**

```python
# Pattern from: backend/src/models/multi_timeframe/enhanced_meta_features.py
def calculate_prediction_stability(
    self,
    predictions: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Calculate rolling stability of predictions.

    CRITICAL: Rolling std is SHIFTED to prevent look-ahead bias.
    """
    n_samples = len(predictions["1H"])
    window = min(self.lookback_window, n_samples // 4)

    stability_features = {}

    for tf in predictions:
        pred_series = pd.Series(predictions[tf])

        # Rolling std of predictions (how stable is the model?)
        stability_raw = pred_series.rolling(window=window, min_periods=5).std()
        stability_shifted = stability_raw.shift(1)  # CRITICAL: Past stability only
        stability_shifted = stability_shifted.fillna(0.5)  # Default: moderate stability

        stability_features[f"pred_stability_{tf.lower()}"] = stability_shifted.values

    return stability_features
```

**Explanation**: Per-timeframe stability features. Each shifted with explicit CRITICAL comment. Default fill value documented.

## Quality Checklist

- [ ] Module docstring explains data leakage prevention policy
- [ ] Every rolling calculation has `.shift(1)` applied
- [ ] CRITICAL comments mark all shift operations
- [ ] Pattern matches `backend/src/models/multi_timeframe/enhanced_meta_features.py`
- [ ] NaN values from shifting are handled with fillna()
- [ ] Fill values are appropriate (median for continuous, 0/False for discrete)
- [ ] Non-shifted calculations documented as intentional

## Common Mistakes

- **Forgetting to shift**: Causes look-ahead bias
  - Wrong: `vol = returns.rolling(20).std()`
  - Correct: `vol = returns.rolling(20).std().shift(1)`

- **Using current bar data**: Leaks future information
  - Wrong: Using close price of current bar to predict current bar
  - Correct: Shift all price-derived features

- **Ignoring NaN from shift**: Causes training failures
  - Wrong: Not handling NaN after shift
  - Correct: `vol_shifted.fillna(vol_shifted.median())`

- **Inconsistent fill values**: Wrong assumptions
  - Wrong: Using 0 for volatility (implies no movement)
  - Correct: Use median for volatility (typical movement)

## Validation

- [ ] Pattern confirmed in `backend/src/models/multi_timeframe/enhanced_meta_features.py`
- [ ] Tests exist in `backend/tests/unit/models/test_enhanced_meta_features.py`
- [ ] Integration tests verify no leakage in `backend/tests/integration/`

## Related Skills

- `validating-time-series-data` - Validate time-series integrity
- `creating-technical-indicators` - Technical indicator patterns

---

<!-- Skill Metadata
Version: 1.0.0
Created: 2026-01-23
Last Verified: 2026-01-23
Last Modified: 2026-01-23
Patterns From: .claude/discovery/codebase-patterns.md v3.0 (Pattern 4.10)
Lines: 220
-->
