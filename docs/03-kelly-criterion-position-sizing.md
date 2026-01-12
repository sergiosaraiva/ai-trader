# Kelly Criterion Position Sizing Implementation

**Date:** January 12, 2026
**Status:** IMPLEMENTED

---

## Executive Summary

Kelly criterion position sizing has been implemented to optimize capital allocation for the MTF Ensemble trading system. The implementation includes:

1. **Full Kelly** - Theoretical optimal for maximum growth
2. **Half Kelly** - Balance between growth and stability (recommended)
3. **Quarter Kelly** - Conservative approach
4. **Confidence-Adjusted Kelly** - Scales with model confidence

### Key Finding

With realistic broker position limits (2-5 standard lots for $100k account), **all Kelly strategies converge to similar results** because they hit the maximum position cap. This is an important practical insight.

---

## 1. Kelly Criterion Theory

### Formula

```
Kelly % = W - (1-W) / R

Where:
  W = Win probability
  R = Win/Loss ratio (avg_win / avg_loss)
```

### Derivation from Trading Stats

For the MTF Ensemble based on WFO results:

| Parameter | Value | Source |
|-----------|-------|--------|
| Win Rate (W) | 53.05% | WFO aggregate |
| Profit Factor | 1.89 | WFO average |
| Win/Loss Ratio (R) | 1.67 | Derived: PF * (1-W) / W |
| **Full Kelly** | **24.98%** | W - (1-W)/R |
| Half Kelly | 12.49% | Full Kelly * 0.5 |
| Quarter Kelly | 6.25% | Full Kelly * 0.25 |

### Expected Value

```
EV = W * R - (1-W) = 0.42 per unit risked
```

This means on average, for every $1 risked, expected return is $0.42.

---

## 2. Implementation Details

### Files Created

| File | Purpose |
|------|---------|
| `src/trading/position_sizing.py` | Kelly position sizing module |
| `scripts/backtest_position_sizing.py` | Strategy comparison backtest |

### KellyPositionSizer Class

```python
from src.trading.position_sizing import (
    KellyPositionSizer,
    KellyParameters,
    PositionSizingConfig,
    SizingStrategy,
)

# Calculate Kelly parameters from WFO
kelly_params = KellyParameters.from_wfo_results(
    wfo_path="models/wfo_validation/wfo_results.json"
)

# Create position sizer
config = PositionSizingConfig(
    strategy=SizingStrategy.HALF_KELLY,
    kelly_params=kelly_params,
    max_lot_size=2.0,  # Realistic limit
)

sizer = KellyPositionSizer(
    account_balance=100000,
    config=config,
)

# Calculate position size
position_size, details = sizer.calculate_position_size(
    confidence=0.65,
    stop_loss_pips=15.0,
    pip_value=10.0,
)
```

### Strategies Available

| Strategy | Description | Risk Level |
|----------|-------------|------------|
| `FIXED` | Fixed percentage (1-2%) | Conservative |
| `QUARTER_KELLY` | 25% of full Kelly (~6%) | Conservative |
| `HALF_KELLY` | 50% of full Kelly (~12%) | Moderate |
| `FULL_KELLY` | 100% Kelly (~25%) | Aggressive |
| `CONFIDENCE_KELLY` | Kelly scaled by confidence | Adaptive |

---

## 3. Backtest Results

### Theoretical vs Practical

**Theoretical (No Limits):**

| Strategy | Position Size | Compound Growth |
|----------|---------------|-----------------|
| Full Kelly | 160 lots | Maximum theoretical |
| Half Kelly | 80 lots | ~75% of max growth |
| Quarter Kelly | 40 lots | ~50% of max growth |
| Fixed 1% | 6.7 lots | Lowest variance |

**Practical (With 2-Lot Limit):**

| Strategy | Final Balance | Return | MaxDD | RAR |
|----------|---------------|--------|-------|-----|
| Fixed | $259,746 | +159.7% | 2.1% | 77.2 |
| Quarter Kelly | $259,746 | +159.7% | 2.1% | 77.2 |
| Half Kelly | $259,746 | +159.7% | 2.1% | 77.2 |
| Full Kelly | $259,746 | +159.7% | 2.1% | 77.2 |

### Key Insight

**With realistic broker limits, all strategies produce identical results** because they all hit the maximum position cap. This demonstrates that:

1. Position limits dominate Kelly calculations in practice
2. Risk management constraints override theoretical optimization
3. For accounts under $500k, broker limits are the binding constraint

---

## 4. Practical Recommendations

### For Different Account Sizes

| Account Size | Recommended Strategy | Max Position |
|--------------|---------------------|--------------|
| $10,000 | Quarter Kelly | 0.5 lots |
| $50,000 | Half Kelly | 1.0 lots |
| $100,000 | Half Kelly | 2.0 lots |
| $500,000 | Half Kelly | 5.0 lots |
| $1,000,000+ | Full Kelly (capped) | 10.0 lots |

### Position Sizing Guidelines

```
Position Size = min(
    Kelly % * Account Balance / (Stop Loss * Pip Value),
    Max Lot Limit
)
```

For $100,000 account with Half Kelly (12%):
```
Risk Amount = $100,000 * 12% = $12,000
Position = $12,000 / (15 pips * $10/pip) = 80 lots
Actual Position = min(80, 2) = 2 lots (capped)
```

### Risk Management Integration

The position sizing module integrates with existing risk management:

1. **Max Position Limit**: Caps individual trade size
2. **Max Exposure**: Limits total open positions
3. **Drawdown Protection**: Reduces size during drawdowns
4. **Confidence Scaling**: Adjusts for model confidence

---

## 5. Code Usage

### Basic Usage

```python
from src.trading.position_sizing import (
    KellyPositionSizer,
    PositionSizingConfig,
    SizingStrategy,
)

# Create sizer with default Half Kelly
sizer = KellyPositionSizer(account_balance=100000)

# Calculate size for a trade
size, details = sizer.calculate_position_size(
    confidence=0.65,
    stop_loss_pips=15.0,
)
print(f"Position: {size:.2f} lots")
```

### With Custom Parameters

```python
from src.trading.position_sizing import KellyParameters

# Custom Kelly parameters
kelly_params = KellyParameters(
    win_rate=0.578,        # 57.8% from baseline
    profit_factor=2.22,    # 2.22 from baseline
    avg_win_pips=22.8,
    avg_loss_pips=14.1,
)

config = PositionSizingConfig(
    strategy=SizingStrategy.CONFIDENCE_KELLY,
    kelly_params=kelly_params,
    kelly_fraction=0.5,
    max_lot_size=5.0,
)

sizer = KellyPositionSizer(
    account_balance=100000,
    config=config,
)
```

### Running Comparison Backtest

```bash
source .venv/bin/activate
python scripts/backtest_position_sizing.py --balance 100000 --confidence 0.55
```

---

## 6. Integration with Live Trading

### Recommended Configuration

For live trading with the MTF Ensemble:

```python
config = PositionSizingConfig(
    strategy=SizingStrategy.HALF_KELLY,
    kelly_fraction=0.5,
    fixed_risk_pct=0.01,      # Fallback: 1% per trade
    max_position_pct=0.03,    # Max 3% account per position
    max_lot_size=2.0,         # Broker limit
    min_lot_size=0.01,        # Micro lot minimum
    max_total_exposure=0.15,  # 15% max total exposure
)
```

### Dynamic Kelly Updates

The sizer supports updating Kelly parameters from recent trades:

```python
# After each trade, record it
sizer.record_trade({
    "pnl_pips": +25,
    "confidence": 0.68,
})

# Periodically update Kelly parameters
sizer.update_kelly_params()  # Updates from last 100 trades
```

---

## 7. Conclusions

### Implementation Status: COMPLETE

1. Kelly criterion position sizing module implemented
2. Multiple strategies available (Fixed, Quarter, Half, Full, Confidence)
3. Backtest comparison script created
4. Integration with existing risk management

### Key Findings

1. **Full Kelly (25%) is too aggressive** - Would require 160 lots per trade
2. **Half Kelly (12%) is recommended** - Good balance of growth and stability
3. **Broker limits dominate** - With realistic limits, strategies converge
4. **Risk management is crucial** - Max position caps override Kelly

### Recommendation

Use **Half Kelly with realistic limits**:
- Kelly Fraction: 50%
- Max Position: 2-5 lots depending on account size
- Max Exposure: 15% of account
- Integrate with confidence scaling for optimal allocation

---

## Appendix: Kelly Mathematics

### Why Kelly Works

Kelly maximizes the expected log of wealth (geometric growth):

```
E[log(W)] = p * log(1 + f * b) + q * log(1 - f)

Where:
  p = win probability
  q = 1 - p (loss probability)
  b = win/loss ratio
  f = fraction to bet
```

### Why Fractional Kelly is Preferred

| Kelly Fraction | Growth Rate | Variance | Drawdown Risk |
|----------------|-------------|----------|---------------|
| Full (100%) | Maximum | Very High | High |
| Half (50%) | 75% of max | 50% lower | Moderate |
| Quarter (25%) | 50% of max | 75% lower | Low |

Half Kelly achieves 75% of optimal growth with 50% of the variance - an excellent trade-off for practical trading.
