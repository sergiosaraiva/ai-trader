# Tier 1 Implementation Quick Reference

> **Goal:** Reduce max drawdown from 42% to ~15-20%
> **File:** `backend/scripts/walk_forward_optimization.py`
> **Estimated Time:** 1-2 hours

## Changes Overview

| Change | Location | Lines |
|--------|----------|-------|
| Add drawdown halt | `run_window_backtest()` | ~608 |
| Add position multiplier function | New function | Top of file |
| Integrate position reduction | Position sizing section | ~536 |
| Update default confidence | Argparse + function params | Multiple |

---

## 1. Add Hard Drawdown Halt

**Location:** Inside `run_window_backtest()`, after line ~608

```python
# Find this line:
max_drawdown = max(max_drawdown, current_drawdown)

# Add immediately after:
# CIRCUIT BREAKER - halt trading at 15% drawdown
MAX_ALLOWED_DRAWDOWN = 0.15
if current_drawdown >= MAX_ALLOWED_DRAWDOWN:
    logger.warning(
        f"Circuit breaker: DD {current_drawdown:.1%} >= {MAX_ALLOWED_DRAWDOWN:.0%}, halting"
    )
    break
```

---

## 2. Add Position Multiplier Function

**Location:** Add near other helper functions (around line ~300)

```python
def get_drawdown_position_multiplier(current_drawdown: float, max_allowed: float = 0.15) -> float:
    """Progressive position reduction based on drawdown."""
    if current_drawdown < 0.05:
        return 1.0
    elif current_drawdown < 0.075:
        return 0.75
    elif current_drawdown < 0.10:
        return 0.50
    elif current_drawdown < max_allowed:
        return 0.25
    else:
        return 0.0
```

---

## 3. Integrate Position Reduction

**Location:** Inside trading loop, after position_lots calculation (~line 536-540)

```python
# Find this block:
position_lots = risk_amount / (sl_pips * pip_dollar_value)
position_lots = min(position_lots, 5.0)

# Replace with:
position_lots = risk_amount / (sl_pips * pip_dollar_value)

# Apply drawdown-based reduction
current_dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
dd_multiplier = get_drawdown_position_multiplier(current_dd)
position_lots = position_lots * dd_multiplier

if position_lots <= 0:
    i += 1
    continue

position_lots = min(position_lots, 5.0)
```

---

## 4. Update Default Confidence

**Location 1:** Argparse (around line ~1270)

```python
# Find:
parser.add_argument("--confidence", "-c", type=float, default=0.55, ...)

# Change to:
parser.add_argument("--confidence", "-c", type=float, default=0.70, ...)
```

**Location 2:** Function signature `run_window_backtest()` (around line ~703)

```python
# Find:
min_confidence: float = 0.55,

# Change to:
min_confidence: float = 0.70,
```

**Location 3:** Function signature `run_window_backtest()` inner call (around line ~345)

```python
# Find:
def run_window_backtest(..., min_confidence: float = 0.55, ...):

# Change to:
def run_window_backtest(..., min_confidence: float = 0.70, ...):
```

---

## Testing Commands

```bash
cd backend

# Run single window test first
python scripts/walk_forward_optimization.py \
    --sentiment --stacking \
    --windows 7 \
    --confidence 0.70

# Expected: Window 7 DD should be ~15% (not 42%)

# Run full WFO validation
python scripts/walk_forward_optimization.py \
    --sentiment --stacking \
    --confidence 0.70

# Compare results to baseline
```

---

## Expected Results

| Metric | Before | After Tier 1 |
|--------|--------|--------------|
| Window 7 Max DD | 42.2% | ~15% |
| Window 7 Pips | 1,023 | ~800-900 |
| Overall Max DD | 42.2% | ~15-18% |
| Total Pips | 14,637 | ~12,000-13,000 |
| Calmar Ratio | 0.87 | ~1.8-2.0 |

---

## Verification Checklist

- [ ] Circuit breaker logs appear for Window 7
- [ ] Max drawdown capped at ~15%
- [ ] Position sizes reduce progressively (check logs)
- [ ] Total pips reduced by ~10-20% (acceptable)
- [ ] Calmar ratio improved (should ~double)
