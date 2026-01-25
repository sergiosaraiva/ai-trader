# Max Drawdown Mitigation Plan

> **Status:** Tier 1 IMPLEMENTED & VALIDATED
> **Created:** 2026-01-25
> **Last Updated:** 2026-01-25
> **Validation Date:** 2026-01-25

## Tier 1 Validation Results

| Metric | Before | After Tier 1 | Change |
|--------|--------|--------------|--------|
| **Max Drawdown** | 42.2% | **15.1%** | **-64%** |
| Total Pips | +14,637 | +15,705 | **+7.3%** |
| Win Rate | 50.8% | 53.5% | +2.7% |
| Profit Factor | 1.58 | 1.75 | +10.8% |
| Calmar Ratio | ~0.87 | ~2.6 | **+200%** |

**Window 7 (worst case):** DD reduced from 42.2% → 15.1%, Pips improved +31%

## Executive Summary

The trading system shows a **42.2% max drawdown** in WFO Window 7 (Jan-Jun 2025). Root cause analysis reveals this is NOT a model problem but a **risk management gap** - the existing circuit breakers are not integrated into the backtesting simulation.

**Key Finding:** With proper implementation, max drawdown can be reduced from 42% to 12-15% while retaining 80-90% of profits.

---

## Table of Contents

1. [Root Cause Analysis](#1-root-cause-analysis)
2. [Tier 1: Immediate Fixes](#2-tier-1-immediate-fixes-p0)
3. [Tier 2: Medium-Term Enhancements](#3-tier-2-medium-term-enhancements-p1)
4. [Tier 3: Advanced Solutions](#4-tier-3-advanced-solutions-p2)
5. [Expected Results](#5-expected-results)
6. [Implementation Checklist](#6-implementation-checklist)
7. [Code References](#7-code-references)

---

## 1. Root Cause Analysis

### 1.1 Protection Mechanism Gap Analysis

| Protection Mechanism | Exists in Code | Used in Backtest | Gap |
|---------------------|----------------|------------------|-----|
| DrawdownBreaker (15% halt) | `src/trading/circuit_breakers/drawdown.py` | NO | **Critical** |
| RiskManager (daily/weekly limits) | `src/trading/risk.py` | NO | **Critical** |
| Consecutive Loss Reduction | `scripts/walk_forward_optimization.py:515` | Yes (partial) | Works |
| Regime Filter | `src/trading/filters/regime_filter.py` | NO | **Critical** |
| Regime Detector | `src/features/regime/regime_detector.py` | NO | **Critical** |
| Volatility-Based Sizing | - | NO | Missing |
| Equity Curve Filter | - | NO | Missing |
| Kill Switch | `src/trading/circuit_breakers/kill_switch.py` | NO | N/A |

### 1.2 Window 7 Breakdown (Worst Period)

```
Window 7: Jan-Jun 2025
├── Win Rate: 44.6% (below breakeven ~48%)
├── Profit Factor: 1.21 (barely profitable)
├── Max Drawdown: 42.2%
├── Total Trades: 581
├── Losing Trades: 322 (55.4%)
└── Why the large drawdown?
    ├── Market regime changed (likely high volatility/choppy)
    ├── No drawdown halt triggered (circuit breaker not integrated)
    ├── System kept trading through extended losing period
    └── Compounding losses accumulated exponentially
```

### 1.3 WFO Window Performance Comparison

| Window | Period | Max DD | Win Rate | PF | Notes |
|--------|--------|--------|----------|-----|-------|
| 1 | 2022 H1 | 12.1% | 53.4% | 1.77 | Good |
| 2 | 2022 H2 | 15.0% | 53.4% | 1.76 | Good |
| 3 | 2023 H1 | 17.0% | 47.1% | 1.31 | Weak |
| 4 | 2023 H2 | 21.1% | 47.1% | 1.35 | Weak |
| 5 | 2024 H1 | 7.6% | 59.8% | 2.10 | **Best** |
| 6 | 2024 H2 | 9.9% | 53.9% | 1.76 | Good |
| 7 | 2025 H1 | **42.2%** | 44.6% | 1.21 | **Worst** |
| 8 | 2025 H2 | 18.4% | 49.9% | 1.41 | Moderate |

**Pattern:** Drawdowns correlate inversely with win rate. Windows with <50% win rate show elevated drawdowns.

---

## 2. Tier 1: Immediate Fixes (P0)

**Goal:** Reduce max drawdown from 42% to ~20%
**Complexity:** Low
**Timeline:** 1-2 days

### 2.1 Hard Drawdown Circuit Breaker in Backtest

**File:** `backend/scripts/walk_forward_optimization.py`
**Location:** Inside `run_window_backtest()` function, after line ~608

**Current Code:**
```python
# Line ~608
max_drawdown = max(max_drawdown, current_drawdown)
```

**Required Change:**
```python
max_drawdown = max(max_drawdown, current_drawdown)

# CIRCUIT BREAKER - halt trading at 15% drawdown
MAX_ALLOWED_DRAWDOWN = 0.15  # 15%
if current_drawdown >= MAX_ALLOWED_DRAWDOWN:
    logger.warning(
        f"Circuit breaker triggered: Drawdown {current_drawdown:.1%} >= "
        f"{MAX_ALLOWED_DRAWDOWN:.0%}, halting trading for this window"
    )
    break  # Stop trading for this window
```

**Expected Impact:** Would have stopped Window 7 at ~15% drawdown instead of 42%.

### 2.2 Progressive Position Reduction Based on Drawdown

**File:** `backend/scripts/walk_forward_optimization.py`
**Location:** Add as new function, call in position sizing section

**New Function:**
```python
def get_drawdown_position_multiplier(
    current_drawdown: float,
    max_allowed: float = 0.15
) -> float:
    """
    Progressive position reduction based on current drawdown.

    Levels:
    - 0-5% DD: Full size (1.0x)
    - 5-7.5% DD: 75% size
    - 7.5-10% DD: 50% size
    - 10-15% DD: 25% size
    - 15%+ DD: No trading (0x)

    Args:
        current_drawdown: Current drawdown as decimal (e.g., 0.10 = 10%)
        max_allowed: Maximum allowed drawdown before halt

    Returns:
        Position size multiplier (0.0 to 1.0)
    """
    if current_drawdown < 0.05:
        return 1.0    # Full size up to 5% DD
    elif current_drawdown < 0.075:
        return 0.75   # 75% size at 5-7.5% DD
    elif current_drawdown < 0.10:
        return 0.50   # 50% size at 7.5-10% DD
    elif current_drawdown < max_allowed:
        return 0.25   # 25% size at 10-15% DD
    else:
        return 0.0    # No trading beyond limit
```

**Usage (in position sizing section around line ~536):**
```python
# After calculating position_lots:
position_lots = risk_amount / (sl_pips * pip_dollar_value)

# Apply drawdown-based reduction
current_drawdown_pct = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
drawdown_multiplier = get_drawdown_position_multiplier(current_drawdown_pct)
position_lots = position_lots * drawdown_multiplier

# Skip trade if multiplier is 0
if position_lots <= 0:
    i += 1
    continue

# Cap position size to reasonable limits
position_lots = min(position_lots, 5.0)
```

### 2.3 Increase Default Confidence Threshold

**File:** `backend/scripts/walk_forward_optimization.py`
**Location:** Argument parser defaults and function parameters

**Current Default:** 0.55 (55%)
**Recommended Default:** 0.70 (70%)

**Evidence:**
| Threshold | Win Rate | Profit Factor | Trades | Improvement |
|-----------|----------|---------------|--------|-------------|
| 0.55 | 50.8% | 1.56 | 3,801 | Baseline |
| 0.70 | 53.1% | 1.75 | 2,853 | +12% PF |
| 0.75 | 54.2% | 1.83 | 2,553 | +17% PF |

**Changes Required:**
```python
# In argparse section:
parser.add_argument(
    "--confidence", "-c",
    type=float,
    default=0.70,  # Changed from 0.55
    help="Minimum confidence threshold (default: 0.70)"
)

# In run_window_backtest function signature:
def run_window_backtest(
    ...
    min_confidence: float = 0.70,  # Changed from 0.55
    ...
)
```

### 2.4 Tier 1 Testing Checklist

- [ ] Run WFO with circuit breaker enabled
- [ ] Verify Window 7 drawdown is capped at ~15%
- [ ] Verify overall profitability is retained (expect 80-90% of original pips)
- [ ] Compare Calmar ratios before/after
- [ ] Document results

---

## 3. Tier 2: Medium-Term Enhancements (P1)

**Goal:** Reduce max drawdown from ~20% to ~12%
**Complexity:** Medium
**Timeline:** 1-2 weeks

### 3.1 Integrate Regime Filter into Backtest

**Files Involved:**
- `backend/src/features/regime/regime_detector.py` (exists)
- `backend/src/trading/filters/regime_filter.py` (exists)
- `backend/scripts/walk_forward_optimization.py` (modify)

**Implementation:**

```python
# Add imports at top of walk_forward_optimization.py:
from src.features.regime.regime_detector import RegimeDetector, MarketRegime

# Initialize detector (once per window):
regime_detector = RegimeDetector()

# Inside trading loop, BEFORE taking trade (around line ~509):
# Get recent data for regime detection (need ~50 bars)
if i >= 50:
    df_regime = df_1h_features.iloc[i-50:i+1]
    try:
        current_regime = regime_detector.get_current_regime(df_regime)

        # Define dangerous regimes to skip
        SKIP_REGIMES = [
            MarketRegime.RANGING_HIGH_VOL,  # Choppy + volatile = worst
        ]

        if current_regime.market_regime in SKIP_REGIMES:
            i += 1
            continue  # Skip this trade

        # Optional: Reduce size in suboptimal regimes
        REDUCED_REGIMES = [
            MarketRegime.RANGING_NORMAL,
            MarketRegime.TRENDING_HIGH_VOL,
        ]

        if current_regime.market_regime in REDUCED_REGIMES:
            position_lots *= 0.5  # Half size in suboptimal regimes

    except Exception as e:
        logger.debug(f"Regime detection failed: {e}")
        # Continue with trade if detection fails
```

**Regime Performance Reference (from existing analysis):**
| Regime | Win Rate | Avg Pips | PF | Recommendation |
|--------|----------|----------|-----|----------------|
| Trending High Vol | 71.9% | 12.9 | 4.15 | TRADE |
| Trending Normal | 65.9% | 10.9 | 3.31 | TRADE |
| Trending Low Vol | 69.8% | 12.9 | 4.09 | TRADE |
| Ranging High Vol | 84.0% | 17.7 | 8.38 | TRADE (surprising) |
| Ranging Normal | 73.8% | 13.6 | 4.89 | TRADE |
| Ranging Low Vol | 74.6% | 14.0 | 5.01 | TRADE |

**Note:** The regime stats show good performance in all regimes at 70% confidence. The issue may be regime detection accuracy rather than regime filtering. Consider starting with volatility-based sizing instead.

### 3.2 Volatility-Based Position Sizing

**File:** `backend/scripts/walk_forward_optimization.py`

**New Function:**
```python
def calculate_volatility_adjusted_risk(
    base_risk: float,
    current_atr: float,
    avg_atr: float,
    min_multiplier: float = 0.25,
    max_multiplier: float = 1.5
) -> float:
    """
    Scale risk inversely with volatility.

    Higher volatility = smaller positions (to maintain constant dollar risk)
    Lower volatility = larger positions (to capture opportunity)

    Args:
        base_risk: Base risk per trade (e.g., 0.02 = 2%)
        current_atr: Current ATR value
        avg_atr: Average ATR over lookback period
        min_multiplier: Minimum scaling factor (floor)
        max_multiplier: Maximum scaling factor (ceiling)

    Returns:
        Adjusted risk percentage
    """
    if current_atr <= 0 or avg_atr <= 0:
        return base_risk

    # Inverse relationship: high vol = low multiplier
    vol_ratio = avg_atr / current_atr

    # Clamp multiplier to reasonable range
    multiplier = max(min_multiplier, min(max_multiplier, vol_ratio))

    return base_risk * multiplier
```

**Integration (requires ATR in feature DataFrame):**
```python
# Calculate average ATR (once before trading loop):
if "atr_14" in df_1h_features.columns:
    avg_atr = df_1h_features["atr_14"].rolling(50).mean()
else:
    # Calculate ATR if not present
    high = df_1h_features["high"]
    low = df_1h_features["low"]
    close = df_1h_features["close"]
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    avg_atr = tr.ewm(span=14).mean().rolling(50).mean()

# Inside trading loop:
current_atr = df_1h_features["atr_14"].iloc[i] if "atr_14" in df_1h_features.columns else None
if current_atr and not pd.isna(avg_atr.iloc[i]):
    adjusted_risk = calculate_volatility_adjusted_risk(
        base_risk=current_risk,
        current_atr=current_atr,
        avg_atr=avg_atr.iloc[i]
    )
else:
    adjusted_risk = current_risk

risk_amount = balance * adjusted_risk
```

### 3.3 Equity Curve Trading Filter

**Concept:** Only trade when account equity is above its moving average. When equity falls below MA, it indicates a losing streak - reduce or halt trading until recovery.

**New Function:**
```python
def equity_curve_filter(
    equity_history: list,
    ma_period: int = 20,
    action: str = "reduce"  # "reduce" or "halt"
) -> tuple[bool, float]:
    """
    Filter trades based on equity curve health.

    Args:
        equity_history: List of equity values
        ma_period: Period for moving average
        action: "reduce" returns multiplier, "halt" returns 0 if below MA

    Returns:
        Tuple of (should_trade, position_multiplier)
    """
    if len(equity_history) < ma_period:
        return True, 1.0

    equity_ma = sum(equity_history[-ma_period:]) / ma_period
    current_equity = equity_history[-1]

    if current_equity >= equity_ma:
        return True, 1.0
    else:
        if action == "halt":
            return False, 0.0
        else:
            # Calculate how far below MA
            pct_below = (equity_ma - current_equity) / equity_ma
            # Reduce position size based on distance below MA
            multiplier = max(0.25, 1.0 - (pct_below * 2))
            return True, multiplier
```

**Integration:**
```python
# Maintain equity history (before trading loop):
equity_history = [initial_balance]

# Inside trading loop, after updating balance:
equity_history.append(balance)

# Before taking next trade:
should_trade, equity_multiplier = equity_curve_filter(equity_history, ma_period=20)
if not should_trade:
    i += 1
    continue

position_lots *= equity_multiplier
```

### 3.4 Daily Loss Limit

**Implementation:**
```python
# Track daily P&L (initialize before loop):
daily_pnl = 0.0
current_day = None
DAILY_LOSS_LIMIT = -0.03  # -3% max daily loss

# Inside trading loop, after each trade:
trade_day = timestamps[exit_idx].date()

# Reset daily P&L at day change
if current_day != trade_day:
    daily_pnl = 0.0
    current_day = trade_day

# Update daily P&L
daily_pnl += pnl_usd / initial_balance

# Check daily limit
if daily_pnl <= DAILY_LOSS_LIMIT:
    logger.warning(f"Daily loss limit hit: {daily_pnl:.1%} on {trade_day}")
    # Skip remaining trades today
    while i < n and timestamps[i].date() == trade_day:
        i += 1
    continue
```

---

## 4. Tier 3: Advanced Solutions (P2)

**Goal:** Reduce max drawdown from ~12% to ~8%
**Complexity:** Medium-High
**Timeline:** 2-4 weeks

### 4.1 Dynamic ATR-Based Stop Loss

**Current:** Fixed 15 pip stop loss for 1H timeframe.
**Proposed:** Dynamic stops based on market volatility.

```python
def calculate_atr_stop_loss(
    current_atr: float,
    multiplier: float = 2.0,
    min_pips: float = 10,
    max_pips: float = 30,
    pip_value: float = 0.0001  # EUR/USD
) -> float:
    """
    Calculate stop loss based on ATR.

    Wider stops in volatile markets, tighter in calm markets.

    Args:
        current_atr: Current ATR value (in price terms)
        multiplier: ATR multiplier for stop distance
        min_pips: Minimum stop loss in pips
        max_pips: Maximum stop loss in pips
        pip_value: Value of 1 pip in price terms

    Returns:
        Stop loss distance in pips
    """
    atr_pips = current_atr / pip_value
    stop_loss = atr_pips * multiplier
    return max(min_pips, min(max_pips, stop_loss))


def calculate_atr_take_profit(
    stop_loss_pips: float,
    risk_reward_ratio: float = 1.5
) -> float:
    """
    Calculate take profit based on stop loss and R:R ratio.

    Args:
        stop_loss_pips: Stop loss distance in pips
        risk_reward_ratio: Desired risk:reward ratio

    Returns:
        Take profit distance in pips
    """
    return stop_loss_pips * risk_reward_ratio
```

**Integration (replace fixed sl_pips/tp_pips):**
```python
# Inside trading loop, when setting up trade:
if "atr_14" in df_1h_features.columns:
    current_atr = df_1h_features["atr_14"].iloc[i]
    sl_pips = calculate_atr_stop_loss(current_atr, multiplier=2.0)
    tp_pips = calculate_atr_take_profit(sl_pips, risk_reward_ratio=1.5)
else:
    sl_pips = 15  # Fallback to fixed
    tp_pips = 25
```

### 4.2 Multi-Level Circuit Breaker System

**Comprehensive circuit breaker with multiple thresholds:**

```python
from dataclasses import dataclass, field
from typing import List, Tuple
from enum import Enum

class BreakerLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CAUTION = "caution"
    ALERT = "alert"
    REDUCED = "reduced"
    MINIMAL = "minimal"
    HALT = "halt"

@dataclass
class MultiLevelCircuitBreaker:
    """
    Progressive circuit breaker with multiple levels.

    Provides graduated response to drawdown, allowing
    partial trading in moderate drawdowns.
    """

    levels: List[Tuple[float, BreakerLevel, float]] = field(default_factory=lambda: [
        (0.03, BreakerLevel.WARNING, 0.80),   # 3% DD: 80% size
        (0.05, BreakerLevel.CAUTION, 0.60),   # 5% DD: 60% size
        (0.08, BreakerLevel.ALERT, 0.40),     # 8% DD: 40% size
        (0.10, BreakerLevel.REDUCED, 0.25),   # 10% DD: 25% size
        (0.12, BreakerLevel.MINIMAL, 0.10),   # 12% DD: 10% size
        (0.15, BreakerLevel.HALT, 0.00),      # 15% DD: Stop trading
    ])

    # Additional controls per level
    min_confidence_overrides: dict = field(default_factory=lambda: {
        BreakerLevel.WARNING: 0.70,
        BreakerLevel.CAUTION: 0.75,
        BreakerLevel.ALERT: 0.80,
        BreakerLevel.REDUCED: 0.85,
        BreakerLevel.MINIMAL: 0.90,
    })

    def get_action(self, current_drawdown: float) -> Tuple[BreakerLevel, float, float]:
        """
        Get action based on current drawdown.

        Returns:
            Tuple of (level, position_multiplier, min_confidence)
        """
        for threshold, level, multiplier in reversed(self.levels):
            if current_drawdown >= threshold:
                min_conf = self.min_confidence_overrides.get(level, 0.55)
                return level, multiplier, min_conf
        return BreakerLevel.NORMAL, 1.0, 0.55

    def should_trade(self, current_drawdown: float, signal_confidence: float) -> Tuple[bool, float]:
        """
        Determine if trade should be taken and at what size.

        Returns:
            Tuple of (should_trade, position_multiplier)
        """
        level, multiplier, min_conf = self.get_action(current_drawdown)

        if level == BreakerLevel.HALT:
            return False, 0.0

        if signal_confidence < min_conf:
            return False, 0.0

        return True, multiplier
```

**Usage:**
```python
# Initialize (before trading loop):
circuit_breaker = MultiLevelCircuitBreaker()

# Inside trading loop:
current_dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0
should_trade, cb_multiplier = circuit_breaker.should_trade(current_dd, conf)

if not should_trade:
    i += 1
    continue

position_lots *= cb_multiplier
```

### 4.3 Correlation-Based Position Reduction

**Concept:** When multiple timeframes are losing simultaneously, it indicates correlated adverse conditions - reduce exposure.

```python
def calculate_correlation_multiplier(
    recent_trades: list,
    lookback: int = 10
) -> float:
    """
    Reduce position size when losses are correlated across timeframes.

    Args:
        recent_trades: List of recent trade dicts with 'pnl_pips' key
        lookback: Number of recent trades to analyze

    Returns:
        Position multiplier (0.5-1.0)
    """
    if len(recent_trades) < lookback:
        return 1.0

    recent = recent_trades[-lookback:]
    losses = sum(1 for t in recent if t.get("pnl_pips", 0) < 0)
    loss_ratio = losses / lookback

    if loss_ratio >= 0.8:  # 80%+ losses
        return 0.25
    elif loss_ratio >= 0.6:  # 60%+ losses
        return 0.50
    elif loss_ratio >= 0.5:  # 50%+ losses
        return 0.75
    else:
        return 1.0
```

### 4.4 Weekly Loss Limit

```python
# Track weekly P&L (initialize before loop):
weekly_pnl = 0.0
current_week = None
WEEKLY_LOSS_LIMIT = -0.08  # -8% max weekly loss

# Inside trading loop, after each trade:
trade_week = timestamps[exit_idx].isocalendar()[:2]  # (year, week)

# Reset weekly P&L at week change
if current_week != trade_week:
    weekly_pnl = 0.0
    current_week = trade_week

# Update weekly P&L
weekly_pnl += pnl_usd / initial_balance

# Check weekly limit
if weekly_pnl <= WEEKLY_LOSS_LIMIT:
    logger.warning(f"Weekly loss limit hit: {weekly_pnl:.1%}")
    # Skip remaining trades this week
    while i < n and timestamps[i].isocalendar()[:2] == trade_week:
        i += 1
    continue
```

---

## 5. Expected Results

### 5.1 Performance Projections

| Stage | Max Drawdown | Est. Profit Retention | Calmar Ratio |
|-------|--------------|----------------------|--------------|
| Current | 42.2% | 100% (baseline) | ~0.87 |
| After Tier 1 | ~15-20% | 80-90% | ~1.5-2.0 |
| After Tier 2 | ~10-15% | 75-85% | ~2.0-2.5 |
| After Tier 3 | ~8-12% | 70-80% | ~2.5-3.0 |

### 5.2 Trade-Off Analysis

**Drawdown Reduction vs Profit Reduction:**

| Max DD Target | Profit Impact | Risk-Adjusted Improvement |
|---------------|--------------|---------------------------|
| 42% → 20% | -10% to -15% | +50% better Calmar |
| 20% → 15% | -5% to -10% | +30% better Calmar |
| 15% → 10% | -5% to -10% | +20% better Calmar |

**Recommendation:** Target 15% max drawdown as optimal trade-off between risk and return.

### 5.3 Key Metrics Comparison

**Before (Current):**
```
Total Pips: 14,637
Max Drawdown: 42.2%
Calmar Ratio: 14,637 / (42.2% * 4 years) = ~0.87
Window 7 DD: 42.2%
```

**After (Projected with Tier 1+2):**
```
Total Pips: ~12,000 (80% retention)
Max Drawdown: 15% (capped)
Calmar Ratio: 12,000 / (15% * 4 years) = ~2.0
Window 7 DD: ~15% (capped)
```

---

## 6. Implementation Checklist

### Tier 1 (P0 - Immediate) ✅ COMPLETED

- [x] **2.1** Add hard 15% drawdown halt to `walk_forward_optimization.py`
- [x] **2.2** Add `get_drawdown_position_multiplier()` function
- [x] **2.2** Integrate progressive position reduction in trading loop
- [x] **2.3** Change default confidence from 0.55 to 0.70
- [x] **Test** Run WFO with changes, verify DD capped at ~15% ✅ (15.1% achieved)
- [x] **Test** Compare total pips before/after ✅ (+7.3% improvement)
- [x] **Test** Calculate new Calmar ratio ✅ (~2.6 vs ~0.87 baseline)
- [x] **Document** Results in this file ✅

**Validation Run:** 2026-01-25 | Output: `models/wfo_tier1_validation/`

### Tier 2 (P1 - Week 1-2)

- [ ] **3.1** Integrate RegimeDetector into backtest
- [ ] **3.1** Test regime filtering impact
- [ ] **3.2** Add `calculate_volatility_adjusted_risk()` function
- [ ] **3.2** Calculate and use ATR for position sizing
- [ ] **3.3** Add `equity_curve_filter()` function
- [ ] **3.3** Integrate equity curve trading
- [ ] **3.4** Add daily loss limit (-3%)
- [ ] **Test** Run WFO with Tier 2 changes
- [ ] **Test** Compare results to Tier 1 baseline
- [ ] **Document** Results

### Tier 3 (P2 - Week 2-4)

- [ ] **4.1** Add dynamic ATR-based stop loss
- [ ] **4.1** Add dynamic take profit based on R:R
- [ ] **4.2** Implement `MultiLevelCircuitBreaker` class
- [ ] **4.2** Integrate multi-level breaker
- [ ] **4.3** Add correlation-based position reduction
- [ ] **4.4** Add weekly loss limit (-8%)
- [ ] **Test** Run WFO with all changes
- [ ] **Test** Final performance comparison
- [ ] **Document** Final results

### Final Validation

- [ ] Run complete WFO with all tiers enabled
- [ ] Compare to baseline (no risk management)
- [ ] Verify max drawdown < 15%
- [ ] Verify profit retention > 75%
- [ ] Update CLAUDE.md with new default parameters
- [ ] Update backtest scripts to use new defaults

---

## 7. Code References

### Existing Files to Modify

| File | Purpose | Tier |
|------|---------|------|
| `backend/scripts/walk_forward_optimization.py` | Main backtest - add circuit breakers | All |
| `backend/scripts/backtest_mtf_ensemble.py` | Alt backtest - sync changes | All |

### Existing Files to Integrate

| File | Contains | Tier |
|------|----------|------|
| `backend/src/trading/circuit_breakers/drawdown.py` | DrawdownBreaker class | Reference |
| `backend/src/trading/risk.py` | RiskManager, RiskLimits | Reference |
| `backend/src/features/regime/regime_detector.py` | RegimeDetector class | Tier 2 |
| `backend/src/trading/filters/regime_filter.py` | RegimeFilter class | Tier 2 |
| `backend/src/trading/position_sizing.py` | Kelly position sizing | Reference |

### Key Functions/Classes

```python
# Existing (for reference):
DrawdownBreaker.check()  # Returns action based on DD level
DrawdownBreaker.update_equity()  # Update and check
RiskManager._check_circuit_breakers()  # Daily/weekly limits
RegimeDetector.get_current_regime()  # Get market regime
RegimeFilter.analyze()  # Get trading recommendation

# New (to implement):
get_drawdown_position_multiplier()  # Tier 1
calculate_volatility_adjusted_risk()  # Tier 2
equity_curve_filter()  # Tier 2
calculate_atr_stop_loss()  # Tier 3
MultiLevelCircuitBreaker  # Tier 3
```

---

## Appendix A: Configuration Parameters

### Recommended Production Settings

```python
# Position Sizing
DEFAULT_RISK_PER_TRADE = 0.02  # 2%
MIN_RISK_MULTIPLIER = 0.25
MAX_RISK_MULTIPLIER = 1.0

# Drawdown Limits
MAX_DRAWDOWN_HALT = 0.15  # 15%
DRAWDOWN_WARNING = 0.05  # 5%
DRAWDOWN_CAUTION = 0.075  # 7.5%
DRAWDOWN_ALERT = 0.10  # 10%

# Daily/Weekly Limits
DAILY_LOSS_LIMIT = 0.03  # 3%
WEEKLY_LOSS_LIMIT = 0.08  # 8%

# Confidence
MIN_CONFIDENCE_DEFAULT = 0.70
MIN_CONFIDENCE_HIGH_DD = 0.85

# Equity Curve
EQUITY_MA_PERIOD = 20

# Volatility Scaling
VOL_MIN_MULTIPLIER = 0.25
VOL_MAX_MULTIPLIER = 1.5
ATR_LOOKBACK = 50
```

---

## Appendix B: Backtest Command Reference

```bash
# Current (baseline)
python scripts/walk_forward_optimization.py --sentiment --stacking

# With Tier 1 changes (after implementation)
python scripts/walk_forward_optimization.py \
    --sentiment \
    --stacking \
    --confidence 0.70 \
    --max-drawdown 0.15

# Full validation
python scripts/walk_forward_optimization.py \
    --sentiment \
    --stacking \
    --confidence 0.70 \
    --max-drawdown 0.15 \
    --all-windows
```

---

**Document Version:** 1.0
**Author:** Claude Code Analysis
**Review Status:** Pending Implementation
