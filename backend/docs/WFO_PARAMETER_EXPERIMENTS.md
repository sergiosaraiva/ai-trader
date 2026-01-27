# WFO Parameter Experiments Guide

This guide shows different WFO configurations you can test to see how results change.

## Overview

The WFO script has **25+ configurable parameters** across 5 categories:
1. **Window Configuration** - Training/test period sizes
2. **Model Features** - Sentiment, stacking, feature selection
3. **Risk Management** - Position sizing, risk reduction, circuit breakers
4. **Trading Rules** - Confidence thresholds, filters
5. **Output** - Where to save results

## Current Baseline Configuration

This is what you ran to get the existing WFO results:

```bash
python scripts/walk_forward_optimization.py --sentiment --stacking
```

**Baseline Parameters:**
- Training window: 24 months
- Test window: 6 months
- Step size: 6 months
- Confidence: 0.70
- Initial balance: $10,000
- Risk per trade: 2%
- Risk reduction: ON
- Sentiment: ON (Daily model only)
- Stacking: ON
- Tier 2 filters: ALL ON

**Baseline Results:**
- 8 windows, 100% profitable
- 887 trades, 56.5% win rate
- +5,239 pips, 15.1% max DD

---

## Experiment Categories

### 1. Window Size Experiments

**Test different training/test period lengths**

#### Experiment 1A: Shorter Training Window (18 months)

```bash
python scripts/walk_forward_optimization.py \
  --sentiment --stacking \
  --train-months 18 \
  --output models/wfo_18mo
```

**Hypothesis:** Shorter training = more adaptive to recent market conditions

**Expected Impact:**
- ✅ Faster adaptation to regime changes
- ⚠️ Less historical data = potentially less stable
- 🔍 Window 7 may perform better (more recent training)

**What to compare:** Consistency rate, Window 7 trade count, overall win rate

---

#### Experiment 1B: Longer Training Window (30 months)

```bash
python scripts/walk_forward_optimization.py \
  --sentiment --stacking \
  --train-months 30 \
  --output models/wfo_30mo
```

**Hypothesis:** Longer training = more stable, less overfitting

**Expected Impact:**
- ✅ More robust to regime changes
- ⚠️ Slower adaptation to new conditions
- 🔍 Window 7 may still struggle (same regime issue)

**What to compare:** Overall consistency, max DD stability

---

#### Experiment 1C: Monthly Rolling (1-month steps)

```bash
python scripts/walk_forward_optimization.py \
  --sentiment --stacking \
  --train-months 24 \
  --test-months 1 \
  --step-months 1 \
  --output models/wfo_monthly_roll
```

**Hypothesis:** More frequent retraining = better adaptation

**Expected Impact:**
- ✅ 36+ windows (vs 8)
- ✅ Better statistical significance
- ⚠️ Much longer runtime (~2-3 hours)
- 🔍 Can identify exact months where performance degrades

**What to compare:** Consistency across more windows, variability

---

### 2. Confidence Threshold Experiments

**Test how confidence filtering affects results**

#### Experiment 2A: Lower Confidence (60%)

```bash
python scripts/walk_forward_optimization.py \
  --sentiment --stacking \
  --confidence 0.60 \
  --output models/wfo_conf60
```

**Hypothesis:** Lower threshold = more trades, potentially lower win rate

**Expected Impact:**
- ✅ More trades per window (maybe 30-50% more)
- ⚠️ Lower win rate (50-55% vs 56.5%)
- 🔍 Window 7 may get 10-15 trades instead of 3

**What to compare:** Total trades, win rate, profit factor

---

#### Experiment 2B: Higher Confidence (80%)

```bash
python scripts/walk_forward_optimization.py \
  --sentiment --stacking \
  --confidence 0.80 \
  --output models/wfo_conf80
```

**Hypothesis:** Higher threshold = fewer trades, higher win rate

**Expected Impact:**
- ⚠️ Fewer trades (maybe 50% less)
- ✅ Higher win rate (60-65%)
- ✅ Higher profit factor
- 🔍 Window 7 may have 0-1 trades

**What to compare:** Win rate, profit factor, trade frequency

---

### 3. Risk Management Experiments

**Test different position sizing and risk approaches**

#### Experiment 3A: Higher Risk (3% per trade)

```bash
python scripts/walk_forward_optimization.py \
  --sentiment --stacking \
  --risk 0.03 \
  --output models/wfo_risk3pct
```

**Hypothesis:** Higher risk = higher returns but higher drawdown

**Expected Impact:**
- ✅ 50% higher returns (if win rate constant)
- ⚠️ Higher max DD (20-25% vs 15%)
- 🔍 More volatile equity curve

**What to compare:** Total return %, max DD, Sharpe ratio

---

#### Experiment 3B: No Risk Reduction

```bash
python scripts/walk_forward_optimization.py \
  --sentiment --stacking \
  --no-risk-reduction \
  --output models/wfo_no_reduction
```

**Hypothesis:** Without progressive reduction, DD increases

**Expected Impact:**
- ⚠️ Higher max DD (18-20%)
- ✅ Slightly higher returns (no reduction after losses)
- 🔍 Longer drawdown periods

**What to compare:** Max DD, recovery time, losing streaks

---

#### Experiment 3C: Conservative (1% risk)

```bash
python scripts/walk_forward_optimization.py \
  --sentiment --stacking \
  --risk 0.01 \
  --output models/wfo_risk1pct
```

**Hypothesis:** Lower risk = lower DD, lower returns

**Expected Impact:**
- ✅ Lower max DD (~7-10%)
- ⚠️ 50% lower returns
- ✅ Smoother equity curve

**What to compare:** Max DD, return/risk ratio, Sharpe

---

### 4. Model Feature Experiments

**Test different model configurations**

#### Experiment 4A: No Sentiment

```bash
python scripts/walk_forward_optimization.py \
  --stacking \
  --output models/wfo_no_sentiment
```

**Hypothesis:** Sentiment adds value on Daily model

**Expected Impact:**
- ⚠️ Lower win rate (54-55% vs 56.5%)
- 🔍 Daily model accuracy drops
- 🔍 Window 7 may perform better (if sentiment was stale)

**What to compare:** Overall win rate, Daily model accuracy

---

#### Experiment 4B: No Stacking (Weighted Average)

```bash
python scripts/walk_forward_optimization.py \
  --sentiment \
  --output models/wfo_no_stacking
```

**Hypothesis:** Stacking meta-learner improves predictions

**Expected Impact:**
- ⚠️ Slightly lower win rate (55-56% vs 56.5%)
- 🔍 Simpler model, faster training
- 🔍 May be more stable (less overfitting risk)

**What to compare:** Win rate, consistency across windows

---

#### Experiment 4C: Minimal Setup (No Sentiment, No Stacking)

```bash
python scripts/walk_forward_optimization.py \
  --output models/wfo_minimal
```

**Hypothesis:** Simpler is better (baseline test)

**Expected Impact:**
- ⚠️ Lower performance (52-54% win rate)
- ✅ Faster training (30-40% faster)
- 🔍 Shows value-add of advanced features

**What to compare:** Performance gap vs full setup

---

### 5. Tier 2 Filter Experiments

**Test advanced risk management filters**

#### Experiment 5A: No Filters

```bash
python scripts/walk_forward_optimization.py \
  --sentiment --stacking \
  --no-regime-filter \
  --no-equity-filter \
  --no-volatility-sizing \
  --output models/wfo_no_filters
```

**Hypothesis:** Filters reduce risk but may reduce returns

**Expected Impact:**
- ⚠️ Higher max DD (18-20%)
- ✅ More trades (10-15% more)
- 🔍 Potentially higher returns

**What to compare:** Max DD, total trades, return/risk

---

#### Experiment 5B: Regime Filter Only

```bash
python scripts/walk_forward_optimization.py \
  --sentiment --stacking \
  --no-equity-filter \
  --no-volatility-sizing \
  --output models/wfo_regime_only
```

**Hypothesis:** Regime filter is most valuable

**Expected Impact:**
- ✅ Better DD control than no filters
- 🔍 Skips trades in ranging_high_vol regime
- 🔍 Window 7 may skip more trades (if bad regime detected)

**What to compare:** Trade distribution by regime

---

## Quick Reference Table

| Experiment | Command Flag | Impact | Runtime |
|------------|-------------|--------|---------|
| **Shorter training (18mo)** | `--train-months 18` | More adaptive | ~45 min |
| **Longer training (30mo)** | `--train-months 30` | More stable | ~60 min |
| **Monthly rolling** | `--step-months 1` | 36 windows | ~3 hours |
| **Lower confidence (60%)** | `--confidence 0.60` | More trades | ~60 min |
| **Higher confidence (80%)** | `--confidence 0.80` | Fewer trades | ~45 min |
| **Higher risk (3%)** | `--risk 0.03` | Higher return/DD | ~60 min |
| **No risk reduction** | `--no-risk-reduction` | Higher DD | ~60 min |
| **Conservative (1%)** | `--risk 0.01` | Lower DD | ~60 min |
| **No sentiment** | (omit `--sentiment`) | Faster, simpler | ~50 min |
| **No stacking** | (omit `--stacking`) | Faster, simpler | ~45 min |
| **Minimal** | No flags | Baseline test | ~40 min |
| **No filters** | `--no-*-filter` flags | More trades | ~55 min |

---

## Recommended Experiment Sequence

### Phase 1: Quick Tests (2-3 hours total)

Test core parameters to see biggest impacts:

```bash
# 1. Lower confidence (more trades, helps Window 7)
python scripts/walk_forward_optimization.py --sentiment --stacking --confidence 0.60 --output models/wfo_conf60

# 2. Shorter training (more adaptive)
python scripts/walk_forward_optimization.py --sentiment --stacking --train-months 18 --output models/wfo_18mo

# 3. Higher risk (stress test)
python scripts/walk_forward_optimization.py --sentiment --stacking --risk 0.03 --output models/wfo_risk3pct
```

**Compare:** Which improves Window 7? Which has best risk-adjusted return?

### Phase 2: Feature Ablation (2 hours total)

Test if advanced features add value:

```bash
# 4. No sentiment
python scripts/walk_forward_optimization.py --stacking --output models/wfo_no_sentiment

# 5. No stacking
python scripts/walk_forward_optimization.py --sentiment --output models/wfo_no_stacking

# 6. Minimal
python scripts/walk_forward_optimization.py --output models/wfo_minimal
```

**Compare:** How much does each feature contribute?

### Phase 3: Deep Dive (optional, 3-5 hours)

If you find promising configurations:

```bash
# 7. Monthly rolling (best config from Phase 1)
python scripts/walk_forward_optimization.py --sentiment --stacking --confidence 0.60 --step-months 1 --output models/wfo_monthly_conf60

# 8. No filters
python scripts/walk_forward_optimization.py --sentiment --stacking --no-regime-filter --no-equity-filter --no-volatility-sizing --output models/wfo_no_filters
```

---

## How to Compare Results

After running experiments, compare using:

### 1. Summary Metrics

```bash
# Extract key metrics from each experiment
cat models/wfo_conf60/wfo_results.json | jq '.summary | {consistency_rate, overall_win_rate, max_drawdown, total_trades}'
cat models/wfo_18mo/wfo_results.json | jq '.summary | {consistency_rate, overall_win_rate, max_drawdown, total_trades}'
```

### 2. Create Comparison Table

| Config | Consistency | Win Rate | Max DD | Trades | Total Pips |
|--------|-------------|----------|--------|--------|------------|
| Baseline (24mo, 70%) | 100% | 56.5% | 15.1% | 887 | +5,239 |
| Conf 60% | ? | ? | ? | ? | ? |
| 18mo training | ? | ? | ? | ? | ? |
| Risk 3% | ? | ? | ? | ? | ? |

### 3. Window 7 Performance

```bash
# Check Window 7 trades in each experiment
cat models/wfo_*/wfo_results.json | jq '.windows[] | select(.window_id == 7) | {config: input_filename, trades: .total_trades, win_rate, total_pips}'
```

### 4. Risk-Adjusted Returns

Calculate Sharpe-like ratios:
```
Sharpe approximation = Total Return % / Max Drawdown %
```

---

## Expected Insights

### Most Likely Improvements:

1. **Lower confidence (0.60)** → Solves Window 7 issue, more trades
2. **Shorter training (18mo)** → Better adaptation to 2025 regime
3. **Monthly rolling** → Better statistical validation

### Trade-offs to Watch:

1. **Higher risk** → Higher returns BUT higher DD (may not be worth it)
2. **No filters** → More trades BUT worse DD (risky)
3. **Simpler models** → Faster BUT lower performance

### What NOT to Expect:

- ❌ Any config to eliminate Window 7 completely (regime change is real)
- ❌ Dramatic improvements (current setup is already good)
- ❌ 100% consistency with all configs (some variance is normal)

---

## Saving & Comparing Results

Each experiment saves to a different directory:

```
models/
├── wfo_validation/        # Baseline (current)
├── wfo_conf60/           # Experiment 1
├── wfo_18mo/             # Experiment 2
├── wfo_risk3pct/         # Experiment 3
└── ...
```

All generate `wfo_results.json` and update `data/backtest_results.json`.

**Tip:** Copy `data/backtest_results.json` after each experiment:
```bash
cp data/backtest_results.json results/backtest_conf60.json
```

---

## Next Steps

1. **Choose 2-3 experiments** from Phase 1
2. **Run them** (can run in parallel if you have multiple cores)
3. **Compare results** using comparison table
4. **Pick the best** configuration
5. **Run Phase 2** to validate feature value
6. **Update CLAUDE.md** with best configuration

---

**Created:** 2026-01-27
**Current Baseline:** 24mo training, 70% confidence, 2% risk, all features ON
**Best Alternative:** TBD (run experiments to find out!)
