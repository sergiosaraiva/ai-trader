# MTF Ensemble Improvements Guide

## 1. Current State Summary

### What Was Implemented
- 3-timeframe ensemble (1H, 4H, Daily) with weighted averaging
- Weights: 60% 1H, 30% 4H, 10% Daily
- Agreement bonus (+5% confidence when all models agree)
- Models saved to `models/mtf_ensemble/`

### Current Backtest Results

| Model | Trades | Win Rate | Profit Factor | Total Pips |
|-------|--------|----------|---------------|------------|
| **1H alone** | 1,183 | **60.0%** | **2.42** | **+9,529** |
| **4H alone** | 337 | 42.7% | 1.48 | +2,308 |
| **Daily alone** | 91 | 33.0% | 0.98 | -55 |
| **Ensemble** | 1,087 | 55.9% | 2.01 | +6,894 |

### Key Finding
**The 1H model alone outperforms the ensemble.** The 4H and Daily models underperform significantly (42.7% and 33.0% win rate), and averaging with weaker models drags down ensemble performance.

---

## 2. Improvement Priorities

### Priority 1: Improve 4H and Daily Models (High Impact)

**Problem**: 4H model has only 42.7% win rate, Daily has 33.0%

**Root Causes**:
1. Fewer training samples (4H: 5,728, Daily: 1,057 vs 1H: 22,367)
2. Triple barrier parameters may not be optimal for longer timeframes
3. Feature set may need adjustment for longer horizons

**Implementation**:
```python
# Adjust Daily model parameters in ImprovedModelConfig.daily_model()
# Current: tp_pips=100.0, sl_pips=50.0, max_holding_bars=10
# Try: tp_pips=150.0, sl_pips=75.0, max_holding_bars=15

# Adjust 4H model parameters
# Current: tp_pips=50.0, sl_pips=25.0, max_holding_bars=18
# Try: tp_pips=75.0, sl_pips=35.0, max_holding_bars=24
```

**Files to Modify**:
- `src/models/multi_timeframe/improved_model.py` - Model configs

### Priority 2: Agreement-Based Filtering (Medium Impact)

**Problem**: Ensemble trades even when models disagree, reducing accuracy

**Implementation**:
- Only trade when at least 2 of 3 models agree
- Higher position size when all 3 agree
- Skip trades when models strongly disagree

**Files to Modify**:
- `src/models/multi_timeframe/mtf_ensemble.py` - Add `min_agreement` filter
- `scripts/backtest_mtf_ensemble.py` - Already has `--agreement` parameter

**Test Command**:
```bash
python scripts/backtest_mtf_ensemble.py --agreement 0.67  # At least 2/3 agree
```

### Priority 3: Learned Weights (Medium Impact)

**Problem**: Fixed weights (60/30/10) may not be optimal

**Implementation**:
Create a meta-model that learns optimal weights based on:
- Recent model performance
- Market regime
- Volatility conditions

**New File to Create**:
- `src/models/multi_timeframe/meta_weight_learner.py`

```python
class MetaWeightLearner:
    """Learn optimal ensemble weights from model predictions."""

    def __init__(self):
        self.model = XGBClassifier(n_estimators=50, max_depth=3)

    def train(self, predictions_df: pd.DataFrame, labels: np.ndarray):
        """
        Input features:
        - 1H prediction, 1H confidence
        - 4H prediction, 4H confidence
        - D prediction, D confidence
        - Agreement score
        - Volatility (ATR)
        - Trend strength (ADX)
        """
        pass
```

### Priority 4: Regime-Based Trading (Lower Impact)

**Problem**: Fixed strategy regardless of market conditions

**Implementation**:
- Detect trending vs ranging markets
- Adjust weights dynamically
- Skip trades in unfavorable regimes

**Already Implemented** (but not optimized):
- `MTFEnsembleConfig.regime_weights` in `mtf_ensemble.py`
- Need to tune the regime detection and weight adjustments

---

## 3. Rollback Procedures

### Before Making Changes

1. **Check current git status**:
```bash
git status
git log --oneline -5
```

2. **Create a backup branch**:
```bash
git checkout -b backup/mtf-ensemble-v1
git add .
git commit -m "Backup: MTF ensemble v1 (55.9% win rate, 2.01 PF)"
git checkout master
```

3. **Create experiment branch**:
```bash
git checkout -b experiment/mtf-improvements
```

### Rollback Commands

**If improvements fail, rollback to working version**:
```bash
# Discard all changes in current branch
git checkout master

# Or restore specific files
git checkout master -- src/models/multi_timeframe/improved_model.py
git checkout master -- src/models/multi_timeframe/mtf_ensemble.py

# Or hard reset to backup
git checkout backup/mtf-ensemble-v1
```

**Restore saved models**:
```bash
# Models are saved in models/mtf_ensemble/
# Current working models:
#   - 1H_model.pkl (67.07% val acc)
#   - 4H_model.pkl (65.43% val acc)
#   - D_model.pkl (59.09% val acc)

# Before training new models, backup:
cp -r models/mtf_ensemble models/mtf_ensemble_backup_v1
```

### Success Criteria for Improvements

Before merging any improvements, verify:

| Metric | Current | Target | Rollback If |
|--------|---------|--------|-------------|
| Win Rate | 55.9% | >= 60% | < 55% |
| Profit Factor | 2.01 | >= 2.2 | < 1.8 |
| High-Conf Win Rate | 59.6% | >= 65% | < 58% |
| Total Pips | +6,894 | >= +8,000 | < +5,000 |

---

## 4. Testing Checklist

Before each improvement:
- [ ] Create experiment branch
- [ ] Backup current models
- [ ] Document current metrics

After each improvement:
- [ ] Run full backtest
- [ ] Compare to baseline metrics
- [ ] Check all target criteria
- [ ] If worse: rollback immediately
- [ ] If better: merge to master

---

## 5. File Structure

```
models/
├── mtf_ensemble/              # Current models
│   ├── 1H_model.pkl
│   ├── 4H_model.pkl
│   ├── D_model.pkl
│   ├── ensemble_config.json
│   └── training_metadata.json
└── mtf_ensemble_backup_v1/    # Backup before improvements

src/models/multi_timeframe/
├── improved_model.py          # Model configs (modify for Priority 1)
├── mtf_ensemble.py            # Ensemble class (modify for Priority 2, 4)
└── meta_weight_learner.py     # NEW (Priority 3)

scripts/
├── train_mtf_ensemble.py      # Training script
├── backtest_mtf_ensemble.py   # Backtest script
└── tune_mtf_parameters.py     # NEW - hyperparameter tuning
```

---

## 6. Quick Reference Commands

```bash
# Train models with current settings
python scripts/train_mtf_ensemble.py --timeframes "1H,4H,D" --weights "0.6,0.3,0.1"

# Backtest with comparison
python scripts/backtest_mtf_ensemble.py --compare

# Backtest with agreement filter
python scripts/backtest_mtf_ensemble.py --agreement 0.67 --compare

# Backtest 1H alone (baseline comparison)
python scripts/backtest_improved.py --timeframe 1H
```
