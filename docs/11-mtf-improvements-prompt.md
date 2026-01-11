# Prompt for Claude Code: MTF Ensemble Improvements

Copy and paste the following prompt into a new clean Claude Code session:

---

## PROMPT START

I need you to improve my **Multi-Timeframe Ensemble** trading model.

### Current State

I have a working MTF Ensemble with these backtest results:

| Model | Win Rate | Profit Factor | Total Pips |
|-------|----------|---------------|------------|
| **1H alone** | 60.0% | 2.42 | +9,529 |
| **4H alone** | 42.7% | 1.48 | +2,308 |
| **Daily alone** | 33.0% | 0.98 | -55 |
| **Ensemble (60/30/10)** | 55.9% | 2.01 | +6,894 |

**Problem**: The 1H model alone beats the ensemble because 4H and Daily models underperform.

### What to Improve

**Priority 1: Improve 4H and Daily Models**
- Adjust triple barrier parameters (TP/SL/max_holding)
- Possibly adjust model hyperparameters
- Target: 4H >= 50% win rate, Daily >= 45% win rate

**Priority 2: Agreement-Based Filtering**
- Only trade when at least 2 of 3 models agree
- Test with `--agreement 0.67` flag in backtest
- Target: Higher win rate with fewer but better trades

**Priority 3: Learned Weights (if time permits)**
- Create meta-model to learn optimal weights
- Based on recent performance and market regime

### Key Files to Read First

1. `docs/10-mtf-ensemble-improvements.md` - Full improvement guide with rollback procedures
2. `src/models/multi_timeframe/improved_model.py` - Model configs to modify
3. `src/models/multi_timeframe/mtf_ensemble.py` - Ensemble implementation
4. `models/mtf_ensemble/training_metadata.json` - Current training results

### Rollback Procedure

**CRITICAL**: Before making ANY changes:

1. Create backup branch:
```bash
git checkout -b backup/mtf-ensemble-v1
git add . && git commit -m "Backup: MTF ensemble v1 (55.9% win rate)"
git checkout master
git checkout -b experiment/mtf-improvements
```

2. Backup models:
```bash
cp -r models/mtf_ensemble models/mtf_ensemble_backup_v1
```

### Success Criteria

Improvements should achieve AT LEAST ONE of:
- Ensemble win rate >= 60% (match 1H alone)
- Ensemble profit factor >= 2.4
- High-confidence (65%) win rate >= 65%

### Rollback Triggers

**Immediately rollback if**:
- Ensemble win rate drops below 55%
- Profit factor drops below 1.8
- Total pips drops below +5,000

### Commands

```bash
# Train with modified parameters
python scripts/train_mtf_ensemble.py --timeframes "1H,4H,D"

# Backtest with comparison
python scripts/backtest_mtf_ensemble.py --compare

# Backtest with agreement filter
python scripts/backtest_mtf_ensemble.py --agreement 0.67 --compare
```

### Important Notes

- Follow CLAUDE.md instructions (proceed without asking for confirmation)
- Test incrementally (one improvement at a time)
- Always compare to baseline (1H alone: 60%, 2.42 PF)
- Document all parameter changes in commit messages

Please start by reading `docs/10-mtf-ensemble-improvements.md`, create the backup, then implement improvements one at a time.

## PROMPT END

---

## Quick Reference

**Current Best Model**: 1H alone (60.0% win rate, 2.42 PF, +9,529 pips)

**Target**: Make ensemble match or exceed 1H alone performance

**Files to modify**:
- `src/models/multi_timeframe/improved_model.py` - TP/SL parameters
- `src/models/multi_timeframe/mtf_ensemble.py` - Ensemble logic

**Backup location**: `models/mtf_ensemble_backup_v1/`
