# Prompt for Claude Code: Multi-Timeframe Ensemble Implementation

Copy and paste the following prompt into a new clean Claude Code session:

---

## PROMPT START

I need you to implement a **Multi-Timeframe Ensemble** trading model for my AI trader project.

### Context

I have an existing trading system with:
- **1H model**: XGBoost, 59.4% win rate, 2.35 profit factor (working well)
- **4H model**: XGBoost, 41.4% win rate, 1.40 profit factor (needs improvement)
- **Daily model**: Not yet trained

The goal is to combine predictions from all 3 timeframes to reduce noise and improve accuracy.

### Architecture

```
SHORT-TERM (1H) ──┐
    60% weight    │
                  ├──> WEIGHTED ENSEMBLE ──> FINAL PREDICTION
MEDIUM-TERM (4H) ─┤
    30% weight    │
                  │
LONG-TERM (Daily)─┘
    10% weight
```

**Key principle**: Short-term is dominant (60%) for entry timing. Medium/long-term (30%+10%) provide trend confirmation and reduce noise.

### What to Implement

**Phase 1: Train Daily Model**
- Use existing `ImprovedModelConfig.daily_model()` from `src/models/multi_timeframe/improved_model.py`
- Train on the same data as 1H/4H models
- Target: >55% validation accuracy

**Phase 2: Create MTF Ensemble Class**
- Create `src/models/multi_timeframe/mtf_ensemble.py`
- Combine predictions from 1H, 4H, Daily models
- Weighted average: 60% short, 30% medium, 10% long
- Agreement bonus: +5% confidence when all models agree
- Use existing `TechnicalEnsemble` in `src/models/ensemble/combiner.py` as reference

**Phase 3: Training Script**
- Create `scripts/train_mtf_ensemble.py`
- Train all 3 models (1H, 4H, D)
- Save to `models/mtf_ensemble/`

**Phase 4: Backtest Script**
- Create `scripts/backtest_mtf_ensemble.py`
- Test ensemble on 20% held-out data
- Compare to individual model performance

### Key Files to Read First

1. `docs/08-multi-timeframe-ensemble-implementation.md` - Full implementation guide
2. `src/models/multi_timeframe/improved_model.py` - Individual model implementation
3. `src/models/ensemble/combiner.py` - Existing ensemble framework
4. `scripts/train_improved_models.py` - Current training script
5. `scripts/backtest_improved.py` - Current backtest script

### Data Location

- 5-minute data: `data/forex/EURUSD_20200101_20251231_5min_combined.csv`
- Existing models: `models/improved_mtf/`

### Success Criteria

1. Daily model trains with >55% validation accuracy
2. Ensemble combines all 3 models correctly
3. Ensemble backtest shows:
   - Win rate >= 59% (at least match 1H model)
   - Profit factor >= 2.0
   - Higher win rate at high confidence levels

### Important Notes

- Follow CLAUDE.md instructions (proceed without asking for confirmation)
- Use XGBoost for all models (proven to work best)
- Test incrementally (Daily model first, then ensemble)
- Keep existing model files intact (new files in `models/mtf_ensemble/`)

Please start by reading the implementation guide at `docs/08-multi-timeframe-ensemble-implementation.md`, then proceed with the implementation.

## PROMPT END

---

## Additional Context (Optional)

If you want to provide more context, you can also mention:

- The project uses triple barrier labeling for realistic trade outcomes
- Features include technical indicators + enhanced features (time, ROC, cross-TF alignment)
- Current best model (1H) uses 115 features
- The existing `TechnicalEnsemble` class has regime-based weight adjustment built in
- Agreement score between models should boost confidence
