# Continuation Prompt for AI Trader Project

Copy and paste the following into a new Claude Code session:

---

## Project Context

I'm working on the **AI Trader** project located at `/home/sergio/ai-trader`. This is a Multi-Timeframe (MTF) Ensemble trading system for EUR/USD forex using XGBoost models.

### Current State (Production-Ready)

The system has achieved:
- **+7,987 pips** total profit on test period
- **57.8% win rate**, **2.22 profit factor**
- 1,103 trades over ~1.2 year test period

### Architecture

Three XGBoost models combined via weighted averaging:
- **1H Model (60% weight)**: 115 features, no sentiment, 67.07% val accuracy
- **4H Model (30% weight)**: 113 features, no sentiment, 65.43% val accuracy
- **Daily Model (10% weight)**: 134 features, WITH VIX/EPU sentiment, 61.54% val accuracy

### Key Finding: Resolution Matching Principle

**Sentiment data resolution must match or be finer than the trading timeframe.**

| Configuration | Total Pips | Profit Factor |
|---------------|------------|---------------|
| Baseline (No Sentiment) | +7,596 | 2.12 |
| **EPU Daily-Only (Current)** | **+7,987** | **2.22** |
| GDELT All Timeframes | +7,273 | 2.09 |

Adding daily EPU/VIX to intraday models (1H, 4H) degraded performance. GDELT hourly sentiment, despite proper resolution, also underperformed.

### Data Available

- **Price**: EUR/USD 5-minute data, 448K bars, 2020-2025 (`data/forex/EURUSD_20200101_20251231_5min_combined.csv`)
- **Sentiment**:
  - EPU + VIX daily (`data/sentiment/sentiment_scores_*.csv`)
  - GDELT hourly (`data/sentiment/gdelt_sentiment_20200101_20251231_hourly.csv`) - downloaded but not used in production

### Key Commands

```bash
# Train with optimal config (sentiment on Daily only)
python scripts/train_mtf_ensemble.py --sentiment

# Backtest
python scripts/backtest_mtf_ensemble.py --model-dir models/mtf_ensemble

# Train baseline (no sentiment)
python scripts/train_mtf_ensemble.py --output models/mtf_ensemble_baseline
```

### Key Documentation

Read these for full context:
1. **`docs/15-current-state-of-the-art.md`** - Comprehensive system documentation
2. **`CLAUDE.md`** - Project guide and conventions
3. **`models/mtf_ensemble/training_metadata.json`** - Current model configuration

### Potential Next Steps

1. **Walk-Forward Optimization**: Rolling window training for robustness validation
2. **Additional Currency Pairs**: Extend to GBP/USD, USD/JPY
3. **Regime Detection**: Adjust weights based on trending/ranging/volatile markets
4. **Real-Time Integration**: Connect to MT5 or broker API for live trading
5. **Risk Management**: Implement Kelly criterion position sizing
6. **Hyperparameter Tuning**: Systematic optimization of XGBoost parameters
7. **Alternative Sentiment Sources**: Test other data sources that might work better than GDELT

### Project Structure

```
ai-trader/
├── data/forex/              # EUR/USD 5-min data
├── data/sentiment/          # EPU, VIX, GDELT
├── models/mtf_ensemble/     # Production models
├── scripts/                 # Training & backtesting
├── src/models/multi_timeframe/  # Core implementation
└── docs/                    # Documentation
```

---

**Please start by reading `docs/15-current-state-of-the-art.md` and `CLAUDE.md` to understand the current state, then let me know what aspect you'd like to work on.**

---

## Alternative Short Version

If you prefer a shorter prompt:

---

I'm working on the **AI Trader** project at `/home/sergio/ai-trader`. It's a production-ready MTF Ensemble forex trading system achieving +7,987 pips with 2.22 profit factor.

Please read these files first:
1. `docs/15-current-state-of-the-art.md` - Complete system documentation
2. `CLAUDE.md` - Project guide

The system uses 3 XGBoost models (1H/4H/Daily) with VIX/EPU sentiment on Daily model only. Key finding: sentiment resolution must match timeframe resolution.

After reading the docs, let me know what you'd like to work on next.

---
