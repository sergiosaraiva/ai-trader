# CLAUDE.md - AI Assets Trader Project Guide

## Interaction Mode

**ALWAYS PROCEED WITHOUT ASKING FOR CONFIRMATION.**

- Do not ask for permission before taking actions
- Do not wait for user confirmation on implementation choices
- Make autonomous decisions based on best practices and project context
- Execute tasks fully without pausing for approval
- If multiple valid approaches exist, choose the most appropriate one and proceed
- Only ask questions if there is genuine ambiguity that cannot be resolved from context

The user trusts Claude to make good decisions. Act decisively and complete tasks end-to-end.

## Project Overview

AI Assets Trader is a **production-ready Multi-Timeframe (MTF) Ensemble trading system** for forex. The system uses XGBoost models across three timeframes (1H, 4H, Daily) combined with sentiment analysis to generate trading predictions.

**Current Status: Production-Ready (WFO Validated)**
- +7,987 pips profit on test period (single split)
- +18,136 pips across 7 WFO windows (100% profitable)
- 57.8% win rate, 2.22 profit factor
- VIX/EPU sentiment integration (Daily model only)
- Walk-Forward Optimization: PASSED (100% consistency)

## Current Performance

| Metric | Value |
|--------|-------|
| **Total Profit** | +7,987 pips |
| **Win Rate** | 57.8% |
| **Profit Factor** | 2.22 |
| **Total Trades** | 1,103 |
| **Avg Pips/Trade** | +7.2 |

### Model Accuracy

| Model | Weight | Val Accuracy | High-Conf (≥60%) |
|-------|--------|--------------|------------------|
| 1H | 60% | 67.07% | 72.14% |
| 4H | 30% | 65.43% | 71.12% |
| Daily | 10% | 61.54% | 64.21% |

## Project Structure

```
ai-trader/
├── data/
│   ├── forex/                     # EUR/USD 5-minute data (448K bars, 2020-2025)
│   │   └── EURUSD_*_5min_combined.csv
│   ├── sentiment/                 # Sentiment data
│   │   ├── sentiment_scores_*.csv # EPU + VIX daily sentiment
│   │   └── gdelt_sentiment_*.csv  # GDELT hourly (available, not used)
│   └── sample/                    # Sample data for development
├── docs/
│   ├── 15-current-state-of-the-art.md  # Comprehensive current state
│   └── ...                        # Architecture and design docs
├── models/
│   └── mtf_ensemble/              # Production models
│       ├── 1H_model.pkl           # 1-hour XGBoost model
│       ├── 4H_model.pkl           # 4-hour XGBoost model
│       ├── D_model.pkl            # Daily XGBoost model (with sentiment)
│       └── training_metadata.json # Configuration and results
├── scripts/
│   ├── train_mtf_ensemble.py      # Training script
│   ├── backtest_mtf_ensemble.py   # Backtesting script
│   ├── download_sentiment_data.py # EPU + VIX download
│   └── download_gdelt_sentiment.py # GDELT download (BigQuery)
├── src/
│   ├── features/
│   │   ├── technical/             # Technical indicators
│   │   │   ├── calculator.py      # TechnicalIndicatorCalculator
│   │   │   └── indicator_registry.py
│   │   └── sentiment/             # Sentiment features
│   │       ├── sentiment_loader.py # EPU/VIX loader
│   │       └── gdelt_loader.py    # GDELT loader
│   ├── models/
│   │   └── multi_timeframe/       # MTF Ensemble (PRIMARY)
│   │       ├── mtf_ensemble.py    # MTFEnsemble class
│   │       ├── improved_model.py  # ImprovedTimeframeModel
│   │       └── enhanced_features.py # EnhancedFeatureEngine
│   ├── simulation/                # Backtesting
│   └── trading/                   # Risk management
└── tests/
```

## MTF Ensemble Architecture

The production system uses a Multi-Timeframe Ensemble with weighted averaging:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MTF ENSEMBLE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   5-min Data ──┬──► 1H Model (60% weight) ───────┐              │
│                │    - 115 features               │              │
│                │    - No sentiment               │              │
│                │                                  │              │
│                ├──► 4H Model (30% weight) ───────┼──► Ensemble  │
│                │    - 113 features               │   Prediction │
│                │    - No sentiment               │              │
│                │                                  │              │
│                └──► Daily Model (10% weight) ────┘              │
│                     - 134 features                              │
│                     - VIX + EPU sentiment                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Optimal Configuration

```python
MTFEnsembleConfig(
    weights={"1H": 0.6, "4H": 0.3, "D": 0.1},
    agreement_bonus=0.05,
    use_regime_adjustment=True,
    include_sentiment=True,
    sentiment_source="epu",
    sentiment_by_timeframe={"1H": False, "4H": False, "D": True},
    trading_pair="EURUSD",
)
```

### Key Finding: Resolution Matching Rule

**Sentiment data resolution must match or be finer than the trading timeframe.**

| Configuration | Total Pips | Profit Factor |
|---------------|------------|---------------|
| Baseline (No Sentiment) | +7,596 | 2.12 |
| **EPU Daily-Only (Optimal)** | **+7,987** | **2.22** |
| GDELT All Timeframes | +7,273 | 2.09 |

Daily EPU/VIX sentiment works for Daily model; adding it to 1H/4H models degrades performance.

## Data Sources

### Price Data
- **Source**: MetaTrader 5
- **Pair**: EUR/USD
- **Resolution**: 5-minute (resampled to 1H, 4H, Daily)
- **Period**: 2020-01-01 to 2025-12-31
- **Records**: 448,586 bars
- **Location**: `data/forex/EURUSD_20200101_20251231_5min_combined.csv`

### Sentiment Data
| Dataset | Source | Resolution | Location |
|---------|--------|------------|----------|
| US EPU | FRED | Daily | `data/sentiment/sentiment_scores_*.csv` |
| VIX | FRED | Daily | `data/sentiment/sentiment_scores_*.csv` |
| GDELT | BigQuery | Hourly | `data/sentiment/gdelt_sentiment_*.csv` |

## Walk-Forward Optimization (WFO)

The model has been validated using walk-forward optimization with 7 rolling windows:

| Window | Test Period | Trades | Win Rate | Pips | PF |
|--------|-------------|--------|----------|------|-----|
| 1 | 2022-H1 | 533 | 57.2% | +3,939 | 2.20 |
| 2 | 2022-H2 | 662 | 54.7% | +4,358 | 1.98 |
| 3 | 2023-H1 | 478 | 47.7% | +1,455 | 1.41 |
| 4 | 2023-H2 | 425 | 48.7% | +1,635 | 1.56 |
| 5 | 2024-H1 | 334 | 62.0% | +2,432 | 2.55 |
| 6 | 2024-H2 | 380 | 56.6% | +2,238 | 2.08 |
| 7 | 2025-H1 | 568 | 47.7% | +2,079 | 1.48 |

**Summary:** 100% consistency (7/7 windows profitable), +18,136 total pips, CV=0.40 (stable)

See `docs/17-walk-forward-optimization-results.md` for full analysis.

## Common Commands

### Training

```bash
# Train with optimal configuration (sentiment on Daily only)
python scripts/train_mtf_ensemble.py --sentiment

# Train without sentiment (baseline)
python scripts/train_mtf_ensemble.py

# Custom sentiment configuration
python scripts/train_mtf_ensemble.py --sentiment-tf "D" --sentiment-source epu

# Custom weights
python scripts/train_mtf_ensemble.py --weights "0.6,0.3,0.1"
```

### Backtesting

```bash
# Backtest the trained model
python scripts/backtest_mtf_ensemble.py --model-dir models/mtf_ensemble

# With comparison to individual models
python scripts/backtest_mtf_ensemble.py --compare
```

### Walk-Forward Optimization

```bash
# Run WFO validation (24-month train, 6-month test windows)
python scripts/walk_forward_optimization.py --sentiment

# Custom window configuration
python scripts/walk_forward_optimization.py --sentiment --train-months 24 --test-months 6 --step-months 6
```

### Data Download

```bash
# Download EPU + VIX sentiment data
python scripts/download_sentiment_data.py --start 2020-01-01 --end 2025-12-31

# Download GDELT sentiment (requires Google Cloud credentials)
export GOOGLE_APPLICATION_CREDENTIALS="credentials/gcloud.json"
python scripts/download_gdelt_sentiment.py --start 2020-01-01 --end 2025-12-31
```

## Feature Engineering

### Feature Categories (per model)

| Category | 1H | 4H | Daily | Description |
|----------|-----|-----|-------|-------------|
| Trend | 15 | 15 | 15 | EMA, SMA, Supertrend, ADX |
| Momentum | 12 | 12 | 12 | RSI, MACD, Stochastic, MFI |
| Volatility | 10 | 10 | 10 | ATR, Bollinger, Keltner |
| Time | 12 | 8 | 6 | Hour, day, session flags |
| Pattern | 15 | 15 | 15 | Highs, lows, engulfing |
| Cross-TF | 8 | 5 | 0 | HTF trend, HTF RSI |
| Sentiment | 0 | 0 | 24 | VIX, EPU, MAs, z-scores |
| **Total** | **115** | **113** | **134** | |

### Top Features by Model

**1H Model**: is_newyork, htf_4H_trend, supertrend_dir, trend_alignment, dist_ema_21
**4H Model**: price_pctl_50, dist_ema_55, htf_D_rsi, macd, rsi_14_pctl
**Daily Model**: sentiment_ma_30, ema_21, sma_50, ema_55, supertrend_10

## Triple Barrier Labeling

| Timeframe | Take Profit | Stop Loss | Max Holding |
|-----------|-------------|-----------|-------------|
| 1H | 25 pips | 15 pips | 12 bars |
| 4H | 50 pips | 25 pips | 18 bars |
| Daily | 150 pips | 75 pips | 15 bars |

## Key Files Reference

### Production Models
| File | Purpose |
|------|---------|
| `models/mtf_ensemble/1H_model.pkl` | 1-hour XGBoost model |
| `models/mtf_ensemble/4H_model.pkl` | 4-hour XGBoost model |
| `models/mtf_ensemble/D_model.pkl` | Daily XGBoost model (with sentiment) |
| `models/mtf_ensemble/training_metadata.json` | Configuration and results |

### Core Implementation
| File | Purpose |
|------|---------|
| `src/models/multi_timeframe/mtf_ensemble.py` | MTFEnsemble class, MTFEnsembleConfig |
| `src/models/multi_timeframe/improved_model.py` | ImprovedTimeframeModel, labeling |
| `src/models/multi_timeframe/enhanced_features.py` | EnhancedFeatureEngine |
| `src/features/sentiment/sentiment_loader.py` | EPU/VIX sentiment loading |
| `src/features/sentiment/gdelt_loader.py` | GDELT hourly sentiment |
| `src/features/technical/calculator.py` | Technical indicator calculation |

### Scripts
| File | Purpose |
|------|---------|
| `scripts/train_mtf_ensemble.py` | Training with all options |
| `scripts/backtest_mtf_ensemble.py` | Backtesting simulation |
| `scripts/walk_forward_optimization.py` | WFO validation (robustness testing) |
| `scripts/download_sentiment_data.py` | EPU + VIX download |
| `scripts/download_gdelt_sentiment.py` | GDELT BigQuery download |

### Documentation
| File | Purpose |
|------|---------|
| `docs/15-current-state-of-the-art.md` | **Comprehensive current state** |
| `docs/17-walk-forward-optimization-results.md` | **WFO validation results** |
| `docs/13-sentiment-analysis-test-results.md` | Sentiment integration results |
| `docs/08-multi-timeframe-ensemble-implementation.md` | MTF implementation details |

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| ML Models | XGBoost |
| Indicators | pandas-ta |
| Data | pandas, numpy |
| Sentiment | FRED API, Google BigQuery |

## Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Win Rate | > 55% | 57.8% |
| Profit Factor | > 2.0 | 2.22 |
| High-Conf Accuracy | > 65% | 72.14% (1H @ ≥60%) |
| Total Pips | > 0 | +7,987 |

## Coding Conventions

### Python Style
- PEP 8 compliant
- Type hints required
- Google-style docstrings
- Max line length: 100

### Time Series Rules
- **CRITICAL**: Always use chronological splits (no future data leakage)
- Train/Val/Test must be sequential in time (60/20/20)
- Store scalers with models
- Validate on out-of-sample data

### Model Development
- Use `ImprovedTimeframeModel` from `src/models/multi_timeframe/improved_model.py`
- Use `EnhancedFeatureEngine` for feature engineering
- Log experiments to metadata JSON

## Notes for Claude

- **ALWAYS PROCEED AUTONOMOUSLY** - Never ask for confirmation
- **PRIMARY MODEL**: MTF Ensemble in `src/models/multi_timeframe/`
- **SENTIMENT**: EPU/VIX on Daily model only (resolution matching principle)
- Always consider data leakage when working with time series
- Use the 5-minute combined data in `data/forex/` for training
- Reference `docs/15-current-state-of-the-art.md` for detailed system documentation
- The optimal configuration is already saved in `models/mtf_ensemble/`
