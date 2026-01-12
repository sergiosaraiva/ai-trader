# AI Trader

A Multi-Timeframe Ensemble Trading System using XGBoost for technical analysis-based forex trading predictions with sentiment analysis integration.

## Performance (Current Production Model)

| Metric | Value |
|--------|-------|
| **Total Profit** | **+7,987 pips** |
| **Win Rate** | 57.8% |
| **Profit Factor** | 2.22 |
| **Total Trades** | 1,103 |
| **Avg Pips/Trade** | +7.2 |

### Model Accuracy

| Model | Val Accuracy | High-Conf (≥60%) |
|-------|--------------|------------------|
| 1H (60% weight) | 67.07% | 72.14% |
| 4H (30% weight) | 65.43% | 71.12% |
| Daily (10% weight) | 61.54% | 64.21% |

## Features

- **Multi-Timeframe Ensemble**: Combines 1H, 4H, and Daily XGBoost models
- **Sentiment Integration**: VIX + EPU sentiment on Daily model (research-optimized)
- **115+ Technical Features**: Trend, momentum, volatility, patterns, cross-TF alignment
- **Triple Barrier Labeling**: Realistic TP/SL/Timeout trade outcomes
- **Confidence-Based Trading**: Higher accuracy on high-confidence signals
- **6 Years of Data**: Trained on EUR/USD 2020-2025 (448K 5-minute bars)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/sergiosaraiva/ai-trader.git
cd ai-trader

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
# Train with optimal configuration (sentiment on Daily only)
python scripts/train_mtf_ensemble.py --sentiment

# Train without sentiment
python scripts/train_mtf_ensemble.py
```

### Run Backtest

```bash
# Backtest the trained model
python scripts/backtest_mtf_ensemble.py --model-dir models/mtf_ensemble
```

### Make Predictions

```python
from src.models.multi_timeframe import MTFEnsemble, MTFEnsembleConfig

# Load trained ensemble
config = MTFEnsembleConfig.with_sentiment("EURUSD")
ensemble = MTFEnsemble(config=config, model_dir="models/mtf_ensemble")
ensemble.load()

# Predict on recent 5-minute data
prediction = ensemble.predict(df_5min_recent)

print(f"Direction: {'LONG' if prediction.direction == 1 else 'SHORT'}")
print(f"Confidence: {prediction.confidence:.1%}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MTF ENSEMBLE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   5-min Data ──┬──► 1H Model (60% weight) ───────┐              │
│                │                                  │              │
│                ├──► 4H Model (30% weight) ───────┼──► Ensemble  │
│                │                                  │   Prediction │
│                └──► Daily Model (10% weight) ────┘              │
│                     + VIX/EPU Sentiment                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
ai-trader/
├── configs/                 # Model and indicator configurations
│   ├── profiles/           # Trading profiles (scalper, trader, investor)
│   └── indicators/         # Technical indicator settings
├── data/
│   ├── forex/              # EUR/USD 5-minute data (2020-2025)
│   ├── sentiment/          # VIX, EPU, GDELT sentiment data
│   └── sample/             # Sample data for development
├── docs/                   # Documentation
├── models/
│   └── mtf_ensemble/       # Trained production models
├── scripts/
│   ├── train_mtf_ensemble.py    # Training script
│   ├── backtest_mtf_ensemble.py # Backtesting script
│   └── download_*.py            # Data download scripts
├── src/
│   ├── features/
│   │   ├── technical/      # Technical indicators
│   │   └── sentiment/      # Sentiment feature engineering
│   ├── models/
│   │   └── multi_timeframe/ # MTF ensemble implementation
│   ├── simulation/         # Backtesting engine
│   └── trading/            # Trading logic and risk management
└── tests/                  # Unit tests
```

## Data

### Price Data
- **Source**: MetaTrader 5
- **Pair**: EUR/USD
- **Resolution**: 5-minute (resampled to 1H, 4H, Daily)
- **Period**: 2020-01-01 to 2025-12-31
- **Records**: 448,586 bars

### Sentiment Data
- **VIX**: CBOE Volatility Index (daily) from FRED
- **EPU**: Economic Policy Uncertainty Index (daily) from FRED
- **GDELT**: Hourly news sentiment (available but not used in production)

## Key Findings

### Sentiment Integration Research

| Configuration | Total Pips | Profit Factor |
|---------------|------------|---------------|
| Baseline (No Sentiment) | +7,596 | 2.12 |
| **EPU Daily-Only (Optimal)** | **+7,987** | **2.22** |
| GDELT All Timeframes | +7,273 | 2.09 |

**Resolution Matching Rule**: Sentiment data resolution must match or be finer than the trading timeframe. Daily EPU/VIX works for Daily model; adding it to 1H/4H models degrades performance.

### What Works
- Multi-timeframe ensemble (60/30/10 weights)
- VIX + EPU sentiment on Daily model only
- Cross-timeframe trend alignment features
- Triple barrier labeling for realistic simulation

### What Doesn't Work
- Sentiment on intraday models (resolution mismatch)
- GDELT hourly sentiment (too noisy despite proper resolution)
- Equal timeframe weights

## Configuration

### Training Options

```bash
# With sentiment (recommended)
python scripts/train_mtf_ensemble.py --sentiment

# Custom sentiment configuration
python scripts/train_mtf_ensemble.py --sentiment-tf "D" --sentiment-source epu

# Custom weights
python scripts/train_mtf_ensemble.py --weights "0.6,0.3,0.1"
```

### Triple Barrier Parameters

| Timeframe | Take Profit | Stop Loss | Max Holding |
|-----------|-------------|-----------|-------------|
| 1H | 25 pips | 15 pips | 12 bars |
| 4H | 50 pips | 25 pips | 18 bars |
| Daily | 150 pips | 75 pips | 15 bars |

## Documentation

- [Current State of the Art](docs/15-current-state-of-the-art.md) - Comprehensive system documentation
- [Architecture Overview](docs/01-architecture-overview.md)
- [Technical Analysis Model Design](docs/02-technical-analysis-model-design.md)
- [Technical Indicators Configuration](docs/03-technical-indicators-configuration.md)
- [Confidence & Uncertainty System](docs/04-confidence-uncertainty-system.md)
- [Trading Robot Design](docs/05-trading-robot-design.md)
- [MTF Ensemble Implementation](docs/08-multi-timeframe-ensemble-implementation.md)
- [Sentiment Analysis Results](docs/13-sentiment-analysis-test-results.md)

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

## License

This project is for educational and research purposes.

## Acknowledgments

Developed with assistance from Claude (Anthropic).
