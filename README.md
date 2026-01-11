# AI Trader

A Multi-Timeframe Ensemble Trading System using machine learning for technical analysis-based trading predictions.

## Performance

| Model | Win Rate | Profit Factor | Total Pips |
|-------|----------|---------------|------------|
| **1H alone** | 60.0% | 2.42 | +9,529 |
| **Ensemble (filter mode)** | 58.5% | 2.33 | +8,263 |
| **Ensemble (strict filter, ≥65% conf)** | 63.3% | - | - |

## Features

- **Multi-Timeframe Analysis**: Combines 1H, 4H, and Daily timeframes
- **Multiple Trading Modes**:
  - Weighted averaging (configurable weights)
  - Filter mode (1H primary, 4H/D as confirmation)
  - Strict filter (requires 4H agreement)
- **Confidence-Based Filtering**: Trade only on high-confidence signals
- **XGBoost Models**: Fast training and inference
- **Triple Barrier Labeling**: TP/SL/Timeout trade outcomes
- **Comprehensive Backtesting**: Full historical simulation with metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/sergiosaraiva/ai-trader.git
cd ai-trader

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train the MTF Ensemble

```bash
python scripts/train_mtf_ensemble.py \
    --timeframes "1H,4H,D" \
    --weights "0.8,0.15,0.05"
```

### 2. Run Backtest

```bash
# Standard weighted mode
python scripts/backtest_mtf_ensemble.py --compare

# Filter mode (recommended)
python scripts/backtest_mtf_ensemble.py --filter-mode --compare

# Strict filter (highest quality trades)
python scripts/backtest_mtf_ensemble.py --strict-filter --compare
```

## Project Structure

```
ai-trader/
├── configs/                 # Model and indicator configurations
│   ├── profiles/           # Trading profiles (scalper, trader, investor)
│   └── indicators/         # Technical indicator settings per timeframe
├── data/
│   └── sample/             # Sample forex data for development
├── docs/                   # Documentation
├── scripts/                # Training and backtesting scripts
├── src/
│   ├── config/            # Configuration management
│   ├── data/              # Data loading and processing
│   ├── features/          # Technical indicators
│   ├── models/            # ML models
│   │   ├── confidence/    # Uncertainty estimation
│   │   ├── ensemble/      # Model combination
│   │   ├── multi_timeframe/  # MTF ensemble implementation
│   │   └── technical/     # Time-horizon models
│   ├── simulation/        # Backtesting engine
│   └── trading/           # Trading logic and risk management
└── tests/                 # Unit tests
```

## Trading Modes

### Weighted Mode (Default)
Combines predictions using configurable weights:
```python
weights = {"1H": 0.8, "4H": 0.15, "D": 0.05}
```

### Filter Mode
Uses 1H as primary signal, 4H/D adjust confidence:
- 4H agreement: +5% confidence boost
- 4H strong disagreement: -15% penalty
- Daily has smaller effect

### Strict Filter Mode
Only trades when 1H AND 4H agree:
- Fewer trades, higher quality
- Best for risk-averse strategies
- 63.3% win rate on high-confidence trades

## Configuration

### Trading Profiles

| Profile | Short-Term | Medium-Term | Long-Term |
|---------|------------|-------------|-----------|
| Scalper | 15m | 1H | 4H |
| Trader | 1H | 4H | 1D |
| Investor | 1D | 1W | 1M |

### Triple Barrier Parameters

| Timeframe | Take Profit | Stop Loss | Max Holding |
|-----------|-------------|-----------|-------------|
| 1H | 25 pips | 15 pips | 12 bars |
| 4H | 50 pips | 25 pips | 18 bars |
| Daily | 150 pips | 75 pips | 15 bars |

## Documentation

- [Architecture Overview](docs/01-architecture-overview.md)
- [Technical Analysis Model Design](docs/02-technical-analysis-model-design.md)
- [Technical Indicators Configuration](docs/03-technical-indicators-configuration.md)
- [Confidence & Uncertainty System](docs/04-confidence-uncertainty-system.md)
- [Trading Robot Design](docs/05-trading-robot-design.md)
- [MTF Ensemble Implementation](docs/08-multi-timeframe-ensemble-implementation.md)

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| ML Models | XGBoost, PyTorch |
| Indicators | pandas-ta |
| Data | pandas, numpy |
| Tracking | MLflow |

## Sample Data

Sample forex data is included in `data/sample/`:
- `EURUSD_daily.csv` - EUR/USD (1286 rows, 2020-2024)
- `GBPUSD_daily.csv` - GBP/USD (1286 rows, 2020-2024)
- `USDJPY_daily.csv` - USD/JPY (1286 rows, 2020-2024)

## License

This project is for educational and research purposes.

## Acknowledgments

Developed with assistance from Claude (Anthropic).
