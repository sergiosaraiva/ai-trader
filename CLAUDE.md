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

AI Assets Trader is a **Technical Analysis-focused** trading system for forex and other assets. The system uses deep learning models to analyze price patterns and technical indicators to generate trading predictions across multiple timeframes.

**Current Focus: Technical Analysis Only** - Fundamental and Sentiment analysis are out of scope for now.

## Current Development Phase

**Phase 1: Technical Analysis Foundation**
- Building core technical analysis module
- Three time-horizon models (Short, Medium, Long-term)
- Ensemble combination
- Backtesting and simulation

## Project Structure

```
ai-trader/
├── data/
│   └── sample/                    # Sample data for development
│       ├── EURUSD_daily.csv       # EUR/USD forex data (1286 rows)
│       ├── GBPUSD_daily.csv       # GBP/USD forex data
│       └── USDJPY_daily.csv       # USD/JPY forex data
├── docs/
│   ├── 01-architecture-overview.md
│   ├── 02-technical-analysis-model-design.md
│   └── 03-technical-indicators-configuration.md
├── src/
│   ├── config/                    # Configuration management
│   ├── data/                      # Data sources and processors
│   │   ├── sources/               # MT5, Alpaca, Yahoo connectors
│   │   ├── processors/            # OHLCV and feature processing
│   │   └── storage/               # Database and cache
│   ├── features/
│   │   └── technical/             # Technical indicators (FOCUS)
│   │       ├── indicators.py      # Main indicator class
│   │       ├── trend.py           # SMA, EMA, ADX, Aroon, etc.
│   │       ├── momentum.py        # RSI, MACD, Stochastic, etc.
│   │       ├── volatility.py      # ATR, Bollinger, Keltner, etc.
│   │       └── volume.py          # OBV, CMF, VWAP, etc.
│   ├── models/
│   │   ├── base.py                # BaseModel, Prediction schema
│   │   ├── technical/             # Technical analysis models (FOCUS)
│   │   │   ├── short_term.py      # CNN-LSTM-Attention (1H-4H)
│   │   │   ├── medium_term.py     # TFT-style (Daily)
│   │   │   └── long_term.py       # N-BEATS + Transformer (Weekly)
│   │   ├── confidence/            # Confidence & uncertainty (CRITICAL)
│   │   │   ├── calibration.py     # Temperature, Platt, Isotonic
│   │   │   ├── uncertainty.py     # MC Dropout, Ensemble disagreement
│   │   │   ├── learned_uncertainty.py  # Beta/Dirichlet outputs (RECOMMENDED)
│   │   │   └── integration.py     # ConfidenceAwarePredictor
│   │   └── ensemble/              # Model combination
│   │       ├── combiner.py        # TechnicalEnsemble
│   │       └── meta_model.py      # Stacking meta-learner
│   ├── simulation/                # Backtesting (FOCUS)
│   │   ├── backtester.py          # Historical simulation
│   │   ├── paper_trading.py       # Live simulation
│   │   └── metrics.py             # Performance metrics
│   └── trading/                   # Trading engine
├── configs/                       # Model configurations (YAML)
│   ├── profiles/                  # Trading profile configurations
│   │   ├── base.yaml              # Base profile (all defaults)
│   │   ├── scalper.yaml           # Scalper profile (15m, 1H, 4H)
│   │   ├── trader.yaml            # Trader profile (1H, 4H, 1D)
│   │   ├── investor.yaml          # Investor profile (1D, 1W, 1M)
│   │   └── assets/                # Asset-specific configurations
│   │       ├── forex.yaml         # Forex trading settings
│   │       ├── crypto.yaml        # Cryptocurrency settings
│   │       └── stocks.yaml        # Stock/equity settings
│   ├── timeframe_transform.yaml   # Timeframe transformation config
│   └── indicators/                # Indicator configurations per model
│       ├── short_term_indicators.yaml
│       ├── medium_term_indicators.yaml
│       └── long_term_indicators.yaml
├── tests/                         # Test suite
├── notebooks/                     # Research notebooks
└── mlruns/                        # MLflow experiments
```

## Sample Data

Sample forex data is available in `data/sample/`:

| File | Symbol | Rows | Date Range | Description |
|------|--------|------|------------|-------------|
| `EURUSD_daily.csv` | EUR/USD | 1286 | 2020-2024 | Primary development data |
| `GBPUSD_daily.csv` | GBP/USD | 1286 | 2020-2024 | Secondary pair |
| `USDJPY_daily.csv` | USD/JPY | 1286 | 2020-2024 | Yen cross |

**CSV Format:**
```csv
Date,Open,High,Low,Close,Volume
2020-01-01,1.11958,1.12068,1.11762,1.11919,74219
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Language | Python 3.11+ | Core development |
| Deep Learning | PyTorch | Model implementation |
| Time Series | Darts | Forecasting utilities |
| Indicators | pandas-ta, TA-Lib | Technical analysis |
| Tracking | MLflow | Experiment management |
| API | FastAPI | REST endpoints |

## Technical Analysis Models

The system uses a **comprehensive Profile Configuration System** with inheritance to support different trading styles and asset classes.

### Profile System Architecture

```
base.yaml (all defaults)
    ├── scalper.yaml (inherits base, overrides for scalping)
    ├── trader.yaml (inherits base, overrides for day/swing trading)
    └── investor.yaml (inherits base, overrides for long-term investing)

Asset Profiles (merged with trading profiles):
    ├── forex.yaml (forex-specific settings: sessions, pairs, spreads)
    ├── crypto.yaml (crypto-specific: 24/7, high volatility, funding rates)
    └── stocks.yaml (stock-specific: market hours, earnings, PDT rules)
```

### Timeframe Profiles

| Profile | Target User | Short-Term | Medium-Term | Long-Term |
|---------|-------------|------------|-------------|-----------|
| **Scalper** | Scalpers, fast day traders | 15m | 1H | 4H |
| **Trader** | Day/Swing traders | 1H | 4H | 1D |
| **Investor** | Long-term investors | 1D | 1W | 1M |

### Profile Configuration Files

**Trading Profiles** (inherit from base):
- `configs/profiles/base.yaml` - Complete base profile with all default settings
- `configs/profiles/scalper.yaml` - Fast scalping profile (15m, 1H, 4H)
- `configs/profiles/trader.yaml` - Active trading profile (1H, 4H, 1D)
- `configs/profiles/investor.yaml` - Long-term investment profile (1D, 1W, 1M)

**Asset Profiles** (merged with trading profiles):
- `configs/profiles/assets/forex.yaml` - Forex pairs, sessions, pip values
- `configs/profiles/assets/crypto.yaml` - Crypto 24/7, volatility, exchanges
- `configs/profiles/assets/stocks.yaml` - Market hours, earnings, circuit breakers

### Profile Sections

Each comprehensive profile includes:
- **timeframes** - Candle sizes, input windows, prediction horizons
- **indicators** - Technical indicators per timeframe (trend, momentum, volatility, volume)
- **models** - Neural network architecture per timeframe
- **training** - Batch size, epochs, learning rate, early stopping
- **ensemble** - Model weights, dynamic regime adjustment
- **signals** - Thresholds, confidence levels, cooldown periods
- **risk** - Position sizing, stop-loss, take-profit, trailing stops
- **sessions** - Trading hours, overlap bonuses, news blackouts
- **data** - History requirements, gap handling
- **regime** - Market regime detection settings
- **backtesting** - Walk-forward, Monte Carlo parameters

### Using Profiles in Code

```python
from src.config.profile_loader import ProfileLoader, load_profile

# Simple usage
config = load_profile("trader", asset="forex")

# Full usage
loader = ProfileLoader()
config = loader.load_profile("scalper", asset="crypto")

# Access values
timeframes = config.timeframes
short_term_candles = config.timeframes.short_term.candle_minutes
rsi_periods = config.indicators.short_term.momentum.rsi.periods

# Validate profile
result = loader.validate_profile("trader")
print(result["valid"], result["errors"], result["warnings"])
```

### 1. Short-Term Model
- **Architecture**: CNN + Bi-LSTM + Multi-Head Attention
- **Scalper Profile**: 15m candles, 192 input (2 days), predicts 15m/30m/1H/2H
- **Trader Profile**: 1H candles, 168 input (7 days), predicts 1H/4H/12H/24H
- **Investor Profile**: 1D candles, 90 input (3 months), predicts 1D/3D/5D/7D
- **File**: `src/models/technical/short_term.py`

### 2. Medium-Term Model
- **Architecture**: Temporal Fusion Transformer (simplified)
- **Scalper Profile**: 1H candles, 168 input (7 days), predicts 1H/2H/4H/8H
- **Trader Profile**: 4H candles, 180 input (30 days), predicts 4H/12H/24H/48H
- **Investor Profile**: 1W candles, 52 input (1 year), predicts 1W/2W/4W
- **File**: `src/models/technical/medium_term.py`

### 3. Long-Term Model
- **Architecture**: N-BEATS + Transformer hybrid
- **Scalper Profile**: 4H candles, 180 input (30 days), predicts 4H/8H/12H/24H
- **Trader Profile**: 1D candles, 90 input (3 months), predicts 1D/3D/5D/7D
- **Investor Profile**: 1M candles, 36 input (3 years), predicts 1M/2M/3M
- **File**: `src/models/technical/long_term.py`

### Ensemble
- **Methods**: Weighted average, Stacking, Attention-based fusion
- **Dynamic Weights**: Adjusted by market regime and recent performance
- **File**: `src/models/ensemble/combiner.py`

## Recommended Pipeline Architecture

The system implements an **enhanced prediction pipeline** with learned uncertainty and confidence-aware trading.

### Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ENHANCED TRADING PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘

Phase 1: DATA COLLECTION
    └── Fetch high-resolution data (1m/5m)
    └── Data quality validation (gaps, outliers)
    └── Store in Parquet format

Phase 2: FEATURE ENGINEERING
    ├── Multi-Timeframe Aggregation (15m/1H/4H/1D/1W)
    ├── Technical Indicators (per timeframe via registry)
    ├── Cross-Timeframe Features (trend alignment, divergence)
    └── Market Regime Detection (trending/ranging/volatile)

Phase 3: LABELING (Triple Barrier Method)
    ├── Take-profit barrier → Label: +1 (profitable)
    ├── Stop-loss barrier → Label: -1 (loss)
    └── Time barrier → Label: 0 (no trade)

Phase 4: MODEL TRAINING (with Learned Uncertainty)
    ├── Short-Term: CNN-LSTM-Attention + Beta Output
    ├── Medium-Term: TFT + Beta Output
    ├── Long-Term: N-BEATS + Beta Output
    └── Each model outputs: (direction, confidence, uncertainty)

Phase 5: ENSEMBLE COMBINATION (Dynamic Weights)
    ├── Weights based on recent performance
    ├── Weights based on market regime
    └── Ensemble disagreement → additional uncertainty

Phase 6: SIGNAL GENERATION (Confidence-Aware)
    ├── Direction from Beta distribution mean
    ├── Confidence from concentration (α + β)
    └── Position sizing based on confidence level
```

### Confidence-Aware Model Outputs

**CRITICAL**: Models use **Beta distribution outputs** instead of sigmoid for learned uncertainty.

| Output Type | Traditional (Sigmoid) | Recommended (Beta) |
|-------------|----------------------|-------------------|
| Prediction | Single value 0-1 | Distribution Beta(α, β) |
| Direction | Derived from threshold | Mean = α/(α+β) |
| Confidence | Derived from distance to 0.5 | **Learned** from concentration |
| Uncertainty | Not available | Explicit from distribution |

**Why Beta is Better:**
- Model explicitly learns "how sure am I?" via concentration (α + β)
- Low concentration = "I don't know" (skip trade)
- High concentration = "I'm confident" (full position)
- Single forward pass (efficient)

### Confidence System Files

```
src/models/confidence/
├── __init__.py                 # Module exports
├── calibration.py              # Temperature, Platt, Isotonic scaling
├── uncertainty.py              # MC Dropout, Ensemble disagreement
├── learned_uncertainty.py      # Beta/Dirichlet/Gaussian outputs (RECOMMENDED)
├── integration.py              # ConfidenceAwarePredictor
├── examples.py                 # Usage demonstrations
└── comparison_demo.py          # Why Beta > Sigmoid
```

### Position Sizing Based on Confidence

```
Confidence Level     Position Size    Action
─────────────────────────────────────────────
≥ 90% (very_high)    100%            Full position
80-90% (high)        75%             Large position
70-80% (moderate)    50%             Medium position
60-70% (low)         25%             Small position
< 60%                0%              NO TRADE
```

### Using the Confidence System

```python
from src.models.confidence import (
    BetaOutputLayer,
    BetaNLLLoss,
    TradingModelWithUncertainty,
    ConfidenceAwarePredictor,
)

# Option 1: Add Beta output to existing model
class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.features = nn.Sequential(...)
        self.output = BetaOutputLayer(hidden_dim)  # Replaces nn.Linear + Sigmoid

    def forward(self, x):
        features = self.features(x)
        return self.output(features)  # Returns BetaPrediction

# Train with Beta loss
model = MyModel(input_dim=50, hidden_dim=128)
criterion = BetaNLLLoss()

# Option 2: Use pre-built model
model = TradingModelWithUncertainty(
    input_dim=50,
    hidden_dim=128,
    output_type='beta'  # or 'dirichlet', 'gaussian', 'all'
)

# Option 3: Confidence-aware ensemble
predictor = ConfidenceAwarePredictor(
    models={'short': short_model, 'medium': medium_model, 'long': long_model},
    model_weights={'short': 0.5, 'medium': 0.3, 'long': 0.2},
    min_confidence=0.60
)
prediction = predictor.predict(features)

if prediction.should_trade:
    execute_trade(
        direction=prediction.direction,
        size=base_position * prediction.position_size
    )
```

### Running Confidence Demos

```bash
# Basic confidence examples
python -m src.models.confidence.examples

# Why Beta > Sigmoid comparison
python -m src.models.confidence.comparison_demo
```

## Technical Indicators Configuration

**IMPORTANT**: Technical indicators are **configurable per model** for flexibility in testing different combinations. Each model (Short, Medium, Long-term) has its own indicator configuration optimized for its timeframe.

### Indicator Configuration Files

| Model | Config File | Features | Focus |
|-------|-------------|----------|-------|
| Short-Term | `configs/indicators/short_term_indicators.yaml` | ~37 | Fast signals, intraday |
| Medium-Term | `configs/indicators/medium_term_indicators.yaml` | ~54 | Balanced, swing |
| Long-Term | `configs/indicators/long_term_indicators.yaml` | ~42 | Major trends, regime |

See `docs/03-technical-indicators-configuration.md` for full specification.

### Default Indicators by Model

#### Short-Term (1H-4H) - Fast Response
| Category | Key Indicators | Rationale |
|----------|---------------|-----------|
| Trend | EMA(8,13,21,55), Supertrend | Quick trend detection |
| Momentum | RSI(7,14), Stochastic(5,3,3), MACD | Fast reversals |
| Volatility | ATR(14), Bollinger(20,2) | Position sizing, squeeze |
| Volume | OBV, VWAP, Force Index | Intraday value |

#### Medium-Term (Daily) - Balanced
| Category | Key Indicators | Rationale |
|----------|---------------|-----------|
| Trend | SMA(20,50,100,200), ADX(14), Aroon | Major levels, trend strength |
| Momentum | RSI(14), MACD, CCI(20), TSI | Confirmation signals |
| Volatility | ATR(14), Bollinger, Keltner | Regime detection |
| Volume | OBV, CMF(20), A/D Line | Volume confirmation |

#### Long-Term (Weekly) - Major Trends
| Category | Key Indicators | Rationale |
|----------|---------------|-----------|
| Trend | SMA(10,20,50), ADX(14), Ichimoku | Major weekly trends |
| Momentum | RSI(14), MACD, ROC(10,20) | Reliable weekly signals |
| Volatility | ATR(14), Donchian(20), HV | Breakouts, regime |
| Volume | OBV | Weekly accumulation |

### Indicator Priority Levels

- **P0 (Critical)**: Must have - RSI, MACD, ATR, major MAs
- **P1 (Important)**: Significantly improves accuracy
- **P2 (Useful)**: Adds value, can be omitted
- **P3 (Optional)**: Nice to have, disable first if needed

### Technical Indicators Implemented

#### Trend (src/features/technical/trend.py)
- Moving Averages: SMA, EMA, WMA, DEMA, TEMA
- ADX, +DI, -DI
- Aroon Up/Down/Oscillator
- Supertrend, Ichimoku
- MA Crossovers

#### Momentum (src/features/technical/momentum.py)
- RSI (multiple periods)
- MACD, Signal, Histogram
- Stochastic %K, %D
- CCI, Momentum, ROC
- Williams %R, MFI
- TSI, Ultimate Oscillator

#### Volatility (src/features/technical/volatility.py)
- ATR, NATR, True Range
- Bollinger Bands (Upper, Middle, Lower, Width, %B)
- Keltner Channel
- Donchian Channel
- Historical Volatility

#### Volume (src/features/technical/volume.py)
- OBV, A/D Line
- Chaikin Money Flow
- VWAP, VPT
- Force Index, EMV

## Common Commands

```bash
# Load a trading profile with asset settings
python -c "
from src.config.profile_loader import ProfileLoader, load_profile

# Load trader profile with forex settings
config = load_profile('trader', asset='forex')
print('Timeframes:', config.timeframes.to_dict())

# List available profiles
loader = ProfileLoader()
print('Profiles:', loader.list_profiles())
print('Assets:', loader.list_assets())
"

# Transform 5-minute data to multiple timeframes with sliding window
python scripts/transform_timeframe.py --config configs/timeframe_transform.yaml

# Or with CLI arguments
python scripts/transform_timeframe.py \
    --input data/forex \
    --output data/forex/derived \
    --base-minutes 5 \
    --target-minutes 15 60 240 1440

# Load and process sample data
python -c "
from src.data.processors import OHLCVProcessor
from src.features.technical import TechnicalIndicators
import pandas as pd

df = pd.read_csv('data/sample/EURUSD_daily.csv', parse_dates=['Date'], index_col='Date')
df.columns = [c.lower() for c in df.columns]

indicators = TechnicalIndicators()
df_features = indicators.calculate_all(df)
print(df_features.head())
"

# Run tests
pytest tests/

# Start API
uvicorn src.api.main:app --reload

# MLflow UI
mlflow ui --port 5000
```

## Development Priorities

### Immediate (Current Sprint)
1. End-to-end training pipeline
2. Model training on sample data
3. Backtesting validation

### Next
4. Walk-forward validation
5. Hyperparameter tuning
6. Performance optimization

### Later
7. Paper trading mode
8. API integration
9. Real-time predictions

## Trading Robot

The system includes a **world-class trading robot** that uses trained models to generate trading signals with configurable risk aversion and comprehensive loss protection.

### Trading Robot Features

1. **Confidence-Based Decisions**: Uses Beta distribution outputs to determine BOTH direction AND confidence
2. **Configurable Risk Profiles**: Five levels from ultra-conservative to ultra-aggressive
3. **Circuit Breakers**: Automatic trading halt on adverse conditions
4. **Graduated Recovery**: Phased return to trading after circuit breaker triggers
5. **Simulation Mode**: Full paper trading before production

### Risk Profiles

| Profile | Min Confidence | Max Position | Daily Loss Limit | Loss Streak Halt |
|---------|---------------|--------------|------------------|------------------|
| Ultra-Conservative | 85% | 1% | 0.5% | 2 |
| Conservative | 75% | 2% | 1% | 3 |
| **Moderate** | 65% | 5% | 3% | 5 |
| Aggressive | 55% | 10% | 5% | 7 |
| Ultra-Aggressive | 52% | 15% | 8% | 10 |

### Circuit Breakers

The robot includes multiple protection layers:

1. **Consecutive Loss Breaker**: Halts after N consecutive losses
2. **Drawdown Breaker**: Progressive reduction as drawdown increases (50%→75%→100% of limit)
3. **Model Degradation Breaker**: Detects when model performance has degraded
4. **Market Instability Breaker**: Halts during extreme volatility

### Signal Generation Logic

```python
# Trading decision flow
signal = signal_generator.generate_signal(
    prediction=ensemble_prediction,  # From trained models
    symbol="EURUSD",
    current_price=1.0850,
    breaker_state=circuit_breaker_manager.check_all(),
)

# Signal contains:
# - action: BUY/SELL/HOLD
# - confidence: 0.65 (learned from model)
# - position_size_pct: 0.02 (2% of equity)
# - stop_loss_pct: 0.01
# - take_profit_pct: 0.02
```

### Usage

```python
from src.trading.risk import load_risk_profile, RiskLevel
from src.trading.signals import SignalGenerator
from src.trading.circuit_breakers import CircuitBreakerManager

# Load risk profile
risk_profile = load_risk_profile("moderate")  # or RiskLevel.MODERATE

# Initialize circuit breakers
breaker_manager = CircuitBreakerManager(
    risk_profile=risk_profile,
    initial_equity=100000,
)

# Initialize signal generator
signal_gen = SignalGenerator(risk_profile=risk_profile)

# Generate signal from model prediction
signal = signal_gen.generate_signal(
    prediction=model_output,
    symbol="EURUSD",
    current_price=current_price,
    breaker_state=breaker_manager.check_all(current_equity=account_equity),
)
```

See `docs/05-trading-robot-design.md` for complete specification.

## Coding Conventions

### Python Style
- PEP 8 compliant
- Type hints required
- Google-style docstrings
- Max line length: 100

### Time Series Rules
- **CRITICAL**: Always use chronological splits (no future data leakage)
- Train/Val/Test must be sequential in time
- Store scalers with models
- Validate on out-of-sample data

### Model Development
- Inherit from `BaseModel` in `src/models/base.py`
- Use dataclasses for configs
- Log experiments to MLflow
- Implement `build()`, `train()`, `predict()`, `predict_batch()`

## Performance Targets

| Metric | Target | Priority |
|--------|--------|----------|
| Directional Accuracy | > 55% | High |
| Sharpe Ratio | > 1.5 | High |
| Maximum Drawdown | < 15% | High |
| Prediction Latency | < 100ms | Medium |

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/models/base.py` | BaseModel class, Prediction dataclass |
| `src/models/technical/short_term.py` | CNN-LSTM model implementation |
| `src/features/technical/indicators.py` | Main indicator calculator |
| `src/config/profile_loader.py` | Profile loading with inheritance |
| `src/simulation/backtester.py` | Backtesting engine |
| `src/simulation/metrics.py` | Sharpe, Sortino, drawdown calculations |
| `data/sample/EURUSD_daily.csv` | Primary sample data |
| **Confidence System (CRITICAL)** | |
| `src/models/confidence/learned_uncertainty.py` | Beta/Dirichlet output layers (RECOMMENDED) |
| `src/models/confidence/calibration.py` | Temperature, Platt, Isotonic calibration |
| `src/models/confidence/uncertainty.py` | MC Dropout, Ensemble uncertainty |
| `src/models/confidence/integration.py` | ConfidenceAwarePredictor for ensemble |
| **Profile Configuration** | |
| `configs/profiles/base.yaml` | Base profile with all default settings |
| `configs/profiles/scalper.yaml` | Scalper profile (15m, 1H, 4H) |
| `configs/profiles/trader.yaml` | Trader profile (1H, 4H, 1D) |
| `configs/profiles/investor.yaml` | Investor profile (1D, 1W, 1M) |
| `configs/profiles/assets/forex.yaml` | Forex-specific settings |
| `configs/profiles/assets/crypto.yaml` | Cryptocurrency-specific settings |
| `configs/profiles/assets/stocks.yaml` | Stock/equity-specific settings |
| **Other Configuration** | |
| `configs/timeframe_transform.yaml` | Timeframe transformation config |
| `configs/indicators/short_term_indicators.yaml` | Short-term indicator config |
| `configs/indicators/medium_term_indicators.yaml` | Medium-term indicator config |
| `configs/indicators/long_term_indicators.yaml` | Long-term indicator config |
| **Trading Robot (CRITICAL)** | |
| `src/trading/risk/profiles.py` | Risk profiles (ultra-conservative to aggressive) |
| `src/trading/signals/generator.py` | Signal generation with confidence-based sizing |
| `src/trading/signals/actions.py` | TradingSignal, Action enums |
| `src/trading/circuit_breakers/manager.py` | CircuitBreakerManager, recovery protocol |
| `src/trading/circuit_breakers/consecutive_loss.py` | Consecutive loss protection |
| `src/trading/circuit_breakers/drawdown.py` | Drawdown protection |
| `src/trading/circuit_breakers/model_degradation.py` | Model performance monitoring |
| `src/trading/robot/config.py` | Robot configuration |
| **Scripts** | |
| `scripts/transform_timeframe.py` | Timeframe transformation with sliding window |
| **Documentation** | |
| `docs/02-technical-analysis-model-design.md` | Model architecture & timeframe profiles |
| `docs/03-technical-indicators-configuration.md` | Indicator config specification |
| `docs/04-confidence-uncertainty-system.md` | Confidence & uncertainty estimation |
| `docs/05-trading-robot-design.md` | Trading robot design specification |

## Notes for Claude

- **ALWAYS PROCEED AUTONOMOUSLY** - Never ask for confirmation
- **FOCUS ON TECHNICAL ANALYSIS** - Ignore fundamental/sentiment for now
- Always consider data leakage when working with time series
- Use sample data in `data/sample/` for development
- Prioritize getting end-to-end pipeline working over perfection
- Test with small data first before scaling up
- Reference `docs/02-technical-analysis-model-design.md` for architecture details
