# Technical Analysis Model Design

## 1. Overview

This document details the architecture and design of the Technical Analysis module - the first and foundational component of the AI Assets Trader system. The module implements three time-horizon models (Short, Medium, Long-term) that analyze price patterns and technical indicators to generate predictions.

## 2. Model Architecture Philosophy

### 2.1 Multi-Timeframe Approach

The forex market exhibits different behaviors at different time scales. Our architecture captures this through specialized models with **configurable timeframe profiles** to support different user types (traders vs investors).

#### 2.1.1 Timeframe Profiles

The system supports multiple **Timeframe Profiles** that define the candle intervals for each model tier. This allows the same model architecture to serve different user needs:

| Profile | Target User | Decision Frequency | Holding Period |
|---------|-------------|-------------------|----------------|
| **Scalper** | Scalpers, fast day traders | Continuous / Every 15m | Minutes to hours |
| **Trader** | Day/Swing traders | Multiple per day | Hours to weeks |
| **Investor** | Long-term investors | Weekly/Monthly | Weeks to months |

#### 2.1.2 Timeframe Selection Rationale

The timeframes were selected based on the following principles:

**1. Factor of 4-6 Rule**
Industry best practice dictates that timeframes in a multi-timeframe system should be 4-6x apart for meaningful analysis. This prevents redundancy while ensuring each tier captures distinct market dynamics.

**2. Signal-to-Noise Optimization**
| Timeframe | Noise Level | ML Suitability | Notes |
|-----------|-------------|----------------|-------|
| 1-5 min | ~80-85% | Poor | Too noisy, microstructure effects dominate |
| 15 min | ~65-70% | Fair | Usable for scalping only |
| **1H** | ~50% | **Good** | Optimal balance for intraday ML |
| **4H** | ~40% | **Very Good** | Sweet spot for swing trading |
| **Daily** | ~30% | **Excellent** | Gold standard for ML models |
| **Weekly** | ~20% | **Excellent** | High reliability, major trends |
| **Monthly** | ~10-15% | **Excellent** | Highest reliability, regime detection |

**3. Data Availability Considerations**
- Lower timeframes require more storage and processing
- Monthly data: only ~12 candles/year (need 3+ years for meaningful training)
- Weekly data: ~52 candles/year (need 2+ years)
- Daily data: ~252 trading days/year (1 year sufficient)
- 15-minute data: ~96 candles/day (6+ months recommended due to noise)

#### 2.1.3 Scalper Profile (Fast Trading)

Optimized for very short-term, almost real-time trading. Follows the **15m → 1H → 4H** progression (4x and 4x factors).

| Model | Candle Timeframe | Minutes | Input Window | Prediction Horizon | Use Case |
|-------|------------------|---------|--------------|-------------------|----------|
| Short-Term | **15m** | 15 | 192 candles (2 days) | 15m, 30m, 1H, 2H | Scalping |
| Medium-Term | **1H** | 60 | 168 candles (7 days) | 1H, 2H, 4H, 8H | Intraday trend |
| Long-Term | **4H** | 240 | 180 candles (30 days) | 4H, 8H, 12H, 24H | Session context |

**Why these timeframes for Scalpers:**

| Timeframe | Selection Rationale |
|-----------|---------------------|
| **15m** | Fastest practical timeframe for ML models. Below 15m, noise dominates (>70%) making ML unreliable. Captures micro-trends and quick reversals. ~65% noise level. |
| **1H** | Provides cleaner signals for intraday trend context. 4x factor from 15m. Standard for day trading bias determination. ~50% noise level. |
| **4H** | Major intraday support/resistance levels. Captures session dynamics (London, NY, Tokyo). 4x factor from 1H. ~40% noise level. |

```
SCALPER PROFILE TIMEFRAME STACK:
────────────────────────────────
Short-Term:   15m (15 min)   →  Scalping signals, quick entries
Medium-Term:  1H (60 min)    →  Intraday trend direction
Long-Term:    4H (240 min)   →  Session context, major levels

Factor progression: 15m → 1H (4x) → 4H (4x)
```

**Important Considerations for Scalper Profile:**
- Higher noise level requires more conservative position sizing
- Transaction costs significantly impact profitability
- Requires low-latency execution infrastructure
- More prone to false signals - use strict stop losses
- Best during high-liquidity sessions (London/NY overlap)
- Avoid during major news events
- Requires more training data (6+ months minimum)

#### 2.1.4 Trader Profile (Default)

Optimized for active trading with faster decision cycles. Follows the **1H → 4H → 1D** progression (4x and 6x factors).

| Model | Candle Timeframe | Hours | Input Window | Prediction Horizon | Use Case |
|-------|------------------|-------|--------------|-------------------|----------|
| Short-Term | **1H** | 1 | 168 candles (7 days) | 1H, 4H, 12H, 24H | Day trading |
| Medium-Term | **4H** | 4 | 180 candles (30 days) | 4H, 12H, 24H, 48H | Swing trading |
| Long-Term | **1D (Daily)** | 24 | 90 candles (3 months) | 1D, 3D, 5D, 7D | Position trading |

**Why these timeframes for Traders:**

| Timeframe | Selection Rationale |
|-----------|---------------------|
| **1H** | Optimal for ML-based intraday prediction. Clean enough for pattern recognition while capturing intraday dynamics. Widely used by quantitative trading firms. |
| **4H** | Industry standard for swing trading. Filters out intraday noise while responsive enough for multi-day moves. 4x factor from 1H provides distinct perspective. |
| **1D** | Gold standard for position trading. Technical indicators (RSI, MACD) were originally designed for daily data. Captures major support/resistance levels. 6x factor from 4H. |

```
TRADER PROFILE TIMEFRAME STACK:
───────────────────────────────
Short-Term:   1H (60 min)    →  Intraday signals, day trading
Medium-Term:  4H (240 min)   →  Swing signals, multi-day positions
Long-Term:    1D (1440 min)  →  Position signals, trend context

Factor progression: 1H → 4H (4x) → 1D (6x)
```

#### 2.1.5 Investor Profile

Optimized for longer-term investment decisions with reduced noise. Follows the **1D → 1W → 1M** progression (7x and ~4x factors).

| Model | Candle Timeframe | Hours | Input Window | Prediction Horizon | Use Case |
|-------|------------------|-------|--------------|-------------------|----------|
| Short-Term | **1D (Daily)** | 24 | 90 candles (3 months) | 1D, 3D, 5D, 7D | Tactical adjustments |
| Medium-Term | **1W (Weekly)** | 168 | 52 candles (1 year) | 1W, 2W, 4W | Position management |
| Long-Term | **1M (Monthly)** | 720 | 36 candles (3 years) | 1M, 2M, 3M | Strategic allocation |

**Why these timeframes for Investors:**

| Timeframe | Selection Rationale |
|-----------|---------------------|
| **1D** | Provides tactical entry/exit timing for investors. Lowest noise level for "fast" signals. Sufficient data points for training (252/year). |
| **1W** | Standard for institutional position management. Captures weekly market cycles and options expiration effects. 7x factor from Daily. Weekly MACD/RSI signals are highly reliable. |
| **1M** | Strategic allocation and regime detection. Highest signal reliability but limited data (12/year). Best for identifying bull/bear market phases. Requires 3+ years historical data. |

```
INVESTOR PROFILE TIMEFRAME STACK:
─────────────────────────────────
Short-Term:   1D (1440 min)   →  Daily tactical signals
Medium-Term:  1W (10080 min)  →  Weekly position signals
Long-Term:    1M (43200 min)  →  Monthly strategic signals

Factor progression: 1D → 1W (7x) → 1M (~4x)
```

#### 2.1.6 Timeframe Profile Comparison

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                           TIMEFRAME PROFILE COMPARISON                                │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│  Model Tier      │  SCALPER Profile   │  TRADER Profile    │  INVESTOR Profile       │
│  ────────────────┼────────────────────┼────────────────────┼─────────────────────────│
│  Short-Term      │  15m (15 min)      │  1H (60 min)       │  1D (1440 min)          │
│  Medium-Term     │  1H (60 min)       │  4H (240 min)      │  1W (10080 min)         │
│  Long-Term       │  4H (240 min)      │  1D (1440 min)     │  1M (43200 min)         │
│                                                                                       │
│  ────────────────┼────────────────────┼────────────────────┼─────────────────────────│
│  Factor (S→M)    │  4x                │  4x                │  7x                     │
│  Factor (M→L)    │  4x                │  6x                │  ~4x                    │
│  ────────────────┼────────────────────┼────────────────────┼─────────────────────────│
│  Typical User    │  Scalpers          │  Day/Swing traders │  Position/Investors     │
│  Check Frequency │  Every 15 min      │  Multiple per day  │  Weekly/Monthly         │
│  Holding Period  │  Minutes to hours  │  Hours to weeks    │  Weeks to months        │
│  Signal Noise    │  High (~65%)       │  Medium (~50%)     │  Low (~20-30%)          │
│  Opportunities   │  Very frequent     │  Frequent          │  Infrequent, high qual  │
│  Min Data Needed │  6+ months         │  1 year            │  3+ years               │
│                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

#### 2.1.7 Timeframe Profile Configuration

Profiles are configured via YAML. See `configs/timeframe_profiles/` for full configuration files.

```yaml
# configs/timeframe_profiles/scalper.yaml (summary)
profile:
  name: "scalper"
  short_term:
    candle_timeframe: "15m"     # 15 minute candles
    input_window: 192           # 2 days of 15m data
    prediction_horizons: [15m, 30m, 1H, 2H]
  medium_term:
    candle_timeframe: "1H"      # 1 hour candles
    input_window: 168           # 7 days of hourly data
    prediction_horizons: [1H, 2H, 4H, 8H]
  long_term:
    candle_timeframe: "4H"      # 4 hour candles
    input_window: 180           # 30 days of 4H data
    prediction_horizons: [4H, 8H, 12H, 24H]
```

```yaml
# configs/timeframe_profiles/trader.yaml (summary)
profile:
  name: "trader"
  short_term:
    candle_timeframe: "1H"      # 1 hour candles
    input_window: 168           # 7 days of hourly data
    prediction_horizons: [1H, 4H, 12H, 24H]
  medium_term:
    candle_timeframe: "4H"      # 4 hour candles
    input_window: 180           # 30 days of 4H data
    prediction_horizons: [4H, 12H, 24H, 48H]
  long_term:
    candle_timeframe: "1D"      # Daily candles
    input_window: 90            # 3 months of daily data
    prediction_horizons: [1D, 3D, 5D, 7D]
```

```yaml
# configs/timeframe_profiles/investor.yaml (summary)
profile:
  name: "investor"
  short_term:
    candle_timeframe: "1D"      # Daily candles
    input_window: 90            # 3 months of daily data
    prediction_horizons: [1D, 3D, 5D, 7D]
  medium_term:
    candle_timeframe: "1W"      # Weekly candles
    input_window: 52            # 1 year of weekly data
    prediction_horizons: [1W, 2W, 4W]
  long_term:
    candle_timeframe: "1M"      # Monthly candles
    input_window: 36            # 3 years of monthly data
    prediction_horizons: [1M, 2M, 3M]
```

#### 2.1.8 Timeframe Reference Table

Standard timeframe definitions used throughout the system:

| Timeframe Code | Minutes | Hours | Days | Description |
|----------------|---------|-------|------|-------------|
| `1M` (minute) | 1 | 0.017 | - | 1 minute (not recommended for ML) |
| `5M` | 5 | 0.083 | - | 5 minutes (scalping only) |
| `15M` | 15 | 0.25 | - | 15 minutes |
| `30M` | 30 | 0.5 | - | 30 minutes |
| `1H` | 60 | 1 | 0.042 | 1 hour |
| `4H` | 240 | 4 | 0.167 | 4 hours |
| `1D` / `24H` | 1440 | 24 | 1 | 1 day |
| `1W` / `168H` | 10080 | 168 | 7 | 1 week |
| `1M` (month) / `720H` | 43200 | 720 | 30 | 1 month (approx) |

### 2.2 Hybrid Architecture

Based on research, hybrid architectures combining multiple neural network types consistently outperform single-architecture models:

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID MODEL ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: OHLCV + Technical Indicators + Temporal Features         │
│                          │                                       │
│  ┌───────────────────────┼───────────────────────┐              │
│  │                       ▼                       │              │
│  │  ┌─────────────────────────────────────────┐  │              │
│  │  │         CNN Feature Extractor           │  │              │
│  │  │  (Local pattern detection)              │  │              │
│  │  └─────────────────────────────────────────┘  │              │
│  │                       │                       │              │
│  │                       ▼                       │              │
│  │  ┌─────────────────────────────────────────┐  │              │
│  │  │      Bi-LSTM Temporal Encoder           │  │              │
│  │  │  (Sequential dependency learning)       │  │              │
│  │  └─────────────────────────────────────────┘  │              │
│  │                       │                       │              │
│  │                       ▼                       │              │
│  │  ┌─────────────────────────────────────────┐  │              │
│  │  │       Multi-Head Attention              │  │              │
│  │  │  (Focus on important time steps)        │  │              │
│  │  └─────────────────────────────────────────┘  │              │
│  │                       │                       │              │
│  └───────────────────────┼───────────────────────┘              │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              Prediction Head(s)                          │    │
│  │  ┌─────────────┐  ┌─────────────────────┐  ┌─────────┐  │    │
│  │  │ Regression  │  │ BETA OUTPUT         │  │Quantile │  │    │
│  │  │ (Gaussian)  │  │ (Direction + Conf)  │  │(Risk)   │  │    │
│  │  │ μ, σ²       │  │ α, β → learned      │  │10%,90%  │  │    │
│  │  └─────────────┘  └─────────────────────┘  └─────────┘  │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  KEY: Beta output learns BOTH direction AND confidence:          │
│       - Direction = α/(α+β) → UP if >0.5, DOWN if <0.5          │
│       - Confidence = f(α+β) → High concentration = certain       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 3. Input Features

### 3.1 Raw Price Data (OHLCV)
```python
raw_features = [
    'open',
    'high',
    'low',
    'close',
    'volume',
    # Derived
    'returns',           # (close - prev_close) / prev_close
    'log_returns',       # log(close / prev_close)
    'range',             # high - low
    'body',              # close - open
    'upper_shadow',      # high - max(open, close)
    'lower_shadow',      # min(open, close) - low
]
```

### 3.2 Technical Indicators

#### Trend Indicators
```python
trend_indicators = {
    # Moving Averages
    'SMA': [5, 10, 20, 50, 100, 200],     # Simple Moving Average
    'EMA': [5, 10, 20, 50, 100, 200],     # Exponential Moving Average
    'WMA': [10, 20, 50],                   # Weighted Moving Average
    'DEMA': [10, 20],                      # Double EMA
    'TEMA': [10, 20],                      # Triple EMA

    # Trend Direction
    'ADX': [14],                           # Average Directional Index
    'PLUS_DI': [14],                       # Plus Directional Indicator
    'MINUS_DI': [14],                      # Minus Directional Indicator
    'AROON_UP': [25],                      # Aroon Up
    'AROON_DOWN': [25],                    # Aroon Down
    'AROON_OSC': [25],                     # Aroon Oscillator

    # Trend Strength
    'TRIX': [14],                          # Triple Exponential Average
    'VORTEX_POS': [14],                    # Vortex Positive
    'VORTEX_NEG': [14],                    # Vortex Negative

    # Price Position
    'PSAR': None,                          # Parabolic SAR
    'SUPERTREND': [10, 3],                 # Supertrend
}
```

#### Momentum Indicators
```python
momentum_indicators = {
    'RSI': [7, 14, 21],                    # Relative Strength Index
    'STOCH_K': [14, 3],                    # Stochastic %K
    'STOCH_D': [14, 3],                    # Stochastic %D
    'MACD': [12, 26, 9],                   # MACD Line
    'MACD_SIGNAL': [12, 26, 9],            # MACD Signal
    'MACD_HIST': [12, 26, 9],              # MACD Histogram
    'CCI': [14, 20],                       # Commodity Channel Index
    'MOM': [10, 14],                       # Momentum
    'ROC': [10, 14],                       # Rate of Change
    'WILLR': [14],                         # Williams %R
    'MFI': [14],                           # Money Flow Index
    'TSI': [25, 13],                       # True Strength Index
    'UO': [7, 14, 28],                     # Ultimate Oscillator
}
```

#### Volatility Indicators
```python
volatility_indicators = {
    'ATR': [14],                           # Average True Range
    'NATR': [14],                          # Normalized ATR
    'TRANGE': None,                        # True Range
    'BB_UPPER': [20, 2],                   # Bollinger Band Upper
    'BB_MIDDLE': [20, 2],                  # Bollinger Band Middle
    'BB_LOWER': [20, 2],                   # Bollinger Band Lower
    'BB_WIDTH': [20, 2],                   # Bollinger Band Width
    'BB_PERCENT': [20, 2],                 # Bollinger %B
    'KC_UPPER': [20, 2],                   # Keltner Channel Upper
    'KC_LOWER': [20, 2],                   # Keltner Channel Lower
    'DC_UPPER': [20],                      # Donchian Channel Upper
    'DC_LOWER': [20],                      # Donchian Channel Lower
    'STDDEV': [20],                        # Standard Deviation
}
```

#### Volume Indicators
```python
volume_indicators = {
    'OBV': None,                           # On-Balance Volume
    'AD': None,                            # Accumulation/Distribution
    'ADOSC': [3, 10],                      # A/D Oscillator
    'CMF': [20],                           # Chaikin Money Flow
    'VWAP': None,                          # Volume Weighted Avg Price
    'VPT': None,                           # Volume Price Trend
    'EMV': [14],                           # Ease of Movement
    'FI': [13],                            # Force Index
    'NVI': None,                           # Negative Volume Index
    'PVI': None,                           # Positive Volume Index
}
```

### 3.3 Temporal Features
```python
temporal_features = [
    'hour_sin',          # sin(2 * pi * hour / 24)
    'hour_cos',          # cos(2 * pi * hour / 24)
    'day_of_week_sin',   # sin(2 * pi * day / 7)
    'day_of_week_cos',   # cos(2 * pi * day / 7)
    'day_of_month_sin',  # sin(2 * pi * day / 31)
    'day_of_month_cos',  # cos(2 * pi * day / 31)
    'month_sin',         # sin(2 * pi * month / 12)
    'month_cos',         # cos(2 * pi * month / 12)
    'is_london_open',    # Binary: London session active
    'is_ny_open',        # Binary: New York session active
    'is_tokyo_open',     # Binary: Tokyo session active
    'is_overlap',        # Binary: Session overlap periods
]
```

### 3.4 Derived/Engineered Features
```python
engineered_features = [
    # Price relative to MAs
    'price_to_sma_20',
    'price_to_sma_50',
    'price_to_sma_200',

    # MA crossover signals
    'sma_5_20_cross',
    'sma_20_50_cross',
    'ema_12_26_cross',

    # Indicator divergences
    'rsi_price_divergence',
    'macd_price_divergence',

    # Volatility regime
    'volatility_regime',     # Low/Normal/High
    'atr_percentile',

    # Support/Resistance proximity
    'dist_to_support',
    'dist_to_resistance',

    # Pattern recognition scores
    'bullish_pattern_score',
    'bearish_pattern_score',
]
```

## 4. Model Specifications

The model specifications below use parameters from the active **Timeframe Profile** (see Section 2.1). The architecture remains the same regardless of profile; only the input timeframe and prediction horizons change.

### 4.1 Short-Term Model

**Purpose**: Fastest predictions within the selected profile

| Profile | Candle TF | Input Window | Predictions |
|---------|-----------|--------------|-------------|
| Trader | 1H | 168 (7 days) | 1H, 4H, 12H, 24H |
| Investor | 24H (Daily) | 90 (3 months) | 1D, 3D, 5D, 7D |

**Architecture**: CNN-LSTM with Attention
```python
class ShortTermModel(nn.Module):
    """
    Input Shape: (batch, sequence_length, features)
    - sequence_length from active TimeframeProfile.short_term.input_window

    Output:
    - price_prediction: Next N candle prices
    - direction_prob: Probability of up/down
    - confidence: Model uncertainty estimate
    """

    # Architecture config (profile-independent)
    architecture_config = {
        # CNN Configuration
        'cnn_filters': [64, 128, 256],
        'cnn_kernel_sizes': [3, 5, 7],
        'cnn_dropout': 0.2,

        # LSTM Configuration
        'lstm_hidden_size': 256,
        'lstm_num_layers': 2,
        'lstm_dropout': 0.3,
        'lstm_bidirectional': True,

        # Attention Configuration
        'attention_heads': 8,
        'attention_dim': 256,

        # Output Configuration (RECOMMENDED: Beta for direction)
        'output_type': 'beta',  # 'beta' (recommended), 'sigmoid', 'gaussian'
        'output_features': ['direction', 'price', 'volatility'],

        # Beta Output Layer (for direction prediction with learned confidence)
        # See src/models/confidence/learned_uncertainty.py
        'beta_config': {
            'min_concentration': 1.0,  # Minimum α and β value
            # Output: BetaPrediction(α, β)
            # - Direction: mean = α/(α+β), >0.5 = UP, <0.5 = DOWN
            # - Confidence: derived from concentration (α+β)
        },
    }

    # Timeframe config (from active profile)
    # These values are loaded from TimeframeProfile at runtime
    timeframe_config = {
        # Trader Profile defaults:
        'sequence_length': 168,                    # From profile.short_term.input_window
        'candle_timeframe': '1H',                  # From profile.short_term.candle_timeframe
        'prediction_horizons': [1, 4, 12, 24],     # From profile.short_term.prediction_horizons
    }
```

### 4.2 Medium-Term Model

**Purpose**: Intermediate predictions for swing/position decisions

| Profile | Candle TF | Input Window | Predictions |
|---------|-----------|--------------|-------------|
| Trader | 4H | 180 (30 days) | 4H, 12H, 24H, 48H |
| Investor | 168H (Weekly) | 52 (1 year) | 1W, 2W, 4W |

**Architecture**: Temporal Fusion Transformer (TFT)
```python
class MediumTermModel(nn.Module):
    """
    Input Shape: (batch, sequence_length, features)
    - sequence_length from active TimeframeProfile.medium_term.input_window

    Output:
    - price_prediction: Next N period prices
    - direction_prob: Probability distribution
    - attention_weights: Interpretability
    """

    # Architecture config (profile-independent)
    architecture_config = {
        # TFT Configuration
        'hidden_size': 256,
        'attention_heads': 4,
        'dropout': 0.1,
        'hidden_continuous_size': 64,
        'num_lstm_layers': 2,

        # Variable Selection
        'static_categoricals': ['currency_pair'],
        'time_varying_known': temporal_features,
        'time_varying_unknown': price_and_indicator_features,

        # Output Configuration (RECOMMENDED: Beta + Quantile)
        'output_type': 'beta',  # 'beta' (recommended) for direction
        'beta_config': {
            'min_concentration': 1.0,
        },

        # Quantile outputs for risk bounds
        'quantiles': [0.1, 0.25, 0.5, 0.75, 0.9],
    }

    # Timeframe config (from active profile)
    timeframe_config = {
        # Trader Profile defaults:
        'sequence_length': 180,                    # From profile.medium_term.input_window
        'candle_timeframe': '4H',                  # From profile.medium_term.candle_timeframe
        'prediction_horizons': [1, 3, 6, 12],      # From profile.medium_term.prediction_horizons
    }
```

### 4.3 Long-Term Model

**Purpose**: Longest-horizon predictions and regime detection

| Profile | Candle TF | Input Window | Predictions |
|---------|-----------|--------------|-------------|
| Trader | 24H (Daily) | 90 (3 months) | 1D, 3D, 5D, 7D |
| Investor | 720H (Monthly) | 36 (3 years) | 1M, 2M, 3M |

**Architecture**: N-BEATS + Transformer Hybrid
```python
class LongTermModel(nn.Module):
    """
    Input Shape: (batch, sequence_length, features)
    - sequence_length from active TimeframeProfile.long_term.input_window

    Output:
    - trend_prediction: Trend direction & strength
    - price_targets: Key price levels
    - regime_classification: Market regime
    """

    # Architecture config (profile-independent)
    architecture_config = {
        # N-BEATS Configuration
        'nbeats_stacks': 30,
        'nbeats_blocks': 3,
        'nbeats_layers': 4,
        'nbeats_width': 256,

        # Transformer Configuration
        'transformer_heads': 8,
        'transformer_layers': 4,
        'transformer_dim': 512,

        # Output Configuration (RECOMMENDED: Beta + Dirichlet)
        'output_type': 'beta',  # For trend direction
        'beta_config': {
            'min_concentration': 1.0,
        },

        # Regime classification uses Dirichlet output
        # See src/models/confidence/learned_uncertainty.py
        'regime_output': 'dirichlet',  # For multi-class with uncertainty
        'regime_classes': ['trending_up', 'trending_down', 'ranging', 'volatile'],
    }

    # Timeframe config (from active profile)
    timeframe_config = {
        # Trader Profile defaults:
        'sequence_length': 90,                     # From profile.long_term.input_window
        'candle_timeframe': '1D',                  # From profile.long_term.candle_timeframe
        'prediction_horizons': [1, 3, 5, 7],       # From profile.long_term.prediction_horizons
    }
```

### 4.4 Loading Models with Timeframe Profile

```python
from src.config import TimeframeProfile
from src.models.technical import ShortTermModel, MediumTermModel, LongTermModel

# Load a timeframe profile
trader_profile = TimeframeProfile.load('configs/timeframe_profiles/trader.yaml')
investor_profile = TimeframeProfile.load('configs/timeframe_profiles/investor.yaml')

# Initialize models with profile
short_model = ShortTermModel(timeframe_profile=trader_profile)
medium_model = MediumTermModel(timeframe_profile=trader_profile)
long_model = LongTermModel(timeframe_profile=trader_profile)

# Or for investors:
short_model_inv = ShortTermModel(timeframe_profile=investor_profile)
medium_model_inv = MediumTermModel(timeframe_profile=investor_profile)
long_model_inv = LongTermModel(timeframe_profile=investor_profile)
```

## 5. Ensemble Architecture

### 5.1 Technical Analysis Ensemble

```
┌─────────────────────────────────────────────────────────────────┐
│              TECHNICAL ANALYSIS ENSEMBLE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │ Short-Term  │   │ Medium-Term │   │ Long-Term   │           │
│  │   Model     │   │   Model     │   │   Model     │           │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               Prediction Standardization                 │   │
│  │  - Normalize to common scale                            │   │
│  │  - Align prediction horizons                            │   │
│  │  - Extract confidence scores                            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               Ensemble Combiner                          │   │
│  │                                                          │   │
│  │  Method 1: Weighted Average                              │   │
│  │  ────────────────────────                               │   │
│  │  final = w_s * short + w_m * medium + w_l * long        │   │
│  │  weights learned from validation performance            │   │
│  │                                                          │   │
│  │  Method 2: Stacking (Meta-Learner)                      │   │
│  │  ────────────────────────────────                       │   │
│  │  meta_features = [short_pred, medium_pred, long_pred,   │   │
│  │                   short_conf, medium_conf, long_conf,   │   │
│  │                   market_regime, volatility]            │   │
│  │  final = MetaModel(meta_features)                       │   │
│  │                                                          │   │
│  │  Method 3: Attention-based Fusion                       │   │
│  │  ──────────────────────────────                         │   │
│  │  attention_weights = softmax(Q @ K.T / sqrt(d))         │   │
│  │  final = attention_weights @ [short, medium, long]      │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               Final Technical Prediction                 │   │
│  │  - Direction (up/down/neutral)                          │   │
│  │  - Magnitude (price target)                             │   │
│  │  - Confidence (0-1 score)                               │   │
│  │  - Time horizon recommendation                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Dynamic Weight Calculation

```python
class DynamicWeightCalculator:
    """
    Adjusts ensemble weights based on:
    - Recent model performance (rolling accuracy)
    - Market regime (trending vs ranging)
    - Volatility conditions
    - Model agreement/disagreement
    """

    def calculate_weights(self, market_state, recent_performance):
        # Base weights from historical performance
        base_weights = self._get_performance_weights(recent_performance)

        # Regime adjustment
        regime_adj = self._get_regime_adjustment(market_state.regime)

        # Volatility adjustment
        vol_adj = self._get_volatility_adjustment(market_state.volatility)

        # Agreement bonus (if models agree, boost confidence)
        agreement_adj = self._get_agreement_adjustment(predictions)

        # Combine and normalize
        final_weights = base_weights * regime_adj * vol_adj
        return final_weights / final_weights.sum()

    def _get_regime_adjustment(self, regime):
        """
        In trending markets: favor medium/long-term
        In ranging markets: favor short-term
        """
        adjustments = {
            'trending_up': {'short': 0.8, 'medium': 1.2, 'long': 1.3},
            'trending_down': {'short': 0.8, 'medium': 1.2, 'long': 1.3},
            'ranging': {'short': 1.3, 'medium': 1.0, 'long': 0.7},
            'volatile': {'short': 1.0, 'medium': 0.9, 'long': 0.6},
        }
        return adjustments.get(regime, {'short': 1, 'medium': 1, 'long': 1})
```

## 6. Training Pipeline

### 6.1 Data Preparation

```python
class DataPipeline:
    """
    Handles data preparation for model training
    """

    def prepare_data(self, symbol, timeframe, start_date, end_date):
        # 1. Fetch raw OHLCV data
        raw_data = self.fetch_ohlcv(symbol, timeframe, start_date, end_date)

        # 2. Calculate technical indicators
        features = self.calculate_indicators(raw_data)

        # 3. Add temporal features
        features = self.add_temporal_features(features)

        # 4. Handle missing values
        features = self.handle_missing(features)

        # 5. Normalize features
        features, scalers = self.normalize(features)

        # 6. Create sequences
        X, y = self.create_sequences(features, self.config.sequence_length)

        # 7. Split data (time-series aware)
        train, val, test = self.time_series_split(X, y)

        return train, val, test, scalers

    def time_series_split(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """
        Chronological split to prevent data leakage
        """
        n = len(X)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        return (
            (X[:train_end], y[:train_end]),
            (X[train_end:val_end], y[train_end:val_end]),
            (X[val_end:], y[val_end:])
        )
```

### 6.2 Training Configuration

```python
training_config = {
    # Optimizer
    'optimizer': 'AdamW',
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'scheduler': 'OneCycleLR',

    # Training
    'batch_size': 64,
    'epochs': 100,
    'early_stopping_patience': 15,
    'gradient_clip': 1.0,

    # Loss functions
    'losses': {
        'price': 'HuberLoss',          # Robust to outliers
        'direction': 'CrossEntropyLoss',
        'volatility': 'GaussianNLLLoss',  # For uncertainty
    },
    'loss_weights': {
        'price': 1.0,
        'direction': 0.5,
        'volatility': 0.3,
    },

    # Regularization
    'dropout': 0.3,
    'label_smoothing': 0.1,

    # Augmentation
    'augmentation': {
        'noise_injection': 0.01,
        'time_masking': 0.1,
    },
}
```

### 6.3 Walk-Forward Validation

```python
class WalkForwardValidator:
    """
    Implements expanding window walk-forward validation
    to simulate real trading conditions
    """

    def validate(self, model, data, n_splits=5):
        results = []

        for i, (train_idx, test_idx) in enumerate(self.get_splits(data, n_splits)):
            # Train on expanding window
            model.fit(data[train_idx])

            # Test on next period
            predictions = model.predict(data[test_idx])
            metrics = self.calculate_metrics(predictions, actuals)

            results.append({
                'split': i,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'metrics': metrics
            })

        return self.aggregate_results(results)
```

## 7. Model Outputs & Predictions

### 7.1 Prediction Schema

```python
@dataclass
class TechnicalPrediction:
    """
    Output structure for technical analysis predictions.

    IMPORTANT: Uses Beta distribution for direction prediction with learned confidence.
    See src/models/confidence/learned_uncertainty.py for implementation.
    """
    timestamp: datetime
    symbol: str

    # Price predictions (Gaussian output with learned variance)
    price_predictions: Dict[str, float]  # {'1h': 1.0825, '4h': 1.0830, ...}
    price_uncertainty: Dict[str, float]  # Learned standard deviation per horizon

    # Direction predictions (Beta distribution output - RECOMMENDED)
    direction: str                        # 'bullish', 'bearish', 'neutral'
    direction_probability: float          # mean = α/(α+β), 0.0 - 1.0

    # Beta distribution parameters (learned from model)
    alpha: float                          # Evidence for UP direction
    beta: float                           # Evidence for DOWN direction
    concentration: float                  # α + β, higher = more certain

    # Confidence & uncertainty (LEARNED, not derived)
    confidence_score: float               # Computed from concentration
    prediction_interval: Tuple[float, float]  # Quantile regression bounds (10%, 90%)

    # Model breakdown
    short_term_signal: float              # -1.0 to 1.0
    medium_term_signal: float             # -1.0 to 1.0
    long_term_signal: float               # -1.0 to 1.0

    # Additional context
    market_regime: str                    # Current detected regime
    key_levels: Dict[str, float]          # Support/resistance levels
    recommended_action: str               # 'buy', 'sell', 'hold', 'wait'

    # Metadata
    model_version: str
    prediction_generated_at: datetime
```

### 7.2 Signal Generation

```python
class SignalGenerator:
    """
    Converts model predictions into actionable trading signals
    """

    def generate_signal(self, prediction: TechnicalPrediction) -> TradingSignal:
        # Combine timeframe signals
        combined_signal = self._combine_signals(
            prediction.short_term_signal,
            prediction.medium_term_signal,
            prediction.long_term_signal,
            weights=self.current_weights
        )

        # Apply confidence threshold
        if prediction.confidence_score < self.min_confidence:
            return TradingSignal(action='HOLD', reason='Low confidence')

        # Generate signal
        if combined_signal > self.buy_threshold:
            return TradingSignal(
                action='BUY',
                strength=combined_signal,
                confidence=prediction.confidence_score,
                stop_loss=prediction.key_levels['support'],
                take_profit=prediction.key_levels['resistance'],
            )
        elif combined_signal < self.sell_threshold:
            return TradingSignal(
                action='SELL',
                strength=abs(combined_signal),
                confidence=prediction.confidence_score,
                stop_loss=prediction.key_levels['resistance'],
                take_profit=prediction.key_levels['support'],
            )
        else:
            return TradingSignal(action='HOLD', reason='Signal within neutral zone')
```

## 8. Performance Metrics

### 8.1 Prediction Accuracy Metrics

```python
prediction_metrics = {
    # Regression metrics (price prediction)
    'MAE': 'Mean Absolute Error',
    'RMSE': 'Root Mean Square Error',
    'MAPE': 'Mean Absolute Percentage Error',

    # Direction metrics
    'Directional_Accuracy': 'Percentage of correct up/down predictions',
    'Precision': 'True positives / (True positives + False positives)',
    'Recall': 'True positives / (True positives + False negatives)',
    'F1_Score': 'Harmonic mean of precision and recall',

    # Probabilistic metrics
    'Brier_Score': 'Mean squared error of probability forecasts',
    'Log_Loss': 'Logarithmic loss for probability predictions',
    'Calibration_Error': 'Expected vs actual probability alignment',
}
```

### 8.2 Trading Performance Metrics

```python
trading_metrics = {
    # Returns
    'Total_Return': 'Cumulative return over period',
    'Annualized_Return': 'Return normalized to yearly basis',
    'Alpha': 'Excess return vs benchmark',

    # Risk-adjusted
    'Sharpe_Ratio': 'Return per unit of risk',
    'Sortino_Ratio': 'Return per unit of downside risk',
    'Calmar_Ratio': 'Return / Maximum drawdown',

    # Risk
    'Max_Drawdown': 'Largest peak-to-trough decline',
    'VaR_95': 'Value at Risk at 95% confidence',
    'CVaR_95': 'Conditional VaR (Expected Shortfall)',

    # Trading quality
    'Win_Rate': 'Percentage of winning trades',
    'Profit_Factor': 'Gross profit / Gross loss',
    'Average_Win_Loss_Ratio': 'Average win / Average loss',
}
```

## 9. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up project structure
- [ ] Implement data pipeline for forex data
- [ ] Create feature engineering module (technical indicators)
- [ ] Build data normalization and preprocessing

### Phase 2: Short-Term Model (Weeks 3-4)
- [ ] Implement CNN-LSTM architecture
- [ ] Add attention mechanism
- [ ] Create training pipeline
- [ ] Implement walk-forward validation

### Phase 3: Medium-Term Model (Weeks 5-6)
- [ ] Implement Temporal Fusion Transformer
- [ ] Integrate with PyTorch Forecasting
- [ ] Train and validate model

### Phase 4: Long-Term Model (Weeks 7-8)
- [ ] Implement N-BEATS + Transformer hybrid
- [ ] Train and validate model
- [ ] Tune hyperparameters

### Phase 5: Ensemble (Weeks 9-10)
- [ ] Implement ensemble combiner
- [ ] Create dynamic weight calculator
- [ ] Integrate all three models
- [ ] Comprehensive backtesting

### Phase 6: Simulation Mode (Weeks 11-12)
- [ ] Build backtesting engine
- [ ] Implement paper trading mode
- [ ] Create performance reporting
- [ ] Dashboard for visualization

## 10. Key Design Decisions

### 10.1 Why Hybrid Architecture?
- **CNNs** excel at detecting local patterns (candlestick patterns, short-term trends)
- **LSTMs** capture long-range dependencies in time series
- **Attention** helps focus on the most relevant historical periods
- **Transformers** (TFT) provide interpretability and handle multiple input types

### 10.2 Why Multiple Timeframes?
- Different trading strategies require different horizons
- Multi-scale analysis captures market dynamics at various levels
- Combining timeframes provides more robust signals
- Allows the system to adapt to different user risk profiles

### 10.3 Why Configurable Timeframe Profiles?
- **User Flexibility**: Traders and investors have fundamentally different needs
- **Same Architecture**: Model architectures work across timeframes; only input data changes
- **Reduced Noise**: Longer timeframes (investor profile) have higher signal-to-noise ratios
- **Appropriate Frequency**: Traders need hourly signals; investors need daily/weekly signals
- **Resource Efficiency**: Train separate model weights per profile, share architecture code
- **A/B Testing**: Easy to experiment with different timeframe combinations

### 10.4 Why Probabilistic Outputs?
- Markets are inherently uncertain
- Point predictions alone are insufficient
- Probability distributions enable proper risk management
- Confidence scores allow position sizing optimization

### 10.5 Why Beta Distribution for Direction Prediction? (CRITICAL)

The system uses **Beta distribution outputs** instead of sigmoid for direction prediction. This is a fundamental architectural choice.

**The Problem with Sigmoid:**
```
Model → Sigmoid → 0.75 → "75% probability UP"

But does this mean:
A) "I'm 75% confident it will go UP"
B) "I'm very confident it will go up by a small amount"
C) "I have no idea, but 0.75 is my best guess"

THE MODEL DOESN'T KNOW! We ASSUME confidence from distance to 0.5.
```

**The Beta Solution:**
```
Model → Beta(α, β) → Distribution over [0, 1]

Examples:
- Beta(α=15, β=3) → "UP with HIGH confidence" (concentration=18)
- Beta(α=3, β=1.5) → "UP with LOW confidence" (concentration=4.5)
- Beta(α=1.1, β=1.1) → "I DON'T KNOW" (concentration=2.2)

The model EXPLICITLY outputs:
- DIRECTION: mean = α/(α+β)
- CONFIDENCE: concentration = α+β
```

**Key Benefits:**
1. **Learned Confidence**: Model learns WHEN to be confident through the loss function
2. **Epistemic Uncertainty**: Low concentration means "I don't have enough evidence"
3. **No Trade Option**: Model can signal "I don't know" with α ≈ β ≈ 1
4. **Position Sizing**: Use concentration directly for position sizing
5. **Single Forward Pass**: No MC Dropout needed (efficient for real-time)

**Position Sizing Based on Confidence:**

| Confidence | Concentration (α+β) | Position Size | Action |
|------------|---------------------|---------------|--------|
| ≥ 0.85     | ≥ 15                | 100%          | Full position |
| 0.70 - 0.85| 8 - 15              | 50%           | Medium position |
| 0.55 - 0.70| 4 - 8               | 25%           | Small position |
| < 0.55     | < 4                 | 0%            | NO TRADE |

**Implementation:** See `src/models/confidence/learned_uncertainty.py`

## 11. References

1. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting (Lim et al., 2021)
2. N-BEATS: Neural basis expansion analysis for interpretable time series forecasting (Oreshkin et al., 2020)
3. Attention Is All You Need (Vaswani et al., 2017)
4. Deep Learning for Financial Applications (Survey, 2023)
5. FinRL: Deep Reinforcement Learning Framework for Trading
6. Darts: Time Series Made Easy in Python

---

*Document Version: 1.2*
*Last Updated: 2026-01-08*
*Author: AI Trader Development Team*

### Changelog
- **v1.2** (2026-01-08): Added Beta distribution output layer for direction prediction with learned confidence. Updated architecture diagrams and model specifications to use Beta outputs. Added section 10.5 explaining why Beta > Sigmoid for trading.
- **v1.1** (2026-01-08): Added configurable Timeframe Profiles (Trader/Investor), updated model specifications to support profile-based configuration
- **v1.0** (2025-01-06): Initial document
