# Sentiment Analysis Integration Design

## Document Information

| Field | Value |
|-------|-------|
| Version | 1.0 |
| Created | 2026-01-11 |
| Status | Design Complete - Ready for Implementation |
| Related Docs | `02-technical-analysis-model-design.md`, `03-technical-indicators-configuration.md` |

---

## 1. Executive Summary

This document describes the technical design for integrating Economic Policy Uncertainty (EPU) sentiment data into the existing technical analysis models. The sentiment scores will be treated as additional input features alongside technical indicators, allowing the neural network to **learn optimal weightings** during training rather than applying fixed manual adjustments.

### Key Design Decisions

1. **Feature Concatenation Approach**: Sentiment features are added as additional input features to the existing model architecture
2. **Learned Weighting**: The model learns how much weight to give sentiment vs technical indicators during training
3. **Dual-Stream Architecture** (Phase 2): Optional enhancement with separate encoding paths for technical and sentiment data
4. **Minimal Architectural Changes**: Existing CNN-LSTM-Attention architecture remains largely unchanged

---

## 2. Available Datasets

### 2.1 Price Data (5-minute intervals, 2020-2025)

| Asset Type | Symbol | File Path | Rows |
|------------|--------|-----------|------|
| Forex | EUR/USD | `data/forex/EURUSD_20200101_20251231_5min_combined.csv` | ~448K |
| Forex | GBP/USD | `data/forex/GBPUSD_20200101_20251231_5min_combined.csv` | ~448K |
| Forex | USD/JPY | `data/forex/USDJPY_20200101_20251231_5min_combined.csv` | ~448K |
| Forex | AUD/USD | `data/forex/AUDUSD_20200101_20251231_5min_combined.csv` | ~448K |
| Forex | EUR/GBP | `data/forex/EURGBP_20200101_20251231_5min_combined.csv` | ~448K |
| Crypto | BTC/USDT | `data/crypto/BTCUSDT_20200101_20251231_5m.csv` | ~631K |
| Crypto | ETH/USDT | `data/crypto/ETHUSDT_20200101_20251231_5m.csv` | ~631K |
| Crypto | BNB/USDT | `data/crypto/BNBUSDT_20200101_20251231_5m.csv` | ~631K |
| Crypto | SOL/USDT | `data/crypto/SOLUSDT_20200101_20251231_5m.csv` | ~567K |
| Crypto | XRP/USDT | `data/crypto/XRPUSDT_20200101_20251231_5m.csv` | ~631K |

### 2.2 Sentiment Data (Daily, 2020-2025)

| File | Description | Columns |
|------|-------------|---------|
| `data/sentiment/sentiment_epu_20200101_20251231_daily.csv` | Full EPU data + derived sentiment | 21 columns |
| `data/sentiment/sentiment_scores_20200101_20251231_daily.csv` | Sentiment scores only | 13 columns |

**Sentiment Columns Available:**

```
Raw EPU Values (higher = more uncertainty):
- EPU_US, EPU_UK, EPU_Europe, EPU_Germany, EPU_Japan, EPU_Australia, EPU_China, EPU_Global

Country Sentiment Scores (range: -0.2 to +0.2):
- Sentiment_US, Sentiment_UK, Sentiment_Europe, Sentiment_Germany
- Sentiment_Japan, Sentiment_Australia, Sentiment_Global

Currency Pair Sentiment (range: -0.2 to +0.2):
- Sentiment_EURUSD, Sentiment_GBPUSD, Sentiment_USDJPY
- Sentiment_AUDUSD, Sentiment_EURGBP, Sentiment_Crypto
```

---

## 3. Sentiment Feature Engineering

### 3.1 Core Sentiment Features

For each trading pair, create the following features from the daily sentiment data:

```python
SENTIMENT_FEATURES = {
    # Raw sentiment score for the pair
    "sentiment_raw": "Direct daily sentiment score (-0.2 to +0.2)",

    # Moving averages (smoothed trends)
    "sentiment_ma_3": "3-day moving average of sentiment",
    "sentiment_ma_7": "7-day moving average of sentiment",
    "sentiment_ma_14": "14-day moving average of sentiment",
    "sentiment_ma_30": "30-day moving average of sentiment",

    # Momentum features
    "sentiment_momentum_3": "sentiment - sentiment_ma_3 (short-term change)",
    "sentiment_momentum_7": "sentiment - sentiment_ma_7 (medium-term change)",
    "sentiment_roc_7": "Rate of change over 7 days",

    # Volatility features
    "sentiment_std_7": "7-day rolling standard deviation",
    "sentiment_std_14": "14-day rolling standard deviation",

    # Regime features
    "sentiment_zscore": "(sentiment - sentiment_ma_30) / sentiment_std_30",
    "sentiment_regime": "Categorical: -1 (bearish), 0 (neutral), 1 (bullish)",

    # Cross-country features (for forex)
    "sentiment_differential": "Base currency sentiment - Quote currency sentiment",
}
```

### 3.2 Feature Count by Model Type

| Model | Technical Features | Sentiment Features | Total Features |
|-------|-------------------|-------------------|----------------|
| Short-Term | ~50 | ~12 | ~62 |
| Medium-Term | ~60 | ~12 | ~72 |
| Long-Term | ~45 | ~12 | ~57 |

### 3.3 Timeframe Alignment Strategy

Since sentiment data is daily but price data is 5-minute:

1. **Forward Fill**: Each day's sentiment applies to all intraday candles until the next day
2. **No Look-Ahead**: Use previous day's sentiment for current day's predictions
3. **Lag Features**: Include sentiment from D-1, D-2, D-3 for temporal context

```python
# Alignment pseudocode
def align_sentiment_to_price(price_df, sentiment_df):
    """
    Align daily sentiment to intraday price data.

    CRITICAL: Use previous day's sentiment to avoid look-ahead bias.
    """
    # Shift sentiment by 1 day to avoid look-ahead
    sentiment_shifted = sentiment_df.shift(1)

    # Create date column from price index
    price_df['date'] = price_df.index.date

    # Merge on date
    merged = price_df.merge(
        sentiment_shifted,
        left_on='date',
        right_index=True,
        how='left'
    )

    # Forward fill any gaps (weekends, holidays)
    merged = merged.ffill()

    return merged
```

---

## 4. Architecture Design

### 4.1 Phase 1: Feature Concatenation (Recommended Starting Point)

The simplest integration - add sentiment features as additional columns to the input tensor.

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1: FEATURE CONCATENATION           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Tensor Shape: (batch, seq_len, features)             │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Features (concatenated):                            │   │
│  │  - OHLCV: 5 features                                │   │
│  │  - Technical Indicators: ~50 features               │   │
│  │  - Temporal Features: ~10 features                  │   │
│  │  - Sentiment Features: ~12 features   <-- NEW       │   │
│  │  ─────────────────────────────────────              │   │
│  │  Total: ~77 features                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │            Existing CNN-LSTM-Attention               │   │
│  │            (No architectural changes)                │   │
│  └─────────────────────────────────────────────────────┘   │
│                          │                                  │
│                          ▼                                  │
│                    Predictions                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Benefits:
- Minimal code changes
- Model learns optimal weighting automatically
- Easy to A/B test (with/without sentiment)

Changes Required:
- Modify FeatureProcessor to include sentiment
- Update config to list sentiment features
- No model architecture changes
```

### 4.2 Phase 2: Dual-Stream Architecture (Future Enhancement)

Separate encoding paths that are fused before the output layer.

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 2: DUAL-STREAM FUSION              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────────┐      ┌────────────────────┐        │
│  │ Technical Features │      │ Sentiment Features │        │
│  │  (seq_len, ~65)    │      │  (seq_len, ~12)    │        │
│  └─────────┬──────────┘      └─────────┬──────────┘        │
│            │                           │                    │
│            ▼                           ▼                    │
│  ┌─────────────────────┐    ┌─────────────────────┐        │
│  │   Technical Stream   │    │   Sentiment Stream  │        │
│  │   CNN-LSTM-Attention │    │   MLP or small LSTM │        │
│  │   Output: 256 dim    │    │   Output: 32 dim    │        │
│  └─────────┬───────────┘    └─────────┬───────────┘        │
│            │                           │                    │
│            └─────────────┬─────────────┘                    │
│                          │                                  │
│                          ▼                                  │
│            ┌─────────────────────────────┐                  │
│            │      Fusion Layer           │                  │
│            │  - Concatenate: 288 dim     │                  │
│            │  - Dense(288, 128, ReLU)    │                  │
│            │  - Dense(128, 64, ReLU)     │                  │
│            └─────────────┬───────────────┘                  │
│                          │                                  │
│                          ▼                                  │
│               ┌─────────────────────┐                       │
│               │    Output Heads     │                       │
│               │  Price, Direction,  │                       │
│               │  Confidence         │                       │
│               └─────────────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Benefits:
- Specialized processing for each data type
- Better representation learning
- Can pre-train streams separately

Changes Required:
- New SentimentEncoder module
- New FusionLayer module
- Modify model forward() method
```

---

## 5. Implementation Plan

### 5.1 New Files to Create

```
src/
├── features/
│   └── sentiment/                          # NEW DIRECTORY
│       ├── __init__.py
│       ├── sentiment_features.py           # Sentiment feature engineering
│       └── sentiment_loader.py             # Load and align sentiment data
├── data/
│   └── processors/
│       └── sentiment_processor.py          # NEW: Process sentiment for training
└── models/
    └── technical/
        └── sentiment_enhanced.py           # NEW: Models with sentiment support
```

### 5.2 Files to Modify

```
src/
├── features/
│   └── technical/
│       └── indicators.py                   # Add sentiment to calculate_all()
├── data/
│   └── processors/
│       └── features.py                     # Add sentiment feature selection
└── models/
    └── technical/
        ├── short_term.py                   # Support sentiment input
        ├── medium_term.py                  # Support sentiment input
        └── long_term.py                    # Support sentiment input
```

### 5.3 Configuration Updates

Create `configs/sentiment/sentiment_features.yaml`:

```yaml
sentiment:
  enabled: true

  # Data source
  data_path: "data/sentiment/sentiment_epu_20200101_20251231_daily.csv"

  # Feature selection per asset type
  forex_features:
    - "sentiment_raw"          # Direct pair sentiment
    - "sentiment_ma_7"
    - "sentiment_ma_14"
    - "sentiment_momentum_7"
    - "sentiment_std_7"
    - "sentiment_zscore"
    - "sentiment_differential" # Base - Quote sentiment

  crypto_features:
    - "sentiment_raw"          # Global/US sentiment
    - "sentiment_ma_7"
    - "sentiment_ma_14"
    - "sentiment_momentum_7"
    - "sentiment_std_7"
    - "sentiment_zscore"

  # Lag features (previous days' sentiment)
  include_lags: [1, 2, 3]      # D-1, D-2, D-3 sentiment

  # Normalization
  normalize: true
  normalization_method: "zscore"  # or "minmax"
```

---

## 6. Data Pipeline Implementation

### 6.1 SentimentLoader Class

```python
# src/features/sentiment/sentiment_loader.py

class SentimentLoader:
    """Load and prepare sentiment data for model training."""

    def __init__(self, sentiment_path: str):
        self.sentiment_path = sentiment_path
        self.sentiment_data = None

    def load(self) -> pd.DataFrame:
        """Load sentiment data from CSV."""
        self.sentiment_data = pd.read_csv(
            self.sentiment_path,
            index_col='Date',
            parse_dates=True
        )
        return self.sentiment_data

    def get_pair_sentiment(self, pair: str) -> pd.Series:
        """
        Get sentiment column for a specific trading pair.

        Args:
            pair: Trading pair (e.g., 'EURUSD', 'BTCUSDT')

        Returns:
            Series with daily sentiment scores
        """
        pair_upper = pair.upper()

        # Forex pairs
        forex_mapping = {
            'EURUSD': 'Sentiment_EURUSD',
            'GBPUSD': 'Sentiment_GBPUSD',
            'USDJPY': 'Sentiment_USDJPY',
            'AUDUSD': 'Sentiment_AUDUSD',
            'EURGBP': 'Sentiment_EURGBP',
        }

        # Crypto pairs use global sentiment
        if 'USDT' in pair_upper or 'BTC' in pair_upper:
            return self.sentiment_data['Sentiment_Crypto']

        return self.sentiment_data.get(
            forex_mapping.get(pair_upper, 'Sentiment_Global')
        )

    def align_to_price_data(
        self,
        price_df: pd.DataFrame,
        pair: str,
        shift_days: int = 1
    ) -> pd.DataFrame:
        """
        Align daily sentiment to intraday price data.

        CRITICAL: Shifts sentiment by shift_days to avoid look-ahead bias.

        Args:
            price_df: DataFrame with DatetimeIndex (intraday)
            pair: Trading pair name
            shift_days: Days to shift sentiment (default 1 for no look-ahead)

        Returns:
            Price DataFrame with sentiment columns added
        """
        if self.sentiment_data is None:
            self.load()

        # Get pair-specific sentiment
        pair_sentiment = self.get_pair_sentiment(pair)

        # Shift to avoid look-ahead bias
        pair_sentiment = pair_sentiment.shift(shift_days)

        # Create date column for merging
        result = price_df.copy()
        result['_date'] = result.index.date

        # Create sentiment dataframe with date index
        sentiment_df = pd.DataFrame({
            'sentiment_raw': pair_sentiment
        })
        sentiment_df.index = sentiment_df.index.date

        # Merge
        result = result.reset_index()
        result = result.merge(
            sentiment_df,
            left_on='_date',
            right_index=True,
            how='left'
        )
        result = result.set_index(result.columns[0])
        result = result.drop('_date', axis=1)

        # Forward fill weekends/holidays
        result['sentiment_raw'] = result['sentiment_raw'].ffill().bfill()

        return result
```

### 6.2 SentimentFeatures Class

```python
# src/features/sentiment/sentiment_features.py

class SentimentFeatures:
    """Generate derived sentiment features."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.ma_periods = self.config.get('ma_periods', [3, 7, 14, 30])
        self.std_periods = self.config.get('std_periods', [7, 14])
        self.lag_days = self.config.get('lag_days', [1, 2, 3])

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all sentiment-derived features.

        Args:
            df: DataFrame with 'sentiment_raw' column

        Returns:
            DataFrame with all sentiment features added
        """
        result = df.copy()

        if 'sentiment_raw' not in result.columns:
            raise ValueError("DataFrame must have 'sentiment_raw' column")

        sent = result['sentiment_raw']

        # Moving averages
        for period in self.ma_periods:
            result[f'sentiment_ma_{period}'] = sent.rolling(
                window=period, min_periods=1
            ).mean()

        # Standard deviations (volatility)
        for period in self.std_periods:
            result[f'sentiment_std_{period}'] = sent.rolling(
                window=period, min_periods=1
            ).std().fillna(0)

        # Momentum (current vs MA)
        for period in self.ma_periods[:3]:  # 3, 7, 14
            ma_col = f'sentiment_ma_{period}'
            if ma_col in result.columns:
                result[f'sentiment_momentum_{period}'] = sent - result[ma_col]

        # Rate of change
        result['sentiment_roc_7'] = sent.pct_change(periods=7).fillna(0)

        # Z-score (deviation from 30-day mean)
        if 'sentiment_ma_30' in result.columns and 'sentiment_std_14' in result.columns:
            std = result['sentiment_std_14'].replace(0, 1e-6)
            result['sentiment_zscore'] = (
                (sent - result['sentiment_ma_30']) / std
            ).clip(-3, 3)

        # Regime classification
        result['sentiment_regime'] = pd.cut(
            sent,
            bins=[-float('inf'), -0.05, 0.05, float('inf')],
            labels=[-1, 0, 1]
        ).astype(float)

        # Lag features
        for lag in self.lag_days:
            result[f'sentiment_lag_{lag}'] = sent.shift(lag)

        # Fill any remaining NaNs
        result = result.ffill().bfill()

        return result

    def get_feature_names(self) -> List[str]:
        """Return list of all sentiment feature names."""
        features = ['sentiment_raw']

        for period in self.ma_periods:
            features.append(f'sentiment_ma_{period}')

        for period in self.std_periods:
            features.append(f'sentiment_std_{period}')

        for period in self.ma_periods[:3]:
            features.append(f'sentiment_momentum_{period}')

        features.extend([
            'sentiment_roc_7',
            'sentiment_zscore',
            'sentiment_regime'
        ])

        for lag in self.lag_days:
            features.append(f'sentiment_lag_{lag}')

        return features
```

---

## 7. Model Integration

### 7.1 Updated FeatureProcessor

```python
# Modifications to src/data/processors/features.py

class FeatureProcessor:
    """Process and prepare features for model training."""

    def __init__(self, include_sentiment: bool = True):
        self.include_sentiment = include_sentiment
        self.sentiment_loader = None
        self.sentiment_features = None

        if include_sentiment:
            from src.features.sentiment import SentimentLoader, SentimentFeatures
            self.sentiment_loader = SentimentLoader(
                'data/sentiment/sentiment_epu_20200101_20251231_daily.csv'
            )
            self.sentiment_features = SentimentFeatures()

    def prepare_features(
        self,
        price_df: pd.DataFrame,
        pair: str,
        include_sentiment: bool = None
    ) -> pd.DataFrame:
        """
        Prepare all features including sentiment.

        Args:
            price_df: OHLCV DataFrame
            pair: Trading pair name
            include_sentiment: Override instance setting

        Returns:
            DataFrame with all features
        """
        result = price_df.copy()

        # Technical indicators (existing)
        from src.features.technical import TechnicalIndicators
        tech = TechnicalIndicators()
        result = tech.calculate_all(result)

        # Temporal features (existing)
        result = self.add_temporal_features(result)
        result = self.add_trading_session_features(result)

        # Sentiment features (NEW)
        use_sentiment = include_sentiment if include_sentiment is not None else self.include_sentiment

        if use_sentiment and self.sentiment_loader:
            result = self.sentiment_loader.align_to_price_data(result, pair)
            result = self.sentiment_features.calculate_all(result)

        return result
```

### 7.2 Updated Model Config

```python
# Example: configs/short_term.yaml with sentiment

model:
  name: "short_term_sentiment"
  version: "2.0.0"

  # Input configuration
  sequence_length: 168
  prediction_horizon: [1, 4, 12, 24]

  # Feature groups
  feature_groups:
    - price
    - returns
    - derived
    - temporal
    - session
    - trend
    - momentum
    - volatility
    - volume
    - sentiment    # NEW

  # Sentiment-specific config
  sentiment:
    enabled: true
    features:
      - sentiment_raw
      - sentiment_ma_7
      - sentiment_ma_14
      - sentiment_momentum_7
      - sentiment_std_7
      - sentiment_zscore
      - sentiment_regime
      - sentiment_lag_1
      - sentiment_lag_2
      - sentiment_lag_3

  # Architecture (unchanged)
  cnn_filters: [64, 128, 256]
  cnn_kernel_sizes: [3, 5, 7]
  lstm_hidden_size: 256
  lstm_num_layers: 2
  attention_heads: 8
```

---

## 8. Training Pipeline

### 8.1 Complete Training Script

```python
# scripts/train_with_sentiment.py

"""
Training script for sentiment-enhanced technical analysis models.

Usage:
    python scripts/train_with_sentiment.py \
        --pair EURUSD \
        --model short_term \
        --sentiment \
        --epochs 100
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from src.features.technical import TechnicalIndicators
from src.features.sentiment import SentimentLoader, SentimentFeatures
from src.data.processors import FeatureProcessor
from src.models.technical import ShortTermModel, MediumTermModel, LongTermModel


def load_price_data(pair: str, data_dir: str = "data") -> pd.DataFrame:
    """Load price data for a trading pair."""

    # Forex data
    forex_files = {
        'EURUSD': 'forex/EURUSD_20200101_20251231_5min_combined.csv',
        'GBPUSD': 'forex/GBPUSD_20200101_20251231_5min_combined.csv',
        'USDJPY': 'forex/USDJPY_20200101_20251231_5min_combined.csv',
        'AUDUSD': 'forex/AUDUSD_20200101_20251231_5min_combined.csv',
        'EURGBP': 'forex/EURGBP_20200101_20251231_5min_combined.csv',
    }

    # Crypto data
    crypto_files = {
        'BTCUSDT': 'crypto/BTCUSDT_20200101_20251231_5m.csv',
        'ETHUSDT': 'crypto/ETHUSDT_20200101_20251231_5m.csv',
        'BNBUSDT': 'crypto/BNBUSDT_20200101_20251231_5m.csv',
        'SOLUSDT': 'crypto/SOLUSDT_20200101_20251231_5m.csv',
        'XRPUSDT': 'crypto/XRPUSDT_20200101_20251231_5m.csv',
    }

    all_files = {**forex_files, **crypto_files}

    if pair.upper() not in all_files:
        raise ValueError(f"Unknown pair: {pair}. Available: {list(all_files.keys())}")

    file_path = Path(data_dir) / all_files[pair.upper()]

    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    df.columns = [c.lower() for c in df.columns]

    return df


def prepare_training_data(
    pair: str,
    include_sentiment: bool = True,
    sequence_length: int = 168,
    prediction_horizon: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict:
    """
    Prepare complete training dataset with optional sentiment.

    Returns:
        Dictionary with X_train, y_train, X_val, y_val, X_test, y_test, metadata
    """
    # Load price data
    print(f"Loading price data for {pair}...")
    price_df = load_price_data(pair)
    print(f"  Loaded {len(price_df):,} candles")

    # Calculate technical indicators
    print("Calculating technical indicators...")
    tech = TechnicalIndicators()
    df = tech.calculate_all(price_df)

    # Add temporal features
    processor = FeatureProcessor(include_sentiment=False)
    df = processor.add_temporal_features(df)
    df = processor.add_trading_session_features(df)

    # Add sentiment features
    if include_sentiment:
        print("Adding sentiment features...")
        sentiment_loader = SentimentLoader(
            'data/sentiment/sentiment_epu_20200101_20251231_daily.csv'
        )
        df = sentiment_loader.align_to_price_data(df, pair)

        sentiment_features = SentimentFeatures()
        df = sentiment_features.calculate_all(df)
        print(f"  Added {len(sentiment_features.get_feature_names())} sentiment features")

    # Handle missing values
    df = df.ffill().bfill()
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # Drop any remaining NaN rows
    initial_len = len(df)
    df = df.dropna()
    print(f"  Dropped {initial_len - len(df)} rows with NaN values")

    print(f"Total features: {len(df.columns)}")

    # Create sequences
    print("Creating sequences...")
    feature_names = df.columns.tolist()
    data = df.values

    # Target is future close price
    target_idx = feature_names.index('close')

    X, y = [], []
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length + prediction_horizon - 1, target_idx])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"  Created {len(X):,} sequences")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    # Time series split
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    print(f"\nData split:")
    print(f"  Train: {train_end:,} samples")
    print(f"  Val: {val_end - train_end:,} samples")
    print(f"  Test: {n - val_end:,} samples")

    return {
        'X_train': X[:train_end],
        'y_train': y[:train_end],
        'X_val': X[train_end:val_end],
        'y_val': y[train_end:val_end],
        'X_test': X[val_end:],
        'y_test': y[val_end:],
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'sequence_length': sequence_length,
        'prediction_horizon': prediction_horizon,
        'pair': pair,
        'include_sentiment': include_sentiment,
    }


def train_model(
    model_type: str,
    data: dict,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
) -> tuple:
    """
    Train a model with prepared data.

    Returns:
        (trained_model, training_history, evaluation_metrics)
    """
    # Select model class
    model_classes = {
        'short_term': ShortTermModel,
        'medium_term': MediumTermModel,
        'long_term': LongTermModel,
    }

    if model_type not in model_classes:
        raise ValueError(f"Unknown model: {model_type}")

    # Create model config
    config = {
        'name': f"{model_type}_sentiment" if data['include_sentiment'] else model_type,
        'version': '2.0.0' if data['include_sentiment'] else '1.0.0',
        'sequence_length': data['sequence_length'],
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
    }

    # Initialize model
    print(f"\nInitializing {model_type} model...")
    model = model_classes[model_type](config)
    model.build()

    # Train
    print("Training model...")
    history = model.train(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
    )

    # Evaluate
    print("\nEvaluating on test set...")
    metrics = model.evaluate(data['X_test'], data['y_test'])

    print(f"\nTest Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

    return model, history, metrics


def main():
    parser = argparse.ArgumentParser(description="Train sentiment-enhanced models")
    parser.add_argument('--pair', type=str, required=True, help='Trading pair')
    parser.add_argument('--model', type=str, default='short_term',
                       choices=['short_term', 'medium_term', 'long_term'])
    parser.add_argument('--sentiment', action='store_true', help='Include sentiment features')
    parser.add_argument('--no-sentiment', dest='sentiment', action='store_false')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save-dir', type=str, default='models/trained')
    parser.set_defaults(sentiment=True)

    args = parser.parse_args()

    print("=" * 60)
    print("SENTIMENT-ENHANCED MODEL TRAINING")
    print("=" * 60)
    print(f"Pair: {args.pair}")
    print(f"Model: {args.model}")
    print(f"Include Sentiment: {args.sentiment}")
    print("=" * 60)

    # Prepare data
    data = prepare_training_data(
        pair=args.pair,
        include_sentiment=args.sentiment,
    )

    # Train model
    model, history, metrics = train_model(
        model_type=args.model,
        data=data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    # Save model
    save_path = Path(args.save_dir) / f"{args.pair}_{args.model}"
    if args.sentiment:
        save_path = Path(str(save_path) + "_sentiment")

    print(f"\nSaving model to {save_path}...")
    model.save(save_path)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# tests/test_sentiment_features.py

import pytest
import pandas as pd
import numpy as np

from src.features.sentiment import SentimentLoader, SentimentFeatures


class TestSentimentLoader:

    def test_load_sentiment_data(self):
        loader = SentimentLoader('data/sentiment/sentiment_epu_20200101_20251231_daily.csv')
        df = loader.load()

        assert df is not None
        assert len(df) > 0
        assert 'Sentiment_EURUSD' in df.columns

    def test_get_pair_sentiment(self):
        loader = SentimentLoader('data/sentiment/sentiment_epu_20200101_20251231_daily.csv')
        loader.load()

        eurusd_sent = loader.get_pair_sentiment('EURUSD')
        assert eurusd_sent is not None
        assert len(eurusd_sent) > 0
        assert eurusd_sent.min() >= -0.25  # Approximate bounds
        assert eurusd_sent.max() <= 0.25

    def test_align_to_price_data(self):
        # Create mock price data
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='5min')
        price_df = pd.DataFrame({
            'open': np.random.randn(len(dates)),
            'high': np.random.randn(len(dates)),
            'low': np.random.randn(len(dates)),
            'close': np.random.randn(len(dates)),
            'volume': np.random.randint(100, 1000, len(dates)),
        }, index=dates)

        loader = SentimentLoader('data/sentiment/sentiment_epu_20200101_20251231_daily.csv')
        result = loader.align_to_price_data(price_df, 'EURUSD')

        assert 'sentiment_raw' in result.columns
        assert len(result) == len(price_df)
        assert result['sentiment_raw'].isna().sum() == 0


class TestSentimentFeatures:

    def test_calculate_all(self):
        # Create mock data with sentiment
        dates = pd.date_range('2024-01-01', '2024-06-30', freq='D')
        df = pd.DataFrame({
            'close': np.random.randn(len(dates)),
            'sentiment_raw': np.random.uniform(-0.2, 0.2, len(dates)),
        }, index=dates)

        features = SentimentFeatures()
        result = features.calculate_all(df)

        assert 'sentiment_ma_7' in result.columns
        assert 'sentiment_momentum_7' in result.columns
        assert 'sentiment_zscore' in result.columns
        assert 'sentiment_regime' in result.columns

    def test_feature_names(self):
        features = SentimentFeatures()
        names = features.get_feature_names()

        assert 'sentiment_raw' in names
        assert len(names) >= 10  # At least 10 sentiment features
```

### 9.2 Integration Tests

```python
# tests/test_sentiment_integration.py

import pytest
import numpy as np

from src.data.processors import FeatureProcessor
from src.models.technical import ShortTermModel


class TestSentimentIntegration:

    def test_feature_processor_with_sentiment(self):
        """Test that FeatureProcessor correctly adds sentiment."""
        processor = FeatureProcessor(include_sentiment=True)

        # Load small sample of real data
        import pandas as pd
        price_df = pd.read_csv(
            'data/forex/EURUSD_20200101_20251231_5min_combined.csv',
            parse_dates=['Date'],
            index_col='Date',
            nrows=10000  # Small sample for testing
        )
        price_df.columns = [c.lower() for c in price_df.columns]

        result = processor.prepare_features(price_df, 'EURUSD')

        # Check sentiment features exist
        sentiment_cols = [c for c in result.columns if 'sentiment' in c.lower()]
        assert len(sentiment_cols) >= 5

    def test_model_accepts_sentiment_features(self):
        """Test that model can train with sentiment features."""
        # Create synthetic data with sentiment
        n_samples = 1000
        seq_len = 168
        n_tech_features = 50
        n_sent_features = 12
        n_features = n_tech_features + n_sent_features

        X = np.random.randn(n_samples, seq_len, n_features).astype(np.float32)
        y = np.random.randn(n_samples).astype(np.float32)

        # Train model
        model = ShortTermModel({'epochs': 2, 'batch_size': 32})
        model.build()

        # Should not raise
        history = model.train(
            X[:800], y[:800],
            X[800:900], y[800:900]
        )

        assert 'train_loss' in history
        assert len(history['train_loss']) > 0
```

### 9.3 A/B Testing Framework

```python
# scripts/ab_test_sentiment.py

"""
A/B test comparing models with and without sentiment features.

Usage:
    python scripts/ab_test_sentiment.py --pair EURUSD --model short_term
"""

def run_ab_test(pair: str, model_type: str, n_runs: int = 5):
    """
    Run A/B test comparing sentiment vs no-sentiment models.
    """
    results = {
        'with_sentiment': [],
        'without_sentiment': [],
    }

    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run + 1}/{n_runs}")
        print(f"{'='*60}")

        # Train with sentiment
        print("\n--- Training WITH sentiment ---")
        data_sent = prepare_training_data(pair, include_sentiment=True)
        model_sent, _, metrics_sent = train_model(model_type, data_sent, epochs=50)
        results['with_sentiment'].append(metrics_sent)

        # Train without sentiment
        print("\n--- Training WITHOUT sentiment ---")
        data_no_sent = prepare_training_data(pair, include_sentiment=False)
        model_no_sent, _, metrics_no_sent = train_model(model_type, data_no_sent, epochs=50)
        results['without_sentiment'].append(metrics_no_sent)

    # Aggregate results
    print("\n" + "="*60)
    print("A/B TEST RESULTS")
    print("="*60)

    for variant in ['with_sentiment', 'without_sentiment']:
        print(f"\n{variant.upper()}:")
        for metric in results[variant][0].keys():
            values = [r[metric] for r in results[variant]]
            print(f"  {metric}: {np.mean(values):.4f} (+/- {np.std(values):.4f})")

    # Statistical significance
    from scipy import stats

    for metric in results['with_sentiment'][0].keys():
        sent_values = [r[metric] for r in results['with_sentiment']]
        no_sent_values = [r[metric] for r in results['without_sentiment']]

        t_stat, p_value = stats.ttest_ind(sent_values, no_sent_values)

        improvement = (np.mean(sent_values) - np.mean(no_sent_values)) / np.mean(no_sent_values) * 100

        print(f"\n{metric}:")
        print(f"  Improvement: {improvement:+.2f}%")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
```

---

## 10. Implementation Prompt for Claude Code

Use this prompt in a new Claude Code session to implement the sentiment integration:

---

### IMPLEMENTATION PROMPT

```
I need you to implement sentiment analysis integration for a trading prediction system.
Please read and follow the design document at docs/04-sentiment-integration-design.md.

PROJECT CONTEXT:
- Read CLAUDE.md for project conventions and guidelines
- Read docs/02-technical-analysis-model-design.md for model architecture
- The project uses PyTorch for deep learning models

DATASETS AVAILABLE:
- Price data: data/forex/*.csv and data/crypto/*.csv (5-minute OHLCV)
- Sentiment data: data/sentiment/sentiment_epu_20200101_20251231_daily.csv

IMPLEMENTATION TASKS (in order):

1. CREATE SENTIMENT FEATURE MODULE
   - Create src/features/sentiment/__init__.py
   - Create src/features/sentiment/sentiment_loader.py (SentimentLoader class)
   - Create src/features/sentiment/sentiment_features.py (SentimentFeatures class)
   - Implement all methods as specified in the design doc Section 6

2. UPDATE FEATURE PROCESSOR
   - Modify src/data/processors/features.py to include sentiment features
   - Add sentiment feature group to select_features() method
   - Ensure backward compatibility (sentiment can be disabled)

3. CREATE TRAINING SCRIPT
   - Create scripts/train_with_sentiment.py as specified in Section 8.1
   - Include CLI arguments for pair, model type, sentiment toggle
   - Implement proper data loading for both forex and crypto pairs

4. WRITE UNIT TESTS
   - Create tests/test_sentiment_features.py with tests from Section 9.1
   - Create tests/test_sentiment_integration.py with tests from Section 9.2
   - Ensure all tests pass

5. CREATE CONFIGURATION
   - Create configs/sentiment/sentiment_features.yaml with settings from Section 5.3
   - Update model configs to support sentiment feature flag

6. VALIDATE END-TO-END
   - Run training on EURUSD with sentiment enabled
   - Verify model trains successfully
   - Compare metrics with/without sentiment

QUALITY REQUIREMENTS:
- All code must have type hints
- All functions must have docstrings
- Follow existing code style in the project
- No data leakage - sentiment must be shifted by 1 day
- Handle edge cases (missing data, weekends, holidays)

TESTING:
- Run pytest tests/ after each major component
- Verify no regressions in existing functionality

Please proceed with implementation, starting with Task 1.
```

---

## 11. Expected Outcomes

### 11.1 Success Metrics

| Metric | Without Sentiment | Target with Sentiment |
|--------|------------------|----------------------|
| Direction Accuracy | ~52-55% | >55% |
| MAE | baseline | -5% to -15% |
| Sharpe Ratio | baseline | +10% to +25% |

### 11.2 Deliverables

1. **New Python modules** in `src/features/sentiment/`
2. **Updated feature processor** with sentiment support
3. **Training script** for sentiment-enhanced models
4. **Unit and integration tests** with >90% coverage
5. **Trained models** for comparison (with/without sentiment)
6. **A/B test results** showing improvement metrics

---

## 12. Future Enhancements (Phase 2+)

1. **Dual-Stream Architecture**: Implement separate encoding paths
2. **Attention-based Fusion**: Learn dynamic sentiment weighting
3. **Real-time Sentiment**: Integrate live news sentiment APIs
4. **Multi-source Sentiment**: Add Twitter, news headlines, etc.
5. **Sentiment Regime Detection**: Classify market sentiment states

---

*Document Version: 1.0*
*Created: 2026-01-11*
*Author: AI Trader Development Team*
