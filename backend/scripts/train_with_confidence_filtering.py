#!/usr/bin/env python3
"""Train model with confidence filtering and regime detection.

Key concepts:
1. Confidence Filtering: Only trade when model is confident (prob > threshold)
2. Regime Detection: Only trade in favorable market conditions (trending)

This filters out low-quality signals and improves effective win rate.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ConfidentLSTMClassifier(nn.Module):
    """LSTM classifier that outputs probability for confidence filtering."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_dropout = nn.Dropout(dropout)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_dropout(x)
        x = torch.relu(self.input_proj(x))
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        x = self.dropout(last_out)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits.squeeze(-1)


def calculate_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate market regime indicators.

    Regimes:
    - Trending UP: ADX > 25, price above MA, positive momentum
    - Trending DOWN: ADX > 25, price below MA, negative momentum
    - Ranging: ADX < 20, price oscillating around MA
    - Volatile: High ATR, large swings
    """
    close = df['close']
    high = df['high']
    low = df['low']

    # ADX-like trend strength (simplified)
    # Using directional movement
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    plus_dm = plus_dm.where(plus_dm > minus_dm, 0).clip(lower=0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0).clip(lower=0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    atr_14 = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_14)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_14)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.rolling(14).mean()

    # Moving averages for trend direction
    ma_20 = close.rolling(20).mean()
    ma_50 = close.rolling(50).mean()

    # Volatility regime
    volatility = close.pct_change().rolling(20).std() * np.sqrt(252)
    vol_ma = volatility.rolling(50).mean()

    # Momentum
    roc_10 = close.pct_change(10)

    # Regime classification
    regime = pd.Series(index=df.index, dtype='object')
    regime[:] = 'unknown'

    # Trending conditions
    trending_up = (adx > 25) & (close > ma_20) & (ma_20 > ma_50) & (roc_10 > 0)
    trending_down = (adx > 25) & (close < ma_20) & (ma_20 < ma_50) & (roc_10 < 0)
    ranging = (adx < 20)
    volatile = (volatility > vol_ma * 1.5)

    regime[trending_up] = 'trending_up'
    regime[trending_down] = 'trending_down'
    regime[ranging & ~volatile] = 'ranging'
    regime[volatile] = 'volatile'

    # Trend strength score (0-100)
    trend_strength = adx.fillna(0)

    # Regime favorability score (higher = better for trading)
    favorability = pd.Series(0.0, index=df.index)
    favorability[trending_up | trending_down] = trend_strength[trending_up | trending_down] / 50  # 0.5-1.0
    favorability[ranging] = 0.3
    favorability[volatile] = 0.2
    favorability = favorability.clip(0, 1)

    return pd.DataFrame({
        'adx': adx,
        'trend_strength': trend_strength,
        'volatility': volatility,
        'regime': regime,
        'favorability': favorability,
        'ma_20': ma_20,
        'ma_50': ma_50,
        'roc_10': roc_10,
    })


def load_and_prepare_data(timeframe: str, threshold: float = 0.0005):
    """Load data with regime features."""
    from src.features.technical.calculator import TechnicalIndicatorCalculator

    data_path = project_root / f"data/forex/derived_proper/{timeframe}/EURUSD_{timeframe}.parquet"
    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    logger.info(f"Loaded {len(df)} rows")

    # Calculate technical features
    calc = TechnicalIndicatorCalculator(model_type="short_term")
    df_features = calc.calculate(df)

    # Calculate regime features
    regime_df = calculate_regime_features(df_features)

    # Add regime features to main dataframe
    for col in ['adx', 'trend_strength', 'volatility', 'favorability']:
        df_features[f'regime_{col}'] = regime_df[col]
    df_features['regime'] = regime_df['regime']

    # Create labels
    returns = df_features["close"].pct_change(1).shift(-1)
    up_mask = returns > threshold
    down_mask = returns < -threshold
    significant_mask = up_mask | down_mask

    # Filter
    df_filtered = df_features[significant_mask].copy()
    labels = up_mask[significant_mask].astype(float)
    regimes = df_filtered['regime'].copy()
    favorability = df_filtered['regime_favorability'].copy()

    logger.info(f"After filtering: {len(df_filtered)} samples")
    logger.info(f"Label distribution: UP {labels.mean():.1%}, DOWN {(1-labels).mean():.1%}")

    # Regime distribution
    regime_counts = regimes.value_counts()
    logger.info(f"Regime distribution:")
    for regime, count in regime_counts.items():
        logger.info(f"  {regime}: {count} ({count/len(regimes):.1%})")

    return df_filtered, labels, regimes, favorability


def create_sequences(features: pd.DataFrame, labels: pd.Series, seq_length: int = 24):
    """Create sequences."""
    feature_cols = [c for c in features.columns
                   if c not in ['open', 'high', 'low', 'close', 'volume', 'regime']]

    X_data = features[feature_cols].values
    y_data = labels.values

    # Normalize
    mean = np.nanmean(X_data, axis=0)
    std = np.nanstd(X_data, axis=0)
    std[std == 0] = 1
    X_data = (X_data - mean) / std
    X_data = np.nan_to_num(X_data, nan=0.0)

    # Create sequences
    X_seq, y_seq, idx_seq = [], [], []
    for i in range(len(X_data) - seq_length):
        X_seq.append(X_data[i:i+seq_length])
        y_seq.append(y_data[i+seq_length])
        idx_seq.append(i+seq_length)

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    idx_seq = np.array(idx_seq)

    valid_mask = ~np.isnan(y_seq)
    return X_seq[valid_mask], y_seq[valid_mask], idx_seq[valid_mask], len(feature_cols)


def train_model(X_train, y_train, X_val, y_val, input_dim, config):
    """Train the model."""
    device = torch.device("cpu")

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    model = ConfidentLSTMClassifier(
        input_dim=input_dim,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(config['epochs']):
        # Train
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch.to(device))
                probs = torch.sigmoid(logits)
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(y_batch.numpy())

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        val_acc = accuracy_score(val_labels, (val_preds > 0.5).astype(int))

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}: Val Acc = {val_acc:.2%}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, best_val_acc


def evaluate_with_filtering(model, X_test, y_test, regimes_test, favorability_test,
                            confidence_thresholds=[0.5, 0.6, 0.7, 0.8]):
    """Evaluate model with confidence and regime filtering."""
    device = torch.device("cpu")
    model.eval()

    # Get predictions and probabilities
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()

    predictions = (probs > 0.5).astype(int)

    # Confidence = distance from 0.5 (how sure the model is)
    confidence = np.abs(probs - 0.5) * 2  # Scale to 0-1

    results = []

    print("\n" + "=" * 80)
    print("EVALUATION WITH CONFIDENCE & REGIME FILTERING")
    print("=" * 80)

    # 1. Baseline (no filtering)
    baseline_acc = accuracy_score(y_test, predictions)
    print(f"\n[BASELINE] No filtering:")
    print(f"  Trades: {len(y_test)} (100%)")
    print(f"  Accuracy: {baseline_acc:.2%}")
    results.append(('Baseline', len(y_test), baseline_acc))

    # 2. Confidence filtering only
    print(f"\n[CONFIDENCE FILTERING]")
    for conf_thresh in confidence_thresholds:
        mask = confidence >= (conf_thresh - 0.5) * 2  # Convert threshold to confidence scale
        if mask.sum() > 0:
            filtered_acc = accuracy_score(y_test[mask], predictions[mask])
            trade_pct = mask.mean() * 100
            print(f"  Confidence >= {conf_thresh:.0%}: {mask.sum()} trades ({trade_pct:.1f}%), Accuracy: {filtered_acc:.2%}")
            results.append((f'Conf>={conf_thresh:.0%}', mask.sum(), filtered_acc))

    # 3. Regime filtering only
    print(f"\n[REGIME FILTERING]")
    for regime in ['trending_up', 'trending_down', 'ranging', 'volatile']:
        mask = regimes_test == regime
        if mask.sum() > 10:
            regime_acc = accuracy_score(y_test[mask], predictions[mask])
            trade_pct = mask.mean() * 100
            print(f"  {regime}: {mask.sum()} trades ({trade_pct:.1f}%), Accuracy: {regime_acc:.2%}")
            results.append((f'Regime:{regime}', mask.sum(), regime_acc))

    # 4. Trending only (up or down)
    trending_mask = (regimes_test == 'trending_up') | (regimes_test == 'trending_down')
    if trending_mask.sum() > 10:
        trending_acc = accuracy_score(y_test[trending_mask], predictions[trending_mask])
        print(f"  TRENDING (up+down): {trending_mask.sum()} trades ({trending_mask.mean()*100:.1f}%), Accuracy: {trending_acc:.2%}")
        results.append(('Trending', trending_mask.sum(), trending_acc))

    # 5. Combined: High confidence + Trending
    print(f"\n[COMBINED FILTERING: Confidence + Regime]")
    for conf_thresh in [0.6, 0.7, 0.8]:
        conf_mask = confidence >= (conf_thresh - 0.5) * 2
        combined_mask = conf_mask & trending_mask
        if combined_mask.sum() > 10:
            combined_acc = accuracy_score(y_test[combined_mask], predictions[combined_mask])
            trade_pct = combined_mask.mean() * 100
            print(f"  Conf>={conf_thresh:.0%} + Trending: {combined_mask.sum()} trades ({trade_pct:.1f}%), Accuracy: {combined_acc:.2%}")
            results.append((f'Conf>={conf_thresh:.0%}+Trend', combined_mask.sum(), combined_acc))

    # 6. Favorability score filtering
    print(f"\n[FAVORABILITY FILTERING]")
    for fav_thresh in [0.3, 0.5, 0.7]:
        mask = favorability_test >= fav_thresh
        if mask.sum() > 10:
            fav_acc = accuracy_score(y_test[mask], predictions[mask])
            trade_pct = mask.mean() * 100
            print(f"  Favorability >= {fav_thresh}: {mask.sum()} trades ({trade_pct:.1f}%), Accuracy: {fav_acc:.2%}")
            results.append((f'Fav>={fav_thresh}', mask.sum(), fav_acc))

    # 7. Best combined strategy
    print(f"\n[BEST STRATEGIES]")
    best_strategies = sorted(results, key=lambda x: x[2] if x[1] > 20 else 0, reverse=True)[:5]
    for name, trades, acc in best_strategies:
        if trades > 20:
            print(f"  {name}: {trades} trades, {acc:.2%} accuracy")

    print("=" * 80)

    return results, probs, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", type=str, default="1H")
    parser.add_argument("--threshold", type=float, default=0.0005)
    parser.add_argument("--seq-length", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    config = vars(args)

    print("\n" + "=" * 80)
    print("CONFIDENCE & REGIME FILTERED TRADING MODEL")
    print("=" * 80)
    print(f"Timeframe: {args.timeframe}, Threshold: {args.threshold:.4%}")
    print("=" * 80)

    # Load data with regimes
    df_features, labels, regimes, favorability = load_and_prepare_data(args.timeframe, args.threshold)

    # Create sequences
    X, y, idx, input_dim = create_sequences(df_features, labels, args.seq_length)

    # Map regimes and favorability to sequence indices
    regimes_arr = regimes.values[idx]
    favorability_arr = favorability.values[idx]

    logger.info(f"Created {len(X)} sequences with {input_dim} features")

    # Split
    n_train = int(len(X) * 0.7)
    n_val = int(len(X) * 0.15)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    regimes_test = regimes_arr[n_train+n_val:]
    favorability_test = favorability_arr[n_train+n_val:]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Train
    model, best_val_acc = train_model(X_train, y_train, X_val, y_val, input_dim, config)
    logger.info(f"Best validation accuracy: {best_val_acc:.2%}")

    # Evaluate with filtering
    results, probs, confidence = evaluate_with_filtering(
        model, X_test, y_test, regimes_test, favorability_test
    )

    # Save model
    output_dir = project_root / "models" / "filtered_trading"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "config": config,
        "results": results,
    }, output_dir / f"filtered_{args.timeframe}.pt")

    logger.info(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
