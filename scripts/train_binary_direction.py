#!/usr/bin/env python3
"""Train binary direction model (up/down only, no neutral class).

This simplified approach:
1. Uses binary labels (up=1, down=0) instead of 3-class
2. Uses a smaller model with stronger regularization
3. Focuses on clear directional moves (filters out small moves)
"""

import argparse
import logging
import sys
import json
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class SimpleLSTMClassifier(nn.Module):
    """Simple LSTM for binary classification with strong regularization."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input projection with dropout
        self.input_dropout = nn.Dropout(dropout)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Output layers with dropout
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for bidirectional
        self.fc2 = nn.Linear(hidden_dim, 1)  # Binary output

    def forward(self, x):
        # x: (batch, seq, features)
        x = self.input_dropout(x)
        x = torch.relu(self.input_proj(x))

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Use last timestep output
        last_out = lstm_out[:, -1, :]

        # Classification head
        x = self.dropout(last_out)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits.squeeze(-1)


def load_and_prepare_data(timeframe: str, threshold: float = 0.0002):
    """Load data and create binary labels."""
    from src.features.technical.calculator import TechnicalIndicatorCalculator

    # Load data
    data_path = project_root / f"data/forex/derived_proper/{timeframe}/EURUSD_{timeframe}.parquet"
    df = pd.read_parquet(data_path)
    df.columns = [c.lower() for c in df.columns]

    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    logger.info(f"Loaded {len(df)} rows from {data_path}")

    # Calculate features
    calc = TechnicalIndicatorCalculator(model_type="short_term")
    df_features = calc.calculate(df)

    # Create binary labels: 1 if price goes up by threshold, 0 if down by threshold
    # Filter out small moves (neither up nor down by threshold)
    returns = df_features["close"].pct_change(1).shift(-1)  # Next period return

    # Create mask for significant moves
    up_mask = returns > threshold
    down_mask = returns < -threshold
    significant_mask = up_mask | down_mask

    # Filter data to only significant moves
    df_filtered = df_features[significant_mask].copy()
    labels = up_mask[significant_mask].astype(float)

    logger.info(f"After filtering small moves: {len(df_filtered)} samples")
    logger.info(f"Label distribution: UP {labels.mean():.1%}, DOWN {(1-labels).mean():.1%}")

    return df_filtered, labels


def create_sequences(features: pd.DataFrame, labels: pd.Series, seq_length: int = 24):
    """Create sequences for LSTM."""
    # Get feature columns (exclude OHLCV for features)
    feature_cols = [c for c in features.columns if c not in ['open', 'high', 'low', 'close', 'volume']]

    X_data = features[feature_cols].values
    y_data = labels.values

    # Normalize features
    mean = np.nanmean(X_data, axis=0)
    std = np.nanstd(X_data, axis=0)
    std[std == 0] = 1
    X_data = (X_data - mean) / std

    # Handle NaNs
    X_data = np.nan_to_num(X_data, nan=0.0)

    # Create sequences
    X_seq = []
    y_seq = []

    for i in range(len(X_data) - seq_length):
        X_seq.append(X_data[i:i+seq_length])
        y_seq.append(y_data[i+seq_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    # Remove any samples with NaN labels
    valid_mask = ~np.isnan(y_seq)
    X_seq = X_seq[valid_mask]
    y_seq = y_seq[valid_mask]

    return X_seq, y_seq, len(feature_cols)


def train_model(
    X_train, y_train, X_val, y_val,
    input_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.3,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    patience: int = 10,
):
    """Train the binary classifier."""
    device = torch.device("cpu")

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    model = SimpleLSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3,
    )  # Note: verbose parameter removed (deprecated in newer PyTorch)

    # Training loop
    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(X_batch)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            train_total += len(y_batch)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)

                val_loss += loss.item() * len(X_batch)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2%} | Val Loss: {val_loss:.4f} Acc: {val_acc:.2%}")

        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logger.info(f"  -> New best validation accuracy: {val_acc:.2%}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model, best_val_acc, best_val_loss


def evaluate_model(model, X_test, y_test, batch_size: int = 256):
    """Evaluate model on test set."""
    device = torch.device("cpu")
    model.eval()

    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = (all_preds == all_labels).mean()

    # Precision and recall for UP class
    up_mask = all_preds == 1
    if up_mask.sum() > 0:
        precision_up = (all_labels[up_mask] == 1).mean()
    else:
        precision_up = 0

    recall_up = (all_preds[all_labels == 1] == 1).mean() if (all_labels == 1).sum() > 0 else 0

    # Precision and recall for DOWN class
    down_mask = all_preds == 0
    if down_mask.sum() > 0:
        precision_down = (all_labels[down_mask] == 0).mean()
    else:
        precision_down = 0

    recall_down = (all_preds[all_labels == 0] == 0).mean() if (all_labels == 0).sum() > 0 else 0

    return {
        "accuracy": accuracy,
        "precision_up": precision_up,
        "recall_up": recall_up,
        "precision_down": precision_down,
        "recall_down": recall_down,
        "predictions": all_preds,
        "probabilities": all_probs,
        "labels": all_labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Train binary direction classifier")
    parser.add_argument("--timeframe", type=str, default="1H", help="Timeframe to use")
    parser.add_argument("--threshold", type=float, default=0.0002, help="Min move threshold for labeling")
    parser.add_argument("--seq-length", type=int, default=24, help="Sequence length")
    parser.add_argument("--hidden-dim", type=int, default=64, help="LSTM hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("BINARY DIRECTION CLASSIFIER")
    print("=" * 70)
    print(f"Timeframe: {args.timeframe}")
    print(f"Threshold: {args.threshold:.4%}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Model: BiLSTM (hidden={args.hidden_dim}, layers={args.num_layers}, dropout={args.dropout})")
    print("=" * 70)

    # Load and prepare data
    df_features, labels = load_and_prepare_data(args.timeframe, args.threshold)

    # Create sequences
    X, y, input_dim = create_sequences(df_features, labels, args.seq_length)
    logger.info(f"Created {len(X)} sequences with {input_dim} features")

    # Chronological split: 70% train, 15% val, 15% test
    n_train = int(len(X) * 0.7)
    n_val = int(len(X) * 0.15)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Train UP ratio: {y_train.mean():.2%}")
    logger.info(f"Val UP ratio: {y_val.mean():.2%}")
    logger.info(f"Test UP ratio: {y_test.mean():.2%}")

    # Train
    model, best_val_acc, best_val_loss = train_model(
        X_train, y_train, X_val, y_val,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
    )

    # Evaluate on test set
    results = evaluate_model(model, X_test, y_test, args.batch_size)

    print("\n" + "=" * 70)
    print("TEST SET RESULTS")
    print("=" * 70)
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Precision (UP):   {results['precision_up']:.2%}")
    print(f"Recall (UP):      {results['recall_up']:.2%}")
    print(f"Precision (DOWN): {results['precision_down']:.2%}")
    print(f"Recall (DOWN):    {results['recall_down']:.2%}")

    # Prediction distribution
    pred_up = (results['predictions'] == 1).mean()
    print(f"\nPrediction distribution: UP {pred_up:.1%}, DOWN {1-pred_up:.1%}")
    print(f"Actual distribution:     UP {y_test.mean():.1%}, DOWN {1-y_test.mean():.1%}")
    print("=" * 70)

    # Save model
    output_dir = project_root / "models" / "binary_direction"
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "seq_length": args.seq_length,
        "threshold": args.threshold,
        "results": {k: float(v) if isinstance(v, (float, np.floating)) else v
                   for k, v in results.items() if k not in ['predictions', 'probabilities', 'labels']},
    }, output_dir / f"binary_{args.timeframe}.pt")

    logger.info(f"Model saved to {output_dir / f'binary_{args.timeframe}.pt'}")

    return results


if __name__ == "__main__":
    main()
