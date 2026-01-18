"""Sequence-based model using CNN-Transformer architecture.

This model takes raw OHLCV sequences as input and learns temporal patterns
directly from price action, complementing the feature-based XGBoost model.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .labeling import AdvancedLabeler, LabelingConfig, LabelMethod

logger = logging.getLogger(__name__)


@dataclass
class SequenceModelConfig:
    """Configuration for sequence model."""

    name: str = "1H"
    base_timeframe: str = "1H"

    # Sequence parameters
    sequence_length: int = 60  # Number of bars to look back

    # Labeling (same as XGBoost for fair comparison)
    tp_pips: float = 25.0
    sl_pips: float = 15.0
    max_holding_bars: int = 12

    # Architecture
    input_features: int = 5  # OHLCV
    cnn_channels: List[int] = None  # [32, 64]
    cnn_kernel_sizes: List[int] = None  # [3, 5]
    transformer_dim: int = 64
    transformer_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.2

    # Training
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 10  # Early stopping

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 5]

    @classmethod
    def hourly_model(cls) -> "SequenceModelConfig":
        return cls(
            name="1H",
            base_timeframe="1H",
            sequence_length=60,  # 60 hours = 2.5 days
            tp_pips=25.0,
            sl_pips=15.0,
            max_holding_bars=12,
        )

    @classmethod
    def four_hour_model(cls) -> "SequenceModelConfig":
        return cls(
            name="4H",
            base_timeframe="4H",
            sequence_length=60,  # 60 * 4H = 10 days
            tp_pips=50.0,
            sl_pips=25.0,
            max_holding_bars=18,
        )


class CNNFeatureExtractor(nn.Module):
    """Extract local patterns using 1D convolutions."""

    def __init__(
        self,
        input_channels: int,
        cnn_channels: List[int],
        kernel_sizes: List[int],
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        in_channels = input_channels

        for out_channels, kernel_size in zip(cnn_channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)
        self.output_channels = cnn_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        # Back to (batch, seq_len, channels)
        x = x.transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    """Add positional information to sequence."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CNNTransformerModel(nn.Module):
    """CNN-Transformer hybrid for sequence classification."""

    def __init__(self, config: SequenceModelConfig):
        super().__init__()
        self.config = config

        # CNN for local pattern extraction
        self.cnn = CNNFeatureExtractor(
            input_channels=config.input_features,
            cnn_channels=config.cnn_channels,
            kernel_sizes=config.cnn_kernel_sizes,
            dropout=config.dropout,
        )

        # Project CNN output to transformer dimension
        self.input_projection = nn.Linear(self.cnn.output_channels, config.transformer_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            config.transformer_dim,
            max_len=config.sequence_length,
            dropout=config.dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.transformer_dim, config.transformer_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.transformer_dim // 2, 2),  # 2 classes: down, up
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)

        # Extract local patterns
        x = self.cnn(x)

        # Project to transformer dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling over sequence
        x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


class SequenceDataset:
    """Prepare sequence data for training."""

    def __init__(
        self,
        config: SequenceModelConfig,
    ):
        self.config = config
        self.labeler = AdvancedLabeler(LabelingConfig(
            method=LabelMethod.TRIPLE_BARRIER,
            tp_pips=config.tp_pips,
            sl_pips=config.sl_pips,
            max_holding_bars=config.max_holding_bars,
        ))

    def prepare(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences and labels.

        Args:
            df: OHLCV DataFrame at target timeframe

        Returns:
            X: (n_samples, seq_length, 5) - OHLCV sequences
            y: (n_samples,) - labels (0=down, 1=up)
        """
        # Create labels
        labels, valid_mask = self.labeler.create_labels(df)

        # Normalize OHLCV data using rolling windows
        df_norm = self._normalize_ohlcv(df)

        # Create sequences
        seq_len = self.config.sequence_length
        n_samples = len(df) - seq_len

        X = []
        y = []

        for i in range(seq_len, len(df)):
            if valid_mask.iloc[i]:
                # Get sequence of normalized OHLCV
                seq = df_norm.iloc[i - seq_len:i][['open', 'high', 'low', 'close', 'volume']].values
                label = labels.iloc[i]

                if not np.isnan(seq).any() and not np.isnan(label):
                    X.append(seq)
                    y.append(int(label))

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        logger.info(f"Prepared {len(X)} sequences of length {seq_len}")
        logger.info(f"Label distribution: {y.mean():.1%} bullish, {1-y.mean():.1%} bearish")

        return X, y

    def _normalize_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize OHLCV data for neural network input.

        Uses percentage returns relative to first bar in each sequence.
        This is computed during sequence creation, but we prepare the
        rolling statistics here.
        """
        df_norm = df.copy()

        # Price as percentage change from rolling mean
        rolling_mean = df['close'].rolling(20).mean()
        rolling_std = df['close'].rolling(20).std()

        for col in ['open', 'high', 'low', 'close']:
            df_norm[col] = (df[col] - rolling_mean) / (rolling_std + 1e-8)

        # Volume as z-score
        vol_mean = df['volume'].rolling(20).mean()
        vol_std = df['volume'].rolling(20).std()
        df_norm['volume'] = (df['volume'] - vol_mean) / (vol_std + 1e-8)

        # Clip extreme values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_norm[col] = df_norm[col].clip(-5, 5)

        return df_norm


class SequenceTrainer:
    """Train sequence model with early stopping."""

    def __init__(self, config: SequenceModelConfig, device: str = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[CNNTransformerModel, Dict]:
        """Train the model.

        Returns:
            Trained model and metrics dict
        """
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        # Create model
        model = CNNTransformerModel(self.config).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )

        # Training loop with early stopping
        best_val_acc = 0
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Train
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                train_loss += loss.item() * len(y_batch)
                train_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                train_total += len(y_batch)

            train_acc = train_correct / train_total

            # Validate
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    logits = model(X_batch)
                    val_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                    val_total += len(y_batch)

            val_acc = val_correct / val_total
            scheduler.step(val_acc)

            logger.info(f"Epoch {epoch+1}/{self.config.epochs}: "
                       f"train_acc={train_acc:.2%}, val_acc={val_acc:.2%}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        model.load_state_dict(best_model_state)

        # Calculate metrics at different confidence levels
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            probs = model.predict_proba(X_val_tensor).cpu().numpy()
            preds = probs.argmax(axis=1)
            confs = probs.max(axis=1)

        metrics = {
            'train_accuracy': train_acc,
            'val_accuracy': best_val_acc,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
        }

        for thresh in [0.55, 0.60, 0.65, 0.70]:
            mask = confs >= thresh
            if mask.sum() > 0:
                acc = (preds[mask] == y_val[mask]).mean()
                metrics[f'val_acc_conf_{int(thresh*100)}'] = acc
                metrics[f'val_samples_conf_{int(thresh*100)}'] = int(mask.sum())

        return model, metrics


class SequencePredictor:
    """Wrapper for inference with trained sequence model."""

    def __init__(
        self,
        model: CNNTransformerModel,
        config: SequenceModelConfig,
        device: str = None,
    ):
        self.model = model
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # For normalizing new data
        self.rolling_mean = None
        self.rolling_std = None

    def predict(self, sequence: np.ndarray) -> Tuple[int, float, float, float]:
        """Predict from a single sequence.

        Args:
            sequence: (seq_length, 5) OHLCV data (already normalized)

        Returns:
            (direction, confidence, prob_up, prob_down)
        """
        with torch.no_grad():
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            probs = self.model.predict_proba(x).cpu().numpy()[0]

        pred = probs.argmax()
        confidence = probs.max()
        prob_down, prob_up = probs[0], probs[1]

        return int(pred), float(confidence), float(prob_up), float(prob_down)

    def predict_batch(self, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batch prediction.

        Args:
            sequences: (n_samples, seq_length, 5)

        Returns:
            (predictions, confidences)
        """
        with torch.no_grad():
            x = torch.FloatTensor(sequences).to(self.device)
            probs = self.model.predict_proba(x).cpu().numpy()

        predictions = probs.argmax(axis=1)
        confidences = probs.max(axis=1)

        return predictions, confidences

    def save(self, path: Path) -> None:
        """Save model."""
        torch.save({
            'model_state': self.model.state_dict(),
            'config': self.config,
        }, path)
        logger.info(f"Saved sequence model to {path}")

    @classmethod
    def load(cls, path: Path, device: str = None) -> "SequencePredictor":
        """Load model."""
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        config = checkpoint['config']
        model = CNNTransformerModel(config)
        model.load_state_dict(checkpoint['model_state'])

        logger.info(f"Loaded sequence model from {path}")
        return cls(model, config, device)
