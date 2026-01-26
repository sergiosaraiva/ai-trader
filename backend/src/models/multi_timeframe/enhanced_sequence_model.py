"""Enhanced sequence model using technical indicators as input.

Instead of just OHLCV, this model takes sequences of technical indicators,
allowing it to learn temporal patterns from RSI, MACD, moving averages, etc.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from .labeling import AdvancedLabeler, LabelingConfig, LabelMethod

logger = logging.getLogger(__name__)


@dataclass
class EnhancedSequenceConfig:
    """Configuration for enhanced sequence model with technical indicators."""

    name: str = "1H"
    base_timeframe: str = "1H"

    # Sequence parameters
    sequence_length: int = 30  # Shorter since features are richer

    # Labeling
    tp_pips: float = 25.0
    sl_pips: float = 15.0
    max_holding_bars: int = 12

    # Input features to use (selected indicators)
    input_features: List[str] = None  # Will be set dynamically

    # Architecture - larger to handle more features
    hidden_dim: int = 128
    cnn_channels: List[int] = None
    cnn_kernel_sizes: List[int] = None
    transformer_dim: int = 128
    transformer_heads: int = 4
    transformer_layers: int = 2
    dropout: float = 0.3

    # Training
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.0005
    weight_decay: float = 1e-4
    patience: int = 10

    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [64, 128]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 5]

    @classmethod
    def hourly_model(cls) -> "EnhancedSequenceConfig":
        return cls(
            name="1H",
            base_timeframe="1H",
            sequence_length=30,  # 30 hours
            tp_pips=25.0,
            sl_pips=15.0,
            max_holding_bars=12,
        )

    @classmethod
    def four_hour_model(cls) -> "EnhancedSequenceConfig":
        return cls(
            name="4H",
            base_timeframe="4H",
            sequence_length=30,  # 5 days
            tp_pips=50.0,
            sl_pips=25.0,
            max_holding_bars=18,
        )


# Key indicators to use as sequence features
DEFAULT_SEQUENCE_FEATURES = [
    # Price-based (normalized)
    "close", "high", "low",
    # Trend
    "ema_8", "ema_21", "ema_55",
    "adx_14", "plus_di_14", "minus_di_14",
    "supertrend_dir_10",
    # Momentum
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "stoch_k_14", "stoch_d_14",
    "cci_20",
    # Volatility
    "atr_14",
    "bb_pctb_20", "bb_width_20",
    # Volume
    "obv",
    # Enhanced features
    "price_roc3", "price_roc6",
    "rsi_14_roc3",
    "returns_zscore",
    "price_pctl_20", "price_pctl_50",
    # Cross-TF (if available)
    "htf_4H_trend", "htf_4H_rsi",
    "htf_D_trend", "htf_D_rsi",
    "trend_alignment",
]


class PositionalEncoding(nn.Module):
    """Add positional information to sequence."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class EnhancedCNNTransformer(nn.Module):
    """CNN-Transformer model for indicator sequences."""

    def __init__(self, n_features: int, config: EnhancedSequenceConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_norm = nn.LayerNorm(n_features)
        self.input_proj = nn.Linear(n_features, config.hidden_dim)

        # CNN for local pattern extraction
        self.cnn_layers = nn.ModuleList()
        in_channels = config.hidden_dim

        for out_channels, kernel_size in zip(config.cnn_channels, config.cnn_kernel_sizes):
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ))
            in_channels = out_channels

        # Project CNN output to transformer dimension
        self.cnn_proj = nn.Linear(config.cnn_channels[-1], config.transformer_dim)

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
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)

        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(config.transformer_dim, 1),
            nn.Softmax(dim=1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.transformer_dim, config.transformer_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.transformer_dim // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)

        # Normalize and project input
        x = self.input_norm(x)
        x = self.input_proj(x)

        # CNN: (batch, seq, hidden) -> (batch, hidden, seq) -> CNN -> (batch, seq, channels)
        x = x.transpose(1, 2)
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        x = x.transpose(1, 2)

        # Project to transformer dim
        x = self.cnn_proj(x)

        # Positional encoding
        x = self.pos_encoding(x)

        # Transformer
        x = self.transformer(x)

        # Attention pooling (weighted average based on learned attention)
        attn_weights = self.attention_pool(x)  # (batch, seq, 1)
        x = (x * attn_weights).sum(dim=1)  # (batch, transformer_dim)

        # Classification
        logits = self.classifier(x)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)


class EnhancedSequenceDataset:
    """Prepare sequence data with technical indicators."""

    def __init__(self, config: EnhancedSequenceConfig):
        self.config = config
        self.labeler = AdvancedLabeler(LabelingConfig(
            method=LabelMethod.TRIPLE_BARRIER,
            tp_pips=config.tp_pips,
            sl_pips=config.sl_pips,
            max_holding_bars=config.max_holding_bars,
        ))
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []

    def prepare(
        self,
        df: pd.DataFrame,
        fit_scaler: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare sequences with technical indicators.

        Args:
            df: DataFrame with technical indicators already calculated
            fit_scaler: Whether to fit the scaler (True for training)

        Returns:
            X: (n_samples, seq_length, n_features)
            y: (n_samples,)
            feature_names: List of feature names used
        """
        # Create labels
        labels, valid_mask = self.labeler.create_labels(df)

        # Select features that exist in the dataframe
        available_features = []
        for feat in DEFAULT_SEQUENCE_FEATURES:
            if feat in df.columns:
                available_features.append(feat)

        self.feature_names = available_features
        logger.info(f"Using {len(available_features)} features for sequence model")

        # Get feature matrix
        feature_data = df[available_features].values

        # Handle NaN by forward filling then backward filling
        feature_df = pd.DataFrame(feature_data, columns=available_features)
        feature_df = feature_df.ffill().bfill()
        feature_data = feature_df.values

        # Scale features
        if fit_scaler:
            feature_data_scaled = self.scaler.fit_transform(feature_data)
        else:
            feature_data_scaled = self.scaler.transform(feature_data)

        # Clip extreme values
        feature_data_scaled = np.clip(feature_data_scaled, -5, 5)

        # Create sequences
        seq_len = self.config.sequence_length
        n_samples = len(df) - seq_len

        X = []
        y = []

        for i in range(seq_len, len(df)):
            if valid_mask.iloc[i]:
                seq = feature_data_scaled[i - seq_len:i]
                label = labels.iloc[i]

                if not np.isnan(seq).any() and not np.isnan(label):
                    X.append(seq)
                    y.append(int(label))

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        logger.info(f"Prepared {len(X)} sequences of shape {X.shape[1:]}")
        logger.info(f"Label distribution: {y.mean():.1%} bullish, {1-y.mean():.1%} bearish")

        return X, y, available_features


class EnhancedSequenceTrainer:
    """Train enhanced sequence model."""

    def __init__(self, config: EnhancedSequenceConfig, device: str = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[EnhancedCNNTransformer, Dict]:
        """Train the model."""
        n_features = X_train.shape[2]

        # Handle class imbalance with weighted loss
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (2 * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val),
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        # Create model
        model = EnhancedCNNTransformer(n_features, self.config).to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Loss with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )

        # Training loop
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

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * len(y_batch)
                train_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                train_total += len(y_batch)

            scheduler.step()
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

            logger.info(f"Epoch {epoch+1}/{self.config.epochs}: "
                       f"train_acc={train_acc:.2%}, val_acc={val_acc:.2%}, "
                       f"lr={optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        model.load_state_dict(best_model_state)
        model.to(self.device)

        # Calculate metrics
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


class EnhancedSequencePredictor:
    """Predictor for enhanced sequence model."""

    def __init__(
        self,
        model: EnhancedCNNTransformer,
        config: EnhancedSequenceConfig,
        scaler: StandardScaler,
        feature_names: List[str],
        device: str = None,
    ):
        self.model = model
        self.config = config
        self.scaler = scaler
        self.feature_names = feature_names
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def predict(self, sequence: np.ndarray) -> Tuple[int, float, float, float]:
        """Predict from a single sequence (already scaled)."""
        with torch.no_grad():
            x = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            probs = self.model.predict_proba(x).cpu().numpy()[0]

        pred = probs.argmax()
        confidence = probs.max()
        prob_down, prob_up = probs[0], probs[1]

        return int(pred), float(confidence), float(prob_up), float(prob_down)

    def predict_batch(self, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Batch prediction."""
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
            'scaler': self.scaler,
            'feature_names': self.feature_names,
        }, path)
        logger.info(f"Saved enhanced sequence model to {path}")

    @classmethod
    def load(cls, path: Path, device: str = None) -> "EnhancedSequencePredictor":
        """Load model."""
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        config = checkpoint['config']
        scaler = checkpoint['scaler']
        feature_names = checkpoint['feature_names']

        model = EnhancedCNNTransformer(len(feature_names), config)
        model.load_state_dict(checkpoint['model_state'])

        logger.info(f"Loaded enhanced sequence model from {path}")
        return cls(model, config, scaler, feature_names, device)
