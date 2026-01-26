"""Short-term prediction model using CNN-LSTM with Attention."""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np

from ..base import Prediction, ModelRegistry
from .base import TechnicalBaseModel


class ShortTermModel(TechnicalBaseModel):
    """
    Short-term prediction model for intraday trading.

    Architecture: CNN + Bi-LSTM + Multi-Head Attention

    Input: 168 hourly candles (7 days)
    Output: Price predictions for 1H, 4H, 12H, 24H ahead
    """

    DEFAULT_CONFIG = {
        "name": "short_term",
        "version": "1.0.0",
        "sequence_length": 168,
        "prediction_horizon": [1, 4, 12, 24],

        # CNN
        "cnn_filters": [64, 128, 256],
        "cnn_kernel_sizes": [3, 5, 7],
        "cnn_dropout": 0.2,

        # LSTM
        "lstm_hidden_size": 256,
        "lstm_num_layers": 2,
        "lstm_dropout": 0.3,
        "lstm_bidirectional": True,

        # Attention
        "attention_heads": 8,
        "attention_dim": 256,

        # Training
        "batch_size": 64,
        "learning_rate": 1e-4,
        "epochs": 100,
        "early_stopping_patience": 15,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize short-term model."""
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

    def build(self) -> None:
        """Build CNN-LSTM-Attention architecture."""
        try:
            import torch
            import torch.nn as nn

            config = self.config

            class CNNLSTMAttention(nn.Module):
                def __init__(self, input_size: int, config: dict):
                    super().__init__()

                    # CNN layers
                    cnn_layers = []
                    in_channels = input_size
                    for i, (filters, kernel) in enumerate(
                        zip(config["cnn_filters"], config["cnn_kernel_sizes"])
                    ):
                        cnn_layers.extend([
                            nn.Conv1d(in_channels, filters, kernel, padding=kernel // 2),
                            nn.BatchNorm1d(filters),
                            nn.ReLU(),
                            nn.Dropout(config["cnn_dropout"]),
                        ])
                        in_channels = filters

                    self.cnn = nn.Sequential(*cnn_layers)

                    # Bi-LSTM
                    lstm_input_size = config["cnn_filters"][-1]
                    self.lstm = nn.LSTM(
                        input_size=lstm_input_size,
                        hidden_size=config["lstm_hidden_size"],
                        num_layers=config["lstm_num_layers"],
                        dropout=config["lstm_dropout"],
                        bidirectional=config["lstm_bidirectional"],
                        batch_first=True,
                    )

                    lstm_output_size = config["lstm_hidden_size"]
                    if config["lstm_bidirectional"]:
                        lstm_output_size *= 2

                    # Multi-Head Attention
                    self.attention = nn.MultiheadAttention(
                        embed_dim=lstm_output_size,
                        num_heads=config["attention_heads"],
                        dropout=0.1,
                        batch_first=True,
                    )

                    # Output heads
                    self.price_head = nn.Sequential(
                        nn.Linear(lstm_output_size, 128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, len(config["prediction_horizon"])),
                    )

                    self.direction_head = nn.Sequential(
                        nn.Linear(lstm_output_size, 64),
                        nn.ReLU(),
                        nn.Linear(64, 3),  # up, down, neutral
                    )

                    self.confidence_head = nn.Sequential(
                        nn.Linear(lstm_output_size, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                        nn.Sigmoid(),
                    )

                def forward(self, x):
                    # x shape: (batch, seq_len, features)
                    # CNN expects (batch, channels, seq_len)
                    x = x.permute(0, 2, 1)
                    x = self.cnn(x)
                    x = x.permute(0, 2, 1)

                    # LSTM
                    lstm_out, _ = self.lstm(x)

                    # Attention
                    attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

                    # Use last time step
                    final = attn_out[:, -1, :]

                    # Outputs
                    price = self.price_head(final)
                    direction = self.direction_head(final)
                    confidence = self.confidence_head(final)

                    return {
                        "price": price,
                        "direction": direction,
                        "confidence": confidence,
                        "attention_weights": attn_weights,
                    }

            # Will set input_size when we know the feature count
            self._model_class = CNNLSTMAttention
            self._model_config = config

        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install torch")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train the model."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Build model with correct input size
        input_size = X_train.shape[2]
        self.model = self._model_class(input_size, self._model_config)
        self.model.to(self.device)

        # Prepare data
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
        )

        if X_val is not None and y_val is not None:
            X_val_t = torch.FloatTensor(X_val)
            y_val_t = torch.FloatTensor(y_val)
            val_dataset = TensorDataset(X_val_t, y_val_t)
            val_loader = DataLoader(val_dataset, batch_size=self.config["batch_size"])
        else:
            val_loader = None

        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config["learning_rate"] * 10,
            epochs=self.config["epochs"],
            steps_per_epoch=len(train_loader),
        )
        criterion = nn.HuberLoss()

        # Training loop
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config["epochs"]):
            # Training
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs["price"][:, 0], y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        outputs = self.model(X_batch)
                        loss = criterion(outputs["price"][:, 0], y_batch)
                        val_loss += loss.item()

                val_loss /= len(val_loader)
                history["val_loss"].append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        self.is_trained = True
        return history

    def predict(self, X: np.ndarray) -> Prediction:
        """Make single prediction."""
        predictions = self.predict_batch(X.reshape(1, *X.shape))
        return predictions[0]

    def predict_batch(self, X: np.ndarray) -> List[Prediction]:
        """Make batch predictions."""
        import torch

        if self.model is None:
            raise ValueError("Model not built or trained")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)

        predictions = []
        price_preds = outputs["price"].cpu().numpy()
        direction_probs = torch.softmax(outputs["direction"], dim=1).cpu().numpy()
        confidence = outputs["confidence"].cpu().numpy()

        for i in range(len(X)):
            # Get direction
            dir_idx = np.argmax(direction_probs[i])
            direction = ["bearish", "neutral", "bullish"][dir_idx]

            # Multi-horizon predictions
            horizons = self.config["prediction_horizon"]
            price_multi = {f"{h}h": float(price_preds[i, j]) for j, h in enumerate(horizons)}

            pred = Prediction(
                timestamp=datetime.now(),
                symbol="",
                price_prediction=float(price_preds[i, 0]),
                price_predictions_multi=price_multi,
                direction=direction,
                direction_probability=float(direction_probs[i, dir_idx]),
                confidence=float(confidence[i, 0]),
                model_name=self.name,
                model_version=self.version,
                prediction_horizon=horizons[0],
            )
            predictions.append(pred)

        return predictions


# Register model
ModelRegistry.register("short_term", ShortTermModel)
