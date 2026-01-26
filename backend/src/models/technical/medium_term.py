"""Medium-term prediction model using Temporal Fusion Transformer."""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np

from ..base import Prediction, ModelRegistry
from .base import TechnicalBaseModel


class MediumTermModel(TechnicalBaseModel):
    """
    Medium-term prediction model for swing trading.

    Architecture: Temporal Fusion Transformer (TFT)

    Input: 90 daily candles (3 months)
    Output: Price predictions for 1D, 3D, 5D, 7D ahead
    """

    DEFAULT_CONFIG = {
        "name": "medium_term",
        "version": "1.0.0",
        "sequence_length": 90,
        "prediction_horizon": [1, 3, 5, 7],

        # TFT Configuration
        "hidden_size": 256,
        "attention_heads": 4,
        "dropout": 0.1,
        "hidden_continuous_size": 64,
        "num_lstm_layers": 2,

        # Quantiles for uncertainty
        "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],

        # Training
        "batch_size": 64,
        "learning_rate": 1e-4,
        "epochs": 100,
        "early_stopping_patience": 15,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize medium-term model."""
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

    def build(self) -> None:
        """Build Temporal Fusion Transformer architecture."""
        try:
            import torch
            import torch.nn as nn

            config = self.config

            class SimpleTFT(nn.Module):
                """Simplified TFT implementation."""

                def __init__(self, input_size: int, config: dict):
                    super().__init__()

                    hidden_size = config["hidden_size"]

                    # Input projection
                    self.input_projection = nn.Linear(input_size, hidden_size)

                    # LSTM encoder
                    self.encoder = nn.LSTM(
                        input_size=hidden_size,
                        hidden_size=hidden_size,
                        num_layers=config["num_lstm_layers"],
                        dropout=config["dropout"],
                        batch_first=True,
                    )

                    # Multi-head attention
                    self.attention = nn.MultiheadAttention(
                        embed_dim=hidden_size,
                        num_heads=config["attention_heads"],
                        dropout=config["dropout"],
                        batch_first=True,
                    )

                    # Gated residual network
                    self.grn = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.ELU(),
                        nn.Linear(hidden_size, hidden_size),
                        nn.Dropout(config["dropout"]),
                    )
                    self.gate = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        nn.Sigmoid(),
                    )
                    self.layer_norm = nn.LayerNorm(hidden_size)

                    # Output heads
                    n_horizons = len(config["prediction_horizon"])
                    n_quantiles = len(config["quantiles"])

                    self.price_head = nn.Linear(hidden_size, n_horizons)
                    self.quantile_head = nn.Linear(hidden_size, n_horizons * n_quantiles)
                    self.direction_head = nn.Linear(hidden_size, 3)

                    self.n_horizons = n_horizons
                    self.n_quantiles = n_quantiles

                def forward(self, x):
                    # Project input
                    x = self.input_projection(x)

                    # LSTM encoding
                    lstm_out, _ = self.encoder(x)

                    # Self-attention
                    attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

                    # Gated residual
                    grn_out = self.grn(attn_out)
                    gate_out = self.gate(attn_out)
                    gated = gate_out * grn_out + (1 - gate_out) * attn_out
                    out = self.layer_norm(gated + lstm_out)

                    # Use last timestep
                    final = out[:, -1, :]

                    # Outputs
                    price = self.price_head(final)
                    quantiles = self.quantile_head(final).view(-1, self.n_horizons, self.n_quantiles)
                    direction = self.direction_head(final)

                    return {
                        "price": price,
                        "quantiles": quantiles,
                        "direction": direction,
                        "attention_weights": attn_weights,
                    }

            self._model_class = SimpleTFT
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

        input_size = X_train.shape[2]
        self.model = self._model_class(input_size, self._model_config)
        self.model.to(self.device)

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

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=1e-5,
        )
        criterion = nn.HuberLoss()

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config["epochs"]):
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

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

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

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config["early_stopping_patience"]:
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
        quantile_preds = outputs["quantiles"].cpu().numpy()
        direction_probs = torch.softmax(outputs["direction"], dim=1).cpu().numpy()

        for i in range(len(X)):
            dir_idx = np.argmax(direction_probs[i])
            direction = ["bearish", "neutral", "bullish"][dir_idx]

            horizons = self.config["prediction_horizon"]
            price_multi = {f"{h}d": float(price_preds[i, j]) for j, h in enumerate(horizons)}

            # Uncertainty from quantiles
            q_idx = self.config["quantiles"].index(0.5) if 0.5 in self.config["quantiles"] else 2
            lower_idx = 0  # 10th percentile
            upper_idx = -1  # 90th percentile

            pred = Prediction(
                timestamp=datetime.now(),
                symbol="",
                price_prediction=float(price_preds[i, 0]),
                price_predictions_multi=price_multi,
                direction=direction,
                direction_probability=float(direction_probs[i, dir_idx]),
                confidence=1.0 - float(quantile_preds[i, 0, upper_idx] - quantile_preds[i, 0, lower_idx]),
                prediction_lower=float(quantile_preds[i, 0, lower_idx]),
                prediction_upper=float(quantile_preds[i, 0, upper_idx]),
                model_name=self.name,
                model_version=self.version,
                prediction_horizon=horizons[0],
            )
            predictions.append(pred)

        return predictions


ModelRegistry.register("medium_term", MediumTermModel)
