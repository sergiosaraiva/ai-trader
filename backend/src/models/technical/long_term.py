"""Long-term prediction model using N-BEATS + Transformer."""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np

from ..base import Prediction, ModelRegistry
from .base import TechnicalBaseModel


class LongTermModel(TechnicalBaseModel):
    """
    Long-term prediction model for position trading.

    Architecture: N-BEATS + Transformer Hybrid

    Input: 52 weekly candles (1 year)
    Output: Price predictions for 1W, 2W, 4W ahead + regime classification
    """

    DEFAULT_CONFIG = {
        "name": "long_term",
        "version": "1.0.0",
        "sequence_length": 52,
        "prediction_horizon": [1, 2, 4],

        # N-BEATS Configuration
        "nbeats_stacks": 3,
        "nbeats_blocks": 3,
        "nbeats_layers": 4,
        "nbeats_width": 256,

        # Transformer Configuration
        "transformer_heads": 8,
        "transformer_layers": 4,
        "transformer_dim": 512,
        "dropout": 0.1,

        # Regime classes
        "regime_classes": ["trending_up", "trending_down", "ranging", "volatile"],

        # Training
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 100,
        "early_stopping_patience": 20,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize long-term model."""
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)

    def build(self) -> None:
        """Build N-BEATS + Transformer architecture."""
        try:
            import torch
            import torch.nn as nn

            config = self.config

            class NBEATSBlock(nn.Module):
                """Single N-BEATS block."""

                def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
                    super().__init__()

                    layers = []
                    for i in range(num_layers):
                        in_features = input_size if i == 0 else hidden_size
                        layers.extend([
                            nn.Linear(in_features, hidden_size),
                            nn.ReLU(),
                        ])

                    self.fc = nn.Sequential(*layers)
                    self.backcast = nn.Linear(hidden_size, input_size)
                    self.forecast = nn.Linear(hidden_size, output_size)

                def forward(self, x):
                    h = self.fc(x)
                    return self.backcast(h), self.forecast(h)

            class NBEATSTransformer(nn.Module):
                """N-BEATS + Transformer hybrid model."""

                def __init__(self, input_size: int, config: dict):
                    super().__init__()

                    seq_len = config["sequence_length"]
                    hidden_size = config["nbeats_width"]
                    n_horizons = len(config["prediction_horizon"])

                    # N-BEATS stacks
                    self.stacks = nn.ModuleList()
                    for _ in range(config["nbeats_stacks"]):
                        blocks = nn.ModuleList()
                        for _ in range(config["nbeats_blocks"]):
                            blocks.append(NBEATSBlock(
                                input_size=seq_len * input_size,
                                hidden_size=hidden_size,
                                num_layers=config["nbeats_layers"],
                                output_size=n_horizons,
                            ))
                        self.stacks.append(blocks)

                    # Transformer for pattern recognition
                    self.input_projection = nn.Linear(input_size, config["transformer_dim"])
                    self.pos_encoding = nn.Parameter(
                        torch.randn(1, seq_len, config["transformer_dim"]) * 0.02
                    )

                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=config["transformer_dim"],
                        nhead=config["transformer_heads"],
                        dim_feedforward=config["transformer_dim"] * 4,
                        dropout=config["dropout"],
                        batch_first=True,
                    )
                    self.transformer = nn.TransformerEncoder(
                        encoder_layer,
                        num_layers=config["transformer_layers"],
                    )

                    # Fusion layer
                    self.fusion = nn.Sequential(
                        nn.Linear(config["transformer_dim"] + n_horizons, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(config["dropout"]),
                    )

                    # Output heads
                    self.price_head = nn.Linear(hidden_size, n_horizons)
                    self.regime_head = nn.Linear(hidden_size, len(config["regime_classes"]))
                    self.trend_strength_head = nn.Sequential(
                        nn.Linear(hidden_size, 1),
                        nn.Sigmoid(),
                    )

                    self.seq_len = seq_len
                    self.input_size = input_size

                def forward(self, x):
                    batch_size = x.size(0)

                    # N-BEATS forward
                    x_flat = x.view(batch_size, -1)
                    forecast_sum = torch.zeros(batch_size, x.size(1)).to(x.device)

                    residual = x_flat
                    for stack in self.stacks:
                        for block in stack:
                            backcast, forecast = block(residual)
                            residual = residual - backcast
                            forecast_sum = forecast_sum + forecast[:, :x.size(1)]

                    nbeats_out = forecast_sum[:, :len(self.price_head.weight)]

                    # Transformer forward
                    x_proj = self.input_projection(x)
                    x_proj = x_proj + self.pos_encoding
                    transformer_out = self.transformer(x_proj)
                    transformer_final = transformer_out[:, -1, :]

                    # Fusion
                    nbeats_pred = nbeats_out[:, :self.price_head.out_features]
                    combined = torch.cat([transformer_final, nbeats_pred], dim=-1)
                    fused = self.fusion(combined)

                    # Outputs
                    price = self.price_head(fused)
                    regime = self.regime_head(fused)
                    trend_strength = self.trend_strength_head(fused)

                    return {
                        "price": price,
                        "regime": regime,
                        "trend_strength": trend_strength,
                        "nbeats_forecast": nbeats_pred,
                    }

            self._model_class = NBEATSTransformer
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
        regime_probs = torch.softmax(outputs["regime"], dim=1).cpu().numpy()
        trend_strength = outputs["trend_strength"].cpu().numpy()

        regime_classes = self.config["regime_classes"]

        for i in range(len(X)):
            regime_idx = np.argmax(regime_probs[i])
            regime = regime_classes[regime_idx]

            # Direction from regime
            if regime == "trending_up":
                direction = "bullish"
            elif regime == "trending_down":
                direction = "bearish"
            else:
                direction = "neutral"

            horizons = self.config["prediction_horizon"]
            price_multi = {f"{h}w": float(price_preds[i, j]) for j, h in enumerate(horizons)}

            pred = Prediction(
                timestamp=datetime.now(),
                symbol="",
                price_prediction=float(price_preds[i, 0]),
                price_predictions_multi=price_multi,
                direction=direction,
                direction_probability=float(regime_probs[i, regime_idx]),
                confidence=float(trend_strength[i, 0]),
                model_name=self.name,
                model_version=self.version,
                prediction_horizon=horizons[0],
            )
            predictions.append(pred)

        return predictions


ModelRegistry.register("long_term", LongTermModel)
