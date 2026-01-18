"""Meta-model for stacking ensemble."""

from typing import Dict, List, Optional, Any
import numpy as np

from ..base import BaseModel, Prediction


class MetaModel(BaseModel):
    """
    Meta-model for stacking ensemble learning.

    Takes outputs from component models as input and learns
    optimal combination weights.
    """

    DEFAULT_CONFIG = {
        "name": "meta_model",
        "version": "1.0.0",
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "epochs": 50,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize meta-model."""
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged_config)
        self.component_names: List[str] = []

    def set_components(self, component_names: List[str]) -> None:
        """Set the names of component models."""
        self.component_names = component_names

    def build(self) -> None:
        """Build meta-model architecture."""
        try:
            import torch
            import torch.nn as nn

            config = self.config
            n_components = len(self.component_names)

            # Input: predictions + confidences from each component
            # Plus additional features like market volatility
            input_size = n_components * 3  # price, direction_prob, confidence

            class MetaNetwork(nn.Module):
                def __init__(self, input_size: int, config: dict):
                    super().__init__()

                    layers = []
                    in_features = input_size

                    for _ in range(config["num_layers"]):
                        layers.extend([
                            nn.Linear(in_features, config["hidden_size"]),
                            nn.ReLU(),
                            nn.Dropout(config["dropout"]),
                        ])
                        in_features = config["hidden_size"]

                    self.features = nn.Sequential(*layers)

                    # Output heads
                    self.price_head = nn.Linear(config["hidden_size"], 1)
                    self.direction_head = nn.Linear(config["hidden_size"], 3)
                    self.confidence_head = nn.Sequential(
                        nn.Linear(config["hidden_size"], 1),
                        nn.Sigmoid(),
                    )
                    self.weight_head = nn.Sequential(
                        nn.Linear(config["hidden_size"], n_components),
                        nn.Softmax(dim=-1),
                    )

                def forward(self, x):
                    features = self.features(x)
                    return {
                        "price": self.price_head(features),
                        "direction": self.direction_head(features),
                        "confidence": self.confidence_head(features),
                        "weights": self.weight_head(features),
                    }

            self.model = MetaNetwork(input_size, config)

        except ImportError:
            raise ImportError("PyTorch required. Install with: pip install torch")

    def prepare_meta_features(
        self, component_predictions: Dict[str, Prediction]
    ) -> np.ndarray:
        """
        Prepare input features from component predictions.

        Args:
            component_predictions: Dict mapping model name to Prediction

        Returns:
            Feature array for meta-model
        """
        features = []

        for name in self.component_names:
            pred = component_predictions.get(name)
            if pred:
                features.extend([
                    pred.price_prediction,
                    pred.direction_probability,
                    pred.confidence,
                ])
            else:
                features.extend([0.0, 0.5, 0.0])

        return np.array(features, dtype=np.float32)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Train meta-model.

        X_train: Meta-features from component predictions
        y_train: Actual target values
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        if self.model is None:
            self.build()

        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"]
        )
        criterion = nn.HuberLoss()

        history = {"train_loss": []}

        for epoch in range(self.config["epochs"]):
            self.model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs["price"].squeeze(), y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

        self.is_trained = True
        return history

    def predict(self, X: np.ndarray) -> Dict[str, Any]:
        """Make prediction from meta-features."""
        import torch

        if self.model is None:
            raise ValueError("Model not built")

        self.model.eval()
        X_tensor = torch.FloatTensor(X).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(X_tensor)

        return {
            "price": float(outputs["price"].squeeze().cpu().numpy()),
            "direction_probs": outputs["direction"].squeeze().cpu().numpy(),
            "confidence": float(outputs["confidence"].squeeze().cpu().numpy()),
            "learned_weights": {
                name: float(w)
                for name, w in zip(
                    self.component_names,
                    outputs["weights"].squeeze().cpu().numpy(),
                )
            },
        }

    def predict_batch(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """Make batch predictions."""
        return [self.predict(x) for x in X]
